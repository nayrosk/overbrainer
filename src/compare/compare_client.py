"""Asks llama-server every question of questions.jsonl, one at a time, and
writes child_answers.jsonl, hardware.json and progress lines. Written by
overbrainer, run by compare.sh in the job's directory, with the Python of the
job's runtime and its standard library only.
"""

import argparse
import http.client
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

QUESTIONS = "questions.jsonl"
ANSWERS = "child_answers.jsonl"
HARDWARE = "hardware.json"
LOG = "server.log"
PROGRESS_EVERY = 2.0
REQUEST_TIMEOUT = 600


def fail(message):
    """Ends the job with `message` on stderr."""
    sys.exit(f"compare: {message}")


def tail(path, count=20):
    """The last `count` lines of the file at `path`, or nothing."""
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            return "".join(file.readlines()[-count:])
    except OSError:
        return ""


def alive(pid):
    """Whether process `pid` runs: a zombie, not yet reaped by the shell, is dead."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as file:
            return file.read().rsplit(") ", 1)[-1][:1] not in ("Z", "X")
    except FileNotFoundError:
        # No /proc (macOS): kill(pid, 0) succeeds on an unreaped zombie, so
        # a dead server still looks alive there.
        pass
    except OSError:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def exited(pid, wait=2.0):
    """Whether process `pid` ended, or ends within `wait` seconds: a server
    that just closed its connections may take a moment to be gone."""
    deadline = time.monotonic() + wait
    while alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def cpu_name():
    """The CPU model, when the system says it."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        named = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return named.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return platform.processor() or None


def listed(command):
    """The non-empty lines `command` prints, or none when it is missing or fails."""
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def after(lines, marker):
    """What follows `marker` on each of `lines` that holds it, when not empty."""
    names = []
    for line in lines:
        if marker in line:
            name = line.split(marker, 1)[1].strip()
            if name:
                names.append(name)
    return names


# The name ROCm tools give an AMD integrated GPU (a Ryzen iGPU): left out
# beside a discrete GPU, as it is not what serves the model.
INTEGRATED_AMD = "AMD Radeon Graphics"


def amd_gpus():
    """The AMD GPUs by name: rocm-smi, else amd-smi, else lspci, else one
    generic. An integrated GPU named "AMD Radeon Graphics" is left out when
    a discrete one is listed too."""
    names = after(listed(["rocm-smi", "--showproductname"]), "Card Series:")
    names = names or after(listed(["amd-smi", "static"]), "MARKET_NAME:")
    if not names:
        for line in listed(["lspci"]):
            if re.search(r"VGA|Display|3D", line) and re.search(r"AMD|ATI", line):
                name = line.split("[AMD/ATI]", 1)[-1] if "[AMD/ATI]" in line else line.split(": ", 1)[-1]
                names.append(name.strip())
    discrete = [name for name in names if name != INTEGRATED_AMD]
    return discrete or names or ["AMD GPU"]


def apple_gpu():
    """The Apple chip whose GPU Metal uses, as `<chip> (Metal)`."""
    chip = listed(["sysctl", "-n", "machdep.cpu.brand_string"])
    return [f"{chip[0] if chip else 'Apple GPU'} (Metal)"]


def gpus_of(build):
    """The GPUs the llama.cpp `build` served on: the NVIDIA ones for a CUDA
    build, the AMD ones for a ROCm build, the Apple chip for the macOS (Metal)
    build, none for a CPU build, whatever GPU the machine has."""
    if build.startswith("ubuntu-cuda-"):
        names = listed(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        return names or ["NVIDIA GPU"]
    if build.startswith("ubuntu-rocm-"):
        return amd_gpus()
    if build.startswith("macos-"):
        return apple_gpu()
    return []


def write_hardware(build):
    """Writes hardware.json: the build, the GPUs it served on, the CPU."""
    with open(HARDWARE, "w", encoding="utf-8") as file:
        json.dump({"build": build, "gpus": gpus_of(build), "cpu": cpu_name()}, file)


def wait_ready(base, pid, seconds):
    """Waits until the server's /health answers 200, at most `seconds`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            fail("llama-server exited before it was ready:\n" + tail(LOG))
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as answer:
                if answer.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    fail(f"llama-server was not ready after {seconds} s:\n" + tail(LOG))


def ask(base, question, settings):
    """Asks one question, streamed; returns its answer line."""
    body = {
        "messages": question["messages"],
        "max_tokens": settings["max_tokens"],
        "temperature": settings["temperature"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = None
    parts = []
    finish = None
    usage = {}
    timings = {}
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as answer:
        for raw in answer:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or ""
                if (text or thought) and first is None:
                    first = time.monotonic() - start
                if text:
                    parts.append(text)
                finish = choice.get("finish_reason") or finish
    seconds = time.monotonic() - start
    output = usage.get("completion_tokens") or timings.get("predicted_n")
    prompt = usage.get("prompt_tokens") or timings.get("prompt_n")
    generating = seconds - (first or 0.0)
    return {
        "id": question["id"],
        "answer": "".join(parts),
        "finish": finish,
        "input_tokens": prompt,
        "output_tokens": output,
        "seconds": round(seconds, 4),
        "first_token_seconds": None if first is None else round(first, 4),
        "tokens_per_second": round(output / generating, 2) if output and generating > 0 else None,
    }


def progress(step, total):
    """Appends a progress line to the job's metrics file."""
    metrics = os.environ.get("OVERBRAINER_METRICS")
    if not metrics:
        return
    with open(metrics, "a", encoding="utf-8") as file:
        file.write(json.dumps({"event": "eval", "time": time.time(), "step": step, "total": total}) + "\n")


def main():
    """Asks every question, then says where the answers are."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--build", required=True)
    args = parser.parse_args()
    settings = {
        "max_tokens": int(os.environ["OVERBRAINER_COMPARE_MAX_TOKENS"]),
        "temperature": float(os.environ["OVERBRAINER_COMPARE_TEMPERATURE"]),
    }
    start_secs = int(os.environ["OVERBRAINER_COMPARE_START_SECS"])
    base = f"http://127.0.0.1:{args.port}"
    write_hardware(args.build)
    wait_ready(base, args.server_pid, start_secs)
    with open(QUESTIONS, encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]
    total = len(questions)
    print(f"compare: asking {total} questions", flush=True)
    progress(0, total)
    last = time.monotonic()
    with open(ANSWERS, "w", encoding="utf-8") as out:
        for step, question in enumerate(questions, 1):
            try:
                line = ask(base, question, settings)
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as error:
                if exited(args.server_pid):
                    fail("llama-server exited while answering:\n" + tail(LOG))
                line = {"id": question["id"], "error": str(error)}
            out.write(json.dumps(line) + "\n")
            out.flush()
            now = time.monotonic()
            if step == total or now - last >= PROGRESS_EVERY:
                progress(step, total)
                last = now
    print(f"compare: wrote {ANSWERS}", flush=True)


if __name__ == "__main__":
    main()

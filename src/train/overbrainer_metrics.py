"""Axolotl plugin shipped by overbrainer.

Appends one JSON line per Trainer log to the file named by $OVERBRAINER_METRICS,
from the main process only. overbrainer tails that file for live metrics.

When the file named by $OVERBRAINER_SNAPSHOT appears (the snapshot request),
saves a checkpoint at the end of the current step, stops training, and writes
snapshot.json next to the request: the proof overbrainer reads once the job ends.
"""

import json
import math
import os
import time

from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

try:
    import torch
    import torch.distributed as dist
except ImportError:
    torch = None

FIELDS = ("loss", "eval_loss", "learning_rate", "grad_norm")

SNAPSHOT_FILE = "snapshot.json"
REASONS = ("requested", "deadline", "cost", "disk")


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class OverbrainerMetricsCallback(TrainerCallback):
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _write(self, record):
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
            file.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write(
                {"event": "begin", "time": time.time(), "max_steps": state.max_steps}
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        record = {
            "event": "log",
            "time": time.time(),
            "step": state.global_step,
            "epoch": _number(state.epoch),
            "max_steps": state.max_steps,
        }
        for field in FIELDS:
            if field in logs:
                record[field] = _number(logs[field])
        self._write(record)


def _reason(path):
    """The reason written in the request, `requested` when it names none."""
    try:
        with open(path, encoding="utf-8") as file:
            reason = file.read().strip()
    except OSError:
        return "requested"
    return reason if reason in REASONS else "requested"


class OverbrainerSnapshotCallback(TrainerCallback):
    """Saves a checkpoint and stops training once the request file exists.

    Rank 0 looks for the file after every step and, under distributed training,
    broadcasts what it found, so every rank stops at the same step. The trainer
    saves the checkpoint before it checks should_training_stop, then calls
    on_save, which writes snapshot.json atomically.
    """

    def __init__(self, request):
        self.request = request
        self.root = os.path.dirname(request) or "."
        self.reason = None

    def _requested(self, args, state):
        found = state.is_world_process_zero and os.path.exists(self.request)
        if torch is None or not (dist.is_available() and dist.is_initialized()):
            return found
        flag = torch.tensor([1 if found else 0], device=args.device)
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def on_step_end(self, args, state, control, **kwargs):
        if self.reason is not None or not self._requested(args, state):
            return
        self.reason = _reason(self.request) if state.is_world_process_zero else "requested"
        control.should_save = True
        control.should_training_stop = True

    def on_save(self, args, state, control, **kwargs):
        if self.reason is None or not state.is_world_process_zero:
            return
        checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        record = {
            "checkpoint": os.path.relpath(checkpoint, self.root),
            "step": state.global_step,
            "time": time.time(),
            "reason": self.reason,
        }
        path = os.path.join(self.root, SNAPSHOT_FILE)
        with open(path + ".tmp", "w", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(path + ".tmp", path)


class OverbrainerMetricsPlugin(BasePlugin):
    def add_callbacks_pre_trainer(self, cfg, model):
        callbacks = []
        path = os.environ.get("OVERBRAINER_METRICS")
        if path:
            callbacks.append(OverbrainerMetricsCallback(path))
        request = os.environ.get("OVERBRAINER_SNAPSHOT")
        if request:
            callbacks.append(OverbrainerSnapshotCallback(request))
        return callbacks

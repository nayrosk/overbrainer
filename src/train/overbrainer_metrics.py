"""Axolotl plugin shipped by overbrainer.

Appends one JSON line per Trainer log to the file named by $OVERBRAINER_METRICS,
from the main process only. overbrainer tails that file for live metrics.
"""

import json
import math
import os
import time

from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

FIELDS = ("loss", "eval_loss", "learning_rate", "grad_norm")


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


class OverbrainerMetricsPlugin(BasePlugin):
    def add_callbacks_pre_trainer(self, cfg, model):
        path = os.environ.get("OVERBRAINER_METRICS")
        return [OverbrainerMetricsCallback(path)] if path else []

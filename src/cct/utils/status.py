"""Append timestamped entries to the log section at the end of results/STATUS.md."""
from __future__ import annotations

import datetime as _dt
import os


def log_status(line: str, path: str = "results/STATUS.md") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- {ts}: {line}\n")

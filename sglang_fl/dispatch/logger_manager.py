# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Centralized logging for dispatch module.

import logging
import os

_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_loggers: dict = {}


def get_logger(name: str = "sglang_fl.dispatch") -> logging.Logger:
    """Get or create a logger for the dispatch module."""
    if name not in _loggers:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)

        level_str = os.environ.get("SGLANG_FL_LOG_LEVEL", "INFO").upper()
        logger.setLevel(_LOG_LEVEL_MAP.get(level_str, logging.INFO))
        _loggers[name] = logger

    return _loggers[name]


def set_log_level(level: str, name: str = "sglang_fl.dispatch") -> None:
    """Set log level for a dispatch logger."""
    logger = get_logger(name)
    logger.setLevel(_LOG_LEVEL_MAP.get(level.upper(), logging.INFO))


# --- Execution-level proof markers ---
# Unlike [OOT-DISPATCH] (which logs selection), [EXEC] proves the function body ran.

_exec_logged: set = set()


def log_exec(op_name: str, backend: str) -> None:
    """
    Write an execution marker to the dispatch log file.

    Called inside backend function bodies to prove actual execution
    (not just dispatch selection). Only logs once per (op, backend) pair.
    """
    key = (op_name, backend)
    if key in _exec_logged:
        return
    _exec_logged.add(key)

    log_path = os.environ.get("SGLANG_FL_DISPATCH_LOG", "").strip()
    if log_path:
        try:
            with open(log_path, "a") as f:
                f.write(f"[EXEC] {op_name} → {backend}\n")
                f.flush()
        except Exception:
            pass

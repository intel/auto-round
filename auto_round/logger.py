# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import lru_cache, wraps
from typing import  Callable, Optional, TypeVar

import auto_round.envs as envs
import torch

T = TypeVar("T", bound="Callable")  # used by `deprecated`


@lru_cache(maxsize=None)
@torch._dynamo.disable()
def warning_once(self, msg, *args):
    """
    Log a warning message only once per unique message/arguments combination.

    Args:
        msg: The warning message format string
        *args: Variable positional arguments for message formatting
    """
    logger.warning(msg, *args, stacklevel=2)


# Define a new logging level TRACE
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def trace(self, message, *args):
    """
    Log a message with the TRACE level.

    Args:
        message: The message format string
        *args: Variable positional arguments for message formatting

    """
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, stacklevel=2)


# Add the trace method to the Logger class
logging.Logger.trace = trace


class AutoRoundFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    cyan = "\x1b[36;1m"
    blue = "\x1b[34;1m"
    _format = "%(asctime)s %(levelname)s %(filename)s L%(lineno)d: %(message)s"

    FORMATS = {
        logging.DEBUG: blue + _format + reset,
        logging.INFO: grey + _format + reset,
        logging.WARNING: yellow + _format + reset,
        logging.ERROR: bold_red + _format + reset,
        logging.CRITICAL: bold_red + _format + reset,
        TRACE_LEVEL: cyan + _format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


logging.Logger.warning_once = warning_once
logger = logging.getLogger("autoround")
logger.setLevel(envs.AR_LOG_LEVEL)
logger.propagate = False
fh = logging.StreamHandler()
fh.setFormatter(AutoRoundFormatter())
logger.addHandler(fh)


def deprecated(future_name: Optional[str] = None, message: Optional[str] = None) -> Callable[[T], T]:
    """
    Decorator to mark functions as deprecated

    :param new_function: Function called in place of deprecated function
    :param message: Deprecation message, replaces default deprecation message
    """

    def decorator(func: T) -> T:
        nonlocal message

        if message is None:
            message = f"{func.__name__} is deprecated and will be removed in a future release"
            if future_name is not None:
                message += f". Please use {future_name} instead."

        @wraps(func)
        def wrapped(*args, **kwargs):
            logger.warning_once(message)
            return func(*args, **kwargs)

        return wrapped

    return decorator

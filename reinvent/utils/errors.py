"""User-facing error reporting helpers

These utilities separate the two output streams that errors need:

* **stderr / terminal** — a short, bordered, plain-English message that the
  user can act on immediately, always visible regardless of log-file settings.
* **log file** — the full traceback for developers, written via the logger at
  ERROR level only when a logger is available.

Typical usage in an entry-point module::

    from reinvent.utils.errors import fatal, format_validation_error

    try:
        val_config = MyConfig(**raw)
    except pydantic.ValidationError as exc:
        fatal(format_validation_error(exc), exc_info=exc)
"""

from __future__ import annotations

__all__ = ["fatal", "format_validation_error"]

import sys
import logging
from typing import Optional

_BORDER = "=" * 72


def format_validation_error(exc) -> str:
    """Return a readable multi-line summary of a pydantic ValidationError.

    :param exc: a ``pydantic.ValidationError`` instance
    :returns: formatted string suitable for printing to the terminal
    """

    lines = ["Configuration error(s) found:"]

    for err in exc.errors():
        loc = " → ".join(str(p) for p in err["loc"]) if err.get("loc") else "(top level)"
        msg = err.get("msg", "invalid value")
        got = err.get("input", "?")
        got_repr = repr(got)
        if len(got_repr) > 80:
            got_repr = got_repr[:77] + "..."
        lines.append(f"  [{loc}] {msg} (got: {got_repr})")

    return "\n".join(lines)


def fatal(
    msg: str,
    log: Optional[logging.Logger] = None,
    exc_info=None,
    exit_code: int = 1,
) -> None:
    """Print a user-facing error to stderr and exit.

    The full traceback is written to the log file only (when *log* and
    *exc_info* are both supplied); the terminal sees a concise, bordered
    plain-text message.

    :param msg: human-readable error description
    :param log: optional logger — when given, ``log.error(msg, exc_info=exc_info)``
                is called so the traceback lands in the log file
    :param exc_info: exception instance or ``(type, value, tb)`` tuple passed
                     to the logger; has no effect when *log* is ``None``
    :param exit_code: process exit code (default 1)
    """

    if log is not None:
        # The logger owns all output: its file handler gets the full message
        # and traceback; its stderr handler (SuppressTracebackFormatter) emits
        # the plain message without the traceback.  When no log file is
        # configured the single stderr handler shows everything, which is the
        # intended "both written to terminal" behaviour.
        log.critical(msg, exc_info=exc_info)
    else:
        # No logger yet (pre-logger CLI checks): write directly to stderr with
        # a visible border so the message cannot be missed.
        print(f"\n{_BORDER}\n{msg}\n{_BORDER}\n", file=sys.stderr, flush=True)

    sys.exit(exit_code)

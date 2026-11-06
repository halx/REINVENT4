"""Tests for reinvent/utils/logmon.py — setup_logger and SuppressTracebackFormatter."""

import logging
import tempfile
import os

import pytest

from reinvent.utils.logmon import setup_logger, SuppressTracebackFormatter

# ---------------------------------------------------------------------------
# setup_logger — handler count and levels
# ---------------------------------------------------------------------------


def test_no_file_single_handler():
    log = setup_logger(name="test_single", level=logging.DEBUG)
    assert len(log.handlers) == 1
    assert isinstance(log.handlers[0], logging.StreamHandler)


def test_with_file_two_handlers(tmp_path):
    log_file = str(tmp_path / "test.log")
    log = setup_logger(name="test_dual", filename=log_file, level=logging.DEBUG)
    assert len(log.handlers) == 2


def test_with_file_file_handler_is_first(tmp_path):
    log_file = str(tmp_path / "test.log")
    log = setup_logger(name="test_file_first", filename=log_file, level=logging.DEBUG)
    file_handlers = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


def test_with_file_stderr_handler_at_warning(tmp_path):
    log_file = str(tmp_path / "test.log")
    log = setup_logger(name="test_stderr_level", filename=log_file, level=logging.DEBUG)
    stderr_handlers = [
        h
        for h in log.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert len(stderr_handlers) == 1
    assert stderr_handlers[0].level == logging.WARNING


def test_with_file_file_handler_at_requested_level(tmp_path):
    log_file = str(tmp_path / "test.log")
    log = setup_logger(name="test_file_level", filename=log_file, level=logging.DEBUG)
    file_handlers = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers[0].level == logging.DEBUG


def test_critical_written_to_log_file(tmp_path):
    log_file = str(tmp_path / "test_critical.log")
    log = setup_logger(name="test_crit_file", filename=log_file, level=logging.DEBUG)
    log.critical("critical test message")
    with open(log_file) as f:
        content = f.read()
    assert "critical test message" in content


# ---------------------------------------------------------------------------
# SuppressTracebackFormatter
# ---------------------------------------------------------------------------


def test_suppress_formatter_removes_traceback():
    formatter = SuppressTracebackFormatter("%(levelname)s: %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="error message",
        args=(),
        exc_info=None,
    )
    try:
        raise ValueError("test exception")
    except ValueError:
        import sys

        record.exc_info = sys.exc_info()

    output = formatter.format(record)
    assert "Traceback" not in output
    assert "ValueError" not in output
    assert "error message" in output


def test_suppress_formatter_does_not_mutate_record():
    """exc_info must be restored so downstream file handlers still see it."""
    formatter = SuppressTracebackFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="msg",
        args=(),
        exc_info=None,
    )
    try:
        raise RuntimeError("test")
    except RuntimeError:
        import sys

        original_exc_info = sys.exc_info()
        record.exc_info = original_exc_info

    formatter.format(record)
    assert record.exc_info is original_exc_info


def test_traceback_in_file_handler_but_not_stderr(tmp_path):
    """Full integration: CRITICAL with exc_info → traceback in file, not on stderr."""
    log_file = str(tmp_path / "split.log")
    log = setup_logger(name="test_split_tb", filename=log_file, level=logging.DEBUG)

    import io

    stderr_capture = io.StringIO()
    for h in log.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.stream = stderr_capture

    try:
        raise ValueError("intentional")
    except ValueError as exc:
        log.critical("split test", exc_info=exc)

    with open(log_file) as f:
        file_content = f.read()

    stderr_content = stderr_capture.getvalue()

    assert "Traceback" in file_content
    assert "ValueError" in file_content
    assert "Traceback" not in stderr_content

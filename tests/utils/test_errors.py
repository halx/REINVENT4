"""Tests for reinvent/utils/errors.py — fatal() and format_validation_error()."""

import sys
import logging

import pytest
import pydantic

from reinvent.utils.errors import fatal, format_validation_error
from reinvent.validation import GlobalConfig

# ---------------------------------------------------------------------------
# format_validation_error
# ---------------------------------------------------------------------------


class _Model(GlobalConfig):
    x: int
    y: str = "ok"


def _make_validation_error(**kwargs):
    try:
        _Model(**kwargs)
    except pydantic.ValidationError as exc:
        return exc
    raise AssertionError("Expected ValidationError was not raised")


def test_format_starts_with_header():
    exc = _make_validation_error(x="not_an_int")
    msg = format_validation_error(exc)
    assert msg.startswith("Configuration error(s) found:")


def test_format_single_field_contains_field_name():
    exc = _make_validation_error(x="not_an_int")
    msg = format_validation_error(exc)
    assert "[x]" in msg


def test_format_extra_field_forbidden():
    exc = _make_validation_error(x=1, unknown_field="bad")
    msg = format_validation_error(exc)
    assert "unknown_field" in msg


def test_format_multiple_errors_all_present():
    exc = _make_validation_error(x="bad", unknown_field="bad")
    msg = format_validation_error(exc)
    lines = [l for l in msg.splitlines() if l.strip().startswith("[")]
    assert len(lines) == 2


def test_format_long_input_truncated():
    """got: ... is capped at 80 chars so cross-field validator errors stay readable."""
    exc = _make_validation_error(x="not_an_int")
    msg = format_validation_error(exc)
    for line in msg.splitlines():
        if "(got:" in line:
            got_part = line[line.index("(got:") :]
            assert len(got_part) <= 90, f"got: section too long: {got_part!r}"


# ---------------------------------------------------------------------------
# fatal()
# ---------------------------------------------------------------------------


def test_fatal_no_logger_exits_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        fatal("something went wrong")
    assert exc_info.value.code == 1


def test_fatal_no_logger_writes_to_stderr(capsys):
    with pytest.raises(SystemExit):
        fatal("something went wrong")
    captured = capsys.readouterr()
    assert "something went wrong" in captured.err


def test_fatal_no_logger_does_not_write_to_stdout(capsys):
    with pytest.raises(SystemExit):
        fatal("something went wrong")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_fatal_custom_exit_code(capsys):
    with pytest.raises(SystemExit) as exc_info:
        fatal("error", exit_code=2)
    assert exc_info.value.code == 2


def test_fatal_with_logger_calls_critical(caplog):
    log = logging.getLogger("test_fatal")
    with caplog.at_level(logging.CRITICAL, logger="test_fatal"):
        with pytest.raises(SystemExit):
            fatal("fatal message", log=log)
    assert any("fatal message" in r.message for r in caplog.records)
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_fatal_with_logger_exc_info_attached(caplog):
    log = logging.getLogger("test_fatal_exc")
    try:
        raise ValueError("inner error")
    except ValueError as exc:
        with caplog.at_level(logging.CRITICAL, logger="test_fatal_exc"):
            with pytest.raises(SystemExit):
                fatal("outer message", log=log, exc_info=exc)
    record = next(r for r in caplog.records if r.levelno == logging.CRITICAL)
    assert record.exc_info is not None

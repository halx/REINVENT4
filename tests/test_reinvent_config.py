"""Tests for ReinventConfig top-level validation and shared path annotated types."""

import tempfile
import os

import pytest
import pydantic

from reinvent.validation import ReinventConfig, ExistingFilePath, WritableFilePath, GlobalConfig

# ---------------------------------------------------------------------------
# run_type Literal
# ---------------------------------------------------------------------------

VALID_RUN_TYPES = [
    "sampling",
    "staged_learning",
    "transfer_learning",
    "scoring",
    "enumeration",
]


@pytest.mark.parametrize("run_type", VALID_RUN_TYPES)
def test_valid_run_type_accepted(run_type):
    cfg = ReinventConfig(run_type=run_type, parameters={})
    assert cfg.run_type == run_type


@pytest.mark.parametrize("bad", ["reinforcement", "SAMPLING", "confabulate", "rl", "tl", ""])
def test_invalid_run_type_rejected(bad):
    with pytest.raises(pydantic.ValidationError) as exc_info:
        ReinventConfig(run_type=bad, parameters={})
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("run_type",) for e in errors)


# ---------------------------------------------------------------------------
# device validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device", ["cpu", "cuda", "cuda:0", "cuda:1", "mps", "mps:0", "xpu", "xpu:0"]
)
def test_valid_device_accepted(device):
    cfg = ReinventConfig(run_type="sampling", device=device, parameters={})
    assert cfg.device == device


@pytest.mark.parametrize("bad_device", ["GPU", "CUDA", "Cpu", "cuda:", "xpu:abc", "tpu", "npu"])
def test_invalid_device_rejected(bad_device):
    with pytest.raises(pydantic.ValidationError) as exc_info:
        ReinventConfig(run_type="sampling", device=bad_device, parameters={})
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("device",) for e in errors)


# ---------------------------------------------------------------------------
# version field
# ---------------------------------------------------------------------------


def test_version_field_has_default():
    cfg = ReinventConfig(run_type="sampling", parameters={})
    assert cfg.version >= 1


def test_version_field_accepts_integer():
    cfg = ReinventConfig(run_type="sampling", version=4, parameters={})
    assert cfg.version == 4


def test_version_field_rejects_zero():
    with pytest.raises(pydantic.ValidationError):
        ReinventConfig(run_type="sampling", version=0, parameters={})


def test_version_field_rejects_negative():
    with pytest.raises(pydantic.ValidationError):
        ReinventConfig(run_type="sampling", version=-1, parameters={})


# ---------------------------------------------------------------------------
# extra fields forbidden
# ---------------------------------------------------------------------------


def test_extra_field_rejected():
    with pytest.raises(pydantic.ValidationError) as exc_info:
        ReinventConfig(run_type="sampling", parameters={}, typo_field="oops")
    errors = exc_info.value.errors()
    assert any("typo_field" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# ExistingFilePath annotated type
# ---------------------------------------------------------------------------


class _FileModel(GlobalConfig):
    f: ExistingFilePath


def test_existing_file_path_accepts_real_file(tmp_path):
    real = tmp_path / "real.txt"
    real.write_text("content")
    m = _FileModel(f=str(real))
    assert m.f == str(real)


def test_existing_file_path_rejects_missing_file():
    with pytest.raises(pydantic.ValidationError) as exc_info:
        _FileModel(f="/tmp/definitely_does_not_exist_xyz.txt")
    assert "File not found" in str(exc_info.value)


def test_existing_file_path_rejects_directory(tmp_path):
    with pytest.raises(pydantic.ValidationError):
        _FileModel(f=str(tmp_path))  # directory, not a file


# ---------------------------------------------------------------------------
# WritableFilePath annotated type
# ---------------------------------------------------------------------------


class _WriteModel(GlobalConfig):
    f: WritableFilePath


def test_writable_file_path_accepts_nonexistent_file_in_existing_dir(tmp_path):
    target = tmp_path / "new_output.pt"
    m = _WriteModel(f=str(target))
    assert m.f == str(target)


def test_writable_file_path_rejects_missing_parent():
    with pytest.raises(pydantic.ValidationError) as exc_info:
        _WriteModel(f="/no_such_dir_xyz/output.pt")
    assert "Parent directory" in str(exc_info.value)


def test_writable_file_path_accepts_existing_file(tmp_path):
    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"")
    m = _WriteModel(f=str(existing))
    assert m.f == str(existing)

"""Integration tests for ModelMetaDataV2 and related functions.

Run with:
    pytest tests/models/integration_tests/test_meta_data_v2.py -m integration --device cpu -v

All tests operate entirely in memory or via tmp_path; no model checkpoint files
from the repository are required.

Positive tests
--------------
  TestRoundTrip       – update→check cycle with every supported tensor type
  TestUpdateLog       – update_log entries are appended correctly
  TestMutationDetects – single-value changes in tensors, metadata, and non-tensor
                        fields are all detected
  TestFileRoundTrip   – torch.save / torch.load round-trip through a temp file

Negative tests (V1 / cross-version / tampering)
------------------------------------------------
  TestV1ModelRejected          – V1 save_dict (pickle hash) returns False from
                                 check_valid_hash_v2 (wrong algorithm)
  TestDispatchRouting          – check_valid_hash_dispatch routes V1→V1 checker
                                 and V2→V2 checker correctly
  TestTamperedHashRejected     – manually overwritten hash_id fails check
  TestMissingHashRejected      – None hash_id fails check
  TestWrongKeyRejected         – weight tensor replaced after hashing fails check
  TestCallerNotMutated         – update and check functions do not modify the
                                 caller's save_dict or its tensors
  TestBfloat16                 – bfloat16 tensors hash without RuntimeError
  TestGradTensor               – tensors with requires_grad=True hash without error
  TestDeviceAgnostic           – hash is identical regardless of compute device
"""

import copy
import io
import platform
import time
import uuid

import numpy as np
import pytest
import torch

from reinvent.models.meta_data import ModelMetaData, update_model_data
from reinvent.models.meta_data2 import (
    ModelMetaDataV2,
    check_valid_hash_dispatch,
    check_valid_hash_v2,
    update_model_data_v2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_v2_meta(**kwargs) -> ModelMetaDataV2:
    defaults = dict(
        hash_id=None,
        model_id=str(uuid.uuid4()),
        original_data_source="test",
        creation_date=1_700_000_000.0,
        model_architecture="RNN",
    )
    defaults.update(kwargs)
    return ModelMetaDataV2(**defaults)


def _make_save_dict(meta=None, *, seed: int = 0) -> dict:
    """Minimal Reinvent-style save_dict with a small weight tensor."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    network = {"weight": torch.randn(4, 4, generator=rng), "bias": torch.randn(4, generator=rng)}
    return {
        "model_type": "Reinvent",
        "version": 4,
        "metadata": meta or _make_v2_meta(),
        "network": network,
        "vocabulary": {"C": 0, "N": 1, "O": 2},
        "max_sequence_length": 128,
    }


def _make_v1_save_dict() -> dict:
    """V1-style save_dict with ModelMetaData (pickle-based hash)."""
    meta_v1 = ModelMetaData(
        hash_id=None,
        hash_id_format="",
        model_id=uuid.uuid4().hex,
        origina_data_source="test",
        creation_date=time.time(),
    )
    raw = {
        "model_type": "Reinvent",
        "version": 4,
        "metadata": meta_v1,
        "network": {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)},
        "vocabulary": {"C": 0},
        "max_sequence_length": 64,
    }
    return update_model_data(raw, comment="v1_test", write_update=False)


# ---------------------------------------------------------------------------
# Positive: round-trip
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRoundTrip:
    def test_basic_round_trip(self):
        sd = update_model_data_v2(_make_save_dict())
        assert check_valid_hash_v2(sd)

    def test_hash_id_is_string(self):
        sd = update_model_data_v2(_make_save_dict())
        assert isinstance(sd["metadata"]["hash_id"], str)
        assert len(sd["metadata"]["hash_id"]) == 32  # xxh3_128 hex = 128 bits = 32 hex chars

    def test_hash_format_recorded(self):
        sd = update_model_data_v2(_make_save_dict())
        assert "xxhash.xxh3_128_hex" in sd["metadata"]["hash_id_format"]

    def test_float32_tensor(self):
        sd = _make_save_dict()
        sd["network"]["weight"] = torch.randn(8, 8)
        assert check_valid_hash_v2(update_model_data_v2(sd))

    def test_float64_tensor(self):
        sd = _make_save_dict()
        sd["network"]["weight"] = torch.randn(4, 4, dtype=torch.float64)
        assert check_valid_hash_v2(update_model_data_v2(sd))

    def test_int32_tensor(self):
        sd = _make_save_dict()
        sd["network"]["weight"] = torch.arange(16, dtype=torch.int32).reshape(4, 4)
        assert check_valid_hash_v2(update_model_data_v2(sd))

    def test_numpy_array_in_dict(self):
        """numpy arrays stored directly in the save_dict are handled."""
        sd = _make_save_dict()
        sd["extra"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert check_valid_hash_v2(update_model_data_v2(sd))

    def test_metadata_as_dataclass(self):
        """update_model_data_v2 accepts metadata as a ModelMetaDataV2 object."""
        sd = _make_save_dict(meta=_make_v2_meta())
        sd = update_model_data_v2(sd)
        assert check_valid_hash_v2(sd)

    def test_metadata_as_dict(self):
        """update_model_data_v2 accepts metadata already as a plain dict."""
        sd = _make_save_dict()
        sd["metadata"] = _make_v2_meta().as_dict()
        sd = update_model_data_v2(sd)
        assert check_valid_hash_v2(sd)

    def test_repeated_updates_chain(self):
        """Multiple sequential updates each produce a valid hash."""
        sd = update_model_data_v2(_make_save_dict(), comment="first")
        sd = update_model_data_v2(sd, comment="second")
        sd = update_model_data_v2(sd, comment="third")
        assert check_valid_hash_v2(sd)

    def test_update_log_prior_hash_chain(self):
        """Each update_log entry's prior_hash equals the hash from the previous update."""
        sd = update_model_data_v2(_make_save_dict(), comment="a")
        h1 = sd["metadata"]["hash_id"]
        sd = update_model_data_v2(sd, comment="b")
        assert sd["metadata"]["update_log"][-1]["prior_hash"] == h1


# ---------------------------------------------------------------------------
# Positive: update_log bookkeeping
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUpdateLog:
    def test_write_update_true_appends_entry(self):
        sd = update_model_data_v2(_make_save_dict(), write_update=True)
        assert len(sd["metadata"]["update_log"]) == 1

    def test_write_update_false_no_timestamp_but_comment_appended(self):
        sd = update_model_data_v2(_make_save_dict(), comment="note", write_update=False)
        log = sd["metadata"]["update_log"]
        assert len(log) == 1
        assert log[0]["timestamp"] is None
        assert log[0]["comment"] == "note"

    def test_no_comment_no_write_no_log_entry(self):
        sd = update_model_data_v2(_make_save_dict(), write_update=False, comment="")
        assert len(sd["metadata"]["update_log"]) == 0

    def test_comment_stored(self):
        sd = update_model_data_v2(_make_save_dict(), comment="my comment")
        assert sd["metadata"]["update_log"][-1]["comment"] == "my comment"

    def test_timestamp_is_recent_float(self):
        before = time.time()
        sd = update_model_data_v2(_make_save_dict(), write_update=True)
        after = time.time()
        ts = sd["metadata"]["update_log"][-1]["timestamp"]
        assert isinstance(ts, float)
        assert before <= ts <= after

    def test_reinvent_version_in_log(self):
        sd = update_model_data_v2(_make_save_dict())
        entry = sd["metadata"]["update_log"][-1]
        assert isinstance(entry["reinvent_version"], str)
        assert entry["reinvent_version"]  # not empty


# ---------------------------------------------------------------------------
# Positive: mutation detection
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMutationDetects:
    def _hashed(self, seed=0):
        return update_model_data_v2(_make_save_dict(seed=seed))

    def test_weight_tensor_change_detected(self):
        sd = self._hashed()
        sd["network"]["weight"][0, 0] += 1e-6
        assert not check_valid_hash_v2(sd)

    def test_bias_tensor_change_detected(self):
        sd = self._hashed()
        sd["network"]["bias"][-1] -= 0.5
        assert not check_valid_hash_v2(sd)

    def test_vocabulary_change_detected(self):
        sd = self._hashed()
        sd["vocabulary"]["X"] = 99
        assert not check_valid_hash_v2(sd)

    def test_max_sequence_length_change_detected(self):
        sd = self._hashed()
        sd["max_sequence_length"] = 256
        assert not check_valid_hash_v2(sd)

    def test_model_type_change_detected(self):
        sd = self._hashed()
        sd["model_type"] = "Linkinvent"
        assert not check_valid_hash_v2(sd)

    def test_metadata_field_change_detected(self):
        sd = self._hashed()
        sd["metadata"]["original_data_source"] = "tampered"
        assert not check_valid_hash_v2(sd)

    def test_extra_key_added_detected(self):
        sd = self._hashed()
        sd["injected"] = "extra"
        assert not check_valid_hash_v2(sd)


# ---------------------------------------------------------------------------
# Positive: file round-trip
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFileRoundTrip:
    def test_torch_save_load_cpu(self, tmp_path):
        sd = update_model_data_v2(_make_save_dict())
        path = tmp_path / "model.pt"
        torch.save(sd, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert check_valid_hash_v2(loaded)

    def test_torch_save_load_preserves_hash_id(self, tmp_path):
        sd = update_model_data_v2(_make_save_dict())
        original_hash = sd["metadata"]["hash_id"]
        path = tmp_path / "model.pt"
        torch.save(sd, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["metadata"]["hash_id"] == original_hash

    def test_two_independent_dicts_same_hash(self):
        """Identical logical content must produce the same hash regardless of object identity."""
        fixed_meta = _make_v2_meta(model_id="fixed-id-for-determinism")
        sd1 = update_model_data_v2(_make_save_dict(meta=fixed_meta, seed=42), write_update=False)
        sd2 = update_model_data_v2(_make_save_dict(meta=fixed_meta, seed=42), write_update=False)
        assert sd1["metadata"]["hash_id"] == sd2["metadata"]["hash_id"]

    def test_different_seeds_different_hash(self):
        sd1 = update_model_data_v2(_make_save_dict(seed=1))
        sd2 = update_model_data_v2(_make_save_dict(seed=2))
        assert sd1["metadata"]["hash_id"] != sd2["metadata"]["hash_id"]


# ---------------------------------------------------------------------------
# Negative: V1 model is rejected by check_valid_hash_v2
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestV1ModelRejected:
    def test_v1_hash_fails_v2_check(self):
        """A V1 model's pickle-based hash must not validate under the V2 algorithm."""
        v1 = _make_v1_save_dict()
        assert not check_valid_hash_v2(v1)

    def test_v1_model_with_dataclass_metadata_fails_v2(self):
        """ModelMetaData stored as a dataclass object is correctly handled and still fails."""
        meta = ModelMetaData(
            hash_id="deadbeefdeadbeefdeadbeefdeadbeef",
            hash_id_format="xxhash.xxh3_128_hex 0.8.0",
            model_id="abc",
            origina_data_source="test",
            creation_date=0.0,
        )
        sd = {
            "model_type": "Reinvent",
            "version": 4,
            "metadata": meta,
            "network": {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)},
            "vocabulary": {},
            "max_sequence_length": 64,
        }
        assert not check_valid_hash_v2(sd)

    def test_v1_dict_metadata_with_v1_format_fails_v2(self):
        """V1 metadata dict carrying a pickle hash must fail the V2 check."""
        v1 = _make_v1_save_dict()
        # V1 stores metadata as a dict after update_model_data
        assert isinstance(v1["metadata"], dict)
        assert not check_valid_hash_v2(v1)


# ---------------------------------------------------------------------------
# Negative: dispatch routing
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDispatchRouting:
    def test_dispatch_routes_v2_to_v2_checker(self):
        sd = update_model_data_v2(_make_save_dict())
        assert check_valid_hash_dispatch(sd) is True

    def test_dispatch_routes_v1_to_v1_checker(self):
        """V1 save_dict with a correct V1 hash must pass through the V1 checker."""
        v1 = _make_v1_save_dict()
        # The V1 check_valid_hash is buggy for legacy (dataclass) metadata but
        # update_model_data stores metadata as a plain dict, so this should pass.
        result = check_valid_hash_dispatch(v1)
        assert result is True

    def test_dispatch_routes_v2_tampered_to_v2_and_fails(self):
        sd = update_model_data_v2(_make_save_dict())
        sd["network"]["weight"][0, 0] += 1.0
        assert check_valid_hash_dispatch(sd) is False

    def test_dispatch_on_empty_format_does_not_crash(self):
        """Missing hash_id_format falls back to V1 path without exception."""
        sd = _make_save_dict()
        sd["metadata"] = {"hash_id": None, "hash_id_format": None}
        # No assertion on the return value — the V1 checker returns False for
        # a None hash; we are only checking there is no uncaught exception.
        check_valid_hash_dispatch(sd)


# ---------------------------------------------------------------------------
# Negative: tampered / missing hash
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTamperedHashRejected:
    def test_wrong_hash_fails(self):
        sd = update_model_data_v2(_make_save_dict())
        sd["metadata"]["hash_id"] = "0" * 32
        assert not check_valid_hash_v2(sd)

    def test_none_hash_fails(self):
        sd = update_model_data_v2(_make_save_dict())
        sd["metadata"]["hash_id"] = None
        assert not check_valid_hash_v2(sd)

    def test_empty_string_hash_fails(self):
        sd = update_model_data_v2(_make_save_dict())
        sd["metadata"]["hash_id"] = ""
        assert not check_valid_hash_v2(sd)

    def test_hash_from_different_model_fails(self):
        """Swapping hash_id from model B into model A must fail."""
        sd_a = update_model_data_v2(_make_save_dict(seed=1))
        sd_b = update_model_data_v2(_make_save_dict(seed=2))
        sd_a["metadata"]["hash_id"] = sd_b["metadata"]["hash_id"]
        assert not check_valid_hash_v2(sd_a)


# ---------------------------------------------------------------------------
# Negative: caller not mutated
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCallerNotMutated:
    def test_update_does_not_modify_caller_metadata(self):
        sd = _make_save_dict()
        original_meta = copy.deepcopy(sd["metadata"])
        update_model_data_v2(sd)
        if isinstance(sd["metadata"], dict):
            assert sd["metadata"].get("hash_id") == original_meta.get("hash_id")
        else:
            assert sd["metadata"].hash_id == original_meta.hash_id

    def test_update_does_not_convert_caller_tensors(self):
        """Caller's tensors must remain as torch.Tensor after update."""
        sd = _make_save_dict()
        update_model_data_v2(sd)
        for v in sd["network"].values():
            assert isinstance(v, torch.Tensor)

    def test_check_does_not_modify_caller(self):
        sd = update_model_data_v2(_make_save_dict())
        hash_before = sd["metadata"]["hash_id"]
        check_valid_hash_v2(sd)
        assert sd["metadata"]["hash_id"] == hash_before
        for v in sd["network"].values():
            assert isinstance(v, torch.Tensor)


# ---------------------------------------------------------------------------
# Edge cases: special tensor types
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBfloat16:
    def test_bfloat16_round_trip(self):
        sd = _make_save_dict()
        sd["network"]["weight"] = torch.randn(4, 4).to(torch.bfloat16)
        assert check_valid_hash_v2(update_model_data_v2(sd))

    def test_bfloat16_mutation_detected(self):
        sd = _make_save_dict()
        sd["network"]["weight"] = torch.ones(4, 4, dtype=torch.bfloat16)
        sd = update_model_data_v2(sd)
        sd["network"]["weight"][0, 0] = torch.tensor(2.0, dtype=torch.bfloat16)
        assert not check_valid_hash_v2(sd)


@pytest.mark.integration
class TestGradTensor:
    def test_requires_grad_does_not_raise(self):
        sd = _make_save_dict()
        sd["network"]["weight"] = torch.randn(4, 4, requires_grad=True)
        sd = update_model_data_v2(sd)
        assert check_valid_hash_v2(sd)


@pytest.mark.integration
class TestDeviceAgnostic:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_and_cpu_same_hash(self):
        """Same weights on CPU vs CUDA must produce the same hash."""
        fixed_meta = _make_v2_meta(model_id="fixed-id-for-determinism")
        sd_cpu = update_model_data_v2(_make_save_dict(meta=fixed_meta, seed=7), write_update=False)
        sd_gpu = _make_save_dict(meta=fixed_meta, seed=7)
        sd_gpu["network"]["weight"] = sd_gpu["network"]["weight"].cuda()
        sd_gpu["network"]["bias"] = sd_gpu["network"]["bias"].cuda()
        sd_gpu = update_model_data_v2(sd_gpu, write_update=False)
        assert sd_cpu["metadata"]["hash_id"] == sd_gpu["metadata"]["hash_id"]

    def test_cpu_hash_stable_across_calls(self):
        """Hashing the same dict twice on CPU must give the same result."""
        sd = _make_save_dict(seed=99)
        h1 = update_model_data_v2(copy.deepcopy(sd), write_update=False)["metadata"]["hash_id"]
        h2 = update_model_data_v2(copy.deepcopy(sd), write_update=False)["metadata"]["hash_id"]
        assert h1 == h2


# ---------------------------------------------------------------------------
# supports_stereochemistry field
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSupportsStereochemistry:
    def test_default_is_false(self):
        meta = _make_v2_meta()
        assert meta.supports_stereochemistry is False

    def test_can_be_set_true(self):
        meta = _make_v2_meta(supports_stereochemistry=True)
        assert meta.supports_stereochemistry is True

    def test_stored_and_hashed_false(self):
        sd = update_model_data_v2(_make_save_dict(meta=_make_v2_meta(supports_stereochemistry=False)))
        assert check_valid_hash_v2(sd)
        assert sd["metadata"]["supports_stereochemistry"] is False

    def test_stored_and_hashed_true(self):
        sd = update_model_data_v2(_make_save_dict(meta=_make_v2_meta(supports_stereochemistry=True)))
        assert check_valid_hash_v2(sd)
        assert sd["metadata"]["supports_stereochemistry"] is True

    def test_mutation_detected(self):
        """Flipping supports_stereochemistry after hashing must invalidate the hash."""
        sd = update_model_data_v2(_make_save_dict(meta=_make_v2_meta(supports_stereochemistry=False)))
        sd["metadata"]["supports_stereochemistry"] = True
        assert not check_valid_hash_v2(sd)

    def test_false_and_true_produce_different_hashes(self):
        sd_false = update_model_data_v2(_make_save_dict(meta=_make_v2_meta(supports_stereochemistry=False)))
        sd_true = update_model_data_v2(_make_save_dict(meta=_make_v2_meta(supports_stereochemistry=True)))
        assert sd_false["metadata"]["hash_id"] != sd_true["metadata"]["hash_id"]

    def test_survives_file_round_trip(self, tmp_path):
        sd = update_model_data_v2(_make_save_dict(meta=_make_v2_meta(supports_stereochemistry=True)))
        path = tmp_path / "model.pt"
        torch.save(sd, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert check_valid_hash_v2(loaded)
        assert loaded["metadata"]["supports_stereochemistry"] is True


# ---------------------------------------------------------------------------
# Pre-populated update_log at construction time
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPrePopulatedUpdateLog:
    def test_pre_populated_entry_survives_hash(self):
        """A log entry set at construction time must be covered by the hash."""
        initial_log = [{"timestamp": None, "comment": "initial", "reinvent_version": "4.0", "prior_hash": None}]
        meta = _make_v2_meta(update_log=initial_log)
        sd = update_model_data_v2(_make_save_dict(meta=meta), write_update=False, comment="")
        assert check_valid_hash_v2(sd)

    def test_pre_populated_comment_present_after_hash(self):
        initial_log = [{"timestamp": None, "comment": "from_config", "reinvent_version": "4.0", "prior_hash": None}]
        meta = _make_v2_meta(update_log=initial_log)
        sd = update_model_data_v2(_make_save_dict(meta=meta), write_update=False, comment="")
        assert sd["metadata"]["update_log"][0]["comment"] == "from_config"

    def test_pre_populated_entry_detected_if_tampered(self):
        initial_log = [{"timestamp": None, "comment": "original", "reinvent_version": "4.0", "prior_hash": None}]
        meta = _make_v2_meta(update_log=initial_log)
        sd = update_model_data_v2(_make_save_dict(meta=meta), write_update=False, comment="")
        sd["metadata"]["update_log"][0]["comment"] = "tampered"
        assert not check_valid_hash_v2(sd)

    def test_subsequent_update_appends_to_pre_populated_log(self):
        initial_log = [{"timestamp": None, "comment": "created", "reinvent_version": "4.0", "prior_hash": None}]
        meta = _make_v2_meta(update_log=initial_log)
        sd = _make_save_dict(meta=meta)
        sd = update_model_data_v2(sd, comment="second")
        assert len(sd["metadata"]["update_log"]) == 2
        assert sd["metadata"]["update_log"][0]["comment"] == "created"
        assert sd["metadata"]["update_log"][1]["comment"] == "second"
        assert check_valid_hash_v2(sd)

    def test_empty_pre_populated_log_behaves_like_default(self):
        meta_empty = _make_v2_meta(update_log=[])
        meta_default = _make_v2_meta()
        sd_empty = update_model_data_v2(_make_save_dict(meta=meta_empty), write_update=False, comment="")
        sd_default = update_model_data_v2(_make_save_dict(meta=meta_default), write_update=False, comment="")
        assert len(sd_empty["metadata"]["update_log"]) == 0
        assert len(sd_default["metadata"]["update_log"]) == 0
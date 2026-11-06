"""Metadata V2 for model files: deterministic, pickle-free hashing.

Improvements over meta_data.py (V1):
  - No pickle: canonical JSON with tensor bytes hex-encoded (big-endian normalised)
  - Tensors: detach().cpu().contiguous() → numpy → big-endian byte order → hex
    eliminates all device/grad-fn/pickle-protocol dependencies
  - Symmetric update/check code paths (V1 had an asymmetry bug where legacy models
    were pickled as a dataclass object in check but as a plain dict in update)
  - detach() before .numpy() prevents RuntimeError on grad-carrying tensors
  - Extended provenance: reinvent_version, torch_version, training_platform, etc.
  - update_log replaces the parallel (updates, comments) design: each entry carries
    prior_hash, forming a verifiable audit chain
  - original_data_source: spelling corrected from V1's origina_data_source
  - update_model_data_v2 never modifies the caller's tensors (no more in-place
    tensor→numpy conversion with a "do not use original" warning)

Routing V1 vs V2
    check_valid_hash_dispatch(save_dict) detects the version automatically.
    V2 hashes are identified by "canonical_json_v1" in hash_id_format.

Canonical serialisation format (canonical_json_v1)
    Every value in save_dict is recursively converted to a JSON-serialisable form:
      torch.Tensor  → {"__type__": "tensor",  "dtype": ">fN", "shape": [...], "hex": "..."}
      np.ndarray    → {"__type__": "tensor",  "dtype": ">fN", "shape": [...], "hex": "..."}
      dict          → keys sorted lexicographically at every nesting level
      UUID          → str
      nan/inf       → {"__type__": "float_special", "value": "nan"|"inf"|"-inf"}
      opaque object → {"__type__": "opaque", "class": "mod.Cls", "repr": repr(obj)}
                       NOTE: stable only when __repr__ is deterministic for that class
    The resulting structure is serialised with json.dumps(sort_keys=True) and
    UTF-8 encoded before hashing with xxhash.xxh3_128.
    Tensor bytes are in big-endian order so the hash is identical on little-endian
    (x86 / ARM) and big-endian machines.
"""

from __future__ import annotations

__all__ = [
    "ModelMetaDataV2",
    "update_model_data_v2",
    "check_valid_hash_v2",
    "check_valid_hash_dispatch",
]

import copy
import json
import platform
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from uuid import UUID

import numpy as np
import torch
import xxhash

try:
    from reinvent.version import __version__ as _REINVENT_VERSION
except ImportError:
    _REINVENT_VERSION = "unknown"

_HASH_FORMAT_V2 = f"xxhash.xxh3_128_hex/{xxhash.VERSION}"


@dataclass
class ModelMetaDataV2:
    """Metadata for model files supporting data provenance and lineage.

    hash_id and hash_id_format are nulled out when computing the hash and must
    not be used as inputs to hash computation.
    model_id and creation_date should be set once at initial training time.
    """

    hash_id: Optional[str]
    hash_id_format: str = _HASH_FORMAT_V2
    version: int = 2

    # --- identity ----------------------------------------------------------------
    model_id: str = ""                # str(uuid.uuid4()), set once at training time
    original_data_source: str = ""    # e.g. "ChEMBL 33" or "PubChem 2023-11-23"
    creation_date: float = field(default_factory=time.time)
    date_format: str = "UNIX epoch"
    model_id_format: str = field(
        default_factory=lambda: f"uuid.uuid4 {platform.python_version()}"
    )

    supports_stereochemistry: bool = False

    # --- environment snapshot at creation/last update time -----------------------
    reinvent_version: str = field(default_factory=lambda: _REINVENT_VERSION)
    torch_version: str = field(default_factory=lambda: torch.__version__)
    python_version: str = field(default_factory=platform.python_version)
    training_platform: str = field(default_factory=platform.platform)

    # --- lineage -----------------------------------------------------------------
    model_architecture: str = ""           # e.g. "RNN", "Transformer"; set by caller
    dataset_hash: Optional[str] = None     # hash of the training data file(s)

    # --- history -----------------------------------------------------------------
    # Replaces V1's parallel (updates: List[float], comments: List[str]).
    # Each entry: {timestamp: float|None, comment: str,
    #              reinvent_version: str, prior_hash: str|None}
    # prior_hash links entries into a verifiable audit chain.
    update_log: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def check_valid_hash(self, save_dict: dict) -> bool:
        return check_valid_hash_v2(save_dict)


# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def update_model_data_v2(
    save_dict: dict,
    comment: str = "",
    write_update: bool = True,
) -> dict:
    """Compute and store a deterministic V2 hash of the model state.

    Returns a shallow copy of save_dict whose metadata key is replaced with a
    plain dict containing the new hash.  The caller's save_dict is not modified
    and no in-place tensor conversion is performed.

    :param save_dict: full model save dictionary
    :param comment: free-text annotation for this update
    :param write_update: when True, record a timestamped entry in update_log
    :returns: updated save_dict with metadata as plain dict and hash stored
    """
    save_dict = {k: v for k, v in sorted(save_dict.items())}
    metadata = copy.deepcopy(save_dict["metadata"])

    if not isinstance(metadata, dict):
        metadata = metadata.as_dict()

    # Migrate V1-only fields to their V2 equivalents (idempotent)
    if "origina_data_source" in metadata:
        metadata.setdefault("original_data_source", metadata.pop("origina_data_source"))
    if "updates" in metadata or "comments" in metadata:
        updates = metadata.pop("updates", [])
        comments = metadata.pop("comments", [])
        if "update_log" not in metadata:
            metadata["update_log"] = []
        for ts, cmt in zip(updates, comments + [None] * len(updates)):
            metadata["update_log"].append(
                {"timestamp": ts, "comment": cmt or "", "reinvent_version": "", "prior_hash": None}
            )

    # Ensure V1 metadata promoted to V2 by this function is versioned correctly
    metadata["version"] = 2

    prior_hash = metadata.get("hash_id")
    metadata["hash_id"] = None
    metadata["hash_id_format"] = None

    # Guard for models migrated from V1 (no update_log key yet)
    if "update_log" not in metadata:
        metadata["update_log"] = []

    if write_update or comment:
        metadata["update_log"].append(
            {
                "timestamp": time.time() if write_update else None,
                "comment": comment,
                "reinvent_version": _REINVENT_VERSION,
                "prior_hash": prior_hash,
            }
        )

    save_dict["metadata"] = metadata
    data = _canonical_bytes(save_dict)
    metadata["hash_id"] = xxhash.xxh3_128_hexdigest(data)
    metadata["hash_id_format"] = _HASH_FORMAT_V2

    return save_dict


def check_valid_hash_v2(save_dict: dict) -> bool:
    """Verify the stored V2 hash of a model file.

    Deep-copies save_dict internally so the caller's dict is not modified.

    :param save_dict: full model save dictionary
    :returns: True if the recomputed hash matches the stored hash
    """
    save_dict = copy.deepcopy(save_dict)
    save_dict = {k: v for k, v in sorted(save_dict.items())}
    metadata = save_dict["metadata"]

    if isinstance(metadata, dict):
        curr_hash_id = metadata.get("hash_id")
        metadata["hash_id"] = None
        metadata["hash_id_format"] = None
    else:
        # V2 metadata is always stored as a dict after update_model_data_v2,
        # but handle dataclass objects defensively.
        curr_hash_id = metadata.hash_id
        metadata = metadata.as_dict()
        metadata["hash_id"] = None
        metadata["hash_id_format"] = None
        save_dict["metadata"] = metadata

    data = _canonical_bytes(save_dict)
    return curr_hash_id == xxhash.xxh3_128_hexdigest(data)


def check_valid_hash_dispatch(save_dict: dict) -> bool:
    """Route hash verification to V2 or V1 based on the metadata version field.

    Reads metadata.version (int): 2 → V2 checker, anything else → V1 checker.
    Falls back to version 1 when the field is absent (legacy models).
    """
    metadata = save_dict.get("metadata", {})
    version = (
        metadata.get("version", 1)
        if isinstance(metadata, dict)
        else getattr(metadata, "version", 1)
    )

    if version == 2:
        return check_valid_hash_v2(save_dict)

    from reinvent.models.meta_data import check_valid_hash as _check_v1
    return _check_v1(save_dict)


# -----------------------------------------------------------------------------
# Canonical serialisation (internal)
# -----------------------------------------------------------------------------

def _canonical_bytes(save_dict: dict) -> bytes:
    """Produce a deterministic byte string suitable for hashing.

    Calls _to_canonical recursively and serialises with json.dumps so that
    the output is unambiguous, human-readable, and reproducible in other
    languages given the same input data.
    """
    return json.dumps(
        _to_canonical(save_dict),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _to_canonical(obj: object) -> object:
    """Recursively map obj to a JSON-serialisable canonical form."""

    # --- JSON primitives ---------------------------------------------------------
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        # JSON does not support nan / inf; use tagged dicts instead
        if obj != obj:
            return {"__type__": "float_special", "value": "nan"}
        if obj == float("inf"):
            return {"__type__": "float_special", "value": "inf"}
        if obj == float("-inf"):
            return {"__type__": "float_special", "value": "-inf"}
        return obj

    # --- UUID → string -----------------------------------------------------------
    if isinstance(obj, UUID):
        return str(obj)

    # --- torch.Tensor ------------------------------------------------------------
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().contiguous()
        # bfloat16 has no numpy native type; reinterpret raw bits as int16
        # (same bit width, avoids lossy float32 promotion)
        if arr.dtype == torch.bfloat16:
            np_arr = arr.view(torch.int16).numpy()
        else:
            np_arr = arr.numpy()
        return _numpy_canonical(np_arr)

    # --- numpy.ndarray -----------------------------------------------------------
    if isinstance(obj, np.ndarray):
        return _numpy_canonical(obj)

    # --- containers --------------------------------------------------------------
    if isinstance(obj, dict):
        return {str(k): _to_canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_to_canonical(v) for v in obj]

    # --- fallback for opaque objects (e.g. tokenizer class instances) -----------
    # For stateless objects (empty __dict__) only the class identity matters;
    # repr() is skipped because the default repr includes a memory address.
    # For objects with instance state, repr() is used as a best-effort; the
    # hash is stable only when __repr__ is deterministic (no memory address).
    cls_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
    state = vars(obj) if hasattr(obj, "__dict__") else None
    if not state:
        return {"__type__": "opaque", "class": cls_name}
    return {
        "__type__": "opaque",
        "class": cls_name,
        "repr": repr(obj),
    }


def _numpy_canonical(arr: np.ndarray) -> dict:
    """Canonical dict for a numpy array.

    Multi-byte dtypes are normalised to big-endian so the hash is identical
    on little-endian (x86 / ARM) and big-endian machines.
    Single-byte dtypes (uint8, int8, bool) need no reordering.
    The hex string encodes the raw bytes in the normalised byte order.
    """
    if arr.dtype.itemsize > 1:
        arr = arr.astype(arr.dtype.newbyteorder(">"))
    return {
        "__type__": "tensor",
        "dtype": str(arr.dtype),   # e.g. ">f4" for big-endian float32
        "shape": list(arr.shape),
        "hex": arr.tobytes().hex(),
    }

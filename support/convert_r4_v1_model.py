#!/bin/env python3
"""Convert a REINVENT 4 V1 model (ModelMetaData hash) to V2 (canonical_json_v1 hash).

V1 models use pickle-based hashing and the ModelMetaData dataclass.  V2 models
use a deterministic, pickle-free canonical JSON serialisation described in
reinvent/models/meta_data2.py.

What this script does
---------------------
1. Load the model file onto CPU.
2. Detect whether the file already carries a V2 hash or has no metadata at all.
3. Extract every V1 metadata field and carry it over to a ModelMetaDataV2 object:
   - origina_data_source  →  original_data_source  (typo corrected)
   - updates (List[float]) + comments (List[str]) → update_log (List[dict])
     Entries are merged pairwise; trailing entries in the longer list are kept.
     prior_hash is set to None for all migrated entries because the V1 hash
     algorithm is not reproducible in V2 (different serialisation).
   - model_architecture is inferred from save_dict["model_type"] when available.
   - reinvent_version, torch_version, python_version, training_platform are filled
     with the values at conversion time, not the original training environment.
     A note to that effect is appended to update_log.
4. Run update_model_data_v2 to compute and store the V2 hash.
5. Save the result to the output file.
6. Reload and verify the hash as a final sanity check.

Usage
-----
  python support/convert_r4_v1_model.py INPUT_MODEL OUTPUT_MODEL [options]

Options
-------
  -s / --source      Override original_data_source (default: keep V1 value)
  -c / --comment     Extra comment appended to the conversion update_log entry
  -a / --architecture Override model_architecture (default: inferred from model_type)
  --dataset-hash     Hash of the training dataset for provenance (optional)
  --stereo           Set supports_stereochemistry=True (default: False)
"""

import logging
import os
import platform
import sys
import time
from itertools import zip_longest
from typing import Optional, TextIO

import torch

from reinvent.models.meta_data import ModelMetaData
from reinvent.models.meta_data2 import (
    ModelMetaDataV2,
    check_valid_hash_v2,
    update_model_data_v2,
)
from reinvent.version import __version__ as _REINVENT_VERSION


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _setup_logger(stream: TextIO = sys.stderr) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s <%(levelname)-4.4s> %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=f"{sys.argv[0]}: convert a REINVENT 4 V1 model to V2 metadata format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "The output file is a drop-in replacement for the input file. "
            "All network weights and vocabulary data are preserved verbatim."
        ),
    )

    parser.add_argument(
        "input_model_file",
        type=os.path.abspath,
        help="REINVENT 4 model with V1 (pickle-based) metadata",
    )
    parser.add_argument(
        "output_model_file",
        type=os.path.abspath,
        help="Output model file with V2 (canonical JSON) metadata",
    )
    parser.add_argument(
        "-s", "--source",
        default=None,
        metavar="TEXT",
        help="Override original_data_source (default: keep value from V1 metadata)",
    )
    parser.add_argument(
        "-c", "--comment",
        default="",
        metavar="TEXT",
        help="Extra comment recorded in the conversion update_log entry",
    )
    parser.add_argument(
        "-a", "--architecture",
        default=None,
        metavar="NAME",
        help="Override model_architecture (default: inferred from model_type key)",
    )
    parser.add_argument(
        "--dataset-hash",
        default=None,
        metavar="HASH",
        help="Hash of the training dataset(s) for provenance",
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        default=False,
        help="Set supports_stereochemistry=True in the V2 metadata (default: False)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# V1 metadata extraction
# ---------------------------------------------------------------------------

def _extract_v1_fields(metadata) -> dict:
    """Return a plain dict of V1 metadata fields regardless of storage form."""
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, ModelMetaData):
        return metadata.as_dict()
    # Fallback: try asdict / __dict__ for any other dataclass-like object
    if hasattr(metadata, "as_dict"):
        return metadata.as_dict()
    if hasattr(metadata, "__dict__"):
        return dict(metadata.__dict__)
    raise TypeError(f"Unrecognised metadata type: {type(metadata)!r}")


def _is_v2(metadata) -> bool:
    """Return True when the metadata version field identifies a V2 model.

    Reads metadata.version (int == 2).  Falls back to inspecting hash_id_format
    for models that were written before the version field was introduced.
    """
    version = (
        metadata.get("version", None)
        if isinstance(metadata, dict)
        else getattr(metadata, "version", None)
    )
    if version is not None:
        return version == 2
    # Fallback for pre-version-field models: inspect hash_id_format string
    fmt = (
        metadata.get("hash_id_format", "")
        if isinstance(metadata, dict)
        else getattr(metadata, "hash_id_format", "")
    ) or ""
    return "canonical_json_v1" in fmt


def _merge_updates_and_comments(
    updates: list,
    comments: list,
) -> list:
    """Combine V1 parallel (updates, comments) lists into V2 update_log entries.

    V1 stores timestamps and comments as independent, potentially unequal lists.
    We merge them pairwise; trailing entries in the longer list become their own
    log entries.  prior_hash is always None for migrated entries because the V1
    hash cannot be reproduced with the V2 algorithm.
    """
    log = []
    for ts, comment in zip_longest(updates, comments):
        # ts can be a float, a tuple, or None (if comments list is longer)
        timestamp: Optional[float] = None
        if isinstance(ts, (int, float)):
            timestamp = float(ts)
        elif isinstance(ts, (list, tuple)) and ts:
            # guard against the List[Tuple] annotation in V1 even though
            # time.time() (a float) is actually appended
            try:
                timestamp = float(ts[0])
            except (TypeError, ValueError):
                timestamp = None

        log.append(
            {
                "timestamp": timestamp,
                "comment": comment or "",
                "reinvent_version": "unknown (migrated from V1)",
                "prior_hash": None,
            }
        )
    return log


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_model(
    in_filename: str,
    out_filename: str,
    source_override: Optional[str] = None,
    comment: str = "",
    architecture_override: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    supports_stereochemistry: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Convert a V1 model file to V2 metadata format and write the result.

    :param in_filename: path to the V1 model file
    :param out_filename: path to write the V2 model file
    :param source_override: if given, replaces original_data_source from V1
    :param comment: extra user comment for the conversion log entry
    :param architecture_override: optional model architecture name
    :param dataset_hash: optional hash of the training dataset
    :param supports_stereochemistry: whether the model supports stereochemistry
    :param logger: logger; a default stderr logger is used when None
    """

    if logger is None:
        logger = _setup_logger()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    logger.info(f"Loading {in_filename}")
    save_dict = torch.load(in_filename, map_location=torch.device("cpu"), weights_only=False)

    if "metadata" not in save_dict:
        logger.error(
            "Model file has no 'metadata' key — run support/add_meta_data.py first "
            "to add V1 metadata, then re-run this script."
        )
        sys.exit(1)

    raw_meta = save_dict["metadata"]

    # ------------------------------------------------------------------
    # 2. Guard: already V2?
    # ------------------------------------------------------------------
    if _is_v2(raw_meta):
        logger.warning(
            "Model already carries a V2 (canonical_json_v1) hash — nothing to do. "
            "The output file will not be written."
        )
        return

    # ------------------------------------------------------------------
    # 3. Extract V1 fields
    # ------------------------------------------------------------------
    v1 = _extract_v1_fields(raw_meta)
    logger.info(
        f"Found V1 metadata: model_id={v1.get('model_id')!r}, "
        f"hash_id={v1.get('hash_id')!r}"
    )

    updates = v1.get("updates") or []
    comments_v1 = v1.get("comments") or []
    migrated_log = _merge_updates_and_comments(updates, comments_v1)
    logger.info(
        f"Migrated {len(updates)} update timestamp(s) and "
        f"{len(comments_v1)} comment(s) into {len(migrated_log)} update_log entry/entries"
    )

    # Infer architecture from top-level model_type when not overridden
    architecture = architecture_override or save_dict.get("model_type", "")

    # Resolve data source: CLI override > V1 value > empty string
    # Note: V1 has a typo "origina_data_source"
    original_source = (
        source_override
        if source_override is not None
        else v1.get("origina_data_source") or v1.get("original_data_source") or ""
    )

    # ------------------------------------------------------------------
    # 4. Build ModelMetaDataV2
    # ------------------------------------------------------------------
    # reinvent_version, torch_version, python_version, training_platform
    # reflect the conversion environment, not the original training environment.
    # This is unavoidable for V1 models; we record this fact in the log.
    conversion_note = (
        "converted from R4 V1 (pickle-based) metadata format; "
        "reinvent_version/torch_version/python_version/training_platform reflect "
        "the conversion environment, not the original training environment"
    )
    if comment:
        conversion_note = f"{conversion_note}; {comment}"

    v2_meta = ModelMetaDataV2(
        hash_id=None,
        model_id=str(v1.get("model_id") or ""),
        original_data_source=original_source,
        creation_date=float(v1.get("creation_date") or time.time()),
        date_format=v1.get("date_format") or "UNIX epoch",
        model_id_format=v1.get("model_id_format") or f"uuid.uuid4 {platform.python_version()}",
        reinvent_version=_REINVENT_VERSION,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        training_platform=platform.platform(),
        model_architecture=architecture,
        supports_stereochemistry=supports_stereochemistry,
        dataset_hash=dataset_hash,
        update_log=migrated_log,  # pre-populated history from V1
    )

    save_dict["metadata"] = v2_meta
    save_dict.pop("tokenizer", None)  # fixed, stateless — not stored in V2 format

    # ------------------------------------------------------------------
    # 5. Hash and save
    # ------------------------------------------------------------------
    # update_model_data_v2 will append one final log entry (the conversion event)
    # and compute the V2 hash.
    logger.info("Computing V2 hash")
    save_dict = update_model_data_v2(save_dict, comment=conversion_note, write_update=True)

    logger.info(f"Writing {out_filename}")
    torch.save(save_dict, out_filename)

    # ------------------------------------------------------------------
    # 6. Verify
    # ------------------------------------------------------------------
    logger.info("Verifying V2 hash")
    saved = torch.load(out_filename, map_location=torch.device("cpu"), weights_only=False)

    if check_valid_hash_v2(saved):
        new_hash = saved["metadata"].get("hash_id", "<unknown>")
        logger.info(f"Hash OK: {new_hash}")
    else:
        logger.error("Hash verification FAILED — output file may be corrupt")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logger = _setup_logger()
    args = _parse_args()

    convert_model(
        in_filename=args.input_model_file,
        out_filename=args.output_model_file,
        source_override=args.source,
        comment=args.comment,
        architecture_override=args.architecture,
        dataset_hash=args.dataset_hash,
        supports_stereochemistry=args.stereo,
        logger=logger,
    )


if __name__ == "__main__":
    main()

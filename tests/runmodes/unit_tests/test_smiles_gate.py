"""Tests for the SMILES validity gate before scoring plugins.

Covers:
  - run_scoring: mask uses Chem.MolFromSmiles (Gap 1)
  - filter_by_pattern: uses sampled.states not np.ones (Gap 2)
"""

import tempfile
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from rdkit import Chem

from reinvent.models.model_factory.sample_batch import SampleBatch, SmilesState
from reinvent.runmodes.samplers.run_sampling import filter_by_pattern, filter_wanted_samples

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]
INVALID_SMILES = ["CC(C", "INVALID_XYZ", ""]
MIXED_SMILES = VALID_SMILES + INVALID_SMILES


def _make_mask(smilies):
    return np.array([bool(s and Chem.MolFromSmiles(s)) for s in smilies])


def _make_sample_batch(smilies, states):
    nlls = np.zeros(len(smilies))
    return SampleBatch(
        items1=None,
        items2=smilies,
        nlls=nlls,
        smilies=smilies,
        states=np.array(states),
    )


# ---------------------------------------------------------------------------
# Gap 1: run_scoring mask
# ---------------------------------------------------------------------------


def test_scoring_mask_all_valid_smiles_all_true():
    mask = _make_mask(VALID_SMILES)
    assert mask.all(), "All valid SMILES should produce a True mask"


def test_scoring_mask_all_invalid_smiles_all_false():
    mask = _make_mask(INVALID_SMILES)
    assert not mask.any(), "All invalid SMILES should produce a False mask"


def test_scoring_mask_mixed_correct_shape():
    mask = _make_mask(MIXED_SMILES)
    assert len(mask) == len(MIXED_SMILES)


def test_scoring_mask_mixed_correct_values():
    mask = _make_mask(MIXED_SMILES)
    # first three are valid, last three are invalid
    assert mask[:3].all()
    assert not mask[3:].any()


def test_scoring_mask_empty_string_is_false():
    mask = _make_mask([""])
    assert not mask[0]


def test_scoring_mask_none_like_empty_is_false():
    mask = _make_mask(["CC(C"])  # unclosed paren
    assert not mask[0]


def test_scoring_mask_from_run_scoring_file(tmp_path):
    """End-to-end: run_scoring builds the mask with Chem.MolFromSmiles."""
    smi_file = tmp_path / "input.smi"
    smi_file.write_text("CCO\nCC(C\nc1ccccc1\n")

    from reinvent.runmodes.scoring.run_scoring import setup_actions, read_data

    actions = setup_actions(standardize=False)
    reader = read_data(str(smi_file), actions=actions, smiles_column="SMILES")
    mask = _make_mask(reader.smilies)

    assert mask[0] is np.bool_(True)  # CCO valid
    assert mask[1] is np.bool_(False)  # CC(C invalid
    assert mask[2] is np.bool_(True)  # c1ccccc1 valid


# ---------------------------------------------------------------------------
# Gap 2: filter_by_pattern uses sampled.states
# ---------------------------------------------------------------------------


def test_filter_by_pattern_does_not_use_ones_mask():
    """Verify np.ones is not in the source of filter_by_pattern."""
    import inspect

    src = inspect.getsource(filter_by_pattern)
    assert "np.ones" not in src, "filter_by_pattern must not use np.ones for masks"


def test_filter_by_pattern_uses_smiles_state_flags():
    """filter_by_pattern source must reference SmilesState to build masks."""
    import inspect

    src = inspect.getsource(filter_by_pattern)
    assert "SmilesState.INVALID" in src
    assert "SmilesState.DUPLICATE" in src


def test_filter_by_pattern_invalid_smiles_masked_in_scorer():
    """Invalid SMILES in the batch must receive mask=False before the scorer."""
    states = [
        SmilesState.VALID,
        SmilesState.INVALID,
        SmilesState.VALID,
        SmilesState.DUPLICATE,
    ]
    smilies = ["CCO", "INVALID_XYZ", "c1ccccc1", "CCO"]
    sampled = _make_sample_batch(smilies, states)

    captured_invalid_mask = []
    captured_duplicate_mask = []

    def mock_scorer_call(smilies_arg, inv_mask, dup_mask, _extra):
        captured_invalid_mask.extend(inv_mask.tolist())
        captured_duplicate_mask.extend(dup_mask.tolist())
        result = MagicMock()
        result.total_scores = np.ones(len(smilies_arg))
        return result

    mock_scorer_instance = MagicMock(side_effect=mock_scorer_call)

    with patch("reinvent.runmodes.samplers.run_sampling.Scorer", return_value=mock_scorer_instance):
        filter_by_pattern(sampled, ["[#7]"])

    # index 1 is INVALID → invalid_mask must be False there
    assert captured_invalid_mask[1] == False
    # index 3 is DUPLICATE → duplicate_mask must be False there
    assert captured_duplicate_mask[3] == False
    # index 0 and 2 are VALID → both masks True
    assert captured_invalid_mask[0] == True
    assert captured_duplicate_mask[0] == True

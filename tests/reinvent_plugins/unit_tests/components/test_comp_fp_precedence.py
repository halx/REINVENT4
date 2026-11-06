import json
import pytest
import numpy as np
import numpy.testing as npt

from unittest.mock import mock_open, patch
from reinvent_plugins.components.database_precedence.comp_fp_precedence import (
    FingerprintPrecedence,
    Parameters,
)


RADIUS = 2
FPSIZE = 64  # small fpsize for testing


@pytest.fixture
def mock_db():
    """Build a small mock fp precedence database as a JSON string.

    Most bits get nll-pos=1.0 (common).
    Bit 5 (hit by benzene, pyridine, acetic acid) gets nll-pos=15.0.
    Bit 33 (hit by CCO, acetic acid, hexane) gets nll-pos=25.0.
    """
    nll_pos = [1.0] * FPSIZE

    nll_pos[5] = 15.0   # hit by benzene, pyridine, acetic acid
    nll_pos[33] = 25.0  # hit by CCO, acetic acid, hexane

    most_common_fragment = [None] * FPSIZE
    most_common_fragment[5] = "c1ccccc1"
    most_common_fragment[33] = "CCO"

    db = {
        "test_fp": {
            "nll-pos": nll_pos,
            "most_common_fragment": most_common_fragment,
            "fp_type": "morgan",
            "radius": RADIUS,
            "fpsize": FPSIZE,
        }
    }
    return json.dumps(db)


def _make_component(mock_db_string, nll_method="total"):
    mock_file = mock_open(read_data=mock_db_string)
    with patch("builtins.open", mock_file):
        params = Parameters(
            database_files=[["mocked/path"]],
            nll_method=[nll_method],
        )
        return FingerprintPrecedence(params)


def test_total_scores_and_fragments(mock_db):
    """Total score = dot(fp, nll_pos) + dot(1-fp, nll_neg) with exact expected values."""
    component = _make_component(mock_db, nll_method="total")

    smiles = ["c1ccccc1", "CCO", "CC(=O)O", "c1ccncc1", "CCCCCC", "INVALID_SMILES"]
    expected_total = [44.5205, 56.1445, 71.1445, 47.2271, 56.6858, 0.0]
    expected_frags = ["ccccc", "C", "C", "ccccc", "C", None]

    results = component(smiles)

    scores = results.scores[0]
    assert len(scores) == len(smiles)
    for actual, expected in zip(scores, expected_total):
        npt.assert_approx_equal(actual, expected, significant=4)

    frags = results.metadata["highest_nll_bit_fragment"]
    assert frags == expected_frags


def test_max_scores(mock_db):
    """Max score = max nll-pos among set bits; total >= max for every molecule."""
    comp_total = _make_component(mock_db, nll_method="total")
    comp_max = _make_component(mock_db, nll_method="max")

    smiles = ["c1ccccc1", "CCO", "c1ccncc1", "INVALID_SMILES"]
    expected_max = [15.0, 25.0, 15.0, 0.0]

    results_total = comp_total(smiles)
    results_max = comp_max(smiles)

    for actual, expected in zip(results_max.scores[0], expected_max):
        npt.assert_approx_equal(actual, expected, significant=4)

    for t, m in zip(results_total.scores[0], results_max.scores[0]):
        assert t >= m



def test_db_without_optional_keys():
    """Component should work with a DB missing most_common_fragment (fp_type, fpsize, radius are required)."""
    nll_pos = [1.0] * FPSIZE
    db = {
        "minimal_db": {
            "nll-pos": nll_pos,
            "fp_type": "morgan",
            "radius": RADIUS,
            "fpsize": FPSIZE,
        }
    }
    db_string = json.dumps(db)
    component = _make_component(db_string, nll_method="total")

    smiles = ["c1ccccc1"]
    results = component(smiles)

    assert len(results.scores[0]) == 1
    assert results.scores[0][0] > 0


def test_db_missing_required_keys():
    """Component should raise KeyError when fp_type, radius, or fpsize are absent."""
    nll_pos = [1.0] * FPSIZE

    for missing_key in ("fp_type", "radius", "fpsize"):
        db = {
            "minimal_db": {
                "nll-pos": nll_pos,
                "fp_type": "morgan",
                "radius": RADIUS,
                "fpsize": FPSIZE,
            }
        }
        del db["minimal_db"][missing_key]
        db_string = json.dumps(db)
        with pytest.raises(KeyError):
            _make_component(db_string)


def test_concatenated_fps_score_additivity(mock_db):
    """Total score for two concatenated DBs equals the sum of their individual scores."""
    nll_pos2 = [2.0] * FPSIZE
    nll_pos2[10] = 20.0
    db2_str = json.dumps({
        "test_fp2": {
            "nll-pos": nll_pos2,
            "fp_type": "morgan",
            "radius": RADIUS,
            "fpsize": FPSIZE,
        }
    })

    smiles = ["c1ccccc1", "CCO", "INVALID_SMILES"]
    comp1 = _make_component(mock_db,  nll_method="total")
    comp2 = _make_component(db2_str, nll_method="total")

    db_strings = [mock_db, db2_str]
    idx = [0]
    def open_se(*a, **kw):
        m = mock_open(read_data=db_strings[idx[0]])
        idx[0] += 1
        return m()

    with patch("builtins.open", side_effect=open_se):
        params = Parameters(database_files=[["mock/0.json", "/mock/1.json"]], nll_method=["total"])
        comp_cat = FingerprintPrecedence(params)

    s1   = comp1(smiles).scores[0]
    s2   = comp2(smiles).scores[0]
    scat = comp_cat(smiles).scores[0]

    for v1, v2, vc in zip(s1, s2, scat):
        npt.assert_approx_equal(vc, v1 + v2, significant=4)

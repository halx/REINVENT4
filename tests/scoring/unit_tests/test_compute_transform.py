"""Tests for compute_transform in reinvent/scoring/compute_scores.py."""

import numpy as np
import pytest

from reinvent.scoring.compute_scores import compute_transform
from reinvent.scoring.transforms import Sigmoid, ReverseSigmoid
from reinvent.scoring.transforms.sigmoids import Parameters as SigParams
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.components.RDKit.comp_physchem import MolecularWeight

SMILIES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCC"]
VALID_MASK = np.array([True, True, True])


def _make_sigmoid():
    params = SigParams(type="sigmoid", low=0.0, high=500.0, k=0.5)
    return Sigmoid(params)


def test_compute_transform_with_sigmoid_returns_bounded_scores():
    transform = _make_sigmoid()
    mw = MolecularWeight()
    cache = {}

    result = compute_transform(
        component_type="molecularweight",
        params=(["MW"], mw, [transform], [1.0], [False]),
        smilies=SMILIES,
        cache=cache,
        valid_mask=VALID_MASK,
    )

    scores = result.transformed_scores[0]
    assert len(scores) == len(SMILIES)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_compute_transform_with_no_transform_is_raw_scores():
    mw = MolecularWeight()
    cache = {}

    result = compute_transform(
        component_type="molecularweight",
        params=(["MW"], mw, [None], [1.0], [False]),
        smilies=SMILIES,
        cache=cache,
        valid_mask=VALID_MASK,
    )

    scores = result.transformed_scores[0]
    # raw molecular weights — should be > 1 for all real molecules
    assert np.all(scores[VALID_MASK] > 1.0)


def test_compute_transform_masks_invalid_smiles():
    mw = MolecularWeight()
    cache = {}
    smilies = ["CC(=O)Oc1ccccc1C(=O)O", "INVALID_XYZ", "c1ccccc1"]
    valid_mask = np.array([True, False, True])

    result = compute_transform(
        component_type="molecularweight",
        params=(["MW"], mw, [None], [1.0], [False]),
        smilies=smilies,
        cache=cache,
        valid_mask=valid_mask,
    )

    scores = result.transformed_scores[0]
    assert scores[1] == 0.0  # masked → zero


def test_compute_transform_populates_cache():
    mw = MolecularWeight()
    cache = {}

    compute_transform(
        component_type="molecularweight",
        params=(["MW"], mw, [None], [1.0], [False]),
        smilies=SMILIES,
        cache=cache,
        valid_mask=VALID_MASK,
    )

    assert len(cache) == len(SMILIES)


def test_compute_transform_uses_cache_on_second_call():
    mw = MolecularWeight()
    cache = {}

    result1 = compute_transform(
        component_type="molecularweight",
        params=(["MW"], mw, [None], [1.0], [False]),
        smilies=SMILIES,
        cache=cache,
        valid_mask=VALID_MASK,
    )
    result2 = compute_transform(
        component_type="molecularweight",
        params=(["MW"], mw, [None], [1.0], [False]),
        smilies=SMILIES,
        cache=cache,
        valid_mask=VALID_MASK,
    )

    np.testing.assert_array_equal(
        result1.transformed_scores[0],
        result2.transformed_scores[0],
    )

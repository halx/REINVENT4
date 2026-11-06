"""Tests for reinvent/utils/config_parse.py pure-function utilities."""

import pytest
import tempfile
import json

from reinvent.utils.config_parse import (
    has_multiple_attachment_points_to_same_atom,
    find_invalid_tokens,
    validate_tokens,
)

# ── has_multiple_attachment_points_to_same_atom ──────────────────────────────


def test_multiple_attachment_points_same_atom_true():
    # Carbon atom carrying two dummy ([*]) neighbours
    smiles = "[*]C([*])CC"
    assert has_multiple_attachment_points_to_same_atom(smiles) is True


def test_multiple_attachment_points_different_atoms_false():
    smiles = "[*]CC[*]"
    assert has_multiple_attachment_points_to_same_atom(smiles) is False


def test_no_attachment_points_false():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    assert has_multiple_attachment_points_to_same_atom(smiles) is False


def test_single_attachment_point_false():
    smiles = "[*]c1ccccc1"
    assert has_multiple_attachment_points_to_same_atom(smiles) is False


def test_invalid_smiles_raises():
    with pytest.raises(RuntimeError):
        has_multiple_attachment_points_to_same_atom("INVALID_SMILES")


# ── find_invalid_tokens ───────────────────────────────────────────────────────


def test_find_invalid_tokens_detects_unknown_element():
    allowed = {"C", "c", "N", "O", "n", "(", ")", "=", "1", "2"}
    # Xe is not in allowed
    invalid = find_invalid_tokens("c1ccncc1[Xe]", allowed)
    assert len(invalid) > 0


def test_find_invalid_tokens_all_valid():
    allowed = {"C", "c", "N", "(", ")", "=", "1"}
    invalid = find_invalid_tokens("c1ccncc1", allowed | {"n"})
    assert len(invalid) == 0


def test_find_invalid_tokens_ignores_attachment_point_wildcards():
    """Tokens containing '*' should be ignored (LibInvent input format)."""
    allowed = {"C", "c", "(", ")", "1"}
    invalid = find_invalid_tokens("[*]c1ccccc1", allowed)
    # [*] contains *, so it is filtered from the check
    assert not any("*" in t for t in invalid)


# ── validate_tokens ───────────────────────────────────────────────────────────


def test_validate_tokens_passes_for_known_smiles():
    allowed_tokens = (
        {"C", "c", "N", "O", "n", "o", "(", ")", "=", "1", "2", "3", "#", "$", "^", "F", "S", "s"},
        set(),
    )
    # should not raise
    validate_tokens("c1ccccc1", allowed_tokens)


def test_validate_tokens_raises_for_unknown_element():
    allowed_tokens = ({"C", "c", "(", ")", "1"}, set())
    with pytest.raises(ValueError, match="not supported"):
        validate_tokens("[Xe]CC", allowed_tokens)


def test_validate_tokens_libinvent_two_vocabularies():
    """When allowed_tokens[1] is non-empty, both SMILES in the pair are checked."""
    voc1 = {"C", "c", "(", ")", "1", "[*]"}
    voc2 = {"C", "N", "(", ")", "="}
    allowed_tokens = (voc1, voc2)
    # Both sub-SMILES valid → no raise
    validate_tokens(["[*]c1ccccc1", "CN"], allowed_tokens)

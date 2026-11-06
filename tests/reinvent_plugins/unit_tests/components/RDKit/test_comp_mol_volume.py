import numpy as np

from reinvent_plugins.components.RDKit.comp_mol_volume import Parameters, MolVolume


def test_comp_mol_volume():
    params = Parameters([0.2, 2.0])
    mol_volume = MolVolume(params)

    # aniline (7 heavy atoms) and a larger pyridine derivative (12 heavy atoms)
    smiles = ["c1ccccc1N", "SCc1ccncc1O"]
    results = mol_volume(smiles)
    vols = np.concatenate(results.scores)

    # Volumes must be finite and physically plausible for small drug-like molecules.
    # Exact values are intentionally not asserted: RDKit's conformer generator can
    # produce slightly different embeddings across versions even with a fixed seed.
    assert np.all(np.isfinite(vols))
    assert np.all(vols > 0)
    assert np.all(vols < 500)

    # The molecule with more heavy atoms must have a larger volume.
    assert vols[1] > vols[0]


def test_comp_mol_volume_none_molecule_scores_nan():
    # an invalid SMILES (None molecule from molcache) must score NaN rather
    # than raising (issue #333)
    params = Parameters([0.2])
    component = MolVolume(params)

    results = component(["not_a_molecule", "c1ccccc1N"])

    assert np.isnan(results.scores[0][0])
    assert not np.isnan(results.scores[0][1])

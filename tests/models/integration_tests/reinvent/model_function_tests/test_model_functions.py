import unittest

import numpy.testing as npt
import pytest
import torch

from reinvent.models import ReinventAdapter
from reinvent.models.reinvent.models.model import Model
from reinvent.runmodes.utils.helpers import set_torch_device
from tests.test_data import PROPANE, BENZENE, METAMIZOLE


@pytest.mark.integration
@pytest.mark.usefixtures("device", "json_config")
class TestReinventModelFunctions(unittest.TestCase):
    def setUp(self):
        save_dict = torch.load(
            self.json_config["PRIOR_PATH"], map_location=self.device, weights_only=False
        )
        self.model = Model.create_from_dict(save_dict, "inference", torch.device(self.device))
        set_torch_device(self.device)
        self.adapter = ReinventAdapter(self.model)

    # --- vocabulary ---

    def test_vocabulary_tokens_contains_common_atoms(self):
        tokens = self.model.vocabulary.tokens()
        for atom in ("C", "O", "N", "c", "n"):
            self.assertIn(atom, tokens)

    def test_vocabulary_size(self):
        tokens = self.model.vocabulary.tokens()
        self.assertEqual(34, len(tokens))
        self.assertIsInstance(tokens, list)

    # --- likelihood_smiles ---

    def test_likelihood_smiles_shape(self):
        nlls = self.adapter.likelihood_smiles([PROPANE, BENZENE])
        self.assertEqual([2], list(nlls.shape))
        self.assertIsInstance(nlls, torch.Tensor)

    def test_likelihood_smiles_values(self):
        nlls = self.adapter.likelihood_smiles([PROPANE, BENZENE])
        # Values from the reference prior; tolerate ≤1 % relative error
        npt.assert_allclose(nlls[0].item(), 20.9116, rtol=0.01)
        npt.assert_allclose(nlls[1].item(), 17.9506, rtol=0.01)

    def test_likelihood_smiles_single(self):
        nlls = self.adapter.likelihood_smiles([METAMIZOLE])
        self.assertEqual([1], list(nlls.shape))
        npt.assert_allclose(nlls[0].item(), 125.4669, rtol=0.01)

    def test_likelihood_smiles_are_positive(self):
        nlls = self.adapter.likelihood_smiles([PROPANE, BENZENE, METAMIZOLE])
        self.assertTrue((nlls > 0).all())

    # --- sample ---

    def test_sample_returns_correct_count(self):
        seqs, smilies, nlls = self.model.sample(batch_size=20)
        self.assertEqual(20, seqs.shape[0])
        self.assertEqual(20, len(smilies))
        self.assertEqual(20, len(nlls))

    def test_sample_sequence_types(self):
        seqs, smilies, nlls = self.model.sample(batch_size=10)
        self.assertIsInstance(seqs, torch.Tensor)
        self.assertIsInstance(smilies, list)
        self.assertIsInstance(nlls, torch.Tensor)

    # --- likelihood / likelihood_smiles consistency ---

    def test_likelihood_consistency(self):
        """likelihood(seqs), likelihood_smiles(smiles) and sample NLLs must agree."""
        seqs, smilies, sample_nlls = self.model.sample(batch_size=64)
        seq_nlls = self.adapter.likelihood(seqs)
        smiles_nlls = self.adapter.likelihood_smiles(smilies)

        npt.assert_array_almost_equal(
            sample_nlls.detach().cpu().numpy(),
            seq_nlls.detach().cpu().numpy(),
            decimal=3,
        )
        npt.assert_array_almost_equal(
            sample_nlls.detach().cpu().numpy(),
            smiles_nlls.detach().cpu().numpy(),
            decimal=3,
        )

    # --- save / load round-trip ---

    def test_save_and_reload(self, tmp_path=None):
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".prior", delete=False) as f:
            path = f.name
        try:
            self.model.save(path)
            self.assertTrue(os.path.isfile(path))
            reloaded = torch.load(path, map_location=self.device, weights_only=False)
            self.assertEqual(reloaded["model_type"], "Reinvent")
            expected_keys = {
                "model_type",
                "version",
                "metadata",
                "vocabulary",
                "max_sequence_length",
                "network",
                "network_params",
            }
            self.assertEqual(expected_keys, set(reloaded.keys()))
        finally:
            os.unlink(path)

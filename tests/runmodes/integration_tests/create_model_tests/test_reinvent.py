import os
import shutil
import pytest
import unittest

import torch

from reinvent.runmodes.create_model.create_reinvent import create_model


def _create(output_file, smiles_path, metadata):
    create_model(
        num_layers=3,
        layer_size=512,
        dropout=0.0,
        max_sequence_length=256,
        cell_type="lstm",
        embedding_layer_size=256,
        layer_normalization=False,
        standardize=True,
        input_smiles_path=smiles_path,
        output_model_path=output_file,
        metadata=metadata,
    )
    return torch.load(output_file, weights_only=False)


@pytest.mark.integration
@pytest.mark.usefixtures("json_config")
class TestCreateModel(unittest.TestCase):
    def setUp(self):
        self.workdir = self.json_config["MAIN_TEST_PATH"]
        self.output_file = os.path.join(self.workdir, "reinvent_empty.model")

        if not os.path.isdir(self.workdir):
            os.makedirs(self.workdir)

    def tearDown(self):
        if os.path.isdir(self.workdir):
            shutil.rmtree(self.workdir)

    def test_create_model(self):
        model = _create(
            self.output_file,
            self.json_config["SMILES_SET_PATH"],
            {"data_source": "pytest", "comment": "pytest"},
        )
        keys = list(model.keys())

        self.assertEqual(
            keys,
            [
                "model_type",
                "version",
                "metadata",
                "vocabulary",
                "max_sequence_length",
                "network",
                "network_params",
            ],
        )

        self.assertEqual(model["model_type"], "Reinvent")
        self.assertEqual(model["max_sequence_length"], 256)

        network_params = model["network_params"]
        self.assertEqual(network_params["dropout"], 0.0)
        self.assertEqual(network_params["layer_size"], 512)
        self.assertEqual(network_params["num_layers"], 3)
        self.assertEqual(network_params["cell_type"], "lstm")
        self.assertEqual(network_params["embedding_layer_size"], 256)

    def test_metadata_supports_stereochemistry_default_false(self):
        model = _create(
            self.output_file,
            self.json_config["SMILES_SET_PATH"],
            {"data_source": "pytest"},
        )
        self.assertFalse(model["metadata"].supports_stereochemistry)

    def test_metadata_supports_stereochemistry_true(self):
        model = _create(
            self.output_file,
            self.json_config["SMILES_SET_PATH"],
            {"data_source": "pytest", "supports_stereochemistry": True},
        )
        self.assertTrue(model["metadata"].supports_stereochemistry)

    def test_metadata_comment_stored_in_update_log(self):
        model = _create(
            self.output_file,
            self.json_config["SMILES_SET_PATH"],
            {"data_source": "pytest", "comment": "initial note"},
        )
        log = model["metadata"].update_log
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["comment"], "initial note")
        self.assertIsNone(log[0]["timestamp"])
        self.assertIsNone(log[0]["prior_hash"])

    def test_metadata_no_comment_empty_update_log(self):
        model = _create(
            self.output_file,
            self.json_config["SMILES_SET_PATH"],
            {"data_source": "pytest"},
        )
        self.assertEqual(model["metadata"].update_log, [])

    def test_metadata_model_architecture_set(self):
        model = _create(
            self.output_file,
            self.json_config["SMILES_SET_PATH"],
            {"data_source": "pytest"},
        )
        self.assertEqual(model["metadata"].model_architecture, "Reinvent")

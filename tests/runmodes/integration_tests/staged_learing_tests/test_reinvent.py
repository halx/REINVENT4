import pytest
from pathlib import Path

import torch

from reinvent.runmodes.RL.run_staged_learning import run_staged_learning
from reinvent.runmodes.utils.helpers import set_torch_device

SCORING = {
    "type": "custom_product",
    "parallel": 1,
    "component": [
        {
            "custom_alerts": {
                "endpoint": [
                    {
                        "name": "Unwanted SMARTS",
                        "weight": 0.79,
                        "params": {
                            "smarts": [
                                "[*;r8]",
                                "[*;r9]",
                                "[*;r10]",
                                "[*;r11]",
                                "[*;r12]",
                                "[*;r13]",
                                "[*;r14]",
                                "[*;r15]",
                                "[*;r16]",
                                "[*;r17]",
                                "[#8][#8]",
                                "[#6;+]",
                                "[#16][#16]",
                                "[#7;!n][S;!$(S(=O)=O)]",
                                "[#7;!n][#7;!n]",
                                "C#C",
                                "C(=[O,S])[O,S]",
                                "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                                "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                                "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                                "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                                "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                                "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
                            ]
                        },
                    }
                ]
            }
        },
        {
            "MolecularWeight": {
                "endpoint": [
                    {
                        "name": "Molecular weight",
                        "weight": 0.342,
                        "transform": {
                            "type": "double_sigmoid",
                            "high": 500.0,
                            "low": 200.0,
                            "coef_div": 500.0,
                            "coef_si": 20.0,
                            "coef_se": 20.0,
                        },
                    }
                ]
            }
        },
    ],
}

EXPECTED_CHECKPOINT_KEYS = [
    "max_sequence_length",
    "metadata",
    "model_type",
    "network",
    "network_params",
    "staged_learning",
    "version",
    "vocabulary",
]


@pytest.fixture
def setup(json_config, tmp_path, pytestconfig):
    device = pytestconfig.getoption("device")
    set_torch_device(device)

    chkpt = str(tmp_path / "stage1.chkpt")
    csv_prefix = str(tmp_path / "summary")

    config = {
        "parameters": {
            "use_checkpoint": False,
            "prior_file": json_config["PRIOR_PATH"],
            "agent_file": json_config["PRIOR_PATH"],
            "batch_size": 50,
            "randomize_smiles": True,
            "summary_csv_prefix": csv_prefix,
        },
        "learning_strategy": {"type": "dap", "sigma": 120, "rate": 0.0001},
        "diversity_filter": {
            "type": "IdenticalMurckoScaffold",
            "bucket_size": 25,
            "minscore": 0.4,
            "minsimilarity": 0.4,
            "penalty_multiplier": 0.5,
        },
        "stage": [
            {
                "chkpt_file": chkpt,
                "termination": "simple",
                "max_score": 0.7,
                "min_steps": 1,
                "max_steps": 5,
                "scoring": SCORING,
            },
        ],
    }
    return config, Path(chkpt)


@pytest.mark.integration
def test_staged_learning(setup, pytestconfig):
    config, checkpoint_file = setup
    device = torch.device(pytestconfig.getoption("device"))

    run_staged_learning(config, device, tb_logdir=None, responder_config=None)

    assert checkpoint_file.exists()

    model = torch.load(checkpoint_file, weights_only=False)
    assert list(model.keys()) == EXPECTED_CHECKPOINT_KEYS
    assert model["model_type"] == "Reinvent"
    assert "staged_learning" in model


@pytest.mark.integration
def test_staged_learning_two_stages(json_config, tmp_path, pytestconfig):
    """Two sequential stages each produce their own checkpoint."""
    device_str = pytestconfig.getoption("device")
    set_torch_device(device_str)
    device = torch.device(device_str)

    chkpt1 = str(tmp_path / "stage1.chkpt")
    chkpt2 = str(tmp_path / "stage2.chkpt")
    csv_prefix = str(tmp_path / "summary")

    config = {
        "parameters": {
            "use_checkpoint": False,
            "prior_file": json_config["PRIOR_PATH"],
            "agent_file": json_config["PRIOR_PATH"],
            "batch_size": 50,
            "randomize_smiles": True,
            "summary_csv_prefix": csv_prefix,
        },
        "learning_strategy": {"type": "dap", "sigma": 120, "rate": 0.0001},
        "diversity_filter": {
            "type": "IdenticalMurckoScaffold",
            "bucket_size": 25,
            "minscore": 0.4,
            "minsimilarity": 0.4,
            "penalty_multiplier": 0.5,
        },
        "stage": [
            {
                "chkpt_file": chkpt1,
                "termination": "simple",
                # score threshold reached before max_steps → clean exit, no break
                "max_score": 0.0,
                "min_steps": 1,
                "max_steps": 100,
                "scoring": SCORING,
            },
            {
                "chkpt_file": chkpt2,
                "termination": "simple",
                "max_score": 0.0,
                "min_steps": 1,
                "max_steps": 100,
                "scoring": SCORING,
            },
        ],
    }

    run_staged_learning(config, device, tb_logdir=None, responder_config=None)

    assert Path(chkpt1).exists(), "Stage 1 checkpoint missing"
    assert Path(chkpt2).exists(), "Stage 2 checkpoint missing"

    for path in (chkpt1, chkpt2):
        model = torch.load(path, weights_only=False)
        assert list(model.keys()) == EXPECTED_CHECKPOINT_KEYS
        assert model["model_type"] == "Reinvent"

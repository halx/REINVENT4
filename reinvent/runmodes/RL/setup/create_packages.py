from __future__ import annotations

from typing import List
import logging

import torch

from . import terminators
from .terminators import terminator_callable
from reinvent.runmodes import RL
from reinvent.runmodes.RL.data_classes import WorkPackage
from reinvent.runmodes.RL.setup.diversity_filter import setup_diversity_filter
from reinvent.runmodes.RL.validation import SectionStage
from reinvent.scoring import Scorer

logger = logging.getLogger(__name__)


def create_package(
    reward_strategy: RL.RLReward,
    stage: SectionStage,
    device: torch.device,
    prior_model_file_path: str,
    rdkit_smiles_flags: dict,
) -> WorkPackage:
    """Create a single work package for one stage

    :param reward_strategy: the reward strategy
    :param stage: the parameters for this stage
    :param device: torch device for intrinsic reward model setup
    :param prior_model_file_path: path to prior model for intrinsic reward setup
    :param rdkit_smiles_flags: RDKit flags for canonicalization
    :return: a work package for this stage
    """
    chkpt_filename = stage.chkpt_file

    scoring_function = Scorer(stage.scoring)

    max_score = stage.max_score
    min_steps = stage.min_steps
    max_steps = stage.max_steps

    terminator_param = stage.termination
    terminator_name = terminator_param.lower().title()

    terminator: terminator_callable = getattr(terminators, f"{terminator_name}Terminator", None)

    if terminator is None:
        valid = ["simple", "null", "plateau"]
        msg = f"Unknown termination criterion '{terminator_param}'. Valid: {', '.join(valid)}"
        logger.critical(msg)
        raise RuntimeError(msg)

    diversity_filter = None

    if stage.diversity_filter:
        diversity_filter = setup_diversity_filter(
            stage.diversity_filter,
            device,
            prior_model_file_path,
            rdkit_smiles_flags,
        )

    return WorkPackage(
        scoring_function,
        reward_strategy,
        max_steps,
        terminator(max_score, min_steps),
        diversity_filter,
        chkpt_filename,
    )


def create_packages(
    reward_strategy: RL.RLReward,
    stages: List[SectionStage],
    device: torch.device,
    prior_model_file_path: str,
    rdkit_smiles_flags: dict,
) -> List[WorkPackage]:
    """Create work packages for all stages

    :param reward_strategy: the reward strategy
    :param stages: the parameters for each work package
    :param device: torch device for intrinsic reward model setup
    :param prior_model_file_path: path to prior model for intrinsic reward setup
    :param rdkit_smiles_flags: RDKit flags for canonicalization
    :return: a list of work packages
    """
    return [
        create_package(reward_strategy, stage, device, prior_model_file_path, rdkit_smiles_flags)
        for stage in stages
    ]

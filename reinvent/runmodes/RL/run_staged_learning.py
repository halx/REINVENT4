"""Multi-stage learning with RL"""

from __future__ import annotations
import logging

import torch

from reinvent.utils import setup_logger, CsvFormatter, config_parse, get_tokens_from_vocabulary
from reinvent.runmodes import Handler, RL, create_adapter
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.RL.data_classes import ModelState
from reinvent.runmodes.RL.memories.diversity_filter import DiversityFilter
from reinvent.runmodes.utils import disable_gradients
from .setup import (
    create_package,
    setup_diversity_filter,
    setup_intrinsic_penalty,
    setup_inception,
    setup_reward_strategy,
)
from .relevance import AGGREGATORS
from .validation import RLConfig

logger = logging.getLogger(__name__)

TRANSFORMERS = ["Mol2Mol", "LinkinventTransformer", "LibinventTransformer", "Pepinvent"]


def run_staged_learning(
    input_config: dict,
    device: torch.device,
    tb_logdir: str,
    responder_config: dict,
    write_config: str = None,
    *args,
    **kwargs,
):
    """Run Reinforcement Learning/Curriculum Learning

    RL can be run in multiple stages (CL) with different parameters, typically
    the scoring function.  A context manager ensures that a checkpoint file is
    written out should the program be terminated.

    :param input_config: the run configuration
    :param device: torch device
    :param tb_logdir: TensorBoard log directory
    :param responder_config: responder configuration
    :param write_config: callable to write config
    """

    config = RLConfig(**input_config)
    stages = config.stage
    num_stages = len(stages)
    logger.info(
        f"Starting {num_stages} {'stages' if num_stages> 1 else 'stage'} of Reinforcement Learning"
    )

    parameters = config.parameters

    # NOTE: The model files are a dictionary with model attributes from
    #       Reinvent and a set of tensors, each with an attribute for the
    #       device (CPU or GPU) and if gradients are required

    prior_model_filename = parameters.prior_file
    agent_model_filename = parameters.agent_file

    # NOTE: Inference mode means here that torch runs eval() on the network:
    #       switch off some specific layers (dropout, batch normal) but that
    #       does not affect autograd
    #       The gradients are switched off for the prior but the optimizer will
    #       not touch those anyway because we pass only the agent network to the
    #       optimizer, see above.
    adapter, _, model_type = create_adapter(prior_model_filename, "inference", device)
    prior = adapter
    disable_gradients(prior)

    rdkit_smiles_flags = dict(allowTautomers=True)

    if model_type in TRANSFORMERS:  # Transformer-based models
        agent_mode = "inference"
        rdkit_smiles_flags.update(sanitize=True, isomericSmiles=True)
        rdkit_smiles_flags2 = dict(isomericSmiles=True)
    else:
        agent_mode = "training"
        rdkit_smiles_flags2 = dict()

    adapter, agent_save_dict, agent_model_type = create_adapter(
        agent_model_filename, agent_mode, device
    )
    agent = adapter

    agent_metadata = agent_save_dict.get("metadata")
    stereo = (
        agent_metadata.get("supports_stereochemistry", False)
        if isinstance(agent_metadata, dict)
        else getattr(agent_metadata, "supports_stereochemistry", False)
    )
    if stereo and not parameters.isomeric_smiles:
        logger.info(
            "Overriding isomeric_smiles=True from agent model metadata (supports_stereochemistry)"
        )
        parameters.isomeric_smiles = True

    if model_type != agent_model_type:
        raise RuntimeError(
            f"Inconsistent model types: prior is {model_type} agent is {agent_model_type}"
        )

    logger.info(f"Using generator {model_type}")
    logger.info(f"Prior read from {prior_model_filename}")
    logger.info(f"Agent read from {agent_model_filename}")

    smilies = None

    if parameters.smiles_file:
        allowed_tokens = get_tokens_from_vocabulary(agent.vocabulary)
        smilies = config_parse.read_smiles_csv_file(parameters.smiles_file, 0, allowed_tokens)

        if not smilies:
            raise RuntimeError(f"No valid SMILES read from {parameters.smiles_file}")

        logger.info(f"Input molecules/fragments read from file {parameters.smiles_file}")

    sampler, _ = setup_sampler(model_type, parameters.model_dump(), agent)
    reward_strategy = setup_reward_strategy(config.learning_strategy, agent, config.inception)

    global_df_only = False
    intrinsic_penalty = None
    diversity_filter: DiversityFilter | None = None

    if parameters.use_checkpoint and "staged_learning" in agent_save_dict:
        logger.info(f"Using diversity filter from {agent_model_filename}")
        diversity_filter = agent_save_dict["staged_learning"]["diversity_filter"]
        if not isinstance(diversity_filter, DiversityFilter):
            logger.warning(
                f"Loaded diversity filter is not an instance of DiversityFilter: "
                f"{type(diversity_filter)}. Ignoring."
            )
            diversity_filter = None
    elif parameters.use_checkpoint:
        logger.warning(
            f"use_checkpoint=True but {agent_model_filename} contains no prior checkpoint "
            f"data ('staged_learning' key missing). Starting with a fresh diversity filter."
        )
        diversity_filter = None
    elif config.diversity_filter:
        diversity_filter = setup_diversity_filter(
            config.diversity_filter,
            device,
            prior_model_filename,
            rdkit_smiles_flags2,
        )
        global_df_only = True
    elif config.intrinsic_penalty:
        intrinsic_penalty = setup_intrinsic_penalty(
            config.intrinsic_penalty,
            device,
            prior_model_filename,
            rdkit_smiles_flags2,
        )

    if parameters.purge_memories:
        logger.info("Purging diversity filter memories after each stage")
    else:
        logger.info("Diversity filter memories are retained between stages")

    inception = None

    # Inception only set up here for the very first step
    if config.inception:  # and model_type == "Reinvent":
        inception = setup_inception(config.inception, diversity_filter, prior)

    if inception is None and model_type == "Reinvent":
        logger.warning("Inception disabled but may speed up convergence")

    summary_csv_prefix = parameters.summary_csv_prefix

    # FIXME: is there a sensible default, this is only needed by Mol2Mol
    distance_threshold = parameters.distance_threshold

    model_learning = getattr(RL, f"{model_type}Learning", None)

    if model_learning is None:
        raise RuntimeError(f"No RL learning class found for model type '{model_type}'")

    if callable(write_config):
        write_config(config.model_dump())

    # --- performance monitoring state initialisation goes here ---

    with Handler() as handler:
        for run, stage in enumerate(stages):
            stage_no = run + 1

            # --- HOOK: dynamic parameter adjustment (e.g. modify stage.max_steps,
            #     stage.scoring, diversity filter params) based on prior-stage
            #     performance metrics captured at the end of the previous iteration ---

            package = create_package(
                reward_strategy, stage, device, prior_model_filename, rdkit_smiles_flags2
            )

            csv_filename = f"{summary_csv_prefix}_{stage_no}.csv"

            setup_logger(
                name="csv",
                filename=csv_filename,
                formatter=CsvFormatter(),
                propagate=False,
                level="INFO",
            )

            logdir = f"{tb_logdir}_{run}" if tb_logdir else None

            logger.info(f"Writing tabular data for stage to {csv_filename}")
            logger.info(f"Starting stage {stage_no} <<<")

            if global_df_only:  # global DF always overwrites stage DFs
                state = ModelState(agent, diversity_filter)
                logger.debug(f"Using global DF")
            else:
                state = ModelState(agent, package.diversity_filter)
                logger.debug(f"Using stage DF")

            # stage overrides the global loss-modulation aggregation
            lm_aggregation = stage.loss_modulation_aggregation or config.loss_modulation_aggregation
            if lm_aggregation not in AGGREGATORS:
                raise ValueError(
                    f"Unknown loss_modulation_aggregation '{lm_aggregation}', "
                    f"expected one of {sorted(AGGREGATORS)}"
                )

            optimize = model_learning(
                max_steps=package.max_steps,
                stage_no=stage_no,
                prior=prior,
                state=state,
                scoring_function=package.scoring_function,
                reward_strategy=package.learning_strategy,
                sampling_model=sampler,
                smilies=smilies,
                distance_threshold=distance_threshold,
                rdkit_smiles_flags=rdkit_smiles_flags,
                inception=inception,
                responder_config=responder_config,
                tb_logdir=logdir,
                tb_isim=parameters.tb_isim,
                intrinsic_penalty=intrinsic_penalty,
                loss_modulation_aggregation=lm_aggregation,
            )

            handler.out_filename = package.out_state_filename
            handler.register_callback(optimize.get_state_dict)

            if inception is not None:
                inception.update(package.scoring_function)

            terminate = optimize(package.terminator)
            state = optimize.state

            # --- HOOK: capture performance metrics (mean score, steps taken,
            #     convergence rate) from this stage for use in the next iteration ---

            if state.diversity_filter and parameters.purge_memories:
                logger.info(f"Purging diversity filter memories in stage {stage_no}")
                state.diversity_filter.purge_memories()

            handler.save()

            if terminate:
                logger.warning(
                    f"Maximum number of steps of {package.max_steps} reached in stage "
                    f"{stage_no}. Terminating all stages."
                )
                break

            logger.info(f"Finished stage {stage_no} >>>")

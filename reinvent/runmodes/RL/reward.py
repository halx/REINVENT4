"""Reward strategies for Reinforcement Learning"""

from __future__ import annotations

__all__ = [
    "ISWeightConfig",
    "RLReward",
    "dap_strategy",
    "mascof_strategy",
    "mauli_strategy",
    "sdap_strategy",
    "dap_reinforce_strategy",
    "mascof_reinforce_strategy",
    "mauli_reinforce_strategy",
]

import logging
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING
import warnings

import numpy as np
import torch

if TYPE_CHECKING:
    from reinvent.models import ModelAdapter
    from reinvent.runmodes.RL.memories.inception import Inception

logger = logging.getLogger(__name__)
warnings.filterwarnings("once", category=FutureWarning)


def format_warning(msg, *args, **kwargs):
    return f"{msg}"


warnings.formatwarning = format_warning


### The reward functions from the LibInvent paper
def dap_strategy(
    agent_lls: torch.Tensor, scores: torch.Tensor, prior_lls: torch.Tensor, sigma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """

    :param agent_lls: agent (actor) log-likelihood
    :param scores: scores for each generated SMILES
    :param prior_lls: prior (critic) log-likelihood
    :param sigma: scores multiplier
    :returns: the loss and the augment NLLs
    """

    augmented_lls = prior_lls + sigma * scores
    loss = torch.pow((augmented_lls - agent_lls), 2)

    return loss, augmented_lls


def mascof_strategy(agent_lls: torch.Tensor, scores: torch.Tensor, *args, **kwargs):
    warnings.warn(
        "MASCOF is deprecated and will be removed in the future.",
        FutureWarning,
        stacklevel=2,
    )

    augmented_lls = scores
    loss = -torch.sum(augmented_lls) * torch.sum(agent_lls)  # scalar

    return loss.view(-1), augmented_lls


def mauli_strategy(
    agent_lls: torch.Tensor,
    scores: torch.Tensor,
    prior_lls: torch.Tensor,
    sigma: int,
):
    warnings.warn(
        "MAULI is deprecated and will be removed in the future.",
        FutureWarning,
        stacklevel=2,
    )

    augmented_lls = prior_lls + sigma * scores
    loss = -torch.sum(augmented_lls) * torch.sum(agent_lls)  # scalar

    return loss.view(-1), augmented_lls


def sdap_strategy(
    agent_lls: torch.Tensor,
    scores: torch.Tensor,
    prior_lls: torch.Tensor,
    sigma: int,
):
    warnings.warn(
        "SDAP is deprecated and will be removed in the future.",
        FutureWarning,
        stacklevel=2,
    )

    augmented_lls = prior_lls + sigma * scores

    reward_score = torch.pow((augmented_lls - agent_lls), 2)
    loss = -reward_score.mean() * agent_lls.mean()  # scalar

    return loss.view(-1), augmented_lls


def reinforce_loss(reward: torch.Tensor, agent_lls: torch.Tensor):
    """Compute the REINFORCE loss function.

    :param reward: reward for each generated SMILES
    :param agent_lls: agent (actor) log-likelihood

    :returns: the loss that can be used to calculate gradients"""
    loss = -torch.mean(reward + reward.detach() * agent_lls).view(-1)
    return loss


def dap_reinforce_strategy(
    agent_lls: torch.Tensor, scores: torch.Tensor, prior_lls: torch.Tensor, sigma: float
):
    """Compute the DAP REINFORCE loss function.

    :param agent_lls: agent (actor) log-likelihood
    :param scores: scores for each generated SMILES
    :param prior_lls: prior (critic) log-likelihood
    :param sigma: scores multiplier

    :returns: the loss that can be used to calculate gradients, and the reward for each generated SMILES
    """
    reward = prior_lls + sigma * scores - agent_lls
    loss = reinforce_loss(reward, agent_lls)
    return loss, reward


def mascof_reinforce_strategy(agent_lls: torch.Tensor, scores: torch.Tensor, *args, **kwargs):
    """Compute the MASCOF REINFORCE loss function.

    :param agent_lls: agent (actor) log-likelihood
    :param scores: scores for each generated SMILES

    :returns: the loss that can be used to calculate gradients, and the reward for each generated SMILES
    """
    loss = reinforce_loss(scores, agent_lls)
    return loss, scores


def mauli_reinforce_strategy(
    agent_lls: torch.Tensor, scores: torch.Tensor, prior_lls: torch.Tensor, sigma: float
):
    """Compute the MAULI REINFORCE loss function.

    :param agent_lls: agent (actor) log-likelihood
    :param scores: scores for each generated SMILES
    :param prior_lls: prior (critic) log-likelihood
    :param sigma: scores multiplier

    :returns: the loss that can be used to calculate gradients, and the reward for each generated SMILES
    """
    reward = prior_lls + sigma * scores
    loss = reinforce_loss(reward, agent_lls)
    return loss, reward


@dataclass
class ISWeightConfig:
    """Importance-sampling weight correction for inception replay.

    :param weighted_scores: apply IS weight correction to inception replay scores
    :param weight_clip: clamp log IS weights to [-clip, clip] before softmax; None disables
    :param weight_temperature: temperature divisor for log IS weights
    """

    weighted_scores: bool = False
    weight_clip: float | None = None
    weight_temperature: float = 1.0


class RLReward:
    def __init__(
        self,
        optimizer,
        sigma=120,
        strategy: Callable = dap_strategy,
        is_weight_config: ISWeightConfig | None = None,
    ):
        """Run the chosen RL reward strategy to optimize the model.

        :param optimizer: torch optimizer for the network
        :param strategy: a callable to compute the loss function
        :param sigma: the scaling hyperparameter
        :param is_weight_config: IS weight correction settings for inception replay
        """

        self._optimizer = optimizer
        self._sigma = sigma
        self._strategy = strategy
        self._is_weight_config = is_weight_config

    def __call__(
        self,
        orig_smilies: list[str],
        scores: np.ndarray,
        agent_nlls: torch.Tensor,
        prior_nlls: torch.Tensor,
        mask_idx: np.ndarray,
        inception: Optional[Inception],
        agent: Optional[ModelAdapter],
        weights: Optional[np.ndarray] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Entry point to run the learning strategy.

        :param orig_smilies: originally sampled SMILES
        :param scores: scores
        :param agent_nlls: agent (actor) NLLa
        :param prior_nlls: prior (critic) NLLs
        :param mask_idx: indices of the valid SMILES, needed for inception
        :param inception: instance of the inception memory
        :param agent: the agent network model, needed for inception
        :param weights: optional per-molecule relevance for loss modulation (NaN = neutral)
        :returns: the negative log likelihoods for agent, prior and augmented, and the mean loss
        """

        scores = torch.from_numpy(scores).to(prior_nlls)

        # FIXME: move NaN filtering before first use of scores in learning
        # FIXME: reconsider NaN/failure handling
        nan_idx = torch.isnan(scores)
        scores_nonnan = scores[~nan_idx]
        agent_lls = -agent_nlls[~nan_idx]  # negated because we need to minimize
        prior_lls = -prior_nlls[~nan_idx]

        loss, augmented_lls = self._strategy(
            agent_lls,
            scores_nonnan,
            prior_lls,
            self._sigma,
        )

        batch_weights = None
        if weights is not None:
            batch_weights = torch.as_tensor(np.asarray(weights, dtype=float)).to(prior_lls)[
                ~nan_idx
            ]

        if inception is not None:
            # mask_idx is in the original (unfiltered) batch space, so intersect it
            # with the non-NaN mask in that same space before indexing.
            valid_mask = np.zeros(len(orig_smilies), dtype=bool)
            valid_mask[mask_idx] = True
            keep_mask = valid_mask & ~nan_idx.cpu().numpy()  # valid AND scored, original space

            to_inception_relevance = (
                np.asarray(weights, dtype=float)[keep_mask] if weights is not None else None
            )  # relevance for the sampled molecules possibly added to the inception batch

            sampled = inception(
                np.array(orig_smilies)[keep_mask],
                scores[keep_mask],
                (-prior_nlls)[keep_mask],
                (-agent_nlls)[keep_mask],
                agent,
                relevance=to_inception_relevance,
            )

            if (
                sampled is not None
            ):  # None when the memory is still empty (e.g. first batch gated out)
                _orig_smilies, _scores, _prior_lls, _stored_agent_lls, _rel_from_inception = sampled

                # compute the agent NLLs for the _current_ state of the agent
                lls = agent.likelihood_smiles(_orig_smilies)

                if isinstance(lls, torch.Tensor):  # Reinvent
                    _agent_lls = -lls
                else:  # all other models
                    _agent_lls = -lls.likelihood

                _scores_t = torch.tensor(_scores).to(_agent_lls)

                if self._is_weight_config is not None and self._is_weight_config.weighted_scores:
                    cfg = self._is_weight_config
                    _stored_lls_t = torch.tensor(_stored_agent_lls).to(_agent_lls)
                    log_w = (_agent_lls.detach() - _stored_lls_t) / cfg.weight_temperature
                    if cfg.weight_clip is not None:
                        log_w = log_w.clamp(-cfg.weight_clip, cfg.weight_clip)
                    w = torch.softmax(log_w, dim=0)
                    _scores_t = _scores_t * w * len(w)  # mean-preserving reweighting

                inception_loss, _ = self._strategy(
                    _agent_lls,
                    _scores_t,
                    torch.tensor(_prior_lls).to(_agent_lls),
                    self._sigma,
                )

                loss = torch.cat((loss, inception_loss), 0)

                if batch_weights is not None:
                    inception_w = torch.tensor(
                        [np.nan if c is None else float(c) for c in _rel_from_inception],
                        dtype=batch_weights.dtype,
                        device=batch_weights.device,
                    )
                    batch_weights = torch.cat((batch_weights, inception_w), 0)

        if batch_weights is not None:
            loss = self._modulate(loss, batch_weights)

        loss = loss.mean()

        self._optimizer.zero_grad()
        loss.backward()

        self._optimizer.step()

        return agent_lls, prior_lls, augmented_lls, loss

    @staticmethod
    def _modulate(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Rescale per-molecule loss by mean-1 normalised relevance; NaN weights stay neutral."""
        finite = torch.isfinite(weights)

        if not bool(finite.any()):
            return loss

        eps = torch.finfo(loss.dtype).eps
        wbar = weights[finite].mean().clamp_min(eps)
        norm = torch.ones_like(loss)
        norm[finite] = weights[finite] / wbar

        return loss * norm

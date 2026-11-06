"""Example/dummy scoring component demonstrating plugin conventions.

File must be named comp_*.py to be discovered by the plugin mechanism.
"""

from __future__ import annotations

__all__ = ["MyScorer"]

from typing import List

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag


# Tag with @add_tag("__parameters"). At most one per file.
# All parameters are lists because endpoints are collected into lists.
@add_tag("__parameters")
@dataclass
class Parameters:
    param_a: List[float]
    param_b: List[str] = Field(default_factory=lambda: ["default"]) # how to declare a default value


# Tag with @add_tag("__component"). Use @add_tag("__component", "filter")
# for filter components or @add_tag("__component", "penalty") for penalties.
@add_tag("__component")
class MyScorer:
    def __init__(self, params: Parameters):
        self.param_a = params.param_a
        self.param_b = params.param_b

        # Required when the component has more than one endpoint.
        self.number_of_endpoints = len(self.param_a)

        # Required when __call__ returns ComponentResults with metadata.
        # Must match the keys in the metadata dict exactly.
        self.metadata_keys = ("metadata_1", "metadata_2")

    def __call__(self, smilies: List[str]) -> ComponentResults:
        scores = []
        detail_a = []
        detail_b = []

        for _ in self.param_a:
            endpoint_scores = np.full(len(smilies), 0.5)
            scores.append(endpoint_scores)

        for smi in smilies:
            detail_a.append(len(smi))
            detail_b.append(smi[:3])

        return ComponentResults(
            scores,
            metadata={self.metadata_keys[0]: detail_a, self.metadata_keys[1]: detail_b},
        )

"""Group count

Allow groups of atoms defined by SMARTS only a certain number of times in a
molecules.
"""

from __future__ import annotations

__all__ = ["GroupCount"]

from typing import List

from rdkit import Chem
import numpy as np
from pydantic.dataclasses import dataclass

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.

    Each endpoint's ``smarts`` may be a single SMARTS string or a list of
    SMARTS strings.  When a list is given the endpoint reports the aggregate
    (total) number of substructure matches over all patterns in that list,
    yielding a single output column.  This mirrors how e.g. TanimotoDistance
    accepts a list of reference SMILES per endpoint.
    """

    smarts: List[str | List[str]]


@add_tag("__component")
class GroupCount:
    def __init__(self, params: Parameters):
        # one list of compiled patterns per endpoint
        self.patterns_per_endpoint = []

        for smarts in params.smarts:
            if isinstance(smarts, str):
                smarts = [smarts]

            patterns = []

            for smart in smarts:
                pattern = Chem.MolFromSmarts(smart)

                if pattern:
                    patterns.append(pattern)

            self.patterns_per_endpoint.append(patterns)

        if not any(self.patterns_per_endpoint):
            raise ValueError(f"{__name__}: no valid SMARTS patterns found")

        self.number_of_endpoints = len(params.smarts)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for patterns in self.patterns_per_endpoint:
            endpoint_scores = []

            for mol in mols:
                if not mol:
                    endpoint_scores.append(np.nan)
                else:
                    total = sum(len(mol.GetSubstructMatches(pattern)) for pattern in patterns)
                    endpoint_scores.append(total)

            scores.append(np.array(endpoint_scores))

        return ComponentResults(scores)

from typing import List, Literal

from pydantic.dataclasses import dataclass
from pydantic import Field

from rdkit.Chem import Mol
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import json

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the FingerprintPrecedence scoring component.

    database_files holds one element per endpoint: a list of JSON paths
    whose fingerprints are concatenated for that endpoint.
    """

    database_files: List[List[str]]
    nll_method: List[
        Literal["total", "max"]
    ] = Field(
        default_factory=lambda: ["total"]
    )
    # "total" = dot(fp_concat, nll_pos_concat) + dot(1-fp_concat, nll_neg_concat)
    # "max"   = max nll-pos over all set bits across all FP types


@add_tag("__component")
class FingerprintPrecedence:
    """Scoring component based on fingerprint bit NLL values.

    Supports Morgan (morgan), AtomPair (ap), RDKit path (rdkit), and
    TopologicalTorsion (tt) fingerprints. Multiple DB files can be concatenated
    for a single endpoint by passing a list of paths in database_files, giving
    a score equal to the sum of per-FP-type Bernoulli NLLs.

    Each JSON must contain:
        - "nll-pos":  list[float]  (length = fpsize)
        - "fp_type":  str          one of: morgan, ap, rdkit, tt  (required)
        - "fpsize":   int          (required)
    Type-specific required keys:
        - morgan: "radius"   (int)
        - rdkit:  "max_path" (int)
    Optional:
        - "most_common_fragment": list[str|null]

    nll-neg is not stored in the DB; it is derived as -log(1 - exp(-nll_pos)).
    """

    def __init__(self, params: Parameters):
        self.metadata_keys = ("highest_nll_bit_fragment",)
        NLL_CAP = 3000.0
        self.nll_method = params.nll_method[0]

        db_paths = params.database_files[0]

        self._fp_configs = []
        nll_pos_parts = []
        nll_neg_parts = []
        self._fragments = []

        for db_path in db_paths:
            raw = json.load(open(db_path.strip()))
            stem = list(raw.keys())[0]
            db = raw[stem]

            fp_type = db["fp_type"]    # required
            fpsize = int(db["fpsize"]) # required

            nll_pos = np.minimum(np.array(db["nll-pos"], dtype=float), NLL_CAP)
            p = np.exp(-nll_pos)
            nll_neg = np.minimum(-np.log(np.maximum(1.0 - p, 1e-300)), NLL_CAP)

            self._fragments.extend(db.get("most_common_fragment", [None] * fpsize))
            nll_pos_parts.append(nll_pos)
            nll_neg_parts.append(nll_neg)

            offset = sum(c["fpsize"] for c in self._fp_configs)
            self._fp_configs.append({
                "fp_type": fp_type,
                "fpsize":  fpsize,
                "gen":     self._make_generator(fp_type, fpsize, db),
                "offset":  offset,
                "params":  db,
            })

        self._nll_pos = np.concatenate(nll_pos_parts)
        self._nll_neg = np.concatenate(nll_neg_parts)

    @staticmethod
    def _make_generator(fp_type, fpsize, db):
        if fp_type == "morgan":
            return rdFingerprintGenerator.GetMorganGenerator(
                radius=int(db["radius"]), fpSize=fpsize)
        elif fp_type == "ap":
            return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fpsize)
        elif fp_type == "rdkit":
            return rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=fpsize, maxPath=int(db["max_path"]))
        elif fp_type == "tt":
            return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fpsize)
        raise ValueError(f"Unknown fp_type: {fp_type!r}. Must be one of: morgan, ap, rdkit, tt")

    def __call__(self, smilies: List[str]) -> ComponentResults:
        scores, metadata = self._compute_scores(smilies)
        return ComponentResults(scores, metadata=metadata)

    @molcache
    def _compute_scores(self, mols: List[Mol]):
        scores = []
        highest_nll_fragments = []

        for mol in mols:
            if mol is None:
                scores.append(0.0)
                highest_nll_fragments.append(None)
                continue

            fp_concat, mappings = self._get_concat_fp_with_mapping(mol)
            set_bits = np.where(fp_concat > 0)[0]

            if self.nll_method == "total":
                score = float(
                    np.dot(fp_concat, self._nll_pos)
                    + np.dot(1 - fp_concat, self._nll_neg)
                )
            else:  # max
                score = float(np.max(self._nll_pos[set_bits])) if len(set_bits) > 0 else 0.0
            scores.append(score)

            if len(set_bits) == 0:
                highest_nll_fragments.append(None)
                continue

            global_bit = int(set_bits[np.argmax(self._nll_pos[set_bits])])
            highest_nll_fragments.append(self._attribute_bit(mol, global_bit, mappings))

        return [np.array(scores, dtype=float)], {self.metadata_keys[0]: highest_nll_fragments}

    def _get_concat_fp_with_mapping(self, mol):
        parts, mappings = [], []
        for cfg in self._fp_configs:
            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            fp = cfg["gen"].GetFingerprintAsNumPy(mol, additionalOutput=ao)
            parts.append(fp)
            mappings.append((cfg["offset"], cfg["fp_type"], ao.GetBitInfoMap(), cfg["params"]))
        return np.concatenate(parts), mappings

    def _attribute_bit(self, mol, global_bit, mappings):
        """Identify which FP type owns this global bit index and extract a fragment."""
        for offset, fp_type, bv_mapping, db_params in mappings:
            fpsize = int(db_params["fpsize"])
            if offset <= global_bit < offset + fpsize:
                local_bit = global_bit - offset
                hits = bv_mapping.get(local_bit, [])
                if not hits:
                    return self._fragments[global_bit]
                return self._extract_fragment(mol, fp_type, hits)
        return self._fragments[global_bit]

    def _extract_fragment(self, mol, fp_type, hits):
        """Type-specific fragment extraction from a bit hit."""
        hit = hits[0]
        try:
            if fp_type == "morgan":
                atom_id, radius = hit
                if radius == 0:
                    a = mol.GetAtomWithIdx(atom_id)
                    sym = a.GetSymbol()
                    return sym.lower() if a.GetIsAromatic() else sym
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_id)
                return Chem.MolToSmiles(Chem.PathToSubmol(mol, env))

            elif fp_type == "ap":
                atom1, atom2 = hit
                path = list(Chem.GetShortestPath(mol, atom1, atom2))
                bonds = [mol.GetBondBetweenAtoms(path[i], path[i + 1]).GetIdx()
                         for i in range(len(path) - 1)
                         if mol.GetBondBetweenAtoms(path[i], path[i + 1])]
                return Chem.MolToSmiles(Chem.PathToSubmol(mol, bonds)) if bonds \
                    else mol.GetAtomWithIdx(atom1).GetSymbol()

            elif fp_type == "rdkit":
                bonds = list(hit)
                return Chem.MolToSmiles(Chem.PathToSubmol(mol, bonds)) if bonds else "?"

            elif fp_type == "tt":
                atoms = list(hit)
                bonds = [mol.GetBondBetweenAtoms(atoms[i], atoms[i + 1]).GetIdx()
                         for i in range(len(atoms) - 1)
                         if mol.GetBondBetweenAtoms(atoms[i], atoms[i + 1])]
                return Chem.MolToSmiles(Chem.PathToSubmol(mol, bonds)) if bonds else "?"
        except Exception:
            pass
        return None

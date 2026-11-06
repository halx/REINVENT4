# Build a fingerprint precedence database (nll-pos + most_common_fragment) from a
# CSV of SMILES. Supports Morgan, AtomPair, RDKit path, and TopologicalTorsion FPs.
# The output JSON can be used directly in the FingerprintPrecedence scoring component,
# or concatenated with other FP-type JSONs.
#
# Examples:
#   python compute_fp_precedence.py --dataset input.csv --smiles_column SMILES \
#     --fp_type morgan --radius 4 --fpsize 8192 --output_file morgan_r4_8192.json
#
#   python compute_fp_precedence.py --dataset input.csv --smiles_column SMILES \
#     --fp_type rdkit --max_path 7 --fpsize 8192 --output_file rdkit_p7_8192.json
#
#   python compute_fp_precedence.py --dataset input.csv --smiles_column SMILES \
#     --fp_type ap --fpsize 8192 --output_file ap_8192.json
#
#   python compute_fp_precedence.py --dataset input.csv --smiles_column SMILES \
#     --fp_type tt --fpsize 8192 --output_file tt_8192.json

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import multiprocessing as mp


NLL_CAP = 3000.0


def clean_and_flatten_smiles(smiles):
    """Convert to canonical SMILES with stereochemistry removed; return None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    except Exception:
        print(f" invalid smiles {smiles}")
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def make_generator(fp_type, fpsize, radius=None, max_path=None):
    if fp_type == "morgan":
        return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpsize)
    elif fp_type == "ap":
        return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fpsize)
    elif fp_type == "rdkit":
        return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpsize, maxPath=max_path)
    elif fp_type == "tt":
        return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fpsize)
    raise ValueError(f"Unknown fp_type: {fp_type!r}")


def find_fragment(mol, fp_type, hit):
    """Extract a representative SMILES fragment for a fingerprint bit hit."""
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
            return Chem.MolToSmiles(Chem.PathToSubmol(mol, bonds)) if bonds else None

        elif fp_type == "tt":
            atoms = list(hit)
            bonds = [mol.GetBondBetweenAtoms(atoms[i], atoms[i + 1]).GetIdx()
                     for i in range(len(atoms) - 1)
                     if mol.GetBondBetweenAtoms(atoms[i], atoms[i + 1])]
            return Chem.MolToSmiles(Chem.PathToSubmol(mol, bonds)) if bonds else None
    except Exception:
        pass
    return None


def process_one(args):
    """
    Process a single SMILES string.
    Returns (fp_array, bit_fragments_dict) or None on failure.
    bit_fragments_dict maps bit_idx -> list of fragment SMILES strings.
    """
    smiles, fp_type, fpsize, max_frags_per_bit, type_params = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)

    gen = make_generator(fp_type, fpsize,
                         radius=type_params.get("radius"),
                         max_path=type_params.get("max_path"))
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = gen.GetFingerprintAsNumPy(mol, additionalOutput=ao)
    bv_mapping = ao.GetBitInfoMap()

    bit_fragments = {}
    # fragment attribution only works reliably for morgan; other generators either
    # return empty bit info maps or yield too many hits per bit to be practical
    if fp_type == "morgan":
        for bit_idx, hits in bv_mapping.items():
            frags = [f for hit in hits for f in [find_fragment(mol, fp_type, hit)] if f]
            if frags:
                bit_fragments[bit_idx] = frags

    return fp, bit_fragments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a fingerprint precedence database from a CSV of SMILES."
    )
    parser.add_argument("--dataset",        type=str, required=True, help="input CSV file")
    parser.add_argument("--smiles_column",  type=str, required=True, help="SMILES column name")
    parser.add_argument("--fp_type",        type=str, required=True,
                        choices=["morgan", "ap", "rdkit", "tt"],
                        help="fingerprint type")
    parser.add_argument("--output_file",    type=str, required=True, help="output JSON path")
    parser.add_argument("--fpsize",         type=int, default=8192,  help="number of bits")
    parser.add_argument("--radius",         type=int, default=None,
                        help="Morgan radius (required for fp_type=morgan)")
    parser.add_argument("--max_path",       type=int, default=7,
                        help="max path length (for fp_type=rdkit)")
    parser.add_argument("--num_cpus",       type=int, default=8)
    parser.add_argument("--max_frags_per_bit", type=int, default=10000,
                        help="max fragment strings stored per bit before subsampling")
    args = parser.parse_args()

    if args.fp_type == "morgan" and args.radius is None:
        parser.error("--radius is required for fp_type=morgan")

    fp_type = args.fp_type
    fpsize = args.fpsize
    num_cpus = args.num_cpus
    max_frags_per_bit = args.max_frags_per_bit

    type_params = {}
    if fp_type == "morgan":
        type_params["radius"] = args.radius
    elif fp_type == "rdkit":
        type_params["max_path"] = args.max_path

    df = pd.read_csv(args.dataset)
    print(f" read in DF size {df.shape}")

    with mp.Pool(num_cpus) as pool:
        df["cleaned_smiles"] = pool.map(clean_and_flatten_smiles, df[args.smiles_column])

    df = df.loc[~df["cleaned_smiles"].isnull()].reset_index(drop=True)
    print(f"after cleaning, df size is {df.shape}")

    smiles_list = df["cleaned_smiles"].tolist()
    task_args = [(smi, fp_type, fpsize, max_frags_per_bit, type_params) for smi in smiles_list]
    print(f"Computing {fp_type} fingerprints for {len(smiles_list):,} molecules ...")

    fp_sum = np.zeros(fpsize, dtype=float)
    all_bit_fragments = defaultdict(list)
    n_valid = 0

    with mp.Pool(num_cpus) as pool:
        for result in tqdm(pool.imap(process_one, task_args), total=len(task_args)):
            if result is None:
                continue
            fp, bit_fragments = result
            fp_sum += fp
            n_valid += 1
            for bit_idx, frags in bit_fragments.items():
                existing = all_bit_fragments[bit_idx]
                existing.extend(frags)
                if len(existing) > max_frags_per_bit:
                    all_bit_fragments[bit_idx] = existing[:max_frags_per_bit]

    print(f"Processed {n_valid:,} valid molecules")

    count_prob = fp_sum / n_valid
    sum_counts = fp_sum.sum()
    pos_prob = fp_sum / sum_counts

    with np.errstate(divide="ignore", invalid="ignore"):
        nll_count = np.where(count_prob > 0, -np.log(count_prob), NLL_CAP)
        nll_pos   = np.where(pos_prob   > 0, -np.log(pos_prob),   NLL_CAP)

    nll_count = np.minimum(nll_count, NLL_CAP)
    nll_pos   = np.minimum(nll_pos,   NLL_CAP)

    print("Aggregating most_common_fragment ...")
    most_common = []
    for i in range(fpsize):
        frags = all_bit_fragments.get(i, [])
        most_common.append(Counter(frags).most_common(1)[0][0] if frags else None)

    n_mapped = sum(1 for f in most_common if f is not None)
    print(f"  {n_mapped:,} / {fpsize} bits have a representative fragment")
    if n_mapped < fpsize * 0.99:
        print(f"  WARNING: {fpsize - n_mapped} bits have no fragment — "
              f"check dataset size or fp parameters")

    stem = os.path.splitext(os.path.basename(args.output_file))[0]
    output_entry = {
        "nll-count":            nll_count.tolist(),
        "nll-pos":              nll_pos.tolist(),
        "most_common_fragment": most_common,
        "fp_type":              fp_type,
        "fpsize":               fpsize,
        "n_molecules":          n_valid,
    }
    if fp_type == "morgan":
        output_entry["radius"] = args.radius
    elif fp_type == "rdkit":
        output_entry["max_path"] = args.max_path

    json.dump({stem: output_entry}, open(args.output_file, "w"))
    print(f"written → {args.output_file}")

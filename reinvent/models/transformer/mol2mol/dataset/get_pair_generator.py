__all__ = ["get_pair_generator"]
from reinvent.models.transformer.mol2mol.dataset import preprocessing


def get_pair_generator(pair_generator_name: str, *args, **kwargs) -> preprocessing.PairGenerator:
    """Returns a PairGenerator object.

    :param pair_generator_name: name to retrieve the PairGenerator object.
                                Supported generators: {'tanimoto', 'mmp', 'scaffold', 'precomputed'}
    :returns: a pair generator instance
    """

    pgn = pair_generator_name.strip().capitalize()
    pair_generator = getattr(preprocessing, f"{pgn}PairGenerator", None)

    if pair_generator is None:
        valid = ["tanimoto", "mmp", "scaffold", "precomputed"]
        raise RuntimeError(
            f"Unknown pair generator '{pair_generator_name}'. " f"Valid choices: {', '.join(valid)}"
        )

    return pair_generator(*args, **kwargs)

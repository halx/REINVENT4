"""Backward-compatibility shim for checkpoints saved before the filters/ refactoring.

Old checkpoints pickle diversity filter objects under the ``memories.*`` module
path.  Re-exporting the concrete class here lets torch.load unpickle those
objects without error.
"""

from .filters.scaffold_similarity import ScaffoldSimilarity

__all__ = ["ScaffoldSimilarity"]

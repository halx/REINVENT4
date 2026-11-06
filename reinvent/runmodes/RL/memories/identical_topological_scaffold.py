"""Backward-compatibility shim for checkpoints saved before the filters/ refactoring.

Old checkpoints pickle diversity filter objects under the ``memories.*`` module
path.  Re-exporting the concrete class here lets torch.load unpickle those
objects without error.
"""

from .filters.identical_topological_scaffold import IdenticalTopologicalScaffold

__all__ = ["IdenticalTopologicalScaffold"]

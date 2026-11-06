"""Backward-compatibility shim for checkpoints saved before the utils/ refactoring.

Old checkpoints pickle ``BucketCounter`` objects under the
``reinvent.runmodes.RL.memories.bucket_counter`` module path.  Re-exporting
the class here lets :func:`torch.load` unpickle those objects without error.
"""

from .utils.bucket_counter import BucketCounter

__all__ = ["BucketCounter"]

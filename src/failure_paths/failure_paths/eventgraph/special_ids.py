from __future__ import annotations

from enum import IntEnum


class ReservedPathId(IntEnum):
    """Reserved negative ``Faalpad_ID`` values used for synthetic graph nodes."""

    START_NODE = -3
    OVERTOPPING = -2
    INDIRECT_MECHANISM = -1

    @classmethod
    def is_scenario(cls, value: int) -> bool:
        """Return whether ``value`` identifies a reserved scenario node."""
        return value in (cls.OVERTOPPING, cls.INDIRECT_MECHANISM)

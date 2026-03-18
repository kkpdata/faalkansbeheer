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


class ScenarioStateId(IntEnum):
    """Boolean-like scenario-state identifiers used for scenario node sub-ids."""

    NEE = 0
    JA = 1

    @classmethod
    def from_label(cls, label: str) -> ScenarioStateId:
        """Map a normalized ``ja``/``nee`` label to its state identifier."""
        normalized = str(label).strip().lower()
        if normalized == "ja":
            return cls.JA
        if normalized == "nee":
            return cls.NEE
        raise ValueError(f"Unsupported scenario state label: {label!r}")

    def transform_pf(self, pf_h: float) -> float:
        """Return direct Pf for ``JA`` and complementary Pf for ``NEE``."""
        if self is ScenarioStateId.JA:
            return pf_h
        return 1.0 - pf_h

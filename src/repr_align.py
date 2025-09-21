# src/repr_align.py
# =============================================================================
# Representational alignment (ideal + helper)
# =============================================================================
# Responsibilities
#   • Define the joint reliability R_joint = min(Rv, Rt).
#   • Ideal alignment I_align = R_joint (identity).
#   • Kept minimal by design; the evaluator computes the measurement S_align.
# =============================================================================

from __future__ import annotations


def joint_reliability(Rv: float, Rt: float) -> float:
    """Joint reliability = bottleneck(Rv, Rt)."""
    return float(min(Rv, Rt))


def compute_I(R_joint: float) -> float:
    """Ideal alignment equals the joint reliability (identity)."""
    return float(R_joint)
# src/superadditivity.py
# =============================================================================
# Superadditivity / inverse effectiveness (ideal)
# =============================================================================
# Responsibilities
#   • Provide the divisive‑normalisation style ideal:
#         I_joint = Rv + Rt − Rv*Rt  (clipped to [0,1])
#   • The evaluator computes S_joint from embedding norms and compares to this.
# =============================================================================

from __future__ import annotations


def compute_I(Rv: float, Rt: float) -> float:
    """
    Divisive‑normalisation style ideal for joint reliability.
    """
    ideal = Rv + Rt - (Rv * Rt)
    return float(min(max(ideal, 0.0), 1.0))
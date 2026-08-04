from __future__ import annotations

from typing import Optional

import numpy as np


def gsas_relative_chi_square_ftol(relative_improvement: float) -> Optional[float]:
    """Return SciPy ``ftol`` for a GSAS-II-style min dM/M setting.

    GSAS-II stops a refinement cycle when the relative improvement in the
    weighted chi-square objective is smaller than the configured ``min dM/M``.
    SciPy's least-squares ``ftol`` is the closest equivalent for the optimizers
    used here.
    """

    value = max(float(relative_improvement), 0.0)
    if value <= 0.0:
        return None
    return max(value, float(np.finfo(float).eps))


def fitting_settings_min_dm_over_m(settings: object) -> float:
    if hasattr(settings, "gsas_min_delta_chi_square"):
        return max(float(getattr(settings, "gsas_min_delta_chi_square")), 0.0)
    return max(float(getattr(settings, "batch_stagnation_relative_improvement", 0.0)), 0.0)

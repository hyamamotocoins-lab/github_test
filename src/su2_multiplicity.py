"""Independent SU(2) singlet multiplicity via integer weight counting.

mu_0 = w_0 - w_2 from the magnetic-weight distribution of a tensor product.
This does not use Clebsch–Gordan coefficients or SymPy and is therefore an
independent reference for the armillary column count.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache


@lru_cache(maxsize=None)
def singlet_multiplicity(reps: tuple[int, ...]) -> int:
    """Return the multiplicity of the trivial irrep in ⊗_ℓ V_{j2_ℓ}.

    Each j2 contributes doubled magnetic weights ``range(-j2, j2 + 1, 2)``.
    """
    if any(not isinstance(j2, int) or isinstance(j2, bool) or j2 < 0 for j2 in reps):
        raise ValueError('Representation labels must be nonnegative integers.')
    weights: Counter[int] = Counter({0: 1})
    for j2 in reps:
        updated: Counter[int] = Counter()
        for total_m2, count in weights.items():
            for local_m2 in range(-j2, j2 + 1, 2):
                updated[total_m2 + local_m2] += count
        weights = updated
    return int(weights[0] - weights[2])


MULTIPLICITY_METHOD = 'weight_count_w0_minus_w2_v1'

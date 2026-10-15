# src/rngd/optimize/metrics.py
def fitness_from_terms(terms: dict, weights: dict, cores_reached: int, n_cores: int) -> float:
    """Re-score cached objective terms under new weights (no re-decoding needed)."""
    total_w = sum(weights[k] for k in terms) or 1.0
    f = sum(weights[k] * terms[k] for k in terms) / total_w
    return f - 0.5 * max(0, min(n_cores, LIMITS["exits"]) - cores_reached)

# src/rngd/optimize/decide.py
def pareto_mask(alts, keys=None) -> np.ndarray:
    keys = keys or list(CRITERIA)
    X = _matrix(alts, keys) * np.array([CRITERIA[k][0] for k in keys])   # larger = better
    n = len(alts)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep

def topsis(alts, weights: dict) -> np.ndarray:
    keys = [k for k in CRITERIA if weights.get(k, 0) > 0]
    if not alts or not keys:
        return np.zeros(len(alts))
    X = _matrix(alts, keys)
    norm = np.sqrt((X ** 2).sum(0))
    norm[norm == 0] = 1
    w = np.array([weights[k] for k in keys], float)
    V = X / norm * (w / w.sum())
    sign = np.array([CRITERIA[k][0] for k in keys])
    ideal = np.where(sign > 0, V.max(0), V.min(0))
    worst = np.where(sign > 0, V.min(0), V.max(0))
    dp = np.sqrt(((V - ideal) ** 2).sum(1))
    dm = np.sqrt(((V - worst) ** 2).sum(1))
    return dm / np.where(dp + dm == 0, 1, dp + dm)

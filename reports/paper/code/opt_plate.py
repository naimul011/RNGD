# src/rngd/optimize/typology.py
def elevators_for(keys: int) -> int:
    """Rule of thumb for select-service hotels: about one elevator per 75 keys, at least two."""
    return max(2, math.ceil(keys / 75))

def _run_ft(shape: str, a: int, b: int, wd: int) -> float:
    """Length of double-loaded corridor the shape offers (corners count once)."""
    return {"Bar": a, "L": a + b - wd, "U": a + 2 * b - 2 * wd, "T": a + b - wd, "H": 2 * b + a - 2 * wd,
            "Courtyard": 2 * a + 2 * b - 4 * wd}[shape]

"""Design decision making over the explored layouts.

The optimizer's fitness picks good layouts; choosing between good layouts is a multi-criteria
decision (more keys vs. lower height vs. less facade per key ...). This module does that step,
deterministically, and separately from the fitness:

  1. Pareto front: a layout is dominated if another one is at least as good on every criterion
     and better on one. Dominated layouts stay browsable but are flagged.
  2. TOPSIS (Hwang & Yoon 1981): normalise each criterion, weight it, and rank by closeness to
     the ideal layout and distance from the worst. Weights come from a preset or sliders.

Jev (TypeSafe's hosted decision model, https://docs.openclaw.ai/plugins/typesafe) can be
plugged in as an optional *advisor* on the final shortlist via jev_choice(): it needs
TYPESAFE_API_KEY and TYPESAFE_JEV_URL in .env. It is a cloud call, so it is never part of the
fitness or the ranking above; when it is not configured the app says so and nothing changes.
"""
from __future__ import annotations

import json
import os

import numpy as np

# criterion -> (direction, label, unit)
CRITERIA = {
    "fitness": (+1, "Fitness (all objectives)", ""),
    "keys": (+1, "Keys placed", "keys"),
    "program_placed": (+1, "Program placed", "share"),
    "daylight": (+1, "Rooms with windows", "share"),
    "corridor_ratio": (-1, "Corridor share", "share"),
    "gross_sf_per_key": (-1, "Gross SF per key", "SF"),
    "facade_ft_per_key": (-1, "Facade per key (cost)", "ft"),
    "height_ft": (-1, "Building height", "ft"),
    "dead_end_ft": (-1, "Dead-end corridor", "ft"),
    "max_travel_ft": (-1, "Exit travel", "ft"),
    "code_pass": (+1, "Code checks passed", "share"),
    "compactness": (+1, "Shared walls (prefab)", "0-1"),
}
PRESETS = {
    "Balanced": {"fitness": 3, "keys": 3, "program_placed": 2, "daylight": 1, "corridor_ratio": 1, "gross_sf_per_key": 1,
                 "facade_ft_per_key": 1, "height_ft": 1, "dead_end_ft": 1, "max_travel_ft": 0.5, "code_pass": 2,
                 "compactness": 0.5},
    "Most keys": {"keys": 5, "program_placed": 3, "fitness": 2, "code_pass": 2, "daylight": 1},
    "Lowest cost": {"gross_sf_per_key": 4, "facade_ft_per_key": 4, "corridor_ratio": 2, "keys": 2, "code_pass": 2,
                    "compactness": 1, "height_ft": 1},
    "Guest experience": {"daylight": 3, "max_travel_ft": 2, "dead_end_ft": 1, "fitness": 2, "keys": 2, "code_pass": 2},
    "Low-rise": {"height_ft": 4, "keys": 3, "code_pass": 2, "fitness": 1},
}


def _matrix(alts, keys):
    return np.array([[float(a.criteria[k]) for k in keys] for a in alts], float)


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


def rank(alts, weights: dict) -> list[dict]:
    """Ranked table rows: TOPSIS closeness, Pareto flag and the criteria."""
    if not alts:
        return []
    score = topsis(alts, weights)
    front = pareto_mask(alts)
    order = np.argsort(-score, kind="stable")
    rows = []
    for r, i in enumerate(order, 1):
        a = alts[i]
        rows.append({"rank": r, "id": a.id, "layout": a.plate.name, "shape": a.plate.shape, "floors": a.floors,
                     "score": round(float(score[i]), 4), "pareto": bool(front[i]), "stage": a.stage,
                     **{k: a.criteria[k] for k in CRITERIA}})
    return rows


def why(row: dict, rows: list[dict]) -> str:
    """One sentence on where a layout is strongest / weakest relative to the others."""
    good, bad = [], []
    for k, (sgn, label, _) in CRITERIA.items():
        vals = np.array([r[k] for r in rows], float)
        if vals.max() == vals.min():
            continue
        pos = (row[k] - vals.min()) / (vals.max() - vals.min())
        pos = pos if sgn > 0 else 1 - pos
        (good if pos >= 0.85 else bad if pos <= 0.15 else []).append(label.lower())
    s = f"Best-in-set on {', '.join(good[:3])}" if good else "No criterion at the top of the set"
    return s + (f"; weakest on {', '.join(bad[:3])}." if bad else ".")


# ------------------------------------------------------------------ Jev (optional, advisory)
def jev_status() -> tuple[bool, str]:
    key, url = os.environ.get("TYPESAFE_API_KEY"), os.environ.get("TYPESAFE_JEV_URL")
    if key and url:
        return True, "Jev configured (TypeSafe); used as an advisor on the shortlist only."
    return False, ("Jev not configured: it is TypeSafe's hosted decision model (no local weights, no public "
                   "endpoint yet). Add TYPESAFE_API_KEY and TYPESAFE_JEV_URL to .env to enable it. The ranking "
                   "uses the local TOPSIS model, which is deterministic.")


def jev_choice(rows: list[dict], question: str, model: str = "typesafe/jev-latest", timeout: float = 20.0) -> dict:
    """Ask Jev a Choice question over 2-255 alternatives; returns {id: probability} or an error."""
    ok, msg = jev_status()
    if not ok:
        return {"ok": False, "message": msg}
    import requests
    alts = [{"id": r["id"], "description": f"{r['layout']}, {r['floors']} guest floors, {r['keys']} keys, "
                                          f"corridor {r['corridor_ratio']:.0%}, {r['gross_sf_per_key']:.0f} gross SF/key, "
                                          f"height {r['height_ft']:.0f} ft, code {r['code_pass']:.0%}"} for r in rows[:255]]
    try:
        resp = requests.post(os.environ["TYPESAFE_JEV_URL"], timeout=timeout,
                             headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
                             json={"model": model, "type": "choice", "question": question, "alternatives": alts})
        resp.raise_for_status()
        data = resp.json()
        probs = data.get("probabilities") or data.get("result") or data
        return {"ok": True, "probabilities": probs, "raw": json.dumps(data)[:2000]}
    except Exception as e:  # network / schema problems never affect the local ranking
        return {"ok": False, "message": f"Jev call failed: {e}"}

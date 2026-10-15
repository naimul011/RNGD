# src/rngd/ifc/explorer.py
def _recenter(meshes: list[dict], outlier_factor: float = 4.0, min_outlier_dist_m: float = 50.0) -> tuple[list[dict], dict]:
    """Translates every mesh so the building sits near the origin, and drops
    elements sitting implausibly far from the rest of the model (see
    STRUCTURAL_CLASSES above). This is a display-space transform applied only
    to the in-memory/cached mesh data -- the source .ifc file on disk is never
    modified. Returns (kept_meshes, report)."""
    structural = [m for m in meshes if m["cls"] in STRUCTURAL_CLASSES]
    reference = structural or meshes
    ref_verts = np.concatenate([m["verts"] for m in reference], axis=0)
    origin = np.median(ref_verts, axis=0)
    diag = float(np.linalg.norm(ref_verts.max(axis=0) - ref_verts.min(axis=0)))
    threshold = max(diag * outlier_factor, min_outlier_dist_m)

    kept, dropped = [], []
    for m in meshes:
        centroid = m["verts"].mean(axis=0)
        dist = float(np.linalg.norm(centroid - origin))
        if dist > threshold:
            dropped.append({"name": m["name"], "cls": m["cls"], "distance_m": round(dist, 1)})
            continue
        m = dict(m, verts=m["verts"] - origin)
        kept.append(m)

    report = {
        "origin_offset_m": [round(x, 2) for x in origin.tolist()],
        "building_diagonal_m": round(diag, 1),
        "excluded_outliers": dropped,
    }
    return kept, report

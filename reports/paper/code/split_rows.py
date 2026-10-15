# src/rngd/hotel/layout.py
def split_rows(program: Program, upper_levels: int):
    """Sort non-guest rows into: per-floor support (small rooms with count >= 2), core
    support on Level 1 (small BOH rooms), Level 1 band rooms, tenant blocks.
    Stairs and corridors are generated, not packed."""
    per_floor, core_l1, band, tenant = [], [], [], []
    for name, count, area, cat in program.rows:
        if cat == "guest" or count == 0:
            continue
        kind = _kind_for(name, cat)
        if kind in ("stair", "corridor"):
            continue
        each = area / count
        if count >= 2 and each < 150:
            # Small repeated rooms (Ice, IT, Jan, Supplies): one per guest floor, total area kept.
            per_floor.append((name, area / upper_levels, kind))
        elif cat == "boh" and area < 150:
            core_l1.append((name, area, kind))
        elif kind == "tenant":
            tenant.append((name, area, kind))
        else:
            band += [(name, each, kind)] * count
    return per_floor, core_l1, band, tenant

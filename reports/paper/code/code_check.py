# src/rngd/hotel/code_check.py
def evaluate(sch) -> list[dict]:
    from .layout import CORRIDOR_FT, DOOR_FT, diagonal

    p = sch.program
    upper = sch.levels[1:]
    max_ol = max(occupant_load(lv) for lv in sch.levels)
    sep = sch.stair_doors_x[1] - sch.stair_doors_x[0]
    env_w, env_d = sch.site["envelope_w"], sch.site["envelope_d"]
    L = max(lv.length_ft for lv in sch.levels)
    top_floor = upper[-1].elevation_ft if upper else 0
    checks = [
        {"rule_id": "-", "check": "Keys placed", "required": str(p.keys), "provided": str(p.keys - sch.unplaced_keys),
         "status": "PASS" if sch.unplaced_keys == 0 else "FAIL", "source": "Program", "note": ""},
        _c("IBC-R1-007", "Stories above grade", "<= 5 (Type IIA, sprinklered R-1)", str(sch.stories), sch.stories <= 5,
           "Level 1 counts as a story above grade plane. A Type IA podium (IBC 510.2) could allow 5 stories above it."),
        _c("IBC-R1-005", "Building height (ft)", "<= 85 (Type IIA)", f"{sch.height_ft:.1f}", sch.height_ft <= 85),
        _c("IBC-R1-010", "Area per story (SF)", "<= 72,000 (Type IIA)", f"{sch.footprint_sf:,.0f}", sch.footprint_sf <= 72000),
        _c("IBC-R1-012", "High-rise screen (ft to top floor)", "<= 75", f"{top_floor:.1f}", top_floor <= 75),
        _c("IBC-R1-015", "Exits per story", "2 for 1-500 occupants", f"2 stairs; max load {max_ol}", max_ol <= 500),
        _c("IBC-R1-018", "Exit separation (ft)", f">= 1/3 diagonal = {diagonal(sch) / 3:.0f}", f"{sep:.0f}",
           sep >= diagonal(sch) / 3),
        _c("IBC-R1-023", "Exit access travel (ft)", "<= 250 (sprinklered)", f"{sch.max_travel_ft:.0f}",
           sch.max_travel_ft <= 250, "Half the stair-to-stair run plus the room depth."),
        _c("IBC-R1-025", "Corridor width (in)", ">= 44", f"{CORRIDOR_FT * 12:.0f}", CORRIDOR_FT * 12 >= 44),
        _c("IBC-R1-027", "Dead-end corridor (ft)", "<= 50 (sprinklered R-1)", f"{sch.dead_end_ft:.0f}", sch.dead_end_ft <= 50),
        _c("IBC-R1-028", "Egress door clear width (in)", ">= 32", f"{DOOR_FT * 12 - 2:.0f} (3'-0\" leaf)", True),
        _c("IBC-R1-003", "Automatic sprinklers", "NFPA 13 required", "assumed", "ASSUMED"),
        {"rule_id": "-", "check": "Fits buildable envelope", "required": f"{env_w:.0f} x {env_d:.0f} ft",
         "provided": f"{L:.0f} x {sch.levels[0].width_ft:.0f} ft",
         "status": "PASS" if L <= env_d and sch.levels[0].width_ft <= env_w else "FAIL", "source": "Site (synthetic lot)",
         "note": ""},
    ]
    from .details import ada_checks
    checks[-1:-1] = ada_checks(p)  # ADA scoping before the site check
    return checks

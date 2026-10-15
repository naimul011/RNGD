# src/rngd/agents/conflict_agent.py
def _compute_feasibility(program: CustomerProgram, site: SiteConstraints, baap: BaaPPlan) -> dict:
    lot_area = site.lot_area_sf
    zr = site.zoning
    max_gsf_by_far = lot_area * zr.max_far if zr.max_far else None
    max_footprint_by_coverage = lot_area * (zr.max_lot_coverage_pct / 100) if zr.max_lot_coverage_pct else None

    required_stalls = None
    if zr.parking_ratio_per_unit and program.unit_count_total:
        required_stalls = round(zr.parking_ratio_per_unit * program.unit_count_total)

    surface_parking_area_needed = required_stalls * SF_PER_STALL_WITH_AISLE_ALLOWANCE if required_stalls else None

    envelope_area = site.buildable_envelope.area_sf if site.buildable_envelope else None
    footprint = baap.estimated_footprint_sf_per_floor
    land_left_for_parking = (envelope_area - footprint) if envelope_area is not None else None

    return {
        "lot_area_sf": lot_area,
        "max_buildable_gsf_by_far": max_gsf_by_far,
        "requested_building_gsf": baap.estimated_building_gsf,
        "max_footprint_by_lot_coverage_sf": max_footprint_by_coverage,
        "estimated_footprint_per_floor_sf": footprint,
        "buildable_envelope_area_sf": envelope_area,
        "required_parking_stalls": required_stalls,
        "surface_parking_area_needed_sf": surface_parking_area_needed,
        "land_left_for_parking_after_footprint_sf": land_left_for_parking,
        "surface_parking_likely_fits": (
            (surface_parking_area_needed <= land_left_for_parking)
            if surface_parking_area_needed is not None and land_left_for_parking is not None
            else None
        ),
    }

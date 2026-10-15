# src/rngd/agents/zoning_agent.py
def _compute_buildable_envelope(lot: dict, setbacks: dict, easements: list[Easement]) -> BuildableEnvelope:
    frontage, depth = lot["frontage_ft"], lot["depth_ft"]
    x_min = setbacks.get("side_west", 0.0)
    x_max = frontage - setbacks.get("side_east", 0.0)
    y_min = setbacks.get("front", 0.0)
    y_max = depth - setbacks.get("rear", 0.0)

    for e in easements:
        if not e.buildable:
            if e.side == "east":
                x_max = min(x_max, frontage - e.width_ft)
            elif e.side == "west":
                x_min = max(x_min, e.width_ft)
            elif e.side == "north":
                y_min = max(y_min, e.width_ft)
            elif e.side == "south":
                y_max = min(y_max, depth - e.width_ft)

    width = max(0.0, x_max - x_min)
    depth_env = max(0.0, y_max - y_min)
    return BuildableEnvelope(
        x_min_ft=x_min,
        x_max_ft=x_max,
        y_min_ft=y_min,
        y_max_ft=y_max,
        width_ft=width,
        depth_ft=depth_env,
        area_sf=round(width * depth_env, 1),
    )

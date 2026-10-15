# src/rngd/hotel/layout.py
# First pass: guest wings, balanced so both sides end near the same x.
wings = []
for group in per:
    sides = {"S": [], "N": []}
    run = {"S": 0.0, "N": STAIR_W_FT}  # north side also has to fit Stair 2
    for r in group:
        s = "S" if run["S"] <= run["N"] else "N"
        sides[s].append(r)
        run[s] += r["area_sf"] / ROOM_DEPTH_FT
    wings.append((sides, run))
length = CORE_W_FT + max(max(run.values()) for _, run in wings)

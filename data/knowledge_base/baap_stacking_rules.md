# RNGD BaaP Stacking, Adjacency, and Repeat Rules (Sample)

> SAMPLE / SYNTHETIC DATA answering the "if/then" rules referenced in RNGD kickoff
> question 31-33. Encodes the kind of authoritative if/then logic RNGD's real BaaP
> source of truth would provide.

## Vertical Stacking Rules
1. **Core alignment (hard rule):** `CORE_STD` modules must occupy the identical (x, y)
   footprint on every floor. Cores do not move, rotate, or resize between floors.
2. **MEP alignment (hard rule):** `MEP_RISER` shafts must stack vertically within the
   same footprint on every floor served.
3. **Unit stacking (soft preference):** Unit modules should stack within +/-2 ft of plan
   offset floor-to-floor to keep structure and risers efficient; occasional offset is
   allowed at the top floor or over transfer structure but should be flagged as an
   exception, not silently accepted.
4. **Repeat structure (soft preference):** A typical floor plate, once resolved, should
   repeat unmodified for all "typical" floors above the ground floor. Only the ground
   floor and top floor are expected to differ.

## Horizontal / Adjacency Rules
5. **Egress distance (hard rule):** No occupied point in a corridor may be more than
   200 ft of travel from a `CORE_STD` exit stair (dead-end segments capped at 50 ft —
   see zoning code 4.4).
6. **Double-loaded corridor (soft preference):** `CORRIDOR_STD` should run between two
   rows of unit modules (double-loaded) rather than serving a single row, to maximize
   BaaP utilization per linear foot of corridor.
7. **Positive adjacency (soft preference):** `AMENITY_LOBBY` should sit adjacent to the
   primary building entrance and to `CORE_STD`; `AMENITY_FIT` should sit near
   `AMENITY_LOBBY`.
8. **Negative adjacency (hard rule):** `BOH_LOADING` must not be adjacent to
   `AMENITY_LOBBY` or directly abut a unit's primary entrance; service access should be
   on a separate building face from the main entrance where the site allows it.
9. **Accessible unit distribution (hard rule):** Required accessible units
   (`ACCESSIBLE_1BR`) must be distributed across the ground floor and floors served by
   an elevator (i.e., not stair-only floors), and located on an accessible route from
   the accessible core.

## Site-Level Placement Rules
10. **Setback clearance (hard rule):** No BaaP module or parking stall may encroach
    into a required setback or a recorded easement.
11. **Fire access (hard rule):** At least one building face must be within 150 ft of a
    fire apparatus access lane meeting zoning code 4.4.
12. **Parking-to-entrance accessible route (hard rule):** At least one accessible
    route (zoning code 4.5) must connect an accessible parking stall to the accessible
    building entrance without crossing a drive aisle without a marked crossing.

## How the Agents Use These Rules
- The **BaaP Matching Agent** uses rules 1-9 to propose a module palette and adjacency
  plan (which modules, how many, roughly how they relate to each other).
- The **Simulation Engine** uses rules 10-12 as hard geometric constraints (never
  violate) and rules 6-8 as soft objectives (bias the physics-style placement toward
  satisfying them, but do not fail if imperfect).
- The **Vision/CV QA Agent** re-measures the rendered layout against rules 10-12 after
  simulation, since these are the rules a raster measurement can actually verify.

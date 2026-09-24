# RNGD Building-as-a-Product (BaaP) Catalog — Sample Module Library

> SAMPLE / SYNTHETIC DATA — standing in for RNGD's actual BaaP data source of truth
> (RNGD kickoff questions 28-38), which may live in Revit families, a database, or
> spreadsheets. The structured, machine-readable twin of this document is
> `data/baap_catalog.json`; agents retrieve narrative context from here and read exact
> numbers from the JSON so the two never disagree.

## Structural Platform
- Structural grid / bay size: **30 ft x 30 ft** typical column bay.
- Floor-to-floor height: **10 ft 6 in** (residential levels).
- Ground floor-to-floor height: **14 ft** (accommodates lobby/amenity/back-of-house).

## Unit Modules
Each unit module is a fixed-footprint BaaP product with defined structural, MEP, and
interior fit-out already resolved — the design problem is *which modules, how many,
and how they stack*, not custom unit design.

| Module ID        | Type              | Width x Depth (ft) | GSF | NSF |
|-------------------|-------------------|---------------------|-----|-----|
| STUDIO_A          | Studio            | 26 x 32             | 550 | 480 |
| ONEBR_A           | 1-Bedroom         | 30 x 34             | 725 | 640 |
| TWOBR_A           | 2-Bedroom         | 34 x 36             | 1020| 900 |
| ACCESSIBLE_1BR    | 1-Bedroom (ADA)   | 32 x 34             | 760 | 665 |

## Non-Unit ("Building" / Back-of-House) BaaP Modules
| Module ID     | Description                                   | Width x Depth (ft) | Notes |
|---------------|------------------------------------------------|---------------------|-------|
| CORE_STD      | Egress core: 2 stairs + 1 elevator + trash chute | 34 x 24           | One required per ≤200 ft exit travel distance; stacks identically every floor. |
| CORRIDOR_STD  | Double-loaded corridor segment                  | 6 ft clear width   | Runs along core-to-core spine; max 50 ft dead-end. |
| MEP_RISER     | Mechanical/electrical/plumbing shaft            | 6 x 6               | One per ~40 units; must align vertically through all floors. |
| AMENITY_LOBBY | Lobby + leasing office                          | ~1,200 GSF          | Ground floor only, at primary entrance. |
| AMENITY_FIT   | Fitness / resident amenity room                 | ~900 GSF            | Ground floor preferred. |
| BOH_LOADING   | Loading, trash staging, mail                    | ~600 GSF            | Ground floor, service-side, away from main entrance. |

## Parking Modules
| Module ID        | Description                     | Dimensions        |
|-------------------|----------------------------------|--------------------|
| STALL_STD         | Standard surface stall           | 9 ft x 18 ft       |
| STALL_COMPACT     | Compact stall                    | 8.5 ft x 16 ft     |
| STALL_ADA         | Accessible stall + 5 ft aisle    | 11 ft x 18 ft + aisle |
| STALL_ADA_VAN     | Van-accessible stall + 8 ft aisle| 12 ft x 18 ft + aisle |
| DRIVE_AISLE       | Two-way 90-degree aisle          | 24 ft clear width  |

## Grossing / Efficiency Factor
- Typical unit-to-gross efficiency for this BaaP kit: **~85%** (i.e. total building GSF
  is approximately sum of unit GSF / 0.85, the remainder being corridors, cores, shafts, and common
  walls). Use this as a planning-level factor only; the BaaP Matching agent recomputes
  it explicitly from the selected module counts where possible.

"""Hotel floor-plan optimizer.

Consumes the same inputs as the design-optimization prototype in QA/ (multi-level
1 ft occupancy grids `fp*_multi.json` and a room program `rooms_hotel.json`) and produces
an optimized corridor + room layout per level, using a memetic genetic algorithm whose
fitness combines deterministic objectives (placement, daylight, efficiency, IBC egress
rules from RNGD's CSV, ADA scoping, adjacency) with an LLM design-review agent grounded
by retrieval (RAG) at a few checkpoints.

Modules: site (grids), program (rooms), engine (genome -> corridor -> rooms), metrics
(objectives + code checks), review (LLM + RAG critic), search (memetic GA), baseline
(evaluate the QA reference designs), render (drawings), report (AI-written PDF).
"""

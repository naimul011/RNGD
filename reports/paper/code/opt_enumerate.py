# src/rngd/optimize/search.py
top = list(range(min(m, 7)))
combos = [list(range(m))] + [list(c) for k in range(1, min(4, len(top)) + 1)
                             for c in itertools.combinations(top, k)]
genomes = []
for c in combos:
    g = engine._seed(self.rng, n_inst, c)
    g[engine.N_SEG * 4:] = 0.5      # rooms in priority order
    genomes.append(g)

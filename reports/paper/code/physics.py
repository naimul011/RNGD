# src/rngd/simulation/physics_engine.py
def _separation_force(a: Body, b: Body) -> tuple[float, float]:
    ax0, ay0, ax1, ay1 = a.bounds()
    bx0, by0, bx1, by1 = b.bounds()
    ox = _overlap_1d(ax0, ax1, bx0, bx1)
    oy = _overlap_1d(ay0, ay1, by0, by1)
    if ox <= 0 or oy <= 0:
        return 0.0, 0.0
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    # Push apart along the axis of least overlap (standard rectangle
    # separating-axis heuristic). `a` must move AWAY from `b`: if a's center
    # is left of b's, a moves further left (negative x), so the sign is the
    # opposite of the naive "a < b -> positive" comparison.
    if ox < oy:
        direction = -1.0 if acx < bcx else 1.0
        return direction * ox, 0.0
    else:
        direction = -1.0 if acy < bcy else 1.0
        return 0.0, direction * oy

def step(self, damping: float = 0.55, gain: float = 0.5) -> float:
    total_penalty = 0.0
    forces = {b.id: [0.0, 0.0] for b in self.bodies}

    for b in self.bodies:
        cfx, cfy = self._containment_force(b)
        forces[b.id][0] += cfx
        forces[b.id][1] += cfy
        total_penalty += abs(cfx) + abs(cfy)

    n = len(self.bodies)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = self.bodies[i], self.bodies[j]
            sx, sy = _separation_force(a, b)
            if sx == 0 and sy == 0:
                continue
            total_penalty += abs(sx) + abs(sy)
            if a.movable:
                forces[a.id][0] += sx * 0.5
                forces[a.id][1] += sy * 0.5
            if b.movable:
                forces[b.id][0] -= sx * 0.5
                forces[b.id][1] -= sy * 0.5

    for b in self.bodies:
        if not b.movable:
            continue
        fx, fy = forces[b.id]
        b.vx = (b.vx + gain * fx) * damping
        b.vy = (b.vy + gain * fy) * damping
        b.x += b.vx
        b.y += b.vy

    self.history.append(total_penalty)
    return total_penalty

def run(self, max_steps: int = 250, tol: float = 0.5) -> dict:
    converged = False
    steps_run = 0
    for i in range(max_steps):
        penalty = self.step()
        steps_run = i + 1
        if penalty < tol:
            converged = True
            break
    return {
        "steps_run": steps_run,
        "max_overlap_penalty": round(self.history[-1] if self.history else 0.0, 3),
        "converged": converged,
        "history": self.history,
    }

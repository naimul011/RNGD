# src/rngd/hotel/drawings.py
def room_xf(r, mirror: bool):
    """Local (u, v) of a guest room -> plan (x, y); v runs away from the corridor."""
    def xf(u, v):
        x = r.x + (r.w - u if mirror else u)
        y = r.y + r.d - v if r.side == "S" else r.y + v
        return x, y
    return xf

def __init__(self, az_deg=-55.0, el_deg=32.0):
    th, ph = math.radians(az_deg), math.radians(el_deg)
    self.c = np.array([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph)])
    self.r = np.array([-math.sin(th), math.cos(th), 0.0])
    self.u = np.cross(self.c, self.r)
    self.items = []  # (depth, polys[(pts, fc, ec, lw)])

def p(self, pts):
    a = np.asarray(pts, float)
    return np.c_[a @ self.r, a @ self.u]

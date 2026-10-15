# src/rngd/hotel/details.py
def ada_required(rooms: int) -> tuple[int, int, int]:
    """(without roll-in, with roll-in, total) mobility rooms for `rooms` guest rooms."""
    for cap, a, b in ADA_MOBILITY:
        if rooms <= cap:
            return a, b, a + b
    if rooms <= 1000:
        return math.ceil(rooms * 0.02), math.ceil(rooms * 0.01), math.ceil(rooms * 0.03)
    extra = math.ceil((rooms - 1000) / 100)
    return 20 + extra, 10 + extra, 30 + 2 * extra

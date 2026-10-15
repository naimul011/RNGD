# src/rngd/agents/baap_matching_agent.py
def _module_counts(unit_count_total: int, unit_mix_pct: dict[str, float]) -> dict[str, int]:
    counts: dict[str, int] = {}
    assigned = 0
    items = list(unit_mix_pct.items())
    for i, (utype, pct) in enumerate(items):
        module_id = UNIT_TYPE_TO_MODULE.get(utype.lower(), "ONEBR_A")
        if i == len(items) - 1:
            n = unit_count_total - assigned  # last bucket absorbs rounding remainder
        else:
            n = round(unit_count_total * pct)
        assigned += n
        counts[module_id] = counts.get(module_id, 0) + max(0, n)
    return counts

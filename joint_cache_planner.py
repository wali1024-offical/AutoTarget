"""Joint dynamic programming over exact anchors and cache representations."""

from __future__ import annotations

from collections.abc import Mapping


Edge = tuple[int, int]


def select_joint_plan(
    edge_costs: Mapping[str, Mapping[Edge, float]],
    *,
    steps: int,
    budget: int,
    require_final: bool,
) -> tuple[list[int], list[dict], float]:
    """Choose exact anchors and the cheapest representation for each segment."""
    if not edge_costs:
        raise ValueError("At least one cache representation is required")
    if not 1 <= budget <= steps:
        raise ValueError("Budget must lie between one and the number of steps")
    if require_final and steps > 1 and budget < 2:
        raise ValueError("A mandatory final anchor requires at least two exact calls")

    def best_edge(anchor: int, boundary: int) -> tuple[float, str]:
        candidates = []
        for representation, costs in edge_costs.items():
            if (anchor, boundary) not in costs:
                raise ValueError(f"Missing edge {(anchor, boundary)} for {representation}")
            candidates.append((float(costs[(anchor, boundary)]), representation))
        return min(candidates)

    # State stores cost, anchors, and completed cache segments.
    states: dict[tuple[int, int], tuple[float, list[int], list[dict]]] = {
        (1, 0): (0.0, [0], [])
    }
    for used in range(1, budget):
        for (state_used, last), (cost, anchors, segments) in list(states.items()):
            if state_used != used:
                continue
            remaining = budget - used
            maximum = steps - remaining
            for next_anchor in range(last + 1, maximum + 1):
                if require_final and remaining == 1 and next_anchor != steps - 1:
                    continue
                edge_cost, representation = best_edge(last, next_anchor)
                candidate = (
                    cost + edge_cost,
                    [*anchors, next_anchor],
                    [
                        *segments,
                        {
                            "anchor": last,
                            "boundary": next_anchor,
                            "representation": representation,
                            "cost": edge_cost,
                        },
                    ],
                )
                key = (used + 1, next_anchor)
                if key not in states or candidate[0] < states[key][0]:
                    states[key] = candidate

    candidates = []
    for (used, last), (cost, anchors, segments) in states.items():
        if used != budget or (require_final and last != steps - 1):
            continue
        edge_cost, representation = best_edge(last, steps)
        candidates.append(
            (
                cost + edge_cost,
                anchors,
                [
                    *segments,
                    {
                        "anchor": last,
                        "boundary": steps,
                        "representation": representation,
                        "cost": edge_cost,
                    },
                ],
            )
        )
    if not candidates:
        raise RuntimeError("No feasible plan found")
    cost, anchors, segments = min(candidates, key=lambda item: item[0])
    return anchors, segments, cost

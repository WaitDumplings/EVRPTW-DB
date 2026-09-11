"""Deterministic physical witnesses for TERRAN-derived synthetic instances."""
from __future__ import annotations

import numpy as np


def native_singleton_candidate_feasible(customer_pos, depot_and_stations, ready, due,
                                        velocity, consumption, battery, charge_rate,
                                        service, horizon):
    """Check a candidate without changing any sampled coordinates/time bounds.

    Uses depot -> [charger a] -> customer -> [charger b] -> depot and prohibits
    a == b when both are chargers, matching the canonical route-local CS rule.
    All calculations are in the original generator's (non-normalized) units.
    """
    if ready > due:
        return False
    stops = np.asarray(depot_and_stations, dtype=float)
    to_customer = np.linalg.norm(stops - np.asarray(customer_pos).reshape(2), axis=1) / velocity
    direct = float(to_customer[0])
    margin = 1e-4
    if (2.0 * direct * consumption < battery - margin
            and max(direct, ready) <= due - margin
            and max(direct, ready) + service + direct < horizon - margin):
        return True
    depot_to_stop = np.linalg.norm(stops - stops[0], axis=1) / velocity
    prefix = depot_to_stop + depot_to_stop * consumption / charge_rate
    prefix[0] = 0.0
    arrival = prefix + to_customer
    start = np.maximum(arrival, ready)
    used = (to_customer[:, None] + to_customer[None, :]) * consumption
    finish = start[:, None] + service + to_customer[None, :]
    finish = finish + np.where(np.arange(len(stops))[None, :] > 0, used / charge_rate, 0.0)
    finish = finish + depot_to_stop[None, :]
    good = (depot_to_stop[:, None] * consumption < battery - margin)
    good = good & (used < battery - margin)
    good = good & (depot_to_stop[None, :] * consumption < battery - margin)
    good = good & (start[:, None] <= due - margin) & (finish < horizon - margin)
    good[np.arange(1, len(stops)), np.arange(1, len(stops))] = False
    return bool(np.any(good))


def singleton_witnesses(instance):
    """Return exactly one independently replayable singleton per customer.

    Returns (routes, missing_customer_ids). Absence of a two-charger witness is
    a bounded-constructor failure, not a claim that the full EVRPTW is infeasible.
    """
    n = instance.num_customers
    nodes = np.r_[0, np.arange(n + 1, instance.num_terminals)]
    customers = np.arange(1, n + 1)
    T = np.asarray(instance.raw_travel_time_matrix_s, dtype=float)
    E = np.asarray(instance.energy_matrix_kwh, dtype=float)
    D = np.asarray(instance.distance_matrix_km, dtype=float)
    B = float(instance.vehicle['battery_capacity_kwh'])
    H = float(instance.working_end_s)
    P = np.r_[1.0, np.asarray(instance.raw['charging_power_kw'], dtype=float)]
    prefix = T[0, nodes] + np.where(nodes > 0, E[0, nodes] / P * 3600.0, 0.0)
    ein = E[np.ix_(customers, nodes)]
    arrival = prefix[None, :] + T[np.ix_(customers, nodes)]
    start = np.maximum(arrival, instance.tw_s[:, 0, None])
    used = ein[:, :, None] + ein[:, None, :]
    depart = start + instance.service_time_s[:, None]
    finish = depart[:, :, None] + T[np.ix_(customers, nodes)][:, None, :]
    finish += np.where(nodes[None, None, :] > 0, used / P[None, None, :] * 3600.0, 0.0)
    finish += T[nodes, 0][None, None, :]
    good = (E[0, nodes][None, :, None] <= B + 1e-6) & (used <= B + 1e-6)
    good &= E[nodes, 0][None, None, :] <= B + 1e-6
    good &= start[:, :, None] <= instance.tw_s[:, 1, None, None] + 1e-6
    good &= finish <= H + 1e-6
    good[:, np.arange(1, len(nodes)), np.arange(1, len(nodes))] = False
    costs = D[0, nodes][None, :, None] + D[np.ix_(customers, nodes)][:, :, None]
    costs = costs + D[np.ix_(customers, nodes)][:, None, :] + D[nodes, 0][None, None, :]
    costs = np.where(good, costs, np.inf)
    routes, missing = [], []
    for k, customer in enumerate(customers):
        if not np.isfinite(costs[k]).any():
            missing.append(int(customer))
            continue
        a, b = np.unravel_index(np.argmin(costs[k]), costs[k].shape)
        routes.append([0] + ([int(nodes[a])] if a else []) + [int(customer)] + ([int(nodes[b])] if b else []) + [0])
    return routes, missing

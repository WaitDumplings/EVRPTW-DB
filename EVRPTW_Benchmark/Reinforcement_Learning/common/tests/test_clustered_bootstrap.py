from __future__ import annotations

import pandas as pd

from EVRPTW_Benchmark.Reinforcement_Learning.common.clustered_bootstrap import (
    paired_parent_family_bootstrap,
)


def test_bootstrap_pairs_methods_and_clusters_parent_families() -> None:
    rows = []
    for family in ("f1", "f2", "f3"):
        for instance in ("a", "b"):
            for method, distance in (("A", 10.0), ("B", 12.0)):
                for seed in (1, 2, 3):
                    rows.append(
                        {
                            "family_id": family,
                            "instance_id": f"{family}-{instance}",
                            "method": method,
                            "seed": seed,
                            "verifier_passed": True,
                            "objective_distance_km": distance + seed * 0.01,
                        }
                    )
    report = paired_parent_family_bootstrap(
        pd.DataFrame(rows), method_a="A", method_b="B", replicates=100, seed=9
    )
    assert report["paired_family_count"] == 3
    assert report["paired_instance_count"] == 6
    assert report["training_seeds_treated_as_independent_instances"] is False
    assert report["common_feasible_distance_difference_km_a_minus_b"]["estimate"] == -2.0

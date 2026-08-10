import pandas as pd

from evrptw_cle.visualization import _boolean_mask, component_color


def test_component_colors_are_deterministic_and_distinct_for_common_case() -> None:
    first = [component_color(rank) for rank in range(1, 21)]
    second = [component_color(rank) for rank in range(1, 21)]
    assert first == second
    assert len(set(first)) == len(first)
    assert first[0] == "#006d8f"


def test_boolean_mask_parses_graphml_string_values() -> None:
    parsed = _boolean_mask(pd.Series(["False", "True", False, True, "0", "1"]))
    assert list(parsed) == [False, True, False, True, False, True]

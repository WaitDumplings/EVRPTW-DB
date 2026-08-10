import geopandas as gpd
import networkx as nx
from shapely.geometry import MultiPolygon, box

from evrptw_cle.connectivity import audit_and_label
from evrptw_cle.operational import OperationalPolicy, select_operational_graph


def _add_bidirectional_edge(graph, u, v, osmid, length=500.0):
    graph.add_edge(u, v, osmid=osmid, highway="residential", length=length)
    graph.add_edge(v, u, osmid=osmid, highway="residential", length=length)


def test_operational_selection_uses_real_outside_nodes_to_join_city_components() -> None:
    graph = nx.MultiDiGraph(crs="EPSG:4326", simplified=True)
    coordinates = {
        1: (0.000, 0.001),
        2: (0.003, 0.001),
        10: (0.008, 0.001),
        11: (0.013, 0.001),
        3: (0.017, 0.001),
        4: (0.020, 0.001),
    }
    for node, (x, y) in coordinates.items():
        graph.add_node(node, x=x, y=y)
    for osmid, (u, v) in enumerate([(1, 2), (2, 10), (10, 11), (11, 3), (3, 4)], 1):
        _add_bidirectional_edge(graph, u, v, osmid)

    boundary = gpd.GeoDataFrame(
        geometry=[
            MultiPolygon(
                [
                    box(-0.001, -0.001, 0.004, 0.003),
                    box(0.016, -0.001, 0.021, 0.003),
                ]
            )
        ],
        crs="EPSG:4326",
    )
    raw = graph.subgraph({1, 2, 3, 4}).copy()
    raw_audit = audit_and_label(raw)
    selection = select_operational_graph(
        graph,
        raw,
        raw_audit,
        boundary,
        OperationalPolicy(
            buffer_ladder_km=(0.0, 2.0),
            min_node_coverage=1.0,
            min_road_length_coverage=1.0,
        ),
    )

    assert selection.summary["selected_buffer_km"] == 2.0
    assert selection.summary["city_node_coverage"] == 1.0
    assert nx.number_weakly_connected_components(selection.graph) == 1
    assert selection.graph.nodes[10]["transit_only"] is True
    assert selection.graph.nodes[1]["service_location_eligible"] is True
    assert selection.graph.number_of_edges() == graph.number_of_edges()


def test_operational_policy_rejects_unordered_buffer_ladder() -> None:
    policy = OperationalPolicy(buffer_ladder_km=(0.0, 5.0, 2.0))
    try:
        policy.validate()
    except ValueError as error:
        assert "increasing" in str(error)
    else:
        raise AssertionError("unordered buffer ladder should fail validation")

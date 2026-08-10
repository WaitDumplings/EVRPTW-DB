import networkx as nx

from evrptw_cle.connectivity import apply_component_policy, audit_and_label


def sample_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node, x, y in [
        (10, 0.0, 0.0),
        (11, 1.0, 0.0),
        (12, 2.0, 0.0),
        (20, 10.0, 0.0),
        (21, 11.0, 0.0),
        (30, 20.0, 0.0),
    ]:
        graph.add_node(node, x=x, y=y)
    graph.add_edge(10, 11)
    graph.add_edge(11, 10)
    graph.add_edge(11, 12)
    graph.add_edge(20, 21)
    return graph


def test_audit_labels_weak_components_by_size() -> None:
    graph = sample_graph()
    audit = audit_and_label(graph)
    assert audit.summary["weak_component_count"] == 3
    assert audit.summary["largest_weak_component_nodes"] == 3
    assert list(audit.components["component_id"]) == ["W0001", "W0002", "W0003"]
    assert graph.nodes[10]["weak_component_id"] == "W0001"
    assert graph.nodes[20]["weak_component_id"] == "W0002"
    assert graph.nodes[30]["weak_component_id"] == "W0003"


def test_largest_weak_policy_is_explicit_and_non_mutating() -> None:
    graph = sample_graph()
    audit_and_label(graph)
    selected = apply_component_policy(graph, "largest_weak")
    assert set(selected.nodes) == {10, 11, 12}
    assert set(graph.nodes) == {10, 11, 12, 20, 21, 30}


def test_all_policy_preserves_every_component() -> None:
    graph = sample_graph()
    audit_and_label(graph)
    selected = apply_component_policy(graph, "all")
    assert set(selected.nodes) == set(graph.nodes)

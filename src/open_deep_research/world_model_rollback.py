"""Phase 5 reduced deterministic rollback repair.

This module computes and optionally applies reduced rollback repairs without
introducing additional model calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from open_deep_research.world_model_scoring import dependents, take_snapshot

_DEPENDENCY_RELATIONS = {"depends_on", "used_in", "derived_from", "locks_in", "compresses"}


@dataclass
class RepairResult:
    contested_claims: list[str]
    retract_nodes: list[str]
    retract_edges: list[tuple[str, str, str]]
    regenerate_nodes: list[str]
    preserved_nodes: list[str]
    reopen_claims: list[str]


def _node_status(node: dict) -> str:
    data = node.get("data", {}) if isinstance(node.get("data"), dict) else {}
    return str(data.get("status", "")).lower()


def _edge_status(edge: dict) -> str:
    data = edge.get("data", {}) if isinstance(edge.get("data"), dict) else {}
    return str(data.get("status", "")).lower()


def _is_active_node(graph, node_id: str) -> bool:
    node = graph.nodes.get(node_id, {})
    return _node_status(node) != "retracted"


def _is_active_edge(graph, edge: dict) -> bool:
    if _edge_status(edge) == "retracted":
        return False
    src_id = edge.get("src_id")
    dst_id = edge.get("dst_id")
    if not src_id or not dst_id:
        return False
    return _is_active_node(graph, src_id) and _is_active_node(graph, dst_id)


def _commitment_target_nodes(graph, commitment_id: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    nodes: list[str] = []
    retract_edges: list[tuple[str, str, str]] = []
    for edge in graph.edges:
        if not _is_active_edge(graph, edge):
            continue
        if edge.get("src_id") != commitment_id:
            continue
        relation = edge.get("relation")
        if relation not in {"locks_in", "depends_on"}:
            continue
        dst_id = edge.get("dst_id")
        if dst_id:
            nodes.append(dst_id)
            retract_edges.append((commitment_id, dst_id, relation))
    return nodes, retract_edges


def _active_consumers(graph, node_id: str) -> set[str]:
    consumers: set[str] = set()
    for edge in graph.edges:
        if not _is_active_edge(graph, edge):
            continue
        if edge.get("dst_id") != node_id:
            continue
        if edge.get("relation") not in _DEPENDENCY_RELATIONS:
            continue
        src_id = edge.get("src_id")
        if src_id:
            consumers.add(src_id)
    return consumers


def _active_commitment_consumers(graph, node_id: str, excluding_commitment_id: str) -> set[str]:
    out: set[str] = set()
    for consumer_id in _active_consumers(graph, node_id):
        consumer = graph.nodes.get(consumer_id, {})
        if consumer.get("type") == "Commitment" and consumer_id != excluding_commitment_id:
            out.add(consumer_id)
    return out


def _unique_justifications_with_preserved(graph, commitment_id: str, config) -> tuple[set[str], set[str], list[tuple[str, str, str]]]:
    action = {"target_node_ids": [commitment_id], "predicted_delta": {"update_nodes": [], "update_edges": []}}
    dependent_nodes = dependents(graph, action, config)

    candidate_nodes, commit_edges = _commitment_target_nodes(graph, commitment_id)
    removable: set[str] = set()
    preserved: set[str] = set()

    for node_id in candidate_nodes:
        node = graph.nodes.get(node_id, {})
        node_type = node.get("type")
        if node_type not in {"Claim", "Assumption", "Hypothesis"}:
            continue
        if not bool(config.wm_preserve_independent):
            removable.add(node_id)
            continue

        other_commitments = _active_commitment_consumers(graph, node_id, commitment_id)
        if other_commitments:
            preserved.add(node_id)
            continue

        consumers = _active_consumers(graph, node_id)
        outside = {consumer_id for consumer_id in consumers if consumer_id not in dependent_nodes}
        if outside:
            preserved.add(node_id)
            continue

        removable.add(node_id)

    return removable, preserved, commit_edges


def justification_set(graph_like: Any, commitment_id: str, config) -> set[str]:
    """Return unique removable justification node ids for this commitment."""
    graph = take_snapshot(graph_like)
    removable, _preserved, _edges = _unique_justifications_with_preserved(graph, commitment_id, config)
    return {commitment_id, *removable}


def reduced_repair(graph_like: Any, fired_commitment: dict, config) -> RepairResult:
    """Build a deterministic reduced rollback repair plan for one stranded commitment."""
    graph = take_snapshot(graph_like)
    commitment_record = fired_commitment.get("commitment_record", {}) if isinstance(fired_commitment, dict) else {}
    commitment_id = str(commitment_record.get("commitment_id") or "")
    claim_id = str(commitment_record.get("claim_id") or "")

    removable_justifications, preserved, retract_edges = _unique_justifications_with_preserved(
        graph, commitment_id, config
    )

    retract_nodes = {commitment_id}
    retract_nodes.update(removable_justifications)
    if claim_id:
        retract_nodes.discard(claim_id)

    action = {
        "target_node_ids": [commitment_id],
        "predicted_delta": {"update_nodes": [], "update_edges": []},
    }
    dep_nodes = dependents(graph, action, config)
    regenerate_nodes: set[str] = set()
    for node_id in dep_nodes:
        if node_id in retract_nodes or node_id in preserved:
            continue
        node = graph.nodes.get(node_id, {})
        if node.get("type") in {"DraftFragment", "Hypothesis"}:
            regenerate_nodes.add(node_id)

    return RepairResult(
        contested_claims=[claim_id] if claim_id else [],
        retract_nodes=sorted(node_id for node_id in retract_nodes if node_id),
        retract_edges=sorted(set(retract_edges)),
        regenerate_nodes=sorted(regenerate_nodes),
        preserved_nodes=sorted(preserved),
        reopen_claims=[claim_id] if claim_id else [],
    )


def apply_repair(graph_like: Any, repair_result: RepairResult, config) -> None:
    """Apply repair via append-only ledger/status helpers without deletion."""
    for claim_id in repair_result.contested_claims:
        graph_like.mark_contested(claim_id, reason="rollback_consistency_violation")

    for node_id in repair_result.retract_nodes:
        graph_like.retract_node(node_id, reason="rollback_reduced_repair")

    for src_id, dst_id, relation in repair_result.retract_edges:
        graph_like.retract_edge(src_id, dst_id, relation, reason="rollback_reduced_repair")

    for node_id in repair_result.regenerate_nodes:
        if str(config.wm_regenerate_dependents) == "drop":
            graph_like.retract_node(node_id, reason="rollback_regenerate_drop")
        else:
            graph_like.mark_stale(node_id, reason="rollback_regenerate_stale")

    rollback_commitment_id = None
    for node_id in repair_result.retract_nodes:
        node = graph_like.get_node(node_id) if hasattr(graph_like, "get_node") else None
        if isinstance(node, dict) and node.get("type") == "Commitment":
            rollback_commitment_id = node_id
            break

    graph_like.rollback_record(
        commitment_id=rollback_commitment_id,
        payload=asdict(repair_result),
    )

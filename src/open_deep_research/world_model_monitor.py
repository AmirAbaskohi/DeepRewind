"""Phase 5 monitor for rollback trigger detection.

This module is read-only: it inspects current graph state and active
commitments, then reports which commitments are stranded.
"""

from __future__ import annotations

from typing import Any, Optional

from open_deep_research.world_model_scoring import compute_belief, source_reliability, take_snapshot


def _edge_weight(edge: dict, config: Any) -> float:
    if edge.get("weight") is not None:
        try:
            return float(edge.get("weight"))
        except Exception:
            pass
    data = edge.get("data", {}) if isinstance(edge.get("data"), dict) else {}
    if data.get("weight") is not None:
        try:
            return float(data.get("weight"))
        except Exception:
            pass
    return float(config.wm_default_edge_weight)


def _status_of_node(node: dict) -> str:
    data = node.get("data", {}) if isinstance(node.get("data"), dict) else {}
    return str(data.get("status", "")).lower()


def _status_of_edge(edge: dict) -> str:
    data = edge.get("data", {}) if isinstance(edge.get("data"), dict) else {}
    return str(data.get("status", "")).lower()


def _is_active_node(graph, node_id: str) -> bool:
    node = graph.nodes.get(node_id, {})
    return _status_of_node(node) != "retracted"


def _is_active_edge(graph, edge: dict) -> bool:
    if _status_of_edge(edge) == "retracted":
        return False
    src_id = edge.get("src_id")
    dst_id = edge.get("dst_id")
    if not src_id or not dst_id:
        return False
    return _is_active_node(graph, src_id) and _is_active_node(graph, dst_id)


def _source_ids_for_evidence(graph, evidence_id: str) -> list[str]:
    out: list[str] = []
    for edge in graph.edges:
        if not _is_active_edge(graph, edge):
            continue
        if edge.get("src_id") != evidence_id:
            continue
        if edge.get("relation") not in {"derived_from", "cites", "from"}:
            continue
        source_id = edge.get("dst_id")
        if not source_id or source_id not in graph.nodes:
            continue
        node = graph.nodes.get(source_id, {})
        if node.get("type") == "Source":
            out.append(source_id)
    return out


def _evidence_reliability(graph, evidence_id: str, config: Any) -> float:
    source_ids = _source_ids_for_evidence(graph, evidence_id)
    if not source_ids:
        return float(config.wm_default_reliability)
    return max(source_reliability(graph, source_id, config) for source_id in source_ids)


def _is_new_since_step(edge: dict, step: Optional[int]) -> bool:
    if step is None:
        return True
    data = edge.get("data", {}) if isinstance(edge.get("data"), dict) else {}
    edge_step = data.get("iteration")
    try:
        return int(edge_step) > int(step)
    except Exception:
        # When the edge does not carry iteration metadata, conservatively keep it visible.
        return True


def consistency_violated(graph_like: Any, commitment_record: dict, config: Any) -> tuple[bool, dict]:
    """Check whether a commitment trigger is fired and consistency is broken."""
    graph = take_snapshot(graph_like)
    claim_id = str(commitment_record.get("claim_id") or "")
    theta = commitment_record.get("theta") if isinstance(commitment_record.get("theta"), dict) else {}
    rho_star = float(theta.get("rho_star", config.wm_theta_rho_star))
    w_star = float(theta.get("w_star", config.wm_theta_w_star))
    step = commitment_record.get("step")

    beta_now = compute_belief(graph, claim_id, config) if claim_id else 0.0
    offending_evidence_ids: list[str] = []

    if claim_id and claim_id in graph.nodes and _is_active_node(graph, claim_id):
        for edge in graph.edges:
            if not _is_active_edge(graph, edge):
                continue
            if edge.get("relation") != "contradicts":
                continue
            if edge.get("dst_id") != claim_id and edge.get("src_id") != claim_id:
                continue
            if not _is_new_since_step(edge, step):
                continue

            other_id = edge.get("dst_id") if edge.get("src_id") == claim_id else edge.get("src_id")
            if not other_id:
                continue
            other_node = graph.nodes.get(other_id, {})
            if other_node.get("type") != "Evidence":
                continue

            rho = _evidence_reliability(graph, other_id, config)
            weight = _edge_weight(edge, config)
            if rho >= rho_star and weight >= w_star:
                offending_evidence_ids.append(other_id)

    qualifying = bool(offending_evidence_ids)
    violated = qualifying and (beta_now < float(config.wm_beta_star))
    detail = {
        "beta_now": beta_now,
        "beta_before": commitment_record.get("beta_at_commit"),
        "offending_evidence_ids": sorted(set(offending_evidence_ids)),
        "qualifying": qualifying,
    }
    return violated, detail


def monitor(graph_like: Any, active_commitments: dict[str, dict], config: Any) -> list[dict]:
    """Check all active commitments and collect the fired subset F."""
    fired: list[dict] = []
    for _, record in (active_commitments or {}).items():
        violated, detail = consistency_violated(graph_like, record, config)
        if violated:
            fired.append({"commitment_record": dict(record), "detail": detail})
    return fired

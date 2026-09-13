"""Grounded world-model scoring over read-only epistemic graph snapshots.

This module computes phase-4 structural quantities without mutating the live graph.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Optional

PREDICTABLE_NODE_TYPES = {
    "Claim",
    "Hypothesis",
    "Assumption",
    "Commitment",
    "DraftFragment",
    "PlanStep",
}

_DEPENDENCY_RELATIONS = {"depends_on", "used_in", "derived_from", "locks_in", "compresses"}


@dataclass
class GraphSnapshot:
    """Read-only style graph snapshot used for pure scoring functions."""

    nodes: dict[str, dict]
    edges: list[dict]
    lockin_terms: dict[str, float]


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _node_status(node: dict) -> str:
    data = node.get("data", {}) if isinstance(node.get("data"), dict) else {}
    return str(data.get("status", "")).lower()


def _edge_status(edge: dict) -> str:
    data = edge.get("data", {}) if isinstance(edge.get("data"), dict) else {}
    return str(data.get("status", "")).lower()


def _is_active_node(graph: GraphSnapshot, node_id: str) -> bool:
    node = graph.nodes.get(node_id, {})
    return _node_status(node) != "retracted"


def _is_active_edge(graph: GraphSnapshot, edge: dict) -> bool:
    if _edge_status(edge) == "retracted":
        return False
    src_id = edge.get("src_id")
    dst_id = edge.get("dst_id")
    if not src_id or not dst_id:
        return False
    return _is_active_node(graph, src_id) and _is_active_node(graph, dst_id)


def _edge_weight(edge: dict, config) -> float:
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


def _iter_edges_for_node(graph: GraphSnapshot, node_id: str) -> list[dict]:
    if not _is_active_node(graph, node_id):
        return []
    return [
        edge
        for edge in graph.edges
        if (edge.get("src_id") == node_id or edge.get("dst_id") == node_id) and _is_active_edge(graph, edge)
    ]


def _source_ids_for_evidence(graph: GraphSnapshot, evidence_id: str) -> list[str]:
    source_ids: list[str] = []
    for edge in graph.edges:
        if not _is_active_edge(graph, edge):
            continue
        if edge.get("src_id") != evidence_id:
            continue
        if edge.get("relation") not in {"derived_from", "cites", "from"}:
            continue
        target_id = edge.get("dst_id")
        if not target_id:
            continue
        target = graph.nodes.get(target_id, {})
        if target.get("type") == "Source":
            source_ids.append(target_id)
    return source_ids


def source_reliability(graph: GraphSnapshot, source_id: str, config) -> float:
    """Return clamped source reliability rho(s)."""
    source = graph.nodes.get(source_id, {})
    data = source.get("data", {}) if isinstance(source.get("data"), dict) else {}
    value = data.get("reliability", config.wm_default_reliability)
    try:
        return _clamp_01(float(value))
    except Exception:
        return _clamp_01(float(config.wm_default_reliability))


def _evidence_source_reliability(graph: GraphSnapshot, evidence_id: str, config) -> float:
    source_ids = _source_ids_for_evidence(graph, evidence_id)
    if not source_ids:
        return _clamp_01(float(config.wm_default_reliability))
    # Take max reliability among linked sources to avoid over-penalizing multi-source evidence.
    return max(source_reliability(graph, source_id, config) for source_id in source_ids)


def compute_belief(graph: GraphSnapshot, claim_id: str, config) -> float:
    """Compute claim belief beta(c) via logistic aggregation (Eq. app_belief)."""
    total = float(config.wm_beta0)
    for edge in _iter_edges_for_node(graph, claim_id):
        relation = edge.get("relation")
        if relation not in {"supports", "contradicts"}:
            continue
        if edge.get("src_id") == claim_id:
            other_id = edge.get("dst_id")
        else:
            other_id = edge.get("src_id")
        if not other_id:
            continue
        other_node = graph.nodes.get(other_id, {})
        if other_node.get("type") != "Evidence":
            continue
        alpha = 1.0 if relation == "supports" else -1.0
        total += alpha * _edge_weight(edge, config) * _evidence_source_reliability(graph, other_id, config)
    # Stable sigmoid.
    if total >= 0:
        z = math.exp(-total)
        return 1.0 / (1.0 + z)
    z = math.exp(total)
    return z / (1.0 + z)


def hypothesis_support(graph: GraphSnapshot, hyp_id: str, config) -> float:
    """Compute g(h) from supporting claims weighted by beta(c) (Eq. app_hyp)."""
    support = float(graph.lockin_terms.get(hyp_id, 0.0))
    for edge in graph.edges:
        if not _is_active_edge(graph, edge):
            continue
        if edge.get("relation") != "supports":
            continue
        if edge.get("dst_id") != hyp_id:
            continue
        claim_id = edge.get("src_id")
        if not claim_id:
            continue
        claim = graph.nodes.get(claim_id, {})
        if claim.get("type") != "Claim":
            continue
        support += _edge_weight(edge, config) * compute_belief(graph, claim_id, config)
    return support


def compute_plausibility(graph: GraphSnapshot, hyp_ids: list[str], config) -> dict[str, float]:
    """Compute hypothesis plausibility softmax (Eq. app_hyp)."""
    if not hyp_ids:
        return {}
    temperature = max(1e-6, float(config.wm_plausibility_temp))
    scores = {hyp_id: hypothesis_support(graph, hyp_id, config) / temperature for hyp_id in hyp_ids}
    max_score = max(scores.values())
    exp_scores = {hyp_id: math.exp(score - max_score) for hyp_id, score in scores.items()}
    normalizer = sum(exp_scores.values())
    if normalizer <= 0.0:
        uniform = 1.0 / float(len(hyp_ids))
        return {hyp_id: uniform for hyp_id in hyp_ids}
    return {hyp_id: exp_scores[hyp_id] / normalizer for hyp_id in hyp_ids}


def entropy(p: dict[str, float]) -> float:
    """Compute entropy H(p) with natural log."""
    total = 0.0
    for value in p.values():
        if value <= 0.0:
            continue
        total -= value * math.log(value)
    return total


def _resolve_hypothesis_target(graph: GraphSnapshot, action: dict, config) -> Optional[str]:
    target_ids = [str(node_id) for node_id in (action.get("target_node_ids") or [])]
    for node_id in target_ids:
        node = graph.nodes.get(node_id, {})
        if node.get("type") == "Hypothesis":
            return node_id

    claim_id = _resolve_claim_target(graph, action)
    if not claim_id:
        return None

    candidates: list[tuple[float, str]] = []
    for edge in graph.edges:
        if edge.get("relation") != "supports" or edge.get("src_id") != claim_id:
            continue
        dst_id = edge.get("dst_id")
        if not dst_id:
            continue
        dst = graph.nodes.get(dst_id, {})
        if dst.get("type") != "Hypothesis":
            continue
        candidates.append((_edge_weight(edge, config), dst_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_claim_target(graph: GraphSnapshot, action: dict) -> Optional[str]:
    target_ids = [str(node_id) for node_id in (action.get("target_node_ids") or [])]
    for node_id in target_ids:
        node = graph.nodes.get(node_id, {})
        if node.get("type") == "Claim":
            return node_id
    # Fallback from prediction metadata, if present.
    for node_delta in (action.get("predicted_delta", {}) or {}).get("update_nodes", []):
        if node_delta.get("type") == "Commitment":
            fields = node_delta.get("fields", {}) if isinstance(node_delta.get("fields"), dict) else {}
            claim_id = fields.get("target_claim_id")
            if claim_id and claim_id in graph.nodes:
                return claim_id
    return None


def take_snapshot(graph_like: Any) -> GraphSnapshot:
    """Create an immutable-by-convention snapshot from an epistemic graph reader."""
    nodes = copy.deepcopy(graph_like.get_nodes()) if hasattr(graph_like, "get_nodes") else {}
    edges = copy.deepcopy(graph_like.get_edges()) if hasattr(graph_like, "get_edges") else []
    return GraphSnapshot(nodes=nodes, edges=edges, lockin_terms={})


def apply_commit_to_snapshot(graph: GraphSnapshot, action: dict, config) -> GraphSnapshot:
    """Return post-commit simulated snapshot without mutating the input graph."""
    snap = GraphSnapshot(
        nodes=copy.deepcopy(graph.nodes),
        edges=copy.deepcopy(graph.edges),
        lockin_terms=copy.deepcopy(graph.lockin_terms),
    )

    predicted_delta = (action or {}).get("predicted_delta") or {}
    for node in predicted_delta.get("update_nodes", []) or []:
        node_id = node.get("id")
        node_type = node.get("type")
        op = node.get("op")
        if node_type not in PREDICTABLE_NODE_TYPES:
            continue
        if op == "update" and node_id in snap.nodes:
            fields = node.get("fields", {}) if isinstance(node.get("fields"), dict) else {}
            snap.nodes[node_id].setdefault("data", {})
            if isinstance(snap.nodes[node_id]["data"], dict):
                snap.nodes[node_id]["data"].update(fields)
        elif op == "create":
            # Created nodes without ids remain unbound in this phase; skip structural insert.
            continue

    for edge in predicted_delta.get("update_edges", []) or []:
        op = edge.get("op")
        src_id = edge.get("src_id")
        dst_id = edge.get("dst_id")
        relation = edge.get("relation")
        if not src_id or not dst_id or not relation:
            continue
        if src_id not in snap.nodes or dst_id not in snap.nodes:
            continue

        if op == "add":
            snap.edges.append(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "relation": relation,
                    "weight": edge.get("weight"),
                    "data": {},
                }
            )
        elif op == "remove":
            snap.edges = [
                e
                for e in snap.edges
                if not (
                    e.get("src_id") == src_id
                    and e.get("dst_id") == dst_id
                    and e.get("relation") == relation
                )
            ]

    hyp_id = _resolve_hypothesis_target(snap, action or {}, config)
    if hyp_id:
        snap.lockin_terms[hyp_id] = float(snap.lockin_terms.get(hyp_id, 0.0)) + float(
            config.wm_commit_lockin_strength
        )

    return snap


def _active_hypotheses(graph: GraphSnapshot, action: dict) -> list[str]:
    target_ids = [
        str(node_id)
        for node_id in (action.get("target_node_ids") or [])
        if node_id in graph.nodes and _is_active_node(graph, str(node_id))
    ]
    start: set[str] = {node_id for node_id in target_ids if node_id in graph.nodes}
    frontier = list(start)
    visited = set(start)
    hops = 2
    while frontier and hops > 0:
        next_frontier: list[str] = []
        for node_id in frontier:
            for edge in _iter_edges_for_node(graph, node_id):
                other = edge.get("dst_id") if edge.get("src_id") == node_id else edge.get("src_id")
                if not other or other in visited:
                    continue
                visited.add(other)
                next_frontier.append(other)
        frontier = next_frontier
        hops -= 1

    return [
        node_id
        for node_id in sorted(visited)
        if graph.nodes.get(node_id, {}).get("type") == "Hypothesis" and _is_active_node(graph, node_id)
    ]


def compute_kappa(graph: GraphSnapshot, action: dict, config) -> float:
    """Compute grounded commitment effect kappa (Eq. app_kappa)."""
    hyp_ids = _active_hypotheses(graph, action)
    k = len(hyp_ids)
    if k < 2:
        return 0.0

    p_t = compute_plausibility(graph, hyp_ids, config)
    post = apply_commit_to_snapshot(graph, action, config)
    p_t1 = compute_plausibility(post, hyp_ids, config)
    denom = math.log(float(k)) if k > 1 else 0.0
    if denom <= 0:
        return 0.0
    value = (entropy(p_t) - entropy(p_t1)) / denom
    return _clamp_01(value)


def _claim_stance_masses(graph: GraphSnapshot, claim_id: str, config) -> tuple[float, float]:
    supp = 0.0
    con = 0.0
    for edge in _iter_edges_for_node(graph, claim_id):
        relation = edge.get("relation")
        if relation not in {"supports", "contradicts"}:
            continue
        if edge.get("src_id") == claim_id:
            other_id = edge.get("dst_id")
        else:
            other_id = edge.get("src_id")
        if not other_id:
            continue
        if graph.nodes.get(other_id, {}).get("type") != "Evidence":
            continue
        mass = _edge_weight(edge, config) * _evidence_source_reliability(graph, other_id, config)
        if relation == "supports":
            supp += mass
        else:
            con += mass
    return supp, con


def compute_lambda(graph: GraphSnapshot, action: dict, config) -> float:
    """Compute grounded information-loss approximation lambda (Eq. app_lambda approx)."""
    claim_id = _resolve_claim_target(graph, action)
    if not claim_id:
        return 0.0
    supp, con = _claim_stance_masses(graph, claim_id, config)
    denom = supp + con
    if denom <= 0:
        return 0.0
    return _clamp_01(con / denom)


def dependents(graph: GraphSnapshot, action: dict, config) -> set[str]:
    """Return transitive predictable dependents D(DeltaV_t)."""
    _ = config
    seeds = set(str(node_id) for node_id in (action.get("target_node_ids") or []) if node_id in graph.nodes)
    seeds = {node_id for node_id in seeds if _is_active_node(graph, node_id)}
    for node in (action.get("predicted_delta", {}) or {}).get("update_nodes", []) or []:
        node_id = node.get("id")
        if node_id and node_id in graph.nodes and _is_active_node(graph, node_id):
            seeds.add(node_id)

    queue = list(seeds)
    visited = set(seeds)
    result: set[str] = set()
    while queue:
        node_id = queue.pop(0)
        node = graph.nodes.get(node_id, {})
        if node.get("type") in PREDICTABLE_NODE_TYPES:
            result.add(node_id)
        for edge in graph.edges:
            if not _is_active_edge(graph, edge):
                continue
            if edge.get("relation") not in _DEPENDENCY_RELATIONS:
                continue
            if edge.get("src_id") != node_id:
                continue
            nxt = edge.get("dst_id")
            if not nxt or nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return result


def compute_gamma(graph: GraphSnapshot, action: dict, config) -> float:
    """Compute grounded normalized recovery cost gamma (Eq. app_gamma)."""
    d = dependents(graph, action, config)
    budget = max(1e-6, float(config.wm_recovery_budget))
    per_node = float(config.wm_cost_retract) + float(config.wm_cost_regen)
    value = (1.0 / budget) * (float(len(d)) * per_node)
    return _clamp_01(value)


def build_trigger(graph: GraphSnapshot, action: dict, predicted_theta: Optional[dict], config) -> Optional[dict]:
    """Resolve trigger theta from predicted value or default fallback."""
    if isinstance(predicted_theta, dict):
        claim_id = predicted_theta.get("claim_id")
        if claim_id and claim_id in graph.nodes and graph.nodes.get(claim_id, {}).get("type") == "Claim":
            return {
                "claim_id": claim_id,
                "rho_star": _clamp_01(predicted_theta.get("rho_star", config.wm_theta_rho_star)),
                "w_star": _clamp_01(predicted_theta.get("w_star", config.wm_theta_w_star)),
            }

    claim_id = _resolve_claim_target(graph, action)
    if not claim_id:
        return None
    return {
        "claim_id": claim_id,
        "rho_star": _clamp_01(float(config.wm_theta_rho_star)),
        "w_star": _clamp_01(float(config.wm_theta_w_star)),
    }


def compute_tc(graph: GraphSnapshot, action: dict, predicted_tc: Optional[float], config) -> float:
    """Compute trigger coverage TC.

    Structural heuristic: count active PlanStep probes linked to the committed
    claim within local structure. More probes imply better chance a future
    contradiction will be detected.
    """
    source = str(config.wm_tc_source)
    if source == "constant":
        return _clamp_01(float(config.wm_default_tc))
    if source == "llm":
        if predicted_tc is None:
            return _clamp_01(float(config.wm_default_tc))
        return _clamp_01(float(predicted_tc))

    claim_id = _resolve_claim_target(graph, action)
    if not claim_id:
        return _clamp_01(float(config.wm_default_tc))

    n_probes = 0
    for edge in _iter_edges_for_node(graph, claim_id):
        other = edge.get("dst_id") if edge.get("src_id") == claim_id else edge.get("src_id")
        if not other:
            continue
        node = graph.nodes.get(other, {})
        if node.get("type") != "PlanStep":
            continue
        data = node.get("data", {}) if isinstance(node.get("data"), dict) else {}
        status = str(data.get("status", "")).lower()
        if status in {"open", "pending", "active", ""}:
            n_probes += 1

    if n_probes <= 0:
        return _clamp_01(float(config.wm_default_tc))
    return _clamp_01(min(1.0, float(n_probes) * float(config.wm_tc_per_probe)))


def compute_irr(kappa: float, lambda_: float, gamma: float, config) -> float:
    """Compute irreversibility risk IRR (Eq. app_irr)."""
    irr = (
        float(config.wm_alpha_kappa) * float(kappa)
        + float(config.wm_alpha_lambda) * float(lambda_)
        + float(config.wm_alpha_gamma) * float(gamma)
    )
    return _clamp_01(irr)


def reversibility_class(irr: float, config) -> str:
    """Map IRR to reversibility class (Eq. app_rev)."""
    if irr < float(config.wm_tau1):
        return "rev"
    if irr < float(config.wm_tau2):
        return "partial"
    return "irrev"


def is_contested(graph: GraphSnapshot, action: dict, config) -> bool:
    """Contested test chi_t from grounded belief and contradiction mass."""
    claim_id = _resolve_claim_target(graph, action)
    if not claim_id:
        return False
    beta = compute_belief(graph, claim_id, config)
    supp, con = _claim_stance_masses(graph, claim_id, config)
    denom = supp + con
    con_norm = (con / denom) if denom > 0 else 0.0
    return abs(beta - 0.5) <= float(config.wm_contested_band) and con_norm >= float(
        config.wm_contested_con_min
    )


def compute_value(graph: GraphSnapshot, action: dict, config) -> float:
    """Compute value term V(a) per configured source."""
    source = str(config.wm_value_source)
    if source == "zero":
        return 0.0

    if source == "plausibility":
        hyp_id = _resolve_hypothesis_target(graph, action, config)
        if not hyp_id:
            return 0.0
        hyp_ids = _active_hypotheses(graph, action)
        plaus = compute_plausibility(graph, hyp_ids, config)
        return float(plaus.get(hyp_id, 0.0))

    return 0.0


def compute_utility(value: float, irr: float, tc: float, config) -> float:
    """Compute utility U = V - eta * IRR * (1 - TC) (Eq. app_utility)."""
    return float(value) - float(config.wm_eta) * float(irr) * (1.0 - float(tc))


def decide(irr: float, tc: float, utility: float, contested: bool, config) -> str:
    """Binary collapse of policy classes into commit/not_commit."""
    if contested:
        return "not_commit"

    gate_mode = str(config.wm_gate_mode)
    if gate_mode == "threshold":
        value = float(irr) * (1.0 - float(tc))
        return "commit" if value <= float(config.wm_tau_commit) else "not_commit"

    if gate_mode == "utility":
        return "commit" if float(utility) >= float(config.wm_u_commit_threshold) else "not_commit"

    return "commit"

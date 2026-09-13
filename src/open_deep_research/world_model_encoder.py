"""State encoder for world-model commitment prediction.

This phase performs graph-context extraction and prompt rendering only.
No LLM calls, scoring, or policy decisions are implemented here.
"""

from __future__ import annotations

from collections import deque
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

OBSERVED_NODE_TYPES = {"Source", "Evidence"}

RELATION_PHRASES = {
    "supports": ("supports", "supported by"),
    "contradicts": ("contradicts", "contradicted by"),
    "depends_on": ("depends on", "depended on by"),
    "compresses": ("compresses", "compressed from"),
    "used_in": ("used in", "uses"),
    "derived_from": ("derived from", "basis for"),
    "cites": ("cites", "cited by"),
    "revises": ("revises", "revised by"),
    "invalidates": ("invalidates", "invalidated by"),
}


@dataclass
class EncodedState:
    prompt: str
    question: Optional[str]
    action: dict
    included_node_ids: list[str]
    predictable_node_ids: list[str]
    observed_node_ids: list[str]
    subgraph_edges: list[dict]
    alias_to_id: dict[str, str]
    id_to_alias: dict[str, str]


def _short(text: Any, limit: int = 140) -> str:
    value = str(text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def _humanize_relation(relation: str) -> str:
    return str(relation or "related_to").replace("_", " ")


def _relation_phrase(edge: dict, from_node_id: str) -> str:
    relation = edge.get("relation", "")
    mapped = RELATION_PHRASES.get(relation)
    if mapped:
        if edge.get("src_id") == from_node_id:
            return mapped[0]
        return mapped[1]
    return _humanize_relation(relation)


def _extract_subgraph_with_distances(
    graph: Any,
    target_node_ids: list[str],
    khop: int,
    max_context_nodes: int = 40,
) -> tuple[list[str], list[dict], dict[str, int]]:
    """Extract an undirected k-hop neighborhood around target nodes.

    Returns:
        node_ids: Included node ids.
        edges: Subgraph edges with both endpoints in node_ids.
        distances: Hop distance from the nearest target node.
    """
    nodes = graph.get_nodes() if hasattr(graph, "get_nodes") else {}
    if not nodes:
        return [], [], {}

    seeds = [node_id for node_id in (target_node_ids or []) if node_id in nodes]
    if not seeds:
        return [], [], {}

    khop = max(0, int(khop))
    max_context_nodes = max(1, int(max_context_nodes))

    visited: set[str] = set()
    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for node_id in seeds:
        if node_id in visited:
            continue
        visited.add(node_id)
        distances[node_id] = 0
        queue.append((node_id, 0))

    while queue and len(visited) < max_context_nodes:
        current_id, depth = queue.popleft()
        if depth >= khop:
            continue
        neighbors = graph.neighbors(current_id) if hasattr(graph, "neighbors") else []
        for _edge, neighbor_id in neighbors:
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            distances[neighbor_id] = depth + 1
            queue.append((neighbor_id, depth + 1))
            if len(visited) >= max_context_nodes:
                break

    ordered_ids = sorted(visited, key=lambda nid: (distances.get(nid, 10**9), nid))

    all_edges = graph.get_edges() if hasattr(graph, "get_edges") else []
    edge_list = [
        {
            "src_id": edge.get("src_id"),
            "dst_id": edge.get("dst_id"),
            "relation": edge.get("relation"),
            "weight": edge.get("weight"),
            "data": edge.get("data", {}),
        }
        for edge in all_edges
        if edge.get("src_id") in visited and edge.get("dst_id") in visited
    ]
    edge_list.sort(key=lambda e: (str(e.get("src_id")), str(e.get("dst_id")), str(e.get("relation"))))

    return ordered_ids, edge_list, distances


def extract_subgraph(
    graph: Any,
    target_node_ids: list[str],
    khop: int,
    max_context_nodes: int = 40,
) -> tuple[list[str], list[dict]]:
    """Extract an undirected k-hop neighborhood around target nodes.

    Returns:
        node_ids: Included node ids.
        edges: Subgraph edges with both endpoints in node_ids.
    """
    node_ids, edges, _distances = _extract_subgraph_with_distances(
        graph=graph,
        target_node_ids=target_node_ids,
        khop=khop,
        max_context_nodes=max_context_nodes,
    )
    return node_ids, edges


def _node_alias(node: dict, counters: dict[str, int]) -> str:
    node_type = node.get("type", "Unknown")
    prefix_by_type = {
        "Claim": "C",
        "Hypothesis": "H",
        "Assumption": "A",
        "Commitment": "K",
        "DraftFragment": "D",
        "PlanStep": "P",
        "Source": "S",
        "Evidence": "E",
    }
    prefix = prefix_by_type.get(node_type, "N")
    counters[prefix] = counters.get(prefix, 0) + 1
    return f"{prefix}{counters[prefix]}"


def _summarize_data(data: dict[str, Any], limit: int = 4) -> str:
    if not isinstance(data, dict) or not data:
        return ""
    keys = [k for k in data.keys() if data.get(k) is not None]
    keys.sort()
    parts = [f"{k}={_short(data[k], 40)}" for k in keys[:limit]]
    return ", ".join(parts)


def encode_state(
    graph: Any,
    action: dict,
    question: Optional[str],
    khop: int,
    encode_message_passing: bool,
    max_context_nodes: int,
) -> EncodedState:
    """Encode local epistemic context for a commitment action.

    The output is an incident-style prompt and structured context object.
    """
    action = action or {}
    target_node_ids = [str(node_id) for node_id in (action.get("target_node_ids") or [])]
    action_obj = {
        "action_id": str(action.get("action_id") or ""),
        "action_type": str(action.get("action_type") or "commitment"),
        "target_node_ids": target_node_ids,
    }

    included_node_ids, subgraph_edges = extract_subgraph(
        graph=graph,
        target_node_ids=target_node_ids,
        khop=khop,
        max_context_nodes=max_context_nodes,
    )

    nodes = graph.get_nodes() if hasattr(graph, "get_nodes") else {}
    included_nodes = [nodes[node_id] for node_id in included_node_ids if node_id in nodes]

    predictable_node_ids = [
        node["id"] for node in included_nodes if node.get("type") in PREDICTABLE_NODE_TYPES
    ]
    observed_node_ids = [
        node["id"] for node in included_nodes if node.get("type") in OBSERVED_NODE_TYPES
    ]

    counters: dict[str, int] = {}
    alias_to_id: dict[str, str] = {}
    id_to_alias: dict[str, str] = {}
    for node in included_nodes:
        alias = _node_alias(node, counters)
        alias_to_id[alias] = node["id"]
        id_to_alias[node["id"]] = alias

    def render_node(node: dict, include_neighbors: bool) -> str:
        alias = id_to_alias.get(node["id"], node["id"])
        label = node.get("label") or node.get("text") or node.get("id")
        data_blob = _summarize_data(node.get("data", {}))
        line = f"- {alias} [{node.get('type', 'Unknown')}]: {_short(label, 180)}"
        if data_blob:
            line += f" | data: {data_blob}"

        if include_neighbors and hasattr(graph, "neighbors"):
            neighbors = graph.neighbors(node["id"])
            snippets: list[str] = []
            for edge, neighbor_id in neighbors:
                if neighbor_id not in id_to_alias:
                    continue
                phrase = _relation_phrase(edge, node["id"])
                neighbor_alias = id_to_alias[neighbor_id]
                neighbor = nodes.get(neighbor_id, {})
                neighbor_label = neighbor.get("label") or neighbor.get("text") or neighbor_id
                snippets.append(
                    f"{phrase} {neighbor_alias}({_short(neighbor_label, 40)})"
                )
                if len(snippets) >= 3:
                    break
            if snippets:
                line += " | incident: " + "; ".join(snippets)
        return line

    predictable_lines = [
        render_node(node, encode_message_passing)
        for node in included_nodes
        if node.get("id") in predictable_node_ids
    ]
    observed_lines = [
        render_node(node, encode_message_passing)
        for node in included_nodes
        if node.get("id") in observed_node_ids
    ]

    edge_lines: list[str] = []
    for edge in subgraph_edges:
        src_alias = id_to_alias.get(edge.get("src_id"), str(edge.get("src_id")))
        dst_alias = id_to_alias.get(edge.get("dst_id"), str(edge.get("dst_id")))
        relation = edge.get("relation") or "related_to"
        relation_text = RELATION_PHRASES.get(relation, (_humanize_relation(relation), ""))[0]
        weight = edge.get("weight")
        weight_suffix = f" [weight={weight}]" if weight is not None else ""
        edge_lines.append(f"- {src_alias} {relation_text} {dst_alias}{weight_suffix}")

    prompt_sections = [
        "Incident Register: Local Epistemic Context",
        f"Question: {question or ''}",
        "",
        "Action",
        f"- action_id: {action_obj['action_id']}",
        f"- action_type: {action_obj['action_type']}",
        f"- targets: {', '.join(id_to_alias.get(t, t) for t in target_node_ids) if target_node_ids else '(none)'}",
        "",
        "Predictable Nodes (model may propose updates only for these)",
    ]
    prompt_sections.extend(predictable_lines or ["- (none)"])
    prompt_sections.extend([
        "",
        "Observed-Only Nodes (internet-grounded context; do NOT predict updates for these)",
    ])
    prompt_sections.extend(observed_lines or ["- (none)"])
    prompt_sections.extend([
        "",
        "Subgraph Relations",
    ])
    prompt_sections.extend(edge_lines or ["- (none)"])

    prompt = "\n".join(prompt_sections)

    return EncodedState(
        prompt=prompt,
        question=question,
        action=action_obj,
        included_node_ids=included_node_ids,
        predictable_node_ids=predictable_node_ids,
        observed_node_ids=observed_node_ids,
        subgraph_edges=subgraph_edges,
        alias_to_id=alias_to_id,
        id_to_alias=id_to_alias,
    )

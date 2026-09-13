"""Epistemic research-graph recording for the Deep Research agent.

Where :mod:`open_deep_research.research_logger` records *what the agent did over
time* (a state graph of actions), this module records *what the agent currently
knows, believes, assumes, cites, commits to, and writes* - an **epistemic
graph** of the agent's evolving research state.

The epistemic graph is made of typed nodes and typed edges:

Node types
    ``Source``        - a document/page the agent found or read.
    ``Evidence``      - a useful piece of information extracted from sources.
    ``Claim``         - an intermediate assertion the agent forms.
    ``Hypothesis``    - a working, falsifiable belief guiding the research.
    ``Assumption``    - something taken as given while reasoning.
    ``Commitment``    - a strengthened commitment to a claim or hypothesis.
    ``DraftFragment`` - a paragraph/section written into the answer.
    ``PlanStep``      - an action/plan step that produced a graph change.

Edge types
    ``supports``      - evidence/claim supports a claim/hypothesis.
    ``contradicts``   - evidence/claim contradicts a claim/hypothesis.
    ``depends_on``    - a claim/hypothesis depends on evidence/assumption/claim.
    ``compresses``    - many items are compressed/summarized into one.
    ``used_in``       - a claim/evidence item is used in a draft fragment.
    ``derived_from``  - evidence is derived from a source.
    ``cites``         - a node cites a source.
    ``revises``       - a newer node revises an older node.
    ``invalidates``   - a newer node invalidates an older node.

Nodes and edges are streamed to a JSON-Lines (``.jsonl``) file - one JSON
object per line - written to ``epistemic_graphs/`` by default (kept separate
from ``research_logs/``). Because every node and edge is appended as its own
event, the graph is append-friendly and preserves history: beliefs are never
silently overwritten. When later information changes an earlier belief we add a
``revises``/``invalidates`` edge to a *new* node instead of mutating the old
one.

The recorder is intentionally dependency-free and defensive: any failure while
recording is swallowed so instrumentation can never break a research run.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Schema version written into every graph file. Bump if the event format
# changes in a way the visualizer needs to know about.
EPISTEMIC_SCHEMA_VERSION = 1

# The canonical set of node types the epistemic graph understands.
NODE_TYPES = (
    "Source",
    "Evidence",
    "Claim",
    "Hypothesis",
    "Assumption",
    "Commitment",
    "DraftFragment",
    "PlanStep",
)

# The canonical set of edge types the epistemic graph understands.
EDGE_TYPES = (
    "supports",
    "contradicts",
    "locks_in",
    "depends_on",
    "compresses",
    "used_in",
    "derived_from",
    "cites",
    "revises",
    "invalidates",
)

# Short, human-readable id prefixes per node type (e.g. ``src1``, ``clm2``).
_NODE_PREFIXES = {
    "Source": "src",
    "Evidence": "ev",
    "Claim": "clm",
    "Hypothesis": "hyp",
    "Assumption": "asm",
    "Commitment": "cmt",
    "DraftFragment": "dft",
    "PlanStep": "plan",
}

# Registry of active epistemic graphs keyed by a stable run identifier so that
# the many LangGraph nodes/subgraphs making up a single research run all write
# to the same graph file.
_GRAPHS: Dict[str, "EpistemicGraph"] = {}
_GRAPHS_LOCK = threading.Lock()


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: str) -> str:
    """Make a string safe to embed in a file name."""
    keep = "-_."
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in value)
    return cleaned[:80] or "run"


def _truncate(value: Any, limit: int = 4000) -> Any:
    """Truncate long strings so the graph file stays readable and bounded."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [truncated {len(value) - limit} chars]"
    return value


def epistemic_graph_enabled(config: Optional[dict]) -> bool:
    """Return whether epistemic-graph recording should be active for this run.

    Controlled by the ``enable_epistemic_graph`` configurable field (or the
    ``ENABLE_EPISTEMIC_GRAPH`` environment variable). Defaults to enabled so
    graphs are captured out of the box.
    """
    env = os.environ.get("ENABLE_EPISTEMIC_GRAPH")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off", "")
    configurable = (config or {}).get("configurable", {}) if config else {}
    value = configurable.get("enable_epistemic_graph")
    if value is None:
        return True
    return bool(value)


def _graph_dir(config: Optional[dict]) -> str:
    """Resolve the directory where epistemic graph files are written."""
    env = os.environ.get("EPISTEMIC_GRAPH_DIR")
    if env:
        return env
    configurable = (config or {}).get("configurable", {}) if config else {}
    return configurable.get("epistemic_graph_dir") or "epistemic_graphs"


def _run_id_from_config(config: Optional[dict]) -> str:
    """Derive a stable run identifier from a LangGraph ``RunnableConfig``.

    Every node and subgraph in a single run shares the same ``thread_id`` (or
    ``run_id``), so we use that to group their nodes/edges into one graph. When
    no such identifier is available we fall back to a single process-level key.
    """
    if not config:
        return "default"
    configurable = config.get("configurable", {}) or {}
    metadata = config.get("metadata", {}) or {}
    candidates = [
        configurable.get("thread_id"),
        metadata.get("thread_id"),
        configurable.get("run_id"),
        metadata.get("run_id"),
        config.get("run_id"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return "default"


class _NullGraph:
    """No-op recorder used when epistemic-graph recording is disabled.

    It exposes the same surface as :class:`EpistemicGraph` so callers never need
    to branch on whether recording is active. Every method is a harmless no-op
    that returns ``None`` (or empty collections) so downstream code is safe.
    """

    enabled = False
    graph_path = None

    def set_question(self, *_args, **_kwargs) -> None:
        return None

    def add_node(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def add_source(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def add_evidence(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def add_claim(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def add_hypothesis(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def add_assumption(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def add_commitment(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def add_draft_fragment(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def add_plan_step(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def add_edge(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def supports(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def contradicts(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def depends_on(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def locks_in(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def compresses(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def used_in(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def derived_from(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def cites(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def revises(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def invalidates(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def record_topic_sources(self, *_args, **_kwargs) -> None:
        return None

    def get_topic_sources(self, *_args, **_kwargs) -> List[str]:
        return []

    def record_topic_evidence(self, *_args, **_kwargs) -> None:
        return None

    def get_topic_evidence(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def mark_contested(self, *_args, **_kwargs) -> None:
        return None

    def retract_node(self, *_args, **_kwargs) -> None:
        return None

    def retract_edge(self, *_args, **_kwargs) -> None:
        return None

    def mark_stale(self, *_args, **_kwargs) -> None:
        return None

    def rollback_record(self, *_args, **_kwargs) -> None:
        return None

    def record_claim(self, *_args, **_kwargs) -> None:
        return None

    def get_claim(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def record_hypothesis(self, *_args, **_kwargs) -> None:
        return None

    def get_hypothesis(self, *_args, **_kwargs) -> Optional[str]:
        return None

    def all_claim_ids(self, *_args, **_kwargs) -> List[str]:
        return []

    def all_evidence_ids(self, *_args, **_kwargs) -> List[str]:
        return []

    def get_nodes(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def get_edges(self) -> List[Dict[str, Any]]:
        return []

    def get_node(self, _node_id: str) -> Optional[Dict[str, Any]]:
        return None

    def neighbors(self, _node_id: str) -> List[tuple[Dict[str, Any], str]]:
        return []

    def finish(self, *_args, **_kwargs) -> None:
        return None


class EpistemicGraph:
    """Accumulates and streams the epistemic graph for a single research run."""

    enabled = True

    def __init__(self, run_id: str, graph_dir: str):
        """Open the graph file and emit the ``run_start`` event."""
        self.run_id = run_id
        self._lock = threading.RLock()
        self._node_counters: Dict[str, int] = {}
        self._edge_counter = 0
        # In-memory mirror used by the world-model encoder. This does not alter
        # the append-only JSONL schema; it only adds read access to current run state.
        self._nodes_by_id: Dict[str, Dict[str, Any]] = {}
        self._edges: List[Dict[str, Any]] = []
        self._neighbors_by_id: Dict[str, List[tuple[Dict[str, Any], str]]] = {}
        # Cross-node working memory shared by every subgraph in this run. These
        # let a later step (e.g. compression, drafting) connect its nodes back
        # to nodes created earlier (e.g. the sources of a research topic).
        self._topic_sources: Dict[str, List[str]] = {}
        self._topic_evidence: Dict[str, str] = {}
        self._claims_by_text: Dict[str, str] = {}
        self._hypotheses_by_text: Dict[str, str] = {}
        self._question: Optional[str] = None
        self._finished = False

        os.makedirs(graph_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_name = f"epistemic_graph_{_sanitize(run_id)}_{stamp}.jsonl"
        self.graph_path = os.path.join(graph_dir, file_name)
        # Line-buffered append so nodes/edges are durable as soon as they occur.
        self._fh = open(self.graph_path, "a", encoding="utf-8")  # noqa: SIM115
        self._write({
            "event": "run_start",
            "schema_version": EPISTEMIC_SCHEMA_VERSION,
            "run_id": run_id,
            "ts": _now_iso(),
        })

    # -- internal helpers ---------------------------------------------------
    def _write(self, record: Dict[str, Any]) -> None:
        """Serialize a single event to the JSONL file (best effort)."""
        try:
            self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            # Never let recording failures propagate into the research run.
            pass

    def _next_node_id(self, node_type: str) -> str:
        prefix = _NODE_PREFIXES.get(node_type, "n")
        self._node_counters[prefix] = self._node_counters.get(prefix, 0) + 1
        return f"{prefix}{self._node_counters[prefix]}"

    def _next_edge_id(self) -> str:
        self._edge_counter += 1
        return f"edge{self._edge_counter}"

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate long string values inside a metadata payload."""
        cleaned: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if isinstance(value, str):
                cleaned[key] = _truncate(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    _truncate(v) if isinstance(v, str) else v for v in value
                ]
            else:
                cleaned[key] = value
        return cleaned

    # -- run metadata -------------------------------------------------------
    def set_question(self, question: Optional[str]) -> None:
        """Record the original user question once, on the first opportunity."""
        with self._lock:
            if self._question is None and question:
                self._question = question
                self._write({
                    "event": "question",
                    "run_id": self.run_id,
                    "ts": _now_iso(),
                    "question": _truncate(question, 8000),
                })

    # -- generic node/edge creation ----------------------------------------
    def add_node(
        self,
        node_type: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> Optional[str]:
        """Record a typed node and return its id.

        Args:
            node_type: One of :data:`NODE_TYPES`.
            text: The content/text of the node (the belief, claim, evidence,
                source title, draft fragment, ...).
            metadata: Arbitrary structured details (topic, url, score, ...).
            label: Optional short human-readable label; defaults to a trimmed
                version of ``text``.
        """
        with self._lock:
            node_id = self._next_node_id(node_type)
            record = {
                "event": "node",
                "run_id": self.run_id,
                "id": node_id,
                "type": node_type,
                "label": _truncate(label or text, 300),
                "text": _truncate(text),
                "metadata": self._clean_metadata(metadata or {}),
                "ts": _now_iso(),
            }
            self._write(record)
            self._nodes_by_id[node_id] = {
                "id": node_id,
                "type": node_type,
                "label": record["label"],
                "text": record["text"],
                "data": dict(record["metadata"]),
            }
            self._neighbors_by_id.setdefault(node_id, [])
            return node_id

    def add_edge(
        self,
        edge_type: str,
        source: Optional[str],
        target: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Record a typed directed edge from ``source`` to ``target``.

        Args:
            edge_type: One of :data:`EDGE_TYPES`.
            source: The id of the source node.
            target: The id of the target node.
            metadata: Arbitrary structured details (rationale, iteration, ...).
        """
        if not source or not target:
            return None
        with self._lock:
            edge_id = self._next_edge_id()
            record = {
                "event": "edge",
                "run_id": self.run_id,
                "id": edge_id,
                "type": edge_type,
                "source": source,
                "target": target,
                "metadata": self._clean_metadata(metadata or {}),
                "ts": _now_iso(),
            }
            self._write(record)
            edge = {
                "id": edge_id,
                "src_id": source,
                "dst_id": target,
                "relation": edge_type,
                "weight": (metadata or {}).get("weight"),
                "data": dict(record["metadata"]),
            }
            self._edges.append(edge)
            self._neighbors_by_id.setdefault(source, []).append((edge, target))
            self._neighbors_by_id.setdefault(target, []).append((edge, source))
            return edge_id

    # -- read accessors for world-model encoding ---------------------------
    def get_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of all current in-memory nodes by id."""
        with self._lock:
            return {node_id: dict(node) for node_id, node in self._nodes_by_id.items()}

    def get_edges(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of all current in-memory edges."""
        with self._lock:
            return [dict(edge) for edge in self._edges]

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return one node by id, if present."""
        with self._lock:
            node = self._nodes_by_id.get(node_id)
            return dict(node) if node else None

    def neighbors(self, node_id: str) -> List[tuple[Dict[str, Any], str]]:
        """Return incident edges and neighboring node ids for ``node_id``."""
        with self._lock:
            pairs = self._neighbors_by_id.get(node_id, [])
            return [(dict(edge), neighbor_id) for edge, neighbor_id in pairs]

    # -- typed node convenience helpers ------------------------------------
    def add_source(
        self,
        title: str,
        url: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a ``Source`` node for a document the agent found or read."""
        meta = dict(metadata or {})
        if url:
            meta.setdefault("url", url)
        text = title or url or "Untitled source"
        return self.add_node("Source", text, meta, label=text)

    def add_evidence(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        sources: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Create an ``Evidence`` node and link it to its ``sources``.

        Each provided source id gets a ``derived_from`` edge (Evidence ->
        Source) and a ``cites`` edge, so the provenance of extracted
        information is explicit in the graph.
        """
        node_id = self.add_node("Evidence", text, metadata)
        for source_id in sources or []:
            self.derived_from(node_id, source_id)
            self.cites(node_id, source_id)
        return node_id

    def add_claim(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a ``Claim`` node for an intermediate assertion."""
        node_id = self.add_node("Claim", text, metadata)
        if node_id:
            self.record_claim(text, node_id)
        return node_id

    def add_hypothesis(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a ``Hypothesis`` node for a working, falsifiable belief."""
        node_id = self.add_node("Hypothesis", text, metadata)
        if node_id:
            self.record_hypothesis(text, node_id)
        return node_id

    def add_assumption(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create an ``Assumption`` node."""
        return self.add_node("Assumption", text, metadata)

    def add_commitment(
        self,
        text: str,
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a ``Commitment`` node, optionally tied to a claim/hypothesis.

        A ``locks_in`` edge is drawn from the commitment to ``target`` (the
        claim or hypothesis being committed to).
        """
        node_id = self.add_node("Commitment", text, metadata)
        if target:
            self.locks_in(node_id, target)
        return node_id

    def add_draft_fragment(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> Optional[str]:
        """Create a ``DraftFragment`` node for written answer content."""
        return self.add_node("DraftFragment", text, metadata, label=label)

    def add_plan_step(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a ``PlanStep`` node describing an action that changed state."""
        return self.add_node("PlanStep", text, metadata)

    # -- typed edge convenience helpers ------------------------------------
    def supports(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that ``source`` supports ``target``."""
        return self.add_edge("supports", source, target, meta or None)

    def contradicts(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that ``source`` contradicts ``target``."""
        return self.add_edge("contradicts", source, target, meta or None)

    def depends_on(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that ``source`` depends on ``target``."""
        return self.add_edge("depends_on", source, target, meta or None)

    def locks_in(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that commitment ``source`` locks in ``target``."""
        return self.add_edge("locks_in", source, target, meta or None)

    def compresses(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that ``source`` compresses/summarizes ``target``."""
        return self.add_edge("compresses", source, target, meta or None)

    def used_in(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that ``source`` is used in ``target`` (e.g. a draft fragment)."""
        return self.add_edge("used_in", source, target, meta or None)

    def derived_from(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that ``source`` is derived from ``target`` (e.g. a source)."""
        return self.add_edge("derived_from", source, target, meta or None)

    def cites(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that ``source`` cites ``target``."""
        return self.add_edge("cites", source, target, meta or None)

    def revises(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that ``source`` revises the older ``target``."""
        return self.add_edge("revises", source, target, meta or None)

    def invalidates(self, source: Optional[str], target: Optional[str], **meta) -> Optional[str]:
        """Record that ``source`` invalidates the older ``target``."""
        return self.add_edge("invalidates", source, target, meta or None)

    # -- cross-node working memory -----------------------------------------
    def record_topic_sources(self, topic: str, source_ids: List[str]) -> None:
        """Remember the source nodes gathered for a research ``topic``."""
        if not topic:
            return
        with self._lock:
            bucket = self._topic_sources.setdefault(topic, [])
            for source_id in source_ids or []:
                if source_id and source_id not in bucket:
                    bucket.append(source_id)

    def get_topic_sources(self, topic: str) -> List[str]:
        """Return the source node ids gathered for a research ``topic``."""
        with self._lock:
            return list(self._topic_sources.get(topic, []))

    def record_topic_evidence(self, topic: str, evidence_id: Optional[str]) -> None:
        """Remember the evidence node synthesized for a research ``topic``."""
        if topic and evidence_id:
            with self._lock:
                self._topic_evidence[topic] = evidence_id

    def get_topic_evidence(self, topic: str) -> Optional[str]:
        """Return the evidence node id synthesized for a research ``topic``."""
        with self._lock:
            return self._topic_evidence.get(topic)

    def record_claim(self, text: str, node_id: Optional[str]) -> None:
        """Index a claim node by its text for later reconciliation lookups."""
        if text and node_id:
            with self._lock:
                self._claims_by_text[text.strip()] = node_id

    def get_claim(self, text: str) -> Optional[str]:
        """Return the claim node id previously recorded for ``text``."""
        with self._lock:
            return self._claims_by_text.get((text or "").strip())

    def record_hypothesis(self, text: str, node_id: Optional[str]) -> None:
        """Index a hypothesis node by its text for later lookups/revisions."""
        if text and node_id:
            with self._lock:
                self._hypotheses_by_text[text.strip()] = node_id

    def get_hypothesis(self, text: str) -> Optional[str]:
        """Return the hypothesis node id previously recorded for ``text``."""
        with self._lock:
            return self._hypotheses_by_text.get((text or "").strip())

    def all_claim_ids(self) -> List[str]:
        """Return the ids of every ``Claim`` node recorded so far."""
        with self._lock:
            return list(self._claims_by_text.values())

    def all_evidence_ids(self) -> List[str]:
        """Return the ids of every ``Evidence`` node recorded so far."""
        with self._lock:
            # Dedupe while preserving insertion order.
            seen: Dict[str, None] = {}
            for evidence_id in self._topic_evidence.values():
                if evidence_id:
                    seen.setdefault(evidence_id, None)
            return list(seen.keys())

    # -- append-only rollback and status helpers --------------------------
    def mark_contested(self, claim_id: Optional[str], reason: str) -> None:
        """Append a contested event and mark claim status as contested."""
        if not claim_id:
            return
        with self._lock:
            node = self._nodes_by_id.get(claim_id)
            if node is not None:
                node.setdefault("data", {})
                if isinstance(node["data"], dict):
                    node["data"]["status"] = "contested"
                    node["data"]["status_reason"] = _truncate(reason, 800)
            self._write({
                "event": "contested",
                "run_id": self.run_id,
                "claim_id": claim_id,
                "reason": _truncate(reason, 800),
                "ts": _now_iso(),
            })

    def retract_node(self, node_id: Optional[str], reason: str) -> None:
        """Append a node retraction event and set node status=retracted."""
        if not node_id:
            return
        with self._lock:
            node = self._nodes_by_id.get(node_id)
            if node is not None:
                node.setdefault("data", {})
                if isinstance(node["data"], dict):
                    node["data"]["status"] = "retracted"
                    node["data"]["status_reason"] = _truncate(reason, 800)
            self._write({
                "event": "retraction",
                "run_id": self.run_id,
                "kind": "node",
                "node_id": node_id,
                "reason": _truncate(reason, 800),
                "ts": _now_iso(),
            })

    def retract_edge(self, src: Optional[str], dst: Optional[str], relation: str, reason: str) -> None:
        """Append edge retraction events and set edge status=retracted."""
        if not src or not dst or not relation:
            return
        with self._lock:
            for edge in self._edges:
                if (
                    edge.get("src_id") == src
                    and edge.get("dst_id") == dst
                    and edge.get("relation") == relation
                ):
                    edge.setdefault("data", {})
                    if isinstance(edge["data"], dict):
                        edge["data"]["status"] = "retracted"
                        edge["data"]["status_reason"] = _truncate(reason, 800)
                    self._write({
                        "event": "retraction",
                        "run_id": self.run_id,
                        "kind": "edge",
                        "src_id": src,
                        "dst_id": dst,
                        "relation": relation,
                        "edge_id": edge.get("id"),
                        "reason": _truncate(reason, 800),
                        "ts": _now_iso(),
                    })

    def mark_stale(self, node_id: Optional[str], reason: str) -> None:
        """Append a stale event and set node status=stale."""
        if not node_id:
            return
        with self._lock:
            node = self._nodes_by_id.get(node_id)
            if node is not None:
                node.setdefault("data", {})
                if isinstance(node["data"], dict):
                    node["data"]["status"] = "stale"
                    node["data"]["status_reason"] = _truncate(reason, 800)
            self._write({
                "event": "stale",
                "run_id": self.run_id,
                "node_id": node_id,
                "reason": _truncate(reason, 800),
                "ts": _now_iso(),
            })

    def rollback_record(self, commitment_id: Optional[str], payload: Optional[Dict[str, Any]]) -> None:
        """Append rollback summary event for auditability."""
        self._write({
            "event": "rollback",
            "run_id": self.run_id,
            "commitment_id": commitment_id,
            "payload": self._clean_metadata(payload or {}),
            "ts": _now_iso(),
        })

    # -- termination --------------------------------------------------------
    def finish(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit the terminal ``run_end`` event and close the file."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._write({
                "event": "run_end",
                "run_id": self.run_id,
                "ts": _now_iso(),
                "metadata": self._clean_metadata(data or {}),
            })
            try:
                self._fh.close()
            except Exception:
                pass


def get_epistemic_graph(config: Optional[dict]):
    """Return the :class:`EpistemicGraph` for this config, creating it if needed.

    Returns a :class:`_NullGraph` when recording is disabled so callers can use
    the returned object unconditionally.
    """
    try:
        if not epistemic_graph_enabled(config):
            return _NullGraph()
        run_id = _run_id_from_config(config)
        with _GRAPHS_LOCK:
            graph = _GRAPHS.get(run_id)
            if graph is None:
                graph = EpistemicGraph(run_id, _graph_dir(config))
                _GRAPHS[run_id] = graph
            return graph
    except Exception:
        # If anything goes wrong setting up recording, degrade to a no-op so the
        # research run itself is unaffected.
        return _NullGraph()


def new_thread_id() -> str:
    """Generate a fresh thread id (handy for ad-hoc scripted runs)."""
    return uuid.uuid4().hex

"""Research state-graph logging for the Deep Research agent.

This module records every meaningful step the agent takes while answering a
question and turns those steps into a directed *state graph*:

* Each state the agent visits (clarification, brief writing, supervisor
  iterations, individual researcher iterations, tool executions, compression,
  final report, ...) becomes a **node**.
* Every transition between states becomes an **edge**.
* For each decision point we also record the *options that were available*, the
  options that were *chosen*, and the options that were *rejected* (for example,
  research units that were dropped because the concurrency limit was exceeded).
* Search queries, retrieved sources, reflections and synthesized findings are
  attached to the relevant nodes.

Events are streamed to a JSON-Lines (``.jsonl``) file - one JSON object per
line - so a run can be inspected live and visualized afterwards with
``scripts/visualize_research_graph.py``.

The logger is intentionally dependency-free and defensive: any failure while
logging is swallowed so that instrumentation can never break a research run.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Schema version written into every log file. Bump if the event format changes
# in a way the visualizer needs to know about.
LOG_SCHEMA_VERSION = 1

# Registry of active runs keyed by a stable run identifier so that the many
# graph nodes/subgraphs that make up a single research run all write to the same
# log file, even though each LangGraph node is invoked independently.
_RUNS: Dict[str, "ResearchRun"] = {}
_RUNS_LOCK = threading.Lock()


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: str) -> str:
    """Make a string safe to embed in a file name."""
    keep = "-_."
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in value)
    return cleaned[:80] or "run"


def _truncate(value: Any, limit: int = 2000) -> Any:
    """Truncate long strings so the log stays readable and bounded."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [truncated {len(value) - limit} chars]"
    return value


def logging_enabled(config: Optional[dict]) -> bool:
    """Return whether research-graph logging should be active for this run.

    Logging is controlled by the ``enable_research_logging`` configurable field
    (or the ``ENABLE_RESEARCH_LOGGING`` environment variable). It defaults to
    enabled so graphs are captured out of the box.
    """
    env = os.environ.get("ENABLE_RESEARCH_LOGGING")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off", "")
    configurable = (config or {}).get("configurable", {}) if config else {}
    value = configurable.get("enable_research_logging")
    if value is None:
        return True
    return bool(value)


def _log_dir(config: Optional[dict]) -> str:
    """Resolve the directory where log files are written."""
    env = os.environ.get("RESEARCH_LOG_DIR")
    if env:
        return env
    configurable = (config or {}).get("configurable", {}) if config else {}
    return configurable.get("research_log_dir") or "research_logs"


def _run_id_from_config(config: Optional[dict]) -> str:
    """Derive a stable run identifier from a LangGraph ``RunnableConfig``.

    Every node and subgraph in a single run shares the same ``thread_id`` (or
    ``run_id``), so we use that to group their events together. When no such
    identifier is available we fall back to a single process-level key.
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


class _NullRun:
    """No-op logger used when logging is disabled.

    It exposes the same surface as :class:`ResearchRun` so callers never need to
    branch on whether logging is active. Every method is a harmless no-op.
    """

    enabled = False
    log_path = None

    def set_question(self, *_args, **_kwargs) -> None:  # noqa: D102
        return None

    def node(self, *_args, **_kwargs) -> Optional[str]:  # noqa: D102
        return None

    def link_delegation(self, *_args, **_kwargs) -> None:  # noqa: D102
        return None

    def delegation_parent(self, *_args, **_kwargs) -> Optional[str]:  # noqa: D102
        return None

    def last_in_scope(self, *_args, **_kwargs) -> Optional[str]:  # noqa: D102
        return None

    def finish(self, *_args, **_kwargs) -> None:  # noqa: D102
        return None


class ResearchRun:
    """Accumulates and streams the state graph for a single research run."""

    enabled = True

    def __init__(self, run_id: str, log_dir: str):
        """Open the log file and emit the ``run_start`` event."""
        self.run_id = run_id
        self._lock = threading.RLock()
        self._counter = 0
        # Last node id seen within each logical scope, used to draw the edge
        # from the previous state to the next state inside that scope.
        self._last_by_scope: Dict[str, str] = {}
        # Maps a research topic to the supervisor "delegate" node that spawned
        # it, so the researcher subgraph can be linked back to the supervisor.
        self._delegation_by_topic: Dict[str, str] = {}
        self._question: Optional[str] = None
        self._finished = False

        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_name = f"research_graph_{_sanitize(run_id)}_{stamp}.jsonl"
        self.log_path = os.path.join(log_dir, file_name)
        # Line-buffered append so events are durable as soon as they happen.
        self._fh = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115
        self._write({
            "event": "run_start",
            "schema_version": LOG_SCHEMA_VERSION,
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
            # Never let logging failures propagate into the research run.
            pass

    def _next_id(self) -> str:
        self._counter += 1
        return f"n{self._counter}"

    # -- public API ---------------------------------------------------------
    def set_question(self, question: Optional[str]) -> None:
        """Record the original user question once, on the first opportunity."""
        with self._lock:
            if self._question is None and question:
                self._question = question
                self._write({
                    "event": "question",
                    "run_id": self.run_id,
                    "ts": _now_iso(),
                    "question": _truncate(question, 4000),
                })

    def node(
        self,
        scope: str,
        kind: str,
        label: str,
        data: Optional[Dict[str, Any]] = None,
        parents: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Record a state node and return its id.

        Args:
            scope: Logical lane the state belongs to (``"main"``,
                ``"supervisor"`` or ``"researcher:<topic>"``). Used to chain
                sequential states automatically.
            kind: Category of the state (e.g. ``"decision"``, ``"action"``,
                ``"search"``, ``"finding"``) - drives node styling in the
                visualizer.
            label: Short human-readable title for the node.
            data: Arbitrary structured details to attach to the node.
            parents: Explicit parent node ids. When omitted, the node is linked
                to the previous node recorded in the same ``scope``.
        """
        with self._lock:
            node_id = self._next_id()
            if parents is None:
                previous = self._last_by_scope.get(scope)
                parents = [previous] if previous else []
            # Drop any parents that are None (defensive).
            parents = [p for p in parents if p]

            record = {
                "event": "node",
                "run_id": self.run_id,
                "id": node_id,
                "scope": scope,
                "kind": kind,
                "label": _truncate(label, 300),
                "parents": parents,
                "data": self._clean_data(data or {}),
                "ts": _now_iso(),
            }
            self._last_by_scope[scope] = node_id
            self._write(record)
            return node_id

    def link_delegation(self, topic: str, node_id: Optional[str]) -> None:
        """Remember which supervisor node delegated a given research topic."""
        if not node_id or not topic:
            return
        with self._lock:
            self._delegation_by_topic[topic] = node_id

    def delegation_parent(self, topic: str) -> Optional[str]:
        """Return the supervisor delegation node id for a research topic."""
        with self._lock:
            return self._delegation_by_topic.get(topic)

    def last_in_scope(self, scope: str) -> Optional[str]:
        """Return the id of the most recent node recorded in a scope."""
        with self._lock:
            return self._last_by_scope.get(scope)

    def _clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate long string values inside the node payload."""
        cleaned: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                cleaned[key] = _truncate(value)
            elif isinstance(value, list):
                cleaned[key] = [_truncate(v) if isinstance(v, str) else v for v in value]
            else:
                cleaned[key] = value
        return cleaned

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
                "data": self._clean_data(data or {}),
            })
            try:
                self._fh.close()
            except Exception:
                pass


def get_logger(config: Optional[dict]):
    """Return the :class:`ResearchRun` for this config, creating it if needed.

    Returns a :class:`_NullRun` when logging is disabled so callers can use the
    returned object unconditionally.
    """
    try:
        if not logging_enabled(config):
            return _NullRun()
        run_id = _run_id_from_config(config)
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
            if run is None:
                run = ResearchRun(run_id, _log_dir(config))
                _RUNS[run_id] = run
            return run
    except Exception:
        # If anything goes wrong setting up logging, degrade to a no-op so the
        # research run itself is unaffected.
        return _NullRun()


def new_thread_id() -> str:
    """Generate a fresh thread id (handy for ad-hoc scripted runs)."""
    return uuid.uuid4().hex

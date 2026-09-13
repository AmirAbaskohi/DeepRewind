#!/usr/bin/env python3
"""Visualize a Deep Research agent run as a state graph.

This script reads one of the JSON-Lines logs produced by
``open_deep_research.research_logger`` (written to ``research_logs/`` by default)
and turns it into a navigable state graph. Each state the agent visited becomes
a node and each transition becomes an edge. Decision nodes show which actions
were *chosen* and which were *rejected*; search nodes show the queries that were
sent and the sources that were retrieved.

Outputs (any combination, controlled by ``--format``):

* ``html``    - a self-contained, interactive HTML page (uses the vis-network
                library from a CDN) with a clickable details panel. Best for
                exploring a run.
* ``dot``     - a Graphviz ``.dot`` file. If the optional ``graphviz`` Python
                package and the Graphviz binaries are installed, a ``.png`` and
                ``.svg`` are also rendered automatically.
* ``mermaid`` - a Mermaid ``.mmd`` flowchart that renders on GitHub or in any
                Mermaid-aware viewer.

Usage examples
--------------
    # Visualize the most recent log into every format
    python scripts/visualize_research_graph.py --latest

    # Visualize a specific log file as interactive HTML only
    python scripts/visualize_research_graph.py research_logs/research_graph_xxx.jsonl --format html

    # List available logs
    python scripts/visualize_research_graph.py --list
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys
import webbrowser
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Visual styling per node kind. Each entry is (fill color, shape).
# ---------------------------------------------------------------------------
KIND_STYLES: Dict[str, Tuple[str, str]] = {
    "start": ("#111827", "ellipse"),
    "decision": ("#fde68a", "diamond"),
    "brief": ("#bfdbfe", "box"),
    "belief": ("#fbcfe8", "box"),
    "proposal": ("#c7d2fe", "box"),
    "delegate": ("#bbf7d0", "box"),
    "rejected": ("#fca5a5", "box"),
    "reflection": ("#e9d5ff", "note"),
    "reasoning": ("#ddd6fe", "note"),
    "search": ("#93c5fd", "box"),
    "finding": ("#86efac", "box"),
    "compress": ("#fcd34d", "box"),
    "report": ("#fdba74", "box"),
    "end": ("#e5e7eb", "ellipse"),
    "tool": ("#f9fafb", "box"),
    "action": ("#fef08a", "box"),
}
DEFAULT_STYLE = ("#f3f4f6", "box")


# ---------------------------------------------------------------------------
# Log loading and graph reconstruction
# ---------------------------------------------------------------------------
def load_events(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL log file and return the list of parsed events."""
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed log line {line_number}: {exc}",
                    file=sys.stderr,
                )
    return events


def build_graph(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconstruct nodes, edges and run metadata from a list of events."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, str]] = []
    question: Optional[str] = None
    run_id: Optional[str] = None
    run_end: Optional[Dict[str, Any]] = None

    for event in events:
        kind = event.get("event")
        if kind == "run_start":
            run_id = event.get("run_id")
        elif kind == "question":
            question = event.get("question")
        elif kind == "run_end":
            run_end = event.get("data", {})
        elif kind == "node":
            node_id = event.get("id")
            if not node_id:
                continue
            nodes[node_id] = {
                "id": node_id,
                "scope": event.get("scope", ""),
                "kind": event.get("kind", ""),
                "label": event.get("label", ""),
                "data": event.get("data", {}),
                "ts": event.get("ts", ""),
                "parents": event.get("parents", []) or [],
            }

    # Build edges from each node's recorded parents.
    for node in nodes.values():
        for parent in node["parents"]:
            if parent in nodes:
                edges.append({"from": parent, "to": node["id"]})

    # Synthesize a single START node and connect all roots (nodes without a
    # known parent) to it, so the graph has one clear entry point.
    start_label = "START"
    if question:
        start_label = "START\n" + _shorten(question, 120)
    start_node = {
        "id": "__start__",
        "scope": "main",
        "kind": "start",
        "label": start_label,
        "data": {"question": question or ""},
        "ts": "",
        "parents": [],
    }
    roots = [n["id"] for n in nodes.values() if not any(p in nodes for p in n["parents"])]
    # Preserve insertion order of roots for a tidy layout.
    ordered_roots = [nid for nid in nodes.keys() if nid in set(roots)]
    for root_id in ordered_roots:
        edges.append({"from": "__start__", "to": root_id})

    ordered_nodes = {"__start__": start_node}
    ordered_nodes.update(nodes)

    return {
        "run_id": run_id,
        "question": question,
        "run_end": run_end,
        "nodes": ordered_nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Label / detail helpers
# ---------------------------------------------------------------------------
def _shorten(text: Any, limit: int) -> str:
    """Return a single-line, length-limited version of any value."""
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def node_caption(node: Dict[str, Any]) -> str:
    """Build the short multi-line caption shown inside a graph node."""
    data = node.get("data", {})
    lines = [node.get("label", node.get("id", ""))]
    kind = node.get("kind")

    if kind == "decision":
        chosen = data.get("chosen_actions") or data.get("chosen_action")
        if chosen:
            lines.append("chose: " + _shorten(", ".join(chosen) if isinstance(chosen, list) else chosen, 60))
        rejected = data.get("rejected_actions") or data.get("rejected_action")
        if rejected:
            lines.append("rejected: " + _shorten(", ".join(rejected) if isinstance(rejected, list) else rejected, 60))
        queries = data.get("search_queries")
        if queries:
            lines.append("queries: " + _shorten("; ".join(queries), 80))
        if data.get("is_sufficient") is not None:
            lines.append("sufficient: " + str(data.get("is_sufficient")))
        if data.get("missing_information"):
            lines.append("missing: " + _shorten(data.get("missing_information"), 80))
    elif kind == "belief":
        beliefs = data.get("beliefs") or []
        if data.get("iteration") is not None:
            lines.append(f"iteration: {data.get('iteration')}/{data.get('max_iterations', '?')}")
        lines.append(f"beliefs: {len(beliefs)} | selected: {len(data.get('selected_beliefs') or [])}")
        for b in beliefs[:5]:
            lines.append(f"[{b.get('score')}] " + _shorten(b.get("belief"), 70))
    elif kind == "reasoning":
        if data.get("iteration") is not None:
            lines.append(f"iteration: {data.get('iteration')}")
        lines.append(_shorten(data.get("reasoning"), 140))
    elif kind == "search":
        queries = data.get("queries") or []
        if queries:
            lines.append("queries: " + _shorten("; ".join(queries), 80))
        lines.append(f"sources: {data.get('num_sources', 0)}")
    elif kind == "delegate":
        lines.append(_shorten(data.get("research_topic"), 90))
        if data.get("score") is not None:
            lines.append(f"score: {data.get('score')}")
    elif kind == "rejected":
        lines.append(_shorten(data.get("research_topic") or data.get("belief"), 90))
        if data.get("score") is not None:
            lines.append(f"score: {data.get('score')}")
    elif kind == "proposal":
        proposals = data.get("proposals") or []
        if data.get("iteration") is not None:
            lines.append(f"iteration: {data.get('iteration')}/{data.get('max_iterations', '?')}")
        lines.append(f"proposed: {len(proposals)} | selected: {len(data.get('selected_research_topics') or [])}")
        for p in proposals[:5]:
            lines.append(f"[{p.get('score')}] " + _shorten(p.get("research_topic"), 70))
    elif kind in ("reflection",):
        lines.append(_shorten(data.get("reflection") or data.get("findings_summary"), 110))
    elif kind == "brief":
        lines.append(_shorten(data.get("research_brief"), 110))
    elif kind in ("finding", "compress"):
        lines.append(_shorten(data.get("compressed_research"), 110))
    elif kind == "report":
        lines.append(_shorten(data.get("status"), 40))
    elif kind == "end":
        if data.get("exit_reason"):
            lines.append("reason: " + _shorten(data.get("exit_reason"), 50))

    return "\n".join(line for line in lines if line)


def node_detail_html(node: Dict[str, Any]) -> str:
    """Build a full HTML detail block for the interactive side panel."""
    parts = [
        f"<h3>{html.escape(node.get('label', ''))}</h3>",
        f"<p><b>id:</b> {html.escape(node.get('id', ''))} &nbsp; "
        f"<b>kind:</b> {html.escape(node.get('kind', ''))}</p>",
        f"<p><b>scope:</b> {html.escape(node.get('scope', ''))}</p>",
        f"<p><b>time:</b> {html.escape(node.get('ts', ''))}</p>",
    ]
    data = node.get("data", {})
    if data:
        pretty = html.escape(json.dumps(data, indent=2, ensure_ascii=False))
        parts.append(f"<pre>{pretty}</pre>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# DOT (Graphviz) output
# ---------------------------------------------------------------------------
def to_dot(graph: Dict[str, Any]) -> str:
    """Render the graph as Graphviz DOT source."""
    lines = [
        "digraph research_state_graph {",
        "  rankdir=TB;",
        "  node [style=\"filled,rounded\", fontname=\"Helvetica\", fontsize=10];",
        "  edge [fontname=\"Helvetica\", fontsize=8, color=\"#6b7280\"];",
    ]
    for node in graph["nodes"].values():
        fill, shape = KIND_STYLES.get(node.get("kind", ""), DEFAULT_STYLE)
        font_color = "#ffffff" if node.get("kind") == "start" else "#111827"
        caption = node_caption(node).replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
        lines.append(
            f"  \"{node['id']}\" ["
            f"label=\"{caption}\", fillcolor=\"{fill}\", shape={shape}, "
            f"fontcolor=\"{font_color}\"];"
        )
    for edge in graph["edges"]:
        lines.append(f"  \"{edge['from']}\" -> \"{edge['to']}\";")
    lines.append("}")
    return "\n".join(lines)


def render_dot(dot_source: str, output_base: str) -> List[str]:
    """Write the DOT file and, if Graphviz is available, render PNG and SVG."""
    written = []
    dot_path = output_base + ".dot"
    with open(dot_path, "w", encoding="utf-8") as handle:
        handle.write(dot_source)
    written.append(dot_path)

    try:
        import graphviz  # type: ignore

        source = graphviz.Source(dot_source)
        for fmt in ("png", "svg"):
            try:
                rendered = source.render(
                    filename=os.path.basename(output_base),
                    directory=os.path.dirname(output_base) or ".",
                    format=fmt,
                    cleanup=True,
                )
                written.append(rendered)
            except Exception as exc:  # graphviz binary missing, etc.
                print(f"Note: could not render {fmt} ({exc}).", file=sys.stderr)
    except ImportError:
        print(
            "Note: the 'graphviz' Python package is not installed, so only the "
            ".dot file was written. Install it (and the Graphviz binaries) to "
            "auto-render PNG/SVG, or paste the .dot into https://dreampuf.github.io/GraphvizOnline/",
            file=sys.stderr,
        )
    return written


# ---------------------------------------------------------------------------
# Mermaid output
# ---------------------------------------------------------------------------
def to_mermaid(graph: Dict[str, Any]) -> str:
    """Render the graph as a Mermaid flowchart."""
    lines = ["flowchart TD"]
    # Class definitions for the different node kinds. Class names are prefixed
    # with "k_" so they never collide with Mermaid reserved words (e.g. "end").
    for kind, style in KIND_STYLES.items():
        text_color = "#ffffff" if kind == "start" else "#111827"
        lines.append(
            f"    classDef k_{kind} fill:{style[0]},stroke:#374151,color:{text_color};"
        )

    def mermaid_id(node_id: str) -> str:
        return node_id.replace("__", "S_").strip("_") or "n0"

    for node in graph["nodes"].values():
        nid = mermaid_id(node["id"])
        caption = node_caption(node).replace("\n", "<br/>")
        caption = caption.replace('"', "'")
        lines.append(f'    {nid}["{caption}"]')
        lines.append(f"    class {nid} k_{node.get('kind', 'tool')};")
    for edge in graph["edges"]:
        lines.append(f"    {mermaid_id(edge['from'])} --> {mermaid_id(edge['to'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive HTML output (vis-network)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Research State Graph - {title}</title>
<script type="text/javascript" src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  html, body {{ margin: 0; height: 100%; font-family: Helvetica, Arial, sans-serif; }}
  #wrap {{ display: flex; height: 100vh; }}
  #graph {{ flex: 1 1 auto; height: 100%; border-right: 1px solid #e5e7eb; }}
  #side {{ width: 380px; padding: 16px; overflow-y: auto; box-sizing: border-box; }}
  #side h2 {{ margin-top: 0; font-size: 16px; }}
  #legend span {{ display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 6px; font-size: 12px; }}
  #detail pre {{ white-space: pre-wrap; word-break: break-word; background: #f9fafb; padding: 8px; border-radius: 6px; font-size: 12px; }}
  .q {{ background: #eef2ff; padding: 10px; border-radius: 8px; font-size: 13px; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="graph"></div>
  <div id="side">
    <h2>Research State Graph</h2>
    <div class="q"><b>Question:</b><br/>{question}</div>
    <p style="font-size:12px;color:#6b7280">run_id: {run_id} &middot; {node_count} states &middot; {edge_count} transitions</p>
    <div id="legend">{legend}</div>
    <hr/>
    <div id="detail"><i>Click a state node to see its full details (queries, chosen/rejected actions, sources, findings...).</i></div>
  </div>
</div>
<script type="text/javascript">
  const nodes = new vis.DataSet({nodes_json});
  const edges = new vis.DataSet({edges_json});
  const details = {details_json};
  const container = document.getElementById('graph');
  const data = {{ nodes: nodes, edges: edges }};
  const options = {{
    layout: {{ hierarchical: {{ enabled: true, direction: 'UD', sortMethod: 'directed', nodeSpacing: 180, levelSeparation: 130 }} }},
    physics: false,
    interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
    nodes: {{ shape: 'box', margin: 8, widthConstraint: {{ maximum: 240 }}, font: {{ size: 12 }} }},
    edges: {{ arrows: 'to', smooth: {{ type: 'cubicBezier' }}, color: {{ color: '#9ca3af' }} }}
  }};
  const network = new vis.Network(container, data, options);
  network.on('click', function(params) {{
    if (params.nodes.length > 0) {{
      const id = params.nodes[0];
      document.getElementById('detail').innerHTML = details[id] || '<i>No details.</i>';
    }}
  }});
</script>
</body>
</html>
"""


def to_html(graph: Dict[str, Any]) -> str:
    """Render the graph as a self-contained interactive HTML page."""
    vis_nodes = []
    vis_edges = []
    details: Dict[str, str] = {}

    for node in graph["nodes"].values():
        fill, shape = KIND_STYLES.get(node.get("kind", ""), DEFAULT_STYLE)
        shape_map = {"diamond": "diamond", "ellipse": "ellipse", "note": "box", "box": "box"}
        font_color = "#ffffff" if node.get("kind") == "start" else "#111827"
        vis_nodes.append({
            "id": node["id"],
            "label": node_caption(node),
            "color": {"background": fill, "border": "#374151"},
            "shape": shape_map.get(shape, "box"),
            "font": {"color": font_color},
        })
        details[node["id"]] = node_detail_html(node)

    for index, edge in enumerate(graph["edges"]):
        vis_edges.append({"id": f"e{index}", "from": edge["from"], "to": edge["to"]})

    legend_items = "".join(
        f'<span style="background:{style[0]}">{kind}</span>'
        for kind, style in KIND_STYLES.items()
    )

    return HTML_TEMPLATE.format(
        title=html.escape(str(graph.get("run_id") or "run")),
        question=html.escape(graph.get("question") or "(no clarified question recorded)"),
        run_id=html.escape(str(graph.get("run_id") or "")),
        node_count=len(graph["nodes"]),
        edge_count=len(graph["edges"]),
        legend=legend_items,
        nodes_json=json.dumps(vis_nodes, ensure_ascii=False),
        edges_json=json.dumps(vis_edges, ensure_ascii=False),
        details_json=json.dumps(details, ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def find_logs(log_dir: str) -> List[str]:
    """Return available log files sorted newest first."""
    pattern = os.path.join(log_dir, "research_graph_*.jsonl")
    files = glob.glob(pattern)
    files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return files


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize a Deep Research agent run as a state graph.",
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        help="Path to a research_graph_*.jsonl log file.",
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("RESEARCH_LOG_DIR", "research_logs"),
        help="Directory to search for logs (default: research_logs).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent log file in --log-dir.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available log files and exit.",
    )
    parser.add_argument(
        "--format",
        default="all",
        choices=["all", "html", "dot", "mermaid"],
        help="Which output format(s) to generate (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated files (default: alongside the log file).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the default web browser.",
    )
    args = parser.parse_args(argv)

    if args.list:
        logs = find_logs(args.log_dir)
        if not logs:
            print(f"No logs found in '{args.log_dir}'.")
            return 1
        print(f"Found {len(logs)} log(s) in '{args.log_dir}':")
        for path in logs:
            print(f"  {path}")
        return 0

    log_file = args.log_file
    if not log_file and args.latest:
        logs = find_logs(args.log_dir)
        if not logs:
            print(f"No logs found in '{args.log_dir}'.", file=sys.stderr)
            return 1
        log_file = logs[0]
    if not log_file:
        parser.error("Provide a log file, or use --latest, or --list.")

    if not os.path.isfile(log_file):
        print(f"Error: log file not found: {log_file}", file=sys.stderr)
        return 1

    events = load_events(log_file)
    if not events:
        print(f"Error: no events found in {log_file}", file=sys.stderr)
        return 1

    graph = build_graph(events)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(log_file))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(log_file))[0]
    output_base = os.path.join(output_dir, base_name)

    written: List[str] = []
    html_path: Optional[str] = None

    if args.format in ("all", "html"):
        html_path = output_base + ".html"
        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(to_html(graph))
        written.append(html_path)

    if args.format in ("all", "dot"):
        written.extend(render_dot(to_dot(graph), output_base))

    if args.format in ("all", "mermaid"):
        mmd_path = output_base + ".mmd"
        with open(mmd_path, "w", encoding="utf-8") as handle:
            handle.write(to_mermaid(graph))
        written.append(mmd_path)

    print(f"Visualized {len(graph['nodes'])} states and {len(graph['edges'])} transitions.")
    print("Wrote:")
    for path in written:
        print(f"  {path}")

    if args.open and html_path:
        webbrowser.open("file://" + os.path.abspath(html_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

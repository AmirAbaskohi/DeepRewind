#!/usr/bin/env python3
"""Visualize a Deep Research agent's *epistemic* research graph.

Where ``scripts/visualize_research_graph.py`` renders *what the agent did over
time* (a state graph of actions), this script renders *what the agent knows,
believes, assumes, cites, commits to, and writes* - the **epistemic graph**
produced by ``open_deep_research.epistemic_graph`` and written to
``epistemic_graphs/`` by default.

The graph is loaded from a JSON-Lines (``.jsonl``) file whose events describe
typed nodes (``Source``, ``Evidence``, ``Claim``, ``Hypothesis``,
``Assumption``, ``Commitment``, ``DraftFragment``, ``PlanStep``) and typed
edges (``supports``, ``contradicts``, ``depends_on``, ``compresses``,
``used_in``, ``derived_from``, ``cites``, ``revises``, ``invalidates``).

Outputs (any combination, controlled by ``--format``):

* ``html``    - a self-contained, interactive HTML page (uses the vis-network
                library from a CDN) with a clickable details panel and a legend
                for node/edge types. Best for exploring an epistemic state.
* ``dot``     - a Graphviz ``.dot`` file. If the optional ``graphviz`` Python
                package and the Graphviz binaries are installed, a ``.png`` and
                ``.svg`` are also rendered automatically.
* ``mermaid`` - a Mermaid ``.mmd`` flowchart that renders on GitHub or in any
                Mermaid-aware viewer.

Usage examples
--------------
    # Visualize the most recent epistemic graph into every format
    python scripts/visualize_epistemic_graph.py --latest

    # Visualize a specific graph file as interactive HTML only
    python scripts/visualize_epistemic_graph.py epistemic_graphs/epistemic_graph_xxx.jsonl --format html

    # List available epistemic graphs
    python scripts/visualize_epistemic_graph.py --list
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
# Visual styling per node type. Each entry is (fill color, shape).
# ---------------------------------------------------------------------------
NODE_TYPE_STYLES: Dict[str, Tuple[str, str]] = {
    "Source": ("#93c5fd", "box"),          # blue
    "Evidence": ("#86efac", "box"),        # green
    "Claim": ("#fde68a", "box"),           # amber
    "Hypothesis": ("#c7d2fe", "hexagon"),  # indigo
    "Assumption": ("#ddd6fe", "hexagon"),  # violet
    "Commitment": ("#fca5a5", "ellipse"),  # red
    "DraftFragment": ("#fdba74", "note"),  # orange
    "PlanStep": ("#e5e7eb", "diamond"),    # gray
    "start": ("#111827", "ellipse"),
}
DEFAULT_NODE_STYLE = ("#f3f4f6", "box")

# Short prefixes shown before a node's caption to make its type obvious even
# without color (useful for grayscale/printed views).
NODE_TYPE_PREFIX: Dict[str, str] = {
    "Source": "[SRC]",
    "Evidence": "[EVD]",
    "Claim": "[CLM]",
    "Hypothesis": "[HYP]",
    "Assumption": "[ASM]",
    "Commitment": "[CMT]",
    "DraftFragment": "[DFT]",
    "PlanStep": "[PLN]",
}

# ---------------------------------------------------------------------------
# Visual styling per edge type. Each entry is (color, dashed?).
# ---------------------------------------------------------------------------
EDGE_TYPE_STYLES: Dict[str, Tuple[str, bool]] = {
    "supports": ("#16a34a", False),      # green solid
    "contradicts": ("#dc2626", True),    # red dashed
    "depends_on": ("#2563eb", False),    # blue solid
    "compresses": ("#d97706", False),    # amber solid
    "used_in": ("#7c3aed", False),       # violet solid
    "derived_from": ("#0891b2", False),  # cyan solid
    "cites": ("#0d9488", True),          # teal dashed
    "revises": ("#ca8a04", True),        # yellow dashed
    "invalidates": ("#b91c1c", True),    # dark red dashed
}
DEFAULT_EDGE_STYLE = ("#9ca3af", False)


# ---------------------------------------------------------------------------
# Graph file loading and reconstruction
# ---------------------------------------------------------------------------
def load_events(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL epistemic-graph file and return the parsed events."""
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
                    f"Warning: skipping malformed line {line_number}: {exc}",
                    file=sys.stderr,
                )
    return events


def build_graph(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconstruct nodes, edges and run metadata from a list of events."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
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
            run_end = event.get("metadata", {})
        elif kind == "node":
            node_id = event.get("id")
            if not node_id:
                continue
            nodes[node_id] = {
                "id": node_id,
                "type": event.get("type", ""),
                "label": event.get("label", ""),
                "text": event.get("text", ""),
                "metadata": event.get("metadata", {}),
                "ts": event.get("ts", ""),
            }
        elif kind == "edge":
            source = event.get("source")
            target = event.get("target")
            if not source or not target:
                continue
            edges.append({
                "id": event.get("id", ""),
                "type": event.get("type", ""),
                "from": source,
                "to": target,
                "metadata": event.get("metadata", {}),
                "ts": event.get("ts", ""),
            })

    # Drop edges that reference unknown nodes (defensive against partial logs).
    edges = [e for e in edges if e["from"] in nodes and e["to"] in nodes]

    return {
        "run_id": run_id,
        "question": question,
        "run_end": run_end,
        "nodes": nodes,
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
        return text[: limit - 1] + "\u2026"
    return text


def node_caption(node: Dict[str, Any]) -> str:
    """Build the short multi-line caption shown inside a graph node."""
    node_type = node.get("type", "")
    prefix = NODE_TYPE_PREFIX.get(node_type, f"[{node_type}]")
    body = node.get("label") or node.get("text") or node.get("id", "")
    lines = [f"{prefix} {node.get('id', '')}", _shorten(body, 90)]

    metadata = node.get("metadata", {})
    topic = metadata.get("research_topic")
    if topic:
        lines.append("topic: " + _shorten(topic, 60))
    if metadata.get("url"):
        lines.append(_shorten(metadata.get("url"), 60))
    if metadata.get("score") is not None:
        lines.append(f"score: {metadata.get('score')}")
    if metadata.get("iteration") is not None:
        lines.append(f"iteration: {metadata.get('iteration')}")
    return "\n".join(line for line in lines if line)


def node_detail_html(node: Dict[str, Any]) -> str:
    """Build a full HTML detail block for the interactive side panel."""
    parts = [
        f"<h3>{html.escape(NODE_TYPE_PREFIX.get(node.get('type', ''), ''))} "
        f"{html.escape(node.get('label', ''))}</h3>",
        f"<p><b>id:</b> {html.escape(node.get('id', ''))} &nbsp; "
        f"<b>type:</b> {html.escape(node.get('type', ''))}</p>",
        f"<p><b>time:</b> {html.escape(node.get('ts', ''))}</p>",
    ]
    text = node.get("text", "")
    if text:
        parts.append(f"<p><b>text:</b></p><pre>{html.escape(str(text))}</pre>")
    metadata = node.get("metadata", {})
    if metadata:
        pretty = html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))
        parts.append(f"<p><b>metadata:</b></p><pre>{pretty}</pre>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# DOT (Graphviz) output
# ---------------------------------------------------------------------------
def to_dot(graph: Dict[str, Any]) -> str:
    """Render the epistemic graph as Graphviz DOT source."""
    lines = [
        "digraph epistemic_graph {",
        "  rankdir=LR;",
        "  node [style=\"filled,rounded\", fontname=\"Helvetica\", fontsize=10];",
        "  edge [fontname=\"Helvetica\", fontsize=8];",
    ]
    for node in graph["nodes"].values():
        fill, shape = NODE_TYPE_STYLES.get(node.get("type", ""), DEFAULT_NODE_STYLE)
        caption = node_caption(node).replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
        lines.append(
            f"  \"{node['id']}\" ["
            f"label=\"{caption}\", fillcolor=\"{fill}\", shape={shape}];"
        )
    for edge in graph["edges"]:
        color, dashed = EDGE_TYPE_STYLES.get(edge.get("type", ""), DEFAULT_EDGE_STYLE)
        style = ", style=dashed" if dashed else ""
        lines.append(
            f"  \"{edge['from']}\" -> \"{edge['to']}\" ["
            f"label=\"{edge.get('type', '')}\", color=\"{color}\", "
            f"fontcolor=\"{color}\"{style}];"
        )
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
    """Render the epistemic graph as a Mermaid flowchart."""
    lines = ["flowchart LR"]
    for node_type, style in NODE_TYPE_STYLES.items():
        text_color = "#ffffff" if node_type == "start" else "#111827"
        lines.append(
            f"    classDef t_{node_type} fill:{style[0]},stroke:#374151,color:{text_color};"
        )

    def mermaid_id(node_id: str) -> str:
        return "n_" + "".join(c if c.isalnum() else "_" for c in node_id)

    for node in graph["nodes"].values():
        nid = mermaid_id(node["id"])
        caption = node_caption(node).replace("\n", "<br/>").replace('"', "'")
        lines.append(f'    {nid}["{caption}"]')
        lines.append(f"    class {nid} t_{node.get('type', 'PlanStep')};")
    for edge in graph["edges"]:
        lines.append(
            f"    {mermaid_id(edge['from'])} -->|{edge.get('type', '')}| {mermaid_id(edge['to'])}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive HTML output (vis-network)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Epistemic Research Graph - {title}</title>
<script type="text/javascript" src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  html, body {{ margin: 0; height: 100%; font-family: Helvetica, Arial, sans-serif; }}
  #wrap {{ display: flex; height: 100vh; }}
  #graph {{ flex: 1 1 auto; height: 100%; border-right: 1px solid #e5e7eb; }}
  #side {{ width: 400px; padding: 16px; overflow-y: auto; box-sizing: border-box; }}
  #side h2 {{ margin-top: 0; font-size: 16px; }}
  #side h4 {{ margin-bottom: 4px; }}
  .legend span {{ display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 6px; font-size: 12px; }}
  .eline {{ display: inline-block; margin: 2px 8px 2px 0; font-size: 12px; }}
  .eswatch {{ display: inline-block; width: 22px; height: 0; border-top-width: 3px; border-top-style: solid; vertical-align: middle; margin-right: 6px; }}
  #detail pre {{ white-space: pre-wrap; word-break: break-word; background: #f9fafb; padding: 8px; border-radius: 6px; font-size: 12px; }}
  .q {{ background: #eef2ff; padding: 10px; border-radius: 8px; font-size: 13px; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="graph"></div>
  <div id="side">
    <h2>Epistemic Research Graph</h2>
    <div class="q"><b>Question:</b><br/>{question}</div>
    <p style="font-size:12px;color:#6b7280">run_id: {run_id} &middot; {node_count} nodes &middot; {edge_count} edges</p>
    <h4>Node types</h4>
    <div class="legend">{node_legend}</div>
    <h4>Edge types</h4>
    <div>{edge_legend}</div>
    <hr/>
    <div id="detail"><i>Click a node to see its full text and metadata (sources, evidence, claims, hypotheses, commitments, draft fragments...).</i></div>
  </div>
</div>
<script type="text/javascript">
  const nodes = new vis.DataSet({nodes_json});
  const edges = new vis.DataSet({edges_json});
  const details = {details_json};
  const container = document.getElementById('graph');
  const data = {{ nodes: nodes, edges: edges }};
  const options = {{
    layout: {{ improvedLayout: true }},
    physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -12000, springLength: 160 }} }},
    interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
    nodes: {{ shape: 'box', margin: 8, widthConstraint: {{ maximum: 240 }}, font: {{ size: 12 }} }},
    edges: {{ arrows: 'to', smooth: {{ type: 'cubicBezier' }}, font: {{ size: 10, align: 'middle' }} }}
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
    """Render the epistemic graph as a self-contained interactive HTML page."""
    vis_nodes = []
    vis_edges = []
    details: Dict[str, str] = {}

    shape_map = {
        "hexagon": "hexagon",
        "ellipse": "ellipse",
        "note": "box",
        "diamond": "diamond",
        "box": "box",
    }
    for node in graph["nodes"].values():
        fill, shape = NODE_TYPE_STYLES.get(node.get("type", ""), DEFAULT_NODE_STYLE)
        vis_nodes.append({
            "id": node["id"],
            "label": node_caption(node),
            "color": {"background": fill, "border": "#374151"},
            "shape": shape_map.get(shape, "box"),
        })
        details[node["id"]] = node_detail_html(node)

    for edge in graph["edges"]:
        color, dashed = EDGE_TYPE_STYLES.get(edge.get("type", ""), DEFAULT_EDGE_STYLE)
        vis_edges.append({
            "id": edge.get("id") or f"{edge['from']}->{edge['to']}",
            "from": edge["from"],
            "to": edge["to"],
            "label": edge.get("type", ""),
            "color": {"color": color},
            "dashes": dashed,
        })

    node_legend = "".join(
        f'<span style="background:{style[0]}">{NODE_TYPE_PREFIX.get(node_type, "")} {node_type}</span>'
        for node_type, style in NODE_TYPE_STYLES.items()
        if node_type != "start"
    )
    edge_legend = "".join(
        f'<span class="eline"><span class="eswatch" style="border-top-color:{style[0]};'
        f'border-top-style:{"dashed" if style[1] else "solid"}"></span>{edge_type}</span>'
        for edge_type, style in EDGE_TYPE_STYLES.items()
    )

    return HTML_TEMPLATE.format(
        title=html.escape(str(graph.get("run_id") or "run")),
        question=html.escape(graph.get("question") or "(no clarified question recorded)"),
        run_id=html.escape(str(graph.get("run_id") or "")),
        node_count=len(graph["nodes"]),
        edge_count=len(graph["edges"]),
        node_legend=node_legend,
        edge_legend=edge_legend,
        nodes_json=json.dumps(vis_nodes, ensure_ascii=False),
        edges_json=json.dumps(vis_edges, ensure_ascii=False),
        details_json=json.dumps(details, ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def find_graphs(graph_dir: str) -> List[str]:
    """Return available epistemic-graph files sorted newest first."""
    pattern = os.path.join(graph_dir, "epistemic_graph_*.jsonl")
    files = glob.glob(pattern)
    files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return files


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize a Deep Research agent's epistemic research graph.",
    )
    parser.add_argument(
        "graph_file",
        nargs="?",
        help="Path to an epistemic_graph_*.jsonl file.",
    )
    parser.add_argument(
        "--graph-dir",
        default=os.environ.get("EPISTEMIC_GRAPH_DIR", "epistemic_graphs"),
        help="Directory to search for epistemic graphs (default: epistemic_graphs).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent graph file in --graph-dir.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available epistemic-graph files and exit.",
    )
    parser.add_argument(
        "--format",
        default="html,dot,mermaid",
        help="Comma-separated outputs to produce: html, dot, mermaid (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write outputs into (default: alongside the graph file).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated HTML in a web browser.",
    )
    args = parser.parse_args(argv)

    if args.list:
        graphs = find_graphs(args.graph_dir)
        if not graphs:
            print(f"No epistemic graphs found in '{args.graph_dir}'.")
            return 0
        print(f"Epistemic graphs in '{args.graph_dir}' (newest first):")
        for path in graphs:
            print(f"  {path}")
        return 0

    # Resolve which graph file to visualize.
    graph_file = args.graph_file
    if not graph_file and args.latest:
        graphs = find_graphs(args.graph_dir)
        if not graphs:
            print(f"No epistemic graphs found in '{args.graph_dir}'.", file=sys.stderr)
            return 1
        graph_file = graphs[0]
    if not graph_file:
        graphs = find_graphs(args.graph_dir)
        if len(graphs) == 1:
            graph_file = graphs[0]
        else:
            parser.print_help()
            print(
                "\nProvide a graph file, or use --latest / --list.",
                file=sys.stderr,
            )
            return 1

    if not os.path.isfile(graph_file):
        print(f"Error: file not found: {graph_file}", file=sys.stderr)
        return 1

    events = load_events(graph_file)
    graph = build_graph(events)

    formats = [fmt.strip().lower() for fmt in args.format.split(",") if fmt.strip()]
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(graph_file))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(graph_file))[0]
    output_base = os.path.join(output_dir, base_name)

    written: List[str] = []

    if "dot" in formats:
        written.extend(render_dot(to_dot(graph), output_base))
    if "mermaid" in formats:
        mmd_path = output_base + ".mmd"
        with open(mmd_path, "w", encoding="utf-8") as handle:
            handle.write(to_mermaid(graph))
        written.append(mmd_path)
    if "html" in formats:
        html_path = output_base + ".html"
        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(to_html(graph))
        written.append(html_path)
        if not args.no_open:
            try:
                webbrowser.open("file://" + os.path.abspath(html_path))
            except Exception:
                pass

    print(
        f"Epistemic graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges."
    )
    if written:
        print("Wrote:")
        for path in written:
            print(f"  {path}")
    else:
        print("No outputs were produced (check --format).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

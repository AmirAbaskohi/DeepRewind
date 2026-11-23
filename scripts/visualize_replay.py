#!/usr/bin/env python3
"""Replay an epistemic graph alongside recorded world-model and rollback events."""

from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path
from typing import Any


def load_events(path: Path, source: str) -> list[dict[str, Any]]:
    """Read a JSONL event stream, rejecting corrupt or incomplete records."""
    events = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(event, dict) or not event.get("event") or not event.get("ts"):
                raise ValueError(f"{path}:{number}: expected an event and timestamp")
            events.append({"source": source, "record": event})
    if not events:
        raise ValueError(f"{path}: no events found")
    return events


def make_replay(
    epistemic_path: Path, world_model_path: Path | None, output: Path, *,
    notice: str = "", initial_event: str | None = None,
) -> Path:
    """Create an offline HTML replay from two streams belonging to the same run."""
    events = load_events(epistemic_path, "graph")
    if world_model_path:
        world_events = load_events(world_model_path, "world")
        graph_ids = {e["record"].get("run_id") for e in events}
        world_ids = {e["record"].get("run_id") for e in world_events}
        if None in graph_ids or len(graph_ids) != 1 or graph_ids != world_ids:
            raise ValueError("Graph and world-model logs must have the same single run_id")
        events.extend(world_events)
    events.sort(key=lambda event: event["record"]["ts"])
    data = json.dumps(events, ensure_ascii=False).replace("<", "\\u003c")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = next(
        (i for i, event in enumerate(events) if event["record"]["event"] == initial_event), 0
    )
    page = HTML.replace("__NOTICE__", html.escape(notice)).replace("__EVENTS__", data)
    page = page.replace("__INITIAL_STEP__", str(start))
    output.write_text(page, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epistemic_graph", type=Path)
    parser.add_argument("--world-model-log", type=Path)
    parser.add_argument("--output", type=Path, help="HTML path (default: next to the graph log)")
    parser.add_argument("--open", action="store_true", help="Open the replay in your browser")
    args = parser.parse_args()
    output = args.output or args.epistemic_graph.with_suffix(".replay.html")
    path = make_replay(args.epistemic_graph, args.world_model_log, output)
    print(f"Replay: {path}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DeepRewind - run replay</title>
<style>
:root{font:14px/1.5 system-ui,sans-serif;color:#17233c;background:#f3f6fb}
body{margin:0} header{background:#14264b;color:white;padding:18px 24px}
h1{font-size:21px;margin:0} header p{margin:4px 0;color:#c7d6ef}
main{padding:18px;max-width:1800px;margin:auto}
.bar,.panel{background:white;border:1px solid #dce4ef;border-radius:12px;padding:14px;box-shadow:0 3px 14px #2034560b}
.bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
button,select{border:1px solid #b9c9dd;background:white;border-radius:7px;padding:7px 10px;color:#17233c;cursor:pointer}
button:hover{background:#e8f0fc} input[type=range]{flex:1;min-width:180px}
.layout{display:grid;grid-template-columns:260px minmax(400px,1fr) 340px;gap:12px;margin-top:12px}
.panel{min-width:0} h2{font-size:15px;margin:0 0 10px} .muted{color:#65758d;font-size:12px}
#timeline{max-height:70vh;overflow:auto} .event{display:block;width:100%;text-align:left;margin:4px 0;font-size:12px}
.event.active{border-color:#315fba;background:#e8f0ff}.event.world{border-left:4px solid #8854c7}
.event.rewind{border-left:4px solid #d74747}.event.graph{border-left:4px solid #5b86b7}
#graph{overflow:auto;max-height:70vh;background:#f8faff;border-radius:8px}
#graph svg{display:block} .node{cursor:pointer} .node:hover rect{stroke:#203d7b;stroke-width:3}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;padding:10px;border-radius:7px;font-size:12px;max-height:40vh;overflow:auto}
.legend{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}.chip{padding:2px 7px;border-radius:5px;background:#edf2fa;font-size:11px}
.warning{background:#fff5da;border-radius:7px;padding:8px;margin:8px 0;color:#714f09}
@media(max-width:1100px){.layout{grid-template-columns:1fr}.panel{max-width:100%}#timeline{max-height:180px}}
</style></head><body>
<header><h1>DeepRewind / Step-by-step replay</h1><p>Recorded graph changes, world-model predictions, and consistency repairs</p></header>
<main><div class="warning" id="notice">__NOTICE__</div><div class="bar"><button id="prev">Previous</button><button id="next">Next</button>
<input id="step" type="range" min="0" value="0"><strong id="counter"></strong>
<label>Show <select id="filter"><option value="all">all events</option><option value="world">world model &amp; rewind</option><option value="rewind">rewind only</option></select></label></div>
<div id="warning" class="warning" hidden></div>
<div class="layout"><section class="panel"><h2>Timeline</h2><p class="muted">Select an event to inspect the graph at that point. Ordering uses recorded timestamps.</p><div id="timeline"></div></section>
<section class="panel"><h2>Epistemic graph at this step</h2><div class="legend"><span class="chip">Blue: source</span><span class="chip">Green: evidence</span><span class="chip">Yellow: claim</span><span class="chip">Purple: hypothesis</span><span class="chip">Red: commitment</span><span class="chip">Gray/dashed: retracted or stale</span></div><p class="muted">Click a node for details. Predicted changes are not shown as actual graph changes.</p><div id="graph"></div></section>
<section class="panel"><h2 id="detail-title">Event details</h2><div id="summary"></div><pre id="details"></pre></section></div></main>
<script>
const events = __EVENTS__;
document.getElementById("notice").hidden = !document.getElementById("notice").textContent;
const range = document.getElementById("step"), timeline = document.getElementById("timeline");
range.max = Math.max(events.length - 1, 0);
range.value = __INITIAL_STEP__;
const colors = {Source:"#cce1ff",Evidence:"#c5f1d4",Claim:"#fff0ae",Hypothesis:"#ded4ff",Assumption:"#eadcff",Commitment:"#ffc4c4",DraftFragment:"#ffd9ad",PlanStep:"#dfe7ef"};
const lanes = ["Source","Evidence","Claim","Hypothesis","Assumption","Commitment","DraftFragment","PlanStep"];
const rewindKinds = new Set(["trigger_fired","rollback","retraction","stale","contested"]);
const label = e => (e.source === "world" ? "WORLD · " : "GRAPH · ") + e.record.event.replaceAll("_"," ");
function showDetail(title, value) {
  document.getElementById("detail-title").textContent = title;
  document.getElementById("details").textContent = JSON.stringify(value, null, 2);
}
function svgElement(name, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg",name);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,String(v));
  return el;
}
function renderGraph(index) {
  const nodes = new Map(), edges = new Map();
  for (const {source,record:r} of events.slice(0,index+1)) {
    if (source !== "graph") continue;
    if (r.event === "node") nodes.set(r.id,{...r,metadata:{...(r.metadata || {})}});
    else if (r.event === "edge") edges.set(r.id,{...r,metadata:{...(r.metadata || {})}});
    else if (r.event === "contested" && nodes.has(r.claim_id)) nodes.get(r.claim_id).metadata.status = "contested";
    else if (r.event === "stale" && nodes.has(r.node_id)) nodes.get(r.node_id).metadata.status = "stale";
    else if (r.event === "retraction") {
      const item = r.kind === "node" ? nodes.get(r.node_id) : edges.get(r.edge_id);
      if (item) item.metadata.status = "retracted";
    }
  }
  const groups = new Map(lanes.map(x => [x,[]]));
  for (const n of nodes.values()) {
    if (!groups.has(n.type)) groups.set(n.type,[]);
    groups.get(n.type).push(n);
  }
  const columns = [...groups].filter(([,items]) => items.length);
  const height = Math.max(220, ...columns.map(([,items]) => 55+items.length*80));
  const svg = svgElement("svg",{width:Math.max(300,columns.length*250),height,role:"img","aria-label":"Epistemic graph"});
  const defs = svgElement("defs",{}), marker = svgElement("marker",{id:"arrow",viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:6,markerHeight:6,orient:"auto-start-reverse"});
  marker.append(svgElement("path",{d:"M 0 0 L 10 5 L 0 10 z",fill:"#8999b0"})); defs.append(marker); svg.append(defs);
  const positions = new Map();
  columns.forEach(([type,items],col) => {
    const heading=svgElement("text",{x:col*250+10,y:22,fill:"#526681","font-weight":"bold","font-size":12});
    heading.textContent=type; svg.append(heading);
    items.forEach((node,row) => positions.set(node.id,{x:col*250+10,y:35+row*80}));
  });
  for (const edge of edges.values()) {
    const a=positions.get(edge.source), b=positions.get(edge.target);
    if (!a || !b) continue;
    const line=svgElement("line",{x1:a.x+105,y1:a.y+28,x2:b.x+105,y2:b.y+28,stroke:edge.type==="contradicts"?"#d74747":"#8999b0","stroke-width":1.5,"marker-end":"url(#arrow)"});
    if (edge.metadata.status==="retracted") line.setAttribute("stroke-dasharray","4 4");
    const title=svgElement("title",{}); title.textContent=edge.type+(edge.metadata.status ? " · "+edge.metadata.status : ""); line.append(title); svg.append(line);
  }
  for (const node of nodes.values()) {
    const p=positions.get(node.id), status=node.metadata.status || "active";
    const group=svgElement("g",{class:"node",tabindex:0,role:"button","aria-label":node.type+" "+node.id});
    const rect=svgElement("rect",{x:p.x,y:p.y,width:210,height:56,rx:7,fill:status==="retracted"||status==="stale"?"#e6e9ef":colors[node.type]||"#e5e7eb",stroke:status==="contested"?"#d74747":"#a8b7c8"});
    if(status==="retracted"||status==="stale") rect.setAttribute("stroke-dasharray","5 3");
    group.append(rect);
    const text=svgElement("text",{x:p.x+8,y:p.y+17,fill:"#1d2b42","font-size":11});
    [node.type+" · "+node.id, String(node.label||node.text||"").replace(/\s+/g," ").slice(0,30), status!=="active"?"["+status+"]":""].forEach((line,i)=>{
      const t=svgElement("tspan",{x:p.x+8,dy:i?15:0});t.textContent=line;text.append(t);
    });
    group.append(text);group.addEventListener("click",()=>showDetail("Node details",node));
    group.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" ")showDetail("Node details",node)});
    svg.append(group);
  }
  document.getElementById("graph").replaceChildren(svg);
  return {nodes:nodes.size,edges:edges.size};
}
function render() {
  const index=Number(range.value), selected=events[index], counts=renderGraph(index);
  document.getElementById("counter").textContent=`${index+1} / ${events.length}`;
  document.getElementById("summary").textContent=`${counts.nodes} nodes · ${counts.edges} edges recorded so far`;
  showDetail(label(selected),selected.record);
  const filter=document.getElementById("filter").value;
  timeline.replaceChildren();
  events.forEach((event,i)=>{
    const isRewind=rewindKinds.has(event.record.event);
    if(filter==="world" && event.source!=="world" && !isRewind) return;
    if(filter==="rewind" && !isRewind) return;
    const b=document.createElement("button");b.className="event "+(isRewind?"rewind":event.source)+(i===index?" active":"");
    b.textContent=`${i+1}. ${label(event)}`; b.onclick=()=>{range.value=i;render()};timeline.append(b);
  });
  timeline.querySelector(".active")?.scrollIntoView({block:"nearest"});
}
const worldEvents=events.filter(e=>e.source==="world");
const predictions=worldEvents.filter(e=>e.record.event==="prediction").length;
const rollbacks=worldEvents.filter(e=>e.record.event==="rollback").length;
const warning=document.getElementById("warning");
if(!worldEvents.length || !predictions || !rollbacks) {
  warning.hidden=false;
  warning.textContent=(!worldEvents.length?"No world-model log supplied. ":!predictions?"No world-model predictions were recorded. ":"")+
    (!rollbacks?"No rollback was recorded. A real run rewinds only if its consistency trigger fires; it is not guaranteed.":"");
}
range.oninput=render; document.getElementById("filter").onchange=render;
document.getElementById("prev").onclick=()=>{range.value=Math.max(0,Number(range.value)-1);render()};
document.getElementById("next").onclick=()=>{range.value=Math.min(events.length-1,Number(range.value)+1);render()};
render();
</script></body></html>"""


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demonstrate a forced commitment and synthetic rollback on a graph snapshot.

The original log is read only. The prediction is a real world-model call, but
the commitment and contradiction are artificial and do not resume the agent.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.configuration import Configuration
from open_deep_research.epistemic_graph import EpistemicGraph
from open_deep_research.world_model import WorldModel
from open_deep_research.world_model_logger import WorldModelRun
from open_deep_research.world_model_monitor import monitor
from open_deep_research.world_model_rollback import apply_repair, reduced_repair
from open_deep_research.world_model_scoring import compute_belief, take_snapshot
from visualize_epistemic_graph import build_graph, load_events
from visualize_replay import make_replay

NOTICE = (
    "SYNTHETIC DEMONSTRATION — This is an isolated graph snapshot, not a continuation "
    "of the live run. The commitment is forced and the contradictory evidence is "
    "fabricated. Only the world-model prediction is a real model response."
)


def copy_context(
    source: Path, target: EpistemicGraph, claim_id: str | None = None, *, question: str = NOTICE
) -> str:
    """Copy a small, active claim neighborhood into an isolated graph."""
    snapshot = build_graph(load_events(str(source)))
    nodes = snapshot["nodes"]
    claims = [
        node for node in nodes.values()
        if node["type"] == "Claim" and node["metadata"].get("status") != "retracted"
    ]
    if claim_id is None:
        if not claims:
            raise ValueError("The snapshot has no active claims yet.")
        claim_id = claims[-1]["id"]
    if claim_id not in {claim["id"] for claim in claims}:
        raise ValueError(f"{claim_id} is not an active claim in the snapshot.")

    active_edges = [
        edge for edge in snapshot["edges"]
        if edge["metadata"].get("status") != "retracted"
    ]
    selected = [claim_id]
    for edge in active_edges:
        other = edge["to"] if edge["from"] == claim_id else (
            edge["from"] if edge["to"] == claim_id else None
        )
        if other and other not in selected and nodes[other]["type"] in {"Evidence", "Hypothesis", "Assumption"}:
            if nodes[other]["metadata"].get("status") != "retracted":
                selected.append(other)
    selected = selected[:12]
    evidence_ids = {node_id for node_id in selected if nodes[node_id]["type"] == "Evidence"}
    for edge in active_edges:
        if edge["from"] in evidence_ids and nodes[edge["to"]]["type"] == "Source":
            if edge["to"] not in selected and nodes[edge["to"]]["metadata"].get("status") != "retracted":
                selected.append(edge["to"])
            if len(selected) >= 20:
                break

    ids: dict[str, str] = {}
    target.set_question(question)
    for original_id in selected:
        node = nodes[original_id]
        metadata = {**node["metadata"], "snapshot_original_id": original_id}
        new_id = target.add_node(node["type"], node["text"], metadata, label=node["label"])
        if not new_id:
            raise RuntimeError(f"Could not copy node {original_id}")
        ids[original_id] = new_id
    for edge in active_edges:
        if edge["from"] in ids and edge["to"] in ids:
            target.add_edge(edge["type"], ids[edge["from"]], ids[edge["to"]], edge["metadata"])
    return ids[claim_id]


def demonstrate(
    source: Path,
    output: Path,
    model_name: str,
    claim_id: str | None = None,
    prediction_client: Any = None,
) -> dict[str, Any]:
    """Run the genuine predictor and monitor around an explicitly forced commitment."""
    output.mkdir(parents=True, exist_ok=False)
    run_id = f"synthetic-demo-{uuid.uuid4()}"
    config = {"configurable": {
        "thread_id": run_id,
        "enable_world_model": True,
        "world_model_model": model_name,
        "wm_enforce_decision": False,
        "wm_enable_rollback": True,
        "wm_enforce_rollback": True,
    }}
    settings = Configuration.from_runnable_config(config)
    graph = EpistemicGraph(run_id, str(output))
    logger = WorldModelRun(run_id, str(output), config, question=NOTICE)
    try:
        target_id = copy_context(source, graph, claim_id)
        world_model = WorldModel(config, prediction_client=prediction_client)
        action = {"action_id": "synthetic-forced-commit", "target_node_ids": [target_id]}
        print("Calling the world model for the isolated snapshot...", flush=True)
        prediction = world_model.predict(graph, action)
        logger.prediction(prediction)
        if prediction.parse_error:
            print("Warning: the world-model response could not be parsed.", file=sys.stderr, flush=True)
        commitment_id = graph.add_commitment(
            "SYNTHETIC: forced commitment for rollback demonstration",
            target=target_id,
            metadata={"synthetic": True, "forced": True, "world_model_decision": prediction.decision},
        )
        if not commitment_id:
            raise RuntimeError("Could not create the synthetic commitment")
        before = compute_belief(take_snapshot(graph), target_id, settings)
        world_model.register_active_commitment(commitment_id, target_id, None, prediction.resolved_theta, 1)
        world_model.active_commitments[commitment_id]["beta_at_commit"] = before
        print(f"Recorded forced commitment; model decision: {prediction.decision}.", flush=True)

        fired = []
        for index in range(1, 21):
            source_id = graph.add_source(
                title=f"SYNTHETIC contradiction source {index} (not real)",
                metadata={"synthetic": True, "reliability": 1.0},
            )
            evidence_id = graph.add_evidence(
                text="SYNTHETIC TEST EVIDENCE: assume the committed claim is contradicted.",
                metadata={"synthetic": True, "iteration": 2},
                sources=[source_id] if source_id else None,
            )
            if not evidence_id or not source_id:
                raise RuntimeError("Could not record synthetic contradiction")
            graph.contradicts(evidence_id, target_id, weight=1.0, synthetic=True, iteration=2)
            fired = monitor(graph, world_model.active_commitments, settings)
            if fired:
                break
        logger.monitor_check({"step": 2, "n_active": 1, "n_fired": len(fired), "synthetic": True})
        if not fired:
            raise RuntimeError("The monitor did not fire after 20 synthetic contradictions; no rollback was fabricated.")
        item = fired[0]
        repair = reduced_repair(graph, item, settings)
        payload = {
            "step": 2, "synthetic": True, "forced_commitment": True,
            "commitment_id": commitment_id, "claim_id": target_id,
            "beta_before": before, "beta_after": item["detail"]["beta_now"],
            "offending_evidence_ids": item["detail"]["offending_evidence_ids"],
            "contested_claims": repair.contested_claims, "retract_nodes": repair.retract_nodes,
            "retract_edges": repair.retract_edges, "regenerate_nodes": repair.regenerate_nodes,
            "preserved_nodes": repair.preserved_nodes, "reopen_claims": repair.reopen_claims,
            "enforced": True,
        }
        apply_repair(graph, repair, settings)
        world_model.unregister_active_commitment(commitment_id)
        logger.trigger_fired(payload)
        logger.rollback(payload)
        logger.finish()
        graph.finish({"synthetic_demo": True})
        replay = make_replay(Path(graph.graph_path), Path(logger.log_path), output / "replay.html", notice=NOTICE)
        return {
            "replay": replay, "graph": Path(graph.graph_path), "world_log": Path(logger.log_path),
            "decision": prediction.decision, "parse_error": prediction.parse_error,
            "beta_before": before, "beta_after": item["detail"]["beta_now"],
            "contradictions": index,
        }
    except Exception:
        logger.finish()
        graph.finish({"synthetic_demo": True, "status": "failed"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="Existing epistemic graph JSONL (read only)")
    parser.add_argument("--model", default="openai:qwen/qwen3.5-27b")
    parser.add_argument("--claim-id", help="Original active claim id (default: latest)")
    parser.add_argument("--output-dir", type=Path, help="New directory for isolated demo outputs")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    output = args.output_dir or Path("reports") / f"synthetic-demo-{uuid.uuid4().hex[:8]}"
    result = demonstrate(args.graph, output, args.model, args.claim_id)
    print(f"Replay: {result['replay']}")
    print(f"Graph log: {result['graph']}")
    print(f"World-model log: {result['world_log']}")
    print(f"Belief: {result['beta_before']:.3f} -> {result['beta_after']:.3f}")


if __name__ == "__main__":
    main()

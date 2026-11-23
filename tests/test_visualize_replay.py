import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_research
import visualize_epistemic_graph
import visualize_replay


def write_events(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_replay_orders_events_and_escapes_embedded_text(tmp_path):
    graph = tmp_path / "graph.jsonl"
    world = tmp_path / "world.jsonl"
    write_events(graph, [
        {"event": "run_start", "run_id": "one", "ts": "2026-01-01T00:00:00+00:00"},
        {"event": "node", "run_id": "one", "ts": "2026-01-01T00:00:02+00:00",
         "id": "c1", "type": "Claim", "text": "</script><script>alert(1)</script>"},
        {"event": "node", "run_id": "one", "ts": "2026-01-01T00:00:03+00:00",
         "id": "k1", "type": "Commitment", "text": "Commit"},
        {"event": "edge", "run_id": "one", "ts": "2026-01-01T00:00:04+00:00",
         "id": "e1", "source": "k1", "target": "c1", "type": "locks_in"},
        {"event": "contested", "run_id": "one", "ts": "2026-01-01T00:00:05+00:00",
         "claim_id": "c1"},
        {"event": "retraction", "run_id": "one", "ts": "2026-01-01T00:00:06+00:00",
         "kind": "node", "node_id": "k1"},
        {"event": "retraction", "run_id": "one", "ts": "2026-01-01T00:00:07+00:00",
         "kind": "edge", "edge_id": "e1"},
    ])
    write_events(world, [
        {"event": "prediction", "run_id": "one", "ts": "2026-01-01T00:00:01+00:00",
         "prediction": {"decision": "commit"}},
        {"event": "rollback", "run_id": "one", "ts": "2026-01-01T00:00:08+00:00",
         "rollback": {"enforced": True, "retract_nodes": ["k1"]}},
    ])
    output = visualize_replay.make_replay(graph, world, tmp_path / "replay.html")
    html = output.read_text(encoding="utf-8")
    assert html.index('"event": "prediction"') < html.index('"event": "node"')
    assert "</script><script>alert(1)</script>" not in html
    assert r"\u003c/script>" in html


def test_replay_rejects_mismatched_and_malformed_logs(tmp_path):
    graph = tmp_path / "graph.jsonl"
    world = tmp_path / "world.jsonl"
    write_events(graph, [{"event": "run_start", "run_id": "one", "ts": "2026-01-01"}])
    write_events(world, [{"event": "run_start", "run_id": "two", "ts": "2026-01-01"}])
    with pytest.raises(ValueError, match="same single run_id"):
        visualize_replay.make_replay(graph, world, tmp_path / "out.html")
    world.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        visualize_replay.make_replay(graph, world, tmp_path / "out.html")


def test_final_graph_replays_rollback_status():
    graph = visualize_epistemic_graph.build_graph([
        {"event": "node", "id": "c1", "type": "Claim"},
        {"event": "node", "id": "k1", "type": "Commitment"},
        {"event": "edge", "id": "e1", "source": "k1", "target": "c1", "type": "locks_in"},
        {"event": "contested", "claim_id": "c1", "reason": "contradiction"},
        {"event": "retraction", "kind": "node", "node_id": "k1", "reason": "repair"},
        {"event": "retraction", "kind": "edge", "edge_id": "e1", "reason": "repair"},
    ])
    assert graph["nodes"]["c1"]["metadata"]["status"] == "contested"
    assert graph["nodes"]["k1"]["metadata"]["status"] == "retracted"
    assert graph["edges"][0]["metadata"]["status"] == "retracted"
    html = visualize_epistemic_graph.to_html(graph)
    assert "status: retracted" in html
    assert '"dashes": true' in html


def test_model_flag_sets_all_roles_and_allows_overrides():
    args = run_research.parse_args([
        "question", "--model", "openai:qwen/qwen3.5-27b",
        "--world-model-model", "openai:other", "--max-tool-calls", "3",
    ])
    config = run_research._build_config(args)["configurable"]
    for role in ("research_model", "summarization_model", "compression_model", "final_report_model"):
        assert config[role] == "openai:qwen/qwen3.5-27b"
    assert config["world_model_model"] == "openai:other"
    assert config["max_react_tool_calls"] == 3

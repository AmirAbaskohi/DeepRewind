import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from demo_illustrative_rewind import demonstrate, generate_scenario


class ExampleModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({
                "claim": "This household's contract credits exported energy at retail rates.",
                "hypothesis": "Retail-rate export credits support a shorter payback.",
                "first_query": "household export compensation",
                "first_finding": "The initial contract summary describes retail-rate credits.",
                "second_query": "verify household export compensation",
                "second_finding": "The signed contract instead sets export credits below retail rates.",
            })
        return json.dumps({
            "update_nodes": [], "update_edges": [],
            "reversibility": {"kappa": 0.1, "lambda": 0.1, "gamma": 0.1},
            "trigger_coverage": 0.9,
        })


def test_illustrative_replay_uses_generated_notes_and_monitor(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_WORLD_MODEL", raising=False)
    graph = tmp_path / "source.jsonl"
    events = [
        {"event": "run_start", "run_id": "original", "ts": "2026-01-01"},
        {"event": "node", "run_id": "original", "ts": "2026-01-02", "id": "clm1",
         "type": "Claim", "text": "Solar policy", "label": "Solar policy", "metadata": {}},
    ]
    graph.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    original = graph.read_bytes()
    model = ExampleModel()
    result = demonstrate(graph, tmp_path / "example", "openai:test", model)
    assert model.calls == 2
    assert graph.read_bytes() == original
    assert result["beta_after"] < result["beta_before"]
    html = result["replay"].read_text()
    assert "not actual search results" in html
    assert "signed contract instead" in html
    assert "SYNTHETIC" not in html
    with result["world_log"].open() as handle:
        rows = [json.loads(line) for line in handle]
    assert [r["event"] for r in rows] == [
        "run_start", "prediction", "monitor_check", "trigger_fired", "rollback", "run_end",
    ]


def test_illustrative_scenario_rejects_missing_fields():
    class Incomplete:
        def invoke(self, messages):
            return '{"claim": "only a claim"}'

    with pytest.raises(ValueError, match="required text field"):
        generate_scenario(Incomplete(), "context")

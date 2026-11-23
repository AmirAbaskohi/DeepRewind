import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from demo_forced_rewind import demonstrate


class FakeModel:
    def invoke(self, messages):
        return json.dumps({
            "update_nodes": [],
            "update_edges": [],
            "reversibility": {"kappa": 0.1, "lambda": 0.1, "gamma": 0.1},
            "trigger_coverage": 0.9,
        })


def test_forced_demo_uses_snapshot_and_real_monitor(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_WORLD_MODEL", raising=False)
    graph = tmp_path / "source.jsonl"
    events = [
        {"event": "run_start", "run_id": "original", "ts": "2026-01-01"},
        {"event": "node", "run_id": "original", "ts": "2026-01-02", "id": "clm1",
         "type": "Claim", "text": "Claim", "label": "Claim", "metadata": {}},
    ]
    graph.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    original = graph.read_bytes()
    result = demonstrate(graph, tmp_path / "demo", "openai:test", prediction_client=FakeModel())
    assert graph.read_bytes() == original
    with result["world_log"].open() as handle:
        rows = [json.loads(line) for line in handle]
    assert [row["event"] for row in rows] == [
        "run_start", "prediction", "monitor_check", "trigger_fired", "rollback", "run_end",
    ]
    assert rows[-2]["rollback"]["synthetic"] is True
    assert result["beta_after"] < result["beta_before"]
    html = result["replay"].read_text()
    assert "SYNTHETIC DEMONSTRATION" in html
    assert "forced commitment" in html
    with pytest.raises(FileExistsError):
        demonstrate(graph, tmp_path / "demo", "openai:test", prediction_client=FakeModel())

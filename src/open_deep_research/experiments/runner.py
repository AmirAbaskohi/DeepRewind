"""Adapter for running one deep-research execution with isolated artifacts."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class RunResult:
    status: str
    final_answer: str
    run_dir: str
    research_log_path: Optional[str]
    epistemic_log_path: Optional[str]
    world_model_log_path: Optional[str]
    wall_clock_s: float
    llm_calls: Optional[int] = None
    token_counts: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def _latest_jsonl(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    files = sorted(path.glob("*.jsonl"))
    if not files:
        return None
    return str(files[-1])


def _build_config(
    *,
    base_config: dict[str, Any],
    run_id: str,
    run_dir: Path,
    seed: int,
    injected_evidence: Optional[list[dict[str, Any]]],
    switch_directive: Optional[dict[str, Any]],
) -> dict[str, Any]:
    cfg = {"configurable": dict((base_config or {}).get("configurable", {}) or {})}
    seed_condition = cfg["configurable"].get("experiment_seed_condition", "seed")
    cfg["configurable"].update(
        {
            "thread_id": run_id,
            "research_log_dir": str(run_dir / "research_logs"),
            "epistemic_graph_dir": str(run_dir / "epistemic_graphs"),
            "world_model_log_dir": str(run_dir / "world_model_logs"),
            "experiment_seed_injection": bool(injected_evidence),
            "experiment_injected_evidence": list(injected_evidence or []),
            "experiment_switch_enabled": bool(switch_directive),
            "experiment_switch_directive": dict(switch_directive or {}),
            "experiment_seed_condition": str(seed_condition),
            "random_seed": int(seed),
        }
    )
    return cfg


def run_once(
    question: str,
    config: dict[str, Any],
    run_dir: str,
    seed: int,
    preamble: Optional[str] = None,
    injected_evidence: Optional[list[dict[str, Any]]] = None,
    switch_directive: Optional[dict[str, Any]] = None,
) -> RunResult:
    """Run one complete deep-research execution using existing graph entrypoints."""
    from langgraph.checkpoint.memory import MemorySaver

    from open_deep_research.deep_researcher import deep_researcher_builder

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "research_logs").mkdir(parents=True, exist_ok=True)
    (run_path / "epistemic_graphs").mkdir(parents=True, exist_ok=True)
    (run_path / "world_model_logs").mkdir(parents=True, exist_ok=True)

    run_id = f"exp-{uuid.uuid4().hex}"
    runnable_config = _build_config(
        base_config=config,
        run_id=run_id,
        run_dir=run_path,
        seed=seed,
        injected_evidence=injected_evidence,
        switch_directive=switch_directive,
    )

    content = question.strip()
    if preamble and preamble.strip():
        content = f"{preamble.strip()}\n\nPrimary research question:\n{question.strip()}"

    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())

    started = time.perf_counter()
    try:
        final_state = asyncio.run(
            graph.ainvoke(
                {"messages": [{"role": "user", "content": content}]},
                runnable_config,
            )
        )
        wall_clock_s = time.perf_counter() - started
        final_answer = str(final_state.get("final_report", "") or "")
        status = "ok"
        error = None
    except Exception as exc:
        wall_clock_s = time.perf_counter() - started
        final_answer = ""
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    result = RunResult(
        status=status,
        final_answer=final_answer,
        run_dir=str(run_path),
        research_log_path=_latest_jsonl(run_path / "research_logs"),
        epistemic_log_path=_latest_jsonl(run_path / "epistemic_graphs"),
        world_model_log_path=_latest_jsonl(run_path / "world_model_logs"),
        wall_clock_s=wall_clock_s,
        llm_calls=None,
        token_counts=None,
        error=error,
    )

    (run_path / "run_result.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if status == "ok":
        (run_path / "_COMPLETE").write_text("ok\n", encoding="utf-8")
    return result

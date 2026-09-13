#!/usr/bin/env python3
"""Run one research question from the console with DeepRewind enabled.

This is a convenience wrapper around ``open_deep_research.deep_researcher``
for ad-hoc use: it runs a single question end-to-end with epistemic-graph
recording, research-state logging, and world-model commitment control turned
on, prints the final report, and (by default) renders the resulting graphs
via the existing ``visualize_epistemic_graph.py`` / ``visualize_research_graph.py``
scripts.

Usage examples
---------------
    # Plain run, world model on (default), graphs rendered automatically
    python scripts/run_research.py "What are the tradeoffs of nuclear vs. renewables for net-zero grids?"

    # Predict-only (shadow) mode: log world-model predictions but never block
    # or roll back a commitment
    python scripts/run_research.py "..." --shadow

    # Disable the world model entirely (plain Open Deep Research behavior)
    python scripts/run_research.py "..." --disable-world-model

    # Skip graph rendering / browser opening
    python scripts/run_research.py "..." --no-visualize
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Make the local package and sibling visualization scripts importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_SCRIPTS = Path(__file__).resolve().parent
for _path in (_SRC, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a LangGraph ``configurable`` dict."""
    enable_world_model = not args.disable_world_model
    enable_rollback = enable_world_model and not args.disable_rollback and not args.shadow
    configurable: dict[str, Any] = {
        "thread_id": args.thread_id or str(uuid.uuid4()),
        "search_api": args.search_api,
        "allow_clarification": args.allow_clarification,
        "max_researcher_iterations": args.max_iterations,
        "max_concurrent_research_units": args.max_concurrent_units,
        "enable_research_logging": True,
        "research_log_dir": args.research_log_dir,
        "enable_epistemic_graph": True,
        "epistemic_graph_dir": args.epistemic_graph_dir,
        "enable_world_model": enable_world_model,
        "world_model_log_dir": args.world_model_log_dir,
        "wm_gate_mode": args.gate_mode,
        "wm_enforce_decision": enable_world_model and not args.shadow,
        "wm_enable_rollback": enable_rollback,
        "wm_enforce_rollback": enable_rollback,
        "wm_recovery_mode": "rollback" if enable_rollback else "none",
    }
    if args.world_model_model:
        configurable["world_model_model"] = args.world_model_model
    for flag_name, config_key in (
        ("research_model", "research_model"),
        ("final_report_model", "final_report_model"),
        ("compression_model", "compression_model"),
        ("summarization_model", "summarization_model"),
    ):
        value = getattr(args, flag_name)
        if value:
            configurable[config_key] = value
    return {"configurable": configurable}


async def _run(question: str, config: dict[str, Any]) -> dict[str, Any]:
    from langgraph.checkpoint.memory import MemorySaver

    from open_deep_research.deep_researcher import deep_researcher_builder

    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    return await graph.ainvoke({"messages": [{"role": "user", "content": question}]}, config)


def _artifact_paths(config: dict[str, Any]) -> dict[str, Optional[str]]:
    """Look up the log/graph files this run produced via their registries."""
    from open_deep_research.epistemic_graph import get_epistemic_graph
    from open_deep_research.research_logger import get_logger
    from open_deep_research.world_model_logger import get_world_model_logger

    return {
        "research_log": getattr(get_logger(config), "log_path", None),
        "epistemic_graph": getattr(get_epistemic_graph(config), "graph_path", None),
        "world_model_log": getattr(get_world_model_logger(config), "log_path", None),
    }


def _render_graphs(paths: dict[str, Optional[str]], open_browser: bool) -> None:
    import visualize_epistemic_graph
    import visualize_research_graph

    epistemic_path = paths.get("epistemic_graph")
    if epistemic_path:
        argv = [epistemic_path, "--format", "html"]
        if not open_browser:
            argv.append("--no-open")
        visualize_epistemic_graph.main(argv)

    research_path = paths.get("research_log")
    if research_path:
        argv = [research_path, "--format", "html"]
        if open_browser:
            argv.append("--open")
        visualize_research_graph.main(argv)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", help="The research question to investigate.")
    parser.add_argument("--thread-id", default=None, help="Explicit LangGraph thread id (default: random uuid).")
    parser.add_argument("--search-api", default="tavily", choices=["tavily", "openai", "anthropic", "none"])
    parser.add_argument("--allow-clarification", action="store_true", help="Let the agent pause to ask a clarifying question before researching.")
    parser.add_argument("--max-iterations", type=int, default=6, help="Max supervisor research iterations.")
    parser.add_argument("--max-concurrent-units", type=int, default=3, help="Max concurrent research sub-agents.")
    parser.add_argument("--research-model", default=None)
    parser.add_argument("--final-report-model", default=None)
    parser.add_argument("--compression-model", default=None)
    parser.add_argument("--summarization-model", default=None)
    parser.add_argument("--world-model-model", default=None, help="Model used for world-model commitment prediction.")
    parser.add_argument("--gate-mode", default="threshold", choices=["threshold", "utility"], help="Commitment gate policy.")
    parser.add_argument("--disable-world-model", action="store_true", help="Run plain Open Deep Research (no commitment prediction/control).")
    parser.add_argument("--shadow", action="store_true", help="Predict-only mode: log world-model predictions but never block a commitment or roll back.")
    parser.add_argument("--disable-rollback", action="store_true", help="Gate commitments but never monitor/roll back stranded ones.")
    parser.add_argument("--research-log-dir", default="research_logs")
    parser.add_argument("--epistemic-graph-dir", default="epistemic_graphs")
    parser.add_argument("--world-model-log-dir", default="world_model_logs")
    parser.add_argument("--output-dir", default="reports", help="Directory to save the final report markdown.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip rendering HTML graphs after the run.")
    parser.add_argument("--open", action="store_true", help="Open the rendered HTML graphs in a browser.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(_REPO_ROOT / ".env")
    args = parse_args(argv)
    config = _build_config(args)

    print(f"Researching: {args.question!r}")
    print(f"World model: {'disabled' if args.disable_world_model else ('shadow' if args.shadow else args.gate_mode)}")
    result = asyncio.run(_run(args.question, config))

    final_report = result.get("final_report", "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{config['configurable']['thread_id']}.md"
    report_path.write_text(final_report, encoding="utf-8")

    print("\n" + "=" * 80)
    print(final_report)
    print("=" * 80)
    print(f"\nSaved report to: {report_path}")

    paths = _artifact_paths(config)
    for label, path in paths.items():
        if path:
            print(f"{label}: {path}")

    if not args.no_visualize:
        _render_graphs(paths, open_browser=args.open)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

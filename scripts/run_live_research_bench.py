"""Run the Open Deep Research agent over the LiveResearchBench dataset.

This script loads the `Salesforce/LiveResearchBench` dataset from the Hugging
Face Hub, passes *only the questions* to the deep research agent, and saves each
run's output. It performs **no evaluation** - it simply collects the agent's
final report together with the research-graph and epistemic-graph artifacts that
the agent produces while it works.

Dataset: https://huggingface.co/datasets/Salesforce/LiveResearchBench

The dataset is gated, so you must accept its terms on the Hub and be logged in
(``huggingface-cli login`` or the ``HF_TOKEN`` environment variable) before
running this script.

For every question the script writes, under ``--output-dir``::

    <output-dir>/
        summary.jsonl                     # one line per completed question
        <qid>/
            question.txt                  # the question passed to the agent
            report.md                     # the agent's final report
            result.json                   # qid, question, report + metadata
            research_logs/                # research_graph_*.jsonl (+ .html/.dot/.mmd)
            epistemic_graphs/             # epistemic_graph_*.jsonl (+ .html/.dot/.mmd)

Example
-------
    python scripts/run_live_research_bench.py --limit 5

    python scripts/run_live_research_bench.py \
        --subset question_only \
        --output-dir liveresearchbench_outputs \
        --max-concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper.
    tqdm = None

# Make the local package importable when running the script directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from open_deep_research.deep_researcher import deep_researcher_builder  # noqa: E402

DATASET_ID = "Salesforce/LiveResearchBench"

# Default agent configuration. These mirror the defaults used in the evaluation
# harness (tests/run_evaluate.py) so behavior is consistent, but they can be
# overridden from the command line.
DEFAULT_CONFIG: Dict[str, Any] = {
    "max_structured_output_retries": 3,
    # Never pause to ask the user for clarification - we run unattended.
    "allow_clarification": False,
    "max_concurrent_research_units": 3,
    "search_api": "tavily",
    "max_researcher_iterations": 3,
    "max_react_tool_calls": 5,
    "summarization_model": "openai:gpt-4o-mini",
    "summarization_model_max_tokens": 8192,
    "research_model": "openai:gpt-4o",
    "research_model_max_tokens": 10000,
    "compression_model": "openai:gpt-4o",
    "compression_model_max_tokens": 10000,
    "final_report_model": "openai:gpt-4o",
    "final_report_model_max_tokens": 10000,
    # Make sure both instrumentation graphs are recorded.
    "enable_research_logging": True,
    "enable_epistemic_graph": True,
}


def _sanitize(value: str) -> str:
    """Make a string safe to use as a directory name."""
    keep = "-_."
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in value)
    return cleaned[:80] or "item"


def load_questions(
    subset: str,
    split: Optional[str],
    limit: Optional[int],
    start_index: int,
) -> List[Dict[str, Any]]:
    """Load (qid, question) pairs from the LiveResearchBench dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "The 'datasets' library is required to load LiveResearchBench.\n"
            "Install it with:  pip install datasets"
        ) from exc

    try:
        dataset = load_dataset(DATASET_ID, subset)
    except Exception as exc:  # gated access / auth problems land here
        raise SystemExit(
            f"Failed to load '{DATASET_ID}': {exc}\n\n"
            "This dataset is gated. Accept its terms on the Hub and log in:\n"
            f"  1. Visit https://huggingface.co/datasets/{DATASET_ID} and request access\n"
            "  2. Authenticate:  huggingface-cli login   (or set HF_TOKEN)"
        ) from exc

    # `load_dataset` returns a DatasetDict; pick the requested split or the
    # first available one.
    if split is not None:
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise SystemExit(
                f"Split '{split}' not found. Available splits: {available}"
            )
        data = dataset[split]
    else:
        first_split = next(iter(dataset.keys()))
        data = dataset[first_split]

    rows = data.select(range(start_index, len(data)))
    items: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        question = row.get("question")
        if not question:
            continue
        qid = row.get("qid") or f"item_{start_index + i:04d}"
        items.append({"qid": str(qid), "question": str(question)})
        if limit is not None and len(items) >= limit:
            break
    return items


def build_config(
    thread_id: str,
    research_log_dir: str,
    epistemic_graph_dir: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a LangGraph RunnableConfig for a single research run."""
    configurable: Dict[str, Any] = dict(DEFAULT_CONFIG)
    configurable.update(overrides)
    configurable["thread_id"] = thread_id
    # Route this run's instrumentation into its own per-question directories.
    configurable["research_log_dir"] = research_log_dir
    configurable["epistemic_graph_dir"] = epistemic_graph_dir
    return {"configurable": configurable}


def _visualize(graph_dir: Path, kind: str) -> None:
    """Render .html/.dot/.mmd files for every .jsonl graph in ``graph_dir``."""
    if kind == "research":
        script = _REPO_ROOT / "scripts" / "visualize_research_graph.py"
        extra = ["--format", "all"]
    else:
        script = _REPO_ROOT / "scripts" / "visualize_epistemic_graph.py"
        extra = ["--format", "html,dot,mermaid", "--no-open"]

    for jsonl in sorted(graph_dir.glob("*.jsonl")):
        try:
            subprocess.run(
                [sys.executable, str(script), str(jsonl), *extra],
                cwd=str(_REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:  # pragma: no cover - visualization is best effort
            print(f"    ! Failed to visualize {jsonl.name}: {exc}", file=sys.stderr)


async def run_one(
    item: Dict[str, Any],
    output_dir: Path,
    overrides: Dict[str, Any],
    visualize: bool,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Run the agent on a single question and persist all artifacts."""
    async with semaphore:
        qid = item["qid"]
        question = item["question"]
        item_dir = output_dir / _sanitize(qid)
        research_dir = item_dir / "research_logs"
        epistemic_dir = item_dir / "epistemic_graphs"
        item_dir.mkdir(parents=True, exist_ok=True)

        thread_id = str(uuid.uuid4())
        config = build_config(
            thread_id=thread_id,
            research_log_dir=str(research_dir),
            epistemic_graph_dir=str(epistemic_dir),
            overrides=overrides,
        )

        (item_dir / "question.txt").write_text(question, encoding="utf-8")

        print(f"[{qid}] running (thread={thread_id})...", flush=True)
        started = datetime.now(timezone.utc)

        graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
        record: Dict[str, Any] = {
            "qid": qid,
            "question": question,
            "thread_id": thread_id,
            "started_at": started.isoformat(),
        }

        try:
            final_state = await graph.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config,
            )
            report = final_state.get("final_report", "")
            (item_dir / "report.md").write_text(report or "", encoding="utf-8")
            record["status"] = "ok"
            record["report"] = report
            record["report_chars"] = len(report or "")
        except Exception as exc:  # keep going on failures
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{qid}] ERROR: {record['error']}", file=sys.stderr, flush=True)

        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["config"] = config["configurable"]

        (item_dir / "result.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        if visualize:
            _visualize(research_dir, "research")
            _visualize(epistemic_dir, "epistemic")

        print(f"[{qid}] done ({record['status']}).", flush=True)
        return record


async def run_all(args: argparse.Namespace) -> None:
    """Load the dataset and run the agent over every question."""
    overrides: Dict[str, Any] = {}
    if args.search_api is not None:
        overrides["search_api"] = args.search_api
    if args.research_model is not None:
        overrides["research_model"] = args.research_model
    if args.max_concurrent_research_units is not None:
        overrides["max_concurrent_research_units"] = args.max_concurrent_research_units

    items = load_questions(
        subset=args.subset,
        split=args.split,
        limit=args.limit,
        start_index=args.start_index,
    )
    if not items:
        raise SystemExit("No questions loaded from the dataset.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"

    print(
        f"Loaded {len(items)} question(s) from {DATASET_ID} "
        f"[subset={args.subset}]. Writing to {output_dir}",
        flush=True,
    )

    semaphore = asyncio.Semaphore(args.max_concurrency)
    tasks = [
        run_one(item, output_dir, overrides, not args.no_visualize, semaphore)
        for item in items
    ]

    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(tasks), desc="LiveResearchBench", unit="q")
    completed = ok = errored = 0

    with summary_path.open("a", encoding="utf-8") as summary_fh:
        for coro in asyncio.as_completed(tasks):
            record = await coro
            slim = {k: v for k, v in record.items() if k != "report"}
            summary_fh.write(json.dumps(slim, ensure_ascii=False, default=str) + "\n")
            summary_fh.flush()

            completed += 1
            if record.get("status") == "ok":
                ok += 1
            else:
                errored += 1
            if progress is not None:
                progress.set_postfix(ok=ok, errored=errored)
                progress.update(1)
            else:
                print(
                    f"Progress: {completed}/{len(tasks)} (ok={ok}, errored={errored})",
                    flush=True,
                )

    if progress is not None:
        progress.close()

    print(f"\nAll done. Summary: {summary_path}", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Open Deep Research agent over the LiveResearchBench "
            "dataset and save outputs + graphs (no evaluation)."
        ),
    )
    parser.add_argument(
        "--subset",
        default="question_only",
        choices=["question_only", "question_with_checklist"],
        help="Dataset subset to load (default: question_only).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to use (default: first available split).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to run (default: all).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index of the first dataset row to include (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        default="liveresearchbench_outputs",
        help="Directory to write outputs into (default: liveresearchbench_outputs).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of questions to run in parallel (default: 1).",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Do not render .html/.dot/.mmd files from the graph .jsonl logs.",
    )
    # Optional agent overrides.
    parser.add_argument(
        "--search-api",
        default=None,
        help="Override the search API (e.g. tavily, openai, anthropic, none).",
    )
    parser.add_argument(
        "--research-model",
        default=None,
        help="Override the research model (e.g. openai:gpt-5).",
    )
    parser.add_argument(
        "--max-concurrent-research-units",
        type=int,
        default=None,
        help="Override the max concurrent research units per question.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Script entry point."""
    load_dotenv(_REPO_ROOT / ".env")
    args = parse_args(argv)
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()

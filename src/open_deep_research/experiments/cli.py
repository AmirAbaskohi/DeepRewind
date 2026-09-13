"""CLI for running and analyzing Phase 6 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from open_deep_research.experiments.analyze import analyze_experiment
from open_deep_research.experiments.config import ExperimentSpec, load_alternatives, load_questions
from open_deep_research.experiments.orchestrate import run_experiment


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _run(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions)
    alternatives = load_alternatives(args.alternatives)

    families = _split_csv(args.families) or ["initial_condition", "switching"]
    arms = _split_csv(args.arms) or ["base", "wm_shadow", "wm_gate", "wm_rollback"]
    seeds = [int(value) for value in (_split_csv(args.seeds) or ["0"])]

    initial_conditions = ["neutral", "counter", "distract", "support"]
    switching_conditions = ["noswitch", "switch"]
    if args.conditions:
        requested = _split_csv(args.conditions)
        initial_conditions = [c for c in requested if c in {"neutral", "counter", "distract", "support"}] or initial_conditions
        switching_conditions = [c for c in requested if c in {"noswitch", "switch"}] or switching_conditions

    spec = ExperimentSpec(
        questions=questions,
        output_root=args.root,
        families=families,
        initial_conditions=initial_conditions,
        switching_conditions=switching_conditions,
        arms=arms,
        seeds=seeds,
        alternatives_by_question=alternatives,
        enable_concurrency=False,
        max_concurrency=1,
    )

    result = run_experiment(spec)
    print(json.dumps({
        "root": result.root,
        "manifest": result.manifest_path,
        "n_total": result.n_total,
        "n_ok": result.n_ok,
        "n_error": result.n_error,
        "n_skipped": result.n_skipped,
    }, indent=2))


def _analyze(args: argparse.Namespace) -> None:
    output = analyze_experiment(args.root)
    print(json.dumps({
        "manifest_rows": int(len(output.manifest_df)),
        "summary_jsonl": output.summary_jsonl,
        "summary_md": output.summary_md,
    }, indent=2))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 experiment harness CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run", help="Run experiment grid")
    run_parser.add_argument("--questions", required=True, help="Path to .jsonl or .txt question list")
    run_parser.add_argument("--root", required=True, help="Experiment output root")
    run_parser.add_argument("--families", default="initial_condition,switching", help="CSV families")
    run_parser.add_argument("--conditions", default="", help="CSV conditions")
    run_parser.add_argument("--arms", default="base,wm_shadow,wm_gate,wm_rollback", help="CSV arms")
    run_parser.add_argument("--seeds", default="0", help="CSV integer seeds")
    run_parser.add_argument("--alternatives", default=None, help="Optional alternatives file (.json/.jsonl)")
    run_parser.set_defaults(func=_run)

    analyze_parser = sub.add_parser("analyze", help="Analyze an existing experiment root")
    analyze_parser.add_argument("--root", required=True, help="Experiment output root")
    analyze_parser.set_defaults(func=_analyze)

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""Experiment orchestrator for Phase 6 study families."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from open_deep_research.configuration import Configuration
from open_deep_research.experiments.config import ExperimentResult, ExperimentRun, ExperimentSpec, arm_to_config
from open_deep_research.experiments.runner import RunResult, run_once
from open_deep_research.experiments.seed_initial_condition import build_seed, elicit_target_proposition
from open_deep_research.experiments.switching import build_switch_setup, elicit_alternatives


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conditions_for_family(spec: ExperimentSpec, family: str) -> list[str]:
    if family == "initial_condition":
        return list(spec.initial_conditions)
    if family == "switching":
        return list(spec.switching_conditions)
    raise ValueError(f"Unknown family: {family}")


def _question_grid(spec: ExperimentSpec) -> list[tuple[dict[str, str], str, str, str, int]]:
    grid: list[tuple[dict[str, str], str, str, str, int]] = []
    for question in spec.questions:
        for family in spec.families:
            for condition in _conditions_for_family(spec, family):
                for arm in spec.arms:
                    for seed in spec.seeds:
                        grid.append((question, family, condition, arm, seed))
    return grid


def run_experiment(
    spec: ExperimentSpec,
    base_config: Optional[dict[str, Any]] = None,
    runner_fn: Callable[..., RunResult] = run_once,
) -> ExperimentResult:
    """Run the full question x family x condition x arm x seed grid."""
    if spec.enable_concurrency:
        # Concurrency is intentionally disabled by default in this phase.
        raise NotImplementedError("Concurrent experiment execution is not enabled in this phase.")

    root = Path(spec.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"

    base = {"configurable": dict((base_config or {}).get("configurable", {}) or {})}
    settings = Configuration.from_runnable_config(base)

    n_total = n_ok = n_error = n_skipped = 0

    for question, family, condition, arm, seed in _question_grid(spec):
        n_total += 1
        qid = str(question["id"])
        qtext = str(question["question"])
        run_dir = root / family / qid / condition / arm / f"seed{seed}"
        run = ExperimentRun(
            question_id=qid,
            question=qtext,
            family=family,
            condition=condition,
            arm=arm,
            seed=seed,
            run_dir=str(run_dir),
        )

        complete_marker = run_dir / "_COMPLETE"
        if complete_marker.exists():
            n_skipped += 1
            record = {
                "ts": _now_iso(),
                "status": "skipped",
                "reason": "resume_completed",
                "run": asdict(run),
            }
            with manifest_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue

        cfg = arm_to_config(arm, base)
        cfg["configurable"]["experiment_seed_injection"] = False
        cfg["configurable"]["experiment_switch_enabled"] = False
        cfg["configurable"]["experiment_switch_after"] = "first_commitment"
        cfg["configurable"]["experiment_switch_n"] = 1

        preamble = ""
        injected_evidence: list[dict[str, Any]] = []
        switch_directive: Optional[dict[str, Any]] = None
        extra: dict[str, Any] = {}

        try:
            if family == "initial_condition":
                target = elicit_target_proposition(qtext, cfg)
                preamble, injected_evidence = build_seed(qtext, condition, target)
                cfg["configurable"]["experiment_seed_condition"] = condition
                cfg["configurable"]["experiment_seed_injection"] = bool(injected_evidence)
                extra["target_proposition"] = target

            elif family == "switching":
                alternatives = spec.alternatives_by_question.get(qid)
                if not alternatives:
                    alternatives = elicit_alternatives(qtext, cfg, k=2)
                preamble, injected_evidence, switch_directive = build_switch_setup(
                    qtext,
                    condition,
                    alternatives,
                    theta_rho_star=float(settings.wm_theta_rho_star),
                    theta_w_star=float(settings.wm_theta_w_star),
                )
                cfg["configurable"]["experiment_seed_condition"] = condition
                cfg["configurable"]["experiment_seed_injection"] = bool(injected_evidence)
                cfg["configurable"]["experiment_switch_enabled"] = bool(switch_directive)
                cfg["configurable"]["experiment_switch_directive"] = dict(switch_directive or {})
                extra["alternatives"] = alternatives

            result = runner_fn(
                question=qtext,
                config=cfg,
                run_dir=str(run_dir),
                seed=seed,
                preamble=preamble,
                injected_evidence=injected_evidence,
                switch_directive=switch_directive,
            )

            if result.status == "ok":
                n_ok += 1
            else:
                n_error += 1

            record = {
                "ts": _now_iso(),
                "status": result.status,
                "run": asdict(run),
                "config_summary": {
                    "enable_world_model": cfg["configurable"].get("enable_world_model"),
                    "wm_enforce_decision": cfg["configurable"].get("wm_enforce_decision"),
                    "wm_enable_rollback": cfg["configurable"].get("wm_enable_rollback"),
                    "wm_enforce_rollback": cfg["configurable"].get("wm_enforce_rollback"),
                    "wm_recovery_mode": cfg["configurable"].get("wm_recovery_mode"),
                    "experiment_seed_injection": cfg["configurable"].get("experiment_seed_injection"),
                    "experiment_switch_enabled": cfg["configurable"].get("experiment_switch_enabled"),
                },
                "result": asdict(result),
                **extra,
            }

        except Exception as exc:
            n_error += 1
            record = {
                "ts": _now_iso(),
                "status": "error",
                "run": asdict(run),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                **extra,
            }

        with manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return ExperimentResult(
        root=str(root),
        manifest_path=str(manifest_path),
        n_total=n_total,
        n_ok=n_ok,
        n_error=n_error,
        n_skipped=n_skipped,
    )

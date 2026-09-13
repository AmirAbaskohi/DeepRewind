"""Analysis utilities for Phase 6 experiments."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from open_deep_research.configuration import Configuration


@dataclass
class AnalysisOutput:
    manifest_df: pd.DataFrame
    divergence_df: pd.DataFrame
    commitment_df: pd.DataFrame
    rollback_df: pd.DataFrame
    susceptibility_df: pd.DataFrame
    calibration_df: pd.DataFrame
    summary_jsonl: str
    summary_md: str


def _table_text(df: pd.DataFrame, limit: Optional[int] = None) -> str:
    if df.empty:
        return "No rows."
    view = df.head(limit) if limit is not None else df
    try:
        return view.to_markdown(index=False)
    except Exception:
        return view.to_string(index=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def _load_manifest(root: Path) -> pd.DataFrame:
    records = _read_jsonl(root / "manifest.jsonl")
    rows: list[dict[str, Any]] = []
    for rec in records:
        run = rec.get("run", {}) if isinstance(rec.get("run"), dict) else {}
        result = rec.get("result", {}) if isinstance(rec.get("result"), dict) else {}
        row = {
            "status": rec.get("status"),
            "question_id": run.get("question_id"),
            "question": run.get("question"),
            "family": run.get("family"),
            "condition": run.get("condition"),
            "arm": run.get("arm"),
            "seed": run.get("seed"),
            "run_dir": run.get("run_dir"),
            "final_answer": result.get("final_answer", ""),
            "research_log_path": result.get("research_log_path"),
            "epistemic_log_path": result.get("epistemic_log_path"),
            "world_model_log_path": result.get("world_model_log_path"),
            "error": rec.get("error") or result.get("error"),
        }
        row["alternatives"] = rec.get("alternatives")
        rows.append(row)
    return pd.DataFrame(rows)


def _default_judge_factory(config: Optional[dict], cache_path: Path):
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage
        from open_deep_research.utils import get_api_key_for_model, get_base_url_for_model
    except Exception:
        def _always_false_agree(_question: str, _a1: str, _a2: str) -> bool:
            return False

        def _always_false_match(_question: str, _answer: str, _alternative: str) -> bool:
            return False

        return _always_false_agree, _always_false_match

    settings = Configuration.from_runnable_config(config)
    model = init_chat_model(
        configurable_fields=("model", "max_tokens", "api_key", "base_url", "temperature")
    ).with_config(
        {
            "model": settings.world_model_model,
            "max_tokens": 120,
            "temperature": 0.0,
            "api_key": get_api_key_for_model(settings.world_model_model, config),
            "base_url": get_base_url_for_model(settings.world_model_model, config),
            "tags": ["langsmith:nostream"],
        }
    )

    cache: dict[str, bool] = {}
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cache = {str(k): bool(v) for k, v in payload.items()}
        except Exception:
            cache = {}

    def _judge(prompt: str, cache_key: str) -> bool:
        if cache_key in cache:
            return cache[cache_key]
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            text = getattr(response, "content", "")
            value = str(text).strip().lower()
            out = value.startswith("true") or value.startswith("yes")
        except Exception:
            # Safe fallback when LLM calls are unavailable.
            out = False
        cache[cache_key] = out
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def answers_agree(question: str, a1: str, a2: str) -> bool:
        key = f"agree::{question}::{a1}::{a2}"
        prompt = (
            "Answer True or False only. Do the two answers below materially agree on the same substantive conclusion for the question?\n\n"
            f"Question: {question}\n"
            f"Answer 1: {a1}\n"
            f"Answer 2: {a2}"
        )
        return _judge(prompt, key)

    def answer_matches_alternative(question: str, answer: str, alternative: str) -> bool:
        key = f"match::{question}::{answer}::{alternative}"
        prompt = (
            "Answer True or False only. Does the answer support or align with this alternative?\n\n"
            f"Question: {question}\n"
            f"Alternative: {alternative}\n"
            f"Answer: {answer}"
        )
        return _judge(prompt, key)

    return answers_agree, answer_matches_alternative


def _world_model_summary(world_model_log_path: Optional[str]) -> dict[str, Any]:
    if not world_model_log_path:
        return {}
    rows = _read_jsonl(Path(world_model_log_path))
    run_end = next((row for row in reversed(rows) if row.get("event") == "run_end"), {})
    return {
        "n_commit": int(run_end.get("n_commit", 0) or 0),
        "n_not_commit": int(run_end.get("n_not_commit", 0) or 0),
        "n_contested": int(run_end.get("n_contested", 0) or 0),
        "n_enforced": int(run_end.get("n_enforced", 0) or 0),
        "n_monitor_checks": int(run_end.get("n_monitor_checks", 0) or 0),
        "n_triggers_fired": int(run_end.get("n_triggers_fired", 0) or 0),
        "n_rollbacks": int(run_end.get("n_rollbacks", 0) or 0),
        "n_preserved": int(run_end.get("n_preserved", 0) or 0),
        "n_reopened": int(run_end.get("n_reopened", 0) or 0),
    }


def _calibration_rows(world_model_log_path: Optional[str]) -> list[dict[str, float]]:
    if not world_model_log_path:
        return []
    rows = _read_jsonl(Path(world_model_log_path))
    out: list[dict[str, float]] = []
    for row in rows:
        if row.get("event") != "prediction":
            continue
        p = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        llm_k = p.get("llm_kappa")
        llm_l = p.get("llm_lambda")
        llm_g = p.get("llm_gamma")
        g_k = p.get("kappa")
        g_l = p.get("lambda_")
        g_g = p.get("gamma")
        if all(v is not None for v in [llm_k, llm_l, llm_g, g_k, g_l, g_g]):
            out.append(
                {
                    "llm_kappa": float(llm_k),
                    "llm_lambda": float(llm_l),
                    "llm_gamma": float(llm_g),
                    "kappa": float(g_k),
                    "lambda_": float(g_l),
                    "gamma": float(g_g),
                }
            )
    return out


def analyze_experiment(
    root: str,
    config: Optional[dict] = None,
    answers_agree: Optional[Callable[[str, str, str], bool]] = None,
    answer_matches_alternative: Optional[Callable[[str, str, str], bool]] = None,
) -> AnalysisOutput:
    """Compute analysis-ready tables from manifest and per-run artifacts."""
    root_path = Path(root)
    analysis_dir = root_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = _load_manifest(root_path)
    if manifest_df.empty:
        raise ValueError("No manifest records found.")

    if answers_agree is None or answer_matches_alternative is None:
        agree_fn, match_fn = _default_judge_factory(config, analysis_dir / "judge_cache.json")
        answers_agree = answers_agree or agree_fn
        answer_matches_alternative = answer_matches_alternative or match_fn

    ok_df = manifest_df[manifest_df["status"] == "ok"].copy()

    # Commitment/rollback profile tables.
    summary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for _, row in ok_df.iterrows():
        wm = _world_model_summary(row.get("world_model_log_path"))
        summary_rows.append({**row.to_dict(), **wm})
        for cal in _calibration_rows(row.get("world_model_log_path")):
            calibration_rows.append({**row.to_dict(), **cal})

    commit_df = pd.DataFrame(summary_rows)
    if commit_df.empty:
        commit_df = pd.DataFrame(
            columns=[
                "question_id",
                "family",
                "condition",
                "arm",
                "seed",
                "n_commit",
                "n_not_commit",
                "n_contested",
                "n_enforced",
            ]
        )

    # a) Final-answer divergence across initial conditions.
    divergence_rows: list[dict[str, Any]] = []
    for (qid, arm), group in ok_df[ok_df["family"] == "initial_condition"].groupby(["question_id", "arm"]):
        values = group[["condition", "question", "final_answer"]].to_dict("records")
        for left, right in itertools.combinations(values, 2):
            agree = answers_agree(left["question"], left["final_answer"], right["final_answer"])
            divergence_rows.append(
                {
                    "question_id": qid,
                    "arm": arm,
                    "condition_a": left["condition"],
                    "condition_b": right["condition"],
                    "agree": bool(agree),
                    "diverged": not bool(agree),
                }
            )
    divergence_df = pd.DataFrame(divergence_rows)

    # c) Rollback activity and recovery in switching runs.
    rollback_rows: list[dict[str, Any]] = []
    switching_df = ok_df[ok_df["family"] == "switching"]
    for _, row in switching_df.iterrows():
        alts = row.get("alternatives") if isinstance(row.get("alternatives"), list) else []
        alt_b = str(alts[1]) if len(alts) > 1 else ""
        recovered_to_b = bool(answer_matches_alternative(row["question"], row["final_answer"], alt_b)) if alt_b else False
        wm = _world_model_summary(row.get("world_model_log_path"))
        rollback_rows.append(
            {
                "question_id": row["question_id"],
                "arm": row["arm"],
                "condition": row["condition"],
                "recovered_to_b": recovered_to_b,
                **wm,
            }
        )
    rollback_df = pd.DataFrame(rollback_rows)

    # d) Seeding susceptibility vs neutral.
    susceptibility_rows: list[dict[str, Any]] = []
    ic_df = ok_df[ok_df["family"] == "initial_condition"]
    for arm, arm_group in ic_df.groupby("arm"):
        flips = 0
        total = 0
        for qid, qgroup in arm_group.groupby("question_id"):
            neutral = qgroup[qgroup["condition"] == "neutral"]
            if neutral.empty:
                continue
            neutral_answer = str(neutral.iloc[0]["final_answer"])
            question = str(neutral.iloc[0]["question"])
            for condition in ["support", "counter", "distract"]:
                comp = qgroup[qgroup["condition"] == condition]
                if comp.empty:
                    continue
                total += 1
                agree = answers_agree(question, neutral_answer, str(comp.iloc[0]["final_answer"]))
                if not agree:
                    flips += 1
        susceptibility_rows.append(
            {
                "arm": arm,
                "flip_fraction": (float(flips) / float(total)) if total else 0.0,
                "n_flips": flips,
                "n_compared": total,
            }
        )
    susceptibility_df = pd.DataFrame(susceptibility_rows)

    # e) Calibration table: correlation between LLM self-estimates and grounded.
    calibration_df = pd.DataFrame(calibration_rows)
    if not calibration_df.empty:
        corr = {
            "corr_kappa": calibration_df[["llm_kappa", "kappa"]].corr().iloc[0, 1],
            "corr_lambda": calibration_df[["llm_lambda", "lambda_"]].corr().iloc[0, 1],
            "corr_gamma": calibration_df[["llm_gamma", "gamma"]].corr().iloc[0, 1],
            "n_points": len(calibration_df),
        }
    else:
        corr = {"corr_kappa": None, "corr_lambda": None, "corr_gamma": None, "n_points": 0}

    summary_jsonl = analysis_dir / "summary.jsonl"
    summary_rows_out = [
        {"table": "divergence", "rows": int(len(divergence_df))},
        {"table": "commitment", "rows": int(len(commit_df))},
        {"table": "rollback", "rows": int(len(rollback_df))},
        {"table": "susceptibility", "rows": int(len(susceptibility_df))},
        {"table": "calibration_corr", **corr},
    ]
    with summary_jsonl.open("w", encoding="utf-8") as fh:
        for row in summary_rows_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_md = analysis_dir / "summary.md"
    lines = [
        "# Experiment Analysis Summary",
        "",
        "## Divergence",
        _table_text(divergence_df, limit=50),
        "",
        "## Commitment Profile",
        _table_text(
            commit_df[
                [
                    "question_id",
                    "family",
                    "condition",
                    "arm",
                    "n_commit",
                    "n_not_commit",
                    "n_contested",
                    "n_enforced",
                ]
            ],
            limit=50,
        ),
        "",
        "## Rollback Activity",
        _table_text(rollback_df, limit=50),
        "",
        "## Seeding Susceptibility",
        _table_text(susceptibility_df),
        "",
        "## Calibration Correlation",
        _table_text(pd.DataFrame([corr])),
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    return AnalysisOutput(
        manifest_df=manifest_df,
        divergence_df=divergence_df,
        commitment_df=commit_df,
        rollback_df=rollback_df,
        susceptibility_df=susceptibility_df,
        calibration_df=calibration_df,
        summary_jsonl=str(summary_jsonl),
        summary_md=str(summary_md),
    )

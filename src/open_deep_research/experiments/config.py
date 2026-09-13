"""Experiment specification, arm mapping, and question loading utilities."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

Family = Literal["initial_condition", "switching"]


@dataclass
class ExperimentRun:
    question_id: str
    question: str
    family: Family
    condition: str
    arm: str
    seed: int
    run_dir: str


@dataclass
class ExperimentSpec:
    questions: list[dict[str, str]]
    output_root: str
    families: list[Family] = field(default_factory=lambda: ["initial_condition", "switching"])
    initial_conditions: list[str] = field(
        default_factory=lambda: ["neutral", "counter", "distract", "support"]
    )
    switching_conditions: list[str] = field(default_factory=lambda: ["noswitch", "switch"])
    arms: list[str] = field(
        default_factory=lambda: ["base", "wm_shadow", "wm_gate", "wm_rollback"]
    )
    seeds: list[int] = field(default_factory=lambda: [0])
    alternatives_by_question: dict[str, list[str]] = field(default_factory=dict)
    enable_concurrency: bool = False
    max_concurrency: int = 1


@dataclass
class ExperimentResult:
    root: str
    manifest_path: str
    n_total: int
    n_ok: int
    n_error: int
    n_skipped: int


def _auto_id(index: int) -> str:
    return f"q{index + 1:05d}"


def load_questions(path: str) -> list[dict[str, str]]:
    """Load questions from .jsonl or .txt where each line is one question."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    out: list[dict[str, str]] = []
    suffix = p.suffix.lower()
    lines = p.read_text(encoding="utf-8").splitlines()

    if suffix == ".txt":
        for i, line in enumerate(lines):
            q = line.strip()
            if not q:
                continue
            out.append({"id": _auto_id(len(out)), "question": q})
        return out

    if suffix == ".jsonl":
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                q = str(payload.get("question", "")).strip()
                if not q:
                    continue
                qid = str(payload.get("id") or payload.get("qid") or _auto_id(len(out)))
                out.append({"id": qid, "question": q})
            else:
                q = str(payload).strip()
                if not q:
                    continue
                out.append({"id": _auto_id(len(out)), "question": q})
        return out

    raise ValueError(f"Unsupported question file extension: {p.suffix}")


def load_alternatives(path: Optional[str]) -> dict[str, list[str]]:
    """Load optional alternatives mapping from json/jsonl.

    Accepted shapes:
    - JSON object: {"qid": ["altA", "altB"]}
    - JSONL lines: {"id": "qid", "alternatives": [...]} or
                   {"question_id": "qid", "alternatives": [...]}.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Alternatives file not found: {path}")

    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Alternatives JSON must be an object mapping question id to list.")
        out: dict[str, list[str]] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                out[str(key)] = [str(item) for item in value if str(item).strip()]
        return out

    out: dict[str, list[str]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            continue
        qid = payload.get("id") or payload.get("qid") or payload.get("question_id")
        alternatives = payload.get("alternatives")
        if qid and isinstance(alternatives, list):
            out[str(qid)] = [str(item) for item in alternatives if str(item).strip()]
    return out


def arm_to_config(arm: str, base_config: dict[str, Any]) -> dict[str, Any]:
    """Map an experiment arm to concrete world-model flags on a copied config."""
    cfg = copy.deepcopy(base_config or {})
    configurable = copy.deepcopy(cfg.get("configurable", {}) or {})

    def set_flags(*, enable_world_model: bool, wm_enforce_decision: bool, wm_enable_rollback: bool, wm_enforce_rollback: bool = False, wm_recovery_mode: str = "none") -> None:
        configurable["enable_world_model"] = enable_world_model
        configurable["wm_enforce_decision"] = wm_enforce_decision
        configurable["wm_enable_rollback"] = wm_enable_rollback
        configurable["wm_enforce_rollback"] = wm_enforce_rollback
        configurable["wm_recovery_mode"] = wm_recovery_mode

    if arm == "base":
        set_flags(
            enable_world_model=False,
            wm_enforce_decision=False,
            wm_enable_rollback=False,
            wm_enforce_rollback=False,
            wm_recovery_mode="none",
        )
    elif arm == "wm_shadow":
        set_flags(
            enable_world_model=True,
            wm_enforce_decision=False,
            wm_enable_rollback=False,
            wm_enforce_rollback=False,
            wm_recovery_mode="none",
        )
    elif arm in {"wm_gate", "ablate_norollback"}:
        set_flags(
            enable_world_model=True,
            wm_enforce_decision=True,
            wm_enable_rollback=False,
            wm_enforce_rollback=False,
            wm_recovery_mode="none",
        )
    elif arm == "wm_rollback":
        set_flags(
            enable_world_model=True,
            wm_enforce_decision=True,
            wm_enable_rollback=True,
            wm_enforce_rollback=True,
            wm_recovery_mode="rollback",
        )
    else:
        raise ValueError(f"Unknown arm: {arm}")

    cfg["configurable"] = configurable
    return cfg

"""Append-only world-model prediction logger.

Phase 1 writes stub commitment predictions without affecting agent behavior.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from open_deep_research.configuration import Configuration, world_model_enabled

LOG_SCHEMA_VERSION = 1

_LOGGERS: dict[str, "WorldModelRun"] = {}
_LOGGERS_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: str) -> str:
    keep = "-_."
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in value)
    return cleaned[:80] or "run"


def _run_id_from_config(config: Optional[dict]) -> str:
    if not config:
        return "default"
    configurable = config.get("configurable", {}) or {}
    metadata = config.get("metadata", {}) or {}
    candidates = [
        configurable.get("thread_id"),
        metadata.get("thread_id"),
        configurable.get("run_id"),
        metadata.get("run_id"),
        config.get("run_id"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return "default"


def _log_dir(config: Optional[dict]) -> str:
    env = os.environ.get("WORLD_MODEL_LOG_DIR")
    if env:
        return env
    configurable = (config or {}).get("configurable", {}) if config else {}
    return configurable.get("world_model_log_dir") or "world_model_logs"


def _truncate(value: Any, limit: int = 4000) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [truncated {len(value) - limit} chars]"
    return value


class _NullWorldModelRun:
    enabled = False
    log_path = None

    def set_question(self, *_args, **_kwargs) -> None:
        return None

    def prediction(self, *_args, **_kwargs) -> None:
        return None

    def finish(self, *_args, **_kwargs) -> None:
        return None

    def monitor_check(self, *_args, **_kwargs) -> None:
        return None

    def trigger_fired(self, *_args, **_kwargs) -> None:
        return None

    def rollback(self, *_args, **_kwargs) -> None:
        return None


class WorldModelRun:
    """Streams world-model events for one run into JSONL."""

    enabled = True

    def __init__(self, run_id: str, log_dir: str, config: Optional[dict], question: Optional[str] = None):
        self.run_id = run_id
        self._lock = threading.RLock()
        self._finished = False
        self._question: Optional[str] = question
        self._n_predictions = 0
        self._n_parse_errors = 0
        self._n_commit = 0
        self._n_not_commit = 0
        self._n_contested = 0
        self._n_enforced = 0
        self._n_monitor_checks = 0
        self._n_triggers_fired = 0
        self._n_rollbacks = 0
        self._n_preserved = 0
        self._n_reopened = 0
        settings = Configuration.from_runnable_config(config)

        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_name = f"world_model_{_sanitize(run_id)}_{stamp}.jsonl"
        self.log_path = os.path.join(log_dir, file_name)
        self._fh = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115

        snapshot = {
            "enable_world_model": settings.enable_world_model,
            "world_model_model": settings.world_model_model,
            "world_model_max_tokens": settings.world_model_max_tokens,
            "wm_prediction_temperature": settings.wm_prediction_temperature,
            "wm_max_retries": settings.wm_max_retries,
            "world_model_log_dir": settings.world_model_log_dir,
            "wm_khop": settings.wm_khop,
            "wm_encode_message_passing": settings.wm_encode_message_passing,
            "wm_max_context_nodes": settings.wm_max_context_nodes,
            "wm_alpha_kappa": settings.wm_alpha_kappa,
            "wm_alpha_lambda": settings.wm_alpha_lambda,
            "wm_alpha_gamma": settings.wm_alpha_gamma,
            "wm_tau_commit": settings.wm_tau_commit,
            "wm_eta": settings.wm_eta,
            "wm_beta0": settings.wm_beta0,
            "wm_plausibility_temp": settings.wm_plausibility_temp,
            "wm_default_reliability": settings.wm_default_reliability,
            "wm_default_edge_weight": settings.wm_default_edge_weight,
            "wm_commit_lockin_strength": settings.wm_commit_lockin_strength,
            "wm_recovery_budget": settings.wm_recovery_budget,
            "wm_cost_retract": settings.wm_cost_retract,
            "wm_cost_regen": settings.wm_cost_regen,
            "wm_theta_rho_star": settings.wm_theta_rho_star,
            "wm_theta_w_star": settings.wm_theta_w_star,
            "wm_contested_band": settings.wm_contested_band,
            "wm_contested_con_min": settings.wm_contested_con_min,
            "wm_tc_source": settings.wm_tc_source,
            "wm_tc_per_probe": settings.wm_tc_per_probe,
            "wm_default_tc": settings.wm_default_tc,
            "wm_tau1": settings.wm_tau1,
            "wm_tau2": settings.wm_tau2,
            "wm_value_source": settings.wm_value_source,
            "wm_gate_mode": settings.wm_gate_mode,
            "wm_u_commit_threshold": settings.wm_u_commit_threshold,
            "wm_enforce_decision": settings.wm_enforce_decision,
            "wm_enable_rollback": settings.wm_enable_rollback,
            "wm_recovery_mode": settings.wm_recovery_mode,
            "wm_enforce_rollback": settings.wm_enforce_rollback,
            "wm_beta_star": settings.wm_beta_star,
            "wm_preserve_independent": settings.wm_preserve_independent,
            "wm_regenerate_dependents": settings.wm_regenerate_dependents,
            "experiment_seed_injection": settings.experiment_seed_injection,
            "experiment_switch_enabled": settings.experiment_switch_enabled,
            "experiment_switch_after": settings.experiment_switch_after,
            "experiment_switch_n": settings.experiment_switch_n,
        }
        self._write({
            "event": "run_start",
            "schema_version": LOG_SCHEMA_VERSION,
            "run_id": run_id,
            "question": _truncate(question, 8000) if question else "",
            "world_model_model": settings.world_model_model,
            "config_snapshot": snapshot,
            "ts": _now_iso(),
        })

    def _write(self, record: dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def set_question(self, question: Optional[str]) -> None:
        with self._lock:
            if self._question is None and question:
                self._question = question

    def prediction(self, payload: Any) -> None:
        with self._lock:
            if is_dataclass(payload):
                data = asdict(payload)
            elif isinstance(payload, dict):
                data = payload
            else:
                data = {"payload": str(payload)}
            self._n_predictions += 1
            if bool(data.get("parse_error")):
                self._n_parse_errors += 1
            if data.get("decision") == "commit":
                self._n_commit += 1
            elif data.get("decision") == "not_commit":
                self._n_not_commit += 1
            if bool(data.get("contested")):
                self._n_contested += 1
            if bool(data.get("enforced")):
                self._n_enforced += 1
            self._write({
                "event": "prediction",
                "run_id": self.run_id,
                "prediction": data,
                "ts": _now_iso(),
            })

    def monitor_check(self, payload: Any) -> None:
        with self._lock:
            data = payload if isinstance(payload, dict) else {"payload": payload}
            self._n_monitor_checks += 1
            self._write({
                "event": "monitor_check",
                "run_id": self.run_id,
                "monitor": data,
                "ts": _now_iso(),
            })

    def trigger_fired(self, payload: Any) -> None:
        with self._lock:
            data = payload if isinstance(payload, dict) else {"payload": payload}
            self._n_triggers_fired += 1
            self._write({
                "event": "trigger_fired",
                "run_id": self.run_id,
                "trigger": data,
                "ts": _now_iso(),
            })

    def rollback(self, payload: Any) -> None:
        with self._lock:
            data = payload if isinstance(payload, dict) else {"payload": payload}
            self._n_rollbacks += 1
            preserved = data.get("preserved_nodes") if isinstance(data, dict) else []
            reopened = data.get("reopen_claims") if isinstance(data, dict) else []
            if isinstance(preserved, list):
                self._n_preserved += len(preserved)
            if isinstance(reopened, list):
                self._n_reopened += len(reopened)
            self._write({
                "event": "rollback",
                "run_id": self.run_id,
                "rollback": data,
                "ts": _now_iso(),
            })

    def finish(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._write({
                "event": "run_end",
                "run_id": self.run_id,
                "n_predictions": self._n_predictions,
                "n_parse_errors": self._n_parse_errors,
                "n_commit": self._n_commit,
                "n_not_commit": self._n_not_commit,
                "n_contested": self._n_contested,
                "n_enforced": self._n_enforced,
                "n_monitor_checks": self._n_monitor_checks,
                "n_triggers_fired": self._n_triggers_fired,
                "n_rollbacks": self._n_rollbacks,
                "n_preserved": self._n_preserved,
                "n_reopened": self._n_reopened,
                "ts": _now_iso(),
            })
            try:
                self._fh.close()
            except Exception:
                pass


def get_world_model_logger(config: Optional[dict], question: Optional[str] = None):
    """Return the world-model logger for this run, creating it when needed."""
    try:
        if not world_model_enabled(config):
            return _NullWorldModelRun()
        run_id = _run_id_from_config(config)
        with _LOGGERS_LOCK:
            logger = _LOGGERS.get(run_id)
            if logger is None:
                logger = WorldModelRun(run_id, _log_dir(config), config, question=question)
                _LOGGERS[run_id] = logger
            elif question:
                logger.set_question(question)
            return logger
    except Exception:
        return _NullWorldModelRun()

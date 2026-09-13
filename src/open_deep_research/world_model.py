"""World-model prediction for commitment actions.

This module predicts local graph deltas, computes grounded structural scores,
and emits a binary commit/not_commit decision.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
from typing import Any, Literal, Optional

try:
    from langchain.chat_models import init_chat_model
except Exception:  # pragma: no cover - fallback for minimal test envs
    class _FallbackChatModel:
        def with_config(self, _config):
            return self

        def invoke(self, _messages):
            raise RuntimeError("langchain is not installed")

    def init_chat_model(*_args, **_kwargs):
        return _FallbackChatModel()

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:  # pragma: no cover - fallback for minimal test envs
    @dataclass
    class HumanMessage:
        content: str

    @dataclass
    class SystemMessage:
        content: str

from open_deep_research.configuration import Configuration, world_model_enabled
from open_deep_research.world_model_encoder import (
    EncodedState,
    OBSERVED_NODE_TYPES,
    PREDICTABLE_NODE_TYPES,
    encode_state,
)
from open_deep_research.world_model_scoring import (
    build_trigger,
    compute_gamma,
    compute_irr,
    compute_kappa,
    compute_lambda,
    compute_tc,
    compute_utility,
    compute_value,
    decide,
    is_contested,
    reversibility_class,
    take_snapshot,
)


def _get_api_key_for_model(model: str, config: Optional[dict]) -> Optional[str]:
    try:
        from open_deep_research.utils import get_api_key_for_model

        return get_api_key_for_model(model, config)
    except Exception:
        return None


def _get_base_url_for_model(model: str, config: Optional[dict]) -> Optional[str]:
    try:
        from open_deep_research.utils import get_base_url_for_model

        return get_base_url_for_model(model, config)
    except Exception:
        return None

CommitDecision = Literal["commit", "not_commit"]


@dataclass
class GraphDelta:
    """Predicted local graph delta over predictable node/edge types."""

    update_nodes: list[dict] = field(default_factory=list)
    update_edges: list[dict] = field(default_factory=list)


@dataclass
class ReversibilityMetadata:
    """Predicted reversibility metadata for a commitment action."""

    kappa: Optional[float] = None
    lambda_: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[dict] = None


@dataclass
class WorldModelPrediction:
    """World-model commitment prediction payload."""

    action_id: str
    target_node_ids: list[str]
    delta: GraphDelta
    reversibility: ReversibilityMetadata
    kappa: Optional[float] = None
    lambda_: Optional[float] = None
    gamma: Optional[float] = None
    reversibility_class: Optional[str] = None
    value: Optional[float] = None
    contested: Optional[bool] = None
    irr: Optional[float] = None
    trigger_coverage: Optional[float] = None
    utility: Optional[float] = None
    decision: CommitDecision = "commit"
    llm_kappa: Optional[float] = None
    llm_lambda: Optional[float] = None
    llm_gamma: Optional[float] = None
    resolved_theta: Optional[dict] = None
    encoded_prompt: Optional[str] = None
    raw_llm_output: Optional[str] = None
    temperature: Optional[float] = None
    n_retries_used: int = 0
    parse_error: bool = False
    latency_ms: Optional[int] = None
    dropped_predictions: list[dict] = field(default_factory=list)
    wm_enforce_decision: bool = False
    enforced: bool = False
    model: str = ""
    timestamp: str = ""


# Mirrors deep_researcher.py model bootstrap, but this phase never invokes it.
_configurable_model = init_chat_model(
        configurable_fields=("model", "max_tokens", "api_key", "base_url", "temperature"),
)


SYSTEM_PREDICTION_PROMPT = """You are a world model of a deep-research agent's epistemic state. You are given the current research-state graph and one candidate COMMIT action. Predict how the graph will change if the action is taken, and estimate the reversibility of that change.

You MUST NOT invent, rewrite, or predict Source or Evidence nodes. Their content is determined by the internet, not by the agent. You may only REFERENCE existing Source/Evidence by their aliases (e.g., E1, S1). Predict changes ONLY to Claim, Hypothesis, Assumption, Commitment, DraftFragment, and PlanStep nodes and the edges among them.

Predict:
1) update_nodes: nodes whose attributes change or are newly created, among predictable types only.
     For each entry use: {"alias":"...","type":"...","op":"update|create","fields":{...}}
2) update_edges: edges added/removed among predictable nodes, or between existing Source/Evidence aliases and predictable nodes.
     For each entry use: {"op":"add|remove","src":"...","dst":"...","relation":"...","weight":0.0}
3) reversibility:
     - kappa in [0,1]: commitment effect (narrowing competing hypotheses)
     - lambda in [0,1]: information loss
     - gamma in [0,1]: normalized recovery cost
     - theta: {"claim_alias":"...","rho_star":0.0,"w_star":0.0} or null
4) optional trigger_coverage in [0,1] if you can estimate how likely future probes
    would catch contradictions for the committed claim.

Respond with ONLY a single JSON object and no prose or markdown fences.
Use this exact top-level shape:
{
    "update_nodes": [
        {"alias": "H1", "type": "Hypothesis", "op": "update", "fields": {"plausibility": 0.61}},
        {"alias": "K2", "type": "Commitment", "op": "create", "fields": {"status": "active"}}
    ],
    "update_edges": [
        {"op": "add", "src": "E1", "dst": "C1", "relation": "supports", "weight": 0.8}
    ],
    "reversibility": {
        "kappa": 0.44,
        "lambda": 0.12,
        "gamma": 0.31,
        "theta": {"claim_alias": "C1", "rho_star": 0.75, "w_star": 0.6}
    },
    "trigger_coverage": 0.5
}

Numeric fields must be plain JSON numbers in [0,1]. Aliases must refer to aliases present in the provided state, except newly created predictable nodes which may introduce a new alias using a predictable-type prefix.
"""


ALIAS_PREFIX_TO_TYPE = {
        "C": "Claim",
        "H": "Hypothesis",
        "A": "Assumption",
        "K": "Commitment",
        "D": "DraftFragment",
        "P": "PlanStep",
        "S": "Source",
        "E": "Evidence",
}


@dataclass
class _PredictionRunMeta:
    n_retries_used: int = 0
    parse_error: bool = False
    dropped_predictions: list[dict] = field(default_factory=list)
    predicted_tc: Optional[float] = None


_MODELS: dict[str, "WorldModel"] = {}
_MODELS_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class WorldModel:
    """World model for commitment prediction and structural commit scoring."""

    def __init__(self, config: Optional[dict], prediction_client: Optional[Any] = None):
        self._config = config or {}
        self._enabled = world_model_enabled(config)
        self._settings = Configuration.from_runnable_config(config)
        self._model_name = self._settings.world_model_model
        self._last_encoded_state: Optional[EncodedState] = None
        self._last_meta = _PredictionRunMeta()
        self.active_commitments: dict[str, dict[str, Any]] = {}

        model_config = {
            "model": self._settings.world_model_model,
            "max_tokens": self._settings.world_model_max_tokens,
            "api_key": _get_api_key_for_model(self._settings.world_model_model, config),
            "base_url": _get_base_url_for_model(self._settings.world_model_model, config),
            "temperature": self._settings.wm_prediction_temperature,
            "tags": ["langsmith:nostream"],
        }
        self._model = prediction_client or _configurable_model.with_config(model_config)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_encoded_state(self) -> Optional[EncodedState]:
        """Most recent encoded state generated during ``predict``."""
        return self._last_encoded_state

    def encode(self, epistemic_graph: Any, action: dict) -> EncodedState:
        """Encode local graph context for the current action (no model call)."""
        action = action or {}
        question = action.get("question")
        if question is not None:
            question = str(question)
        return encode_state(
            graph=epistemic_graph,
            action=action,
            question=question,
            khop=self._settings.wm_khop,
            encode_message_passing=self._settings.wm_encode_message_passing,
            max_context_nodes=self._settings.wm_max_context_nodes,
        )

    def register_active_commitment(
        self,
        commitment_id: str,
        claim_id: Optional[str],
        hypothesis_id: Optional[str],
        theta: Optional[dict],
        step: int,
    ) -> None:
        """Register a live commitment for rollback monitoring."""
        if not commitment_id:
            return
        self.active_commitments[commitment_id] = {
            "commitment_id": commitment_id,
            "claim_id": claim_id,
            "hypothesis_id": hypothesis_id,
            "theta": dict(theta or {}),
            "step": int(step),
        }

    def unregister_active_commitment(self, commitment_id: str) -> None:
        """Remove a commitment from the active monitor registry."""
        if commitment_id:
            self.active_commitments.pop(commitment_id, None)

    def _invoke_model(self, messages: list[Any]) -> str:
        response = self._model.invoke(messages)
        if isinstance(response, str):
            return response
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in content
            )
        return str(content)

    def _extract_json_object(self, text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        return cleaned[start:end + 1]

    def _parse_json(self, text: str) -> Optional[dict]:
        json_blob = self._extract_json_object(text)
        if not json_blob:
            return None
        try:
            loaded = json.loads(json_blob)
        except Exception:
            return None
        if isinstance(loaded, dict):
            return loaded
        return None

    def _is_predictable_alias(self, alias: str) -> bool:
        if not isinstance(alias, str) or not alias:
            return False
        alias = alias.strip()
        match = re.match(r"^([A-Z])[0-9]+$", alias)
        if not match:
            return False
        alias_type = ALIAS_PREFIX_TO_TYPE.get(match.group(1))
        return alias_type in PREDICTABLE_NODE_TYPES

    def _clamp_01(self, value: Any) -> Optional[float]:
        try:
            number = float(value)
        except Exception:
            return None
        return max(0.0, min(1.0, number))

    def _run_prediction(self, encoded_state: EncodedState) -> tuple[GraphDelta, ReversibilityMetadata, str]:
        """Run world-model prediction and parse response robustly."""
        self._last_meta = _PredictionRunMeta()

        user_prompt = encoded_state.prompt
        instruction_suffix = ""
        parsed: Optional[dict] = None
        raw_text = ""

        max_attempts = max(1, int(self._settings.wm_max_retries) + 1)
        for attempt in range(max_attempts):
            messages = [
                SystemMessage(content=SYSTEM_PREDICTION_PROMPT),
                HumanMessage(content=user_prompt + instruction_suffix),
            ]
            raw_text = self._invoke_model(messages)
            parsed = self._parse_json(raw_text)
            if parsed is not None:
                self._last_meta.n_retries_used = attempt
                break
            if attempt < max_attempts - 1:
                instruction_suffix = (
                    "\n\nYour previous output was not valid JSON. "
                    "Return ONLY the JSON object."
                )

        if parsed is None:
            self._last_meta.parse_error = True
            self._last_meta.n_retries_used = max_attempts - 1
            return GraphDelta(), ReversibilityMetadata(), raw_text

        alias_to_id = dict(encoded_state.alias_to_id or {})
        created_aliases: set[str] = set()

        update_nodes_raw = parsed.get("update_nodes")
        if not isinstance(update_nodes_raw, list):
            update_nodes_raw = []
        update_edges_raw = parsed.get("update_edges")
        if not isinstance(update_edges_raw, list):
            update_edges_raw = []

        update_nodes: list[dict] = []
        for item in update_nodes_raw:
            if not isinstance(item, dict):
                self._last_meta.dropped_predictions.append({"kind": "node", "reason": "not_object", "item": item})
                continue
            alias = str(item.get("alias") or "").strip()
            node_type = str(item.get("type") or "").strip()
            op = str(item.get("op") or "").strip().lower()
            fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}

            if node_type in OBSERVED_NODE_TYPES:
                self._last_meta.dropped_predictions.append({
                    "kind": "node",
                    "reason": "observed_type_masked",
                    "alias": alias,
                    "type": node_type,
                })
                continue
            if node_type not in PREDICTABLE_NODE_TYPES:
                self._last_meta.dropped_predictions.append({
                    "kind": "node",
                    "reason": "non_predictable_type",
                    "alias": alias,
                    "type": node_type,
                })
                continue
            if op not in ("update", "create"):
                self._last_meta.dropped_predictions.append({
                    "kind": "node",
                    "reason": "invalid_op",
                    "alias": alias,
                    "op": op,
                })
                continue

            resolved_id = alias_to_id.get(alias)
            if op == "update" and resolved_id is None:
                self._last_meta.dropped_predictions.append({
                    "kind": "node",
                    "reason": "unknown_alias_for_update",
                    "alias": alias,
                })
                continue

            if op == "create" and resolved_id is None:
                if not self._is_predictable_alias(alias):
                    self._last_meta.dropped_predictions.append({
                        "kind": "node",
                        "reason": "invalid_create_alias",
                        "alias": alias,
                    })
                    continue
                created_aliases.add(alias)

            update_nodes.append({
                "alias": alias,
                "id": resolved_id,
                "type": node_type,
                "op": op,
                "fields": fields,
            })

        update_edges: list[dict] = []
        for item in update_edges_raw:
            if not isinstance(item, dict):
                self._last_meta.dropped_predictions.append({"kind": "edge", "reason": "not_object", "item": item})
                continue
            op = str(item.get("op") or "").strip().lower()
            src_alias = str(item.get("src") or "").strip()
            dst_alias = str(item.get("dst") or "").strip()
            relation = str(item.get("relation") or "").strip() or "related_to"
            weight = item.get("weight")
            src_id = alias_to_id.get(src_alias)
            dst_id = alias_to_id.get(dst_alias)

            if op not in ("add", "remove"):
                self._last_meta.dropped_predictions.append({
                    "kind": "edge",
                    "reason": "invalid_op",
                    "edge": item,
                })
                continue

            if src_id is None and src_alias not in created_aliases:
                self._last_meta.dropped_predictions.append({
                    "kind": "edge",
                    "reason": "unknown_src_alias",
                    "edge": item,
                })
                continue
            if dst_id is None and dst_alias not in created_aliases:
                self._last_meta.dropped_predictions.append({
                    "kind": "edge",
                    "reason": "unknown_dst_alias",
                    "edge": item,
                })
                continue

            if (src_alias in created_aliases and src_alias[:1] in {"S", "E"}) or (
                dst_alias in created_aliases and dst_alias[:1] in {"S", "E"}
            ):
                self._last_meta.dropped_predictions.append({
                    "kind": "edge",
                    "reason": "created_observed_alias_blocked",
                    "edge": item,
                })
                continue

            src_type = ALIAS_PREFIX_TO_TYPE.get(src_alias[:1], "") if src_alias else ""
            dst_type = ALIAS_PREFIX_TO_TYPE.get(dst_alias[:1], "") if dst_alias else ""
            # Keep edges if at least one endpoint is predictable.
            if src_type in OBSERVED_NODE_TYPES and dst_type in OBSERVED_NODE_TYPES:
                self._last_meta.dropped_predictions.append({
                    "kind": "edge",
                    "reason": "observed_to_observed_blocked",
                    "edge": item,
                })
                continue

            resolved_weight: Optional[float] = None
            if weight is not None:
                try:
                    resolved_weight = float(weight)
                except Exception:
                    resolved_weight = None

            update_edges.append({
                "op": op,
                "src": src_alias,
                "src_id": src_id,
                "dst": dst_alias,
                "dst_id": dst_id,
                "relation": relation,
                "weight": resolved_weight,
            })

        reversibility_payload = parsed.get("reversibility")
        reversibility_payload = reversibility_payload if isinstance(reversibility_payload, dict) else {}
        theta_raw = reversibility_payload.get("theta")
        theta: Optional[dict] = None
        if isinstance(theta_raw, dict):
            claim_alias = str(theta_raw.get("claim_alias") or "").strip()
            claim_id = alias_to_id.get(claim_alias)
            rho_star = self._clamp_01(theta_raw.get("rho_star"))
            w_star = self._clamp_01(theta_raw.get("w_star"))
            if claim_id:
                theta = {
                    "claim_alias": claim_alias,
                    "claim_id": claim_id,
                    "rho_star": rho_star,
                    "w_star": w_star,
                }

        reversibility = ReversibilityMetadata(
            kappa=self._clamp_01(reversibility_payload.get("kappa")),
            lambda_=self._clamp_01(reversibility_payload.get("lambda")),
            gamma=self._clamp_01(reversibility_payload.get("gamma")),
            theta=theta,
        )

        tc_value = self._clamp_01(parsed.get("trigger_coverage"))
        self._last_meta.predicted_tc = tc_value

        return GraphDelta(update_nodes=update_nodes, update_edges=update_edges), reversibility, raw_text

    def _compute_grounded_scores(
        self,
        epistemic_graph: Any,
        action: dict,
        delta: GraphDelta,
        llm_reversibility: ReversibilityMetadata,
    ) -> dict[str, Any]:
        """Compute phase-4 grounded quantities from a read-only snapshot."""
        snapshot = take_snapshot(epistemic_graph)
        score_action = {
            **(action or {}),
            "predicted_delta": {
                "update_nodes": list(delta.update_nodes or []),
                "update_edges": list(delta.update_edges or []),
            },
            "included_node_ids": list(getattr(self._last_encoded_state, "included_node_ids", []) or []),
            "alias_to_id": dict(getattr(self._last_encoded_state, "alias_to_id", {}) or {}),
        }

        kappa = compute_kappa(snapshot, score_action, self._settings)
        lambda_ = compute_lambda(snapshot, score_action, self._settings)
        gamma = compute_gamma(snapshot, score_action, self._settings)
        irr = compute_irr(kappa, lambda_, gamma, self._settings)
        rev_class = reversibility_class(irr, self._settings)
        resolved_theta = build_trigger(snapshot, score_action, llm_reversibility.theta, self._settings)
        predicted_tc = self._last_meta.predicted_tc
        tc = compute_tc(snapshot, score_action, predicted_tc, self._settings)
        value = compute_value(snapshot, score_action, self._settings)
        utility = compute_utility(value, irr, tc, self._settings)
        contested = is_contested(snapshot, score_action, self._settings)
        decision = decide(irr, tc, utility, contested, self._settings)

        return {
            "kappa": kappa,
            "lambda_": lambda_,
            "gamma": gamma,
            "irr": irr,
            "reversibility_class": rev_class,
            "resolved_theta": resolved_theta,
            "trigger_coverage": tc,
            "value": value,
            "utility": utility,
            "contested": contested,
            "decision": decision,
        }

    def predict(self, epistemic_graph: Any, action: dict) -> WorldModelPrediction:
        """Predict graph delta + reversibility for a commitment action.

        Args:
            epistemic_graph: Current in-memory epistemic graph state.
            action: Action payload containing at least ``action_id`` and
                ``target_node_ids`` when available.
        """
        action = action or {}
        action_id = str(action.get("action_id") or "")
        target_node_ids = [str(node_id) for node_id in (action.get("target_node_ids") or [])]
        self._last_encoded_state = self.encode(epistemic_graph, action)

        if not self.enabled:
            return WorldModelPrediction(
                action_id=action_id,
                target_node_ids=target_node_ids,
                delta=GraphDelta(),
                reversibility=ReversibilityMetadata(),
                kappa=None,
                lambda_=None,
                gamma=None,
                reversibility_class=None,
                value=None,
                contested=None,
                llm_kappa=None,
                llm_lambda=None,
                llm_gamma=None,
                resolved_theta=None,
                irr=None,
                trigger_coverage=None,
                utility=None,
                decision="commit",
                encoded_prompt=self._last_encoded_state.prompt,
                raw_llm_output=None,
                temperature=self._settings.wm_prediction_temperature,
                n_retries_used=0,
                parse_error=False,
                latency_ms=0,
                dropped_predictions=[],
                wm_enforce_decision=bool(self._settings.wm_enforce_decision),
                enforced=False,
                model=self._model_name,
                timestamp=_now_iso(),
            )

        started = time.perf_counter()
        delta, reversibility, raw_text = self._run_prediction(self._last_encoded_state)
        elapsed_ms = int(round((time.perf_counter() - started) * 1000.0))
        grounded = self._compute_grounded_scores(epistemic_graph, action, delta, reversibility)

        return WorldModelPrediction(
            action_id=action_id,
            target_node_ids=target_node_ids,
            delta=delta,
            reversibility=reversibility,
            kappa=grounded.get("kappa"),
            lambda_=grounded.get("lambda_"),
            gamma=grounded.get("gamma"),
            reversibility_class=grounded.get("reversibility_class"),
            value=grounded.get("value"),
            contested=grounded.get("contested"),
            irr=grounded.get("irr"),
            trigger_coverage=grounded.get("trigger_coverage"),
            utility=grounded.get("utility"),
            decision=grounded.get("decision", "commit"),
            llm_kappa=reversibility.kappa,
            llm_lambda=reversibility.lambda_,
            llm_gamma=reversibility.gamma,
            resolved_theta=grounded.get("resolved_theta"),
            encoded_prompt=self._last_encoded_state.prompt,
            raw_llm_output=raw_text,
            temperature=self._settings.wm_prediction_temperature,
            n_retries_used=self._last_meta.n_retries_used,
            parse_error=self._last_meta.parse_error,
            latency_ms=elapsed_ms,
            dropped_predictions=list(self._last_meta.dropped_predictions),
            wm_enforce_decision=bool(self._settings.wm_enforce_decision),
            enforced=False,
            model=self._model_name,
            timestamp=_now_iso(),
        )


def get_world_model(config: Optional[dict]) -> WorldModel:
    """Return one world-model instance per run id."""
    run_id = _run_id_from_config(config)
    with _MODELS_LOCK:
        model = _MODELS.get(run_id)
        if model is None:
            model = WorldModel(config)
            _MODELS[run_id] = model
        return model

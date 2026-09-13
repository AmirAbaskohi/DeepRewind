"""Alternative-switching setup helpers for Phase 6 experiments."""

from __future__ import annotations

import json
from typing import Optional

from open_deep_research.configuration import Configuration


def elicit_alternatives(question: str, config: Optional[dict], k: int = 2) -> list[str]:
    """Elicit at least two competing alternatives for a question."""
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage
        from open_deep_research.utils import get_api_key_for_model, get_base_url_for_model
    except Exception:
        return [f"Alternative A for: {question}", f"Alternative B for: {question}"]

    settings = Configuration.from_runnable_config(config)
    model = init_chat_model(
        configurable_fields=("model", "max_tokens", "api_key", "base_url", "temperature")
    ).with_config(
        {
            "model": settings.world_model_model,
            "max_tokens": 300,
            "temperature": 0.0,
            "api_key": get_api_key_for_model(settings.world_model_model, config),
            "base_url": get_base_url_for_model(settings.world_model_model, config),
            "tags": ["langsmith:nostream"],
        }
    )
    prompt = (
        "Return a JSON array with at least two competing answer alternatives for the question. "
        "Each alternative must be short and mutually distinguishable. Return JSON only.\n\n"
        f"Question: {question}"
    )
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", "")
        text = content if isinstance(content, str) else str(content)
        parsed = json.loads(text)
        if isinstance(parsed, list):
            values = [str(item).strip() for item in parsed if str(item).strip()]
            if len(values) >= 2:
                return values[: max(2, k)]
    except Exception:
        pass
    # Fallback keeps experiments running even if model output is malformed.
    return [f"Alternative A for: {question}", f"Alternative B for: {question}"]


def build_switch_setup(
    question: str,
    condition: str,
    alternatives: list[str],
    theta_rho_star: float,
    theta_w_star: float,
) -> tuple[str, list[dict], Optional[dict]]:
    """Build preamble, initial A-support evidence, and optional switch directive."""
    _ = question
    condition = str(condition).strip().lower()
    alternatives = [str(a).strip() for a in alternatives if str(a).strip()]
    if len(alternatives) < 2:
        alternatives = alternatives + ["Competing alternative"]

    alt_a = alternatives[0]
    alt_b = alternatives[1]

    if condition == "noswitch":
        return "", [], None

    if condition != "switch":
        raise ValueError(f"Unknown switching condition: {condition}")

    preamble = (
        "For the initial part of this run, treat the first alternative below as the leading hypothesis "
        "that deserves early investigation:\n"
        f"- Preferred initial alternative: {alt_a}\n"
        f"- Competing alternative: {alt_b}"
    )

    initial_support = [
        {
            "claim_hint": alt_a,
            "stance": "support",
            "text": f"Preliminary indications favor alternative A: {alt_a}",
            "source_label": "Synthetic A-support prior",
            "reliability": max(0.8, float(theta_rho_star)),
            "weight": max(0.8, float(theta_w_star)),
        }
    ]

    switch_directive = {
        "condition": "switch",
        "alternative_a": alt_a,
        "alternative_b": alt_b,
        "switch_evidence": [
            {
                "claim_hint": alt_a,
                "stance": "contradict",
                "text": f"New high-confidence contradiction against alternative A: {alt_a}",
                "source_label": "Synthetic switch contradiction",
                "reliability": max(0.95, float(theta_rho_star)),
                "weight": max(0.95, float(theta_w_star)),
                "attach_to_commitment_claim": True,
            },
            {
                "claim_hint": alt_b,
                "stance": "support",
                "text": f"New high-confidence support for alternative B: {alt_b}",
                "source_label": "Synthetic switch support",
                "reliability": max(0.95, float(theta_rho_star)),
                "weight": max(0.95, float(theta_w_star)),
            },
        ],
    }
    return preamble, initial_support, switch_directive

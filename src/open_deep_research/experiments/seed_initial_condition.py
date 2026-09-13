"""Initial-condition seeding helpers for Phase 6 experiments."""

from __future__ import annotations

from typing import Any, Optional

from open_deep_research.configuration import Configuration


def elicit_target_proposition(question: str, config: Optional[dict]) -> str:
    """Elicit one concise target proposition for support/counter seeding."""
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage
        from open_deep_research.utils import get_api_key_for_model, get_base_url_for_model
    except Exception:
        return question.strip()

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
    prompt = (
        "Given the research question below, provide exactly one concise, testable target proposition "
        "that can be supported or contradicted. Return one sentence only.\n\n"
        f"Question: {question}"
    )
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        return str(content).strip() or question.strip()
    except Exception:
        return question.strip()


def build_seed(
    question: str,
    condition: str,
    target_proposition: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Build preamble and synthetic evidence for an initial-condition arm."""
    condition = str(condition).strip().lower()
    target = target_proposition.strip() or question.strip()

    if condition == "neutral":
        return "", []

    if condition == "support":
        preamble = (
            "Starting context for this run includes prior high-confidence indications that the "
            "following proposition is likely true. Treat this as provisional context, not proof: "
            f"{target}"
        )
        evidence = [
            {
                "claim_hint": target,
                "stance": "support",
                "text": f"Independent reports align with proposition: {target}",
                "source_label": "Synthetic supportive brief",
                "reliability": 0.9,
                "weight": 0.9,
            }
        ]
        return preamble, evidence

    if condition == "counter":
        preamble = (
            "Starting context for this run includes prior high-confidence indications that the "
            "following proposition is likely false. Treat this as provisional context, not proof: "
            f"{target}"
        )
        evidence = [
            {
                "claim_hint": target,
                "stance": "contradict",
                "text": f"Independent reports dispute proposition: {target}",
                "source_label": "Synthetic counter brief",
                "reliability": 0.9,
                "weight": 0.9,
            }
        ]
        return preamble, evidence

    if condition == "distract":
        distractor_claim = (
            "Peripheral operational details and adjacent trivia are highly salient and should be "
            "tracked before deciding core conclusions"
        )
        preamble = (
            "Starting context for this run emphasizes plausible but potentially irrelevant side factors. "
            "Do not assume they answer the core question."
        )
        evidence = [
            {
                "claim_hint": distractor_claim,
                "stance": "distractor",
                "text": "Several reports focus on peripheral process details unrelated to the central proposition.",
                "source_label": "Synthetic distractor brief",
                "reliability": 0.85,
                "weight": 0.2,
            }
        ]
        return preamble, evidence

    raise ValueError(f"Unknown initial-condition seed condition: {condition}")

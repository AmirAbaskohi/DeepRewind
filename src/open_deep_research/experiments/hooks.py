"""Default-off experiment hooks for synthetic evidence injection."""

from __future__ import annotations

from typing import Any, Optional


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def inject_synthetic_evidence_batch(
    egraph: Any,
    evidence_items: list[dict[str, Any]],
    condition: str,
    reason: str,
) -> dict[str, Any]:
    """Inject synthetic Source/Evidence/Claim structures into the epistemic graph."""
    claim_by_hint: dict[str, str] = {}
    injected_claim_ids: list[str] = []
    injected_evidence_ids: list[str] = []

    for item in evidence_items or []:
        claim_hint = str(item.get("claim_hint") or "").strip()
        if not claim_hint:
            continue
        stance = str(item.get("stance") or "neutral").strip().lower()
        text = str(item.get("text") or "").strip() or claim_hint
        source_label = str(item.get("source_label") or "Synthetic source").strip()
        reliability = _as_float(item.get("reliability"), 0.5)
        weight = _as_float(item.get("weight"), 1.0)

        claim_id = claim_by_hint.get(claim_hint)
        if not claim_id:
            claim_id = egraph.add_claim(
                text=claim_hint,
                metadata={
                    "synthetic": True,
                    "experiment_seed": condition,
                    "reason": reason,
                },
            )
            if claim_id:
                claim_by_hint[claim_hint] = claim_id
                injected_claim_ids.append(claim_id)

        source_id = egraph.add_source(
            title=source_label,
            metadata={
                "synthetic": True,
                "experiment_seed": condition,
                "reason": reason,
                "reliability": reliability,
            },
        )
        evidence_id = egraph.add_evidence(
            text=text,
            metadata={
                "synthetic": True,
                "experiment_seed": condition,
                "reason": reason,
                "stance": stance,
                "reliability": reliability,
            },
            sources=[source_id] if source_id else None,
        )

        if evidence_id:
            injected_evidence_ids.append(evidence_id)

        if evidence_id and claim_id:
            if stance == "contradict":
                egraph.contradicts(evidence_id, claim_id, weight=weight, synthetic=True, experiment_seed=condition)
            else:
                # support/neutral/distractor all remain observed evidence; only
                # contradict carries explicit reversal semantics.
                low_weight = 0.2 if stance in {"neutral", "distractor"} else weight
                egraph.supports(evidence_id, claim_id, weight=low_weight, synthetic=True, experiment_seed=condition)

    return {
        "claim_ids": injected_claim_ids,
        "evidence_ids": injected_evidence_ids,
    }


def maybe_apply_seed_injection(egraph: Any, configurable: Any) -> dict[str, Any]:
    """Apply initial-condition synthetic seed injection when enabled."""
    if not bool(getattr(configurable, "experiment_seed_injection", False)):
        return {"applied": False, "count": 0}

    condition = str(getattr(configurable, "experiment_seed_condition", "seed") or "seed")
    evidence_items = list(getattr(configurable, "experiment_injected_evidence", None) or [])
    if not evidence_items:
        return {"applied": False, "count": 0}

    payload = inject_synthetic_evidence_batch(
        egraph=egraph,
        evidence_items=evidence_items,
        condition=condition,
        reason="initial_condition_seed",
    )
    return {
        "applied": True,
        "count": len(payload.get("evidence_ids", [])),
        **payload,
    }


def should_fire_switch(
    *,
    configurable: Any,
    switch_directive: Optional[dict[str, Any]],
    first_commitment_seen: bool,
    step_index: int,
) -> bool:
    """Determine whether a switch injection should fire at this point."""
    if not bool(getattr(configurable, "experiment_switch_enabled", False)):
        return False
    if not switch_directive:
        return False

    mode = str(getattr(configurable, "experiment_switch_after", "first_commitment"))
    if mode == "n_steps":
        return step_index >= int(getattr(configurable, "experiment_switch_n", 1))
    return first_commitment_seen


def maybe_apply_switch_injection(
    *,
    egraph: Any,
    configurable: Any,
    switch_directive: Optional[dict[str, Any]],
    commitment_claim_id: Optional[str],
) -> dict[str, Any]:
    """Inject synthetic switch evidence for rollback-trigger study."""
    if not bool(getattr(configurable, "experiment_switch_enabled", False)):
        return {"applied": False, "count": 0}
    if not switch_directive:
        return {"applied": False, "count": 0}

    evidence_items = list((switch_directive or {}).get("switch_evidence") or [])
    if not evidence_items:
        return {"applied": False, "count": 0}

    # Bind target contradiction to the active commitment claim when requested.
    adjusted: list[dict[str, Any]] = []
    for item in evidence_items:
        copy_item = dict(item)
        if commitment_claim_id and copy_item.get("attach_to_commitment_claim"):
            copy_item["claim_hint"] = str(copy_item.get("claim_hint") or "")
            copy_item["claim_id_override"] = commitment_claim_id
        adjusted.append(copy_item)

    # Build claims via hint text first.
    payload = inject_synthetic_evidence_batch(
        egraph=egraph,
        evidence_items=adjusted,
        condition="switch",
        reason="alternative_switch",
    )

    # Explicitly add high-confidence contradiction to the actual commitment claim
    # so monitor/rollback can fire on that exact target.
    if commitment_claim_id:
        for item in adjusted:
            if str(item.get("stance", "")).lower() != "contradict":
                continue
            text = str(item.get("text") or "").strip()
            source_label = str(item.get("source_label") or "Switch source").strip()
            reliability = _as_float(item.get("reliability"), 0.95)
            weight = _as_float(item.get("weight"), 0.95)
            source_id = egraph.add_source(
                title=source_label,
                metadata={
                    "synthetic": True,
                    "experiment_seed": "switch",
                    "reason": "alternative_switch_targeted",
                    "reliability": reliability,
                },
            )
            evidence_id = egraph.add_evidence(
                text=text,
                metadata={
                    "synthetic": True,
                    "experiment_seed": "switch",
                    "reason": "alternative_switch_targeted",
                    "stance": "contradict",
                    "reliability": reliability,
                },
                sources=[source_id] if source_id else None,
            )
            if evidence_id:
                payload.setdefault("evidence_ids", []).append(evidence_id)
                egraph.contradicts(
                    evidence_id,
                    commitment_claim_id,
                    weight=weight,
                    synthetic=True,
                    experiment_seed="switch",
                )

    return {
        "applied": True,
        "count": len(payload.get("evidence_ids", [])),
        **payload,
    }

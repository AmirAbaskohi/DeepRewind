"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
    world_model_enabled,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    epistemic_reconciliation_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_belief_prompt,
    research_findings_summary_prompt,
    research_question_proposal_prompt,
    research_reasoning_prompt,
    research_system_prompt,
    sufficiency_decision_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    BeliefSet,
    ClarifyWithUser,
    ConductResearch,
    EpistemicReconciliation,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchProposal,
    ResearchQuestion,
    SufficiencyDecision,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_base_url_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)
from open_deep_research.research_logger import get_logger
from open_deep_research.epistemic_graph import get_epistemic_graph


# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url"),
)


def _summarize_tool_calls(tool_calls) -> list:
    """Build a compact, JSON-serializable summary of a message's tool calls.

    Used when logging an agent decision so the state graph records which tools
    were selected and the arguments that drove each call (search queries,
    reflections, delegated research topics, ...).
    """
    summary = []
    for tool_call in tool_calls or []:
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {}) or {}
        entry = {"name": name, "id": tool_call.get("id")}
        if name == "tavily_search":
            entry["queries"] = args.get("queries", [])
        elif name == "think_tool":
            entry["reflection"] = args.get("reflection", "")
        elif name == "ConductResearch":
            entry["research_topic"] = args.get("research_topic", "")
        elif name == "ResearchComplete":
            entry["args"] = {}
        else:
            entry["args"] = args
        summary.append(entry)
    return summary


def _parse_search_sources(observation: str) -> list:
    """Extract source titles and URLs from a formatted Tavily search result.

    The ``tavily_search`` tool returns results as a structured string where each
    source is introduced by a ``--- SOURCE n: <title> ---`` header followed by a
    ``URL: <url>`` line. We parse those markers to record what information the
    agent actually retrieved for the state graph.
    """
    sources = []
    if not isinstance(observation, str):
        return sources
    title = None
    for line in observation.splitlines():
        stripped = line.strip()
        if stripped.startswith("--- SOURCE") and stripped.endswith("---"):
            # Format: "--- SOURCE 1: Some Title ---"
            inner = stripped[len("--- SOURCE"):-len("---")].strip()
            if ":" in inner:
                title = inner.split(":", 1)[1].strip()
            else:
                title = inner
        elif stripped.startswith("URL:"):
            url = stripped[len("URL:"):].strip()
            sources.append({"title": title or "", "url": url})
            title = None
    return sources


def _split_report_sections(report_text: str) -> list:
    """Split a markdown report into ``(heading, section_text)`` pairs.

    Sections are delimited by markdown ATX headings (lines beginning with
    ``#``). Content before the first heading (if any) is returned under an
    empty heading. When the report has no headings, the whole report is
    returned as a single unnamed section.
    """
    sections = []
    current_heading = ""
    current_lines: list = []

    def _flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_heading.strip(), text))

    for line in (report_text or "").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            _flush()
            current_heading = stripped.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    _flush()
    return sections


def _format_prior_findings(prior_findings: str) -> str:
    """Format the running findings summary for inclusion in a planning prompt."""
    if prior_findings.strip():
        return (
            "Here is a running summary of what has already been researched and "
            "learned so far:\n" + prior_findings.strip()
        )
    return "No research has been conducted yet. This is the first iteration."


def _research_model_config(configurable: Configuration, config: RunnableConfig) -> dict:
    """Build the standard research-model invocation config."""
    return {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }


async def generate_and_score_beliefs(
    research_brief: str,
    configurable: Configuration,
    config: RunnableConfig,
    prior_findings: str = "",
    iteration: int = 1,
    max_iterations: int = 1,
):
    """Form, score, and select working beliefs about the topic for an iteration.

    A belief is a falsifiable hypothesis/assumption that guides which research
    questions are worth pursuing. The model generates ``num_research_proposals``
    candidate beliefs, scores each 0-10, and selects the best
    ``max_selected_research_proposals``.

    Returns ``(beliefs, selected_beliefs)`` where ``beliefs`` is a list of dicts
    ``{"belief", "score", "rationale"}`` sorted by descending score, and
    ``selected_beliefs`` is the chosen subset of belief strings.
    """
    max_selected = configurable.max_selected_research_proposals
    belief_prompt = research_belief_prompt.format(
        date=get_today_str(),
        research_brief=research_brief,
        prior_findings=_format_prior_findings(prior_findings),
        iteration=iteration,
        max_iterations=max_iterations,
        num_candidates=configurable.num_research_proposals,
        max_selected=max_selected,
    )
    belief_model = (
        configurable_model
        .with_structured_output(BeliefSet)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(_research_model_config(configurable, config))
    )
    result = await belief_model.ainvoke([HumanMessage(content=belief_prompt)])

    beliefs = [
        {"belief": b.belief, "score": b.score, "rationale": b.rationale}
        for b in (result.beliefs or [])
    ]
    beliefs.sort(key=lambda b: b["score"], reverse=True)

    valid = {b["belief"] for b in beliefs}
    selected_beliefs = []
    for belief in (result.selected_beliefs or []):
        if belief in valid and belief not in selected_beliefs:
            selected_beliefs.append(belief)
    # Always act on at least one belief if any were generated.
    if not selected_beliefs and beliefs:
        selected_beliefs = [beliefs[0]["belief"]]
    selected_beliefs = selected_beliefs[:max_selected]
    return beliefs, selected_beliefs


async def propose_and_score_research_questions(
    research_brief: str,
    seed_topics: list,
    configurable: Configuration,
    config: RunnableConfig,
    beliefs: list = None,
    prior_findings: str = "",
    iteration: int = 1,
    max_iterations: int = 1,
):
    """Generate, score, and select candidate research questions before delegating.

    Given the overall research brief, the supervisor's initial topic ideas, the
    working ``beliefs`` for this iteration, and a summary of everything learned
    so far (``prior_findings``), ask the model to propose
    ``num_research_proposals`` distinct candidate questions, score each from 0 to
    10, and recommend the best ones to research this iteration (at most
    ``max_selected_research_proposals``, possibly fewer).

    Returns a tuple ``(proposals, selected_topics)`` where ``proposals`` is a
    list of dicts ``{"research_topic", "score", "rationale"}`` sorted by score
    (highest first) and ``selected_topics`` is the chosen subset of topic
    strings, capped by both the selection limit and the concurrency limit.
    """
    max_selected = min(
        configurable.max_selected_research_proposals,
        configurable.max_concurrent_research_units,
    )
    seed_topics_text = "\n".join(
        f"- {topic}" for topic in seed_topics if topic
    ) or "- (No specific ideas provided.)"
    beliefs_text = "\n".join(
        f"- {belief}" for belief in (beliefs or []) if belief
    ) or "- (No specific beliefs were formed.)"
    proposal_prompt = research_question_proposal_prompt.format(
        date=get_today_str(),
        research_brief=research_brief,
        seed_topics=seed_topics_text,
        beliefs=beliefs_text,
        num_proposals=configurable.num_research_proposals,
        max_selected=max_selected,
        prior_findings=_format_prior_findings(prior_findings),
        iteration=iteration,
        max_iterations=max_iterations,
    )
    proposal_model = (
        configurable_model
        .with_structured_output(ResearchProposal)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(_research_model_config(configurable, config))
    )
    result = await proposal_model.ainvoke([HumanMessage(content=proposal_prompt)])

    # Normalize proposals into plain dicts sorted by descending score.
    proposals = [
        {
            "research_topic": p.research_topic,
            "score": p.score,
            "rationale": p.rationale,
        }
        for p in (result.proposals or [])
    ]
    proposals.sort(key=lambda p: p["score"], reverse=True)

    valid_topics = {p["research_topic"] for p in proposals}
    # Honor the model's selection, but keep only valid topics and dedupe while
    # preserving order.
    selected_topics = []
    for topic in (result.selected_research_topics or []):
        if topic in valid_topics and topic not in selected_topics:
            selected_topics.append(topic)
    # Always research at least one question per iteration if any were proposed.
    if not selected_topics and proposals:
        selected_topics = [proposals[0]["research_topic"]]
    # Enforce the maximum number of selected research units.
    selected_topics = selected_topics[:max_selected]
    return proposals, selected_topics


async def reason_over_findings(
    research_brief: str,
    beliefs: list,
    new_findings: list,
    configurable: Configuration,
    config: RunnableConfig,
) -> str:
    """Reason about an iteration's findings in light of the working beliefs.

    ``new_findings`` is a list of ``(research_topic, compressed_research)``
    tuples from this iteration. Returns the model's reasoning text assessing
    which beliefs were confirmed/refuted and what remains open.
    """
    if not new_findings:
        return ""
    beliefs_text = "\n".join(
        f"- {belief}" for belief in (beliefs or []) if belief
    ) or "- (No specific beliefs were formed.)"
    new_findings_text = "\n\n".join(
        f"=== Findings for: {topic} ===\n{compressed}"
        for topic, compressed in new_findings
    )
    reasoning_prompt = research_reasoning_prompt.format(
        date=get_today_str(),
        research_brief=research_brief,
        beliefs=beliefs_text,
        new_findings=new_findings_text,
    )
    reasoning_model = (
        configurable_model
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(_research_model_config(configurable, config))
    )
    response = await reasoning_model.ainvoke([HumanMessage(content=reasoning_prompt)])
    return getattr(response, "content", "") or ""


async def decide_research_sufficiency(
    research_brief: str,
    findings_summary: str,
    reasoning: str,
    configurable: Configuration,
    config: RunnableConfig,
):
    """Decide whether the gathered information is sufficient to stop research.

    Returns a dict ``{"is_sufficient", "reasoning", "missing_information"}``.
    """
    decision_prompt = sufficiency_decision_prompt.format(
        date=get_today_str(),
        research_brief=research_brief,
        findings_summary=findings_summary.strip() or "(No findings yet.)",
        reasoning=reasoning.strip() or "(No reasoning provided.)",
    )
    decision_model = (
        configurable_model
        .with_structured_output(SufficiencyDecision)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(_research_model_config(configurable, config))
    )
    result = await decision_model.ainvoke([HumanMessage(content=decision_prompt)])
    return {
        "is_sufficient": bool(result.is_sufficient),
        "reasoning": result.reasoning or "",
        "missing_information": result.missing_information or "",
    }


async def summarize_research_findings(
    research_brief: str,
    prior_summary: str,
    new_findings: list,
    configurable: Configuration,
    config: RunnableConfig,
) -> str:
    """Fold the latest iteration's findings into a concise running summary.

    ``new_findings`` is a list of ``(research_topic, compressed_research)``
    tuples produced in the most recent research iteration. The returned string
    is an updated running summary that integrates the new findings with
    ``prior_summary`` and highlights remaining gaps, which drives the next
    iteration.
    """
    if not new_findings:
        return prior_summary
    new_findings_text = "\n\n".join(
        f"=== Findings for: {topic} ===\n{compressed}"
        for topic, compressed in new_findings
    )
    summary_prompt = research_findings_summary_prompt.format(
        date=get_today_str(),
        research_brief=research_brief,
        prior_summary=prior_summary.strip() or "(No previous summary - this is the first iteration.)",
        new_findings=new_findings_text,
    )
    summary_model = (
        configurable_model
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(_research_model_config(configurable, config))
    )
    response = await summary_model.ainvoke([HumanMessage(content=summary_prompt)])
    return getattr(response, "content", "") or prior_summary


async def reconcile_epistemic_state(
    research_brief: str,
    existing_items: list,
    new_findings: list,
    configurable: Configuration,
    config: RunnableConfig,
) -> list:
    """Reconcile an iteration's findings against previously held beliefs/claims.

    ``existing_items`` is a list of dicts ``{"kind", "text", "node_id"}`` for the
    claims and hypotheses recorded so far. ``new_findings`` is a list of
    ``(research_topic, compressed_research)`` tuples from this iteration.

    The model identifies, for each confident relationship, whether the new
    findings ``supports``/``contradicts``/``revises``/``invalidates`` an existing
    item. Returns a list of dicts
    ``{"subject_topic", "relation", "object_text", "rationale"}`` limited to
    valid relations that reference known topics and existing item texts.
    """
    if not existing_items or not new_findings:
        return []
    existing_text = "\n".join(
        f"- [{item['kind']}] {item['text']}" for item in existing_items if item.get("text")
    )
    new_findings_text = "\n\n".join(
        f"=== Findings for: {topic} ===\n{compressed}"
        for topic, compressed in new_findings
    )
    reconciliation_prompt = epistemic_reconciliation_prompt.format(
        date=get_today_str(),
        research_brief=research_brief,
        existing_items=existing_text,
        new_findings=new_findings_text,
    )
    reconciliation_model = (
        configurable_model
        .with_structured_output(EpistemicReconciliation)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(_research_model_config(configurable, config))
    )
    try:
        result = await reconciliation_model.ainvoke(
            [HumanMessage(content=reconciliation_prompt)]
        )
    except Exception:
        # Reconciliation is best-effort; never let it break the research loop.
        return []

    valid_relations = {"supports", "contradicts", "revises", "invalidates"}
    valid_topics = {topic for topic, _ in new_findings}
    valid_texts = {item["text"].strip() for item in existing_items if item.get("text")}
    reconciled = []
    for relation in (getattr(result, "relations", None) or []):
        relation_type = (relation.relation or "").strip().lower()
        subject_topic = (relation.subject_topic or "").strip()
        object_text = (relation.object_text or "").strip()
        if relation_type not in valid_relations:
            continue
        if subject_topic not in valid_topics:
            continue
        if object_text not in valid_texts:
            continue
        reconciled.append({
            "subject_topic": subject_topic,
            "relation": relation_type,
            "object_text": object_text,
            "rationale": relation.rationale or "",
        })
    return reconciled


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    logger = get_logger(config)
    logger.set_question(get_buffer_string(state.get("messages", [])))
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        logger.node(
            scope="main",
            kind="decision",
            label="Clarify with user (skipped)",
            data={
                "allow_clarification": False,
                "available_actions": ["ask_clarifying_question", "proceed_to_research"],
                "chosen_action": "proceed_to_research",
                "rejected_actions": ["ask_clarifying_question"],
            },
        )
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        logger.node(
            scope="main",
            kind="decision",
            label="Clarify with user (question asked)",
            data={
                "allow_clarification": True,
                "available_actions": ["ask_clarifying_question", "proceed_to_research"],
                "chosen_action": "ask_clarifying_question",
                "rejected_actions": ["proceed_to_research"],
                "question": response.question,
            },
        )
        logger.finish({"status": "ended_for_clarification"})
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        logger.node(
            scope="main",
            kind="decision",
            label="Clarify with user (no clarification needed)",
            data={
                "allow_clarification": True,
                "available_actions": ["ask_clarifying_question", "proceed_to_research"],
                "chosen_action": "proceed_to_research",
                "rejected_actions": ["ask_clarifying_question"],
                "verification": response.verification,
            },
        )
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Log the generated research brief as a state in the research graph.
    logger = get_logger(config)
    logger.set_question(get_buffer_string(state.get("messages", [])))
    logger.node(
        scope="main",
        kind="brief",
        label="Write research brief",
        data={"research_brief": response.research_brief},
    )

    # Default-off experiment hook: optionally inject synthetic seed evidence
    # into the epistemic graph before supervisor reasoning begins.
    if bool(getattr(configurable, "experiment_seed_injection", False)):
        try:
            from open_deep_research.experiments.hooks import maybe_apply_seed_injection

            egraph = get_epistemic_graph(config)
            seed_result = maybe_apply_seed_injection(egraph, configurable)
            if seed_result.get("applied"):
                logger.node(
                    scope="main",
                    kind="action",
                    label="Experiment seed injection",
                    data={
                        "applied": True,
                        "count": seed_result.get("count", 0),
                        "condition": getattr(configurable, "experiment_seed_condition", None),
                    },
                )
        except Exception:
            # Experiment scaffolding must never alter normal execution.
            pass

    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Log the supervisor's decision for this iteration: which of the available
    # tools it selected and with what arguments.
    iteration = state.get("research_iterations", 0) + 1
    logger = get_logger(config)
    tool_summaries = _summarize_tool_calls(getattr(response, "tool_calls", None))
    chosen = [t["name"] for t in tool_summaries]
    available = ["think_tool", "ConductResearch", "ResearchComplete"]
    # The first supervisor decision is linked to the research brief so the main
    # flow and the supervisor lane form a single connected graph.
    supervisor_parents = None
    if iteration == 1:
        brief_node = logger.last_in_scope("main")
        if brief_node:
            supervisor_parents = [brief_node]
    supervisor_node_id = logger.node(
        scope="supervisor",
        kind="decision",
        label=f"Supervisor decision (iteration {iteration})",
        data={
            "iteration": iteration,
            "available_actions": available,
            "chosen_actions": chosen,
            "rejected_actions": [a for a in available if a not in chosen],
            "tool_calls": tool_summaries,
            "reasoning": getattr(response, "content", "") or "",
        },
        parents=supervisor_parents,
    )
    # Remember delegation nodes so spawned researchers can be linked back to the
    # supervisor iteration that created them.
    for tool_call in tool_summaries:
        if tool_call["name"] == "ConductResearch":
            logger.link_delegation(tool_call.get("research_topic", ""), supervisor_node_id)

    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    logger = get_logger(config)
    egraph = get_epistemic_graph(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        if exceeded_allowed_iterations:
            exit_reason = "exceeded_max_iterations"
        elif no_tool_calls:
            exit_reason = "no_tool_calls"
        else:
            exit_reason = "research_complete"
        logger.node(
            scope="supervisor",
            kind="end",
            label="Supervisor finished research phase",
            data={
                "exit_reason": exit_reason,
                "research_iterations": research_iterations,
                "max_researcher_iterations": configurable.max_researcher_iterations,
            },
        )
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
        # Record the supervisor's strategic reflection as a state node.
        logger.node(
            scope="supervisor",
            kind="reflection",
            label="Supervisor reflection (think_tool)",
            data={"reflection": reflection_content},
        )
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            research_brief = state.get("research_brief", "")
            scoring_enabled = configurable.enable_research_proposal_scoring

            if scoring_enabled:
                # Iterative belief-driven research loop. Each iteration forms a
                # chain of nodes (depth, not width): form beliefs -> propose &
                # score research questions -> research -> reason over findings ->
                # summarize -> decide sufficiency. The next iteration hangs below
                # the previous iteration's decision node, so the graph grows in
                # depth. The loop stops early once an iteration decides the
                # gathered information is sufficient.
                seed_topics = [
                    tool_call["args"]["research_topic"]
                    for tool_call in conduct_research_calls
                ]
                max_selected = min(
                    configurable.max_selected_research_proposals,
                    configurable.max_concurrent_research_units,
                )
                max_iterations = configurable.max_research_proposal_iterations
                findings_summary = state.get("findings_summary", "") or ""
                all_compressed_results = []
                all_raw_notes = []
                completed_iterations = 0
                # Epistemic graph working memory across iterations. Maps a
                # belief's text to its Hypothesis node id (so later iterations
                # can revise it), and accumulates every claim/hypothesis so new
                # findings can be reconciled against them.
                hypothesis_id_by_belief = {}
                epistemic_items = []
                claim_topic_by_id = {}
                pending_reopen_topics = set()
                switch_injected = False
                # The spine is the deepening chain that connects iterations. It
                # starts at the supervisor decision that requested research.
                spine = logger.last_in_scope("supervisor")

                for proposal_iteration in range(1, max_iterations + 1):
                    completed_iterations = proposal_iteration
                    # Snapshot the claims/hypotheses that existed BEFORE this
                    # iteration, so this iteration's new findings can be
                    # reconciled against them (support/contradict/revise/
                    # invalidate) without a claim reconciling against itself.
                    prior_epistemic_items = list(epistemic_items)

                    # --- 1. Belief action: propose, score, and select beliefs ---
                    beliefs, selected_beliefs = await generate_and_score_beliefs(
                        research_brief=research_brief,
                        configurable=configurable,
                        config=config,
                        prior_findings=findings_summary,
                        iteration=proposal_iteration,
                        max_iterations=max_iterations,
                    )
                    belief_node = logger.node(
                        scope="supervisor",
                        kind="belief",
                        label=f"Formed beliefs (iteration {proposal_iteration})",
                        data={
                            "iteration": proposal_iteration,
                            "max_iterations": max_iterations,
                            "num_candidates": len(beliefs),
                            "max_selected": max_selected,
                            "beliefs": beliefs,
                            "selected_beliefs": selected_beliefs,
                        },
                        parents=[spine] if spine else None,
                    )
                    spine = belief_node
                    selected_belief_set = set(selected_beliefs)
                    for b in beliefs:
                        if b["belief"] not in selected_belief_set:
                            logger.node(
                                scope="supervisor",
                                kind="rejected",
                                label="Rejected belief (low score)",
                                data={
                                    "status": "rejected_low_score",
                                    "belief": b["belief"],
                                    "score": b["score"],
                                    "rationale": b["rationale"],
                                    "iteration": proposal_iteration,
                                },
                                parents=[belief_node],
                            )

                    # Epistemic graph: each selected belief is a working
                    # Hypothesis. When a belief recurs in a later iteration (it
                    # was re-formed after seeing new evidence), the new node
                    # revises the previous one instead of overwriting it.
                    scored_belief_by_text = {b["belief"]: b for b in beliefs}
                    for belief_text in selected_beliefs:
                        scored = scored_belief_by_text.get(belief_text, {})
                        hypothesis_id = egraph.add_hypothesis(
                            text=belief_text,
                            metadata={
                                "iteration": proposal_iteration,
                                "score": scored.get("score"),
                                "rationale": scored.get("rationale", ""),
                            },
                        )
                        prior_hypothesis_id = hypothesis_id_by_belief.get(belief_text)
                        if prior_hypothesis_id and hypothesis_id:
                            egraph.revises(
                                hypothesis_id,
                                prior_hypothesis_id,
                                iteration=proposal_iteration,
                                reason="belief re-formed after new evidence",
                            )
                        if hypothesis_id:
                            hypothesis_id_by_belief[belief_text] = hypothesis_id
                            epistemic_items.append({
                                "kind": "Hypothesis",
                                "text": belief_text,
                                "node_id": hypothesis_id,
                            })

                    # --- 2. Propose, score, and select research questions ---
                    proposals, selected_topics = await propose_and_score_research_questions(
                        research_brief=research_brief,
                        seed_topics=(seed_topics if proposal_iteration == 1 else []) + sorted(pending_reopen_topics),
                        configurable=configurable,
                        config=config,
                        beliefs=selected_beliefs,
                        prior_findings=findings_summary,
                        iteration=proposal_iteration,
                        max_iterations=max_iterations,
                    )
                    proposal_node = logger.node(
                        scope="supervisor",
                        kind="proposal",
                        label=f"Proposed research questions (iteration {proposal_iteration})",
                        data={
                            "iteration": proposal_iteration,
                            "max_iterations": max_iterations,
                            "num_proposals": len(proposals),
                            "max_selected": max_selected,
                            "proposals": proposals,
                            "selected_research_topics": selected_topics,
                            "selected_beliefs": selected_beliefs,
                        },
                        parents=[spine],
                    )
                    spine = proposal_node
                    selected_set = set(selected_topics)
                    pending_reopen_topics.difference_update(selected_set)
                    # Selected proposals become dispatched delegations (chosen);
                    # the rest are rejected because of a lower score.
                    for topic in selected_topics:
                        score = next(
                            (p["score"] for p in proposals if p["research_topic"] == topic),
                            None,
                        )
                        delegate_node = logger.node(
                            scope="supervisor",
                            kind="delegate",
                            label="Delegate research unit",
                            data={
                                "status": "dispatched",
                                "research_topic": topic,
                                "score": score,
                                "iteration": proposal_iteration,
                            },
                            parents=[proposal_node],
                        )
                        logger.link_delegation(topic, delegate_node)
                    for p in proposals:
                        if p["research_topic"] not in selected_set:
                            logger.node(
                                scope="supervisor",
                                kind="rejected",
                                label="Rejected research question (low score)",
                                data={
                                    "status": "rejected_low_score",
                                    "research_topic": p["research_topic"],
                                    "score": p["score"],
                                    "rationale": p["rationale"],
                                    "iteration": proposal_iteration,
                                },
                                parents=[proposal_node],
                            )

                    # --- 3. Research: execute the selected questions in parallel ---
                    research_tasks = [
                        researcher_subgraph.ainvoke({
                            "researcher_messages": [
                                HumanMessage(content=topic)
                            ],
                            "research_topic": topic
                        }, config)
                        for topic in selected_topics
                    ]
                    tool_results = await asyncio.gather(*research_tasks)

                    # Record each synthesized finding, linked to its delegation.
                    round_results = []
                    finding_nodes = []
                    claim_id_by_topic = {}
                    for observation, topic in zip(tool_results, selected_topics):
                        compressed = observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded")
                        round_results.append((topic, compressed))
                        all_compressed_results.append((topic, compressed))
                        finding_node = logger.node(
                            scope="supervisor",
                            kind="finding",
                            label="Research unit result returned",
                            data={
                                "research_topic": topic,
                                "compressed_research": compressed,
                                "iteration": proposal_iteration,
                            },
                            parents=[p for p in [logger.delegation_parent(topic)] if p],
                        )
                        finding_nodes.append(finding_node)
                        all_raw_notes.extend(observation.get("raw_notes", []))

                        # Epistemic graph: a finding forms an intermediate Claim.
                        # The topic's Evidence supports and compresses into the
                        # Claim, and the Claim depends on the beliefs that guided
                        # this iteration's research.
                        claim_id = egraph.add_claim(
                            text=compressed,
                            metadata={
                                "research_topic": topic,
                                "iteration": proposal_iteration,
                            },
                        )
                        claim_id_by_topic[topic] = claim_id
                        if claim_id:
                            claim_topic_by_id[claim_id] = topic
                        evidence_id = egraph.get_topic_evidence(topic)
                        if evidence_id and claim_id:
                            egraph.supports(evidence_id, claim_id, research_topic=topic)
                            egraph.compresses(evidence_id, claim_id, research_topic=topic)
                        for belief_text in selected_beliefs:
                            hypothesis_id = hypothesis_id_by_belief.get(belief_text)
                            if hypothesis_id and claim_id:
                                egraph.depends_on(
                                    claim_id,
                                    hypothesis_id,
                                    iteration=proposal_iteration,
                                )
                        if claim_id:
                            epistemic_items.append({
                                "kind": "Claim",
                                "text": compressed,
                                "node_id": claim_id,
                            })

                    # --- 4. Reasoning: assess findings against the beliefs ---
                    # Reasoning hangs BELOW the findings so each iteration is
                    # strictly deeper than its research branch.
                    reasoning_text = await reason_over_findings(
                        research_brief=research_brief,
                        beliefs=selected_beliefs,
                        new_findings=round_results,
                        configurable=configurable,
                        config=config,
                    )
                    reasoning_node = logger.node(
                        scope="supervisor",
                        kind="reasoning",
                        label=f"Reasoned over findings (iteration {proposal_iteration})",
                        data={
                            "iteration": proposal_iteration,
                            "reasoning": reasoning_text,
                            "selected_beliefs": selected_beliefs,
                        },
                        parents=finding_nodes or [spine],
                    )
                    spine = reasoning_node

                    # --- 5. Summarize findings into the running summary ---
                    findings_summary = await summarize_research_findings(
                        research_brief=research_brief,
                        prior_summary=findings_summary,
                        new_findings=round_results,
                        configurable=configurable,
                        config=config,
                    )
                    summary_node = logger.node(
                        scope="supervisor",
                        kind="reflection",
                        label=f"Summarized findings (iteration {proposal_iteration})",
                        data={
                            "iteration": proposal_iteration,
                            "findings_summary": findings_summary,
                        },
                        parents=[spine],
                    )
                    spine = summary_node

                    # Epistemic graph: reconcile this iteration's findings with
                    # the claims/hypotheses held before it. Each relationship
                    # becomes a typed edge from the new topic's Claim to the
                    # existing item, preserving history instead of overwriting.
                    if prior_epistemic_items:
                        reconciled = await reconcile_epistemic_state(
                            research_brief=research_brief,
                            existing_items=prior_epistemic_items,
                            new_findings=round_results,
                            configurable=configurable,
                            config=config,
                        )
                        item_node_by_text = {
                            item["text"].strip(): item["node_id"]
                            for item in prior_epistemic_items
                        }
                        for relation in reconciled:
                            subject_id = claim_id_by_topic.get(relation["subject_topic"])
                            object_id = item_node_by_text.get(relation["object_text"].strip())
                            if not subject_id or not object_id:
                                continue
                            edge_meta = {
                                "iteration": proposal_iteration,
                                "rationale": relation["rationale"],
                            }
                            egraph.add_edge(
                                relation["relation"], subject_id, object_id, edge_meta
                            )

                    # Phase 5 monitor: after observed graph updates are applied,
                    # check all active commitments for stranded consistency.
                    if (
                        world_model_enabled(config)
                        and bool(getattr(configurable, "wm_enable_rollback", False))
                        and str(getattr(configurable, "wm_recovery_mode", "rollback")) == "rollback"
                    ):
                        try:
                            from open_deep_research.world_model import get_world_model
                            from open_deep_research.world_model_logger import get_world_model_logger
                            from open_deep_research.world_model_monitor import monitor as wm_monitor
                            from open_deep_research.world_model_rollback import apply_repair, reduced_repair

                            wm = get_world_model(config)
                            wm_logger = get_world_model_logger(config, question=research_brief)
                            fired = wm_monitor(egraph, wm.active_commitments, configurable)
                            wm_logger.monitor_check({
                                "step": proposal_iteration,
                                "n_active": len(wm.active_commitments),
                                "n_fired": len(fired),
                            })

                            for fired_item in fired:
                                commitment_record = fired_item.get("commitment_record", {})
                                detail = fired_item.get("detail", {})
                                plan = reduced_repair(egraph, fired_item, configurable)
                                commitment_id = str(commitment_record.get("commitment_id") or "")
                                claim_id = str(commitment_record.get("claim_id") or "")

                                trigger_payload = {
                                    "step": proposal_iteration,
                                    "commitment_id": commitment_id,
                                    "claim_id": claim_id,
                                    "offending_evidence_ids": detail.get("offending_evidence_ids", []),
                                    "beta_before": detail.get("beta_before"),
                                    "beta_after": detail.get("beta_now"),
                                    "intended_plan": {
                                        "contested_claims": list(plan.contested_claims),
                                        "retract_nodes": list(plan.retract_nodes),
                                        "retract_edges": list(plan.retract_edges),
                                        "regenerate_nodes": list(plan.regenerate_nodes),
                                        "preserved_nodes": list(plan.preserved_nodes),
                                        "reopen_claims": list(plan.reopen_claims),
                                    },
                                }

                                enforced = False
                                if bool(getattr(configurable, "wm_enforce_rollback", False)):
                                    apply_repair(egraph, plan, configurable)
                                    wm.unregister_active_commitment(commitment_id)
                                    for reopen_claim_id in plan.reopen_claims:
                                        reopen_topic = claim_topic_by_id.get(reopen_claim_id)
                                        if not reopen_topic:
                                            reopen_node = egraph.get_node(reopen_claim_id) or {}
                                            reopen_topic = reopen_node.get("text")
                                        if reopen_topic:
                                            pending_reopen_topics.add(reopen_topic)
                                    enforced = True

                                wm_logger.trigger_fired(trigger_payload)
                                wm_logger.rollback({
                                    **trigger_payload,
                                    "enforced": enforced,
                                    "contested_claims": list(plan.contested_claims),
                                    "retract_nodes": list(plan.retract_nodes),
                                    "retract_edges": list(plan.retract_edges),
                                    "regenerate_nodes": list(plan.regenerate_nodes),
                                    "preserved_nodes": list(plan.preserved_nodes),
                                    "reopen_claims": list(plan.reopen_claims),
                                })
                        except Exception:
                            # Monitoring and rollback scaffolding must never
                            # alter core research behavior.
                            pass

                    # --- 6. Decision: is the information sufficient? ---
                    decision = await decide_research_sufficiency(
                        research_brief=research_brief,
                        findings_summary=findings_summary,
                        reasoning=reasoning_text,
                        configurable=configurable,
                        config=config,
                    )
                    is_last_iteration = proposal_iteration >= max_iterations
                    will_continue = (
                        (not decision["is_sufficient"]) or bool(pending_reopen_topics)
                    ) and (not is_last_iteration)
                    decision_node = logger.node(
                        scope="supervisor",
                        kind="decision",
                        label=f"Sufficiency decision (iteration {proposal_iteration})",
                        data={
                            "iteration": proposal_iteration,
                            "available_actions": ["continue_research", "end_research"],
                            "chosen_action": "continue_research" if will_continue else "end_research",
                            "rejected_actions": ["end_research"] if will_continue else ["continue_research"],
                            "is_sufficient": decision["is_sufficient"],
                            "reasoning": decision["reasoning"],
                            "missing_information": decision["missing_information"],
                            "reached_max_iterations": is_last_iteration,
                        },
                        parents=[spine],
                    )
                    spine = decision_node

                    # Epistemic graph: when the agent judges the evidence
                    # sufficient it is making a stronger Commitment to this
                    # iteration's claims (and the hypotheses behind them).
                    if decision["is_sufficient"]:
                        suppressed_any_commitment = False
                        for topic, claim_id in claim_id_by_topic.items():
                            if not claim_id:
                                continue
                            should_commit = True
                            prediction = None
                            target_hypothesis_ids = [
                                hypothesis_id_by_belief[belief_text]
                                for belief_text in selected_beliefs
                                if hypothesis_id_by_belief.get(belief_text)
                            ]
                            if world_model_enabled(config):
                                try:
                                    from open_deep_research.world_model import get_world_model
                                    from open_deep_research.world_model_logger import get_world_model_logger

                                    wm = get_world_model(config)
                                    wm_logger = get_world_model_logger(config, question=research_brief)
                                    action = {
                                        "action_id": f"commit:{proposal_iteration}:{topic}",
                                        "target_node_ids": [claim_id, *target_hypothesis_ids],
                                        "metadata": {
                                            "iteration": proposal_iteration,
                                            "research_topic": topic,
                                        },
                                    }
                                    prediction = wm.predict(egraph, action)
                                    wm_enforce = bool(getattr(configurable, "wm_enforce_decision", False))
                                    enforced = wm_enforce and prediction.decision == "not_commit"
                                    prediction.wm_enforce_decision = wm_enforce
                                    prediction.enforced = enforced
                                    wm_logger.prediction(prediction)
                                    if enforced:
                                        should_commit = False
                                        suppressed_any_commitment = True
                                except Exception:
                                    # World-model scaffolding must never alter
                                    # core research behavior.
                                    pass
                            if not should_commit:
                                # Active mode: retain as contested and continue
                                # researching rather than locking in commitment.
                                contested_marker_id = egraph.add_plan_step(
                                    text=f"Retain as contested for: {topic}",
                                    metadata={
                                        "iteration": proposal_iteration,
                                        "research_topic": topic,
                                        "status": "retain_as_contested",
                                        "reason": "world_model_not_commit",
                                    },
                                )
                                if contested_marker_id:
                                    egraph.depends_on(contested_marker_id, claim_id)
                                continue
                            commitment_id = egraph.add_commitment(
                                text=f"Committed to findings for: {topic}",
                                target=claim_id,
                                metadata={
                                    "iteration": proposal_iteration,
                                    "research_topic": topic,
                                    "reason": decision["reasoning"],
                                    "wm_theta": (prediction.resolved_theta if prediction else None),
                                    "wm_decision": (prediction.decision if prediction else None),
                                },
                            )
                            if commitment_id and world_model_enabled(config):
                                try:
                                    from open_deep_research.world_model import get_world_model
                                    from open_deep_research.world_model_scoring import compute_belief, take_snapshot

                                    wm = get_world_model(config)
                                    beta_at_commit = compute_belief(take_snapshot(egraph), claim_id, configurable)
                                    wm.register_active_commitment(
                                        commitment_id=commitment_id,
                                        claim_id=claim_id,
                                        hypothesis_id=(target_hypothesis_ids[0] if target_hypothesis_ids else None),
                                        theta=(prediction.resolved_theta if prediction else None),
                                        step=proposal_iteration,
                                    )
                                    wm.active_commitments[commitment_id]["beta_at_commit"] = beta_at_commit
                                except Exception:
                                    pass

                            # Default-off experiment hook: inject scheduled
                            # switch evidence after the first commitment or an
                            # n-step schedule, then let monitor/rollback react.
                            if not switch_injected and bool(getattr(configurable, "experiment_switch_enabled", False)):
                                try:
                                    from open_deep_research.experiments.hooks import (
                                        maybe_apply_switch_injection,
                                        should_fire_switch,
                                    )

                                    directive = getattr(configurable, "experiment_switch_directive", None)
                                    if should_fire_switch(
                                        configurable=configurable,
                                        switch_directive=directive,
                                        first_commitment_seen=bool(commitment_id),
                                        step_index=proposal_iteration,
                                    ):
                                        switch_result = maybe_apply_switch_injection(
                                            egraph=egraph,
                                            configurable=configurable,
                                            switch_directive=directive,
                                            commitment_claim_id=claim_id,
                                        )
                                        if switch_result.get("applied"):
                                            logger.node(
                                                scope="supervisor",
                                                kind="action",
                                                label="Experiment switch injection",
                                                data={
                                                    "iteration": proposal_iteration,
                                                    "applied": True,
                                                    "count": switch_result.get("count", 0),
                                                    "commitment_id": commitment_id,
                                                    "claim_id": claim_id,
                                                },
                                                parents=[spine] if spine else None,
                                            )
                                            switch_injected = True
                                except Exception:
                                    pass
                            for belief_text in selected_beliefs:
                                hypothesis_id = hypothesis_id_by_belief.get(belief_text)
                                if commitment_id and hypothesis_id:
                                    egraph.depends_on(
                                        commitment_id,
                                        hypothesis_id,
                                        iteration=proposal_iteration,
                                    )

                        if suppressed_any_commitment:
                            # Active mode chose not_commit for at least one item;
                            # do not terminate this proposal iteration as sufficient.
                            continue

                    # Stop early when the information is judged sufficient.
                    if decision["is_sufficient"] and not pending_reopen_topics:
                        break

                # The iterative loop owns the research phase end-to-end. Return
                # the consolidated findings to the supervisor's tool calls and
                # end the research phase.
                if all_compressed_results:
                    consolidated = "\n\n".join(
                        f"=== Research findings for: {topic} ===\n{compressed}"
                        for topic, compressed in all_compressed_results
                    )
                else:
                    consolidated = "No research findings were produced."
                for index, tool_call in enumerate(conduct_research_calls):
                    content = consolidated if index == 0 else (
                        "See the consolidated research findings returned for the related ConductResearch call."
                    )
                    all_tool_messages.append(ToolMessage(
                        content=content,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ))

                logger.node(
                    scope="supervisor",
                    kind="end",
                    label="Supervisor finished iterative research",
                    data={
                        "exit_reason": "research_sufficient" if completed_iterations < max_iterations else "reached_max_iterations",
                        "completed_iterations": completed_iterations,
                        "max_research_proposal_iterations": max_iterations,
                        "num_findings": len(all_compressed_results),
                    },
                    parents=[spine] if spine else None,
                )

                end_messages = supervisor_messages + all_tool_messages
                return Command(
                    goto=END,
                    update={
                        "supervisor_messages": all_tool_messages,
                        "notes": get_notes_from_tool_calls(end_messages),
                        "raw_notes": ["\n".join(all_raw_notes)] if all_raw_notes else [],
                        "findings_summary": findings_summary,
                        "research_brief": research_brief,
                    }
                )

            # Original behavior (scoring disabled): dispatch the supervisor's own
            # topics, capped by the concurrency limit.
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            selected_topics = [
                tool_call["args"]["research_topic"]
                for tool_call in allowed_conduct_research_calls
            ]
            # Record which research units were dispatched (chosen) and which
            # were dropped because of the concurrency limit (rejected).
            for tool_call in allowed_conduct_research_calls:
                topic = tool_call["args"]["research_topic"]
                delegate_node = logger.node(
                    scope="supervisor",
                    kind="delegate",
                    label="Delegate research unit",
                    data={
                        "status": "dispatched",
                        "research_topic": topic,
                        "max_concurrent_research_units": configurable.max_concurrent_research_units,
                    },
                )
                # Link the researcher subgraph for this topic to its delegation.
                logger.link_delegation(topic, delegate_node)
            for tool_call in overflow_conduct_research_calls:
                logger.node(
                    scope="supervisor",
                    kind="rejected",
                    label="Rejected research unit (concurrency limit)",
                    data={
                        "status": "rejected_concurrency_limit",
                        "research_topic": tool_call["args"]["research_topic"],
                        "max_concurrent_research_units": configurable.max_concurrent_research_units,
                    },
                )

            # Execute the selected research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=topic)
                    ],
                    "research_topic": topic
                }, config)
                for topic in selected_topics
            ]

            tool_results = await asyncio.gather(*research_tasks)

            # Record each synthesized finding, linked back to its delegation node.
            compressed_results = []
            for observation, topic in zip(tool_results, selected_topics):
                compressed = observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded")
                compressed_results.append((topic, compressed))
                logger.node(
                    scope="supervisor",
                    kind="finding",
                    label="Research unit result returned",
                    data={
                        "research_topic": topic,
                        "compressed_research": compressed,
                    },
                    parents=[p for p in [logger.delegation_parent(topic)] if p],
                )
                # Epistemic graph: the finding forms a Claim supported and
                # compressed by the topic's Evidence.
                claim_id = egraph.add_claim(
                    text=compressed,
                    metadata={"research_topic": topic},
                )
                evidence_id = egraph.get_topic_evidence(topic)
                if evidence_id and claim_id:
                    egraph.supports(evidence_id, claim_id, research_topic=topic)
                    egraph.compresses(evidence_id, claim_id, research_topic=topic)

            # Create tool messages with research results (1:1 with calls)
            for (topic, compressed), tool_call in zip(compressed_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=compressed,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                logger.node(
                    scope="supervisor",
                    kind="end",
                    label="Supervisor finished (research execution error)",
                    data={
                        "exit_reason": "research_execution_error",
                        "error": str(e),
                    },
                )
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "base_url": get_base_url_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Log the researcher's decision: which tools (search/think/complete) it chose
    # and the search queries it dispatched. The first decision for a topic is
    # linked back to the supervisor delegation node that spawned this researcher.
    research_topic = state.get("research_topic", "")
    iteration = state.get("tool_call_iterations", 0) + 1
    logger = get_logger(config)
    scope = f"researcher:{research_topic}"
    tool_summaries = _summarize_tool_calls(getattr(response, "tool_calls", None))
    chosen = [t["name"] for t in tool_summaries]
    available = sorted({tool.name for tool in tools if hasattr(tool, "name")})
    queries = [q for t in tool_summaries if t["name"] == "tavily_search" for q in t.get("queries", [])]
    parents = None
    if iteration == 1:
        delegation_parent = logger.delegation_parent(research_topic)
        if delegation_parent:
            parents = [delegation_parent]
    logger.node(
        scope=scope,
        kind="decision",
        label=f"Researcher decision (iteration {iteration})",
        data={
            "research_topic": research_topic,
            "iteration": iteration,
            "available_actions": available,
            "chosen_actions": chosen,
            "rejected_actions": [a for a in available if a not in chosen],
            "search_queries": queries,
            "tool_calls": tool_summaries,
            "reasoning": getattr(response, "content", "") or "",
        },
        parents=parents,
    )

    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    logger = get_logger(config)
    egraph = get_epistemic_graph(config)
    research_topic = state.get("research_topic", "")
    scope = f"researcher:{research_topic}"
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        logger.node(
            scope=scope,
            kind="end",
            label="Researcher finished (no tool calls)",
            data={"research_topic": research_topic, "exit_reason": "no_tool_calls"},
        )
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # Record each executed tool as a state node: searches with their queries and
    # retrieved sources, and reflections with their content.
    for observation, tool_call in zip(observations, tool_calls):
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {}) or {}
        if name == "tavily_search":
            sources = _parse_search_sources(observation)
            logger.node(
                scope=scope,
                kind="search",
                label="Web search executed",
                data={
                    "research_topic": research_topic,
                    "queries": args.get("queries", []),
                    "num_sources": len(sources),
                    "sources": sources,
                },
            )
            # Epistemic graph: every retrieved source becomes a Source node so
            # later evidence/claims can cite it.
            source_ids = []
            for source in sources:
                source_id = egraph.add_source(
                    title=source.get("title", ""),
                    url=source.get("url", ""),
                    metadata={
                        "research_topic": research_topic,
                        "queries": args.get("queries", []),
                    },
                )
                if source_id:
                    source_ids.append(source_id)
            egraph.record_topic_sources(research_topic, source_ids)
        elif name == "think_tool":
            logger.node(
                scope=scope,
                kind="reflection",
                label="Researcher reflection (think_tool)",
                data={"research_topic": research_topic, "reflection": args.get("reflection", "")},
            )
        elif name == "ResearchComplete":
            logger.node(
                scope=scope,
                kind="action",
                label="Researcher signalled ResearchComplete",
                data={"research_topic": research_topic},
            )
        else:
            logger.node(
                scope=scope,
                kind="tool",
                label=f"Tool executed: {name}",
                data={
                    "research_topic": research_topic,
                    "tool": name,
                    "args": args,
                    "result": observation if isinstance(observation, str) else str(observation),
                },
            )
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        logger.node(
            scope=scope,
            kind="end",
            label="Researcher finished research loop",
            data={
                "research_topic": research_topic,
                "exit_reason": "exceeded_max_tool_calls" if exceeded_iterations else "research_complete",
                "tool_call_iterations": state.get("tool_call_iterations", 0),
                "max_react_tool_calls": configurable.max_react_tool_calls,
            },
        )
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    logger = get_logger(config)
    egraph = get_epistemic_graph(config)
    research_topic = state.get("research_topic", "")
    scope = f"researcher:{research_topic}"
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "base_url": get_base_url_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            logger.node(
                scope=scope,
                kind="compress",
                label="Compressed research findings",
                data={
                    "research_topic": research_topic,
                    "compressed_research": str(response.content),
                    "status": "success",
                },
            )
            # Epistemic graph: the compressed research is Evidence extracted and
            # summarized from the topic's sources. Link it to those sources with
            # derived_from/cites, and record a compresses edge from each source.
            topic_sources = egraph.get_topic_sources(research_topic)
            evidence_id = egraph.add_evidence(
                text=str(response.content),
                metadata={"research_topic": research_topic, "status": "success"},
                sources=topic_sources,
            )
            for source_id in topic_sources:
                egraph.compresses(evidence_id, source_id, research_topic=research_topic)
            egraph.record_topic_evidence(research_topic, evidence_id)
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    logger.node(
        scope=scope,
        kind="compress",
        label="Compression failed (max retries exceeded)",
        data={"research_topic": research_topic, "status": "error"},
    )
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    logger = get_logger(config)
    egraph = get_epistemic_graph(config)
    # Link the final report node to the end of the supervisor lane (falling back
    # to the main lane) so it joins the rest of the state graph.
    report_parents = [p for p in [logger.last_in_scope("supervisor"), logger.last_in_scope("main")] if p]
    report_parents = report_parents[:1]

    def _record_draft_fragments(report_text: str, status: str) -> None:
        """Record the final answer as DraftFragment nodes in the epistemic graph.

        The whole report becomes an overarching DraftFragment that every
        recorded Claim and Evidence node is ``used_in``; each markdown section
        becomes its own DraftFragment ``derived_from`` the overarching draft, so
        the epistemic graph shows which knowledge fed the written answer.
        """
        text = str(report_text or "")
        if not text.strip():
            return
        answer_id = egraph.add_draft_fragment(
            text=text,
            metadata={"role": "final_report", "status": status, "num_notes": len(notes)},
            label="Final report",
        )
        if not answer_id:
            return
        for claim_id in egraph.all_claim_ids():
            egraph.used_in(claim_id, answer_id, role="final_report")
        for evidence_id in egraph.all_evidence_ids():
            egraph.used_in(evidence_id, answer_id, role="final_report")
        for heading, section_text in _split_report_sections(text):
            section_id = egraph.add_draft_fragment(
                text=section_text,
                metadata={"role": "section", "heading": heading, "status": status},
                label=heading or "Section",
            )
            egraph.derived_from(section_id, answer_id, role="section")

    def _log_report(report_text: str, status: str) -> None:
        """Record the final report state node and close the run log."""
        logger.node(
            scope="main",
            kind="report",
            label="Generate final report",
            data={
                "status": status,
                "num_notes": len(notes),
                "final_report": str(report_text),
            },
            parents=report_parents or None,
        )
        _record_draft_fragments(report_text, status)
        if world_model_enabled(config):
            try:
                from open_deep_research.world_model_logger import get_world_model_logger

                get_world_model_logger(config).finish()
            except Exception:
                # World-model scaffolding must never alter report generation.
                pass
        egraph.finish({"status": status})
        logger.finish({"status": status})

    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "base_url": get_base_url_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # Return successful report generation
            _log_report(final_report.content, "success")
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        error_report = f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}"
                        _log_report(error_report, "error_token_limit")
                        return {
                            "final_report": error_report,
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                _log_report(f"Error generating final report: {e}", "error")
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    _log_report("Error generating final report: Maximum retries exceeded", "error_max_retries")
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
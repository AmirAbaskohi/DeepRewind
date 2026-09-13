"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Literal, Optional

try:
    from langchain_core.runnables import RunnableConfig
except Exception:  # pragma: no cover - fallback for minimal test envs
    RunnableConfig = dict[str, Any]
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    enable_research_logging: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to log every step the agent takes as a research state graph (JSONL) for later visualization."
            }
        }
    )
    research_log_dir: str = Field(
        default="research_logs",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "research_logs",
                "description": "Directory where research state-graph logs (JSONL) are written when logging is enabled."
            }
        }
    )
    enable_epistemic_graph: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to record the agent's evolving epistemic state (sources, evidence, claims, hypotheses, assumptions, commitments, draft fragments) as an epistemic graph (JSONL) for later visualization."
            }
        }
    )
    epistemic_graph_dir: str = Field(
        default="epistemic_graphs",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "epistemic_graphs",
                "description": "Directory where epistemic-graph files (JSONL) are written when epistemic-graph recording is enabled. Kept separate from the research state-graph logs."
            }
        }
    )
    enable_world_model: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Whether to enable world-model commitment prediction scaffolding and logging."
            }
        }
    )
    world_model_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model configured for the world model. In Phase 1 this is logged but not invoked."
            }
        }
    )
    world_model_log_dir: str = Field(
        default="world_model_logs",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "world_model_logs",
                "description": "Directory where world-model logs (JSONL) are written when world-model logging is enabled."
            }
        }
    )
    world_model_max_tokens: int = Field(
        default=2000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 2000,
                "description": "Maximum output tokens configured for the world model."
            }
        }
    )
    wm_prediction_temperature: float = Field(
        default=0.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.0,
                "description": "Temperature for world-model predictions."
            }
        }
    )
    wm_max_retries: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 2,
                "min": 0,
                "max": 10,
                "description": "Maximum world-model regeneration attempts after invalid JSON output."
            }
        }
    )
    wm_alpha_kappa: float = Field(
        default=0.34,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.34,
                "description": "Placeholder IRR weight for kappa. Used in later world-model phases."
            }
        }
    )
    wm_alpha_lambda: float = Field(
        default=0.33,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.33,
                "description": "Placeholder IRR weight for lambda. Used in later world-model phases."
            }
        }
    )
    wm_alpha_gamma: float = Field(
        default=0.33,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.33,
                "description": "Placeholder IRR weight for gamma. Used in later world-model phases."
            }
        }
    )
    wm_tau_commit: float = Field(
        default=0.5,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.5,
                "description": "Placeholder commit threshold for later world-model gating phases."
            }
        }
    )
    wm_eta: float = Field(
        default=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1.0,
                "description": "Placeholder utility penalty weight for later world-model phases."
            }
        }
    )
    wm_khop: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 2,
                "min": 1,
                "max": 6,
                "description": "Neighborhood hop count used by the world-model encoder subgraph extraction."
            }
        }
    )
    wm_encode_message_passing: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether world-model encoding includes short neighbor context inline for each node."
            }
        }
    )
    wm_max_context_nodes: int = Field(
        default=40,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 40,
                "min": 5,
                "max": 200,
                "description": "Maximum nodes included in encoded world-model context."
            }
        }
    )
    wm_beta0: float = Field(
        default=0.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.0,
                "description": "Belief prior beta0 for world-model grounded claim belief computation."
            }
        }
    )
    wm_plausibility_temp: float = Field(
        default=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1.0,
                "description": "Temperature for grounded hypothesis plausibility softmax."
            }
        }
    )
    wm_default_reliability: float = Field(
        default=0.5,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.5,
                "description": "Default source reliability when missing from source metadata."
            }
        }
    )
    wm_default_edge_weight: float = Field(
        default=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1.0,
                "description": "Default edge weight when missing for grounded scoring."
            }
        }
    )
    wm_commit_lockin_strength: float = Field(
        default=4.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4.0,
                "description": "Pseudo-support added to committed hypothesis in post-commit simulation."
            }
        }
    )
    wm_recovery_budget: float = Field(
        default=10.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10.0,
                "description": "Recovery budget used in grounded recovery-cost normalization."
            }
        }
    )
    wm_cost_retract: float = Field(
        default=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1.0,
                "description": "Per-node retract cost used by grounded gamma computation."
            }
        }
    )
    wm_cost_regen: float = Field(
        default=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1.0,
                "description": "Per-node regeneration cost used by grounded gamma computation."
            }
        }
    )
    wm_theta_rho_star: float = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.7,
                "description": "Default reliability threshold for rollback trigger."
            }
        }
    )
    wm_theta_w_star: float = Field(
        default=0.6,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.6,
                "description": "Default edge-weight threshold for rollback trigger."
            }
        }
    )
    wm_contested_band: float = Field(
        default=0.1,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.1,
                "description": "Belief band around 0.5 for contested-claim detection."
            }
        }
    )
    wm_contested_con_min: float = Field(
        default=0.5,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.5,
                "description": "Minimum normalized contradiction mass for contested-claim detection."
            }
        }
    )
    wm_tc_source: Literal["structural", "constant", "llm"] = Field(
        default="structural",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "structural",
                "description": "Source for trigger-coverage estimation.",
                "options": [
                    {"label": "Structural", "value": "structural"},
                    {"label": "Constant", "value": "constant"},
                    {"label": "LLM", "value": "llm"}
                ]
            }
        }
    )
    wm_tc_per_probe: float = Field(
        default=0.34,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.34,
                "description": "Trigger-coverage contribution per active structural probe."
            }
        }
    )
    wm_default_tc: float = Field(
        default=0.5,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.5,
                "description": "Default trigger coverage for constant/fallback modes."
            }
        }
    )
    wm_tau1: float = Field(
        default=0.33,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.33,
                "description": "Lower irreversibility class threshold."
            }
        }
    )
    wm_tau2: float = Field(
        default=0.66,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.66,
                "description": "Upper irreversibility class threshold."
            }
        }
    )
    wm_value_source: Literal["zero", "plausibility"] = Field(
        default="zero",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "zero",
                "description": "Value term source for world-model utility.",
                "options": [
                    {"label": "Zero", "value": "zero"},
                    {"label": "Plausibility", "value": "plausibility"}
                ]
            }
        }
    )
    wm_gate_mode: Literal["threshold", "utility"] = Field(
        default="threshold",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "threshold",
                "description": "Binary decision gate mode for world-model commit control.",
                "options": [
                    {"label": "Threshold", "value": "threshold"},
                    {"label": "Utility", "value": "utility"}
                ]
            }
        }
    )
    wm_u_commit_threshold: float = Field(
        default=0.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.0,
                "description": "Utility threshold for commit when wm_gate_mode is utility."
            }
        }
    )
    wm_enforce_decision: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Whether to actively enforce not_commit decisions (default shadow mode)."
            }
        }
    )
    wm_enable_rollback: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Master rollback monitor switch. When false, monitor/repair never run and behavior matches Phase 4."
            }
        }
    )
    wm_recovery_mode: Literal["none", "rollback"] = Field(
        default="rollback",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "rollback",
                "description": "Recovery strategy mode. Currently supports none or reduced rollback.",
                "options": [
                    {"label": "None", "value": "none"},
                    {"label": "Rollback", "value": "rollback"}
                ]
            }
        }
    )
    wm_enforce_rollback: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Whether rollback plans are actively applied. False logs intended repairs only (shadow mode)."
            }
        }
    )
    wm_beta_star: float = Field(
        default=0.5,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.5,
                "description": "Consistency threshold beta*. Commitments are consistent when beta(claim) >= beta*."
            }
        }
    )
    wm_preserve_independent: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Preserve independently supported nodes during reduced rollback repairs."
            }
        }
    )
    wm_regenerate_dependents: Literal["mark_stale", "drop"] = Field(
        default="mark_stale",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "mark_stale",
                "description": "How to regenerate dependent fragments in reduced rollback mode.",
                "options": [
                    {"label": "Mark stale", "value": "mark_stale"},
                    {"label": "Drop", "value": "drop"}
                ]
            }
        }
    )
    experiment_seed_injection: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable synthetic initial-condition seed injection hooks for experiment runs."
            }
        }
    )
    experiment_switch_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable mid-run alternative-switch injection hooks for experiment runs."
            }
        }
    )
    experiment_switch_after: Literal["first_commitment", "n_steps"] = Field(
        default="first_commitment",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "first_commitment",
                "description": "Switch injection schedule selector for experiments.",
                "options": [
                    {"label": "First commitment", "value": "first_commitment"},
                    {"label": "N steps", "value": "n_steps"}
                ]
            }
        }
    )
    experiment_switch_n: int = Field(
        default=1,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1,
                "min": 1,
                "description": "Step index used when experiment_switch_after is n_steps."
            }
        }
    )
    experiment_injected_evidence: Optional[List[dict[str, Any]]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Optional synthetic evidence payload used by experiment hooks."
            }
        }
    )
    experiment_seed_condition: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Initial-condition experiment label attached to synthetic nodes."
            }
        }
    )
    experiment_switch_directive: Optional[dict[str, Any]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Optional switch directive payload consumed by experiment hooks."
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    enable_research_proposal_scoring: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "When delegating research, first propose multiple candidate research questions, score each, then research only the best ones."
            }
        }
    )
    num_research_proposals: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 2,
                "max": 20,
                "step": 1,
                "description": "How many candidate research questions to propose and score before selecting the best ones to research."
            }
        }
    )
    max_research_proposal_iterations: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of belief-research-reasoning iterations. Each iteration forms beliefs, proposes and scores research questions, researches the best ones, reasons over the findings, summarizes, and decides whether the information is sufficient. The loop stops early once an iteration decides the information is sufficient."
            }
        }
    )
    max_selected_research_proposals: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of scored research questions to actually research. Fewer may be selected when additional questions would be redundant."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True


def world_model_enabled(config: Optional[dict]) -> bool:
    """Return whether world-model scaffolding should be active for this run.

    Controlled by the ``enable_world_model`` configurable field (or the
    ``ENABLE_WORLD_MODEL`` environment variable). Defaults to disabled so
    existing behavior remains unchanged unless explicitly enabled.
    """
    env = os.environ.get("ENABLE_WORLD_MODEL")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off", "")
    configurable = (config or {}).get("configurable", {}) if config else {}
    value = configurable.get("enable_world_model")
    if value is None:
        return False
    return bool(value)
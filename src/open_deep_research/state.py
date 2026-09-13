"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class ScoredResearchQuestion(BaseModel):
    """A candidate research question with a quality score."""

    research_topic: str = Field(
        description="A standalone research question/topic, described in high detail (at least a paragraph). Do NOT use acronyms or abbreviations.",
    )
    score: float = Field(
        description="A score from 0 (useless) to 10 (essential) rating how valuable, relevant, non-redundant, and feasible this research question is for answering the overall research brief.",
    )
    rationale: str = Field(
        description="A concise justification for the assigned score.",
    )

class ResearchProposal(BaseModel):
    """A set of candidate research questions, scored and ranked, with the best ones selected."""

    proposals: list[ScoredResearchQuestion] = Field(
        description="A list of distinct candidate research questions, each scored from 0 to 10.",
    )
    selected_research_topics: list[str] = Field(
        description="The best research topics chosen from the proposals (each must exactly match the 'research_topic' of one proposal). Select the strongest non-overlapping questions only. Never select more than the allowed maximum; selecting fewer is encouraged when additional questions would be redundant or low value.",
    )

class ScoredBelief(BaseModel):
    """A candidate belief/assumption about the research topic with a quality score."""

    belief: str = Field(
        description="A clear, falsifiable hypothesis or working assumption about the research topic that could guide what to investigate.",
    )
    score: float = Field(
        description="A score from 0 (useless) to 10 (essential) rating how plausible, relevant to the brief, and useful this belief is for guiding productive research.",
    )
    rationale: str = Field(
        description="A concise justification for the assigned score.",
    )

class BeliefSet(BaseModel):
    """A set of candidate beliefs, scored and ranked, with the best ones selected."""

    beliefs: list[ScoredBelief] = Field(
        description="A list of distinct candidate beliefs/assumptions, each scored from 0 to 10.",
    )
    selected_beliefs: list[str] = Field(
        description="The best beliefs chosen to act on (each must exactly match the 'belief' text of one candidate). Never select more than the allowed maximum; selecting fewer is encouraged when the others are redundant or low value.",
    )

class SufficiencyDecision(BaseModel):
    """Decision on whether the information gathered so far is sufficient to stop research."""

    reasoning: str = Field(
        description="Reasoning about what has been learned so far and whether it comprehensively answers the research brief.",
    )
    is_sufficient: bool = Field(
        description="True if the information gathered is sufficient to comprehensively answer the research brief and research should STOP. False if important information is still missing and research should CONTINUE.",
    )
    missing_information: str = Field(
        description="If not sufficient, a clear description of what important information is still missing and should be researched next. Leave empty if sufficient.",
    )

class EpistemicRelation(BaseModel):
    """A single relationship between a new finding and an existing belief/claim."""

    subject_topic: str = Field(
        description="The research topic whose new findings drive this relationship (must exactly match one of the provided research topics).",
    )
    relation: str = Field(
        description="The relationship the new findings have to the existing item. Must be exactly one of: 'supports', 'contradicts', 'revises', 'invalidates'.",
    )
    object_text: str = Field(
        description="The exact text of the existing claim or hypothesis that is supported, contradicted, revised, or invalidated (must exactly match one of the provided existing items).",
    )
    rationale: str = Field(
        description="A concise justification for asserting this relationship.",
    )

class EpistemicReconciliation(BaseModel):
    """How the latest findings relate to previously formed claims and hypotheses."""

    relations: list[EpistemicRelation] = Field(
        default_factory=list,
        description="Relationships where the new findings support, contradict, revise, or invalidate an existing claim or hypothesis. Only include relationships you are confident about; return an empty list if the new findings neither reinforce nor conflict with any existing item.",
    )

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    findings_summary: str = ""

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
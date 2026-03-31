"""
Pydantic models and state schemas for the OmniResearch engine.

This module contains all data structures used across the system:
- ResearchReport: Structured output for individual research agents
- SynthesisReport: Meta-analysis output comparing multiple agents
- ResearchDependencies: Dependency injection container for PydanticAI
- ResearchState: LangGraph state schema for workflow orchestration
"""

from typing import TypedDict, List, Dict, Optional, Annotated
import operator
from dataclasses import dataclass

from pydantic import BaseModel


class ResearchReport(BaseModel):
    """
    Structured output model for individual research agent reports.

    Enforces type safety and ensures all research agents return consistent,
    structured data matching the RESEARCHER_PROMPT_TEMPLATE requirements.
    Used as the output_type for all research agents.

    Fields:
        agent_name: Identifier for which AI agent generated this report
        query: Original research question being investigated
        executive_summary: High-level overview of findings
        key_findings: Main insights discovered during research
        current_data_statistics: Quantitative data and metrics found
        recent_developments: Latest news and updates on the topic
        source_analysis: Evaluation of source credibility and reliability
        conclusion: Final assessment and implications
        evidence_sources: URLs and references used for verification
    """
    agent_name: str
    query: str
    executive_summary: str
    key_findings: List[str]
    current_data_statistics: List[str]
    recent_developments: List[str]
    source_analysis: str
    conclusion: str
    evidence_sources: List[str]


class SynthesisReport(BaseModel):
    """
    Meta-analysis output model for comparing multiple research agent findings.

    Structures the synthesis agent's comparative analysis of the research
    reports. Identifies consensus, conflicts, and unique insights across
    different AI models' research approaches.

    Fields:
        query: Original research question
        agents_agreed: Points where all agents reached consensus
        agents_disagreed: Areas of conflict between agent findings
        disagreement_reasons: Analysis of why conflicts occurred
        combined_conclusions: Synthesized insights from all agents
        unique_insights_by_agent: Agent-specific discoveries not found by others
        actionable_insights: Practical next steps derived from research
    """
    query: str
    agents_agreed: List[str]
    agents_disagreed: List[str]
    disagreement_reasons: List[str]
    combined_conclusions: List[str]
    unique_insights_by_agent: Dict[str, List[str]]
    actionable_insights: List[str]


@dataclass
class ResearchDependencies:
    """
    Dependency injection container for PydanticAI research agents.

    Provides runtime context and shared state for agent operations.
    PydanticAI uses dependency injection to pass this data to system prompts
    and tool functions.

    Attributes:
        query: The research topic being investigated
        agent_name: Human-readable identifier for the agent instance
        search_count: Tracks API calls to enforce rate limits (mutable state)
    """
    query: str
    agent_name: str
    search_count: int = 0


class ResearchState(TypedDict):
    """
    LangGraph state schema for multi-agent research workflow coordination.

    Defines the shared state flowing through the LangGraph workflow.
    Uses Annotated types with reducers to handle parallel execution.

    State Evolution:
        1. Initial: {query, research_reports=[], aggregated_data=None, ...}
        2. After each agent: research_reports grows, agents_completed increments
        3. After synthesis: final_synthesis populated, aggregated_data added

    Reducer Functions:
        - operator.add for research_reports: Merges lists from parallel agents
        - operator.add for agents_completed: Counts completed agents
    """
    query: str
    research_reports: Annotated[List[Dict], operator.add]
    aggregated_data: Optional[Dict]
    final_synthesis: Optional[str]
    agents_completed: Annotated[int, operator.add]
    stream_mode: bool

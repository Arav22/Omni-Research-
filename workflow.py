"""
LangGraph workflow construction and node functions for the OmniResearch engine.

This module handles:
- Bridge functions between LangGraph state and PydanticAI agents
- Terminal output formatting for research reports
- Workflow graph construction with parallel execution

Production features:
    - Protected synthesis node with graceful degradation
    - Per-node execution timing with structured logging
    - Full traceback preservation in error logs
    - Agent-level error isolation — failures don't cascade
"""

import time
from typing import Dict

from colorama import Fore, Style
from langgraph.graph import StateGraph, END

from models import ResearchReport, ResearchDependencies, ResearchState
from agents import get_research_agents, get_synthesis_agent, AGENT_CONFIGS
from logging_config import get_logger

logger = get_logger("workflow")


# ============================================================================
# REPORT OUTPUT FORMATTING
# ============================================================================

def print_full_report(report: ResearchReport, agent_name: str) -> None:
    """
    Print a detailed research report to the terminal with colored formatting.

    Args:
        report: The ResearchReport to display
        agent_name: Display name of the agent that produced the report
    """
    print(Fore.YELLOW + Style.BRIGHT + f"\n  📋 FULL RESEARCH REPORT - {agent_name}")
    print(Fore.CYAN + "  " + "=" * 60)

    if report.executive_summary:
        print(Fore.WHITE + Style.BRIGHT + "  📊 EXECUTIVE SUMMARY:")
        print(Fore.LIGHTBLUE_EX + f"  {report.executive_summary}")
        print()

    if report.key_findings:
        print(Fore.WHITE + Style.BRIGHT + "  🔍 KEY FINDINGS:")
        for i, finding in enumerate(report.key_findings, 1):
            print(Fore.WHITE + f"  {i}. " + Fore.LIGHTBLUE_EX + f"{finding}")
        print()

    if report.current_data_statistics:
        print(Fore.WHITE + Style.BRIGHT + "  📈 CURRENT DATA & STATISTICS:")
        for i, stat in enumerate(report.current_data_statistics, 1):
            print(Fore.WHITE + f"  {i}. " + Fore.CYAN + f"{stat}")
        print()

    if report.recent_developments:
        print(Fore.WHITE + Style.BRIGHT + "  🚀 RECENT DEVELOPMENTS:")
        for i, dev in enumerate(report.recent_developments, 1):
            print(Fore.WHITE + f"  {i}. " + Fore.YELLOW + f"{dev}")
        print()

    if report.source_analysis:
        print(Fore.WHITE + Style.BRIGHT + "  🔬 SOURCE ANALYSIS:")
        print(Fore.LIGHTMAGENTA_EX + f"  {report.source_analysis}")
        print()

    if report.conclusion:
        print(Fore.WHITE + Style.BRIGHT + "  🎯 CONCLUSION:")
        print(Fore.LIGHTGREEN_EX + f"  {report.conclusion}")
        print()

    if report.evidence_sources:
        print(Fore.WHITE + Style.BRIGHT + "  🔗 EVIDENCE SOURCES:")
        for i, source in enumerate(report.evidence_sources, 1):
            print(Fore.WHITE + f"  {i}. " + Fore.LIGHTGREEN_EX + f"{source}")

    print(Fore.CYAN + "  " + "=" * 60)
    print()


# ============================================================================
# LANGGRAPH NODE FUNCTIONS
# ============================================================================

async def execute_research_agent(state: ResearchState, agent_key: str) -> ResearchState:
    """
    Core LangGraph node function bridging LangGraph with PydanticAI agents.

    This function serves as the integration layer between LangGraph's workflow
    orchestration and PydanticAI's agent execution model.

    Workflow:
        1. Extracts research query from LangGraph state
        2. Creates ResearchDependencies for PydanticAI agent
        3. Executes PydanticAI agent with search tools
        4. Handles success/failure scenarios gracefully
        5. Returns state update for LangGraph reducers

    Error Handling:
        - Agent failures don't stop the workflow
        - Creates dummy reports for failed agents
        - Provides meaningful error information
        - Full tracebacks logged for debugging

    Args:
        state: Current LangGraph ResearchState
        agent_key: Agent identifier ("claude", "openai", "zai")

    Returns:
        Partial state update with new research report and incremented agent count.
    """
    config = AGENT_CONFIGS[agent_key]
    agent_dependencies = ResearchDependencies(
        query=state["query"],
        agent_name=config["display_name"]
    )

    pydantic_agent = get_research_agents()[agent_key]
    instruction_message = "Begin your comprehensive research analysis using the provided tools."

    start_time = time.monotonic()

    try:
        logger.info(
            "Starting research agent: %s (model: %s)",
            agent_dependencies.agent_name, config["model"],
            extra={"agent_name": agent_dependencies.agent_name, "query": state["query"]},
        )
        print(Fore.CYAN + "[STARTING] " + Fore.MAGENTA + f"{agent_dependencies.agent_name}")

        execution_result = await pydantic_agent.run(instruction_message, deps=agent_dependencies)
        research_report = execution_result.output

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Agent completed: %s — %d findings, %d sources (%.0fms)",
            agent_dependencies.agent_name,
            len(research_report.key_findings),
            len(research_report.evidence_sources),
            elapsed_ms,
            extra={
                "agent_name": agent_dependencies.agent_name,
                "duration_ms": round(elapsed_ms),
                "result_count": len(research_report.key_findings),
            },
        )

        print(Fore.GREEN + "[COMPLETED] " + Fore.MAGENTA + f"{agent_dependencies.agent_name}")
        print(Fore.WHITE + "  → Found " + Fore.CYAN + f"{len(research_report.key_findings)}" + Fore.WHITE + " key findings")
        print(Fore.WHITE + "  → Used " + Fore.CYAN + f"{len(research_report.evidence_sources)}" + Fore.WHITE + " sources")

        print_full_report(research_report, agent_dependencies.agent_name)

        if not research_report or not hasattr(research_report, 'agent_name'):
            raise ValueError(f"Agent {agent_key} returned invalid report structure")

        return {
            "research_reports": [research_report.model_dump()],
            "agents_completed": 1
        }

    except Exception as error:
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Log with full traceback — this is the key production improvement
        logger.error(
            "Agent failed: %s — %s: %s (%.0fms)",
            agent_dependencies.agent_name,
            type(error).__name__,
            str(error),
            elapsed_ms,
            extra={
                "agent_name": agent_dependencies.agent_name,
                "duration_ms": round(elapsed_ms),
            },
            exc_info=True,  # Preserves full traceback chain
        )

        print(Fore.RED + "[FAILED] " + Fore.MAGENTA + f"{agent_dependencies.agent_name}" + Fore.RED + f": {str(error)}")
        print(Fore.YELLOW + "  → Workflow will continue with other agents")
        print()

        dummy_report = ResearchReport(
            agent_name=agent_dependencies.agent_name,
            query=agent_dependencies.query,
            executive_summary=f"Agent failed during execution: {str(error)}",
            key_findings=[f"Agent failed: {str(error)}"],
            current_data_statistics=["N/A - Agent failed"],
            recent_developments=["N/A - Agent failed"],
            source_analysis="N/A - Agent failed during execution",
            conclusion="Unable to complete research due to agent failure",
            evidence_sources=["N/A - Agent failed"]
        )

        return {
            "research_reports": [dummy_report.model_dump()],
            "agents_completed": 1
        }


# Thin LangGraph node wrappers — each delegates to the shared execution function
async def claude_research_node(state: ResearchState) -> ResearchState:
    """LangGraph node for Claude research agent."""
    return await execute_research_agent(state, "claude")


async def openai_research_node(state: ResearchState) -> ResearchState:
    """LangGraph node for OpenAI research agent."""
    return await execute_research_agent(state, "openai")


async def zai_research_node(state: ResearchState) -> ResearchState:
    """LangGraph node for Z-AI research agent."""
    return await execute_research_agent(state, "zai")


async def synthesis_node(state: ResearchState) -> ResearchState:
    """
    Synthesis node — analyzes all research reports into a meta-analysis.

    The SYNTHESIS_PROMPT_TEMPLATE already contains detailed instructions;
    the LLM handles all comparative analysis based on that prompt.

    Error Handling:
        - Catches all exceptions to prevent workflow crash
        - Returns None synthesis on failure (CLI handles fallback display)
        - Full traceback preserved in logs for debugging
    """
    start_time = time.monotonic()

    # Count how many reports are real vs dummy (from failed agents)
    reports = [ResearchReport(**report) for report in state["research_reports"]]
    valid_reports = [r for r in reports if not r.executive_summary.startswith("Agent failed")]
    failed_count = len(reports) - len(valid_reports)

    logger.info(
        "Synthesis starting — %d total reports (%d valid, %d failed agents)",
        len(reports), len(valid_reports), failed_count,
    )

    if not valid_reports:
        logger.warning("All research agents failed — skipping synthesis")
        return {
            "final_synthesis": None,
            "aggregated_data": {
                "total_reports": len(reports),
                "valid_reports": 0,
                "agents_used": [report.agent_name for report in reports],
                "synthesis_agent": "GPT-4.1-mini",
                "synthesis_status": "skipped_all_agents_failed",
            }
        }

    try:
        synthesis_message = (
            f"Analyze these {len(reports)} research reports from different AI agents. "
            "Follow your system prompt to create a comprehensive meta-analysis."
        )

        result = await get_synthesis_agent().run(synthesis_message, deps=reports)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Synthesis completed in %.0fms — analyzed %d reports",
            elapsed_ms, len(reports),
            extra={"duration_ms": round(elapsed_ms), "result_count": len(reports)},
        )

        return {
            "final_synthesis": result.output.model_dump_json(indent=2),
            "aggregated_data": {
                "total_reports": len(reports),
                "valid_reports": len(valid_reports),
                "agents_used": [report.agent_name for report in reports],
                "synthesis_agent": "GPT-4.1-mini",
                "synthesis_status": "completed",
            }
        }

    except Exception as error:
        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.error(
            "Synthesis failed — %s: %s (%.0fms)",
            type(error).__name__,
            str(error),
            elapsed_ms,
            extra={"duration_ms": round(elapsed_ms)},
            exc_info=True,  # Full traceback preserved
        )

        return {
            "final_synthesis": None,
            "aggregated_data": {
                "total_reports": len(reports),
                "valid_reports": len(valid_reports),
                "agents_used": [report.agent_name for report in reports],
                "synthesis_agent": "GPT-4.1-mini",
                "synthesis_status": f"failed: {type(error).__name__}: {str(error)}",
            }
        }


# ============================================================================
# WORKFLOW GRAPH CONSTRUCTION
# ============================================================================

def create_research_workflow():
    """
    Constructs the LangGraph workflow for parallel multi-agent research.

    Architecture (fan-out / fan-in):

        ┌─────────────┐  ┌──────────────┐  ┌─────────────┐
        │   Claude     │  │   OpenAI     │  │    Z-AI     │
        └──────┬───────┘  └──────┬───────┘  └──────┬──────┘
               └─────────────────┼─────────────────┘
                                 │
                        ┌────────┴────────┐
                        │   Synthesis     │
                        └────────┬────────┘
                                 │
                              ┌──┴──┐
                              │ END │
                              └─────┘

    Returns:
        Compiled LangGraph application ready for execution.
    """
    logger.info("Building research workflow graph")

    workflow = StateGraph(ResearchState)

    # Register nodes
    workflow.add_node("claude_research", claude_research_node)
    workflow.add_node("openai_research", openai_research_node)
    workflow.add_node("zai_research", zai_research_node)
    workflow.add_node("synthesis", synthesis_node)

    # Parallel execution via multiple entry points
    workflow.set_entry_point("claude_research")
    workflow.set_entry_point("openai_research")
    workflow.set_entry_point("zai_research")

    # All research → synthesis → end
    workflow.add_edge("claude_research", "synthesis")
    workflow.add_edge("openai_research", "synthesis")
    workflow.add_edge("zai_research", "synthesis")
    workflow.add_edge("synthesis", END)

    compiled = workflow.compile()
    logger.info("Research workflow graph compiled successfully")

    return compiled

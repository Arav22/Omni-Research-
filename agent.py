"""
OmniResearch — CLI Entry Point

Production-grade entry point that orchestrates the multi-agent research
workflow and handles user interaction (CLI arguments, streaming output,
result display).

Production features:
    - Logging initialized at startup with configurable level
    - Per-run correlation IDs for end-to-end tracing
    - Proper traceback preservation on unhandled errors
    - Structured error reporting

Usage:
    python agent.py                                    # Interactive mode
    python agent.py --query "your topic"               # CLI mode
    python agent.py --query "your topic" --no-stream   # Batch mode
    LOG_LEVEL=DEBUG python agent.py                    # Debug logging
"""

import os
import json
import asyncio
import argparse
from typing import Dict

from colorama import Fore, Style, init
init(autoreset=True)

from dotenv import load_dotenv
load_dotenv()

from logging_config import setup_logging, get_logger, set_correlation_id
from workflow import create_research_workflow

# Initialize logging BEFORE anything else
setup_logging()
logger = get_logger("cli")


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

async def run_research_workflow(query: str, stream: bool = True) -> Dict:
    """
    Execute the complete research workflow with professional streaming output.

    Args:
        query: Research topic to investigate
        stream: If True, show real-time progress; if False, batch mode

    Returns:
        Final LangGraph state dict with research_reports and final_synthesis.

    Raises:
        ValueError: If required environment variables are missing.
        RuntimeError: If no result is received from the workflow stream.
    """
    # Generate a unique correlation ID for this research run
    cid = set_correlation_id()
    logger.info(
        "Starting research workflow",
        extra={"query": query, "status": "starting"},
    )

    required_env_vars = ["TAVILY_API_KEY", "OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(
            "Missing required environment variables: %s",
            missing_vars,
            extra={"status": "env_error"},
        )
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    app = create_research_workflow()

    initial_state = {
        "query": query,
        "research_reports": [],
        "aggregated_data": None,
        "final_synthesis": None,
        "agents_completed": 0,
        "stream_mode": stream
    }

    # Workflow header
    print("\n" + Fore.BLUE + Style.BRIGHT + "=" * 80)
    print(Fore.BLUE + Style.BRIGHT + "🔬 OMNI RESEARCH ENGINE")
    print(Fore.BLUE + Style.BRIGHT + "=" * 80)
    print(Fore.WHITE + Style.BRIGHT + f"Research Query: " + Fore.YELLOW + f"{query}")
    print(Fore.WHITE + Style.BRIGHT + f"Streaming Mode: " + Fore.CYAN + f"{'Enabled' if stream else 'Disabled'}")
    print(Fore.WHITE + Style.BRIGHT + f"Correlation ID:  " + Fore.CYAN + f"{cid}")
    print(Fore.BLUE + Style.BRIGHT + "=" * 80)

    if stream:
        logger.info("Running in streaming mode")
        print(Fore.YELLOW + Style.BRIGHT + "\n[WORKFLOW] Starting parallel research agents...")

        completed_agents = set()
        synthesis_started = False
        result = None

        async for chunk in app.astream(initial_state, stream_mode="values"):
            result = chunk

            current_reports = len(chunk.get('research_reports', []))
            has_synthesis = chunk.get('final_synthesis') is not None

            # Show research agent completions
            if current_reports > len(completed_agents):
                agent_names = ["CLAUDE", "OPENAI", "Z-AI"]
                if len(completed_agents) < len(agent_names):
                    agent_type = agent_names[len(completed_agents)]
                    print(Fore.YELLOW + Style.BRIGHT + f"\n[STEP 1/2] RESEARCH AGENT: " + Fore.MAGENTA + Style.BRIGHT + f"{agent_type}")
                    print(Fore.CYAN + "-" * 50)
                    print(Fore.WHITE + "Status: " + Fore.GREEN + Style.BRIGHT + "COMPLETED")
                    print(Fore.WHITE + "Agent: " + Fore.MAGENTA + f"{agent_type} Research Agent")
                    print(Fore.WHITE + "Task: " + Fore.LIGHTBLUE_EX + "Web research and analysis")
                    completed_agents.add(agent_type)
                    print(Fore.CYAN + "-" * 50)

            # Show synthesis processing
            if current_reports >= 3 and not synthesis_started:
                print(Fore.YELLOW + Style.BRIGHT + "\n[STEP 2/2] SYNTHESIS ANALYSIS")
                print(Fore.CYAN + "-" * 50)
                print(Fore.WHITE + "Status: " + Fore.YELLOW + Style.BRIGHT + "PROCESSING")
                print(Fore.WHITE + "Agent: " + Fore.MAGENTA + "GPT-4.1-mini Synthesis Agent")
                print(Fore.WHITE + "Task: " + Fore.LIGHTBLUE_EX + "Cross-agent analysis and meta-synthesis")
                print(Fore.CYAN + "-" * 50)
                synthesis_started = True

            # Show synthesis completion
            if has_synthesis and synthesis_started:
                print(Fore.YELLOW + Style.BRIGHT + "\n[STEP 2/2] SYNTHESIS ANALYSIS")
                print(Fore.CYAN + "-" * 50)
                print(Fore.WHITE + "Status: " + Fore.GREEN + Style.BRIGHT + "COMPLETED")
                print(Fore.WHITE + "Reports Analyzed: " + Fore.CYAN + f"{current_reports}")
                print(Fore.CYAN + "-" * 50)
                break

        if not result:
            logger.error("No result received from workflow stream")
            raise RuntimeError("No result received from workflow stream")

    else:
        logger.info("Running in batch mode")
        print(Fore.YELLOW + Style.BRIGHT + "\n[WORKFLOW] Executing in batch mode...")
        result = await app.ainvoke(initial_state)

    # Completion summary
    report_count = len(result['research_reports'])
    has_synthesis = result['final_synthesis'] is not None

    logger.info(
        "Workflow completed — %d reports, synthesis=%s",
        report_count,
        "yes" if has_synthesis else "no",
        extra={
            "query": query,
            "result_count": report_count,
            "status": "completed",
        },
    )

    print(Fore.GREEN + Style.BRIGHT + "\n" + "=" * 80)
    print(Fore.GREEN + Style.BRIGHT + "✅ OMNI RESEARCH — WORKFLOW COMPLETED")
    print(Fore.GREEN + Style.BRIGHT + "=" * 80)
    print(Fore.WHITE + "Total Research Reports: " + Fore.CYAN + Style.BRIGHT + f"{report_count}")
    synthesis_status = Fore.GREEN + "Generated" if has_synthesis else Fore.RED + "Failed"
    print(Fore.WHITE + "Synthesis Report: " + synthesis_status)
    print(Fore.WHITE + "Correlation ID:   " + Fore.CYAN + f"{cid}")
    print(Fore.GREEN + Style.BRIGHT + "=" * 80)

    return result


async def run_research_with_query(query: str) -> Dict:
    """Utility function to run research with a specific query (for testing/automation)."""
    return await run_research_workflow(query, stream=True)


# ============================================================================
# RESULT DISPLAY
# ============================================================================

def display_synthesis(synthesis_data: dict) -> None:
    """Pretty-print the synthesis report sections."""
    print(Fore.CYAN + Style.BRIGHT + f"\n📊 WHERE ALL AGENTS AGREED ({len(synthesis_data.get('agents_agreed', []))}):")
    for i, finding in enumerate(synthesis_data.get('agents_agreed', []), 1):
        print(Fore.WHITE + f"  {i}. " + Fore.LIGHTBLUE_EX + f"{finding}")

    print(Fore.YELLOW + Style.BRIGHT + f"\n⚡ WHERE AGENTS DISAGREED ({len(synthesis_data.get('agents_disagreed', []))}):")
    for i, conflict in enumerate(synthesis_data.get('agents_disagreed', []), 1):
        print(Fore.WHITE + f"  {i}. " + Fore.LIGHTYELLOW_EX + f"{conflict}")

    print(Fore.RED + Style.BRIGHT + f"\n🔍 WHY DISAGREEMENTS HAPPENED ({len(synthesis_data.get('disagreement_reasons', []))}):")
    for i, reason in enumerate(synthesis_data.get('disagreement_reasons', []), 1):
        print(Fore.WHITE + f"  {i}. " + Fore.LIGHTRED_EX + f"{reason}")

    print(Fore.BLUE + Style.BRIGHT + f"\n🧠 COMBINED CONCLUSIONS ({len(synthesis_data.get('combined_conclusions', []))}):")
    for i, conclusion in enumerate(synthesis_data.get('combined_conclusions', []), 1):
        print(Fore.WHITE + f"  {i}. " + Fore.LIGHTBLUE_EX + f"{conclusion}")

    print(Fore.MAGENTA + Style.BRIGHT + "\n💡 UNIQUE INSIGHTS BY AGENT:")
    for agent, insights in synthesis_data.get('unique_insights_by_agent', {}).items():
        print(Fore.WHITE + f"  📋 {agent}:")
        for i, insight in enumerate(insights, 1):
            print(Fore.WHITE + f"    {i}. " + Fore.LIGHTMAGENTA_EX + f"{insight}")
        print()

    print(Fore.GREEN + Style.BRIGHT + f"\n🎯 ACTIONABLE INSIGHTS ({len(synthesis_data.get('actionable_insights', []))}):")
    for i, insight in enumerate(synthesis_data.get('actionable_insights', []), 1):
        print(Fore.WHITE + f"  {i}. " + Fore.LIGHTGREEN_EX + f"{insight}")


def display_fallback_reports(reports: list) -> None:
    """Display individual reports when synthesis fails."""
    print(Fore.RED + Style.BRIGHT + "\n❌ SYNTHESIS FAILED - SHOWING INDIVIDUAL REPORTS:")
    print(Fore.RED + Style.BRIGHT + "-" * 80)
    for i, report in enumerate(reports, 1):
        print(Fore.CYAN + Style.BRIGHT + f"\n📋 REPORT {i} - " + Fore.MAGENTA + f"{report.get('agent_name', 'Unknown')}:")

        if report.get('executive_summary'):
            print(Fore.WHITE + "📊 Summary: " + Fore.LIGHTBLUE_EX + f"{report.get('executive_summary')}")

        for j, finding in enumerate(report.get('key_findings', []), 1):
            print(Fore.WHITE + f"  {j}. " + Fore.LIGHTBLUE_EX + f"{finding}")

        if report.get('conclusion'):
            print(Fore.WHITE + "🎯 Conclusion: " + Fore.LIGHTGREEN_EX + f"{report.get('conclusion')}")

        print(Fore.WHITE + f"\n🔗 Sources: " + Fore.CYAN + f"{len(report.get('evidence_sources', []))} sources")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    async def main():
        """Run the research workflow with user-provided query."""

        parser = argparse.ArgumentParser(description="OmniResearch — Multi-Agent Research Engine")
        parser.add_argument('--query', '-q', type=str, help='Research query (if not provided, will prompt interactively)')
        parser.add_argument('--no-stream', action='store_true', help='Disable streaming output')
        args = parser.parse_args()

        print(Fore.BLUE + Style.BRIGHT + "🔬 OMNI RESEARCH ENGINE")
        print(Fore.BLUE + Style.BRIGHT + "=" * 60)

        # Get query from command line or interactive input
        if args.query:
            query = args.query.strip()
            print(Fore.WHITE + f"Query provided via command line: " + Fore.YELLOW + f'"{query}"')
        else:
            print(Fore.WHITE + "Enter your research topic below:")
            print(Fore.CYAN + "Examples:")
            print(Fore.LIGHTBLUE_EX + "  • Latest developments in artificial intelligence")
            print(Fore.LIGHTBLUE_EX + "  • Climate change impact on agriculture")
            print(Fore.LIGHTBLUE_EX + "  • Cryptocurrency market trends 2025")
            print(Fore.LIGHTBLUE_EX + "  • Space exploration recent achievements")
            print()
            print(Fore.CYAN + "💡 Tip: You can also use: " + Fore.WHITE + 'python3 agent.py --query "your topic here"')
            print()

            query = input(Fore.YELLOW + Style.BRIGHT + "🔍 Research Topic: " + Fore.WHITE + Style.NORMAL).strip()

        # Validate input
        if not query:
            logger.warning("Empty query provided")
            print(Fore.RED + "❌ No query provided. Exiting...")
            return

        if len(query) < 10:
            logger.warning("Short query: '%s' (%d chars)", query, len(query))
            print(Fore.YELLOW + "⚠️  Query seems quite short. Consider adding more detail for better results.")
            if not args.query:
                confirm = input(Fore.WHITE + "Continue anyway? (y/N): ").strip().lower()
                if confirm != 'y':
                    print(Fore.CYAN + "👋 Goodbye!")
                    return

        logger.info("User query accepted: '%s'", query, extra={"query": query})
        print(Fore.GREEN + f"\n✅ Starting research on: " + Fore.YELLOW + f'"{query}"')
        print()

        stream_mode = not args.no_stream
        result = await run_research_workflow(query, stream=stream_mode)

        # Final results
        print(Fore.GREEN + Style.BRIGHT + "\nFINAL RESULTS SUMMARY")
        print(Fore.GREEN + Style.BRIGHT + "=" * 60)
        print(Fore.WHITE + f"Query: " + Fore.YELLOW + f"{result['query']}")
        print(Fore.WHITE + f"Reports Generated: " + Fore.CYAN + f"{len(result['research_reports'])}")
        synthesis_color = Fore.GREEN if result['final_synthesis'] else Fore.RED
        print(Fore.WHITE + f"Synthesis Available: " + synthesis_color + f"{'Yes' if result['final_synthesis'] else 'No'}")

        if result['final_synthesis']:
            print(Fore.GREEN + Style.BRIGHT + "\nSYNTHESIS REPORT:")
            print(Fore.GREEN + Style.BRIGHT + "-" * 80)
            try:
                synthesis_data = json.loads(result['final_synthesis'])
                display_synthesis(synthesis_data)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Failed to parse synthesis JSON: %s",
                    str(exc),
                    exc_info=True,
                )
                print(result['final_synthesis'])
        else:
            display_fallback_reports(result['research_reports'])

        print(Fore.BLUE + Style.BRIGHT + "-" * 80)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Research interrupted by user (KeyboardInterrupt)")
        print(Fore.YELLOW + "\n\n⚠️  Research interrupted by user")
        print(Fore.CYAN + "👋 Goodbye!")
    except Exception as e:
        logger.critical(
            "Unhandled exception: %s: %s",
            type(e).__name__,
            str(e),
            exc_info=True,  # Full traceback preserved in logs
        )
        print(Fore.RED + f"\n❌ An error occurred: {type(e).__name__}: {str(e)}")
        print(Fore.YELLOW + "Please check your environment variables and try again.")
        print(Fore.CYAN + "📄 Full traceback available in logs/research.error.log")
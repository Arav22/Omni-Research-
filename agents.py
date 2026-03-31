"""
PydanticAI agent configuration using a factory pattern.

Eliminates code duplication by creating all research agents through a single
factory function. Each agent gets identical system prompts and search tools,
differing only in the underlying LLM model.

Agent creation is lazy — agents are only instantiated on first access,
allowing test imports without requiring API keys.

Production features:
    - Structured logging for agent lifecycle events
    - Logged agent creation with model and configuration details
    - Provider initialization tracking
"""

import os
from typing import List, Optional

from pydantic_ai import Agent, RunContext, PromptedOutput
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from models import ResearchReport, SynthesisReport, ResearchDependencies
from prompts import RESEARCHER_PROMPT_TEMPLATE, SYNTHESIS_PROMPT_TEMPLATE
from search import search_web
from logging_config import get_logger

logger = get_logger("agents")


# Maximum searches per agent during testing
# TODO: PRODUCTION - Increase to 5-10
MAX_SEARCHES_PER_AGENT = 2

# Agent registry: key -> (model_name, display_name)
AGENT_CONFIGS = {
    "claude": {
        "model": "anthropic/claude-3.5-haiku",
        "display_name": "Claude Research Agent",
    },
    "openai": {
        "model": "openai/gpt-4.1-mini",
        "display_name": "OpenAI Research Agent",
    },
    "zai": {
        "model": "z-ai/glm-4.5",
        "display_name": "Z-AI Research Agent",
    },
}


# ============================================================================
# LAZY INITIALIZATION — avoids requiring API keys at import time
# ============================================================================

_openrouter_provider: Optional[OpenRouterProvider] = None
_research_agents: Optional[dict] = None
_synthesis_agent: Optional[Agent] = None


def _get_provider() -> OpenRouterProvider:
    """Get or create the shared OpenRouter provider."""
    global _openrouter_provider
    if _openrouter_provider is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY environment variable is not set")
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        _openrouter_provider = OpenRouterProvider(api_key=api_key)
        logger.info("OpenRouter provider initialized")
    return _openrouter_provider


def create_research_agent(model_name: str) -> Agent:
    """
    Factory function to create a PydanticAI research agent.

    Creates an agent with:
    - Dynamic system prompt injected with query and agent name
    - Rate-limited web search tool via Tavily
    - Structured ResearchReport output
    - Retry and instrumentation support

    Args:
        model_name: OpenRouter model identifier (e.g. "anthropic/claude-3.5-haiku")

    Returns:
        Configured PydanticAI Agent ready for research execution.
    """
    provider = _get_provider()

    logger.info(
        "Creating research agent with model: %s",
        model_name,
        extra={"agent_name": model_name},
    )

    agent = Agent(
        OpenAIModel(model_name, provider=provider),
        deps_type=ResearchDependencies,
        output_type=PromptedOutput(
            ResearchReport,
            name='research_report',
            description='Generate a comprehensive research report with structured fields'
        ),
        retries=3,
        instrument=True
    )

    @agent.system_prompt
    def system_prompt(ctx: RunContext[ResearchDependencies]) -> str:
        """Dynamic system prompt with injected query and agent identity."""
        return RESEARCHER_PROMPT_TEMPLATE.format(
            query=ctx.deps.query,
            agent_name=ctx.deps.agent_name
        )

    @agent.tool
    async def search_tavily(ctx: RunContext[ResearchDependencies], query: str) -> str:
        """Search for information using Tavily API."""
        if ctx.deps.search_count >= MAX_SEARCHES_PER_AGENT:
            logger.debug(
                "Search limit reached for %s (%d max)",
                ctx.deps.agent_name, MAX_SEARCHES_PER_AGENT,
                extra={"agent_name": ctx.deps.agent_name},
            )
            return f"Search limit reached ({MAX_SEARCHES_PER_AGENT} max for testing). Use existing findings."
        ctx.deps.search_count += 1
        logger.debug(
            "Agent %s search %d/%d: %s",
            ctx.deps.agent_name, ctx.deps.search_count,
            MAX_SEARCHES_PER_AGENT, query,
            extra={"agent_name": ctx.deps.agent_name, "search_query": query},
        )
        return await search_web(query)

    logger.info(
        "Research agent created: %s (retries=3, instrument=True)",
        model_name,
        extra={"agent_name": model_name},
    )
    return agent


def get_research_agents() -> dict:
    """
    Get the research agents dict, creating them lazily on first call.

    Returns:
        Dict mapping agent keys ("claude", "openai", "zai") to Agent instances.
    """
    global _research_agents
    if _research_agents is None:
        logger.info("Initializing %d research agents", len(AGENT_CONFIGS))
        _research_agents = {
            key: create_research_agent(config["model"])
            for key, config in AGENT_CONFIGS.items()
        }
        logger.info(
            "All research agents initialized: %s",
            list(_research_agents.keys()),
        )
    return _research_agents


def get_synthesis_agent() -> Agent:
    """
    Get the synthesis agent, creating it lazily on first call.

    Returns:
        Configured PydanticAI Agent for meta-analysis.
    """
    global _synthesis_agent
    if _synthesis_agent is None:
        provider = _get_provider()

        logger.info("Creating synthesis agent with model: openai/gpt-4.1-mini")

        _synthesis_agent = Agent(
            OpenAIModel("openai/gpt-4.1-mini", provider=provider),
            deps_type=List[ResearchReport],
            output_type=PromptedOutput(
                SynthesisReport,
                name='synthesis_report',
                description='Generate a comprehensive meta-analysis report comparing multiple research findings'
            ),
            retries=3,
            instrument=True
        )

        @_synthesis_agent.system_prompt
        def synthesis_system_prompt(ctx: RunContext[List[ResearchReport]]) -> str:
            """Dynamic prompt that extracts the query from input reports."""
            query = ctx.deps[0].query if ctx.deps else "research query"
            return SYNTHESIS_PROMPT_TEMPLATE.format(query=query)

        logger.info("Synthesis agent created (retries=3, instrument=True)")

    return _synthesis_agent


# Convenience property-like access for backward compatibility
# These are accessed as module-level attributes via __getattr__
def __getattr__(name):
    """Lazy module-level attribute access for research_agents and synthesis_agent."""
    if name == "research_agents":
        return get_research_agents()
    if name == "synthesis_agent":
        return get_synthesis_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

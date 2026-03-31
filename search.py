"""
Web search infrastructure for the OmniResearch engine using Tavily API.

Provides a singleton TavilyClient and centralized search function
shared by all research agents. The singleton pattern avoids recreating
the client on every search call.

Production features:
    - Retry with exponential backoff on transient failures
    - Per-search timing and structured logging
    - Graceful degradation with informative error messages
"""

import os
import time
import asyncio
from typing import Optional

from tavily import TavilyClient

from logging_config import get_logger

logger = get_logger("search")


# Singleton Tavily client — created once, reused across all searches
_tavily_client: Optional[TavilyClient] = None


def get_tavily_client() -> TavilyClient:
    """
    Get or create the singleton TavilyClient instance.

    Returns:
        Shared TavilyClient configured with TAVILY_API_KEY.

    Raises:
        ValueError: If TAVILY_API_KEY environment variable is not set.
    """
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.error("TAVILY_API_KEY environment variable is not set")
            raise ValueError("TAVILY_API_KEY environment variable is not set")
        _tavily_client = TavilyClient(api_key=api_key)
        logger.info("TavilyClient initialized successfully")
    return _tavily_client


def reset_tavily_client() -> None:
    """Reset the singleton client (useful for testing)."""
    global _tavily_client
    _tavily_client = None
    logger.debug("TavilyClient singleton reset")


# ---------------------------------------------------------------------------
# Retry Configuration
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0          # 1s, 2s, 4s exponential
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    OSError,
)


async def search_web(query: str) -> str:
    """
    Centralized web search function using Tavily API for all research agents.

    Includes retry with exponential backoff for transient failures and
    structured logging for every search attempt.

    TESTING CONFIGURATION (Current Settings):
        - Timeout: 10 seconds
        - Results: 1 per search
        - Search depth: "basic"
        - Raw content: Disabled

    PRODUCTION RECOMMENDATIONS:
        - Timeout: 30+ seconds
        - Results: 3-5 per search
        - Search depth: "advanced"
        - Consider enabling raw content

    Args:
        query: Natural language search query from research agent

    Returns:
        Formatted string containing search results with titles, URLs, and content.
        Returns error message if search fails or times out after all retries.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        start_time = time.monotonic()
        try:
            logger.info(
                "Search attempt %d/%d: %s",
                attempt, MAX_RETRIES, query,
                extra={"search_query": query, "retry_attempt": attempt},
            )

            # TODO: PRODUCTION - Increase timeout to 30+ seconds
            async with asyncio.timeout(10):
                client = get_tavily_client()
                response = client.search(
                    query=query,
                    search_depth="basic",      # TODO: PRODUCTION - Use "advanced"
                    max_results=1,             # TODO: PRODUCTION - Increase to 3-5
                    include_answer=True,
                    include_raw_content=False
                )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            result_count = len(response.get('results', []))

            logger.info(
                "Search completed in %.0fms — %d results for: %s",
                elapsed_ms, result_count, query,
                extra={
                    "search_query": query,
                    "duration_ms": round(elapsed_ms),
                    "result_count": result_count,
                    "retry_attempt": attempt,
                },
            )

            results = []
            if response.get('answer'):
                results.append(f"Answer: {response['answer']}")

            for result in response.get('results', []):
                content = result.get('content', 'N/A')
                results.append(
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"Content: {content}\n"
                )

            return "\n\n".join(results)

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            last_exception = asyncio.TimeoutError(
                f"Search timed out after {elapsed_ms:.0f}ms"
            )
            logger.warning(
                "Search timeout (attempt %d/%d, %.0fms): %s",
                attempt, MAX_RETRIES, elapsed_ms, query,
                extra={
                    "search_query": query,
                    "duration_ms": round(elapsed_ms),
                    "retry_attempt": attempt,
                },
            )

        except RETRYABLE_EXCEPTIONS as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            last_exception = exc
            logger.warning(
                "Retryable search error (attempt %d/%d): %s — %s",
                attempt, MAX_RETRIES, type(exc).__name__, str(exc),
                extra={
                    "search_query": query,
                    "duration_ms": round(elapsed_ms),
                    "retry_attempt": attempt,
                },
                exc_info=True,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Non-retryable search error: %s — %s",
                type(exc).__name__, str(exc),
                extra={
                    "search_query": query,
                    "duration_ms": round(elapsed_ms),
                    "retry_attempt": attempt,
                },
                exc_info=True,
            )
            return f"Error during search: {type(exc).__name__}: {str(exc)}"

        # Exponential backoff before retry
        if attempt < MAX_RETRIES:
            backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.info("Retrying in %.1fs...", backoff)
            await asyncio.sleep(backoff)

    # All retries exhausted
    logger.error(
        "Search failed after %d attempts: %s — last error: %s",
        MAX_RETRIES, query, str(last_exception),
        extra={"search_query": query, "retry_attempt": MAX_RETRIES},
    )
    return f"Search failed after {MAX_RETRIES} attempts: {str(last_exception)}"

"""
Tests for the Tavily search module.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from search import search_web, get_tavily_client, reset_tavily_client


# ============================================================================
# Singleton Client Tests
# ============================================================================

class TestTavilyClientSingleton:
    """Tests for the singleton TavilyClient pattern."""

    def setup_method(self):
        """Reset the singleton before each test."""
        reset_tavily_client()

    def teardown_method(self):
        """Clean up after each test."""
        reset_tavily_client()

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key-123"})
    @patch("search.TavilyClient")
    def test_get_client_creates_instance(self, mock_client_class):
        """First call should create a new TavilyClient."""
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        client = get_tavily_client()

        mock_client_class.assert_called_once_with(api_key="test-key-123")
        assert client is mock_instance

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key-123"})
    @patch("search.TavilyClient")
    def test_get_client_returns_same_instance(self, mock_client_class):
        """Subsequent calls should return the same cached instance."""
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        client1 = get_tavily_client()
        client2 = get_tavily_client()

        # Only one TavilyClient should be created
        mock_client_class.assert_called_once()
        assert client1 is client2

    @patch.dict("os.environ", {}, clear=True)
    def test_get_client_raises_without_api_key(self):
        """Should raise ValueError when TAVILY_API_KEY is not set."""
        import os
        os.environ.pop("TAVILY_API_KEY", None)

        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            get_tavily_client()

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("search.TavilyClient")
    def test_reset_client(self, mock_client_class):
        """reset_tavily_client should force a new instance on next call."""
        mock_client_class.return_value = MagicMock()

        client1 = get_tavily_client()
        reset_tavily_client()
        client2 = get_tavily_client()

        # Two instances should have been created
        assert mock_client_class.call_count == 2


# ============================================================================
# search_web Tests
# ============================================================================

class TestSearchWeb:
    """Tests for the search_web async function."""

    def setup_method(self):
        reset_tavily_client()

    def teardown_method(self):
        reset_tavily_client()

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("search.TavilyClient")
    async def test_successful_search_with_answer(self, mock_client_class):
        """Should format response with answer and results."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "AI is advancing rapidly.",
            "results": [
                {
                    "title": "AI News",
                    "url": "https://example.com/ai",
                    "content": "Latest developments in AI..."
                }
            ]
        }
        mock_client_class.return_value = mock_client

        result = await search_web("artificial intelligence trends")

        assert "AI is advancing rapidly" in result
        assert "AI News" in result
        assert "https://example.com/ai" in result
        assert "Latest developments in AI" in result

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("search.TavilyClient")
    async def test_successful_search_without_answer(self, mock_client_class):
        """Should handle responses without an answer field."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com",
                    "content": "Some content"
                }
            ]
        }
        mock_client_class.return_value = mock_client

        result = await search_web("test query")

        assert "Answer:" not in result
        assert "Result 1" in result

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("search.TavilyClient")
    async def test_non_retryable_error_returns_error_message(self, mock_client_class):
        """Non-retryable exceptions should return error message immediately."""
        mock_client = MagicMock()
        mock_client.search.side_effect = RuntimeError("API limit exceeded")
        mock_client_class.return_value = mock_client

        result = await search_web("test query")

        assert "Error during search" in result
        assert "API limit exceeded" in result
        # Non-retryable errors should only call search once
        assert mock_client.search.call_count == 1

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("search.TavilyClient")
    async def test_empty_results(self, mock_client_class):
        """Should handle empty search results gracefully."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_client_class.return_value = mock_client

        result = await search_web("obscure topic with no results")

        # Should return empty string (no answer, no results)
        assert result == ""

    @pytest.mark.asyncio
    @patch("search.MAX_RETRIES", 1)  # Force single retry for faster test
    @patch("search.BASE_BACKOFF_SECONDS", 0.01)
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("search.TavilyClient")
    async def test_timeout_returns_failure_message(self, mock_client_class):
        """Should return a failure message after timeout retries are exhausted."""
        mock_client = MagicMock()
        mock_client.search.side_effect = lambda **kwargs: (_ for _ in ()).throw(asyncio.TimeoutError())
        mock_client_class.return_value = mock_client

        result = await search_web("test")

        assert "failed" in result.lower() or "timed out" in result.lower()

    @pytest.mark.asyncio
    @patch("search.MAX_RETRIES", 3)
    @patch("search.BASE_BACKOFF_SECONDS", 0.01)  # Fast backoff for tests
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("search.TavilyClient")
    async def test_retryable_error_retries_and_succeeds(self, mock_client_class):
        """Should retry on transient errors and succeed if a later attempt works."""
        mock_client = MagicMock()
        # First call fails, second succeeds
        mock_client.search.side_effect = [
            ConnectionError("connection reset"),
            {
                "answer": "Retry worked!",
                "results": [{"title": "Success", "url": "https://example.com", "content": "Content"}]
            }
        ]
        mock_client_class.return_value = mock_client

        result = await search_web("test retry")

        assert "Retry worked!" in result
        assert mock_client.search.call_count == 2

    @pytest.mark.asyncio
    @patch("search.MAX_RETRIES", 2)
    @patch("search.BASE_BACKOFF_SECONDS", 0.01)
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("search.TavilyClient")
    async def test_retryable_error_exhausts_retries(self, mock_client_class):
        """Should return failure message when all retries are exhausted."""
        mock_client = MagicMock()
        mock_client.search.side_effect = ConnectionError("persistent failure")
        mock_client_class.return_value = mock_client

        result = await search_web("test exhausted retries")

        assert "failed after" in result.lower()
        assert mock_client.search.call_count == 2

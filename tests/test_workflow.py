"""
Tests for the LangGraph workflow construction.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from models import ResearchReport, ResearchDependencies, ResearchState


# ============================================================================
# Helper Fixtures
# ============================================================================

def make_test_state(**overrides) -> ResearchState:
    """Create a minimal valid ResearchState for testing."""
    defaults = {
        "query": "test research topic",
        "research_reports": [],
        "aggregated_data": None,
        "final_synthesis": None,
        "agents_completed": 0,
        "stream_mode": False,
    }
    defaults.update(overrides)
    return defaults


def make_test_report(**overrides) -> ResearchReport:
    """Create a valid ResearchReport for testing."""
    defaults = {
        "agent_name": "Test Agent",
        "query": "test query",
        "executive_summary": "Summary of test findings.",
        "key_findings": ["Finding A", "Finding B"],
        "current_data_statistics": ["50% growth"],
        "recent_developments": ["New breakthrough"],
        "source_analysis": "Reliable academic sources.",
        "conclusion": "Test conclusion reached.",
        "evidence_sources": ["https://example.com/source1"],
    }
    defaults.update(overrides)
    return ResearchReport(**defaults)


# ============================================================================
# Workflow Graph Tests
# ============================================================================

class TestWorkflowCreation:
    """Tests for the LangGraph workflow construction."""

    @patch("workflow.get_research_agents")
    @patch("workflow.get_synthesis_agent")
    def test_workflow_compiles(self, mock_synth, mock_agents):
        """create_research_workflow should return a compiled graph without error."""
        from workflow import create_research_workflow

        app = create_research_workflow()
        assert app is not None

    @patch("workflow.get_research_agents")
    @patch("workflow.get_synthesis_agent")
    def test_workflow_has_expected_nodes(self, mock_synth, mock_agents):
        """The compiled workflow should contain all expected node names."""
        from workflow import create_research_workflow

        app = create_research_workflow()
        # graph.nodes is a dict mapping node names -> callables in LangGraph
        node_names = list(app.get_graph().nodes)
        assert "claude_research" in node_names
        assert "openai_research" in node_names
        assert "zai_research" in node_names
        assert "synthesis" in node_names


# ============================================================================
# execute_research_agent Tests
# ============================================================================

class TestExecuteResearchAgent:
    """Tests for the core research agent execution bridge."""

    @pytest.mark.asyncio
    async def test_agent_failure_returns_dummy_report(self):
        """When an agent raises an exception, a dummy report should be returned."""
        from workflow import execute_research_agent

        state = make_test_state()

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = RuntimeError("Model unavailable")

        with patch("workflow.get_research_agents", return_value={"claude": mock_agent}):
            result = await execute_research_agent(state, "claude")

        # Should still return a valid state update
        assert result["agents_completed"] == 1
        assert len(result["research_reports"]) == 1

        report = result["research_reports"][0]
        assert "failed" in report["executive_summary"].lower()
        assert "Model unavailable" in report["executive_summary"]

    @pytest.mark.asyncio
    async def test_successful_agent_returns_report(self):
        """A successful agent run should return the serialized report."""
        from workflow import execute_research_agent

        state = make_test_state()
        test_report = make_test_report(agent_name="Claude Research Agent")

        mock_result = MagicMock()
        mock_result.output = test_report

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch("workflow.get_research_agents", return_value={"claude": mock_agent}):
            result = await execute_research_agent(state, "claude")

        assert result["agents_completed"] == 1
        assert len(result["research_reports"]) == 1
        assert result["research_reports"][0]["agent_name"] == "Claude Research Agent"
        assert result["research_reports"][0]["key_findings"] == ["Finding A", "Finding B"]


# ============================================================================
# print_full_report Tests
# ============================================================================

class TestPrintFullReport:
    """Tests for the report printing function."""

    def test_print_full_report_does_not_crash(self, capsys):
        """Printing a valid report should not raise any exceptions."""
        from workflow import print_full_report

        report = make_test_report()
        print_full_report(report, "Test Agent")

        captured = capsys.readouterr()
        assert "FULL RESEARCH REPORT" in captured.out
        assert "EXECUTIVE SUMMARY" in captured.out
        assert "KEY FINDINGS" in captured.out

    def test_print_report_with_empty_fields(self, capsys):
        """Report with empty lists should print without error."""
        from workflow import print_full_report

        report = make_test_report(
            key_findings=[],
            current_data_statistics=[],
            recent_developments=[],
            evidence_sources=[],
            executive_summary="",
            source_analysis="",
            conclusion=""
        )
        print_full_report(report, "Empty Agent")
        captured = capsys.readouterr()
        assert "FULL RESEARCH REPORT" in captured.out


# ============================================================================
# Prompt Consistency Tests
# ============================================================================

class TestPromptConsistency:
    """Tests verifying prompt templates match the actual system configuration."""

    def test_synthesis_prompt_says_three_not_four(self):
        """The synthesis prompt should reference 3 agents, not 4."""
        from prompts import SYNTHESIS_PROMPT_TEMPLATE
        assert "three" in SYNTHESIS_PROMPT_TEMPLATE.lower()
        assert "four" not in SYNTHESIS_PROMPT_TEMPLATE.lower()

    def test_agent_configs_has_three_agents(self):
        """AGENT_CONFIGS should define exactly 3 research agents."""
        from agents import AGENT_CONFIGS
        assert len(AGENT_CONFIGS) == 3
        assert set(AGENT_CONFIGS.keys()) == {"claude", "openai", "zai"}

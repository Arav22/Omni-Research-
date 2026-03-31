"""
Tests for Pydantic models and data structures.
"""

import pytest
from models import ResearchReport, SynthesisReport, ResearchDependencies


# ============================================================================
# ResearchReport Tests
# ============================================================================

class TestResearchReport:
    """Tests for the ResearchReport Pydantic model."""

    def _make_report(self, **overrides) -> ResearchReport:
        """Helper to create a valid ResearchReport with sensible defaults."""
        defaults = {
            "agent_name": "Test Agent",
            "query": "test query",
            "executive_summary": "This is a test summary.",
            "key_findings": ["Finding 1", "Finding 2"],
            "current_data_statistics": ["Stat 1"],
            "recent_developments": ["Dev 1"],
            "source_analysis": "Sources are reliable.",
            "conclusion": "Test conclusion.",
            "evidence_sources": ["https://example.com"],
        }
        defaults.update(overrides)
        return ResearchReport(**defaults)

    def test_valid_report_creation(self):
        """A report with all required fields should construct without error."""
        report = self._make_report()
        assert report.agent_name == "Test Agent"
        assert report.query == "test query"
        assert len(report.key_findings) == 2

    def test_report_serialization_roundtrip(self):
        """model_dump → construct should produce an equivalent object."""
        original = self._make_report()
        data = original.model_dump()
        restored = ResearchReport(**data)
        assert restored == original

    def test_report_json_roundtrip(self):
        """model_dump_json → model_validate_json should preserve data."""
        original = self._make_report()
        json_str = original.model_dump_json()
        restored = ResearchReport.model_validate_json(json_str)
        assert restored == original

    def test_empty_lists_allowed(self):
        """Reports with empty list fields should be valid."""
        report = self._make_report(
            key_findings=[],
            current_data_statistics=[],
            recent_developments=[],
            evidence_sources=[]
        )
        assert report.key_findings == []
        assert report.evidence_sources == []

    def test_missing_required_field_raises(self):
        """Omitting a required field should raise a ValidationError."""
        with pytest.raises(Exception):
            ResearchReport(
                agent_name="Test",
                # missing query and other fields
            )

    def test_report_field_types(self):
        """Verify list fields are actually lists and string fields are strings."""
        report = self._make_report()
        assert isinstance(report.key_findings, list)
        assert isinstance(report.executive_summary, str)
        assert isinstance(report.evidence_sources, list)


# ============================================================================
# SynthesisReport Tests
# ============================================================================

class TestSynthesisReport:
    """Tests for the SynthesisReport Pydantic model."""

    def _make_synthesis(self, **overrides) -> SynthesisReport:
        """Helper to create a valid SynthesisReport with sensible defaults."""
        defaults = {
            "query": "test query",
            "agents_agreed": ["Point 1"],
            "agents_disagreed": ["Conflict 1"],
            "disagreement_reasons": ["Reason 1"],
            "combined_conclusions": ["Conclusion 1"],
            "unique_insights_by_agent": {
                "Agent A": ["Insight 1"],
                "Agent B": ["Insight 2"],
            },
            "actionable_insights": ["Action 1"],
        }
        defaults.update(overrides)
        return SynthesisReport(**defaults)

    def test_valid_synthesis_creation(self):
        """A synthesis with all required fields should construct without error."""
        report = self._make_synthesis()
        assert report.query == "test query"
        assert len(report.agents_agreed) == 1

    def test_synthesis_serialization_roundtrip(self):
        """model_dump → construct should produce an equivalent object."""
        original = self._make_synthesis()
        data = original.model_dump()
        restored = SynthesisReport(**data)
        assert restored == original

    def test_unique_insights_dict_structure(self):
        """unique_insights_by_agent should be a Dict[str, List[str]]."""
        report = self._make_synthesis()
        assert isinstance(report.unique_insights_by_agent, dict)
        for key, value in report.unique_insights_by_agent.items():
            assert isinstance(key, str)
            assert isinstance(value, list)

    def test_empty_synthesis_fields(self):
        """All list/dict fields can be empty."""
        report = self._make_synthesis(
            agents_agreed=[],
            agents_disagreed=[],
            disagreement_reasons=[],
            combined_conclusions=[],
            unique_insights_by_agent={},
            actionable_insights=[]
        )
        assert report.agents_agreed == []
        assert report.unique_insights_by_agent == {}


# ============================================================================
# ResearchDependencies Tests
# ============================================================================

class TestResearchDependencies:
    """Tests for the ResearchDependencies dataclass."""

    def test_default_search_count(self):
        """search_count should default to 0."""
        deps = ResearchDependencies(query="test", agent_name="Test Agent")
        assert deps.search_count == 0

    def test_custom_search_count(self):
        """search_count should accept custom initial values."""
        deps = ResearchDependencies(query="test", agent_name="Test", search_count=5)
        assert deps.search_count == 5

    def test_search_count_mutable(self):
        """search_count should be mutable (for rate-limiting tracking)."""
        deps = ResearchDependencies(query="test", agent_name="Test")
        deps.search_count += 1
        assert deps.search_count == 1
        deps.search_count += 1
        assert deps.search_count == 2

    def test_fields_stored_correctly(self):
        """All fields should be stored exactly as provided."""
        deps = ResearchDependencies(query="AI research trends", agent_name="Claude Agent")
        assert deps.query == "AI research trends"
        assert deps.agent_name == "Claude Agent"

# 🔬 OmniResearch

A **production-grade** parallel multi-agent research engine that combines **PydanticAI** and **LangGraph** to conduct comprehensive research analysis using multiple AI models simultaneously.

## 📍 Repository

**GitHub Repository**: [https://github.com/Arav22/Omni-Research-.git](https://github.com/Arav22/Omni-Research-.git)

## 🌟 Features

### Core Engine
- **Parallel Multi-Agent Research**: 3 AI agents (Claude, OpenAI, Z-AI) research simultaneously
- **Structured Output**: Type-safe Pydantic models ensure consistent data formats
- **Cross-Agent Synthesis**: Meta-analysis agent compares and consolidates findings
- **Real-time Web Search**: Integrated Tavily API for current information
- **Streaming Output**: Real-time progress tracking with colored terminal output

### Production-Grade Infrastructure
- **Structured Logging**: JSON-formatted rotating log files + colored console output
- **Correlation IDs**: Per-run tracing across all agents, searches, and synthesis
- **Retry with Backoff**: Exponential backoff (1s → 2s → 4s) on transient search failures
- **Error Isolation**: Individual agent failures don't crash the workflow
- **Protected Synthesis**: Synthesis node with graceful degradation
- **Full Tracebacks**: Exception chains preserved in log files for debugging
- **Configurable Log Levels**: `DEBUG`, `INFO`, `WARNING`, `ERROR` via `LOG_LEVEL` env var

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Claude Research │    │ OpenAI Research │    │  Z-AI Research  │
│     Agent       │    │     Agent       │    │     Agent       │
│   (Haiku 3.5)   │    │  (GPT-4.1-mini) │    │   (GLM-4.5)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                       ┌─────────────────┐
                       │   Synthesis     │
                       │     Agent       │
                       │  (GPT-4.1-mini) │
                       └─────────────────┘
                                │
                           📋 Final Report
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Agent Framework** | [PydanticAI](https://github.com/pydantic/pydantic-ai) | Type-safe agents with structured outputs |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | Fan-out/fan-in parallel workflow |
| **LLM Provider** | [OpenRouter](https://openrouter.ai/) | Multi-model API access |
| **Web Search** | [Tavily](https://tavily.com/) | Real-time web search |
| **Data Validation** | [Pydantic](https://pydantic.dev/) | Structured output models |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) | Fast Python dependency management |

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **uv** — Install from [docs.astral.sh/uv](https://docs.astral.sh/uv/)
- **OpenRouter API Key** — Get from [openrouter.ai](https://openrouter.ai/)
- **Tavily API Key** — Get from [tavily.com](https://tavily.com/)

### 1. Clone & Setup

```bash
git clone https://github.com/Arav22/Omni-Research-.git
cd Omni-Research-

# Install dependencies with uv (creates .venv automatically)
uv sync
```

### 2. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit with your API keys
```

**Required environment variables:**
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Optional logging configuration:**
```env
LOG_LEVEL=INFO          # DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_DIR=logs            # Directory for log files
LOG_JSON=false          # Set "true" for JSON-formatted console output
```

### 3. Run

```bash
# Interactive mode (recommended)
uv run python agent.py

# CLI mode with specific query
uv run python agent.py --query "Latest developments in artificial intelligence"

# Batch mode (no streaming)
uv run python agent.py --query "Climate change impact" --no-stream

# Debug logging
LOG_LEVEL=DEBUG uv run python agent.py --query "your query"
```

## 📁 Project Structure

```
omni-research/
├── agent.py              # CLI entry point — args, streaming output, result display
├── agents.py             # PydanticAI agent factory — research & synthesis agents
├── models.py             # Pydantic models & LangGraph state schema
├── workflow.py           # LangGraph workflow graph — fan-out/fan-in nodes
├── search.py             # Tavily search — singleton client, retry, backoff
├── prompts.py            # System prompt templates for research & synthesis
├── logging_config.py     # Production logging — console, JSON files, correlation IDs
├── pyproject.toml        # Project metadata & dependencies (uv)
├── uv.lock               # Locked dependency versions
├── .python-version       # Python version pin (3.13)
├── pytest.ini            # Pytest configuration
├── env.example           # Environment variables template
├── .gitignore            # Git ignore rules
├── tests/                # Test suite (46 tests)
│   ├── conftest.py       # Shared test configuration
│   ├── test_models.py    # Pydantic model validation tests
│   ├── test_search.py    # Search module tests (retry, backoff, mocked Tavily)
│   ├── test_workflow.py  # Workflow graph & node function tests
│   └── test_logging_config.py  # Logging module tests
└── logs/                 # Runtime logs (gitignored)
    ├── research.log          # All log levels (JSON, 10MB rotating)
    └── research.error.log    # Errors only (JSON, 10MB rotating)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `agent.py` | CLI entry point — argument parsing, streaming output, result display |
| `agents.py` | Agent factory — creates PydanticAI agents via shared factory function |
| `models.py` | `ResearchReport`, `SynthesisReport`, `ResearchDependencies`, `ResearchState` |
| `workflow.py` | LangGraph graph with parallel fan-out/fan-in + protected synthesis |
| `search.py` | Tavily search with singleton client, retry (3x), exponential backoff |
| `prompts.py` | `RESEARCHER_PROMPT_TEMPLATE` and `SYNTHESIS_PROMPT_TEMPLATE` |
| `logging_config.py` | Centralized logging — console + rotating JSON files + correlation IDs |

## 🔍 How It Works

### Research Process Flow

1. **Input**: User provides research query
2. **Correlation ID**: Unique `run-xxxx` ID generated for end-to-end tracing
3. **Parallel Execution**: 3 research agents start simultaneously:
   - **Claude Agent** (Haiku 3.5): Analytical research approach
   - **OpenAI Agent** (GPT-4.1-mini): Comprehensive analysis
   - **Z-AI Agent** (GLM-4.5): Efficient information gathering
4. **Web Search**: Each agent uses Tavily with retry and exponential backoff
5. **Error Isolation**: Failed agents produce dummy reports — workflow continues
6. **Synthesis**: Meta-analysis agent compares all findings
7. **Output**: Structured report with agreements, disagreements, and insights

### Output Structure

- **Points of Agreement** — Where all agents reached consensus
- **Areas of Disagreement** — Conflicting findings between agents
- **Disagreement Analysis** — Why conflicts occurred
- **Combined Conclusions** — Synthesized insights from all agents
- **Unique Insights** — Agent-specific discoveries
- **Actionable Insights** — Practical next steps

## ⚙️ Configuration

### Testing vs Production Settings

The system is **currently configured for testing** with reduced limits to minimize API costs:

| Setting | Testing (Current) | Production (Recommended) |
|---------|-------------------|--------------------------|
| **Searches per agent** | 2 | 5–10 |
| **Results per search** | 1 | 3–5 |
| **Search timeout** | 10 seconds | 30+ seconds |
| **Search depth** | `"basic"` | `"advanced"` |
| **Retry attempts** | 3 | 3 |

To switch to production, update the `TODO: PRODUCTION` comments in `search.py` and `agents.py`.

## 📊 Logging & Observability

### Log Files

| File | Content | Format |
|------|---------|--------|
| `logs/research.log` | All log levels (DEBUG+) | JSON, 10MB rotating, 5 backups |
| `logs/research.error.log` | Errors only | JSON, 10MB rotating, 5 backups |

### Console Output

```
2026-03-31 04:30:00.123 [INFO    ] [run-a1b2c3d4e5f6] research.workflow: Agent completed: Claude — 5 findings (4523ms)
```

### Structured JSON Log Entry

```json
{
  "timestamp": "2026-03-31T04:30:00.123456+00:00",
  "level": "INFO",
  "logger": "research.workflow",
  "correlation_id": "run-a1b2c3d4e5f6",
  "message": "Agent completed: Claude Research Agent — 5 findings, 3 sources (4523ms)",
  "agent_name": "Claude Research Agent",
  "duration_ms": 4523,
  "result_count": 5
}
```

## 🧪 Testing

```bash
# Run full test suite (46 tests)
uv run pytest tests/ -v

# Run specific test module
uv run pytest tests/test_logging_config.py -v
uv run pytest tests/test_search.py -v
uv run pytest tests/test_workflow.py -v
uv run pytest tests/test_models.py -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `logging_config.py` | 13 | Correlation IDs, formatters, setup, cleanup |
| `search.py` | 7 | Singleton, retry, backoff, timeout, errors |
| `workflow.py` | 6 | Graph construction, agent execution, reporting |
| `models.py` | 8 | Validation, serialization, field types |
| **Total** | **46** | **All passing ✅** |

## 🔧 Troubleshooting

### Common Issues

| Error | Solution |
|-------|----------|
| `ValueError: Missing required environment variables` | Create `.env` from `env.example` with valid API keys |
| `ModuleNotFoundError` | Run `uv sync` to install dependencies |
| `Search failed after 3 attempts` | Check network, increase timeouts for production |
| Unhandled crash | Check `logs/research.error.log` for full traceback |

### Debug Mode

```bash
# Enable full debug logging
LOG_LEVEL=DEBUG uv run python agent.py --query "test"

# Check error logs after a crash
cat logs/research.error.log | python -m json.tool
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Run the test suite: `uv run pytest tests/ -v`
4. Create a pull request

### Code Standards

- PEP 8 formatting with type hints on all functions
- Comprehensive docstrings (Google style)
- Error handling with full traceback preservation
- Structured logging (no bare `print()` for operational info)
- Tests for all new functionality

### Extending

- **New Agents**: Add to `AGENT_CONFIGS` in `agents.py` + node in `workflow.py`
- **New Search Providers**: Extend `search_web()` in `search.py`
- **New Output Formats**: Add Pydantic models in `models.py`

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 🙏 Acknowledgments

- **[PydanticAI](https://github.com/pydantic/pydantic-ai)** — Type-safe agent framework
- **[LangGraph](https://github.com/langchain-ai/langgraph)** — Workflow orchestration
- **[OpenRouter](https://openrouter.ai/)** — Multi-model API access
- **[Tavily](https://tavily.com/)** — Real-time web search
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package management

---

**🔬 OmniResearch — See the whole picture.**
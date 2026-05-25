# 🤖 LangGraph Agentic AI Platform — v2

A production-grade multi-agent AI platform built on **LangGraph** and **Streamlit**.  
Upgrades the original single-graph chatbot into a fully modular, observable, streaming multi-agent system.

---

## 📁 Project Structure

```
AgenticChatbot_v2/
│
├── app.py                          # Streamlit entrypoint
├── requirements.txt
│
└── src/
    ├── main.py                     # Orchestration layer
    │
    ├── agents/                     # Specialist agents
    │   ├── supervisor_agent.py     # Routes queries to the right specialist
    │   ├── chat_agent.py           # General conversation (memory-aware)
    │   ├── tool_agent.py           # Tool-augmented reasoning
    │   ├── news_agent.py           # AI news fetch → summarise → save
    │   ├── research_agent.py       # Deep multi-source research
    │   └── formatter_agent.py      # Polishes final responses
    │
    ├── graph/
    │   ├── state.py                # Unified AgentState TypedDict
    │   └── graph_builder.py        # Compiles + caches LangGraph graphs
    │
    ├── tools/
    │   └── tool_registry.py        # Tool definitions + retry/fallback
    │
    ├── memory/
    │   └── memory_manager.py       # Per-session short-term memory
    │
    ├── observability/
    │   └── logging_config.py       # Structured logging + metrics + tracing
    │
    ├── llm/
    │   └── provider.py             # Cached LLM initialisation
    │
    ├── config/
    │   └── settings.py             # All config in one place
    │
    └── ui/
        ├── loadui.py               # Sidebar controls
        └── display_result.py       # Chat rendering + streaming
```

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set these environment variables (or enter them in the sidebar):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq LLM access |
| `TAVILY_API_KEY` | Web search + news |

---

## 🎯 Use Cases

| Use Case | Description |
|---|---|
| **Basic Chatbot** | Context-aware conversation with rolling memory |
| **Chatbot with Web** | Live web search via Tavily, Wikipedia, arXiv |
| **AI News** | Daily / weekly / monthly AI news digest |
| **Multi-Agent Research** | Supervisor routes to specialist → formatter polishes output |

---

## 🧠 Feature 1 — Memory System

`SessionMemory` holds a capped deque of `AnyMessage` objects per session.  
`MemoryManager` manages all active sessions and evicts stale ones.

Every graph input is built via `session_memory.build_graph_state(user_message)` 
which prepends history so the LLM always sees the full conversation context.

---

## 🤖 Feature 2 — Multi-Agent Collaboration

```
User Query
   │
   ▼
SupervisorAgent          ← LLM-based semantic classifier
   │
   ├─→ chat_agent        ← general Q&A / conversation
   ├─→ tool_agent        ← web/wiki/arxiv/calculator
   ├─→ research_agent    ← deep multi-source research
   └─→ formatter_agent   ← direct formatting requests
         │
         ▼
   FormatterAgent        ← polishes every response
         │
         ▼
        END
```

Routing is LLM-based (not keyword-based) using a structured JSON decision.

---

## 🛠 Feature 3 — Real Tool Calling

Registered tools in `tool_registry.py`:
- **TavilySearch** — real-time web search
- **Wikipedia** — encyclopaedic lookup
- **Arxiv** — scientific paper search
- **Calculator** — safe math expression evaluator

All tools include:
- Retry with exponential back-off (configurable in `settings.py`)
- Graceful fallback stub response on total failure
- Per-tool call metrics tracking

---

## ⚡ Feature 4 — Streaming Responses

- **Basic Chatbot**: token-level streaming via `graph.stream(stream_mode="values")`
- **Chatbot with Web**: full invoke (tool loops require complete graph traversal), then stream render
- **Multi-Agent Research**: event-level streaming with live agent transition indicators
- **AI News**: spinner during pipeline execution; markdown render on completion

---

## 📊 Feature 5 — Observability / Monitoring

`MetricsCollector` tracks (in-memory, session-scoped):
- Total requests, errors, tool calls
- Per-agent and per-tool call counts
- Response latency histogram (last 500 samples, rolling average)

Structured logs via Python `logging` with timestamp, level, module, and message.  
All node entry/exit is wrapped in `trace_node()` context manager.

**LangSmith**: set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to enable.

Metrics are visible in the **Session Metrics** expander in the sidebar.

---

## 🚀 Feature 6 — Performance Optimisations

| Optimisation | Where |
|---|---|
| LLM instance cached per `(api_key, model_name)` | `llm/provider.py` |
| Compiled graph cached per `(usecase, model_name)` | `graph/graph_builder.py` |
| Tool instances cached at module level | `tools/tool_registry.py` |
| Session memory evicts stale sessions | `memory/memory_manager.py` |
| No redundant Streamlit reruns (flag reset after consume) | `main.py` |
| Retry with back-off avoids thundering-herd on tool failures | `tools/tool_registry.py` |
| Single unified `AgentState` eliminates duplicate state conversion | `graph/state.py` |

---

## 🔧 Configuration

All tuneable parameters live in `src/config/settings.py`:

```python
MEMORY_CONFIG.max_history_messages = 20     # rolling context window
MEMORY_CONFIG.session_ttl_seconds  = 3600   # 1 hour idle session eviction
TOOL_CONFIG.max_retries            = 2      # tool call retry limit
TOOL_CONFIG.tavily_max_results     = 5      # articles per search
```

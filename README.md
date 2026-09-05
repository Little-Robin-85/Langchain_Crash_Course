# LangChain Crash Course: AI Agents with Google Gemini & LangChain 1.0

This repository is a complete, production-grade recreation of the **LangChain Full Crash Course - AI Agents in Python** (based on the tutorial by [NeuralNine](https://youtu.be/J7j5tCB_y4w?list=PLeCnTKzTkM94)), fully adapted for the **latest Google Gemini models and standards**.

Every module contains **verbose educational commentary**, explaining:
- **The Concept**: Why the pattern exists and when to use it.
- **The Architecture**: How LangChain 1.0 and LangGraph manage state, tools, and message flows.
- **The Implementation (The How and The Why)**: Line-by-line rationale for function decorators, type hints, parameters, and return types.
- **Gemini-Specific Nuances**: Native tool calling, JSON schema enforcement, native multimodal vision, and embedding dimensions.

---

## 📚 Table of Contents & Roadmap

| Module | Script | Topic & Concepts Covered |
| :--- | :--- | :--- |
| **01** | [`01_simple_agent_weather.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/01_simple_agent_weather.py) | **Simple AI Agent & Tool Calling**: The ReAct decision loop, `@tool` decorator, OpenAPI schema translation, `create_agent`, and invoking external APIs (`wttr.in`). |
| **02** | [`02_standalone_model_inference.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/02_standalone_model_inference.py) | **Standalone Model Inference**: When NOT to use an agent, `init_chat_model` universal abstraction, temperature controls, and token usage inspection. |
| **03** | [`03_conversation_history.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/03_conversation_history.py) | **Conversations & Message Roles**: Managing stateless LLMs, `SystemMessage`, `HumanMessage`, `AIMessage`, and retaining multi-turn context. |
| **04** | [`04_streaming_responses.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/04_streaming_responses.py) | **Streaming Responses**: Minimizing Time-to-First-Token (TTFT), `model.stream()`, `AIMessageChunk`, and live stdout flushing for chat UIs. |
| **05** | [`05_advanced_agent_context_memory_structured_output.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/05_advanced_agent_context_memory_structured_output.py) | **Advanced Agent**: Out-of-band `ToolRuntime[Context]`, multi-turn thread memory with `InMemorySaver`, and Pydantic `WeatherResponse` schema validation. |
| **06** | [`06_multimodal_input.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/06_multimodal_input.py) | **Multimodal Vision**: Gemini native multimodal understanding, Base64/Data URI encoding, MIME type detection, and chart/diagram analysis. |
| **07** | [`07_rag_example.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/07_rag_example.py) | **RAG (Retrieval-Augmented Generation)**: Google embeddings (`gemini-embedding-001`), in-memory FAISS vector indexing, similarity search, and `create_retriever_tool`. |
| **08** | [`08_dynamic_system_prompt.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/08_dynamic_system_prompt.py) | **Dynamic System Prompts**: Middleware `@dynamic_prompt`, inspecting runtime user roles (`child`, `beginner`, `expert`), and adapting system instructions on-the-fly. |
| **09** | [`09_dynamic_model_choice.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/09_dynamic_model_choice.py) | **Dynamic Model Choice**: Cost/latency optimization with `@wrap_model_call`, tiered routing between `gemini-3.5-flash-lite` and `gemini-3.6-flash`. |
| **10** | [`10_custom_agent_middleware.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/10_custom_agent_middleware.py) | **Custom Middleware & Lifecycle Hooks**: Subclassing `AgentMiddleware`, tracking `before_agent`, `before_model`, `after_model`, and `after_agent`. |
| **11** | [`11_builtin_middleware_examples.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/11_builtin_middleware_examples.py) | **Built-in Middleware**: Context compression with `SummarizationMiddleware`, catalog of production patterns (HITL, rate limiters, fallback). |
| **12** | [`12_langsmith_tracing.py`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/12_langsmith_tracing.py) | **LangSmith Tracing Deep Dive**: End-to-end telemetry, capturing direct trace URLs, custom tags/metadata, and `@traceable` functions. |

---

## 🛠️ Environment Setup & Quick Start

### 1. Virtual Environment & Dependencies

This repository uses Python 3.12 (managed via `uv` or standard Python `venv`) to ensure compatibility with FAISS, NumPy, and LangChain:

```bash
# Create a Python 3.12 virtual environment
uv venv .venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate

# Install all required dependencies
uv pip install -r requirements.txt
```

### 2. Environment Configuration (`.env`)

Ensure your `.env` file in the project root contains your API keys (see [`.env.example`](file:///home/ljdeveney/Projects/Langchain_Crash_Course/.env.example) for a template):

```ini
# Google AI Studio API Key
GEMINI_API_KEY="your_gemini_api_key"
GOOGLE_API_KEY="your_gemini_api_key"

# Models
GEMINI_MODEL="gemini-3.5-flash-lite"
ADVANCED_MODEL="gemini-3.6-flash"
Embedding="gemini-embedding-001"

# LangSmith Tracing & Observability (Optional)
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://eu.api.smith.langchain.com"
LANGCHAIN_PROJECT="LangChain-Gemini-Crash-Course"
```

---

## 🚀 Running the Modules

You can execute any module independently using the virtual environment:

```bash
# 1. Simple Agent with Weather Tool
.venv/bin/python 01_simple_agent_weather.py

# 2. Standalone Model Inference
.venv/bin/python 02_standalone_model_inference.py

# 3. Multi-turn Conversation History
.venv/bin/python 03_conversation_history.py

# 4. Streaming Tokens to Stdout
.venv/bin/python 04_streaming_responses.py

# 5. Advanced Agent (Context, Checkpointer Memory, Structured Pydantic Output)
.venv/bin/python 05_advanced_agent_context_memory_structured_output.py

# 6. Multimodal Visual Reasoning over Sample Chart
.venv/bin/python 06_multimodal_input.py

# 7. RAG (Vector Similarity Search & Retrieval Agent with FAISS)
.venv/bin/python 07_rag_example.py

# 8. Dynamic System Prompt Adaptation (Child vs Expert)
.venv/bin/python 08_dynamic_system_prompt.py

# 9. Dynamic Model Choice (Tiered Flash-Lite vs Flash Routing)
.venv/bin/python 09_dynamic_model_choice.py

# 10. Custom Middleware & Lifecycle Hooks
.venv/bin/python 10_custom_agent_middleware.py

# 11. Built-in Summarization Middleware
.venv/bin/python 11_builtin_middleware_examples.py

# 12. LangSmith Tracing & Direct Trace URL Capture
.venv/bin/python 12_langsmith_tracing.py
```

---

## 🔍 Key Architectural Insights

### 1. LangChain 1.0 Agent Paradigm (`create_agent`)
In legacy LangChain, agents relied on `initialize_agent` or `AgentExecutor`, which suffered from complex parsing errors and rigid execution loops. LangChain 1.0 unifies agents on top of **LangGraph**:
- Every agent is an executable directed graph.
- Middleware runs as lifecycle intercepts.
- State is preserved across turns using a `checkpointer` (such as `InMemorySaver`).

### 2. Google Gemini Best Practices
- **Provider Identifier**: Use `"google_genai:<model_name>"` with `init_chat_model` or `create_agent`.
- **Model Tiers**:
  - `gemini-3.5-flash-lite`: Recommended for high-frequency tool calling, lightweight reasoning, and summarization.
  - `gemini-3.6-flash`: Recommended for deep multi-turn synthesis, complex visual analysis, and strict JSON schemas.
- **Embeddings**: `gemini-embedding-001` yields rich 3,072-dimensional vectors for semantic search.
- **Structured Outputs**: Direct integration with Pydantic ensures zero schema hallucination.

### 3. LangSmith Observability
Because `.env` includes `LANGCHAIN_TRACING_V2="true"` and your LangSmith credentials, every agent execution, tool call parameter, model latency, and token count is automatically streamed to your LangSmith dashboard for real-time inspection.

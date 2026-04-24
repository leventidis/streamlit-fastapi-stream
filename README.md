## Streamlit + FastAPI with Streaming

A minimal working example of streaming intermediate agent reasoning steps to a user in real time, before the final answer is ready. Supports both Azure OpenAI and native OpenAI — see `.env.example`.

```
pip install -r requirements.txt

# Run backend (in one terminal)
fastapi dev src/api.py

# Run frontend (in another terminal)
streamlit run src/frontend.py
```

---

## How Streaming Works

```
Streamlit frontend  ←── SSE ──  FastAPI /stream  ←── astream_events ──  LangGraph graph
```

### 1. LangGraph — `astream_events`

The graph is executed using `graph.astream_events(version="v2")` instead of the usual `graph.invoke()`. This emits a continuous stream of fine-grained events as the graph runs — including which node is active and every token produced by an LLM — rather than waiting for the full result. Each event carries a `metadata.langgraph_node` field identifying its source node.

### 2. FastAPI — Server-Sent Events (SSE)

The `/stream` endpoint returns a `StreamingResponse` with `media_type="text/event-stream"`. An async generator consumes the LangGraph event stream and translates it into three typed SSE events:

| SSE event | Triggered when | Contains |
|---|---|---|
| `status` | An LLM call starts inside a node | Short human-readable label, e.g. *"Creating a plan..."* |
| `trace` | A node finishes | JSON with node name, completion label, and elapsed time in ms |
| `answer` | A token arrives from the final answer node | One token chunk |

The SSE wire format is simple — two lines per event followed by a blank line:
```
event: trace
data: {"node": "planning", "label": "Plan created", "duration_ms": 843}

event: answer
data: Paris is the capital
```

### 3. Streamlit — Live trace panel + streaming answer

The frontend consumes the SSE stream line-by-line using `requests.post(..., stream=True)`.

- **`st.status()`** acts as the live trace panel. It shows a spinner while the agent works, receives a ✅ checkmark for each completed node via `.write()`, and transitions to a green *Done* state when the stream ends. Crucially, it **stays visible** after completion — giving the user a persistent view of the agent's reasoning steps, similar to how Claude Code and VS Code Copilot display their tool calls.
- Answer tokens are appended into an `st.empty()` placeholder below the trace panel, producing the word-by-word streaming effect.
- Past traces are stored in `st.session_state` and replayed as a collapsed `st.expander` in chat history.

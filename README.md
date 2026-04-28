## Streamlit + FastAPI with Streaming

A proof-of-concept that demonstrates **streaming agent reasoning traces** to a user in real time — before the final answer is ready. The experience is modelled on how tools like VS Code Copilot and Claude Code display their reasoning steps live as they work through a problem.

Supports both **Azure OpenAI** and **native OpenAI** — see `.env.example`.

```
pip install -r requirements.txt

# Run backend (in one terminal)
fastapi dev src/api.py

# Run frontend (in another terminal)
streamlit run src/frontend.py
```

---

## Architecture

```
┌──────────────────┐   HTTP SSE stream    ┌──────────────────┐   astream_events   ┌──────────────────┐
│  Streamlit UI    │ ◄─────────────────── │  FastAPI /stream │ ◄───────────────── │  LangGraph graph │
│  (frontend.py)   │                      │  (api.py)        │                    │  (graph.py)      │
└──────────────────┘                      └──────────────────┘                    └──────────────────┘
```

The three layers have clean responsibilities:

| Layer | Responsibility |
|---|---|
| **LangGraph** | Executes the agent graph and emits fine-grained events for every node transition and LLM token |
| **FastAPI** | Translates raw LangGraph events into typed SSE events and forwards them over an HTTP stream |
| **Streamlit** | Consumes the SSE stream and renders a live trace panel + streaming answer |

---

## The Graph (`graph.py`)

The graph has four nodes. The `orchestrate` node is a router — it decides at runtime whether the question needs planning or can be answered directly:

```
START
  └── orchestrate ──┬── (planning_required=True)  ──► planning ──► generate_joke ──► answer_question ──► END
                    └── (planning_required=False) ───────────────────────────────► answer_question ──► END
```

A helper `_make_llm()` factory selects `AzureChatOpenAI` or `ChatOpenAI` based on environment variables — the graph itself is provider-agnostic.

---

## How Streaming Works

### Step 1 — LangGraph emits fine-grained events

`graph.astream_events(version="v2")` is the key API. Unlike `graph.invoke()` (which returns only the final result) or `graph.astream()` (which returns state after each node), `astream_events` emits a **continuous event stream** during execution. Each event has:

- `event` — the event type (e.g. `on_chain_start`, `on_chat_model_stream`)
- `name` — the node or chain name that produced it
- `metadata.langgraph_node` — the active graph node
- `run_id` — a unique ID for correlating start/end pairs (used for timing)
- `data` — event-specific payload (tokens, outputs, usage, etc.)

The events used by this application:

| LangGraph event | When it fires | Used for |
|---|---|---|
| `on_chain_start` | A tracked node begins execution | Record `time.monotonic()` for timing |
| `on_chat_model_start` | An LLM call inside a node starts | Emit `status` SSE (node is active) |
| `on_chat_model_stream` | A token arrives from an LLM | Emit `node_output` (intermediate) or `answer` (final node) |
| `on_chat_model_end` | An LLM call finishes | Accumulate token usage |
| `on_chain_end` | A tracked node finishes | Emit `trace` SSE with timing; emit `routing` for orchestrate |

### Step 2 — FastAPI translates events to typed SSE

The `llm_chat_generator` async generator in `api.py` filters and transforms the LangGraph event stream into **six typed SSE events** sent to the client:

```
event: status
data: {"node": "planning", "label": "Creating a plan..."}

event: node_output
data: {"node": "planning", "token": "Step 1: Understand"}

event: routing
data: {"planning_required": true}

event: trace
data: {"node": "planning", "label": "Plan created", "duration_ms": 1843}

event: answer
data: "Based on the plan, here is"

event: done
data: {"input_tokens": 312, "output_tokens": 187, "elapsed_seconds": 4.2}
```

Each SSE event is two lines followed by a blank line — this is the standard wire format:
```
event: <type>
data: <json payload>

```

Key design decisions in `api.py`:
- `node_start_times` keyed by `run_id` enables accurate per-node timing using `on_chain_start` / `on_chain_end` pairs.
- `announced_nodes` prevents duplicate `status` events if a node is retried.
- `node_output` is only emitted for intermediate nodes (defined in `NODE_OUTPUT_LABELS`); `answer` is only emitted for the final `answer_question` node — this keeps reasoning content and the final answer on separate channels.
- The `orchestrate` node emits both a `trace` event (timing) and a `routing` event (decision) when it finishes.

### Step 3 — Streamlit renders the live trace and streams the answer

`frontend.py` opens a persistent HTTP connection with `requests.post(..., stream=True)` and processes lines as they arrive. The rendering is built around two core primitives:

**`st.status()`** — the collapsible trace container. It displays a spinner and label while the agent is working. Inside it, a **single `st.empty()` placeholder** holds the entire trace panel. On every incoming event, the placeholder is completely replaced with freshly rendered markdown — this is what makes the trace appear to "update live" as each node progresses.

**`render_trace_markdown(trace_state)`** — a pure function that converts the current in-memory trace state (an ordered dict of nodes) into structured markdown. Each node section shows:
- ⏳ spinner icon while running
- ✅ checkmark with elapsed time when done
- The node's streaming content indented as a blockquote

This is the same pattern used by Claude Code and VS Code Copilot: a structured, updating panel that shows what each step did, with the final answer appearing below once it is ready.

**State tracked per request:**

| Variable | Type | Purpose |
|---|---|---|
| `trace_state` | `dict[node → {label, content, status, duration_ms}]` | Live trace, re-rendered on every event |
| `node_outputs` | `dict[node → str]` | Accumulated intermediate content (for history) |
| `trace_steps` | `list[dict]` | Compact step summary stored in chat history |
| `routing` | `dict` | Planning decision from orchestrate |
| `stats` | `dict` | Token counts and elapsed time from `done` event |

**After streaming completes:**
- The trace collapses to `✅ Agent trace` (still re-expandable)
- A routing badge shows whether the question was planned or answered directly
- Token/timing stats appear as a caption below the answer
- The full `ChatMessage` (including `trace_steps`) is stored in `st.session_state` for history replay

---

## Event Order for a Planned Question

For a question that triggers planning, the full SSE sequence is:

```
status      {"node": "orchestrate", "label": "Routing question..."}
trace       {"node": "orchestrate", "label": "Route decided", "duration_ms": 31}
routing     {"planning_required": true}
status      {"node": "planning", "label": "Creating a plan..."}
node_output {"node": "planning", "token": "Step 1: ..."}     ← repeated per token
node_output {"node": "planning", "token": " understand"}     ← repeated per token
trace       {"node": "planning", "label": "Plan created", "duration_ms": 1843}
status      {"node": "generate_joke", "label": "Generating a joke..."}
node_output {"node": "generate_joke", "token": "Why did..."}  ← repeated per token
trace       {"node": "generate_joke", "label": "Joke generated", "duration_ms": 612}
status      {"node": "answer_question", "label": "Formulating answer..."}
answer      "Based on the plan,"                              ← repeated per token
answer      " here is the answer..."                         ← repeated per token
trace       {"node": "answer_question", "label": "Answer generated", "duration_ms": 3201}
done        {"input_tokens": 312, "output_tokens": 187, "elapsed_seconds": 6.1}
```

For a direct (no-planning) question, the `planning` and `generate_joke` nodes are skipped entirely.

---

## File Structure

```
src/
  api.py            Backend: SSE event generator, FastAPI app
  graph.py          LangGraph graph definition and node functions
  frontend.py       Streamlit UI: SSE consumer, trace renderer, chat history
  chat_message.py   ChatMessage dataclass with render() for history replay
  models.py         Pydantic request models
  components/
    sidebar.py      Sidebar component
.env.example        Environment variable template
```

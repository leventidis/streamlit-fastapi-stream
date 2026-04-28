import json
import requests
import streamlit as st
from chat_message import ChatMessage, NODE_OUTPUT_LABELS
from components.sidebar import sidebar

API_URL = "http://localhost:8000/stream"

# Icon per graph node — extend this as the graph grows
NODE_ICONS: dict[str, str] = {
    "orchestrate": "🔀",
    "planning": "📋",
    "generate_joke": "😄",
    "answer_question": "💬",
}

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="LLM Streamer", page_icon="🤖", layout="wide")

sidebar()


def parse_sse_line(line: str) -> tuple[str | None, str | None]:
    """Parse a single SSE line, returning (event_type, data) or (None, None)."""
    if line.startswith("event:"):
        return "event", line[len("event:"):].strip()
    if line.startswith("data:"):
        return "data", json.loads(line[len("data:"):].removeprefix(" "))
    return None, None


def render_trace_markdown(trace_state: dict[str, dict]) -> str:
    """Render the live trace state as structured markdown.

    Each node gets a section that looks like:

        ⏳ 📋 **Creating a plan...**          ← while running
        > Step 1: Understand the question...  ← streaming tokens
        > Step 2: ...

        ✅ 📋 **Plan created** `1843 ms`      ← after completion
        > Step 1: ...
        > Step 2: ...

    Sections are separated by horizontal rules so each node is visually distinct.
    """
    if not trace_state:
        return "*Starting...*"

    sections: list[str] = []
    for node_name, data in trace_state.items():
        is_done = data["status"] == "done"
        status_icon = "✅" if is_done else "⏳"
        node_icon = NODE_ICONS.get(node_name, "🔹")
        duration = f" `{data['duration_ms']} ms`" if is_done and data.get("duration_ms") else ""
        header = f"{status_icon} {node_icon} **{data['label']}**{duration}"

        content = data.get("content", "").strip()
        if content:
            quoted = "\n".join(
                f"> {line}" if line.strip() else ">"
                for line in content.split("\n")
            )
            sections.append(f"{header}\n{quoted}")
        else:
            sections.append(header)

    return "\n\n---\n\n".join(sections)


# ── Chat history ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    message.render()

if not st.session_state.messages:
    st.info("Ask me anything! I'll come up with a plan and answer your question.")


# ── Live interaction ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me something..."):
    user_message = ChatMessage(role="user", content=prompt)
    user_message.render()

    with st.chat_message("assistant"):
        # st.status is the collapsible trace container.
        # It stays expanded and shows a spinner while the agent works,
        # then collapses with a checkmark once the answer is rendered —
        # mimicking the Copilot / Claude Code "reasoning steps" panel.
        trace_status = st.status("🤖 Agent is reasoning...", expanded=True)
        with trace_status:
            # Single st.empty() that we replace on every event — gives full
            # control over the trace layout without Streamlit ordering issues.
            trace_placeholder = st.empty()

        # The final streamed answer appears below the trace panel
        answer_placeholder = st.empty()

        current_event: str | None = None
        # trace_state is an ordered dict: node_name → {label, content, status, duration_ms}
        trace_state: dict[str, dict] = {}
        node_outputs: dict[str, str] = {}   # accumulated content per node (for history)
        stats: dict = {}
        routing: dict = {}
        trace_steps: list[dict] = []        # compact step list (for history)
        response_text = ""

        def update_trace() -> None:
            trace_placeholder.markdown(render_trace_markdown(trace_state))

        with requests.post(API_URL, json={"prompt": prompt}, stream=True) as response:
            for raw_line in response.iter_lines(decode_unicode=True):
                kind, value = parse_sse_line(raw_line)

                if kind == "event":
                    current_event = value

                elif kind == "data":

                    if current_event == "status":
                        # value: {"node": "planning", "label": "Creating a plan..."}
                        node = value["node"]
                        if node not in trace_state:
                            trace_state[node] = {
                                "label": value["label"],
                                "content": "",
                                "status": "running",
                                "duration_ms": None,
                            }
                            update_trace()

                    elif current_event == "routing":
                        # value: {"planning_required": bool}
                        routing.update(value)
                        route_text = "📋 Planning required" if value.get("planning_required") else "⚡ Direct answer"
                        if "orchestrate" in trace_state:
                            trace_state["orchestrate"]["content"] = route_text
                            update_trace()

                    elif current_event == "trace":
                        # value: {"node": "planning", "label": "Plan created", "duration_ms": 843}
                        node = value["node"]
                        trace_steps.append(value)
                        if node in trace_state:
                            trace_state[node].update({
                                "label": value["label"],
                                "status": "done",
                                "duration_ms": value.get("duration_ms"),
                            })
                        else:
                            # Node had no prior status event (e.g. direct answer_question)
                            trace_state[node] = {
                                "label": value["label"],
                                "content": "",
                                "status": "done",
                                "duration_ms": value.get("duration_ms"),
                            }
                        update_trace()

                    elif current_event == "node_output":
                        # value: {"node": "planning", "token": "some text"}
                        node = value["node"]
                        token = value["token"]
                        node_outputs[node] = node_outputs.get(node, "") + token
                        if node in trace_state:
                            trace_state[node]["content"] = node_outputs[node]
                            update_trace()

                    elif current_event == "answer":
                        # value: plain token string
                        response_text += value
                        answer_placeholder.markdown(response_text + "▌")

                    elif current_event == "done":
                        stats.update(value)

        answer_placeholder.markdown(response_text)

        # Collapse trace once the answer is fully shown — user can still click to re-expand
        trace_status.update(label="✅ Agent trace", state="complete", expanded=False)

        if routing.get("planning_required") is not None:
            label, color = ("🗺 Planned", "green") if routing["planning_required"] else ("⚡ Direct", "orange")
            st.badge(label, color=color)

        if stats:
            st.caption(
                f"⏱ {stats['elapsed_seconds']}s · "
                f"↑ {stats['input_tokens']} tokens in · "
                f"↓ {stats['output_tokens']} tokens out"
            )

    st.session_state.messages.append(user_message)
    st.session_state.messages.append(ChatMessage(
        role="assistant",
        content=response_text,
        node_outputs=node_outputs,
        stats=stats,
        planning_required=routing.get("planning_required"),
        trace_steps=trace_steps,
    ))

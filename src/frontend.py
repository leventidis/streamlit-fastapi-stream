import json
import requests
import streamlit as st
from chat_message import ChatMessage, NODE_OUTPUT_LABELS
from components.sidebar import sidebar

API_URL = "http://localhost:8000/stream"

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


# ── Chat history ─────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    message.render()

if not st.session_state.messages:
    st.info("Ask me anything! I'll come up with a plan and answer your question.")


# ── Live interaction ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me something..."):
    user_message = ChatMessage(role="user", content=prompt)
    user_message.render()

    with st.chat_message("assistant"):
        # st.status acts as a live collapsible trace panel — stays visible after completion
        trace_status = st.status("Agent is working...", expanded=True)
        answer_placeholder = st.empty()

        current_event = None
        node_outputs: dict[str, str] = {}
        stats: dict = {}
        routing: dict = {}
        trace_steps: list[dict] = []
        response_text = ""

        with requests.post(API_URL, json={"prompt": prompt}, stream=True) as response:
            for raw_line in response.iter_lines(decode_unicode=True):
                kind, value = parse_sse_line(raw_line)

                if kind == "event":
                    current_event = value

                elif kind == "data":
                    if current_event == "status":
                        trace_status.update(label=value, state="running")

                    elif current_event == "routing":
                        routing.update(value)

                    elif current_event == "trace":
                        # A node finished — append a checkmark with timing inside the trace panel
                        trace_steps.append(value)
                        duration = f"  `{value['duration_ms']} ms`" if "duration_ms" in value else ""
                        trace_status.write(f"✅ {value['label']}{duration}")

                    elif current_event == "node_output":
                        # value is formatted as "node_name:token"
                        node, _, token = value.partition(":")
                        node_outputs[node] = node_outputs.get(node, "") + token

                    elif current_event == "answer":
                        response_text += value
                        answer_placeholder.markdown(response_text + "▌")

                    elif current_event == "done":
                        stats.update(value)

        answer_placeholder.markdown(response_text)
        trace_status.update(label="Done", state="complete", expanded=True)

        if routing.get("planning_required") is not None:
            label, color = ("🗺 Planned", "green") if routing["planning_required"] else ("⚡ Direct", "orange")
            st.badge(label, color=color)

        for node, content in node_outputs.items():
            label = NODE_OUTPUT_LABELS.get(node, node)
            with st.expander(label):
                st.markdown(content)

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

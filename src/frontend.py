import streamlit as st
import requests

API_URL = "http://localhost:8000/stream"

st.title("FastAPI + Streamlit LLM Streamer")

if "messages" not in st.session_state:
    st.session_state.messages = []

def parse_sse_line(line: str) -> tuple[str | None, str | None]:
    """Parse a single SSE line, returning (event_type, data) or (None, None)."""
    if line.startswith("event:"):
        return "event", line[len("event:"):].strip()
    if line.startswith("data:"):
        return "data", line[len("data:"):].removeprefix(" ")
    return None, None

def stream_response(prompt, status_placeholder):
    """Consume the SSE stream, show status events in a box, yield answer chunks."""
    current_event = None

    with requests.post(API_URL, json={"prompt": prompt}, stream=True) as response:
        for raw_line in response.iter_lines(decode_unicode=True):
            kind, value = parse_sse_line(raw_line)

            if kind == "event":
                current_event = value
            elif kind == "data":
                if current_event == "status":
                    status_placeholder.status(value, state="running")
                elif current_event == "answer":
                    status_placeholder.empty()
                    yield value

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me something..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_text = st.write_stream(stream_response(prompt, status_placeholder))

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response_text})
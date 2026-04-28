import os
import json
import time
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from enum import StrEnum

from graph import build_graph
from models import PromptRequest

graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    
    has_azure = os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")
    has_openai = os.getenv("OPENAI_API_KEY")

    if not has_azure and not has_openai:
        raise EnvironmentError(
            "No LLM credentials found. Set either:\n"
            "  - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT  (Azure OpenAI)\n"
            "  - OPENAI_API_KEY  (native OpenAI)"
        )
    
    global graph
    graph = build_graph()
    yield

app = FastAPI(lifespan=lifespan)

# Status messages shown while a node is running (on_chat_model_start)
NODE_STATUS_MESSAGES = {
    "orchestrate": "Routing question...",
    "planning": "Creating a plan...",
    "generate_joke": "Generating a joke...",
    "answer_question": "Formulating answer...",
}

NODE_OUTPUT_LABELS = {
    "planning": "📋 Plan",
    "generate_joke": "😄 Joke",
}

# Trace labels emitted when a node completes (on_chain_end)
NODE_TRACE_LABELS = {
    "orchestrate": "Route decided",
    "planning": "Plan created",
    "generate_joke": "Joke generated",
    "answer_question": "Answer generated",
}

class GraphEvent(StrEnum):
    ON_CHAT_MODEL_START = "on_chat_model_start"
    ON_CHAT_MODEL_STREAM = "on_chat_model_stream"
    ON_CHAT_MODEL_END = "on_chat_model_end"
    ON_CHAIN_START = "on_chain_start"
    ON_CHAIN_END = "on_chain_end"

def sse_event(event_type: str, data) -> str:
    """Format a Server-Sent Event with a typed event field."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

async def llm_chat_generator(prompt: str) -> AsyncIterator[str]:
    """Streams SSE events using LangGraph.

    Event types emitted:
      status  — short human-readable label while a node is running
      trace   — JSON payload emitted when a node completes (includes duration_ms)
      answer  — token-by-token chunks from the final answer node
    """
    announced_nodes: set[str] = set()
    node_start_times: dict[str, float] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.perf_counter()

    async for event in graph.astream_events(
        {"question": prompt},
        version="v2"
    ):
        node = event.get("metadata", {}).get("langgraph_node")
        event_name = event["event"]
        run_id = event.get("run_id")

        # Record start time when a tracked node begins
        if event_name == GraphEvent.ON_CHAIN_START and event.get("name") in NODE_TRACE_LABELS:
            node_start_times[run_id] = time.monotonic()

        # Yield status updates when a new node starts
        # Payload is a dict so the frontend can identify the node and label
        elif event_name == GraphEvent.ON_CHAT_MODEL_START and node in NODE_STATUS_MESSAGES:
            if node not in announced_nodes:
                announced_nodes.add(node)
                yield sse_event("status", {"node": node, "label": NODE_STATUS_MESSAGES[node]})

        elif event_name == GraphEvent.ON_CHAIN_END and event.get("name") in NODE_TRACE_LABELS:
            node_name = event["name"]
            duration_ms = round((time.monotonic() - node_start_times.pop(run_id, time.monotonic())) * 1000)
            yield sse_event("trace", {
                "node": node_name,
                "label": NODE_TRACE_LABELS[node_name],
                "duration_ms": duration_ms,
            })
            # Routing decision is also emitted alongside the trace for the orchestrate node
            if node_name == "orchestrate":
                output = event["data"].get("output", {})
                planning_required = output.get("planning_required", False)
                yield sse_event("routing", {"planning_required": planning_required})

        elif event_name == GraphEvent.ON_CHAT_MODEL_STREAM:
            content = event["data"]["chunk"].content
            if not content:
                continue

            # Stream intermediate node tokens — payload is a dict for clean frontend parsing
            if node in NODE_OUTPUT_LABELS:
                yield sse_event("node_output", {"node": node, "token": content})

            # Stream the final answer token by token
            elif node == "answer_question":
                yield sse_event("answer", content)

        elif event["event"] == GraphEvent.ON_CHAT_MODEL_END:
            usage = event["data"].get("output", {}).usage_metadata or {}
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)

    elapsed = round(time.perf_counter() - start_time, 2)
    
    yield sse_event("done", {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "elapsed_seconds": elapsed,
    })


@app.post("/stream")
async def stream_chat(request: PromptRequest):
    return StreamingResponse(llm_chat_generator(request.prompt), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
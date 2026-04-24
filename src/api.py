import os
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from graph import build_graph
from models import PromptRequest

graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")
    
    global graph
    graph = build_graph()
    yield

app = FastAPI(lifespan=lifespan)

async def llm_chat_generator(prompt: str) -> AsyncIterator[str]:
    """Streams LLM response using LangGraph."""
    async for event in graph.astream_events(
        {"question": prompt},
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content


@app.post("/stream")
async def stream_chat(request: PromptRequest):
    return StreamingResponse(llm_chat_generator(request.prompt), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
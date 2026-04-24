from pydantic import BaseModel


class PromptRequest(BaseModel):
    """Request model for incoming prompts."""
    prompt: str

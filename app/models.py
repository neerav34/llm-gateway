from typing import List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request body.

    Clients talk to the gateway exactly as they would to OpenAI/Groq,
    so existing SDKs work unchanged by just pointing base_url at us.
    """

    messages: List[ChatMessage]
    model: Optional[str] = None  # None -> gateway picks the provider default
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False  # streaming passthrough lands in Day 5

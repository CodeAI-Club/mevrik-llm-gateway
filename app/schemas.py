"""OpenAI-compatible request schemas.

extra='allow' ensures any additional fields (temperature, top_p, tools, etc.)
are passed through transparently to the vLLM backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any], None] = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False

    model_config = ConfigDict(extra="allow")


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stream: bool = False

    model_config = ConfigDict(extra="allow")


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

    model_config = ConfigDict(extra="allow")


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]

    model_config = ConfigDict(extra="allow")


class ScoreRequest(BaseModel):
    model: str

    model_config = ConfigDict(extra="allow")

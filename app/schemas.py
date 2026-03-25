"""OpenAI-compatible request schemas.

extra='allow' ensures any additional fields (temperature, top_p, tools, etc.)
are passed through transparently to the vLLM backend.

Each schema validates the minimum required fields and lets everything
else flow through — this is intentional for forward-compatibility with
new vLLM parameters without gateway code changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


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
    """OpenAI-compatible embedding request.

    vLLM accepts:
      - input: str | list[str] | list[int] | list[list[int]]
      - encoding_format: "float" | "base64" (optional)
    """

    model: str
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class RerankRequest(BaseModel):
    """vLLM rerank endpoint request.

    vLLM (v0.6+) expects:
      - model: str
      - query: str
      - documents: list[str]
      - top_n: int (optional, defaults to len(documents))
    """

    model: str
    query: str
    documents: Union[List[str], List[Dict[str, Any]]]
    top_n: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class ScoreRequest(BaseModel):
    """vLLM cross-encoder scoring request.

    vLLM expects:
      - model: str
      - text_1: str | list[str]
      - text_2: str | list[str]
    OR (depending on version):
      - input: list[dict] with "text_1"/"text_2" pairs
    """

    model: str
    # Accept both formats — extra="allow" passes any additional fields through
    text_1: Optional[Union[str, List[str]]] = None
    text_2: Optional[Union[str, List[str]]] = None

    model_config = ConfigDict(extra="allow")

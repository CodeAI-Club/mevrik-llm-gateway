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


# ---------------------------------------------------------------------------
# Rerank schemas — support vLLM, Cohere, and Jina formats
# ---------------------------------------------------------------------------


class RerankRequest(BaseModel):
    """Unified rerank request supporting multiple API conventions.

    vLLM (v0.6+) expects:
      - model: str
      - query: str
      - documents: list[str] | list[dict]
      - top_n: int (optional, defaults to len(documents))

    Cohere API sends:
      - model: str
      - query: str
      - documents: list[str] | list[dict with "text" key]
      - top_n: int
      - return_documents: bool (optional)
      - max_chunks_per_doc: int (optional)

    Jina API sends:
      - model: str
      - query: str
      - documents: list[str] | list[dict]
      - top_n: int

    The gateway normalizes all formats before proxying to vLLM.
    """

    model: str
    query: str
    documents: Union[List[str], List[Dict[str, Any]]]
    top_n: Optional[int] = None

    # Cohere-compat fields (accepted but may not be forwarded to vLLM)
    return_documents: Optional[bool] = None
    max_chunks_per_doc: Optional[int] = None

    model_config = ConfigDict(extra="allow")

    def normalize_documents(self) -> List[str]:
        """Extract plain-text list from documents, regardless of input format.

        Handles:
          - list[str] → returned as-is
          - list[dict] with "text" key → extracts text values
          - list[dict] with "content" key → extracts content (multimodal)
        """
        if not self.documents:
            return []

        first = self.documents[0]
        if isinstance(first, str):
            return self.documents  # type: ignore[return-value]

        # Dict documents — extract text
        result: List[str] = []
        for doc in self.documents:
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or str(doc)
                if isinstance(text, list):
                    # Multimodal content list — join text parts
                    parts = [
                        item.get("text", "")
                        for item in text
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    result.append(" ".join(parts) if parts else str(text))
                else:
                    result.append(str(text))
            else:
                result.append(str(doc))
        return result

    def to_vllm_payload(self) -> dict:
        """Build the payload dict that vLLM's /rerank endpoint expects."""
        payload: dict = {
            "model": self.model,
            "query": self.query,
            "documents": self.documents,  # pass through as-is (vLLM handles both)
        }
        if self.top_n is not None:
            payload["top_n"] = self.top_n

        # Forward any extra fields the caller sent (future-proofing)
        extras = self.model_dump(
            exclude={
                "model",
                "query",
                "documents",
                "top_n",
                "return_documents",
                "max_chunks_per_doc",
            },
            exclude_none=True,
        )
        payload.update(extras)
        return payload


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


# ---------------------------------------------------------------------------
# Rerank response schemas (for documentation / normalization)
# ---------------------------------------------------------------------------


class RerankResultItem(BaseModel):
    """Single rerank result."""

    index: int
    relevance_score: float
    document: Optional[Dict[str, Any]] = None  # populated when return_documents=True

    model_config = ConfigDict(extra="allow")


class RerankResponse(BaseModel):
    """Normalized rerank response compatible with Cohere/Jina/vLLM formats."""

    id: Optional[str] = None
    results: List[RerankResultItem]
    meta: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")

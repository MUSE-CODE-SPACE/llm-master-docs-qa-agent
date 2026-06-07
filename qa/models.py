"""Pydantic models — the shape of every request/response. / 요청·응답의 형태."""
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    content: str
    source: str
    chunk_index: int
    distance: float


class Citation(BaseModel):
    source: str
    chunk_index: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    confidence: str = "medium"
    retrieved_count: int = 0

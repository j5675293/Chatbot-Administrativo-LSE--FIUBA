"""
Schemas Pydantic para la API.

Autor: Juan Ruiz Otondo - CEIA FIUBA
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RetrievalModeEnum(str, Enum):
    rag = "rag"
    graph = "graph"
    hybrid = "hybrid"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    mode: RetrievalModeEnum = RetrievalModeEnum.hybrid
    program_filter: Optional[str] = None
    session_id: Optional[str] = None


class SourceCitation(BaseModel):
    document_name: str
    page_numbers: list[int] = []
    section_title: str = ""
    text_snippet: str = ""
    score: float = 0.0
    source_type: str = "rag"


class ChatResponse(BaseModel):
    answer: str
    formatted_answer: str = ""
    sources: list[SourceCitation] = []
    confidence: float
    method: str
    warnings: list[str] = []
    fallback_contacts: list[str] = []
    processing_time_ms: float = 0.0


class ComparisonRequest(BaseModel):
    question: str = Field(..., min_length=3)
    program_filter: Optional[str] = None


class ComparisonResponse(BaseModel):
    rag_answer: ChatResponse
    graph_answer: ChatResponse
    hybrid_answer: ChatResponse


class DocumentInfo(BaseModel):
    filename: str
    document_type: str
    program_codes: list[str]
    topics: list[str]
    chunks_count: int = 0


class GraphStats(BaseModel):
    nodes: int
    edges: int
    density: float = 0.0
    node_types: dict = {}
    edge_types: dict = {}


class FeedbackRequest(BaseModel):
    session_id: str = ""
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    method: str = "hybrid"
    confidence: float = 0.0
    is_correct: Optional[bool] = None
    is_complete: Optional[bool] = None
    user_comment: str = ""
    expected_answer: str = ""


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str = "ok"


class FeedbackStatsResponse(BaseModel):
    total_entries: int
    avg_rating: float
    correct_rate: float
    complete_rate: float
    by_method: dict = {}
    by_rating: dict = {}
    low_rated_questions: list = []
    improvement_suggestions: list[str] = []


class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    documents_loaded: int
    index_size: int
    graph_nodes: int

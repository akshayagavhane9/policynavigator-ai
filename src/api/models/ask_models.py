from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class AskRequest(BaseModel):
    question: str
    answer_style: str = "Strict policy quote"
    k: int = 5
    rewrite_query: bool = True


class Citation(BaseModel):
    source: str
    chunk_id: str
    rank: int
    similarity: Optional[float] = None
    text: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    confidence_label: str
    confidence_score: float
    used_query: str
    latency_ms: int
    hallucination_flag: Optional[bool] = False
    hallucination_risk: Optional[str] = "unknown"


class ScenarioRequest(BaseModel):
    scenario: str


class ScenarioResponse(BaseModel):
    analysis: str


class QuizItem(BaseModel):
    question: str
    options: List[str]
    answer: str
    explanation: str


class QuizRequest(BaseModel):
    num_questions: int = 3


class QuizResponse(BaseModel):
    quiz: List[QuizItem]

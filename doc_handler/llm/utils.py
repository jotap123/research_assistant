from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ConversationState(BaseModel):
    """Encapsulates the conversation state."""
    memory: List[Any] = Field(default_factory=list)
    context: Optional[str] = ""
    action_plan: Optional[Dict] = None
    summary: Optional[str] = None


class AgentConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_search_results: int = 3
    memory_summarizer_threshold: int = 8


class RetrievalAction(str, Enum):
    SEARCH = "search"
    PDF = "pdf"
    ERROR = "error"
    NONE = "none"


class ActionPlan(BaseModel):
    action: RetrievalAction
    reasoning: str
    search_query: Optional[str]
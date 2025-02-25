from enum import Enum
from typing import Optional
from pydantic import BaseModel
from langgraph.graph import MessagesState


class AgentConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_search_results: int = 3
    memory_summarizer_threshold: int = 8


class RetrievalAction(str, Enum):
    SEARCH = "search"
    ERROR = "error"
    NONE = "none"


class State(MessagesState):
    summary: str = None
    context: Optional[str] = ""
    curated_query: Optional[str] = ""
    action_plan: RetrievalAction

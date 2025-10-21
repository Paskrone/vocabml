from typing import List, Optional
from pydantic import BaseModel, Field

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=512)

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class RecommendRequest(BaseModel):
    query: str
    lang: str = Field("deu", pattern="^(deu|eng)$")
    top_k: int = Field(20, ge=1, le=200)
    min_score: Optional[float] = Field(None, ge=-1.0, le=1.0)

class RecommendItem(BaseModel):
    id: int
    lang: str
    text: str
    score: float

class RecommendResponse(BaseModel):
    items: List[RecommendItem]

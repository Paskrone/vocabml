from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import EmbedRequest, EmbedResponse, RecommendRequest, RecommendResponse, RecommendItem
from .encoder import embed_texts
from .service import recommend

app = FastAPI(title="VocabML FastAPI ML-Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/embed", response_model=EmbedResponse)
def post_embed(req: EmbedRequest):
    if any((t is None or str(t).strip() == "") for t in req.texts):
        raise HTTPException(status_code=400, detail="Empty text in batch")
    embs = embed_texts(req.texts)
    return {"embeddings": embs.tolist()}

@app.post("/recommend", response_model=RecommendResponse)
def post_recommend(req: RecommendRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")
    qvec = embed_texts([req.query])[0]
    items = recommend(qvec, req.lang, req.top_k, req.min_score)
    return {"items": [RecommendItem(**it) for it in items]}

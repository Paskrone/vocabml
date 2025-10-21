import os
import threading
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

_MODEL = None
_LOCK = threading.Lock()

def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                model_id = os.getenv("MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
                _MODEL = SentenceTransformer(model_id)
    return _MODEL

def embed_texts(texts: List[str]) -> np.ndarray:
    m = get_model()
    emb = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256)
    return emb.astype(np.float32)

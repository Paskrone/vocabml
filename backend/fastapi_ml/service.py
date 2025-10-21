from typing import List, Optional
import numpy as np
from .db import get_conn, put_conn

def to_vector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"

def recommend(qvec: np.ndarray, lang: str, top_k: int, min_score: Optional[float]) -> List[dict]:
    qlit = to_vector_literal(qvec)
    sql = f"""
        WITH q AS (
          SELECT {('%s')}::vector(384) AS v
        )
        SELECT s.id, s.lang, s.text,
               1 - (e.emb <=> (SELECT v FROM q)) AS score
        FROM sentence_embeddings e
        JOIN sentences s ON s.id = e.sentence_id
        WHERE s.lang = %s
        ORDER BY e.emb <=> (SELECT v FROM q)
        LIMIT %s
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (qlit, lang, top_k))
            rows = cur.fetchall()
    finally:
        put_conn(conn)

    items = []
    for rid, rlang, rtext, rscore in rows:
        if min_score is None or rscore >= min_score:
            items.append({"id": rid, "lang": rlang, "text": rtext, "score": float(rscore)})
    return items

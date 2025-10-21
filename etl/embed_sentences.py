# etl/embed_sentences.py
import argparse, numpy as np, psycopg2
from sentence_transformers import SentenceTransformer

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--lang-filter", nargs="*", default=["deu","eng"], dest="lang_filter")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    print("📦 Lade Modell:", args.model)
    m = SentenceTransformer(args.model)

    print("🔌 Verbinde zu Postgres …")
    con = psycopg2.connect(args.dsn); con.autocommit = True
    cur = con.cursor()

    # Hole IDs+Texte ohne vorhandenes Embedding
    q = """
      SELECT s.id, s.text
      FROM sentences s
      LEFT JOIN sentence_embeddings e ON e.sentence_id = s.id
      WHERE e.sentence_id IS NULL
        AND s.lang = ANY(%s)
      ORDER BY s.id
      {}
    """.format(f"LIMIT {args.limit}" if args.limit else "")
    cur.execute(q, (args.lang_filter,))
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    print(f"🧮 zu embeddende Sätze: {len(ids):,}")

    # Encode + UPSERT in Batches
    done = 0
    for chunk_ids, chunk_texts in zip(batched(ids, args.batch_size), batched(texts, args.batch_size)):
        embs = m.encode(chunk_texts, batch_size=args.batch_size, normalize_embeddings=True)
        payload = [(int(i), np.asarray(e, dtype=np.float32).tolist()) for i, e in zip(chunk_ids, embs)]
        cur.executemany(
            "INSERT INTO sentence_embeddings (sentence_id, emb) VALUES (%s, %s) "
            "ON CONFLICT (sentence_id) DO UPDATE SET emb = EXCLUDED.emb",
            payload
        )
        done += len(chunk_ids)
        print(f"  ✅ geschrieben: {done:,}/{len(ids):,}")

    cur.close(); con.close()
    print("🎉 Fertig.")

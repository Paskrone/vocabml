import argparse, pandas as pd
from sqlalchemy import create_engine, text

DDL_SENTENCES = """
CREATE TABLE IF NOT EXISTS sentences (
  id   BIGINT PRIMARY KEY,
  lang TEXT NOT NULL,
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sentences_lang ON sentences(lang);
"""

DDL_EXAMPLES = """
CREATE TABLE IF NOT EXISTS examples_deu_eng (
  src_id  BIGINT NOT NULL,
  tgt_id  BIGINT NOT NULL,
  src_text TEXT NOT NULL,
  tgt_text TEXT NOT NULL,
  PRIMARY KEY (src_id, tgt_id)
);
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True, help="z.B. postgresql+psycopg2://user:pass@host:5432/db")
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--mode", choices=["append","replace"], default="append")
    ap.add_argument("--chunksize", type=int, default=50000)
    args = ap.parse_args()

    eng = create_engine(args.dsn)
    with eng.begin() as conn:
        if args.mode == "replace":
            conn.execute(text("DROP TABLE IF EXISTS examples_deu_eng"))
            conn.execute(text("DROP TABLE IF EXISTS sentences"))
        conn.execute(text(DDL_SENTENCES))
        conn.execute(text(DDL_EXAMPLES))

    s = pd.read_parquet(args.sentences)
    e = pd.read_parquet(args.examples)

    # dtypes
    s["id"] = s["id"].astype("int64")
    e["src_id"] = e["src_id"].astype("int64")
    e["tgt_id"] = e["tgt_id"].astype("int64")

    s.to_sql("sentences", eng, if_exists="append", index=False, method="multi", chunksize=args.chunksize)
    e.to_sql("examples_deu_eng", eng, if_exists="append", index=False, method="multi", chunksize=args.chunksize)

    print("✅ Load complete.")

if __name__ == "__main__":
    main()

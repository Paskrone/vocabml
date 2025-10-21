import argparse, pandas as pd
from pathlib import Path

def norm_text(s: pd.Series) -> pd.Series:
    return (s.fillna("")
              .astype(str)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_len", type=int, default=3)      # minimale Zeichenlänge
    ap.add_argument("--max_len", type=int, default=240)    # maximale Zeichenlänge
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    s = pd.read_parquet(args.sentences)
    e = pd.read_parquet(args.examples)

    # Normalize
    s["text"] = norm_text(s["text"])
    e["src_text"] = norm_text(e["src_text"])
    e["tgt_text"] = norm_text(e["tgt_text"])

    # Sentence-level dedupe
    s = s.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Pair-level filters
    e = e.drop_duplicates(subset=["src_id","tgt_id"])
    # identische Texte raus
    e = e[e["src_text"].str.lower() != e["tgt_text"].str.lower()]
    # Längenfilter
    e = e[e["src_text"].str.len().between(args.min_len, args.max_len)]
    e = e[e["tgt_text"].str.len().between(args.min_len, args.max_len)]

    # Option: super-kurze Tokens (nur 1 Wort) raus
    e = e[e["src_text"].str.count(r"\w+") >= 2]
    e = e[e["tgt_text"].str.count(r"\w+") >= 2]

    s.to_parquet(out / "sentences_clean.parquet", index=False)
    e.to_parquet(out / "examples_deu_eng_clean.parquet", index=False)

    print("✅ Cleaned:",
          f"sentences={len(s):,}",
          f"examples={len(e):,}",
          f"→ {out/'sentences_clean.parquet'}",
          f"→ {out/'examples_deu_eng_clean.parquet'}", sep="\n")

if __name__ == "__main__":
    main()

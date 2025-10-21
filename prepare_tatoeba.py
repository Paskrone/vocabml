import argparse
import pandas as pd
from pathlib import Path

def sniff_sep(sample_path: Path) -> str:
    with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    return "," if (first.count(",") >= first.count("\t")) else "\t"

def load_sentences(path: Path, lang_expected: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        names=["id","lang","text"],
        header=None,
        quoting=3,
        dtype={"id":"int64","lang":"string","text":"string"},
        on_bad_lines="skip",
        encoding="utf-8"
    )
    if lang_expected:
        df = df[df["lang"] == lang_expected]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deu", required=True, type=Path, help="Pfad zu deu_sentences_small.tsv")
    ap.add_argument("--eng", required=True, type=Path, help="Pfad zu eng_sentences_small.tsv")
    ap.add_argument("--links", required=True, type=Path, help="Pfad zu links_small.tsv (csv oder tsv)")
    ap.add_argument("--outdir", default=Path("."), type=Path, help="Output-Ordner für Parquet")
    ap.add_argument("--chunksize", default=1_000_000, type=int, help="Zeilen pro Links-Chunk")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("📥 Lade Sätze (DEU)…")
    deu = load_sentences(args.deu, "deu")
    deu_ids = set(deu["id"].astype("int64").tolist())
    print(f"  DEU: {len(deu)} Sätze, {len(deu_ids)} IDs")

    print("📥 Lade Sätze (ENG)…")
    eng = load_sentences(args.eng, "eng")
    eng_ids = set(eng["id"].astype("int64").tolist())
    print(f"  ENG: {len(eng)} Sätze, {len(eng_ids)} IDs")

    print("🔎 Erkenne Separator der Links-Datei…")
    sep_links = sniff_sep(args.links)
    print(f"  Links-Separator: {repr(sep_links)}")

    print("🪄 Filtere Links (deu → eng) in Chunks…")
    pairs = []
    reader = pd.read_csv(
        args.links,
        sep=sep_links,
        names=["a","b"],
        header=None,
        dtype={"a":"int64","b":"int64"},
        chunksize=args.chunksize,
        encoding="utf-8",
        on_bad_lines="skip",
        engine="python",
    )
    total_kept = 0
    for i, chunk in enumerate(reader, 1):
        mask = chunk["a"].isin(deu_ids) & chunk["b"].isin(eng_ids)
        keep = chunk.loc[mask, ["a","b"]]
        total_kept += len(keep)
        pairs.append(keep)
        if i % 10 == 0:
            print(f"  …Chunk {i}: bisher behalten {total_kept:,} Paare")

    if pairs:
        links_deu_eng = pd.concat(pairs, ignore_index=True).drop_duplicates()
    else:
        links_deu_eng = pd.DataFrame(columns=["a","b"], dtype="int64")
    print(f"✅ Gesamt behalten: {len(links_deu_eng):,} Paare")

    print("🔗 Joine Texte…")
    examples = (
        links_deu_eng
        .merge(deu, left_on="a", right_on="id")
        .merge(eng, left_on="b", right_on="id", suffixes=("_deu","_eng"))
        .rename(columns={
            "a":"src_id","b":"tgt_id",
            "text_deu":"src_text","text_eng":"tgt_text"
        })[["src_id","tgt_id","src_text","tgt_text"]]
    )

    print("💾 Speichere Parquet…")
    sentences = pd.concat([deu[["id","lang","text"]], eng[["id","lang","text"]]], ignore_index=True)
    sentences.to_parquet(args.outdir / "sentences.parquet", index=False)
    examples.to_parquet(args.outdir / "examples_deu_eng.parquet", index=False)

    print("🎉 Fertig!")
    print(f"  → {args.outdir / 'sentences.parquet'}")
    print(f"  → {args.outdir / 'examples_deu_eng.parquet'}")
    print(f"  Sätze: {len(sentences):,} | Beispiele: {len(examples):,}")

if __name__ == "__main__":
    main()

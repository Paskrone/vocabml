import sys
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def interpret_similarity(score: float) -> str:
    if score >= 0.90: return "🟢 Fast identisch"
    if score >= 0.70: return "🟩 Starke Nähe"
    if score >= 0.50: return "🟨 Mittlere Nähe / thematisch verwandt"
    if score >= 0.30: return "🟧 Schwach verwandt"
    return "🔴 Keine Verbindung"

def sim(a: str, b: str) -> float:
    ea = model.encode(a, normalize_embeddings=True)
    eb = model.encode(b, normalize_embeddings=True)
    return float(util.cos_sim(ea, eb).item())

def main():
    if len(sys.argv) >= 3:
        s1 = " ".join(sys.argv[1:-1]).strip('"')
        s2 = sys.argv[-1].strip('"')
    else:
        print("Eingabe leer – wechsle in den interaktiven Modus.")
        s1 = input("Satz 1: ").strip()
        s2 = input("Satz 2: ").strip()

    score = sim(s1, s2)
    print(f"\n🧩 Satz 1: {s1}\n🧩 Satz 2: {s2}")
    print(f"\n➡️ Ähnlichkeitswert: {score:.3f}")
    print(interpret_similarity(score))

if __name__ == "__main__":
    main()

from sentence_transformers import SentenceTransformer, util

# Modell laden
model = SentenceTransformer("all-MiniLM-L6-v2")

def interpret_similarity(score: float) -> str:
    """Gibt textliche Einschätzung basierend auf dem Score zurück."""
    if score >= 0.9:
        return "🟢 Fast identisch (z. B. Synonyme oder gleichbedeutende Sätze)"
    elif score >= 0.7:
        return "🟩 Starke inhaltliche Nähe (ähnliche Bedeutung, anderes Wording)"
    elif score >= 0.4:
        return "🟨 Thematisch ähnlich, aber in Kontext oder Details unterschiedlich"
    elif score >= 0.1:
        return "🟧 Schwach verwandt oder entfernt ähnliche Themen"
    else:
        return "🔴 Keine inhaltliche Verbindung"

def compare_sentences(sentence1: str, sentence2: str):
    """Berechnet Ähnlichkeit und druckt Ergebnis."""
    emb1 = model.encode(sentence1, normalize_embeddings=True)
    emb2 = model.encode(sentence2, normalize_embeddings=True)

    score = util.cos_sim(emb1, emb2).item()
    interpretation = interpret_similarity(score)

    print(f"\n🧩 Satz 1: {sentence1}")
    print(f"🧩 Satz 2: {sentence2}")
    print(f"\n➡️ Ähnlichkeitswert: {score:.3f}")
    print(f"{interpretation}\n")

# Beispielaufrufe
pairs = [
    ("Ich trinke Wasser", "Ich trinke H2O"),
    ("Ich esse ein Brötchen", "Ich frühstücke"),
    ("Ich fahre zur Arbeit", "Ich gehe spazieren"),
    ("Die Sonne scheint", "Ich programmiere eine App")
]

for s1, s2 in pairs:
    compare_sentences(s1, s2)

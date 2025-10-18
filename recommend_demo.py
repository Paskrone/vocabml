from sentence_transformers import SentenceTransformer, util
import numpy as np

# 1) Modell: probiere für Deutsch auch: paraphrase-multilingual-MiniLM-L12-v2
model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # oder "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# 2) Thema/Kontext aus mehreren Seeds bauen
seeds = [
    "Party", "jemanden kennenlernen", "attraktiv", "nach der Nummer fragen",
    "Kontakt austauschen", "WhatsApp", "schreiben", "Date ausmachen"
]
seed_emb = model.encode(seeds, normalize_embeddings=True)
topic_vec = seed_emb.mean(axis=0)

# 3) Kandidaten: angereichert (Wort + Erklärung + Beispiel)
candidates = [
    {
        "id": "ask_whatsapp",
        "text": "WhatsApp fragen. Bedeutung: nach dem Messenger fragen, um Kontakt zu halten. "
                "Beispiel: Hast du WhatsApp? Dann schreibe ich dir später."
    },
    {
        "id": "ask_instagram",
        "text": "Instagram austauschen. Bedeutung: Nutzername erfragen, um in Kontakt zu bleiben. "
                "Beispiel: Bist du auf Instagram? Wie heißt du dort?"
    },
    {
        "id": "ask_number",
        "text": "Nach der Telefonnummer fragen. Bedeutung: Direkt nach Nummer fragen, um später zu schreiben. "
                "Beispiel: Kann ich deine Nummer haben?"
    },
    {
        "id": "ÖPNV_ticket",
        "text": "Fahrkarte kaufen. Bedeutung: Ticket für Bus/Bahn kaufen. "
                "Beispiel: Wo kann ich eine Fahrkarte kaufen?"
    },
]

cand_texts = [c["text"] for c in candidates]
cand_emb = model.encode(cand_texts, normalize_embeddings=True)

# 4) Scores + Ranking
scores = util.cos_sim(topic_vec, cand_emb).cpu().numpy().ravel()
order = scores.argsort()[::-1]
print(f"Modell: {model_name}\n")
for i in order:
    print(f"{candidates[i]['id']:>15} | score={scores[i]:.3f} | {candidates[i]['text'][:70]}...")

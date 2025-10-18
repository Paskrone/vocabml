import argparse
from sentence_transformers import SentenceTransformer, util
import numpy as np


# --- 1) Modelle registrieren ---
MODEL_IDS = {
"MiniLM-en": "all-MiniLM-L6-v2",
"MiniLM-multi": "paraphrase-multilingual-MiniLM-L12-v2",
"distiluse-multi": "distiluse-base-multilingual-cased-v2",
"LaBSE": "sentence-transformers/LaBSE",
}


# --- 2) Testdaten ---
DEFAULT_CORPUS = [
# ÖPNV & Verkehr
    "Bus", "Haltestelle", "Fahrkarte kaufen", "Verspätung", "U-Bahn", "S-Bahn", "Straßenbahn", "Taxi",
    "Fahrrad", "E-Scooter", "Parkplatz", "Stau", "Ampel", "Kreuzung", "Fußgängerzone", "Fahrplan",
    "Abfahrt", "Ankunft", "Umsteigen", "Endstation", "Waggon", "Schaffner", "Fahrkartenautomat",
    
    # Arbeit & Beruf
    "Ich gehe heute zur Arbeit", "Büro", "Kollege", "Meeting", "Präsentation", "Deadline", "Projekt",
    "Team", "Chef", "Gehalt", "Urlaub", "Krankheit", "Überstunden", "Pause", "Kantine", "Kaffeemaschine",
    "Computer", "Laptop", "Internet", "E-Mail", "Anruf", "Termin", "Besprechung", "Bericht",
    
    # Kommunikation & Technologie
    "WhatsApp austauschen", "Telefonnummer", "Instagram", "Facebook", "Snapchat", "TikTok", "YouTube",
    "Chat", "Nachricht", "Anruf", "Videoanruf", "Sprachnachricht", "Emoji", "Hashtag", "Like",
    "Follower", "Post", "Story", "Live-Stream", "Podcast", "Blog", "Website", "App", "Software",
    
    # Essen & Trinken
    "Restaurant", "Café", "Bar", "Bäckerei", "Supermarkt", "Lebensmittel", "Einkaufen", "Kochen",
    "Backen", "Frühstück", "Mittagessen", "Abendessen", "Snack", "Getränk", "Kaffee", "Tee", "Bier",
    "Wein", "Wasser", "Saft", "Pizza", "Pasta", "Salat", "Fleisch", "Fisch", "Vegetarisch", "Vegan",
    
    # Gesundheit & Körper
    "Arzt", "Krankenhaus", "Apotheke", "Medikament", "Schmerzen", "Kopfschmerzen", "Fieber", "Husten",
    "Schnupfen", "Allergie", "Sport", "Fitness", "Laufen", "Schwimmen", "Radfahren", "Gym", "Yoga",
    "Pilates", "Tanzen", "Wandern", "Klettern", "Schlafen", "Erholung", "Entspannung", "Stress",
    
    # Shopping & Mode
    "Einkaufen", "Geschäft", "Laden", "Kaufhaus", "Online-Shop", "Kleidung", "Schuhe", "Jacke",
    "Hose", "Hemd", "Rock", "Kleid", "Pullover", "T-Shirt", "Jeans", "Anzug", "Krawatte", "Schmuck",
    "Uhr", "Handtasche", "Rucksack", "Sonnenbrille", "Kosmetik", "Parfüm", "Drogerie", "Preis",
    
    # Freizeit & Unterhaltung
    "Kino", "Film", "Theater", "Konzert", "Musik", "Buch", "Zeitung", "Zeitschrift", "Zeitschrift",
    "Spiel", "Puzzle", "Brettspiel", "Videospiel", "Spielen", "Hobby", "Sammeln", "Malen", "Zeichnen",
    "Fotografieren", "Lesen", "Schreiben", "Musik hören", "Party", "Feier", "Geburtstag", "Hochzeit",
    
    # Familie & Beziehungen
    "Familie", "Eltern", "Mutter", "Vater", "Schwester", "Bruder", "Großeltern", "Onkel", "Tante",
    "Cousin", "Cousine", "Freund", "Freundin", "Partner", "Ehemann", "Ehefrau", "Kind", "Baby",
    "Hund", "Katze", "Haustier", "Liebe", "Kuss", "Umarmung", "Streit", "Versöhnung", "Hochzeit",
    
    # Wohnen & Haushalt
    "Wohnung", "Haus", "Zimmer", "Küche", "Badezimmer", "Schlafzimmer", "Wohnzimmer", "Balkon",
    "Garten", "Möbel", "Sofa", "Tisch", "Stuhl", "Bett", "Schrank", "Regal", "Lampen", "Heizung",
    "Klimaanlage", "Waschmaschine", "Trockner", "Geschirrspüler", "Kühlschrank", "Herd", "Ofen",
    
    # Wetter & Jahreszeiten
    "Wetter", "Sonne", "Regen", "Schnee", "Wolken", "Wind", "Sturm", "Gewitter", "Nebel", "Hitze",
    "Kälte", "Wärme", "Frühling", "Sommer", "Herbst", "Winter", "Jahreszeit", "Monat", "Woche",
    "Tag", "Nacht", "Morgen", "Mittag", "Abend", "Zeit", "Uhr", "Datum", "Kalender",
    
    # Bildung & Lernen
    "Schule", "Universität", "Studium", "Lernen", "Prüfung", "Test", "Hausaufgaben", "Buch",
    "Stift", "Papier", "Computer", "Internet", "Forschung", "Wissenschaft", "Mathematik", "Deutsch",
    "Englisch", "Französisch", "Spanisch", "Geschichte", "Geographie", "Biologie", "Chemie", "Physik",
    
    # Gefühle & Emotionen
    "glücklich", "traurig", "wütend", "ängstlich", "nervös", "entspannt", "müde", "energisch",
    "aufgeregt", "gelangweilt", "interessiert", "überrascht", "verwirrt", "stolz", "schüchtern",
    "mutig", "stark", "schwach", "gesund", "krank", "sicher", "unsicher", "zufrieden", "unzufrieden",
    
    # Reisen & Urlaub
    "Reisen", "Urlaub", "Ferien", "Hotel", "Pension", "Hostel", "Camping", "Strand", "Berg", "See",
    "Fluss", "Wald", "Stadt", "Land", "Ausland", "Inland", "Flugzeug", "Zug", "Auto", "Schiff",
    "Koffer", "Rucksack", "Kamera", "Souvenir", "Karte", "Reiseführer", "Geld", "Währung",
    
    # Zeit & Zahlen
    "Zeit", "Stunde", "Minute", "Sekunde", "Jahr", "Monat", "Woche", "Tag", "Nacht", "Morgen",
    "Mittag", "Nachmittag", "Abend", "gestern", "heute", "morgen", "früh", "spät", "schnell",
    "langsam", "oft", "selten", "immer", "nie", "manchmal", "meistens", "manchmal"
]
DEFAULT_QUERIES = [
"Fahrkarte",
"Lass uns mit dem Bus fahren",
"Wo finde ich den Fahrplan?",
]


def encode_norm(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)




def run_once(model_name: str, model_id: str, queries: list[str], corpus: list[str], topk: int = 5):
    print(f"\n===== {model_name} ({model_id}) =====")
    m = SentenceTransformer(model_id)
    E = encode_norm(m, corpus)  # (N, D)

    for q in queries:
        qv = encode_norm(m, [q])  # (1, D)
        sims = util.cos_sim(qv, E)[0].cpu().numpy()  # (N,)
        order = np.argsort(-sims)[:topk]
        print(f"\nQuery: {q}")
        for i in order:
            print(f" {corpus[i]:20s} | {sims[i]:.3f}")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODEL_IDS.keys()), help="Welche Modelle testen")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--query", action="append", help="Zusätzliche Query (mehrfach möglich)")
    args = ap.parse_args()

    queries = DEFAULT_QUERIES.copy()
    if args.query:
        queries = args.query

    corpus = DEFAULT_CORPUS

    for name in args.models:
        run_once(name, MODEL_IDS[name], queries, corpus, topk=args.topk)




if __name__ == "__main__":
    main()
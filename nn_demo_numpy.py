import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")  # funktioniert für Wörter & Sätze

# 1) Dein Korpus (Wörter ODER Sätze). Für Wörter: gerne anreichern (Definition/Beispiel)
corpus = [
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

# 2) Embeddings vorrechnen & normalisieren (gut für Cosine)
E = model.encode(corpus, normalize_embeddings=True)   # shape: (N, 384)

def topk_similar(query_text: str, k: int = 5):
    q = model.encode([query_text], normalize_embeddings=True)  # (1,384)
    # Cosine mit util (oder E @ q.T weil normalisiert)
    sims = util.cos_sim(q, E).cpu().numpy().ravel()            # (N,)
    idx = np.argsort(-sims)[:k]
    return [(corpus[i], float(sims[i])) for i in idx]

print("=== Wort → ähnliche Wörter ===")
for t, s in topk_similar("Fahrkarte", k=10):
    print(f"{t:25s} | {s:.3f}")

print("\n=== Satz → ähnliche Sätze/Wörter ===")
for t, s in topk_similar("Lass uns mit dem Bus fahren", k=10):
    print(f"{t:25s} | {s:.3f}")
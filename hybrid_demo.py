import numpy as np
from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# ===== 1) Modell: nimm den Sieger aus deinem Benchmark =====
MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_ID)

# ===== 2) Enriched Items (Wort + Definition + Beispiel + Tags) =====
@dataclass
class Item:
    surface: str
    definition: str
    example: str
    tags: List[str]

def T(x: Item) -> str:
    # Alles in EINEN Text gießen → ein Vektor mit viel Bedeutung
    return f"{x.surface} — {x.definition}. Beispiel: {x.example}. Tags: {', '.join(x.tags)}."

items = [
    # ÖPNV & Verkehr
    Item("Fahrkarte", "Ticket für den öffentlichen Nahverkehr",
         "Ich kaufe eine Fahrkarte am Automaten.", ["oepnv","verkehr"]),
    Item("Fahrplan", "Übersicht der Abfahrts- und Ankunftszeiten",
         "Der Fahrplan hängt an der Haltestelle.", ["oepnv","verkehr"]),
    Item("Haltestelle", "Ort, an dem Busse/Bahnen halten",
         "Wir treffen uns an der Haltestelle.", ["oepnv","verkehr"]),
    Item("Straßenbahn", "Schienenfahrzeug für den Stadtverkehr",
         "Die Straßenbahn kommt alle zehn Minuten.", ["oepnv","verkehr"]),
    Item("Bus", "Großes Fahrzeug für den öffentlichen Transport",
         "Der Bus ist überfüllt in der Rush Hour.", ["oepnv","verkehr"]),
    Item("Taxi", "Mietwagen mit Fahrer für Einzelpersonen",
         "Ich rufe ein Taxi für den Flughafen.", ["verkehr","dienstleistung"]),
    Item("Fahrrad", "Zweirad mit Pedalen für Fortbewegung",
         "Ich fahre mit dem Fahrrad zur Arbeit.", ["verkehr","sport"]),
    Item("Auto", "Vierrädriges Fahrzeug für Personenverkehr",
         "Das Auto steht in der Garage.", ["verkehr","technik"]),
    Item("Motorrad", "Zweirad mit Motor für schnelle Fortbewegung",
         "Er fährt gerne Motorrad am Wochenende.", ["verkehr","sport"]),
    Item("U-Bahn", "Unterirdisches Verkehrsmittel in Städten",
         "Die U-Bahn bringt mich schnell ins Zentrum.", ["oepnv","verkehr"]),
    Item("S-Bahn", "Stadtbahn für regionale Verbindungen",
         "Die S-Bahn fährt alle 15 Minuten.", ["oepnv","verkehr"]),
    
    # Mode & Kleidung
    Item("Krawatte", "Schmales Tuch als Kleidungsteil am Hals",
         "Er trägt eine rote Krawatte zum Anzug.", ["mode"]),
    Item("Hemd", "Oberteil mit Kragen und Knöpfen",
         "Das weiße Hemd passt gut zur Hose.", ["mode"]),
    Item("Jeans", "Robuste Hose aus Denim-Stoff",
         "Ich trage gerne Jeans in der Freizeit.", ["mode"]),
    Item("Schuhe", "Fußbekleidung für Schutz und Komfort",
         "Diese Schuhe sind sehr bequem.", ["mode"]),
    Item("Jacke", "Äußere Bekleidung für Schutz vor Wetter",
         "Die Jacke hält mich warm im Winter.", ["mode"]),
    Item("Hut", "Kopfbedeckung zum Schutz oder als Accessoire",
         "Er trägt einen schwarzen Hut.", ["mode"]),
    Item("Schal", "Langes Tuch zum Umlegen um Hals oder Schultern",
         "Der Schal wärmt mich bei kaltem Wetter.", ["mode"]),
    Item("Handschuhe", "Bekleidung für die Hände",
         "Die Handschuhe halten meine Hände warm.", ["mode"]),
    Item("Schmuck", "Dekorative Gegenstände zur Verschönerung",
         "Sie trägt eleganten Schmuck zur Party.", ["mode","luxus"]),
    Item("Uhr", "Zeitmessgerät am Handgelenk",
         "Meine Uhr zeigt die genaue Zeit.", ["mode","technik"]),
    
    # Essen & Trinken
    Item("Mittag", "Tageszeit um 12 Uhr",
         "Wir essen Mittag in der Kantine.", ["alltag"]),
    Item("Frühstück", "Erste Mahlzeit des Tages",
         "Das Frühstück ist die wichtigste Mahlzeit.", ["alltag","essen"]),
    Item("Abendessen", "Letzte Mahlzeit des Tages",
         "Wir essen um 18 Uhr zu Abend.", ["alltag","essen"]),
    Item("Kaffee", "Heißes Getränk aus gerösteten Bohnen",
         "Ich trinke morgens gerne Kaffee.", ["getränk","alltag"]),
    Item("Tee", "Heißes Aufgussgetränk aus Kräutern oder Blättern",
         "Grüner Tee ist sehr gesund.", ["getränk","gesundheit"]),
    Item("Brot", "Gebackenes Nahrungsmittel aus Mehl",
         "Frisches Brot schmeckt am besten.", ["essen","grundnahrung"]),
    Item("Butter", "Milchprodukt zum Bestreichen",
         "Butter macht das Brot geschmackvoller.", ["essen","milchprodukt"]),
    Item("Käse", "Milchprodukt mit festem oder weichem Zustand",
         "Der Käse ist sehr würzig.", ["essen","milchprodukt"]),
    Item("Fleisch", "Tierisches Nahrungsmittel",
         "Das Fleisch ist gut durchgebraten.", ["essen","protein"]),
    Item("Fisch", "Wassertier als Nahrungsmittel",
         "Frischer Fisch ist sehr gesund.", ["essen","protein","gesundheit"]),
    Item("Gemüse", "Pflanzliche Nahrungsmittel",
         "Gemüse enthält viele Vitamine.", ["essen","gesundheit"]),
    Item("Obst", "Süße pflanzliche Nahrungsmittel",
         "Obst ist ein gesunder Snack.", ["essen","gesundheit"]),
    Item("Wasser", "Flüssigkeit zum Trinken",
         "Wasser ist lebenswichtig für den Körper.", ["getränk","gesundheit"]),
    Item("Bier", "Alkoholisches Getränk aus Hopfen und Malz",
         "Ein kaltes Bier schmeckt nach der Arbeit.", ["getränk","alkohol"]),
    Item("Wein", "Alkoholisches Getränk aus Trauben",
         "Roter Wein passt gut zum Essen.", ["getränk","alkohol"]),
    
    # Arbeit & Büro
    Item("Computer", "Elektronisches Gerät zur Datenverarbeitung",
         "Der Computer läuft den ganzen Tag.", ["technik","arbeit"]),
    Item("Laptop", "Tragbarer Computer für unterwegs",
         "Ich arbeite mit dem Laptop im Café.", ["technik","arbeit"]),
    Item("Tastatur", "Eingabegerät mit Tasten für Computer",
         "Die Tastatur ist sehr leise.", ["technik","arbeit"]),
    Item("Maus", "Eingabegerät zur Steuerung des Cursors",
         "Die Maus funktioniert nicht richtig.", ["technik","arbeit"]),
    Item("Monitor", "Bildschirm für die Anzeige von Inhalten",
         "Der Monitor ist sehr scharf.", ["technik","arbeit"]),
    Item("Drucker", "Gerät zum Ausdrucken von Dokumenten",
         "Der Drucker braucht neues Papier.", ["technik","arbeit"]),
    Item("Telefon", "Gerät zur Sprachkommunikation",
         "Das Telefon klingelt ständig.", ["technik","kommunikation"]),
    Item("Handy", "Tragbares Telefon für unterwegs",
         "Mein Handy ist leer.", ["technik","kommunikation"]),
    Item("E-Mail", "Elektronische Nachricht über Internet",
         "Ich bekomme viele E-Mails täglich.", ["technik","kommunikation"]),
    Item("Meeting", "Zusammenkunft für Besprechungen",
         "Das Meeting dauert zwei Stunden.", ["arbeit","kommunikation"]),
    Item("Präsentation", "Vortrag mit visuellen Elementen",
         "Die Präsentation war sehr überzeugend.", ["arbeit","kommunikation"]),
    Item("Bericht", "Schriftliche Zusammenfassung von Informationen",
         "Ich schreibe einen Bericht über das Projekt.", ["arbeit","dokumentation"]),
    Item("Projekt", "Vorhaben mit bestimmten Zielen",
         "Das Projekt ist fast fertig.", ["arbeit","organisation"]),
    Item("Deadline", "Frist für die Fertigstellung",
         "Die Deadline ist nächste Woche.", ["arbeit","organisation"]),
    Item("Kollege", "Person, die im gleichen Unternehmen arbeitet",
         "Mein Kollege hilft mir bei der Aufgabe.", ["arbeit","sozial"]),
    Item("Chef", "Vorgesetzter in einem Unternehmen",
         "Der Chef ist mit der Arbeit zufrieden.", ["arbeit","hierarchie"]),
    Item("Gehalt", "Regelmäßige Bezahlung für Arbeit",
         "Das Gehalt wird monatlich überwiesen.", ["arbeit","finanzen"]),
    Item("Urlaub", "Freie Zeit für Erholung",
         "Ich nehme nächsten Monat Urlaub.", ["arbeit","freizeit"]),
    
    # Freizeit & Hobbys
    Item("Buch", "Gedrucktes Werk zum Lesen",
         "Ich lese gerne spannende Bücher.", ["freizeit","bildung"]),
    Item("Film", "Bewegte Bilder zur Unterhaltung",
         "Der Film war sehr emotional.", ["freizeit","unterhaltung"]),
    Item("Musik", "Klangliche Kunst zur Unterhaltung",
         "Musik entspannt mich nach der Arbeit.", ["freizeit","unterhaltung"]),
    Item("Sport", "Körperliche Aktivität zur Fitness",
         "Sport hält mich fit und gesund.", ["freizeit","gesundheit"]),
    Item("Fußball", "Ballsport mit zwei Mannschaften",
         "Fußball spielen macht mir Spaß.", ["sport","freizeit"]),
    Item("Tennis", "Racketsport mit Ball",
         "Tennis spielen ist anstrengend.", ["sport","freizeit"]),
    Item("Schwimmen", "Fortbewegung im Wasser",
         "Schwimmen trainiert alle Muskeln.", ["sport","gesundheit"]),
    Item("Laufen", "Schnelle Fortbewegung zu Fuß",
         "Laufen in der Natur entspannt mich.", ["sport","gesundheit"]),
    Item("Fahrradfahren", "Fortbewegung mit Fahrrad",
         "Fahrradfahren ist gut für die Umwelt.", ["sport","verkehr"]),
    Item("Spaziergang", "Gemütliches Gehen zur Erholung",
         "Ein Spaziergang im Park ist entspannend.", ["freizeit","gesundheit"]),
    Item("Reisen", "Besuch fremder Orte",
         "Reisen erweitert den Horizont.", ["freizeit","bildung"]),
    Item("Fotografie", "Kunst des Fotografierens",
         "Fotografie ist mein Hobby.", ["freizeit","kunst"]),
    Item("Malen", "Kunst mit Farben und Pinsel",
         "Malen entspannt mich sehr.", ["freizeit","kunst"]),
    Item("Kochen", "Zubereitung von Speisen",
         "Kochen macht mir großen Spaß.", ["freizeit","essen"]),
    Item("Garten", "Bereich mit Pflanzen und Blumen",
         "Der Garten blüht im Frühling.", ["freizeit","natur"]),
    Item("Wandern", "Längere Wanderung in der Natur",
         "Wandern in den Bergen ist wunderschön.", ["freizeit","natur","sport"]),
    
    # Gesundheit & Medizin
    Item("Arzt", "Mediziner für Behandlung von Krankheiten",
         "Ich gehe morgen zum Arzt.", ["gesundheit","medizin"]),
    Item("Krankenhaus", "Einrichtung für medizinische Behandlung",
         "Das Krankenhaus ist sehr modern.", ["gesundheit","medizin"]),
    Item("Medikament", "Arzneimittel zur Behandlung",
         "Das Medikament hilft gegen Schmerzen.", ["gesundheit","medizin"]),
    Item("Schmerzen", "Unangenehme körperliche Empfindung",
         "Die Schmerzen sind heute besser.", ["gesundheit","symptom"]),
    Item("Fieber", "Erhöhte Körpertemperatur",
         "Das Kind hat hohes Fieber.", ["gesundheit","symptom"]),
    Item("Husten", "Reflexartiges Ausstoßen von Luft",
         "Der Husten hält schon lange an.", ["gesundheit","symptom"]),
    Item("Schnupfen", "Entzündung der Nasenschleimhaut",
         "Der Schnupfen ist sehr lästig.", ["gesundheit","symptom"]),
    Item("Kopfschmerzen", "Schmerzen im Kopfbereich",
         "Kopfschmerzen können verschiedene Ursachen haben.", ["gesundheit","symptom"]),
    Item("Zahnarzt", "Mediziner für Zahnbehandlung",
         "Der Zahnarzt Termin ist nächste Woche.", ["gesundheit","medizin"]),
    Item("Zähne", "Harte Strukturen im Mund",
         "Gesunde Zähne sind wichtig.", ["gesundheit","anatomie"]),
    Item("Impfung", "Schutz vor Krankheiten durch Impfstoff",
         "Die Impfung ist sehr wichtig.", ["gesundheit","vorsorge"]),
    Item("Apotheke", "Einrichtung für Medikamente",
         "Die Apotheke ist um die Ecke.", ["gesundheit","medizin"]),
    Item("Krankenversicherung", "Versicherung für Gesundheitskosten",
         "Die Krankenversicherung übernimmt die Kosten.", ["gesundheit","versicherung"]),
    Item("Gesundheit", "Zustand körperlichen Wohlbefindens",
         "Gesundheit ist das Wichtigste im Leben.", ["gesundheit","grundbedürfnis"]),
    
    # Technik & Elektronik
    Item("Internet", "Globales Netzwerk für Datenübertragung",
         "Das Internet ist sehr schnell heute.", ["technik","kommunikation"]),
    Item("WLAN", "Drahtlose Internetverbindung",
         "Das WLAN funktioniert nicht.", ["technik","kommunikation"]),
    Item("App", "Anwendungsprogramm für mobile Geräte",
         "Diese App ist sehr nützlich.", ["technik","software"]),
    Item("Website", "Internetseite mit Informationen",
         "Die Website ist gut gestaltet.", ["technik","internet"]),
    Item("Smartphone", "Intelligentes mobiles Telefon",
         "Das Smartphone ist sehr praktisch.", ["technik","kommunikation"]),
    Item("Tablet", "Flaches tragbares Computer-Gerät",
         "Das Tablet ist perfekt zum Lesen.", ["technik","computer"]),
    Item("Kamera", "Gerät zum Aufnehmen von Bildern",
         "Die Kamera macht sehr gute Fotos.", ["technik","fotografie"]),
    Item("Fernseher", "Gerät zur Anzeige von Fernsehsendungen",
         "Der Fernseher ist zu laut.", ["technik","unterhaltung"]),
    Item("Lautsprecher", "Gerät zur Wiedergabe von Ton",
         "Die Lautsprecher haben guten Klang.", ["technik","audio"]),
    Item("Kopfhörer", "Gerät zum Hören von Audio",
         "Die Kopfhörer sind sehr bequem.", ["technik","audio"]),
    Item("Batterie", "Energiespeicher für elektronische Geräte",
         "Die Batterie ist leer.", ["technik","energie"]),
    Item("Ladekabel", "Kabel zum Aufladen von Geräten",
         "Das Ladekabel ist kaputt.", ["technik","zubehör"]),
    Item("USB", "Standard für Datenübertragung",
         "Der USB-Anschluss funktioniert nicht.", ["technik","verbindung"]),
    Item("Bluetooth", "Drahtlose Datenübertragung",
         "Bluetooth verbindet die Geräte.", ["technik","verbindung"]),
    Item("GPS", "System zur Positionsbestimmung",
         "Das GPS zeigt den richtigen Weg.", ["technik","navigation"]),
    
    # Natur & Umwelt
    Item("Baum", "Große Pflanze mit Stamm und Ästen",
         "Der Baum spendet Schatten.", ["natur","pflanze"]),
    Item("Blume", "Schöne Pflanze mit bunten Blüten",
         "Die Blume duftet wunderbar.", ["natur","pflanze"]),
    Item("Gras", "Kurze grüne Pflanzen",
         "Das Gras ist frisch gemäht.", ["natur","pflanze"]),
    Item("Sonne", "Stern, der Licht und Wärme spendet",
         "Die Sonne scheint hell.", ["natur","wetter"]),
    Item("Regen", "Wasser, das vom Himmel fällt",
         "Der Regen macht alles nass.", ["natur","wetter"]),
    Item("Schnee", "Gefrorenes Wasser in kristalliner Form",
         "Der Schnee bedeckt alles weiß.", ["natur","wetter"]),
    Item("Wind", "Bewegung der Luft",
         "Der Wind ist sehr stark heute.", ["natur","wetter"]),
    Item("Wolke", "Ansammlung von Wassertropfen am Himmel",
         "Die Wolke sieht aus wie ein Tier.", ["natur","wetter"]),
    Item("Berg", "Hohe Erhebung der Erdoberfläche",
         "Der Berg ist sehr steil.", ["natur","geographie"]),
    Item("Fluss", "Fließendes Gewässer",
         "Der Fluss fließt ins Meer.", ["natur","wasser"]),
    Item("Meer", "Großes salziges Gewässer",
         "Das Meer ist sehr ruhig heute.", ["natur","wasser"]),
    Item("Wald", "Großes Gebiet mit vielen Bäumen",
         "Der Wald ist sehr dicht.", ["natur","pflanze"]),
    Item("Tier", "Lebewesen, das sich bewegen kann",
         "Das Tier lebt in der Wildnis.", ["natur","biologie"]),
    Item("Vogel", "Tier mit Federn und Flügeln",
         "Der Vogel singt schön.", ["natur","biologie"]),
    Item("Hund", "Häufiges Haustier",
         "Der Hund ist sehr treu.", ["natur","haustier"]),
    Item("Katze", "Beliebtes Haustier",
         "Die Katze schläft gerne.", ["natur","haustier"]),
    
    # Wohnen & Haushalt
    Item("Haus", "Gebäude zum Wohnen",
         "Das Haus hat einen großen Garten.", ["wohnen","gebäude"]),
    Item("Wohnung", "Abgeschlossene Räume zum Wohnen",
         "Die Wohnung ist sehr hell.", ["wohnen","gebäude"]),
    Item("Zimmer", "Abgeschlossener Raum in einem Gebäude",
         "Das Zimmer ist gemütlich eingerichtet.", ["wohnen","raum"]),
    Item("Küche", "Raum zur Zubereitung von Speisen",
         "Die Küche ist sehr modern.", ["wohnen","raum"]),
    Item("Badezimmer", "Raum mit Badewanne oder Dusche",
         "Das Badezimmer ist sauber.", ["wohnen","raum"]),
    Item("Schlafzimmer", "Raum zum Schlafen",
         "Das Schlafzimmer ist ruhig.", ["wohnen","raum"]),
    Item("Wohnzimmer", "Hauptraum zum Leben und Entspannen",
         "Das Wohnzimmer ist gemütlich.", ["wohnen","raum"]),
    Item("Tür", "Öffnung zum Betreten oder Verlassen",
         "Die Tür ist verschlossen.", ["wohnen","möbel"]),
    Item("Fenster", "Öffnung für Licht und Luft",
         "Das Fenster ist offen.", ["wohnen","möbel"]),
    Item("Tisch", "Möbelstück mit flacher Oberfläche",
         "Der Tisch steht in der Mitte.", ["wohnen","möbel"]),
    Item("Stuhl", "Möbelstück zum Sitzen",
         "Der Stuhl ist sehr bequem.", ["wohnen","möbel"]),
    Item("Bett", "Möbelstück zum Schlafen",
         "Das Bett ist sehr weich.", ["wohnen","möbel"]),
    Item("Schrank", "Möbelstück zur Aufbewahrung",
         "Der Schrank ist voll.", ["wohnen","möbel"]),
    Item("Licht", "Elektrische Beleuchtung",
         "Das Licht ist zu hell.", ["wohnen","elektrik"]),
    Item("Heizung", "Gerät zur Raumwärmung",
         "Die Heizung ist an.", ["wohnen","klima"]),
    Item("Kühlschrank", "Gerät zur Kühlung von Lebensmitteln",
         "Der Kühlschrank ist voll.", ["wohnen","haushalt"]),
    Item("Waschmaschine", "Gerät zum Waschen von Kleidung",
         "Die Waschmaschine läuft.", ["wohnen","haushalt"]),
    Item("Geschirrspüler", "Gerät zum Spülen von Geschirr",
         "Der Geschirrspüler ist neu.", ["wohnen","haushalt"]),
    Item("Staubsauger", "Gerät zum Reinigen von Böden",
         "Der Staubsauger ist sehr leise.", ["wohnen","haushalt"]),
    Item("Besen", "Gerät zum Kehren",
         "Der Besen steht in der Ecke.", ["wohnen","haushalt"]),
    
    # Soziale Beziehungen & Freundschaft
    Item("Freund", "Person, mit der man eine enge Beziehung hat",
         "Mein Freund hilft mir immer.", ["sozial","beziehung"]),
    Item("Freundin", "Weibliche Person, mit der man eine enge Beziehung hat",
         "Meine Freundin ist sehr nett.", ["sozial","beziehung"]),
    Item("Partner", "Person in einer Beziehung",
         "Mein Partner und ich gehen ins Kino.", ["sozial","beziehung"]),
    Item("Familie", "Gruppe verwandter Menschen",
         "Die Familie isst zusammen zu Abend.", ["sozial","beziehung"]),
    Item("Liebe", "Starke emotionale Zuneigung",
         "Liebe ist das Wichtigste im Leben.", ["sozial","beziehung"]),
    Item("Beziehung", "Verbindung zwischen Menschen",
         "Eine gute Beziehung braucht Vertrauen.", ["sozial","beziehung"]),
    Item("Treffen", "Zusammenkunft von Menschen",
         "Wir haben ein Treffen vereinbart.", ["sozial","kommunikation"]),
    Item("Party", "Gesellschaftliche Feier",
         "Die Party war sehr lustig.", ["sozial","freizeit"]),
    Item("Gesellschaft", "Gruppe von Menschen",
         "Die Gesellschaft ist sehr gemischt.", ["sozial","alltag"]),
    Item("Menschen", "Individuen der Spezies Homo sapiens",
         "Menschen sind soziale Wesen.", ["sozial","alltag"]),
]

# ===== 3) Embeddings & BM25 vorbereiten =====
embed_texts = [T(x) for x in items]
E = model.encode(embed_texts, normalize_embeddings=True)
E = np.asarray(E, dtype=np.float32)

bm25_corpus = [ (x.surface + " " + x.definition + " " + x.example + " " + " ".join(x.tags)).lower().split() for x in items ]
bm25 = BM25Okapi(bm25_corpus)

def normalize01(a: np.ndarray) -> np.ndarray:
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

def normalize_bm25(bm):
    """Robustere BM25-Normalisierung mit Z-Score + Sigmoid statt Min-Max"""
    bm = np.asarray(bm, dtype=np.float32)
    mu, sigma = float(bm.mean()), float(bm.std() + 1e-9)
    z = (bm - mu) / sigma
    # squash in [0,1]
    return 1.0 / (1.0 + np.exp(-z))

def choose_alpha(q: str):
    """Dynamisches Alpha basierend auf Query-Länge"""
    toks = q.strip().split()
    return 0.85 if len(toks) <= 2 else 0.65

def detect_relevant_tags(query: str) -> List[str]:
    """Intelligente Tag-Erkennung basierend auf Query-Inhalt"""
    query_lower = query.lower()
    
    # Tag-Keyword-Mapping
    tag_keywords = {
        'oepnv': ['fahrkarte', 'fahrplan', 'haltestelle', 'bus', 'bahn', 'u-bahn', 's-bahn', 'straßenbahn', 'öpnv', 'verkehr', 'fahren', 'zug', 'tram'],
        'verkehr': ['auto', 'fahrrad', 'motorrad', 'taxi', 'verkehr', 'fahren', 'straße'],
        'essen': ['essen', 'trinken', 'kochen', 'mahlzeit', 'frühstück', 'mittag', 'abendessen', 'essen gehen', 'restaurant', 'küche', 'speise'],
        'getränk': ['trinken', 'wasser', 'kaffee', 'tee', 'bier', 'wein', 'saft'],
        'arbeit': ['arbeit', 'büro', 'computer', 'meeting', 'projekt', 'kollege', 'chef', 'gehalt', 'urlaub'],
        'technik': ['computer', 'handy', 'internet', 'app', 'software', 'technik', 'elektronik'],
        'gesundheit': ['arzt', 'krankenhaus', 'medizin', 'gesundheit', 'krank', 'schmerzen', 'fieber'],
        'freizeit': ['freizeit', 'hobby', 'sport', 'reisen', 'lesen', 'film', 'musik', 'spielen'],
        'sport': ['sport', 'fußball', 'tennis', 'schwimmen', 'laufen', 'fitness', 'training'],
        'natur': ['natur', 'baum', 'blume', 'wald', 'berg', 'sonne', 'regen', 'tier'],
        'wohnen': ['haus', 'wohnung', 'zimmer', 'küche', 'badezimmer', 'wohnen', 'einrichten'],
        'mode': ['kleidung', 'mode', 'hemd', 'jeans', 'schuhe', 'jacke', 'anziehen', 'tragen'],
        'sozial': ['freund', 'freunde', 'freundschaft', 'beziehung', 'partner', 'kollege', 'familie', 'menschen', 'treffen', 'party', 'gesellschaft'],
        'beziehung': ['liebe', 'verliebt', 'partner', 'freund', 'freundin', 'beziehung', 'zusammen', 'paar']
    }
    
    detected_tags = []
    for tag, keywords in tag_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_tags.append(tag)
    
    # Fallback: Wenn keine Tags erkannt werden, keine Filterung
    return detected_tags if detected_tags else None

def mmr(query_vec, cand_vecs, lam=0.7, topk=5):
    """Original MMR auf Embeddings"""
    selected, pool = [], list(range(len(cand_vecs)))
    simq = cand_vecs @ query_vec
    while pool and len(selected) < topk:
        best, bestv = None, -1e9
        for i in pool:
            div = 0 if not selected else max(cand_vecs[i] @ cand_vecs[j] for j in selected)
            v = lam*simq[i] - (1-lam)*div
            if v > bestv: best, bestv = i, v
        selected.append(best); pool.remove(best)
    return selected

def mmr_hybrid(cand_vecs, cand_scores, lam=0.7, topk=5):
    """MMR mit Hybrid-Score als Relevanz, Cosine als Diversität"""
    selected, pool = [], list(range(len(cand_scores)))
    while pool and len(selected) < topk:
        best, bestv = None, -1e9
        for i in pool:
            div = 0 if not selected else max(cand_vecs[i] @ cand_vecs[j] for j in selected)
            v = lam*cand_scores[i] - (1-lam)*div
            if v > bestv:
                best, bestv = i, v
        selected.append(best); pool.remove(best)
    return selected

def search(query: str, k=5, alpha=None, must_have_tags=None, min_score=0.0):
    """Verbesserte Suche mit robustem BM25 und MMR auf Hybrid-Score"""
    if alpha is None:
        alpha = choose_alpha(query)
    
    qv = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    cos = E @ qv  # Cosine (normalised)
    bm = bm25.get_scores(query.lower().split())
    bm = normalize_bm25(bm)  # Robuste Z-Score + Sigmoid Normalisierung
    score = alpha * cos + (1 - alpha) * bm

    # Tag-Filter (hart)
    idxs = list(np.argsort(-score))
    if must_have_tags:
        idxs = [i for i in idxs if set(must_have_tags) & set(items[i].tags)]

    # Score-Floor
    idxs = [i for i in idxs if score[i] >= min_score]

    # Top-N Shortlist für MMR
    order = np.argsort(-score)[: min(200, len(score))]
    if must_have_tags:
        order = [i for i in order if set(must_have_tags) & set(items[i].tags)]
    order = [i for i in order if score[i] >= min_score]
    
    # MMR mit Hybrid-Score
    cand_vecs = E[order]
    cand_scores = score[order]
    sel_local = mmr_hybrid(cand_vecs, cand_scores, lam=0.7, topk=min(k, len(order)))
    final_idxs = [order[j] for j in sel_local]
    
    final = [(items[i].surface, float(score[i]), items[i].tags) for i in final_idxs]
    return final

if __name__ == "__main__":
    # ===== 4) Drei Modi vergleichen =====
    q = "Wann kommt der Bus?"

    print("\n--- A) Nur Cosine (ohne Kontext, nur surface) → zum Vergleich ---")
    E_surface = model.encode([x.surface for x in items], normalize_embeddings=True)
    qv = model.encode([q], normalize_embeddings=True)[0]
    sims = (E_surface @ qv).astype(float)
    for i in np.argsort(-sims)[:5]:
        print(f"{items[i].surface:15s} | {sims[i]:.3f} | tags={items[i].tags}")

    print("\n--- B) Enriched Embeddings + Hybrid (BM25+Cosine), kein Tag-Filter ---")
    for name, s, tags in search(q, k=5, alpha=0.7, must_have_tags=None, min_score=0.0):
        print(f"{name:15s} | {s:.3f} | tags={tags}")

    # Intelligente Tag-Erkennung
    detected_tags = detect_relevant_tags(q)
    tag_info = f"detected_tags={detected_tags}" if detected_tags else "no_tag_filter"
    
    print(f"\n--- C) Enriched + Hybrid + Tag-Filter ({tag_info}) + MMR-Hybrid + Score-Floor ---")
    for name, s, tags in search(q, k=5, alpha=None, must_have_tags=detected_tags, min_score=0.35):
        print(f"{name:15s} | {s:.3f} | tags={tags}")
    
    print("\n--- D) Nur Hybrid-Score Sortierung (ohne MMR) für Vergleich ---")
    # Simuliere reine Score-Sortierung ohne MMR
    qv = model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
    cos = E @ qv
    bm = bm25.get_scores(q.lower().split())
    bm = normalize_bm25(bm)
    alpha = choose_alpha(q)
    score = alpha * cos + (1 - alpha) * bm
    
    # Items mit Score >= 0.35 (mit Tag-Filter falls erkannt)
    filtered_items = [(i, score[i]) for i in range(len(items)) 
                     if score[i] >= 0.35]
    
    if detected_tags:
        filtered_items = [(i, s) for i, s in filtered_items 
                         if set(detected_tags) & set(items[i].tags)]
    
    filtered_items.sort(key=lambda x: x[1], reverse=True)
    
    for i, s in filtered_items[:5]:
        print(f"{items[i].surface:15s} | {s:.3f} | tags={items[i].tags}")

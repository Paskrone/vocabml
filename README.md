# VocabML

Ein intelligentes Vokabular-Lernsystem mit hybridem Suchansatz, das semantische Embeddings mit BM25-Textsuche kombiniert.

## Features

- **Hybride Suche**: Kombination aus semantischen Embeddings (SentenceTransformers) und BM25-Textsuche
- **Enriched Items**: Vokabeln mit Definition, Beispiel und Tags für bessere Kontextualisierung
- **MMR-Diversität**: Maximal Marginal Relevance für vielfältige Suchergebnisse
- **Intelligente Tag-Erkennung**: Automatische Kategorisierung basierend auf Query-Inhalt
- **Dynamische Gewichtung**: Adaptive Alpha-Parameter je nach Query-Länge

## Installation

```bash
pip install sentence-transformers rank-bm25 numpy
```

## Verwendung

### Grundlegende Suche
```python
from hybrid_demo import search

# Einfache Suche
results = search("Wann kommt der Bus?", k=5)
for name, score, tags in results:
    print(f"{name}: {score:.3f} | {tags}")
```

### Erweiterte Suche mit Tag-Filter
```python
# Suche mit spezifischen Tags
results = search("gesundes Essen", k=5, must_have_tags=["gesundheit", "essen"])
```

## Architektur

### Hybrid-Score
- **Cosine-Similarity**: Semantische Ähnlichkeit über SentenceTransformers
- **BM25**: Lexikalische Textsuche für exakte Wort-Matches
- **Dynamisches Alpha**: Gewichtung je nach Query-Länge

### MMR-Diversität
- Verhindert redundante Suchergebnisse
- Balanciert Relevanz und Diversität
- Optimiert für bessere Benutzererfahrung

## Modelle

- **Standard**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Optimiert**: Für deutsche Texte und mehrsprachige Inhalte

## Dateien

- `hybrid_demo.py`: Hauptimplementierung mit hybridem Suchsystem
- `benchmark_models.py`: Modellvergleich und Performance-Tests
- `nn_demo_numpy.py`: Reine Neural Network Demo
- `recommend_demo.py`: Empfehlungssystem-Demo
- `compare_similarity.py`: Similarity-Vergleichstools

## Performance

Das System verwendet exakte Nearest Neighbor Suche (keine ANN) und ist für Datensätze bis ~10.000 Items optimiert. Für größere Datensätze empfiehlt sich eine ANN-Implementierung mit FAISS oder ähnlichen Bibliotheken.

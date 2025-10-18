#!/usr/bin/env python3
"""
Test-Script für verschiedene Query-Typen mit intelligenter Tag-Erkennung
"""

from hybrid_demo import search, detect_relevant_tags, items

def test_query(query: str):
    print(f"\n{'='*60}")
    print(f"QUERY: '{query}'")
    print('='*60)
    
    # Tag-Erkennung
    detected_tags = detect_relevant_tags(query)
    tag_info = f"detected_tags={detected_tags}" if detected_tags else "no_tag_filter"
    print(f"Erkannte Tags: {tag_info}")
    
    # Suche
    results = search(query, k=5, alpha=None, must_have_tags=detected_tags, min_score=0.35)
    
    print(f"\nTop {len(results)} Ergebnisse:")
    for i, (name, score, tags) in enumerate(results, 1):
        print(f"{i}. {name:15s} | {score:.3f} | tags={tags}")

if __name__ == "__main__":
    # Verschiedene Query-Typen testen
    test_queries = [
        "Fahrkarte",
        "Lass uns essen gehen", 
        "Ich brauche ein neues Handy",
        "Sport machen",
        "Arzttermin",
        "Wohnung einrichten",
        "Mode kaufen",
        "Natur erleben",
        "Computer reparieren",
        "Reisen planen"
    ]
    
    for query in test_queries:
        test_query(query)

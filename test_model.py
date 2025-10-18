from sentence_transformers import SentenceTransformer
import numpy as np, torch

print("numpy:", np.__version__)
print("torch:", torch.__version__)

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode("Hallo Welt", normalize_embeddings=True)
print("shape:", emb.shape, "first10:", emb[:10])

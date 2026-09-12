from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import os

print("Loading dataset...")
dataset = load_dataset("fancyzhx/ag_news")

texts = dataset["train"]["text"][:50000]

print(f"Loaded {len(texts)} texts")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True
)

os.makedirs("data", exist_ok=True)

np.save("data/embeddings.npy", embeddings)
np.save("data/texts.npy", np.array(texts, dtype=object))

print("Embeddings shape:", embeddings.shape)
print("Saved successfully!")
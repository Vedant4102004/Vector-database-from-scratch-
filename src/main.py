import numpy as np
from sentence_transformers import SentenceTransformer

from ivf_index import IVFIndex


print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading embeddings and texts...")
vectors = np.load("data/embeddings.npy")
texts = np.load("data/texts.npy", allow_pickle=True)

print("Building IVF index...")
index = IVFIndex(vectors, n_clusters=500)
index.build()

print("\nVector Database From Scratch")
print("--------------------------------")
print("Type a statement to search.")
print("Type 'exit' to quit.\n")

while True:
    query_text = input("Enter a statement: ").strip()

    if query_text.lower() == "exit":
        print("Goodbye!")
        break

    if not query_text:
        continue

    query_vector = model.encode(query_text)

    results = index.search(
        query_vector,
        top_k=5,
        n_probe=10
    )

    print("\nTop matches:")
    print("--------------------------------")

    for rank, (idx, score) in enumerate(results, start=1):
        print(f"{rank}. {score:.4f} | {texts[idx]}")

    print()
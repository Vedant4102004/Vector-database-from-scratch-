import numpy as np
from ivf_index import IVFIndex

vectors = np.load("data/embeddings.npy")
texts = np.load("data/texts.npy", allow_pickle=True)

index = IVFIndex(vectors, n_clusters=500)

print("Building IVF index...")
index.build(iterations=10)
print("IVF index built.")

results = index.search(vectors[0], top_k=5, n_probe=10)

print("\nTop results:")

for idx, score in results:
    print(f"{score:.4f} | {texts[idx]}")

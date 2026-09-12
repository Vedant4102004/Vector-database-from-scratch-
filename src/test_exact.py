import numpy as np
from exact_index import ExactIndex

vectors = np.load("data/embeddings.npy")
texts = np.load("data/texts.npy", allow_pickle=True)

index = ExactIndex(vectors)

results = index.search(vectors[0], top_k=5)

for idx, score in results:
    print(f"{score:.4f} | {texts[idx]}")
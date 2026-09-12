import numpy as np
from ivf_index import IVFIndex

vectors = np.load("data/embeddings.npy")

index = IVFIndex(vectors, n_clusters=500)

print("Building IVF index...")
index.build(iterations=10)
print("IVF index built.")

new_vector = vectors[0].copy()

print("\nTesting insert...")

new_index = index.insert(new_vector)

print(f"Inserted vector at index: {new_index}")

results = index.search(
    new_vector,
    top_k=5,
    n_probe=10
)

print("\nSearch after insert:")

for idx, score in results:
    print(f"{idx} | {score:.4f}")

print("\nTesting delete...")

deleted = index.delete(new_index)

print(f"Delete successful: {deleted}")

print("\nOperations test completed.")
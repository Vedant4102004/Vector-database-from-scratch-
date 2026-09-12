import time
import numpy as np
import matplotlib.pyplot as plt

from exact_index import ExactIndex
from ivf_index import IVFIndex


vectors = np.load("data/embeddings.npy")

print("Loading indexes...")

exact_index = ExactIndex(vectors)

ivf_index = IVFIndex(vectors, n_clusters=500)

print("Building IVF index...")
ivf_index.build(iterations=10)
print("IVF index built.")

rng = np.random.default_rng(42)
query_indices = rng.choice(len(vectors), 500, replace=False)
queries = vectors[query_indices]

print("\nComputing exact ground truth...")

ground_truth = []

start = time.perf_counter()

for query in queries:
    results = exact_index.search(query, top_k=10)
    ground_truth.append([idx for idx, _ in results])

exact_time = time.perf_counter() - start
exact_qps = len(queries) / exact_time

print(f"Exact search: {exact_qps:.2f} queries/sec")

n_probe_values = [1, 5, 10, 25, 50]
recalls = []
qps_values = []

print("\nIVF Benchmark")
print("-" * 55)
print("n_probe | Recall@10 | Queries/sec")
print("-" * 55)

for n_probe in n_probe_values:

    start = time.perf_counter()

    total_recall = 0.0

    for i, query in enumerate(queries):
        results = ivf_index.search(
            query,
            top_k=10,
            n_probe=n_probe
        )

        predicted = {idx for idx, _ in results}
        actual = set(ground_truth[i])

        total_recall += len(predicted & actual) / len(actual)

    elapsed = time.perf_counter() - start

    recall = total_recall / len(queries)
    qps = len(queries) / elapsed

    recalls.append(recall)
    qps_values.append(qps)

    print(f"{n_probe:7d} | {recall:9.4f} | {qps:12.2f}")


plt.figure()
plt.plot(n_probe_values, recalls, marker="o")
plt.xlabel("n_probe")
plt.ylabel("Recall@10")
plt.title("IVF Recall vs n_probe")
plt.grid(True)
plt.savefig("plots/recall_vs_nprobe.png", dpi=150)
plt.close()


plt.figure()
plt.plot(n_probe_values, qps_values, marker="o")
plt.xlabel("n_probe")
plt.ylabel("Queries per second")
plt.title("IVF Speed vs n_probe")
plt.grid(True)
plt.savefig("plots/speed_vs_nprobe.png", dpi=150)
plt.close()


print("\nPlots saved:")
print("plots/recall_vs_nprobe.png")
print("plots/speed_vs_nprobe.png")
import time
import numpy as np
import matplotlib.pyplot as plt

from exact_index import ExactIndex
from ivf_index import IVFIndex


vectors = np.load("data/embeddings.npy")

print("Loading indexes...")

exact_index = ExactIndex(vectors)

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


n_clusters = 500

print(f"\nBuilding IVF index with {n_clusters} clusters...")

ivf_index = IVFIndex(
    vectors,
    n_clusters=n_clusters
)

ivf_index.build(iterations=10)

print("IVF index built.")


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

    print(
        f"{n_probe:7d} | "
        f"{recall:9.4f} | "
        f"{qps:12.2f}"
    )


plt.figure()
plt.plot(
    n_probe_values,
    recalls,
    marker="o"
)

plt.xlabel("n_probe")
plt.ylabel("Recall@10")
plt.title("IVF Recall vs n_probe")
plt.grid(True)
plt.savefig(
    "plots/recall_vs_nprobe.png",
    dpi=150
)
plt.close()


plt.figure()
plt.plot(
    n_probe_values,
    qps_values,
    marker="o"
)

plt.xlabel("n_probe")
plt.ylabel("Queries per second")
plt.title("IVF Speed vs n_probe")
plt.grid(True)
plt.savefig(
    "plots/speed_vs_nprobe.png",
    dpi=150
)
plt.close()


print("\nPlots saved:")
print("plots/recall_vs_nprobe.png")
print("plots/speed_vs_nprobe.png")


print("\n")
print("=" * 65)
print("Effect of Number of Clusters")
print("=" * 65)

cluster_values = [100, 250, 500, 1000]

cluster_recalls = []
cluster_qps = []

fixed_n_probe = 10

print(
    f"{'Clusters':>10} | "
    f"{'Recall@10':>10} | "
    f"{'Queries/sec':>12}"
)

print("-" * 65)

for cluster_count in cluster_values:

    print(f"Building IVF with {cluster_count} clusters...")

    index = IVFIndex(
        vectors,
        n_clusters=cluster_count
    )

    index.build(iterations=10)

    start = time.perf_counter()

    total_recall = 0.0

    for i, query in enumerate(queries):

        results = index.search(
            query,
            top_k=10,
            n_probe=fixed_n_probe
        )

        predicted = {idx for idx, _ in results}
        actual = set(ground_truth[i])

        total_recall += len(predicted & actual) / len(actual)

    elapsed = time.perf_counter() - start

    recall = total_recall / len(queries)
    qps = len(queries) / elapsed

    cluster_recalls.append(recall)
    cluster_qps.append(qps)

    print(
        f"{cluster_count:10d} | "
        f"{recall:10.4f} | "
        f"{qps:12.2f}"
    )


plt.figure()
plt.plot(
    cluster_values,
    cluster_recalls,
    marker="o"
)

plt.xlabel("Number of clusters")
plt.ylabel("Recall@10")
plt.title("IVF Recall vs Number of Clusters")
plt.grid(True)
plt.savefig(
    "plots/recall_vs_clusters.png",
    dpi=150
)
plt.close()


plt.figure()
plt.plot(
    cluster_values,
    cluster_qps,
    marker="o"
)

plt.xlabel("Number of clusters")
plt.ylabel("Queries per second")
plt.title("IVF Speed vs Number of Clusters")
plt.grid(True)
plt.savefig(
    "plots/speed_vs_clusters.png",
    dpi=150
)
plt.close()


print("\nAdditional plots saved:")
print("plots/recall_vs_clusters.png")
print("plots/speed_vs_clusters.png")
import numpy as np


class IVFIndex:
    def __init__(self, vectors, n_clusters=500):
        self.vectors = vectors.astype(np.float32)

        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norms + 1e-10)

        self.n_clusters = n_clusters
        self.centroids = None
        self.inverted_lists = {}

    def build(self, iterations=10):
        n = len(self.vectors)

        rng = np.random.default_rng(42)

        centroids = np.empty(
            (self.n_clusters, self.vectors.shape[1]),
            dtype=np.float32
        )

        first_index = rng.integers(n)
        centroids[0] = self.vectors[first_index]

        closest_distances = 1 - (self.vectors @ centroids[0])
        closest_distances = np.maximum(closest_distances, 0)

        for i in range(1, self.n_clusters):
            probabilities = closest_distances / closest_distances.sum()

            next_index = rng.choice(
                n,
                p=probabilities
            )

            centroids[i] = self.vectors[next_index]

            distances = 1 - (self.vectors @ centroids[i])
            distances = np.maximum(distances, 0)

            closest_distances = np.minimum(
                closest_distances,
                distances
            )

        self.centroids = centroids

        for _ in range(iterations):
            similarities = self.vectors @ self.centroids.T
            assignments = np.argmax(similarities, axis=1)

            new_centroids = np.zeros_like(self.centroids)

            for i in range(self.n_clusters):
                cluster_vectors = self.vectors[assignments == i]

                if len(cluster_vectors) > 0:
                    centroid = cluster_vectors.mean(axis=0)
                    centroid /= np.linalg.norm(centroid) + 1e-10
                    new_centroids[i] = centroid
                else:
                    new_centroids[i] = self.centroids[i]

            self.centroids = new_centroids

        self.inverted_lists = {}

        for i, cluster_id in enumerate(assignments):
            if cluster_id not in self.inverted_lists:
                self.inverted_lists[cluster_id] = []

            self.inverted_lists[cluster_id].append(i)

    def search(self, query, top_k=5, n_probe=10):
        query = query.astype(np.float32)

        query_norm = np.linalg.norm(query)
        query = query / (query_norm + 1e-10)

        centroid_scores = self.centroids @ query

        probe_clusters = np.argpartition(
            -centroid_scores,
            min(n_probe, self.n_clusters) - 1
        )[:n_probe]

        candidate_indices = []

        for cluster_id in probe_clusters:
            candidate_indices.extend(
                self.inverted_lists.get(int(cluster_id), [])
            )

        if not candidate_indices:
            return []

        candidate_vectors = self.vectors[candidate_indices]

        scores = candidate_vectors @ query

        top_indices = np.argpartition(
            -scores,
            min(top_k, len(scores)) - 1
        )[:top_k]

        top_indices = top_indices[
            np.argsort(-scores[top_indices])
        ]

        return [
            (
                int(candidate_indices[index]),
                float(scores[index])
            )
            for index in top_indices
        ]

    def insert(self, vector):
        vector = vector.astype(np.float32)

        norm = np.linalg.norm(vector)
        vector = vector / (norm + 1e-10)

        index = len(self.vectors)

        self.vectors = np.vstack([
            self.vectors,
            vector
        ])

        similarities = self.centroids @ vector
        cluster_id = int(np.argmax(similarities))

        if cluster_id not in self.inverted_lists:
            self.inverted_lists[cluster_id] = []

        self.inverted_lists[cluster_id].append(index)

        return index

    def delete(self, index):
            if index < 0 or index >= len(self.vectors):
                return False
    
            for cluster_id, indices in self.inverted_lists.items():
                if index in indices:
                    indices.remove(index)
                    break
    
            return True
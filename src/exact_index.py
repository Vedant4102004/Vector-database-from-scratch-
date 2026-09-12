import numpy as np


class ExactIndex:
    def __init__(self, vectors):
        self.vectors = vectors.astype(np.float32)

        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norms + 1e-10)

    def search(self, query, top_k=5):
        query = query.astype(np.float32)

        query_norm = np.linalg.norm(query)
        query = query / (query_norm + 1e-10)

        scores = self.vectors @ query

        top_indices = np.argpartition(
            -scores,
            min(top_k, len(scores)) - 1
        )[:top_k]

        top_indices = top_indices[
            np.argsort(-scores[top_indices])
        ]

        return [
            (int(index), float(scores[index]))
            for index in top_indices
        ]
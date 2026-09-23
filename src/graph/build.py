from __future__ import annotations

import numpy as np


def build_knn_edges(
    embeddings: np.ndarray,
    k: int = 5,
    max_degree: int | None = None,
) -> list[tuple[int, int, float]]:
    """Build a simple semantic k-NN edge list.

    Input embeddings should already be L2-normalized.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)

    edges = []
    for source in range(len(embeddings)):
        neighbors = np.argsort(-similarity[source])[:k]
        for target in neighbors:
            edges.append((source, int(target), float(similarity[source, target])))

    return edges

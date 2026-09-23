import numpy as np

from retrieval.faiss_index import FAISSIndex


def test_faiss_search():
    embeddings = np.eye(4, dtype=np.float32)
    index = FAISSIndex(4)
    index.add(embeddings)

    scores, ids = index.search(embeddings[:1], k=2)

    assert ids.shape == (1, 2)
    assert scores.shape == (1, 2)
    assert ids[0, 0] == 0

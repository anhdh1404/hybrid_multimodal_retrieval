from pathlib import Path

import numpy as np

from src.retrieval.faiss_index import FAISSIndex


def main():
    embedding_path = Path("data/embeddings/image_embeddings.npy")
    index_path = Path("data/indices/image_index.faiss")

    embeddings = np.load(embedding_path).astype("float32")

    index = FAISSIndex(embeddings.shape[1])
    index.add(embeddings)
    index.save(index_path)

    print("Embeddings:", embeddings.shape)
    print("Index:", index_path)


if __name__ == "__main__":
    main()

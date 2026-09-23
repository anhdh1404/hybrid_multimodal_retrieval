import sys

from src.encoders.clip_encoder import CLIPEncoder
from src.retrieval.faiss_index import FAISSIndex


def main():
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        raise SystemExit(
            'Usage: python scripts/run_search.py "a dog running on the beach"'
        )

    encoder = CLIPEncoder()
    index = FAISSIndex.load("data/indices/image_index.faiss")

    query_embedding = encoder.encode_text([query]).cpu().numpy()
    scores, ids = index.search(query_embedding, k=10)

    print("\nQuery:", query)
    print("\nTop results:")
    for rank, (image_id, score) in enumerate(zip(ids[0], scores[0]), start=1):
        print(f"{rank:02d}. image_index={image_id}, score={score:.4f}")


if __name__ == "__main__":
    main()

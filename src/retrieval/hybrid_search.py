from __future__ import annotations


class HybridSearchEngine:
    """Pipeline controller for Stage 1 retrieval + Stage 2 reranking.

    Stage 2 is intentionally left as an interface so BLIP-2 can be added
    after the basic CLIP + FAISS pipeline is working.
    """

    def __init__(self, bi_encoder, image_index, reranker=None):
        self.bi_encoder = bi_encoder
        self.image_index = image_index
        self.reranker = reranker

    def text_to_image_search(self, query: str, k1: int = 100, k2: int = 10):
        query_embedding = self.bi_encoder.encode_text([query]).cpu().numpy()
        scores, ids = self.image_index.search(query_embedding, k1)

        candidates = [
            {"image_id": int(image_id), "clip_score": float(score)}
            for image_id, score in zip(ids[0], scores[0])
            if image_id >= 0
        ]

        if self.reranker is None:
            return candidates[:k2]

        return self.reranker.rerank(query, candidates, top_k=k2)

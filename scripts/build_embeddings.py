from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.encoders.clip_encoder import CLIPEncoder
from src.flickr30k.dataset import Flickr30KDataset


def main():
    dataset = Flickr30KDataset("data/images", "data/results.csv")
    paths = dataset.image_paths()

    encoder = CLIPEncoder()

    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings = []

    batch_size = 16
    for start in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[start:start + batch_size]
        embeddings = encoder.encode_image_paths([str(p) for p in batch_paths])
        all_embeddings.append(embeddings.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_dir / "image_embeddings.npy", embeddings)

    with open(output_dir / "image_ids.txt", "w", encoding="utf-8") as f:
        for path in paths:
            f.write(path.name + "\n")

    print("Saved:", embeddings.shape)


if __name__ == "__main__":
    main()

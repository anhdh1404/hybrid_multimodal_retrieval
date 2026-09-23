from pathlib import Path

from src.flickr30k.dataset import Flickr30KDataset


def main():
    dataset = Flickr30KDataset(
        image_dir="data/images",
        captions_csv="data/results.csv",
    )

    print("Images:", len(dataset))
    print("CSV:", dataset.captions.shape)
    print("First images:")
    for path in dataset.image_paths()[:5]:
        print(" -", path)


if __name__ == "__main__":
    main()

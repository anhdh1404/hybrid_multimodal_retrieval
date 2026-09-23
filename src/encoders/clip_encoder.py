from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:
    """Small wrapper around Hugging Face CLIP.

    The encoder maps text and images into the same embedding space.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        return F.normalize(features, p=2, dim=-1)

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(
            images=list(images),
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_image_features(**inputs)
        return F.normalize(features, p=2, dim=-1)

    def encode_image_paths(self, paths: Sequence[str]) -> torch.Tensor:
        images = [Image.open(path).convert("RGB") for path in paths]
        try:
            return self.encode_images(images)
        finally:
            for image in images:
                image.close()

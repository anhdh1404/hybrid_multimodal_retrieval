from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    IMAGE = "image"
    CAPTION = "caption"


class EdgeType(str, Enum):
    SEMANTIC = "semantic"
    PAIRED = "paired"
    COOCCUR = "cooccur"


@dataclass
class ImageNode:
    image_id: str
    path: str


@dataclass
class CaptionNode:
    caption_id: str
    image_id: str
    text: str

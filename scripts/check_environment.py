import sys

print("Python:", sys.version)

try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
except Exception as exc:
    print("PyTorch check failed:", repr(exc))

try:
    import faiss
    print("FAISS:", faiss.__version__)
except Exception as exc:
    print("FAISS check failed:", repr(exc))

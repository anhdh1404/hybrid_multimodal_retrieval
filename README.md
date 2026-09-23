# Hybrid Multimodal Retrieval

A personal implementation of a multimodal image retrieval system based on Flickr30K.

## Goal

Build the system progressively:

Flickr30K
→ CLIP embeddings
→ FAISS retrieval
→ BLIP-2 re-ranking
→ Graph representation
→ Multi-hop graph retrieval

Supported retrieval directions:

- Text → Image
- Image → Image
- Image → Text

## Project structure

```text
hybrid_multimodal_retrieval/
├── configs/
│   ├── clip_config.yaml
│   ├── faiss_config.yaml
│   ├── blip2_config.yaml
│   └── graph_config.yaml
├── data/
│   ├── images/          # Flickr30K images - DO NOT commit to Git
│   ├── embeddings/     # generated embeddings
│   ├── indices/        # FAISS indices
│   ├── graphs/         # graph artifacts
│   └── results/        # experiment outputs
├── notebooks/
├── scripts/
│   ├── check_environment.py
│   ├── prepare_flickr30k.py
│   ├── build_embeddings.py
│   ├── build_faiss_index.py
│   └── run_search.py
├── src/
│   ├── encoders/
│   │   └── clip_encoder.py
│   ├── flickr30k/
│   │   └── dataset.py
│   ├── retrieval/
│   │   ├── faiss_index.py
│   │   └── hybrid_search.py
│   └── graph/
│       ├── schema.py
│       └── build.py
├── tests/
├── requirements.txt
├── setup.py
└── .gitignore
```

## Local vs Kaggle

Use VS Code for writing and testing the source code.

Use GitHub for version control.

Use Kaggle GPU for expensive model inference and experiments.

Typical workflow:

```text
VS Code
  ↓
git add / commit / push
  ↓
GitHub
  ↓
git clone / pull
  ↓
Kaggle GPU
  ↓
run scripts
```

## First setup

### Local

```bash
python -m venv .venv
# Windows:
.venv\Scriptsctivate

pip install -r requirements.txt
pip install -e .
```

### Kaggle

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -r requirements.txt
!pip install -e .
```

Then check:

```python
!python scripts/check_environment.py
```

## Dataset

The expected layout is:

```text
data/
├── images/
└── results.csv
```

Do not upload the full Flickr30K dataset to GitHub.

On Kaggle, attach the Flickr30K dataset and copy/symlink the required files into the expected `data/` layout.

## Development order

1. Environment check
2. Flickr30K loader
3. CLIP image/text embeddings
4. FAISS image/text indexes
5. Text → Image retrieval
6. Image → Image retrieval
7. Image → Text retrieval
8. BLIP-2 re-ranking
9. Evaluation
10. Graph construction
11. Multi-hop retrieval

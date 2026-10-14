# Embedding Atlas

**Images and words placed in one shared embedding space**, computed locally with
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) — Google's multimodal
embedding model (740M params, text + images + video + audio → one 768-dim vector space).

The artworks come from the dataset behind
[Google's t-SNE Map](https://artsexperiments.withgoogle.com/tsnemap/)
(194,370 artworks from Google Arts & Culture, arranged by visual similarity).
We sample ~1,250 of them, embed each image and ~145 words with EmbeddingGemma 2,
and jointly project everything to 2D with UMAP — so you can watch which words
drift toward *marble sculpture*, *sunset*, or *skulls and vanitas*.

## Open it

Open `tsnemap-embeddings/index.html` (any static server; it fetches `data.json` + `atlas*.jpg`).

- **Gemma 2 space / Original t-SNE** — animated morph between our joint embedding layout
  and the original visual-similarity map: same artworks, two geometric interpretations
- **Wave field** — WebGL mode replicating the original experiment's vertex shader verbatim:
  `y += (sin(t + x/1800) − sin(t + z/1600)) · amp · 500`, with terrain height from point density
  (their `heightmap.png` trick, re-derived from our own projection)
- hover for detail cards · click for the detail panel · search jumps to words or titles

## Repo layout

| path | what |
|---|---|
| `tsnemap-embeddings/` | the visualization + its data (`data.json`, `atlas*.jpg` sprite pages) |
| `pipeline/` | full pipeline (see below) |
| `data/` | working dir for the pipeline — downloaded thumbnails, raw bins, embeddings (ignored except small metadata) |

## Pipeline

```bash
uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python pipeline/build_dataset.py 2000   # parse tsnemap bins, sample, fetch metadata + thumbnails
.venv/bin/python pipeline/embed.py                # EmbeddingGemma 2: 1,247 images + 145 words → 768d
.venv/bin/python pipeline/project.py              # joint UMAP, thumbnail atlas, data.json
```

1. **build_dataset.py** — reverse-engineers the t-SNE Map data format:
   `atlasN.bin` (21-byte records: 14-char artwork ID + sprite rect), `tsne.bin`
   (int16 x / elevation / z per artwork), and their metadata API for titles + image URLs.
   Samples IDs stratified across the (x, z) plane and downloads thumbnails.
2. **embed.py** — images via `model.encode([{"image": ...}, ...])` (one document per image;
   a single dict with a list would *pool* them — the interleaving feature), words as text,
   on MPS at 140 vision soft-tokens (2.3× faster, 0.97 cosine agreement vs 280).
3. **project.py** — joint UMAP (768d → PCA-50 → 2D), then anchors each word to the weighted
   centroid of its top-8 most similar artworks (closing the cross-modal gap: unanchored,
   words form their own island outside the image cloud) with a light repulsion pass,
   dominant colors, thumbnail atlases, and precomputed neighbours.

Model weights are downloaded to the HuggingFace cache (`~/.cache/huggingface`) and are
**never** committed — see `.gitignore`.

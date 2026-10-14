# Embedding Atlas — EmbeddingGemma 2 × Google's t-SNE Map

An interactive 2D map where **artwork images and words live in one shared embedding space**,
computed with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) — Google's
multimodal embedding model (text + images → a single 768-dim vector space).

Images are sampled from the dataset behind
[Google's t-SNE Map experiment](https://artsexperiments.withgoogle.com/tsnemap/)
(194,370 artworks from Google Arts & Culture, mapped by visual similarity).

## What you're looking at

- **1,247 artworks** sampled evenly across the original t-SNE map's extent
- **165 words** (subjects, materials, moods, abstract probes) embedded as text
- Everything jointly projected to 2D with **UMAP** (cosine space, 768d → PCA-50 → 2D)

Words land *inside* the image cloud — so you can see which words drift toward
"marble sculpture" vs "sunset over the sea", and which artworks attract abstract
probes like "silence" or "the passage of time".

## Features

| | |
|---|---|
| **Gemma 2 space / Original t-SNE** | animated morph between our joint embedding layout and the original visual-similarity map — same artworks, two geometric interpretations |
| **Wave field** | WebGL mode replicating the original experiment's vertex shader verbatim: `y += (sin(t + x/1800) − sin(t + z/1600)) · amp · 500`, with terrain height from point density (their `heightmap.png` trick) |
| hover | thumbnail, title, creator, museum + the 3 nearest words |
| click | detail panel: 3 nearest words, 5 similar artworks (cosine), links |
| search | jump-to word or title |

## Pipeline (in `/tmp/tsne`, scripts included)

1. **build_dataset.py** — reverse-engineered the t-SNE Map's data format:
   - `atlasN.bin` (GCS): 21-byte records → 14-char artwork ID + atlas-sprite rect
   - `tsne.bin`: int16 (x, elevation, z) per artwork, in atlas order
   - `/freefall/api/items/<id>` → title, creator, museum, image URL
   - samples ~1,200 IDs stratified across the (x, z) plane, downloads thumbnails
2. **embed.py** — EmbeddingGemma 2 via `sentence-transformers`:
   `model.encode({"image": [PIL.Image, ...]})` for artworks,
   `model.encode([words])` for text (plain + `SearchQuery`-prefixed variants), on MPS.
3. **project.py** — joint UMAP (fit on images + words together), dominant colors,
   packs thumbnails into atlas pages, precomputes nearest neighbours, emits `data.json`.

## Files here

- `index.html` — the visualization (self-contained; fetches `data.json` + `atlas*.jpg`)
- `data.json` — positions in both layouts, metadata, neighbour indices (~300 KB)
- `atlas0..34.jpg` — 6×6 thumbnail sprite pages (160px cells)

Images © their respective museums/partners via Google Arts & Culture.

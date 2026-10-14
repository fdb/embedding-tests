#!/usr/bin/env python3
"""Embed tsnemap artwork images + concept words with EmbeddingGemma 2 (shared 768-d space)."""
import json, os, time
import numpy as np
import torch

OUT = os.environ.get("EMB_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.abspath(OUT)
items = json.load(open(f"{OUT}/items_raw.json"))
print(f"{len(items)} items")

from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer(
    "google/embeddinggemma-2",
    model_kwargs={"torch_dtype": torch.float32},
    device="mps" if torch.backends.mps.is_available() else "cpu",
)
model[0].processor.image_processor.max_soft_tokens = 140   # 2x faster, cos-agreement 0.97 vs 280
print("device:", model.device)

# ---------------- images (one document per image, NOT pooled) ----------------
imgs, valid = [], []
for it in items:
    p = f"{OUT}/thumbs/{it['id']}.jpg"
    if os.path.exists(p) and os.path.getsize(p) > 500:
        try:
            imgs.append({"image": Image.open(p).convert("RGB")})
            valid.append(it)
        except Exception:
            pass
print(f"loaded {len(imgs)} images")

B = 4
embs = []
t0 = time.time()
for i in range(0, len(imgs), B):
    e = model.encode(imgs[i:i+B], batch_size=B, show_progress_bar=False)
    embs.append(np.asarray(e))
    if (i // B) % 40 == 0 and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if (i // B) % 40 == 0:
        done, tot = min(i+B, len(imgs)), len(imgs)
        rate = (i+B) / (time.time()-t0)
        print(f"  img {done}/{tot}  ({rate:.2f}/s, eta {(tot-done)/max(rate,1e-9):.0f}s)")
img_emb = np.concatenate(embs)
assert img_emb.ndim == 2 and img_emb.shape == (len(imgs), 768), img_emb.shape
print("img_emb:", img_emb.shape, f"{time.time()-t0:.0f}s")

# ---------------- words ----------------
WORDS = [
    "portrait", "self-portrait", "a woman's face", "an old man with a beard",
    "landscape", "mountains", "forest", "the sea", "a storm at sea", "rivers and lakes",
    "sunset", "night sky", "flowers", "roses", "trees", "a garden",
    "horses", "dogs", "cattle in a field", "birds", "a hunt", "fish",
    "religious scene", "the virgin mary", "crucifixion", "saints", "angels", "a madonna and child",
    "greek mythology", "gods and goddesses", "biblical story",
    "war and battle", "soldiers", "a shipwreck", "naval battle", "ships and harbours",
    "a city view", "village houses", "a church interior", "a castle", "ruins",
    "still life with fruit", "flowers in a vase", "a banquet table", "bread and wine",
    "music", "musical instruments", "a lute player", "dancing", "a concert",
    "reading and writing", "books and letters", "a scholar in his study", "maps and globes",
    "money and coins", "a merchant", "market scene", "kitchen scene", "cooking",
    "children playing", "a family at home", "mother and child", "lovers", "a letter writer",
    "work and labour", "farmers", "blacksmith", "laundry", "fishing boats",
    "bathing", "a nude figure", "the human body", "dancers",
    "a king on horseback", "aristocrats", "peasants", "beggars", "clergy",
    "death and mourning", "skulls and vanitas", "a funeral", "martyrdom",
    "dreams and visions", "the supernatural", "ghosts and demons", "witchcraft",
    "abstract art", "geometric shapes", "color study", "doodles and sketches", "calligraphy",
    "a drawing in ink", "a woodcut print", "an etching", "a watercolour", "an oil painting in a gold frame",
    "marble sculpture", "bronze statue", "porcelain", "ivory carving", "tapestry",
    "embroidery and textile", "furniture", "a wooden chair", "ceramics and pottery", "glassware",
    "jewelry", "armour and weapons", "swords", "coins", "a crown",
    "peace and quiet", "chaos and violence", "joy and celebration", "melancholy", "solitude",
    "luxury and wealth", "poverty", "power and authority", "faith and devotion", "science and reason",
    "spring", "summer", "autumn", "winter", "rain", "fog",
    "daylight", "candlelight", "shadow", "reflection",
    "an emotion", "a memory", "time passing", "a story being told", "silence",
    "movement", "stillness", "order", "confusion", "beauty",
    "a face looking at me", "a window onto the world", "the passage of time", "the divine",
]
print(f"{len(WORDS)} words")
w_plain = np.asarray(model.encode(WORDS, normalize_embeddings=True))
w_query = np.asarray(model.encode([f"task: search result | query: {w}" for w in WORDS], normalize_embeddings=True))

img_norm = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)

for probe in ["horses", "marble sculpture", "sunset", "religious scene", "skulls and vanitas"]:
    wi = WORDS.index(probe)
    for name, wv in [("plain", w_plain), ("query", w_query)]:
        sims = img_norm @ wv[wi]
        top = [valid[j]["title"][:38] for j in np.argsort(-sims)[:3]]
        print(f"[{probe}] {name}: {top}")

np.savez_compressed(f"{OUT}/embeddings.npz", img=img_norm.astype(np.float32),
                    w_plain=w_plain.astype(np.float32), w_query=w_query.astype(np.float32))
json.dump({"words": WORDS}, open(f"{OUT}/words.json", "w"))
json.dump(valid, open(f"{OUT}/items_valid.json", "w"))
print("saved embeddings.npz")

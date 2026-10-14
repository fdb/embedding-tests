#!/usr/bin/env python3
"""Project EmbeddingGemma2 embeddings to 2D (UMAP, joint fit), pack a thumbnail atlas,
and emit data.json for the visualization."""
import json, os
import numpy as np
from PIL import Image

OUT = os.environ.get("EMB_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.abspath(OUT)
VIZ = os.environ.get("EMB_VIZ_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tsnemap-embeddings")
VIZ = os.path.abspath(VIZ)
os.makedirs(VIZ, exist_ok=True)

d = np.load(f"{OUT}/embeddings.npz")
img, w_plain, w_query = d["img"], d["w_plain"], d["w_query"]
items = json.load(open(f"{OUT}/items_valid.json"))
WORDS = json.load(open(f"{OUT}/words.json"))["words"]
N, W = len(img), len(w_plain)
print(f"{N} images, {W} words")

# ---- joint UMAP: images + words share one space ----
import umap
X = np.vstack([img, w_plain, w_query])

# --- quick PCA to 50d for speed/stability ---
Xc = X - X.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
X50 = (U[:, :50] * S[:50]).astype(np.float32)

reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="euclidean",
                    n_components=2, random_state=42)
xy = reducer.fit_transform(X50)
print("umap done", xy.shape)

# normalize: images define the frame
img_xy, wplain_xy, wquery_xy = xy[:N], xy[N:N+W], xy[N+W:]
def norm(a):
    lo, hi = a.min(0), a.max(0)
    return (a - lo) / (hi - lo + 1e-9)
frame_lo, frame_hi = img_xy.min(0), img_xy.max(0)
def in_frame(a):
    b = (a - frame_lo) / (frame_hi - frame_lo + 1e-9)
    return b
img_xy, wplain_xy, wquery_xy = in_frame(img_xy), in_frame(wplain_xy), in_frame(wquery_xy)

# report word extents so we know how far outside the image cloud words land
for nm, wxy in [("plain", wplain_xy), ("query", wquery_xy)]:
    print(f"word extents [raw {nm}]: x {wxy[:,0].min():.2f}..{wxy[:,0].max():.2f}  y {wxy[:,1].min():.2f}..{wxy[:,1].max():.2f}")

# ---- close the modality gap: anchor each word toward the weighted centroid
#      of its top-matching artworks (blend raw UMAP with the anchor) ----
wsims = w_plain @ img.T                      # W x N, cosines
K = 8
top = np.argsort(-wsims, 1)[:, :K]
wgt = np.take_along_axis(wsims, top, 1) ** 4
wgt /= wgt.sum(1, keepdims=True)
cent = (img_xy[top] * wgt[:, :, None]).sum(1)
ANCHOR = 1.0                                  # 0 = raw UMAP, 1 = fully pinned to centroid
wplain_xy = (1-ANCHOR) * wplain_xy + ANCHOR * cent
wsims_q = w_query @ img.T
top_q = np.argsort(-wsims_q, 1)[:, :K]
wgt_q = np.take_along_axis(wsims_q, top_q, 1) ** 4
wgt_q /= wgt_q.sum(1, keepdims=True)
cent_q = (img_xy[top_q] * wgt_q[:, :, None]).sum(1)
wquery_xy = (1-ANCHOR) * wquery_xy + ANCHOR * cent_q

# light label repulsion so pinned words don't stack (keeps them inside the frame)
def relax(wxy, iters=40, minD=0.045):
    W = len(wxy)
    for _ in range(iters):
        delta = wxy[:, None, :] - wxy[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-9
        push = (delta / dist[:, :, None]) * np.clip((minD - dist) / minD, 0, 1)[:, :, None] * 0.5
        np.fill_diagonal(push[..., 0], 0); np.fill_diagonal(push[..., 1], 0)
        wxy = wxy + push.sum(1)
        wxy = np.clip(wxy, 0.005, 0.995)
    return wxy
wplain_xy = relax(wplain_xy)
wquery_xy = relax(wquery_xy)
for nm, wxy in [("plain", wplain_xy), ("query", wquery_xy)]:
    print(f"word extents [anchored {nm}]: x {wxy[:,0].min():.2f}..{wxy[:,0].max():.2f}  y {wxy[:,1].min():.2f}..{wxy[:,1].max():.2f}")

# original t-SNE map coords (x, z plane) for the comparison view
ox = np.array([it["tx"] for it in items], float)
oz = np.array([it["tz"] for it in items], float)
o = np.stack([ox, oz], 1)
olo, ohi = o.min(0), o.max(0)
# pad a little for the map frame
pad = 0.02
o_xy = (o - olo) / (ohi - olo + 1e-9)
o_xy = o_xy * (1 - 2*pad) + pad

# ---- thumbnail atlas ----
CELL, GRID = 160, 6          # 6x6 = 36 per page
per_page = GRID * GRID
pages = (N + per_page - 1) // per_page
meta_items = []
for pg in range(pages):
    sheet = Image.new("RGB", (CELL*GRID, CELL*GRID), (18, 18, 22))
    for k in range(per_page):
        idx = pg * per_page + k
        if idx >= N: break
        p = f"{OUT}/thumbs/{items[idx]['id']}.jpg"
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((CELL, CELL), Image.LANCZOS)
            # center in cell on dark background
            gx, gy = (k % GRID) * CELL, (k // GRID) * CELL
            ox0 = gx + (CELL - im.width)//2; oy0 = gy + (CELL - im.height)//2
            sheet.paste(im, (ox0, oy0))
        except Exception:
            pass
    sheet.save(f"{VIZ}/atlas{pg}.jpg", quality=82)
    print(f"atlas{pg}.jpg", sheet.size)

for idx, it in enumerate(items):
    pg, k = idx // per_page, idx % per_page
    # dominant color
    try:
        im = Image.open(f"{OUT}/thumbs/{it['id']}.jpg").convert("RGB").resize((6, 6))
        col = np.asarray(im).reshape(-1, 3).mean(0).astype(int)
        hexc = "#%02x%02x%02x" % tuple(col)
    except Exception:
        hexc = "#666666"
    meta_items.append(dict(
        id=it["id"], t=it["title"][:70], c=it["creator"][:40], p=it["partner"][:40],
        x=round(float(img_xy[idx][0]), 5), y=round(float(img_xy[idx][1]), 5),
        ox=round(float(o_xy[idx][0]), 5), oy=round(float(o_xy[idx][1]), 5),
        pg=pg, k=k, col=hexc,
        ratio=None,  # filled below from thumb dims
    ))
    try:
        im = Image.open(f"{OUT}/thumbs/{it['id']}.jpg")
        meta_items[-1]["ratio"] = round(im.width / im.height, 3)
    except Exception:
        pass

# ---- precomputed neighbours (cosine) for detail panels ----
IN = img / np.linalg.norm(img, axis=1, keepdims=True)
sims = IN @ IN.T
np.fill_diagonal(sims, -1)
nn_idx = np.argsort(-sims, 1)[:, :5]          # 5 nearest artworks
wsims = IN @ w_plain.T                        # N x W
top_words = np.argsort(-wsims, 1)[:, :3]      # 3 best-matching words per artwork
for i, it in enumerate(meta_items):
    it["nn"] = [int(j) for j in nn_idx[i]]
    it["bw"] = [int(j) for j in top_words[i]]
widx = w_plain @ img.T                        # W x N
word_nn = np.argsort(-widx, 1)[:, :6]         # 6 nearest artworks per word

out = dict(
    grid=GRID, cell=CELL, pages=pages, n=N,
    items=meta_items,
    words=WORDS, word_nn=[[int(j) for j in row] for row in word_nn],
    wplain=[[round(float(a),5), round(float(b),5)] for a, b in wplain_xy],
    wquery=[[round(float(a),5), round(float(b),5)] for a, b in wquery_xy],
)
json.dump(out, open(f"{VIZ}/data.json", "w"), separators=(",", ":"))
print("data.json:", os.path.getsize(f"{VIZ}/data.json")//1024, "KB")

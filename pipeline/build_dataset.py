#!/usr/bin/env python3
"""Build a sampled dataset from Google's t-SNE Map (artsexperiments.withgoogle.com/tsnemap).

Data sources (reverse-engineered from the site's JS):
  - https://storage.googleapis.com/ct_static/data/2016-10-21/atlas/atlasN.bin
      21-byte records: 14-byte ASCII MID, byte a, uint16 x (LE, /4), uint16 y (LE, /4), uint8 w, uint8 h (/4)
      (x,y,w,h = sprite position inside the atlas texture, which is the rendered t-SNE map)
  - https://artsexperiments.withgoogle.com/freefall/data/tsne.bin
      int16 (LE) triples: tsne_x, tsne_y, tsne_z per asset, in atlas order
  - https://artsexperiments.withgoogle.com/freefall/api/items/<MID>  -> title/creator/partner/image_url
"""
import json, os, struct, sys, time, urllib.request, concurrent.futures as cf

BASE = "https://artsexperiments.withgoogle.com"
ATLAS_API = "https://storage.googleapis.com/ct_static/data/2016-10-21/atlas"
OUT = os.environ.get("EMB_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
TARGET_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
IMG_SIZE = 400

def fetch(url, binary=False, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read() if binary else json.loads(r.read().decode())
        except Exception as e:
            if t == tries - 1:
                print(f"  ! failed {url[:90]}: {e}")
                return None
            time.sleep(1.5 * (t + 1))

# ---- 1. atlas bins: MIDs in order + sprite positions in the map texture ----
mids, sprites = [], []
for i in range(100):
    data = fetch(f"{ATLAS_API}/atlas{i}.bin", binary=True)
    if data is None:
        print(f"stopped at atlas{i}.bin")
        break
    n = len(data) // 21
    for k in range(n):
        rec = data[k*21:(k+1)*21]
        mid = rec[:14].decode("ascii", "ignore").strip("\x00")
        x, y = struct.unpack("<HH", rec[15:19])
        mids.append(mid)
        sprites.append((x/4.0, y/4.0, rec[19]/4.0, rec[20]/4.0))
    print(f"atlas{i}.bin: {n} records (total {len(mids)})")

# ---- 2. tsne coordinates ----
data = fetch(f"{BASE}/freefall/data/tsne.bin", binary=True)
vals = struct.unpack("<" + "h" * (len(data)//2), data)
assert len(vals) == len(mids)*3, f"tsne mismatch: {len(vals)//3} vs {len(mids)}"
ts = [(vals[i*3], vals[i*3+1], vals[i*3+2]) for i in range(len(mids))]
xs, ys = [t[0] for t in ts], [t[1] for t in ts]
print(f"tsne.bin OK: {len(ts)} points, x {min(xs)}..{max(xs)}, y {min(ys)}..{max(ys)}")

# ---- 3. sample evenly across the original t-SNE plane (grid cells) ----
import numpy as np
P = np.array([(t[0], t[2]) for t in ts], dtype=np.float64)  # (tsne_x, tsne_z) = map plane; tsne_y is elevation
lo, hi = P.min(0), P.max(0)
side = int(np.ceil(np.sqrt(TARGET_SAMPLE)))
cells = {}
q = ((P - lo) / (hi - lo + 1e-9) * (side - 1)).astype(int)
for idx in range(len(P)):
    cells.setdefault((q[idx, 0], q[idx, 1]), []).append(idx)
rng = np.random.default_rng(42)
picks = []
for cell, idxs in cells.items():
    picks.append(int(idxs[rng.integers(len(idxs))]))
picks = picks[:TARGET_SAMPLE]
print(f"sampled {len(picks)} of {len(P)} across {side}x{side} grid")

# ---- 4. metadata + images ----
items = []
def grab(i):
    mid = mids[i]
    meta = fetch(f"{BASE}/freefall/api/items/{mid}")
    if not meta or not meta.get("image_url"):
        return None
    return dict(id=mid, tx=ts[i][0], ty=ts[i][1], tz=ts[i][2],
                ax=sprites[i][0], ay=sprites[i][1],
                title=meta.get("title") or "", creator=meta.get("creator") or "",
                partner=meta.get("partner") or "", url=meta["image_url"])

with cf.ThreadPoolExecutor(8) as ex:
    for j, it in enumerate(ex.map(grab, picks)):
        if it:
            items.append(it)
        if (j+1) % 200 == 0:
            print(f"meta {j+1}/{len(picks)}")
print(f"metadata ok for {len(items)}")

os.makedirs(f"{OUT}/thumbs", exist_ok=True)
def dl(it):
    p = f"{OUT}/thumbs/{it['id']}.jpg"
    if os.path.exists(p) and os.path.getsize(p) > 500:
        return it["id"]
    u = it["url"]
    if not u.startswith("http"):
        u = "https:" + u
    b = fetch(f"{u}=w{IMG_SIZE}", binary=True)
    if b and len(b) > 500:
        open(p, "wb").write(b)
        return it["id"]
    return None

ok = []
with cf.ThreadPoolExecutor(8) as ex:
    for j, mid in enumerate(ex.map(dl, items)):
        if mid:
            ok.append(mid)
        if (j+1) % 200 == 0:
            print(f"img {j+1}/{len(items)}")
items = [it for it in items if it["id"] in set(ok)]
print(f"images ok for {len(items)}")

json.dump(items, open(f"{OUT}/items_raw.json", "w"))
print("saved items_raw.json")

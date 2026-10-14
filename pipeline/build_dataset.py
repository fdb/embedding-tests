#!/usr/bin/env python3
"""Build a sampled dataset from Google's t-SNE Map (artsexperiments.withgoogle.com/tsnemap).

Data sources (reverse-engineered from the site's JS):
  - https://storage.googleapis.com/ct_static/data/2016-10-21/atlas/atlasN.bin
      21-byte records: 14-byte ASCII MID, byte a, uint16 x (LE, /4), uint16 y (LE, /4), uint8 w, uint8 h (/4)
      (x,y,w,h = sprite position inside the atlas texture, which is the rendered t-SNE map)
  - https://artsexperiments.withgoogle.com/freefall/data/tsne.bin
      int16 (LE) triples: tsne_x, tsne_y, tsne_z per asset, in atlas order
  - https://artsexperiments.withgoogle.com/freefall/api/items/<MID>  -> title/creator/partner/image_url

A prebuilt bundle (thumbnails + metadata + embeddings) can be served from any static host;
restore it with --from-zip <url|path> in a single download instead of ~2,500 requests.
The bundle, thumbs/ and raw bins stay out of git (see .gitignore).
"""
import json, os, re, struct, sys, time, urllib.request, zipfile, concurrent.futures as cf

BASE = "https://artsexperiments.withgoogle.com"
ATLAS_API = "https://storage.googleapis.com/ct_static/data/2016-10-21/atlas"
OUT = os.environ.get("EMB_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
IMG_SIZE = 400

import argparse
ap = argparse.ArgumentParser(description="Build the tsnemap sample dataset. "
    "By default samples + fetches from Google; --from-zip restores a prebuilt bundle in one download.")
ap.add_argument("target", nargs="?", type=int, default=1500, help="target sample size (Google fetch mode)")
ap.add_argument("--from-zip", default=os.environ.get("EMB_SAMPLE_ZIP"),
                help="path or URL to a bundle zip (thumbs/, items_raw.json, ...) — skips Google entirely")
ap.add_argument("--fresh", action="store_true", help="re-download images even if present on disk")
args = ap.parse_args()
TARGET_SAMPLE = args.target


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


# ---- 0. prebuilt bundle: one download instead of ~2,500 Google requests ----
if args.from_zip:
    zp = args.from_zip
    if re.match(r"^https?://", zp):
        t0 = time.time()
        print(f"downloading bundle from {zp} …")
        data = fetch(zp, binary=True, tries=2)
        assert data, "bundle download failed"
        zp = os.path.join(OUT, "_sample.zip")
        open(zp, "wb").write(data)
        print(f"  {len(data)/1e6:.1f} MB in {time.time()-t0:.0f}s")
    with zipfile.ZipFile(zp) as z:
        z.extractall(OUT)
    items = json.load(open(f"{OUT}/items_raw.json"))
    # prune thumbnails not referenced by items_raw.json (stale files from earlier runs)
    keep = {it["id"] + ".jpg" for it in items}
    tdir = f"{OUT}/thumbs"
    if os.path.isdir(tdir):
        stale = [f for f in os.listdir(tdir) if f.endswith(".jpg") and f not in keep]
        for f in stale:
            os.remove(os.path.join(tdir, f))
    nth = len([f for f in os.listdir(tdir) if f.endswith(".jpg")]) if os.path.isdir(tdir) else 0
    print(f"bundle restored: {len(items)} items, {nth} thumbnails -> {OUT}")
    sys.exit(0)

TARGET_SAMPLE = args.target

# ---- 0b. already-built local data: reuse unless --fresh ----
if not args.fresh and os.path.exists(f"{OUT}/items_raw.json") and os.path.isdir(f"{OUT}/thumbs"):
    items = json.load(open(f"{OUT}/items_raw.json"))
    nth = len([f for f in os.listdir(f"{OUT}/thumbs") if f.endswith(".jpg")])
    if len(items) == nth:
        print(f"reusing existing data in {OUT}: {len(items)} items with thumbnails (use --fresh to rebuild)")
        sys.exit(0)
    print(f"partial data in {OUT} ({nth}/{len(items)} thumbnails) — rebuilding from Google")

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
have = set(os.listdir(f"{OUT}/thumbs")) if os.path.isdir(f"{OUT}/thumbs") else set()
def dl(it):
    p = f"{OUT}/thumbs/{it['id']}.jpg"
    if not args.fresh and it["id"] + ".jpg" in have and os.path.getsize(p) > 500:
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

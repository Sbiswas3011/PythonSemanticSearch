# import io
# from pathlib import Path
# import rawpy
# from PIL import Image
# from tqdm import tqdm

# # SRC = Path(r"E:\Kabita Active\202616May")
# # SRC = Path(r"E:\Kabita Active")
# SRCS = [
#     Path(r"E:\\"),
#     Path(r"F:\\"),         # add as many as you like
#     Path(r"G:\\"),
#     Path(r"H:\\"),
# ]
# OUT = Path(r"C:\Users\JAYANTA\Desktop\PythonCases\jpegs")
# OUT.mkdir(exist_ok=True)
# FAILED_LOG = Path("failed.jsonl")
# DUPES_LOG = Path("dupes.jsonl")
# CLEAR_FIRST = True  

# if CLEAR_FIRST:
#     old = list(OUT.glob("*.jpg"))
#     for p in old:
#         p.unlink()
#     print(f"Deleted {len(old)} old JPEGs")  

# ROT = {3: 180, 5: 90, 6: -90}  # LibRaw flip codes -> PIL rotate degrees

# def cr2_to_jpeg(cr2: Path, out: Path):
#     with rawpy.imread(str(cr2)) as raw:
#         try:
#             thumb = raw.extract_thumb()
#             if thumb.format == rawpy.ThumbFormat.JPEG:
#                 img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
#             else:
#                 img = Image.fromarray(thumb.data)
#             if raw.sizes.flip in ROT:
#                 img = img.rotate(ROT[raw.sizes.flip], expand=True)
#         except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
#             # fallback: full develop (slower, auto-rotates)
#             img = Image.fromarray(raw.postprocess(half_size=True, use_camera_wb=True))
#     img.thumbnail((1600, 1600))
#     img.save(out, "JPEG", quality=90)

# found = {}
# for src in SRCS:
#     if not src.exists():
#         print(f"WARNING: {src} does not exist, skipping")
#         continue
#     n_before = len(found)
#     for p in src.rglob("*.CR2"):
#         found.setdefault(str(p.resolve()).lower(), p)   # same file reached twice -> kept once
#     print(f"{src}: {len(found) - n_before} new CR2 files")

# # files = sorted(SRC.rglob("*.CR2"))
# # print(f"Found {len(files)} CR2 files")

# files = sorted(found.values())
# print(f"Found {len(files)} CR2 files in total")

# seen = set()
# ok, failed, dupes = 0, [], []

# for cr2 in tqdm(files, unit="img"):
#     name = cr2.stem.lower()
#     if name in seen:                      # same filename in another folder
#         dupes.append(str(cr2))
#         continue
#     seen.add(name)

#     try:
#         cr2_to_jpeg(cr2, OUT / f"{cr2.stem}.jpg")
#         ok += 1
#     except Exception as e:
#         failed.append((str(cr2), str(e)))

# print(f"\nConverted: {ok}")
# print(f"Failed:    {len(failed)}")
# print(f"Skipped (duplicate filename): {len(dupes)}")
# for item in failed[:10]:
#     print("  FAILED", item)
# for item in dupes[:10]:
#     print("  DUPLICATE", item)


import io, json, os, re, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import rawpy
from PIL import Image, ImageOps
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None                 # trusted local files

MANIFEST = Path("manifest.jsonl")
RESULTS = Path("convert_results.jsonl")       # one line per attempt; the outcome log
OUT = Path(r"C:\Users\JAYANTA\Desktop\PythonCases\jpegs")
OUT.mkdir(exist_ok=True)

MAX_SIDE = 1600
QUALITY = 90
MIN_NONRAW_SIDE = 512                         # JPG/PNG smaller than this = icon/thumbnail
THUMB_MIN_SIDE = 1200                         # RAW preview smaller than this -> develop the RAW
WORKERS = 8                                   # test 1 vs 4 for a minute before the long run
RETRY_FAILED = False                          # True = retry failed/too_small ids
FRESH_START = False                           # True = wipe results and JPEGs, redo everything
RAW_EXTS = {".cr2", ".cr3"}
ROT = {3: 180, 5: 90, 6: -90}                 # LibRaw flip codes -> PIL rotate degrees

if FRESH_START:
    RESULTS.write_text("", encoding="utf-8")
    for p in list(OUT.glob("*.jpg")) + list(OUT.glob("*.tmp")):
        p.unlink()


def safe_name(photo_id):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", photo_id).strip(" .")
    if s != photo_id or not s:                # sanitising changed it: add a hash so ids can't merge
        s = (s or "x") + "_" + hashlib.sha1(photo_id.encode("utf-8")).hexdigest()[:6]
    return s


def flatten(img):
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.clip(np.asarray(img).astype(np.float32) / 256, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "L")
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        return bg
    return img.convert("RGB")


class TooSmall(Exception):
    pass


def load_raw(path):
    """returns (image, how) where how is 'preview' or 'developed'"""
    with rawpy.imread(path) as raw:
        img, how = None, "preview"
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(thumb.data))
                img.load()
                img = img.convert("RGB")
            else:
                img = Image.fromarray(thumb.data)
            if raw.sizes.flip in ROT:
                img = img.rotate(ROT[raw.sizes.flip], expand=True)
            if max(img.size) < THUMB_MIN_SIDE:
                img = None                    # preview too small
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            img = None
        if img is None:
            img = Image.fromarray(raw.postprocess(half_size=True, use_camera_wb=True))
            how = "developed"                 # slower; auto-rotates
    return img, how


def load_plain(path):
    with Image.open(path) as im:
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))          # JPEG only: reduced-size decode
        img = flatten(ImageOps.exif_transpose(im))
    if max(img.size) < MIN_NONRAW_SIDE:
        raise TooSmall(f"{img.size[0]}x{img.size[1]}")
    return img, "plain"


def convert_one(m):
    name = safe_name(m["id"]) + ".jpg"
    out = OUT / name
    errors, unreachable = [], 0
    candidates = [m["src"]] + [p for p in m["all_paths"] if p != m["src"]]   # fall back to other copies
    for p in candidates:
        if not os.path.exists(p):                      # drive unplugged or file moved
            errors.append({"path": p, "error": "not found"})
            unreachable += 1
            continue
        try:
            img, how = load_raw(p) if Path(p).suffix.lower() in RAW_EXTS else load_plain(p)
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            tmp = out.with_name(name + ".tmp")
            img.save(tmp, format="JPEG", quality=QUALITY)
            tmp.replace(out)                           # atomic: a .jpg is complete or absent
            return {"id": m["id"], "status": "ok", "jpg": name, "used": p, "how": how, "errors": errors}
        except TooSmall as e:
            return {"id": m["id"], "status": "too_small", "jpg": None, "used": p, "errors": [str(e)]}
        except Exception as e:
            errors.append({"path": p, "error": f"{type(e).__name__}: {e}"})
    status = "unavailable" if unreachable == len(candidates) else "failed"
    return {"id": m["id"], "status": status, "jpg": None, "used": None, "errors": errors}


# ---------- what is left to do ----------
manifest = [json.loads(l) for l in open(MANIFEST, encoding="utf-8") if l.strip()]
last = {}
if RESULTS.exists():
    for line in open(RESULTS, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            last[r["id"]] = r                          # latest attempt wins


def finished(m):
    r = last.get(m["id"])
    if r is None:
        return False
    s = r["status"]
    if s == "ok":
        return (OUT / r["jpg"]).exists()               # redo if the file went missing
    if s == "unavailable" or RETRY_FAILED:
        return False                                   # drive was unplugged: always retry
    if s == "failed":                                  # retry only if a copy we never tried has appeared
        tried = {e["path"] for e in r["errors"]}
        return all(p in tried for p in m["all_paths"])
    return True                                        # too_small stays skipped


pending = [m for m in manifest
           if m.get("status", "present") == "present" and m.get("src") and not finished(m)]
pending.sort(key=lambda m: m["src"].lower())           # sequential disk access, kinder to HDDs
print(f"{len(pending):,} to convert, {len(manifest) - len(pending):,} already done or skipped")

# ---------- convert, logging each result as it lands ----------
counts = {}
with open(RESULTS, "a", encoding="utf-8") as log, ThreadPoolExecutor(WORKERS) as ex, \
        tqdm(total=len(pending), unit="img") as bar:
    for i in range(0, len(pending), 256):              # chunks keep memory flat
        for res in ex.map(convert_one, pending[i:i + 256]):
            log.write(json.dumps(res, ensure_ascii=False) + "\n")
            log.flush()
            last[res["id"]] = res
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            bar.update(1)

tot = {}
for r in last.values():
    tot[r["status"]] = tot.get(r["status"], 0) + 1
print("this run:", counts or "nothing to do")
print("overall: ", tot, "->", RESULTS)
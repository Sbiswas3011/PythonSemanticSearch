import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import exifread

# SRC = Path(r"E:\Kabita Active\202616May")   # your real folder
SRC = Path(r"E:\Kabita Active")   # your real folder
SRCS = [
    Path(r"E:\\"),
    Path(r"F:\\"),         # add as many as you like
    Path(r"G:\\"),
    Path(r"H:\\"),
]
OUT_FILE = Path("metadata.jsonl")

if OUT_FILE.exists():
    OUT_FILE.unlink()
    print("Deleted old metadata.jsonl", flush=True)

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
}
RDF = "{%s}" % NS["rdf"]

def li_values(el):
    """Collect text from rdf:li items under an Alt/Bag/Seq element."""
    if el is None:
        return []
    return [li.text.strip() for li in el.iter(RDF + "li") if li.text and li.text.strip()]

def parse_xmp(path: Path) -> dict:
    out = {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return out
    for desc in root.iter(RDF + "Description"):
        # simple values can be stored as attributes...
        for key, name in (("xmp", "Rating"), ("xmp", "Label")):
            v = desc.get("{%s}%s" % (NS[key], name))
            if v:
                out[name.lower()] = v
        # ...or as child elements
        for name in ("Rating", "Label"):
            el = desc.find("xmp:" + name, NS)
            if el is not None and el.text:
                out[name.lower()] = el.text.strip()
        title = li_values(desc.find("dc:title", NS))
        descr = li_values(desc.find("dc:description", NS))
        kws = li_values(desc.find("dc:subject", NS))
        if title: out["title"] = title[0]
        if descr: out["description"] = descr[0]
        if kws:   out["keywords"] = kws
    return out

def ratio(tag):
    try:
        return float(tag.values[0])
    except Exception:
        return None

def gps_coord(coord_tag, ref_tag):
    try:
        d, m, s = [float(x) for x in coord_tag.values]
        v = d + m / 60 + s / 3600
        return -v if str(ref_tag) in ("S", "W") else v
    except Exception:
        return None

def parse_cr2(path: Path) -> dict:
    with open(path, "rb") as f:
        t = exifread.process_file(f, details=False)   # details=False skips thumbnails/makernote, much faster
    out = {
        "make": str(t["Image Make"]) if "Image Make" in t else None,
        "model": str(t["Image Model"]) if "Image Model" in t else None,
        "lens": str(t["EXIF LensModel"]) if "EXIF LensModel" in t else None,
        "date": str(t["EXIF DateTimeOriginal"]) if "EXIF DateTimeOriginal" in t else None,
        "iso": int(str(t["EXIF ISOSpeedRatings"])) if "EXIF ISOSpeedRatings" in t else None,
        "focal_length": ratio(t["EXIF FocalLength"]) if "EXIF FocalLength" in t else None,
        "f_number": ratio(t["EXIF FNumber"]) if "EXIF FNumber" in t else None,
        "exposure": str(t["EXIF ExposureTime"]) if "EXIF ExposureTime" in t else None,
    }
    if "GPS GPSLatitude" in t and "GPS GPSLongitude" in t:
        out["lat"] = gps_coord(t["GPS GPSLatitude"], t.get("GPS GPSLatitudeRef", ""))
        out["lon"] = gps_coord(t["GPS GPSLongitude"], t.get("GPS GPSLongitudeRef", ""))
    return {k: v for k, v in out.items() if v is not None}

# # 1. list the files first so we know the total
# print("Scanning folder...", flush=True)
# files = [p for p in SRC.rglob("*") if p.suffix.lower() in (".cr2", ".xmp")]
# total = len(files)
# print(f"Found {total} files (CR2 + XMP)", flush=True)

# 1. list the files from every source
print("Scanning folders...", flush=True)
found = {}
for src in SRCS:
    if not src.exists():
        print(f"WARNING: {src} does not exist, skipping", flush=True)
        continue
    n_before = len(found)
    for p in src.rglob("*"):
        if p.suffix.lower() in (".cr2", ".xmp"):
            found.setdefault(str(p.resolve()).lower(), p)   # same file reached twice -> kept once
    print(f"{src}: {len(found) - n_before} new files", flush=True)

files = sorted(found.values())
total = len(files)
print(f"Found {total} files (CR2 + XMP) in total", flush=True)

# 2. parse each file, merging CR2 + sidecar by folder + stem
records = {}
failed = []
t0 = time.time()

for n, p in enumerate(files, start=1):
    ext = p.suffix.lower()
    key = (str(p.parent).lower(), p.stem.lower())
    rec = records.setdefault(key, {"id": p.stem})
    try:
        if ext == ".cr2":
            rec["raw_path"] = str(p)
            rec.update(parse_cr2(p))
        else:
            rec.update(parse_xmp(p))
    except Exception as e:
        failed.append((str(p), str(e)))

    # print every 100 files, and on the last one
    if n % 100 == 0 or n == total:
        pct = n / total * 100
        rate = n / (time.time() - t0)
        eta_min = (total - n) / rate / 60
        print(f"{pct:5.1f}%  {n}/{total}  {rate:.0f} files/s  ETA {eta_min:.1f} min", flush=True)

# 3. write the result
# with open(OUT_FILE, "w", encoding="utf-8") as f:
#     for rec in records.values():
#         f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# print(f"\n{len(records)} records written")
# print(f"{len(failed)} files failed")
# for item in failed[:10]:
#     print("  FAILED", item)

cr2_records = sorted((r for r in records.values() if "raw_path" in r),
                     key=lambda r: Path(r["raw_path"]))
xmp_only = [r for r in records.values() if "raw_path" not in r]

kept, dupes, first = [], [], {}
for r in cr2_records:
    name = r["id"].lower()
    if name in first:
        dupes.append((r["raw_path"], first[name]))   # (dropped file, the one kept)
    else:
        first[name] = r["raw_path"]
        kept.append(r)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for rec in kept:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open("metadata_duplicates.txt", "w", encoding="utf-8") as f:
    for dropped, kept_path in dupes:
        f.write(f"DROPPED: {dropped}\n   KEPT:  {kept_path}\n")

print(f"\n{len(kept)} records written (one per CR2)")
print(f"{len(dupes)} duplicate filenames dropped -> metadata_duplicates.txt")
print(f"{len(xmp_only)} XMP files with no CR2 dropped")
print(f"{len(failed)} files failed")
for item in failed[:10]:
    print("  FAILED", item)
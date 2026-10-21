import os, json, time
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

SRCS = [Path("E:/"), Path("F:/"), Path("G:/"), Path("H:/"), Path("I:/")]
RAW_EXTS = {".cr2", ".cr3"}
EXTS = RAW_EXTS | {".jpg", ".jpeg", ".png"}
SKIP_DIRS = {"$recycle.bin", "system volume information"}
EXCLUDE_FOLDERS = []                  # lowercase path prefixes, e.g. r"e:\office\cummins"
MIN_NONRAW_BYTES = 100_000            # drops icons/thumbnails/cache images (non-RAW only)
DRIVE_ORDER = {"E:": 0, "F:": 1, "G:": 2, "H:": 3, "I": 4}   # preferred copy to convert

MANIFEST = Path("manifest.jsonl")     # one line per distinct photo (drives the converter)
INVENTORY = Path("inventory.jsonl")   # every file that passed the filters
DROPPED = Path("dropped.jsonl")       # every file removed by a filter, with reason
SUMMARY = Path("scan_summary.json")   # all the stats, so nothing needs printing

for f in (MANIFEST, INVENTORY, DROPPED):
    f.write_text("", encoding="utf-8")          # start every run empty

excl = tuple(x.lower() for x in EXCLUDE_FOLDERS)
found = {}                                      # resolved lowercase path -> record
per_drive = defaultdict(Counter)
dropped = Counter()
drive_secs = {}
t0 = time.time()

with open(DROPPED, "w", encoding="utf-8") as dfh:
    def drop(path, reason, size=None):
        dropped[reason] += 1
        dfh.write(json.dumps({"path": path, "reason": reason, "size": size}, ensure_ascii=False) + "\n")

    for src in SRCS:
        if not src.exists():
            print(f"{src}  not found, skipped")
            continue
        t_drive = time.time()
        with tqdm(desc=str(src), unit=" files", leave=False) as bar:
            for root, dirs, names in os.walk(src, onerror=lambda e: None):
                dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
                if excl and root.lower().startswith(excl):
                    dirs[:] = []
                    continue
                for n in names:
                    ext = os.path.splitext(n)[1].lower()
                    if ext not in EXTS:
                        continue
                    p = Path(root) / n
                    key = str(p.resolve()).lower()
                    if key in found:                      # same file reached twice
                        continue
                    try:
                        size = p.stat().st_size
                    except OSError:
                        drop(str(p), "unreadable_stat"); continue
                    if size == 0:
                        drop(str(p), "zero_bytes", size); continue
                    if ext not in RAW_EXTS and size < MIN_NONRAW_BYTES:
                        drop(str(p), "small_nonraw", size); continue
                    found[key] = {"path": str(p), "stem": p.stem.lower(), "ext": ext, "size": size}
                    per_drive[str(src)][ext] += 1
                    bar.update(1)
        drive_secs[str(src)] = round(time.time() - t_drive, 1)
        print(f"{src}  {sum(per_drive[str(src)].values()):>8,} files  ({drive_secs[str(src)]:.0f}s)")

# ---- inventory ----
with open(INVENTORY, "w", encoding="utf-8") as f:
    for rec in found.values():
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---- group by (stem, size) = photo identity ----
groups = defaultdict(list)
for rec in found.values():
    groups[(rec["stem"], rec["size"])].append(rec)

with open(MANIFEST, "w", encoding="utf-8") as f:
    for (stem, size), rs in groups.items():
        rs.sort(key=lambda r: (DRIVE_ORDER.get(Path(r["path"]).drive.upper(), 9), r["path"].lower()))
        f.write(json.dumps({
            "id": f"{stem}__{size}",
            "stem": stem,
            "ext": rs[0]["ext"],
            "size": size,
            "src": rs[0]["path"],
            "n_copies": len(rs),
            "all_paths": [r["path"] for r in rs],
        }, ensure_ascii=False) + "\n")

# ---- summary: details to file, one line to console ----
n_ids = len(groups)
n_raw = sum(1 for rs in groups.values() if rs[0]["ext"] in RAW_EXTS)
n_copies = len(found) - n_ids
elapsed = time.time() - t0

SUMMARY.write_text(json.dumps({
    "elapsed_sec": round(elapsed, 1),
    "files_kept": len(found),
    "distinct_photos": n_ids,
    "raw": n_raw,
    "jpg_png": n_ids - n_raw,
    "exact_copies_merged": n_copies,
    "old_stem_only_count": len({r["stem"] for r in found.values()}),
    "dropped_by_reason": dict(dropped),
    "per_drive": {d: dict(c) for d, c in per_drive.items()},
    "drive_scan_secs": drive_secs,
    "config": {"min_nonraw_bytes": MIN_NONRAW_BYTES, "exclude_folders": EXCLUDE_FOLDERS},
}, indent=2), encoding="utf-8")

print(f"\n{n_ids:,} photos ({n_raw:,} RAW, {n_ids - n_raw:,} JPG/PNG) "
      f"from {len(found):,} files | {n_copies:,} copies merged | {sum(dropped.values()):,} dropped | {elapsed:.0f}s")
print(f"-> {MANIFEST}, {INVENTORY}, {DROPPED}, {SUMMARY}")
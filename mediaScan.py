import os
import json
from pathlib import Path
from collections import defaultdict
import hashlib
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from itertools import count

SRCS = [Path("C:/"), Path("D:/"), Path("E:/"), Path("F:/"), Path("G:/"), Path("H:/"), Path("I:/")]
OUTPUT = Path("media_files.jsonl")

EXTS = {
    "audio": {
        ".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"
    },
    "video": {
        ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm",
        ".m4v", ".mpeg", ".mpg", ".3gp", ".mts", ".m2ts"
    },
    "documents": {
        ".pdf", ".doc", ".docx", ".docm", ".rtf", ".odt", ".txt"
    },
    "images": {
        ".jpg", ".jpeg", ".png"
    },
    "raw_images": {
        ".cr2", ".cr3"
    }
}
SKIP_DIRS = {"$recycle.bin", "system volume information"}

EXCLUDE_FOLDERS = [
    r"C:\Users\JAYANTA\Desktop\PythonCases",
    r"C:\Users\JAYANTA\Desktop\LLM_Work",
]

def scan_files():
    """Scan drives and write matching files to one JSONL file."""
    ext_to_category = {
        ext: category
        for category, extensions in EXTS.items()
        for ext in extensions
    }

    excluded = tuple(
        os.path.normcase(os.path.normpath(p)).rstrip("\\") + "\\"
        for p in EXCLUDE_FOLDERS
    )

    with OUTPUT.open("w", encoding="utf-8") as out:
        for drive in SRCS:
            if not drive.exists():
                print(f"{drive} not found, skipping")
                continue

            print(f"Scanning {drive}...")
            stack = [str(drive)]

            while stack:
                directory = stack.pop()
                normalized = os.path.normcase(
                    os.path.normpath(directory)
                )

                if any(
                    normalized == p.rstrip("\\") or normalized.startswith(p)
                    for p in excluded
                ):
                    continue

                try:
                    with os.scandir(directory) as entries:
                        for e in entries:
                            try:
                                if e.is_dir(follow_symlinks=False):
                                    if (
                                        e.name.lower() not in SKIP_DIRS
                                        and not e.is_junction()
                                    ):
                                        stack.append(e.path)

                                elif e.is_file(follow_symlinks=False):
                                    ext = Path(e.name).suffix.lower()
                                    category = ext_to_category.get(ext)

                                    if category:
                                        out.write(json.dumps({
                                            "path": os.path.normpath(e.path),
                                            "category": category,
                                            "extension": ext,
                                            "size_bytes": e.stat().st_size
                                        }, ensure_ascii=False) + "\n")

                            except OSError:
                                continue
                except OSError:
                    continue

    print(f"Scan complete. Results saved to {OUTPUT.resolve()}")


def summarize():
    """Summarize file counts and storage usage by extension."""
    stats = defaultdict(lambda: {"count": 0, "bytes": 0})
    total_count = 0
    total_bytes = 0

    with OUTPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            ext = record["extension"]
            size = record["size_bytes"]

            stats[ext]["count"] += 1
            stats[ext]["bytes"] += size
            total_count += 1
            total_bytes += size

    print("\nSummary by extension:")
    for ext, data in sorted(stats.items()):
        print(
            f"{ext:8} {data['count']:>8,} files | "
            f"{data['bytes'] / (1024 ** 3):.3f} GB"
        )

    print(
        f"\nTOTAL: {total_count:,} files | "
        f"{total_bytes / (1024 ** 3):.3f} GB"
    )
WORKERS = 8  # Try 4–8 initially
SAMPLE = 64 * 1024
_counter = count(1)
_t0 = time.time()
TOTAL = None  # set in find_duplicates after building candidates

def quick_hash(item):
    path, size = item
    n = next(_counter)
    if n % 500 == 0:
        elapsed = time.time() - _t0
        rate = n / elapsed
        eta = (TOTAL - n) / rate / 60 if TOTAL else 0
        print(f"{n:,}/{TOTAL:,}  {rate:.0f} files/s  ETA {eta:.1f} min  (last: {path})", flush=True)
    h = hashlib.blake2b(digest_size=16)
    try:
        with open(path, "rb") as f:
            h.update(f.read(SAMPLE))                  # start
            if size > 3 * SAMPLE:
                f.seek(size // 2 - SAMPLE // 2)
                h.update(f.read(SAMPLE))              # middle
                f.seek(-SAMPLE, 2)
                h.update(f.read(SAMPLE))              # end
        return path, size, h.hexdigest()
    except OSError:
        return path, size, None
    
def find_duplicates():
    by_size = defaultdict(list)
    with OUTPUT.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["size_bytes"] > 0:
                by_size[item["size_bytes"]].append(item["path"])

    candidates = [
        (p, size)
        for size, paths in by_size.items()
        if len(paths) > 1
        for p in paths
    ]
    print(f"Candidates: {len(candidates):,}", flush=True)
    global TOTAL, _t0
    TOTAL = len(candidates)
    _t0 = time.time()

    groups = defaultdict(list)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for path, size, digest in tqdm.tqdm(
            # pool.map(quick_hash, candidates, chunksize=64),
            pool.map(quick_hash, candidates),
            total=len(candidates), desc="Hashing", unit="file",
        ):
            if digest:
                groups[(size, digest)].append(path)

    dupes = [g for g in groups.values() if len(g) > 1]
    for g in dupes:
        print("\nLikely duplicates:")
        for p in g:
            print(f"  {p}")
    print(f"\nFound {len(dupes)} groups.")

if __name__ == "__main__":
    # scan_files()
    # summarize()
    find_duplicates()
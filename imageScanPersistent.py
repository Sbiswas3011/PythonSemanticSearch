import os, json, time, shutil
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

SRCS = [Path("E:/"), Path("F:/"), Path("G:/"), Path("H:/"), Path("I:/")]
RAW_EXTS = {".cr2", ".cr3"}
EXTS = RAW_EXTS | {".jpg", ".jpeg", ".png"}
SKIP_DIRS = {"$recycle.bin", "system volume information"}
EXCLUDE_FOLDERS = []                    # lowercase path prefixes
MIN_NONRAW_BYTES = 100_000
DRIVE_ORDER = {"E:": 0, "F:": 1, "G:": 2, "H:": 3, "I:": 4}
LOG_NEW = True                          # False = don't write a line per new photo

MANIFEST = Path("manifest.jsonl")            # persistent, merged every run
MANIFEST_PREV = Path("manifest.prev.jsonl")  # copy from before this run
CHANGES = Path("scan_changes.jsonl")         # persistent, append-only
INVENTORY = Path("inventory.jsonl")          # rewritten each run
DROPPED = Path("dropped.jsonl")              # rewritten each run
SUMMARY = Path("scan_summary.json")          # rewritten each run

for f in (INVENTORY, DROPPED):
    f.write_text("", encoding="utf-8")

now = time.strftime("%Y-%m-%dT%H:%M:%S")
drive_of = lambda p: os.path.splitdrive(p)[0].upper()
sort_key = lambda p: (DRIVE_ORDER.get(drive_of(p), 9), p.lower())
excl = tuple(x.lower() for x in EXCLUDE_FOLDERS)

# ---------- load previous manifest ----------
old = {}
if MANIFEST.exists():
    for line in open(MANIFEST, encoding="utf-8"):
        if line.strip():
            m = json.loads(line)
            old[m["id"]] = m
first_run = not old
old_present_on = Counter(drive_of(p) for m in old.values()
                         if m.get("status", "present") == "present" for p in m["all_paths"])


# ---------- scan ----------
def iter_files(root):
    stack = [os.path.normpath(str(root))]
    while stack:
        d = stack.pop()
        if excl and d.lower().startswith(excl):
            continue
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name.lower() in SKIP_DIRS or e.is_junction():
                            continue
                        stack.append(os.path.normpath(e.path))
                    elif e.is_file(follow_symlinks=False):
                        yield e
                except OSError:
                    continue


found = {}
per_drive = defaultdict(Counter)
dropped = Counter()
scanned = set()                         # drives whose scan results we trust this run
t0 = time.time()

with open(DROPPED, "w", encoding="utf-8") as dfh:
    def drop(path, reason, size=None):
        dropped[reason] += 1
        dfh.write(json.dumps({"path": path, "reason": reason, "size": size}, ensure_ascii=False) + "\n")

    for src in SRCS:
        if not src.exists():
            print(f"{src}  not found, left unchanged")
            continue
        t_drive = time.time()
        with tqdm(desc=str(src), unit=" files", leave=False) as bar:
            for e in iter_files(src):
                stem, ext = os.path.splitext(e.name)
                ext = ext.lower()
                if ext not in EXTS:
                    continue
                path = os.path.normpath(e.path)
                key = os.path.normcase(path)
                if key in found:
                    continue
                try:
                    size = e.stat().st_size
                except OSError:
                    drop(path, "unreadable_stat"); continue
                if size == 0:
                    drop(path, "zero_bytes", size); continue
                if ext not in RAW_EXTS and size < MIN_NONRAW_BYTES:
                    drop(path, "small_nonraw", size); continue
                found[key] = {"path": path, "stem": stem.lower(), "ext": ext, "size": size}
                per_drive[str(src)][ext] += 1
                bar.update(1)
        d, n_found = src.drive.upper(), sum(per_drive[str(src)].values())
        if n_found == 0 and old_present_on[d] > 0:
            print(f"{src}  0 files but manifest expects {old_present_on[d]:,}: treated as not mounted, left unchanged")
        else:
            scanned.add(d)
            print(f"{src}  {n_found:>8,} files  ({time.time()-t_drive:.0f}s)")

with open(INVENTORY, "w", encoding="utf-8") as f:
    for rec in found.values():
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- merge ----------
seen = defaultdict(list)                # id -> records found now
for rec in found.values():
    seen[f"{rec['stem']}__{rec['size']}"].append(rec)

merged, ev, events = {}, Counter(), []

def log(event, **kw):
    ev[event] += 1
    if first_run or (event == "new" and not LOG_NEW):
        return
    events.append({"run": now, "event": event, **kw})

for id_, rs in seen.items():
    found_paths = [r["path"] for r in rs]
    m = old.get(id_)
    if m is None:                                               # brand-new photo
        paths = sorted(found_paths, key=sort_key)
        merged[id_] = {"id": id_, "stem": rs[0]["stem"], "ext": rs[0]["ext"], "size": rs[0]["size"],
                       "src": paths[0], "n_copies": len(paths), "all_paths": paths,
                       "status": "present", "first_seen": now, "last_seen": now}
        log("new", id=id_, paths=paths)
        continue

    # known photo: scanned drives are authoritative, unscanned drives keep their old paths
    cand = set(found_paths) | {p for p in m["all_paths"] if drive_of(p) not in scanned}
    ordered = [p for p in m["all_paths"] if p in cand]          # keep old order
    ordered += sorted(cand - set(ordered), key=sort_key)        # then newly found
    src = m["src"] if m.get("src") in cand else ordered[0]      # src stays unless its file is gone
    paths = [src] + [p for p in ordered if p != src]

    added = sorted(set(paths) - set(m["all_paths"]))
    removed = sorted(set(m["all_paths"]) - set(paths))
    if m.get("status") == "missing":
        log("reappeared", id=id_, paths=paths)
    elif added or removed:
        log("paths_changed", id=id_, added=added, removed=removed)
    if src != m.get("src"):
        log("src_changed", id=id_, old=m.get("src"), new=src)

    m2 = {**m, "src": src, "all_paths": paths, "n_copies": len(paths),
          "status": "present", "first_seen": m.get("first_seen", now), "last_seen": now}
    m2.pop("missing_since", None)
    merged[id_] = m2

for id_, m in old.items():                                      # photos not found in this scan
    if id_ in seen:
        continue
    if m.get("status") == "missing":
        merged[id_] = m                                         # still missing, unchanged
        continue
    keep = [p for p in m["all_paths"] if drive_of(p) not in scanned]
    if keep:                                                    # still on a drive we didn't scan
        if len(keep) != len(m["all_paths"]):
            src = m["src"] if m["src"] in keep else keep[0]
            paths = [src] + [p for p in keep if p != src]
            log("paths_changed", id=id_, added=[], removed=sorted(set(m["all_paths"]) - set(paths)))
            if src != m["src"]:
                log("src_changed", id=id_, old=m["src"], new=src)
            merged[id_] = {**m, "src": src, "all_paths": paths, "n_copies": len(paths)}
        else:
            merged[id_] = m
    else:                                                       # gone everywhere: keep entry, old paths
        merged[id_] = {**m, "status": "missing", "missing_since": now}
        log("missing", id=id_, last_paths=m["all_paths"])

# ---------- write ----------
if MANIFEST.exists():
    shutil.copy2(MANIFEST, MANIFEST_PREV)
tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
with open(tmp, "w", encoding="utf-8") as f:
    for m in merged.values():
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
tmp.replace(MANIFEST)

if first_run or events:
    with open(CHANGES, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run": now, "event": "baseline" if first_run else "run_summary",
                            **dict(ev)}) + "\n")
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

n_present = sum(1 for m in merged.values() if m["status"] == "present")
n_missing = len(merged) - n_present
SUMMARY.write_text(json.dumps({
    "run": now, "elapsed_sec": round(time.time() - t0, 1),
    "photos_total": len(merged), "present": n_present, "missing": n_missing,
    "changes": dict(ev), "drives_scanned": sorted(scanned),
    "dropped_by_reason": dict(dropped),
    "per_drive": {d: dict(c) for d, c in per_drive.items()},
}, indent=2), encoding="utf-8")

print(f"\n{len(merged):,} photos ({n_present:,} present, {n_missing:,} missing) | "
      f"+{ev['new']:,} new | {ev['paths_changed']:,} paths changed | "
      f"{ev['missing']:,} newly missing | {ev['reappeared']:,} reappeared | {time.time()-t0:.0f}s")
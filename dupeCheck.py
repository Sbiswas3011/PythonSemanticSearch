# import json
# from collections import Counter, defaultdict
# from pathlib import Path

# groups = [json.loads(l) for l in open("dupe_groups.jsonl", encoding="utf-8")]

# # 1. group-size distribution, and how many files each kind accounts for dropping
# size_hist = Counter()
# dropped_by_kind = Counter()
# for g in groups:
#     n = len(g["files"])
#     size_hist[n if n < 10 else "10+"] += 1
#     dropped_by_kind[g["kind"]] += n - 1
# print("Group size -> number of groups")
# for k in sorted(size_hist, key=lambda x: 99 if x == "10+" else x):
#     print(f"  {k:>4} files: {size_hist[k]}")
# print("\nFiles dropped per kind:", dict(dropped_by_kind))

# # 2. how many distinct drives each group spans
# drive_span = Counter(len({Path(f["path"]).drive for f in g["files"]}) for g in groups)
# print("\nGroups by number of distinct drives:", dict(sorted(drive_span.items())))

# # 3. how many distinct folders each group spans
# folder_span = Counter(len({str(Path(f["path"]).parent) for f in g["files"]}) for g in groups)
# print("Groups by number of distinct folders:", dict(sorted(folder_span.items())))

# # 4. the 10 biggest groups
# print("\nBiggest groups:")
# for g in sorted(groups, key=lambda g: -len(g["files"]))[:10]:
#     print(f"  {g['stem']}  x{len(g['files'])}  [{g['kind']}]")

# # 5. which stems collide most, by name pattern (camera counters roll over at 9999)
# def prefix(stem):
#     return "".join(c for c in stem if not c.isdigit())[:8]
# pref = Counter(prefix(g["stem"]) for g in groups if g["kind"] == "same_ext_diff_size")
# print("\nCollision groups by filename prefix:", pref.most_common(8))

# # 6. a few sample collision groups, to eyeball
# print("\nSample collisions:")
# for g in [g for g in groups if g["kind"] == "same_ext_diff_size"][:5]:
#     print(" ", g["stem"])
#     for f in g["files"]:
#         print(f"     {f['size']:>12,}  {f['path']}")

# import json
# from collections import defaultdict
# from pathlib import Path

# DRIVE_ORDER = {"E:": 0, "F:": 1, "G:": 2, "H:": 3}    # which copy to prefer when converting
# recs = [json.loads(l) for l in open("inventory.jsonl", encoding="utf-8")]

# groups = defaultdict(list)
# for r in recs:
#     groups[(r["stem"], r["size"])].append(r)

# print("files found:             ", len(recs))
# print("distinct (stem,size):    ", len(groups))
# print("old stem-only count:     ", len({r["stem"] for r in recs}))

# with open("manifest.jsonl", "w", encoding="utf-8") as f:
#     for (stem, size), rs in groups.items():
#         rs.sort(key=lambda r: (DRIVE_ORDER.get(Path(r["path"]).drive.upper(), 9), r["path"].lower()))
#         f.write(json.dumps({
#             "id": f"{stem}__{size}",            # unique and stable across runs
#             "stem": stem,
#             "size": size,
#             "src": rs[0]["path"],               # copy to convert
#             "all_paths": [r["path"] for r in rs],
#         }, ensure_ascii=False) + "\n")
# print("wrote manifest.jsonl")

# import json
# from collections import defaultdict, Counter

# RAW = {".cr2", ".cr3"}
# recs = [json.loads(l) for l in open("inventory.jsonl", encoding="utf-8")]
# print("files by ext:", dict(Counter(r["ext"] for r in recs)))

# by_stem = defaultdict(list)
# for r in recs:
#     by_stem[r["stem"]].append(r)

# extra = Counter()          # extra (stem,size) entries beyond 1 per stem, by category
# examples = defaultdict(list)
# for stem, rs in by_stem.items():
#     sizes = {r["size"] for r in rs}
#     if len(sizes) < 2:
#         continue
#     has_raw = any(r["ext"] in RAW for r in rs)
#     has_non = any(r["ext"] not in RAW for r in rs)
#     cat = ("RAW + exports" if has_raw and has_non
#            else "RAW only, sizes differ" if has_raw
#            else "JPG/PNG only, sizes differ")
#     extra[cat] += len(sizes) - 1
#     if len(examples[cat]) < 3:
#         examples[cat].append((stem, sorted({(r["ext"], r["size"]) for r in rs})))

# print("\nExtra entries by category (these sum to 168,383 - 83,298 = 85,085):")
# for k, v in extra.most_common():
#     print(f"  {k:28} {v}")
# for k, ex in examples.items():
#     print("\n", k)
#     for stem, items in ex:
#         print("  ", stem, items)

# import json
# from collections import Counter
# from pathlib import Path

# recs = [json.loads(l) for l in open("inventory.jsonl", encoding="utf-8")]
# RAW = {".cr2", ".cr3"}
# nonraw = [r for r in recs if r["ext"] not in RAW]

# print("non-RAW files:", len(nonraw))
# for lim in (50_000, 100_000, 200_000, 500_000):
#     print(f"  under {lim//1000} KB:", sum(r["size"] < lim for r in nonraw))

# small = [r for r in nonraw if r["size"] < 200_000]
# folders = Counter("/".join(Path(r["path"]).parts[:3]) for r in small)
# print("\ntop folders holding small non-RAW files:")
# for f, n in folders.most_common(15):
#     print(f"  {n:7}  {f}")




# import json
# from collections import Counter
# from pathlib import Path

# total = 0
# n_by_copies = Counter()          # how many photos have 2, 3, 4... copies
# pairs = Counter()                # kept-drive -> skipped-drive
# skipped_drives = Counter()
# folders = Counter()
# examples = []

# for line in open("manifest.jsonl", encoding="utf-8"):
#     m = json.loads(line)
#     if m["n_copies"] < 2:
#         continue
#     skipped = m["all_paths"][1:]
#     total += len(skipped)
#     n_by_copies[m["n_copies"]] += 1
#     for p in skipped:
#         pairs[(Path(m["src"]).drive, Path(p).drive)] += 1
#         skipped_drives[Path(p).drive] += 1
#         folders[str(Path(p).parent)] += 1
#     if len(examples) < 10:
#         examples.append(m)

# print(f"exact copies skipped: {total}   (should be 23004)")
# print("photos by number of copies:", dict(sorted(n_by_copies.items())))
# print("skipped copies by drive:   ", dict(skipped_drives))

# print("\nkept-on -> skipped-on:")
# for (k, s), n in pairs.most_common():
#     print(f"  {k} -> {s}: {n}")

# print("\ntop 15 folders holding skipped copies:")
# for f, n in folders.most_common(15):
#     print(f"  {n:6}  {f}")

# print("\nexamples:")
# for m in examples:
#     print(f"  {m['id']}  ({m['size']:,} bytes, {m['n_copies']} copies)")
#     print(f"      KEPT:    {m['src']}")
#     for p in m["all_paths"][1:]:
#         print(f"      skipped: {p}")

import json
from collections import Counter
from pathlib import Path

hits = []                                   # manifest entries: kept on F:, also on G:
for line in open("manifest.jsonl", encoding="utf-8"):
    m = json.loads(line)
    if Path(m["src"]).drive.upper() != "F:":
        continue
    g_copies = [p for p in m["all_paths"][1:] if Path(p).drive.upper() == "G:"]
    if g_copies:
        hits.append((m, g_copies))

print(f"photos kept on F: with a copy on G:  {len(hits)}")
print(f"G: copies involved:                   {sum(len(g) for _, g in hits)}")

# which F folders / G folders are involved
f_fold = Counter(str(Path(m["src"]).parent) for m, _ in hits)
g_fold = Counter(str(Path(p).parent) for _, g in hits for p in g)
print("\nTop F: folders (kept copy):")
for f, n in f_fold.most_common(10):
    print(f"  {n:6}  {f}")
print("\nTop G: folders (duplicate copy):")
for f, n in g_fold.most_common(10):
    print(f"  {n:6}  {f}")

print("\nExamples:")
for m, g in hits[:10]:
    print(f"  {m['id']}")
    print(f"      F: {m['src']}")
    for p in g:
        print(f"      G: {p}")
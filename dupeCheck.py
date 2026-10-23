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
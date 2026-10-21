from pathlib import Path
import numpy as np

SHARD_DIR = Path("shards")
DIM = 768  # Replace with your actual embedding dimension

checked = 0
valid = 0
corrupted = []
invalid_shape = []
empty = []

for shard in sorted(SHARD_DIR.glob("shard_*.npz")):
    checked += 1

    try:
        with np.load(shard, allow_pickle=False) as data:
            if "ids" not in data or "vectors" not in data:
                corrupted.append((shard.name, "Missing ids or vectors"))
                continue

            ids = data["ids"]
            vectors = data["vectors"]

            if ids.ndim != 1:
                invalid_shape.append((shard.name, f"IDs shape: {ids.shape}"))
                continue

            if vectors.shape != (len(ids), DIM):
                invalid_shape.append(
                    (shard.name, f"IDs: {len(ids)}, vectors: {vectors.shape}")
                )
                continue

            if not np.issubdtype(vectors.dtype, np.number):
                invalid_shape.append((shard.name, f"Non-numeric dtype: {vectors.dtype}"))
                continue

            if not np.isfinite(vectors).all():
                invalid_shape.append((shard.name, "NaN or infinite values"))
                continue

            if len(ids) == 0:
                empty.append(shard.name)

            valid += 1

    except Exception as e:
        corrupted.append((shard.name, repr(e)))

print(f"\nChecked: {checked}")
print(f"Valid: {valid}")
print(f"Corrupted/unreadable: {len(corrupted)}")
print(f"Invalid contents/shapes: {len(invalid_shape)}")
print(f"Empty shards: {len(empty)}")

if corrupted:
    print("\n--- CORRUPTED / UNREADABLE ---")
    for name, reason in corrupted:
        print(f"{name}: {reason}")

if invalid_shape:
    print("\n--- INVALID CONTENTS ---")
    for name, reason in invalid_shape:
        print(f"{name}: {reason}")

if empty:
    print("\n--- EMPTY SHARDS ---")
    for name in empty:
        print(name)
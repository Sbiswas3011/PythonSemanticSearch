import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import argparse, inspect
import sys

MODEL_DIR = r"C:\Users\JAYANTA\Desktop\LLM_Work\models\embeddinggemma2"
JPEG_DIR = Path(r"C:\Users\JAYANTA\Desktop\PythonCases\jpegs")
VECS_FILE = "embeddings.npz"
RECORDS_FILE = "convert_results.jsonl"
CHUNK = 128
DIM = 768
BATCH = 8
IMG_KWARGS = {"image": {"max_soft_tokens": 1120}}
QUERY = "a flamingo flying alone"

SHARD_DIR = Path("shards")
SHARD_DIR.mkdir(exist_ok=True)

def generate_embeddings(model_dir=MODEL_DIR,jpeg_dir=JPEG_DIR,records_file=RECORDS_FILE,vecs_file=VECS_FILE,chunk_size=CHUNK,batch_size=BATCH,dim=DIM,shard_dir=SHARD_DIR):
    # 1. Load the model
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    model = SentenceTransformer(
        model_dir,
        config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": dtype},
    )

    print("device:", model.device, "| dtype:", dtype)

    # 2. Read conversion results and match successful JPEGs
    records, seen = [], set()

    with open(records_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            if rec.get("status") != "ok" or not rec.get("jpg"):
                continue

            if rec["id"] in seen:
                continue

            jpg = jpeg_dir / rec["jpg"]

            if jpg.is_file():
                seen.add(rec["id"])
                records.append({
                    "id": rec["id"],
                    "jpg": jpg,
                    "date": rec.get("date"),
                    "raw_path": rec.get("raw_path"),
                })

    total = len(records)
    print(total, "successful JPEGs matched")
    print("cuda available:", torch.cuda.is_available())
    print("torch version:", torch.__version__)
    print("model device:", model.device)

    # 3. Generate embeddings in resumable shards
    failed = []
    t0 = time.time()
    done_this_run = 0

    for start in range(0, total, chunk_size):
        shard = shard_dir / f"shard_{start:07d}.npz"

        # Skip shards already generated
        if shard.exists():
            continue

        chunk = records[start:start + chunk_size]

        images, ok_ids = [], []

        for r in chunk:
            try:
                with Image.open(r["jpg"]) as im:
                    images.append(im.convert("RGB"))

                ok_ids.append(r["id"])

            except Exception as e:
                failed.append((r["id"], repr(e)))

        if images:
            vecs = model.encode(
                [{"image": im} for im in images],
                batch_size=batch_size,
                truncate_dim=dim,
                normalize_embeddings=True,
                show_progress_bar=False,
                processing_kwargs=IMG_KWARGS,
            ).astype(np.float32)
        else:
            vecs = np.zeros((0, dim), dtype=np.float32)

        del images

        # Write to a temporary file before replacing the final shard
        tmp = shard.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(ok_ids), vectors=vecs)
        tmp.replace(shard)

        done_this_run += len(chunk)
        elapsed = time.time() - t0
        rate = done_this_run / elapsed if elapsed > 0 else 0
        remaining = total - (start + len(chunk))

        eta = remaining / rate / 60 if rate > 0 else 0

        print(
            f"{start + len(chunk)}/{total}  "
            f"{1 / rate:.2f} s/img  ETA {eta:.0f} min"
            if rate > 0
            else f"{start + len(chunk)}/{total}",
            flush=True,
        )

    # 4. Save failures
    if failed:
        Path("failed.json").write_text(
            json.dumps(failed, indent=1),
            encoding="utf-8",
        )
        print(len(failed), "images failed, see failed.json")

    # 5. Merge existing shards
    parts = []

    for p in sorted(shard_dir.glob("shard_*.npz")):
        with np.load(p, allow_pickle=False) as data:
            parts.append((data["ids"].copy(), data["vectors"].copy()))

    if not parts:
        print("No shards available to merge.")
        return model, records, None

    ids = np.concatenate([p[0] for p in parts])
    vecs = np.concatenate([p[1] for p in parts])

    np.savez(vecs_file, ids=ids, vectors=vecs)

    print("saved", vecs_file, vecs.shape)

    return model, records, vecs

def test_search(model_dir=MODEL_DIR,jpeg_dir=JPEG_DIR,records_file=RECORDS_FILE,vecs_file=VECS_FILE,query=QUERY):

    if vecs_file is None:
        print("No embeddings available.")
        return

    data = np.load(vecs_file)
    ids, vecs = data["ids"], data["vectors"]

    dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
    
    model = SentenceTransformer(
        model_dir,
        config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": dtype},
    )

    print("device:", model.device, "| dtype:", dtype)

    q = model.encode(
        query,
        prompt_name="SearchQuery",
        truncate_dim=DIM,
        normalize_embeddings=True,
    )

    records, seen = [], set()
    
    with open(records_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            if rec.get("status") != "ok" or not rec.get("jpg"):
                continue

            if rec["id"] in seen:
                continue

            jpg = jpeg_dir / rec["jpg"]

            if jpg.is_file():
                seen.add(rec["id"])
                records.append({
                    "id": rec["id"],
                    "jpg": jpg,
                    "date": rec.get("date"),
                    "raw_path": rec.get("raw_path"),
                })

    scores = vecs @ q

    # Match IDs to metadata by ID instead of assuming array positions match.
    record_by_id = {r["id"]: r for r in records}
    ids = np.load("embeddings.npz", allow_pickle=False)["ids"]

    for i in scores.argsort()[::-1][:5]:
        rec = record_by_id.get(str(ids[i]), {})

        print(
            round(float(scores[i]), 3),
            ids[i],
            rec.get("date"),
            rec.get("raw_path"),
        )

def merge_shards(shard_dir=SHARD_DIR, vecs_file=VECS_FILE):  
    shards = sorted(Path(shard_dir).glob("shard_*.npz"))

    if not shards:
        print("No shards found.")
        return

    ids, vecs = [], []

    for shard in shards:
        with np.load(shard, allow_pickle=False) as p:
            ids.append(p["ids"].copy())
            vecs.append(p["vectors"].copy())

    all_ids = np.concatenate(ids)
    all_vecs = np.concatenate(vecs)

    np.savez(
        vecs_file,
        ids=all_ids,
        vectors=all_vecs,
    )

    print(
        f"Saved {len(all_ids)} embeddings to {vecs_file}"
    )

def check_shards(shard_dir=SHARD_DIR, dim=DIM):
    checked = 0
    valid = 0
    corrupted = []
    invalid_shape = []
    empty = []

    for shard in sorted(Path(shard_dir).glob("shard_*.npz")):
        checked += 1

        try:
            with np.load(shard, allow_pickle=False) as data:
                if "ids" not in data or "vectors" not in data:
                    corrupted.append(
                        (shard.name, "Missing ids or vectors")
                    )
                    continue

                ids = data["ids"]
                vectors = data["vectors"]

                if ids.ndim != 1:
                    invalid_shape.append(
                        (shard.name, f"IDs shape: {ids.shape}")
                    )
                    continue

                if vectors.shape != (len(ids), dim):
                    invalid_shape.append(
                        (
                            shard.name,
                            f"IDs: {len(ids)}, vectors: {vectors.shape}",
                        )
                    )
                    continue

                if not np.issubdtype(vectors.dtype, np.number):
                    invalid_shape.append(
                        (shard.name, f"Non-numeric dtype: {vectors.dtype}")
                    )
                    continue

                if not np.isfinite(vectors).all():
                    invalid_shape.append(
                        (shard.name, "NaN or infinite values")
                    )
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

    return {
        "checked": checked,
        "valid": valid,
        "corrupted": corrupted,
        "invalid_shape": invalid_shape,
        "empty": empty,
    }

# order here = order they run in
STEPS = {
    "gen": generate_embeddings,
    "check_s": check_shards,
    "merge_s": merge_shards,
    "test": test_search,
}

def main(argv=None):
    p = argparse.ArgumentParser(description="Photo pipeline")
    for name in STEPS:
        p.add_argument(f"--{name}", action="store_true", help=f"run the {name} step")
    p.add_argument("--all", action="store_true", help="run every step in order")

    # one flag per distinct parameter name across all steps
    seen = {}
    for fn in STEPS.values():
        for pname, param in inspect.signature(fn).parameters.items():
            if pname in seen:
                continue
            seen[pname] = param
            p.add_argument(
                f"--{pname.replace('_', '-')}",
                dest=pname,
                type=type(param.default) if param.default is not inspect.Parameter.empty
                     and param.default is not None else str,
                default=None,                      # None = "not given", function default applies
                help=f"(default: {param.default})",
            )

    args = p.parse_args(argv)
    selected = [n for n in STEPS if args.all or getattr(args, n)]
    if not selected:
        p.print_help()
        return 1

    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        params = inspect.signature(STEPS[name]).parameters
        kwargs = {k: v for k, v in vars(args).items() if k in params and v is not None}
        STEPS[name](**kwargs)
    return 0

if __name__ == "__main__":
    sys.exit(main())
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import argparse, inspect
import sys
import os
import re
import subprocess
import uuid
from collections import Counter

import psycopg
from dotenv import load_dotenv
from tqdm import tqdm
from pgvector.psycopg import register_vector

load_dotenv()

MODEL_DIR = r"C:\Users\JAYANTA\Desktop\LLM_Work\models\embeddinggemma2"
JPEG_DIR = Path(r"C:\Users\JAYANTA\Desktop\PythonCases\jpegs")
VECS_FILE = "embeddings.npz"
RECORDS_FILE = "convert_results.jsonl"
DATABASE_URL = os.getenv("DATABASE_URL")
CHUNK = 128
DIM = 768
BATCH = 8
IMG_KWARGS = {"image": {"max_soft_tokens": 1120}}
QUERY = "a flamingo flying alone"

NS = uuid.UUID("6f1c0d2e-0000-4000-8000-000000000001")  # never change this once used
EXTENSIONS = ["jpg", "jpeg", "cr2", "cr3"]
IGNORE_DIRS = ["$RECYCLE.BIN", "System Volume Information"]
FIELDS = ["$Directory/$FileName", "$Model", "$SerialNumber", "$ImageUniqueID",
          "$LensSerialNumber", "$SubSecDateTimeOriginal", "$OriginalDocumentID",
          "$DocumentID", "$InstanceID"]

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

def normalize_source(source):
    source = source.replace("\\", "/")
    return source if source.endswith("/") else source + "/"

def scan_drive(source, out_file, err_file):
    """Run exiftool over the drive. Returns exiftool's exit code."""
    source = normalize_source(source)
    if not os.path.exists(source):
        raise FileNotFoundError(f"{source} not found, nothing scanned.")

    cmd = ["exiftool", "-r", "-fast"]
    for ext in EXTENSIONS:
        cmd += ["-ext", ext]
    for d in IGNORE_DIRS:
        cmd += ["-i", d]
    cmd += ["-q", "-f", "-p", "|".join(FIELDS), source]

    with open(out_file, "wb") as out, open(err_file, "wb") as err:
        return subprocess.run(cmd, stdout=out, stderr=err).returncode

def clean(value):
    """Convert ExifTool's missing-value markers into None."""
    value = value.strip().strip("*")
    return None if value in ("", "-") else value

def get_capture_key(image_uid, model, serial, timestamp):
    if image_uid:
        key = f"image:{image_uid}"

    elif model and serial and timestamp:
        key = f"camera:{model}|serial:{serial}|time:{timestamp}"

    else:
        return None

    return uuid.uuid5(uuid.NAMESPACE_URL, key)

def load_embedding_lookup(npz_path):
    if not npz_path or not os.path.isfile(npz_path):
        print(f"Embeddings file not found: {npz_path}. Skipping embedding import.")
        return {}

    data = np.load(npz_path, allow_pickle=True)

    ids = data["ids"]
    vectors = data["vectors"]

    lookup = {}

    for i, value in enumerate(ids):
        parts = str(value).rsplit("__", 1)
        if len(parts) != 2:
            continue

        filename, size = parts

        try:
            key = (filename, int(size))
        except ValueError:
            continue

        lookup[key] = np.asarray(vectors[i], dtype=np.float32)

    return lookup

def store_scan(source, vecs_file=VECS_FILE, database_url=DATABASE_URL):
    if not database_url:
        raise ValueError("Provide --database-url or set DATABASE_URL")

    embedding_lookup = load_embedding_lookup(vecs_file)

    sql = """
        INSERT INTO files (
            drive_id,
            path,
            filename,
            extension,
            size,
            mtime_ns,
            model,
            serial_number,
            image_unique_id,
            lens_serial_number,
            subsec_datetime_original,
            original_document_id,
            document_id,
            instance_id,
            capture_key,
            tracking,
            embedding,
            contrastive_embedding
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s
        )
        ON CONFLICT (path) DO UPDATE SET
            drive_id = EXCLUDED.drive_id,
            filename = EXCLUDED.filename,
            extension = EXCLUDED.extension,
            size = EXCLUDED.size,
            mtime_ns = EXCLUDED.mtime_ns,
            model = EXCLUDED.model,
            serial_number = EXCLUDED.serial_number,
            image_unique_id = EXCLUDED.image_unique_id,
            lens_serial_number = EXCLUDED.lens_serial_number,
            subsec_datetime_original = EXCLUDED.subsec_datetime_original,
            original_document_id = EXCLUDED.original_document_id,
            document_id = EXCLUDED.document_id,
            instance_id = EXCLUDED.instance_id,
            capture_key = EXCLUDED.capture_key,
            tracking = EXCLUDED.tracking,
            embedding = COALESCE(EXCLUDED.embedding, files.embedding),
            contrastive_embedding = COALESCE(EXCLUDED.contrastive_embedding, files.contrastive_embedding),
            status = 'present',
            last_seen = NOW()
    """

    batch = []
    inserted = 0
    matched = 0
    unmatched = 0
    skipped = 0
    batch_size = 1000

    with psycopg.connect(database_url) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            with open(source, "r", encoding="utf-8", errors="replace") as f:

                for line_no, line in enumerate(f, 1):
                    line = line.strip()

                    if not line:
                        continue

                    fields = line.split("|", 8)

                    if len(fields) != 9:
                        print(f"Invalid line {line_no}: expected 9 fields")
                        skipped += 1
                        continue

                    (
                        path,
                        model,
                        serial,
                        image_uid,
                        lens_serial,
                        timestamp,
                        original_document_id,
                        document_id,
                        instance_id,
                    ) = [clean(x) for x in fields]

                    try:
                        stat = os.stat(path)
                    except OSError as e:
                        print(f"Cannot stat {path}: {e}")
                        skipped += 1
                        continue

                    filename = os.path.basename(path)
                    filename, ext = os.path.splitext(filename)
                    extension = ext.lower().lstrip(".")

                    drive_id = (
                        os.path.splitdrive(path)[0]
                        .upper()
                        .rstrip(":")
                    )

                    capture_key = get_capture_key(
                        image_uid,
                        model,
                        serial,
                        timestamp
                    )

                    tracking = capture_key is not None

                    # Match using filename + actual filesystem size
                    embedding = embedding_lookup.get(
                        (filename, stat.st_size)
                    )

                    if embedding is not None:
                        embedding = np.asarray(
                            embedding, dtype=np.float32
                        )
                        matched += 1
                    else:
                        unmatched += 1

                    batch.append((
                        drive_id,
                        path,
                        filename,
                        extension,
                        stat.st_size,
                        stat.st_mtime_ns,
                        model,
                        serial,
                        image_uid,
                        lens_serial,
                        timestamp,
                        original_document_id,
                        document_id,
                        instance_id,
                        capture_key,
                        tracking,
                        embedding,
                        embedding  # Store the same embedding in both columns for now
                    ))

                    if len(batch) >= batch_size:
                        cur.executemany(sql, batch)
                        inserted += len(batch)
                        conn.commit()
                        batch.clear()

                        print(
                            f"Stored: {inserted:,} | "
                            f"Embedding matches: {matched:,} | "
                            f"No match: {unmatched:,}",
                            flush=True
                        )

            if batch:
                cur.executemany(sql, batch)
                inserted += len(batch)
                conn.commit()

    print(
        f"\nScan storage complete."
        f"\nStored: {inserted:,}"
        f"\nEmbedding matches: {matched:,}"
        f"\nNo embedding match: {unmatched:,}"
        f"\nSkipped: {skipped:,}",
        flush=True,
    )
# order here = order they run in
STEPS = {
    "scan": scan_drive,
    "store": store_scan,
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
        print(f"\n=== performing {name} ===", flush=True)
        params = inspect.signature(STEPS[name]).parameters
        kwargs = {k: v for k, v in vars(args).items() if k in params and v is not None}
        print(f"\n=== {name} completed ===", flush=True)
        STEPS[name](**kwargs)
    return 0

if __name__ == "__main__":
    sys.exit(main())
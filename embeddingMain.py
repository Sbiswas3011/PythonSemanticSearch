import json
import time
from pathlib import Path
import hashlib
import io
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
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import psycopg
from dotenv import load_dotenv
from tqdm import tqdm
from pgvector.psycopg import register_vector
import rawpy
from PIL import Image, ImageOps

load_dotenv()

MODEL_DIR = r"C:\Users\JAYANTA\Desktop\LLM_Work\models\embeddinggemma2"
JPEG_DIR = Path(r"C:\Users\JAYANTA\Desktop\PythonCases\jpegs_v2")
VECS_FILE = "embeddings_new.npz"
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

SELECT_TEMPLATE = """
WITH ranked AS (
    SELECT
        id,
        capture_key,
        path,
        filename,
        extension,
        size,
        ROW_NUMBER() OVER (
            PARTITION BY capture_key
            ORDER BY size DESC, id
        ) AS rn
    FROM files
    WHERE capture_key IS NOT NULL
      {extra_where}
)
SELECT
    id,
    capture_key::text AS capture_key,
    path,
    filename,
    extension,
    size
FROM ranked
WHERE rn = 1
ORDER BY path;
"""


def _fetch(database_url, extra_where=""):
    sql = SELECT_TEMPLATE.format(extra_where=extra_where)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    return [
        {
            "id": file_id,
            "capture_key": capture_key,
            "output_id": capture_key,
            "src": path,
            "filename": filename,
            "extension": extension,
            "size": size,
        }
        for file_id, capture_key, path, filename, extension, size in rows
    ]

def fetch_unembedded_candidates(database_url):
    """One file per capture_key (largest), only where embedding IS NULL."""
    return _fetch(database_url, "AND embedding IS NULL")

def fetch_all_candidates(database_url):
    """One file per capture_key (largest), regardless of embedding."""
    return _fetch(database_url)

def generate_embeddings(model_dir=MODEL_DIR,jpeg_dir=JPEG_DIR,database_url=DATABASE_URL,vecs_file=VECS_FILE,chunk_size=CHUNK,batch_size=BATCH,dim=DIM,shard_dir=SHARD_DIR):
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

     # 2. Fetch files from DB and match their extracted JPEGs
    candidates = fetch_all_candidates(database_url)

    records = []

    for candidate in candidates:
        jpg = jpeg_dir / f'{candidate["output_id"]}.jpg'

        if not jpg.is_file():
            continue

        records.append({
            "id": candidate["id"],
            "capture_key": candidate["capture_key"],
            "jpg": jpg,
            "raw_path": candidate["src"],
        })

    total = len(records)

    print(f"{total:,} database files matched with JPEGs")
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

OUT = Path("jpegs_v2")
RESULTS = Path("extract_results.jsonl")

MAX_SIDE = 1600
QUALITY = 90
MIN_NONRAW_SIDE = 512
THUMB_MIN_SIDE = 1200
WORKERS = 4
FETCH = "all"  # "all" or "unembedded"

RETRY_FAILED = False
FRESH_START = False

RAW_EXTS = {".cr2", ".cr3"}
ROT = {3: 180, 5: 90, 6: -90}

Image.MAX_IMAGE_PIXELS = None  # trusted local files

def safe_name(value):
    value = str(value)
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")

    if cleaned != value or not cleaned:
        cleaned = (cleaned or "x") + "_" + hashlib.sha1(
            value.encode("utf-8")
        ).hexdigest()[:6]

    return cleaned

def flatten(img):
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.clip(
            np.asarray(img).astype(np.float32) / 256,
            0,
            255,
        ).astype(np.uint8)
        img = Image.fromarray(arr, "L")

    if img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        return bg

    return img.convert("RGB")

class TooSmall(Exception):
    pass

def load_raw(path):
    """Return (image, method), where method is 'preview' or 'developed'."""
    with rawpy.imread(path) as raw:
        img = None
        how = "preview"

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
                img = None

        except (
            rawpy.LibRawNoThumbnailError,
            rawpy.LibRawUnsupportedThumbnailError,
        ):
            img = None

        if img is None:
            img = Image.fromarray(
                raw.postprocess(
                    half_size=True,
                    use_camera_wb=True,
                )
            )
            how = "developed"

    return img, how

def load_plain(path):
    with Image.open(path) as im:
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))
        img = flatten(ImageOps.exif_transpose(im))

    if max(img.size) < MIN_NONRAW_SIDE:
        raise TooSmall(f"{img.size[0]}x{img.size[1]}")

    return img, "plain"

def convert_one(candidate, out):
    path = candidate["src"]

    output_name = safe_name(candidate["output_id"]) + ".jpg"
    output_path = out / output_name

    errors = []

    if not os.path.exists(path):
        errors.append({"path": path, "error": "not found"})
        status = "unavailable"
    else:
        try:
            suffix = Path(path).suffix.lower()

            if suffix in RAW_EXTS:
                img, how = load_raw(path)
            else:
                img, how = load_plain(path)

            img.thumbnail((MAX_SIDE, MAX_SIDE))

            tmp_path = output_path.with_name(output_name + ".tmp")
            img.save(tmp_path, format="JPEG", quality=QUALITY)
            tmp_path.replace(output_path)

            return {
                "id": candidate["id"],
                "capture_key": candidate["capture_key"],
                "output_id": candidate["output_id"],
                "status": "ok",
                "jpg": output_name,
                "used": path,
                "how": how,
                "errors": errors,
            }

        except TooSmall as exc:
            status = "too_small"
            errors.append({"path": path, "error": f"too_small: {exc}"})

        except Exception as exc:
            status = "failed"
            errors.append({
                "path": path,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "id": candidate["id"],
        "capture_key": candidate["capture_key"],
        "output_id": candidate["output_id"],
        "status": status,
        "jpg": None,
        "used": None,
        "errors": errors,
    }

def load_last_results(results):
    last = {}

    if results.exists():
        with open(results, "r", encoding="utf-8") as log:
            for line in log:
                if line.strip():
                    result = json.loads(line)
                    last[result["output_id"]] = result

    return last

def finished(candidate, last, out):
    result = last.get(candidate["output_id"])

    if result is None:
        return False

    status = result["status"]

    if status == "ok":
        return bool(result.get("jpg")) and (out / result["jpg"]).exists()

    if status == "unavailable" or (
        status in ("failed", "too_small") and RETRY_FAILED
    ):
        return False

    return True

def extract_images(
    database_url=DATABASE_URL,
    fresh_start=FRESH_START,
    out=OUT,
    results=RESULTS,
    fetch=FETCH,
):
    if not database_url:
        raise ValueError(
            "Set DATABASE_URL, for example:\n"
            "DATABASE_URL=postgres://user:password@localhost:5432/photo_search"
        )

    out = Path(out)
    results = Path(results)
    out.mkdir(parents=True, exist_ok=True)

    if fresh_start:
        results.write_text("", encoding="utf-8")
        for path in list(out.glob("*.jpg")) + list(out.glob("*.tmp")):
            path.unlink()

    print("Fetching extraction candidates from PostgreSQL...", flush=True)
    if fetch == "unembedded":
        manifest = fetch_unembedded_candidates(database_url)
    elif fetch == "all":
        manifest = fetch_all_candidates(database_url)
    print(f"{len(manifest):,} candidate images selected", flush=True)

    last = load_last_results(results)

    pending = [
        candidate
        for candidate in manifest
        if not finished(candidate, last, out)
    ]
    pending.sort(key=lambda candidate: candidate["src"].lower())

    print(
        f"{len(pending):,} to extract, "
        f"{len(manifest) - len(pending):,} already done or skipped",
        flush=True,
    )

    counts = {}
    convert = partial(convert_one, out=out)

    with (
        open(results, "a", encoding="utf-8") as log,
        ThreadPoolExecutor(max_workers=WORKERS) as executor,
        tqdm(total=len(pending), unit="img") as bar,
    ):
        for start in range(0, len(pending), 256):
            chunk = pending[start:start + 256]

            for result in executor.map(convert, chunk):
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                log.flush()

                last[result["output_id"]] = result
                counts[result["status"]] = (
                    counts.get(result["status"], 0) + 1
                )
                bar.update(1)

    totals = {}
    for result in last.values():
        totals[result["status"]] = totals.get(result["status"], 0) + 1

    print("This run:", counts or "nothing to do")
    print("Overall:", totals, "->", results)

OUT = Path("jpegs_v2")
RESULTS = Path("extract_results.jsonl")

MAX_SIDE = 1600
QUALITY = 90
MIN_NONRAW_SIDE = 512
THUMB_MIN_SIDE = 1200
WORKERS = 4
FETCH = "all"  # "all" or "unembedded"

RETRY_FAILED = False
FRESH_START = False

RAW_EXTS = {".cr2", ".cr3"}
ROT = {3: 180, 5: 90, 6: -90}

Image.MAX_IMAGE_PIXELS = None  # trusted local files

# order here = order they run in
STEPS = {
    "scan": scan_drive,
    "store": store_scan,
    "extract": extract_images,
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
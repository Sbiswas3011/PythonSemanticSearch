
import os
import re
import uuid
import threading
from pathlib import Path
from collections import OrderedDict

import numpy as np
import psycopg
import torch
import torch.nn.functional as F

from flask import Flask, request, jsonify, redirect, abort
from flask_cors import CORS
from flask import send_from_directory, abort

from PIL import Image
from sentence_transformers import SentenceTransformer
from pgvector.psycopg import register_vector

import boto3
from botocore.exceptions import ClientError


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).parent

if os.name == "nt":
    MODEL_DIR = r"C:\Users\JAYANTA\Desktop\LLM_Work\models\embeddinggemma2"
else:
    MODEL_DIR = "/opt/semantic-search/model/embeddinggemma2"

DATABASE_URL = os.environ["DATABASE_URL"]

DIM = 768
EMBED_DIMS = (768, 512, 256, 128)

PAGE = 24

IMG_KWARGS = {"image": {"max_soft_tokens": 1120}}

S3_BUCKET = os.environ.get(
    "S3_BUCKET", "thumbnail-s3-semanticsearch"
)
S3_PREFIX = os.environ.get("S3_PREFIX", "jpegs").strip("/")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

JPEG_DIR = r"C:\Users\JAYANTA\Desktop\PythonCases\jpegs_v2"

# ============================================================
# INITIALIZATION
# ============================================================

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": [
            "https://sbiswas3011.github.io",
            "http://localhost:5173",
        ]
    }
})

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

model = SentenceTransformer(
    MODEL_DIR,
    device="cuda" if torch.cuda.is_available() else "cpu",
    config_kwargs={"audio_config": None},
)

s3 = boto3.client("s3", region_name=AWS_REGION)

# Model access and correction updates
model_lock = threading.Lock()
db_lock = threading.Lock()

# Cache query vectors, not complete search results.
query_cache = OrderedDict()
image_cache = OrderedDict()

CACHE_MAX = 20


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def db_health():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1


# ============================================================
# QUERY ENCODING
# ============================================================

def parse_query(q):
    """Parse expressions such as: birds + outdoors - cage."""

    parts = re.split(r"\s+([+-])\s+", q.strip()) if q.strip() else []

    terms = []
    sign = 1

    for part in parts:
        part = part.strip()

        if not part:
            continue
        elif part == "+":
            sign = 1
        elif part == "-":
            sign = -1
        else:
            terms.append((part, sign))

    return terms


def encode_query(q="", image=None, dim=0):
    """Create one normalized query vector."""

    if not 0 <= dim < len(EMBED_DIMS):
        raise ValueError("Invalid embedding dimension.")

    target_dim = EMBED_DIMS[dim]
    terms = parse_query(q)

    if image is None and not any(sign > 0 for _, sign in terms):
        raise ValueError("Query requires an image or positive text term.")

    qv = np.zeros(target_dim, dtype=np.float32)

    with model_lock:

        # Image query
        if image is not None:
            vector = model.encode(
                {"image": image},
                truncate_dim=target_dim,
                normalize_embeddings=True,
                convert_to_numpy=True,
                processing_kwargs=IMG_KWARGS,
            )

            qv += np.asarray(vector, dtype=np.float32).reshape(-1)

        # Text query
        if terms:
            texts = [text for text, _ in terms]

            vectors = model.encode(
                texts,
                prompt_name="SearchQuery",
                truncate_dim=target_dim,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            for (_, sign), vector in zip(terms, vectors):
                qv += sign * vector

    norm = np.linalg.norm(qv)

    if norm < 1e-8:
        raise ValueError("Query vectors cancel out.")

    return (qv / norm).astype(np.float32)


# ============================================================
# DATABASE SEARCH
# ============================================================

def search_db(query_vector, dim=0, offset=0, limit=PAGE):
    """
    Search PostgreSQL using cosine distance.

    contrastive_embedding is used for ranking.
    """

    target_dim = EMBED_DIMS[dim]

    # Dimensions are selected from a fixed whitelist.
    if target_dim == DIM:
        expression = "contrastive_embedding"
    else:
        expression = f"subvector(contrastive_embedding, 1, {target_dim})"

    sql = f"""
        SELECT
            id,
            capture_key,
            path,
            1 - ({expression} <=> %s) AS score
        FROM files
        WHERE contrastive_embedding IS NOT NULL
        ORDER BY {expression} <=> %s
        LIMIT %s OFFSET %s
    """

    count_sql = """
        SELECT COUNT(*)
        FROM files
        WHERE contrastive_embedding IS NOT NULL
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(count_sql)
            total = cur.fetchone()[0]

            cur.execute(
                sql,
                (query_vector, query_vector, limit, offset)
            )

            rows = cur.fetchall()

    results = []

    for file_id, capture_key, path, score in rows:
        results.append({
            "id": str(file_id),
            "capture_key": capture_key,
            "score": round(float(score), 4),
            "raw_path": path,
        })

    return total, results


# ============================================================
# QUERY CACHE
# ============================================================

def cache_query(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)

    while len(cache) > CACHE_MAX:
        cache.popitem(last=False)


def get_cached_query(cache, key):
    if key not in cache:
        return None

    cache.move_to_end(key)
    return cache[key]


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():
    try:
        return jsonify(
            status="ok",
            # database=db_health(),
        )
    except Exception:
        app.logger.exception("Health check failed")
        return jsonify(status="error"), 503


# ============================================================
# SEARCH
# ============================================================

@app.route("/search", methods=["GET", "POST"])
def search():

    q = request.values.get("q", "").strip()
    search_id = request.values.get("search_id")

    try:
        offset = max(0, int(request.values.get("offset", 0)))
        dim = int(request.values.get("dim", 0))

        if not 0 <= dim < len(EMBED_DIMS):
            return jsonify(error="Invalid dimension"), 400

        image = None

        if request.method == "POST":
            if search_id is None and "image" in request.files:
                image = Image.open(request.files["image"].stream).convert("RGB")

        if search_id is not None:

            cached = get_cached_query(image_cache, search_id)

            if cached is None:
                return jsonify(error="Image search expired."), 404

            query_vector, cached_dim = cached

            if cached_dim != dim:
                return jsonify(error="Search dimension mismatch."), 400

        else:

            if not q and image is None:
                return jsonify(total=0, results=[])

            cache_key = (q, dim) if image is None else None

            query_vector = (
                get_cached_query(query_cache, cache_key)
                if cache_key is not None
                else None
            )

            if query_vector is None:
                query_vector = encode_query(q, image=image, dim=dim)

                if cache_key is not None:
                    cache_query(query_cache, cache_key, query_vector)

            if image is not None:
                search_id = str(uuid.uuid4())

                cache_query(
                    image_cache,
                    search_id,
                    (query_vector, dim)
                )

        total, results = search_db(
            query_vector,
            dim=dim,
            offset=offset,
            limit=PAGE,
        )

        response = {
            "total": total,
            "results": results,
        }

        if search_id:
            response["search_id"] = search_id

        return jsonify(response)

    except (ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400

    except Exception:
        app.logger.exception("Search failed")
        return jsonify(error="Search failed."), 500


# ============================================================
# IMAGE DELIVERY
# ============================================================

@app.route("/img/<path:rid>")
def img(rid):

    if "/" in rid or "\\" in rid or not rid:
        abort(404)

    key = f"{S3_PREFIX}/{rid}.jpg"

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": key,
            },
            ExpiresIn=3600,
        )

        return redirect(url)

    except ClientError:
        app.logger.exception("Could not generate S3 URL")
        abort(502)

@app.route("/img/local/<path:rid>")
def img_local(rid):

    if "/" in rid or "\\" in rid or not rid:
        abort(404)

    filename = f"{rid}.jpg"

    try:
        return send_from_directory(
            JPEG_DIR,
            filename,
            mimetype="image/jpeg"
        )

    except (FileNotFoundError, OSError):
        abort(404)

# ============================================================
# CONTRASTIVE CORRECTION
# ============================================================

def fetch_embedding(file_id):
    """Fetch the current contrastive embedding and capture key."""

    sql = """
        SELECT id, capture_key, contrastive_embedding
        FROM files
        WHERE id = %s
        LIMIT 1
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (file_id,))
            return cur.fetchone()

def update_embedding(file_id, capture_key, vector):
    """
    Update the selected image.

    If capture_key exists, update all sibling files.
    Otherwise update only the selected row.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            if capture_key:
                cur.execute(
                    """
                    UPDATE files
                    SET contrastive_embedding = %s
                    WHERE capture_key = %s
                    """,
                    (vector, capture_key),
                )
            else:
                cur.execute(
                    """
                    UPDATE files
                    SET contrastive_embedding = %s
                    WHERE id = %s
                    """,
                    (vector, file_id),
                )

        conn.commit()

def contrastive_update(
    image_vector,
    wrong_text_vector,
    correct_text_vector,
    margin=0.15,
    preserve_weight=2.0,
    lr=0.01,
    steps=100,
):
    """Optimize one image embedding."""

    original = F.normalize(
        torch.as_tensor(image_vector, dtype=torch.float32),
        dim=0,
    ).detach()

    wrong = F.normalize(
        torch.as_tensor(wrong_text_vector, dtype=torch.float32),
        dim=0,
    ).detach()

    correct = F.normalize(
        torch.as_tensor(correct_text_vector, dtype=torch.float32),
        dim=0,
    ).detach()

    updated = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.Adam([updated], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        v = F.normalize(updated, dim=0)

        sim_wrong = torch.dot(v, wrong)
        sim_correct = torch.dot(v, correct)

        ranking_loss = F.relu(
            margin + sim_wrong - sim_correct
        )

        preservation_loss = 1 - torch.dot(v, original)

        loss = ranking_loss + preserve_weight * preservation_loss

        loss.backward()
        optimizer.step()

    return F.normalize(updated.detach(), dim=0).cpu().numpy()

@app.route("/correct", methods=["POST"])
def correct_image():

    data = request.get_json(silent=True) or {}

    try:
        # V2 uses database ID, not an in-memory array index.
        file_id = int(data["id"])

        margin = float(data.get("margin", 0.15))
        preserve_weight = float(data.get("preserve_weight", 2.0))
        lr = float(data.get("lr", 0.01))
        steps = int(data.get("steps", 200))
        search_dim = int(data.get("search_dim", 0))

        wrong_text = data["wrong"]
        correct_text = data["correct"]

        if not 0 <= search_dim < len(EMBED_DIMS):
            return jsonify(error="Invalid dimension"), 400

        if not wrong_text.strip() or not correct_text.strip():
            raise ValueError("Both concepts are required.")

        if "+" in wrong_text or "-" in wrong_text:
            raise ValueError("Wrong text cannot contain '+' or '-'.")

        if "+" in correct_text or "-" in correct_text:
            raise ValueError("Correct text cannot contain '+' or '-'.")

        with db_lock:

            row = fetch_embedding(file_id)

            if row is None or row[2] is None:
                return jsonify(error="Embedding not found."), 404

            _, capture_key, current_vector = row

            target_dim = EMBED_DIMS[search_dim]

            current_vector = np.asarray(
                current_vector,
                dtype=np.float32,
            )[:target_dim]

            with model_lock:
                text_vectors = model.encode(
                    [wrong_text, correct_text],
                    prompt_name="SearchQuery",
                    truncate_dim=target_dim,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )

            updated = contrastive_update(
                image_vector=current_vector,
                wrong_text_vector=text_vectors[0],
                correct_text_vector=text_vectors[1],
                margin=margin,
                preserve_weight=preserve_weight,
                lr=lr,
                steps=steps,
            )

            # IMPORTANT: restore the full 768-dimensional vector.
            # Only the selected prefix was optimized.
            if target_dim < DIM:
                full_vector = np.asarray(
                    fetch_embedding(file_id)[2],
                    dtype=np.float32,
                ).copy()

                full_vector[:target_dim] = updated

                # Renormalize the full vector.
                full_vector /= max(np.linalg.norm(full_vector), 1e-8)

                updated = full_vector

            update_embedding(file_id, capture_key, updated)

        query_cache.clear()
        image_cache.clear()

        return jsonify(
            success=True,
            id=file_id,
            wrong=wrong_text,
            correct=correct_text,
        )

    except (KeyError, ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400

    except Exception:
        app.logger.exception("Correction failed")
        return jsonify(error="Correction failed."), 500


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":

    print("Checking database...", flush=True)

    if not db_health():
        raise RuntimeError("Database unavailable.")

    print("Warming up model...", flush=True)

    with model_lock:
        model.encode(
            ["warmup"],
            prompt_name="SearchQuery",
            convert_to_numpy=True,
        )

    print("Ready.", flush=True)

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False,
        use_reloader=False,
    )

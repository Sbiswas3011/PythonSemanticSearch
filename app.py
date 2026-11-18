import json
import threading
from collections import OrderedDict
from pathlib import Path
import uuid
from flask import send_from_directory, redirect, abort
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import re, os
from flask_cors import CORS
from botocore.exceptions import ClientError
import boto3
from PIL import Image, ImageOps
 
# ---------- settings ----------
BASE = Path(__file__).parent
# MODEL_DIR = r"C:\Users\JAYANTA\Desktop\LLM_Work\models\embeddinggemma2"
# JPEG_DIR = Path(r"C:\Users\JAYANTA\Desktop\PythonCases\jpegs")
# MODEL_DIR = "/opt/semantic-search/model/embeddinggemma2"
# JPEG_DIR = BASE / "jpegs"

if os.name == "nt":   # your Windows machine
    MODEL_DIR = r"C:\Users\JAYANTA\Desktop\LLM_Work\models\embeddinggemma2"
    JPEG_DIR = Path(r"C:\Users\JAYANTA\Desktop\PythonCases\jpegs")
else:                 # Linux server / Docker
    MODEL_DIR = "/opt/semantic-search/model/embeddinggemma2"
    JPEG_DIR = BASE / "jpegs"
EMB_FILE = BASE / "embeddings.npz"
CONTRASTIVE_EMB_FILE = BASE / "contrastive_embeddings.npz"
META_FILE = BASE / "convert_results.jsonl"

DIM = 768      # must match the DIM used when embedding
PAGE = 24      # results per page
IMAGE_SIZES = {
    "tiny": 128,
    "small": 224,
    "medium": 384,
    "large": 768,
    "original": None,
}
EMBED_DIMS = (768, 512, 256, 128)
# ------------------------------

S3_BUCKET = os.environ.get(
    "S3_BUCKET", "thumbnail-s3-semanticsearch"
)
S3_PREFIX = os.environ.get("S3_PREFIX", "jpegs").strip("/")

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

s3 = boto3.client("s3", region_name=AWS_REGION)

print("Loading model...", flush=True)
model = SentenceTransformer(MODEL_DIR, device="cpu", config_kwargs={"audio_config": None})
 
data = np.load(EMB_FILE)
ids, vecs = data["ids"], data["vectors"]

if os.path.exists(CONTRASTIVE_EMB_FILE):
    dataContrastive = np.load(CONTRASTIVE_EMB_FILE)
    Cvecs = dataContrastive["vectors"]
else:
    Cvecs = vecs.copy()
    np.savez_compressed(
        CONTRASTIVE_EMB_FILE,
        ids=ids,
        vectors=Cvecs,
    )
 
meta = {}
with open(META_FILE, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        meta[r["id"]] = r
 
print(f"{len(ids)} images loaded.", flush=True)
 
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://sbiswas3011.github.io",
            "http://localhost:5173",
            "http://192.168.29.220:5173",
            "https://haven-poem-important-philosophy.trycloudflare.com"
        ]
    }
})
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
lock = threading.Lock()          # the model isn't safe to call from several threads at once
cache = OrderedDict()            # query -> (order, scores), keeps the last 20 queries
image_cache = OrderedDict()
CACHE_MAX = 20

def prepare_embeddings(embeddings):
    result = {}

    for dim in (768, 512, 256, 128):
        if dim == 768:
            result[dim] = embeddings
            continue

        v = embeddings[:, :dim].copy()

        v /= np.maximum(
            np.linalg.norm(v, axis=1, keepdims=True),
            1e-8
        )

        result[dim] = v

    return result

vecs_by_dim = prepare_embeddings(vecs)
Cvecs_by_dim = prepare_embeddings(Cvecs)

def preprocess_image(image, size="tiny"):
    image = ImageOps.exif_transpose(image).convert("RGB")

    max_size = IMAGE_SIZES.get(size, 128)

    if max_size is not None:
        image.thumbnail(
            (max_size, max_size),
            Image.Resampling.LANCZOS
        )

    return image

def rank(q="", contrastive=False, image=None, search_id=None, dim=0):

    if search_id is not None:
        if search_id not in image_cache:
            raise ValueError("Image search expired or not found.")

        image_cache.move_to_end(search_id)
        return image_cache[search_id]
    
    # Preserve existing text cache behaviour.
    cache_key = (q, contrastive, dim) if image is None else None

    if cache_key is not None and cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    # Parse text expressions.
    parts = re.split(r"\s+([+-])\s+", q.strip()) if q.strip() else []

    terms = []
    sign = 1

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part == "+":
            sign = 1
        elif part == "-":
            sign = -1
        else:
            terms.append((part, sign))

    if image is None and not any(s > 0 for _, s in terms):
        raise ValueError("Query must contain an image or positive text term.")

    # Build combined query vector.
    qv = np.zeros(EMBED_DIMS[dim], dtype=np.float32)

    # Encode uploaded image.
    if image is not None:
        image_vector = model.encode(
            {"image": image},
            truncate_dim=EMBED_DIMS[dim],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        qv += np.asarray(image_vector).reshape(-1)

    # Encode text terms.
    if terms:
        texts = [text for text, _ in terms]

        vectors = model.encode(
            texts,
            prompt_name="SearchQuery",
            truncate_dim=EMBED_DIMS[dim],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        for (_, sig), vector in zip(terms, vectors):
            qv += sig * vector

    norm = np.linalg.norm(qv)

    if norm < 1e-8:
        raise ValueError("Query vectors cancel out.")

    qv /= norm

    # Choose embedding set.
    image_vecs = Cvecs_by_dim[EMBED_DIMS[dim]] if contrastive else vecs_by_dim[EMBED_DIMS[dim]]

    scores = image_vecs @ qv
    order = scores.argsort()[::-1]

    result = (order, scores)

    # Cache text-only searches.
    if cache_key is not None:
        cache[cache_key] = result

        if len(cache) > CACHE_MAX:
            cache.popitem(last=False)

    return result


def contrastive_update(
    index,
    image_vec,
    wrong_text_vec,
    correct_text_vec,
    margin=0.15,
    preserve_weight=2.0,
    lr=0.01,
    steps=200,
    search_dim=0
):
    # Keep a normalized copy of the original embedding.
    original = F.normalize(
        torch.as_tensor(image_vec, dtype=torch.float32).detach(),
        dim=0,
    )

    wrong = F .normalize(
        torch.as_tensor(wrong_text_vec, dtype=torch.float32),
        dim=0,
    ).detach()

    correct = F.normalize(
        torch.as_tensor(correct_text_vec, dtype=torch.float32),
        dim=0,
    ).detach()

    # Only optimize the image vector, not the embedding model.
    updated = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.Adam([updated], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        v = F.normalize(updated, dim=0)

        sim_wrong = torch.dot(v, wrong)
        sim_correct = torch.dot(v, correct)

        # Encourage the correct concept to outrank the wrong one.
        ranking_loss = F.relu(
            margin + sim_wrong - sim_correct
        )

        # Preserve the original image representation.
        preservation_loss = 1 - torch.dot(v, original)

        loss = (
            ranking_loss
            + preserve_weight * preservation_loss
        )

        loss.backward()
        optimizer.step()

    result = F.normalize(updated.detach(), dim=0).cpu().numpy()

    # Update the image embedding in memory.
    # Cvecs[index] = result.astype(Cvecs.dtype, copy=False)
    Cvecs_by_dim[EMBED_DIMS[search_dim]][index] = result.astype(Cvecs_by_dim[EMBED_DIMS[search_dim]].dtype, copy=False)

    # Cached rankings are now stale.
    with lock:
        cache.clear()

    return result
 
@app.route("/health")
def health():
    return {"status": "ok"}, 200

@app.route("/search", methods=["GET", "POST"])
def search():

    search_id = request.values.get("search_id")
    q = request.args.get("q", "").strip()
    image = None

    if request.method == "POST":
        q = request.form.get("q", "").strip()

        if search_id is None and "image" in request.files:
            size = request.form.get("size", "tiny").lower()

            if size not in IMAGE_SIZES:
                return jsonify(error="Invalid image size"), 400

            try:
                image = Image.open(
                    request.files["image"].stream
                )
                image = preprocess_image(image, size)
            except Exception:
                return jsonify(error="Invalid image file"), 400
            
    try:
        offset = max(0, int(request.values.get("offset", 0)))
    except ValueError:
        offset = 0

    if not q and image is None and search_id is None:
        return jsonify(total=0, results=[])

    dim = int(request.form.get("dim", 0))

    if not 0 <= dim < len(EMBED_DIMS):
        return jsonify(error="Invalid dimension"), 400
    
    order, scores = rank(q, True, image=image, search_id=search_id, dim=dim)

    if image is not None and search_id is None:
        search_id = str(uuid.uuid4())

        image_cache[search_id] = (order, scores)

        if len(image_cache) > CACHE_MAX:
            image_cache.popitem(last=False)

    results = []
    for i in order[offset:offset + PAGE]:
        r = meta.get(str(ids[i]), {})
        results.append({
            "index": int(i),
            "id": str(ids[i]),
            "score": round(float(scores[i]), 3),
            "raw_path": r.get("used", ""),
        })

    response = {
        "total": len(order),
        "results": results,
    }

    if search_id:
        response["search_id"] = search_id

    return jsonify(response)

@app.route("/img/<path:rid>")
def img(rid):
    # Uploaded filenames are based on IDs without ".jpg".
    # Reject path traversal and unexpected characters.
    # s3 routing handling
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
        app.logger.exception("Could not generate S3 image URL")
        abort(502)

@app.route("/correct", methods=["POST"])
def correct_image():
    data = request.get_json(silent=True) or {}

    try:
        index = int(data["index"])
        margin = float(data["margin"])
        preserve_weight = float(data["preserve_weight"])
        lr = float(data["lr"])
        steps = int(data["steps"])
        wrong_text = data["wrong"]
        correct_text = data["correct"]
        search_dim = int(data.get("search_dim", 0))

        if not 0 <= search_dim < len(EMBED_DIMS):
            return jsonify(error="Invalid dimension"), 400

        if not isinstance(wrong_text, str) or not wrong_text.strip():
            raise ValueError("Invalid wrong concept")

        if not isinstance(correct_text, str) or not correct_text.strip():
            raise ValueError("Invalid correct concept")

        if "+" in wrong_text or "-" in wrong_text:
            raise ValueError("Wrong text cannot contain '+' or '-' characters")
        
        if "+" in correct_text or "-" in correct_text:
            raise ValueError("Correct text cannot contain '+' or '-' characters")

        if not 0 <= index < len(vecs):
            return jsonify(error="Invalid embedding index"), 400

        # Encode both concepts using the same query pipeline.

        text_vecs = model.encode(
            [wrong_text, correct_text],
            prompt_name="SearchQuery",
            truncate_dim=EMBED_DIMS[search_dim],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Update the selected image embedding.
        updated = contrastive_update(
            index=index,
            image_vec=vecs_by_dim[EMBED_DIMS[search_dim]][index],
            wrong_text_vec=text_vecs[0],
            correct_text_vec=text_vecs[1],
            margin=margin,
            preserve_weight=preserve_weight,
            lr=lr,
            steps=steps,
            search_dim=search_dim
        )

        # Invalidate cached search rankings.
        with lock:
            cache.clear()
            image_cache.clear()

        return jsonify(
            success=True,
            index=index,
            wrong=wrong_text,
            correct=correct_text,
        )

    except (KeyError, ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    # 127.0.0.1 = reachable only from this computer
    # use_reloader=False so the model isn't loaded twice
    # app.run(host="127.0.0.1", port=8080, debug=False, use_reloader=False)
    app.run(host="0.0.0.0", port=8000, debug=False, ssl_context='adhoc')
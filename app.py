import json
import threading
from collections import OrderedDict
from pathlib import Path
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
# ------------------------------

S3_BUCKET = os.environ.get(
    "S3_BUCKET", "thumbnail-s3-semanticsearch"
)
S3_PREFIX = os.environ.get("S3_PREFIX", "jpegs").strip("/")

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

s3 = boto3.client("s3", region_name=AWS_REGION)

print("Loading model...", flush=True)
model = SentenceTransformer(MODEL_DIR, device="cpu", config_kwargs={"audio_config": None, "vision_config": None})
 
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
            "https://sbiswas3011.github.io/PythonSemanticSearch",
            "http://localhost:5173"
        ]
    }
})
lock = threading.Lock()          # the model isn't safe to call from several threads at once
cache = OrderedDict()            # query -> (order, scores), keeps the last 20 queries

def rank(q, contrastive=False):
    # Separate cached results for original and contrastive embeddings.
    cache_key = (q, contrastive)

    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    # Parse expressions like:
    # human + outdoors - cage + wildlife - indoors
    parts = re.split(r"\s+([+-])\s+", q)

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

    if not terms or not any(s > 0 for _, s in terms):
        raise ValueError("Query must contain at least one positive term.")

    # Encode every phrase separately.
    with lock:
        texts = [text for text, _ in terms]
        vectors = model.encode(
            texts,
            prompt_name="SearchQuery",
            truncate_dim=DIM,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    # Build the combined query vector.
    qv = np.zeros(DIM, dtype=np.float32)

    for (_, sig), vector in zip(terms, vectors):
        qv += sig * vector

    norm = np.linalg.norm(qv)
    if norm < 1e-8:
        raise ValueError("Positive and negative terms cancel out.")

    qv /= norm

    # Choose the embedding set.
    image_vecs = Cvecs if contrastive else vecs

    scores = image_vecs @ qv
    order = scores.argsort()[::-1]

    cache[cache_key] = (order, scores)

    if len(cache) > 20:
        cache.popitem(last=False)

    return cache[cache_key]

def contrastive_update(
    index,
    image_vec,
    wrong_text_vec,
    correct_text_vec,
    margin=0.15,
    preserve_weight=2.0,
    lr=0.01,
    steps=200,
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
    Cvecs[index] = result.astype(Cvecs.dtype, copy=False)

    # Cached rankings are now stale.
    with lock:
        cache.clear()

    return result
 
 
@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")
 
@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        offset = 0
    if not q:
        return jsonify(total=0, results=[])
 
    order, scores = rank(q,True)
    results = []
    for i in order[offset:offset + PAGE]:
        r = meta.get(str(ids[i]), {})
        results.append({
            "index": int(i), 
            "id": str(ids[i]),
            "score": round(float(scores[i]), 3),
            "raw_path": r.get("used", ""),
        })
    return jsonify(total=len(order), results=results)

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
        with lock:
            text_vecs = model.encode(
                [wrong_text, correct_text],
                prompt_name="SearchQuery",
                truncate_dim=DIM,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        # Update the selected image embedding.
        updated = contrastive_update(
            index=index,
            image_vec=vecs[index],
            wrong_text_vec=text_vecs[0],
            correct_text_vec=text_vecs[1],
            margin=margin,
            preserve_weight=preserve_weight,
            lr=lr,
            steps=steps
        )

        # Invalidate cached search rankings.
        with lock:
            cache.clear()

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
    app.run(host="0.0.0.0", port=8000, debug=False)
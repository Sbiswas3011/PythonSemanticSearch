import json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_DIR = r"C:\Users\JAYANTA\Desktop\LLM_Work\models\embeddinggemma2"
JPEG_DIR = Path(r"C:\Users\JAYANTA\Desktop\PythonCases\jpegs")
CHUNK = 128
DIM = 768
BATCH = 8
IMG_KWARGS = {"image": {"max_soft_tokens": 1120}}
SHARD_DIR = Path("shards")
SHARD_DIR.mkdir(exist_ok=True)

# 1. load the model
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
model = SentenceTransformer(MODEL_DIR,
                            config_kwargs={"audio_config": None},
                            model_kwargs={"torch_dtype": dtype})

print("device:", model.device, "| dtype:", dtype)

# 2. Read conversion results and match successful JPEGs
records, seen = [], set()

with open("convert_results.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)

        if rec.get("status") != "ok" or not rec.get("jpg"):
            continue

        if rec["id"] in seen:
            continue

        jpg = JPEG_DIR / rec["jpg"]

        if jpg.is_file():
            seen.add(rec["id"])
            records.append({
                "id": rec["id"],
                "jpg": jpg,
            })

total = len(records)
print(total, "successful JPEGs matched")

print("cuda available:", torch.cuda.is_available())
print("torch version:", torch.__version__)
print("model device:", model.device)
# print("cpu threads:", torch.get_num_threads())

failed = []
t0, done_this_run = time.time(), 0
for start in range(0, total, CHUNK):
    shard = SHARD_DIR / f"shard_{start:07d}.npz"
    if shard.exists():                       # already done in a previous run
        # try:
        #     with np.load(shard, allow_pickle=False) as data:
        #         ids = data["ids"]
        #         vectors = data["vectors"]
        #         if vectors.shape == (len(ids), DIM):
        #             continue
        # except (OSError, ValueError, EOFError):
        #     pass  # Invalid shard: regenerate it
        continue
    chunk = records[start:start + CHUNK]

    images, ok_ids = [], []
    for r in chunk:
        try:
            with Image.open(r["jpg"]) as im:
                images.append(im.convert("RGB"))
            ok_ids.append(r["id"])
        except Exception as e:
            failed.append((r["id"], repr(e)))

    if images:
        vecs = model.encode([{"image": im} for im in images], batch_size=BATCH,
                            truncate_dim=DIM, normalize_embeddings=True,
                            show_progress_bar=False, processing_kwargs=IMG_KWARGS)
        vecs = vecs.astype(np.float32)
    else:
        vecs = np.zeros((0, DIM), np.float32)
    del images

    tmp = shard.with_suffix(".tmp.npz")      # write then rename = no half-written shards
    np.savez(tmp, ids=np.array(ok_ids), vectors=vecs)
    tmp.replace(shard)

    done_this_run += len(chunk)
    rate = done_this_run / (time.time() - t0)
    remaining = total - (start + len(chunk))
    print(f"{start + len(chunk)}/{total}  {1/rate:.2f} s/img  ETA {remaining/rate/60:.0f} min", flush=True)

if failed:
    Path("failed.json").write_text(json.dumps(failed, indent=1))
    print(len(failed), "images failed, see failed.json")

# --- merge whatever shards exist (run this anytime, even mid-run in another process) ---
parts = [np.load(p) for p in sorted(SHARD_DIR.glob("shard_*.npz"))]
ids  = np.concatenate([p["ids"] for p in parts])
vecs = np.concatenate([p["vectors"] for p in parts])
np.savez("embeddings.npz", ids=ids, vectors=vecs)
print("saved embeddings.npz", vecs.shape)

# all_vecs = []
# t0 = time.time()
# for start in range(0, total, CHUNK):
#     chunk = records[start:start + CHUNK]
#     images = []
#     for r in chunk:
#         with Image.open(JPEG_DIR / (r["id"] + ".jpg")) as im:
#             images.append(im.convert("RGB"))     # convert() makes a copy, so the file can close

#     vecs = model.encode(images, batch_size=BATCH, truncate_dim=DIM,
#                         normalize_embeddings=True, show_progress_bar=False, processing_kwargs=IMG_KWARGS)
#     all_vecs.append(vecs.astype(np.float32))
#     del images                                   # free the RAM before the next chunk

#     done = start + len(chunk)
#     rate = done / (time.time() - t0)
#     print(f"{done / total * 100:5.1f}%  {done}/{total}  {1 / rate:.2f} sec/img  "
#           f"ETA {(total - done) / rate / 60:.1f} min", flush=True)

# # 4. save to one file
# vecs = np.concatenate(all_vecs)
# ids = np.array([r["id"] for r in records])
# np.savez("embeddings.npz", ids=ids, vectors=vecs)
# print("saved embeddings.npz", vecs.shape)

# 5. try a search
q = model.encode("a flamingo flying alone", prompt_name="SearchQuery",
                 truncate_dim=DIM, normalize_embeddings=True)
scores = vecs @ q
for i in scores.argsort()[::-1][:5]:
    print(round(float(scores[i]), 3), records[i]["id"], records[i]["date"], records[i]["raw_path"])
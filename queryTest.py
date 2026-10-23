import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_DIR = r"C:\Users\JAYANTA\Desktop\LLM_Work\models\embeddinggemma2"
DIM = 256

# load model (text only is enough for queries, so skip vision and audio to load faster)
model = SentenceTransformer(MODEL_DIR,
                            config_kwargs={"audio_config": None})

# load saved embeddings
data = np.load("embeddings.npz")
ids, vecs = data["ids"], data["vectors"]

# load metadata, keyed by id
meta = {}
for line in open("metadata.jsonl", encoding="utf-8"):
    r = json.loads(line)
    meta[r["id"]] = r

print(f"{len(ids)} images loaded. Type a query, or 'q' to quit.")

while True:
    query = input("\nsearch> ").strip()
    if query.lower() in ("q", "quit", "exit"):
        break
    if not query:
        continue

    print("Encoding...")
    q = model.encode(query, prompt_name="SearchQuery",
                     truncate_dim=DIM, normalize_embeddings=True)
    print("Encoding Complete...")

    # print("Scoring...")
    scores = vecs @ q
    # print("Scoring Complete...")

    for i in scores.argsort()[::-1][:5]:
        r = meta[str(ids[i])]
        print(f"{scores[i]:.3f}  {r['id']}  {r.get('date', '')}  {r['raw_path']} ")
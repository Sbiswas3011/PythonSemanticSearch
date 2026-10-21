import numpy as np
from pathlib import Path

shards = sorted(Path("shards").glob("shard_*.npz"))

with np.load(shards[0]) as p:
    ids = [p["ids"]]
    vecs = [p["vectors"]]

for shard in shards[1:]:
    with np.load(shard) as p:
        ids.append(p["ids"])
        vecs.append(p["vectors"])

np.savez("embeddings.npz",
         ids=np.concatenate(ids),
         vectors=np.concatenate(vecs))

print(f"Saved {sum(len(x) for x in ids)} embeddings to embeddings.npz")
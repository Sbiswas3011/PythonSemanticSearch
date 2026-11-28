
import numpy as np
import umap
import hdbscan
import plotly.express as px

from sklearn.preprocessing import normalize


NPZ_FILE = "embeddings.npz"

MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 5

# Start small; increase once performance is acceptable.
MAX_SAMPLES = 20000


# -----------------------------
# Load embeddings
# -----------------------------

data = np.load(NPZ_FILE, allow_pickle=False)

ids = data["ids"]
vectors = data["vectors"].astype(np.float32)

valid = np.isfinite(vectors).all(axis=1)

ids = ids[valid]
vectors = vectors[valid]

vectors = normalize(vectors)

if MAX_SAMPLES and len(vectors) > MAX_SAMPLES:
    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(vectors),
        MAX_SAMPLES,
        replace=False
    )

    ids = ids[indices]
    vectors = vectors[indices]

print(f"Loaded {len(vectors):,} embeddings")


# -----------------------------
# HDBSCAN clustering (768D)
# -----------------------------

print("Running HDBSCAN...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    cluster_selection_method="eom",
    core_dist_n_jobs=-1,
)

labels = clusterer.fit_predict(vectors)

print(
    f"Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}"
)
print(f"Noise: {np.sum(labels == -1):,}")


# -----------------------------
# UMAP dimensionality reduction
# -----------------------------

print("Reducing to 3D...")

reducer = umap.UMAP(
    n_components=3,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    low_memory=True,
)

coords = reducer.fit_transform(vectors)


# -----------------------------
# Interactive visualization
# -----------------------------

print("Opening visualization...")

fig = px.scatter_3d(
    x=coords[:, 0],
    y=coords[:, 1],
    z=coords[:, 2],
    color=labels.astype(str),
    hover_name=ids.astype(str),
    hover_data={
        "Cluster": labels,
    },
    labels={
        "color": "Cluster",
        "x": "UMAP 1",
        "y": "UMAP 2",
        "z": "UMAP 3",
    },
    title="Semantic Image Embedding Clusters",
)

fig.update_traces(
    marker=dict(
        size=2,
        opacity=0.75,
    )
)

fig.update_layout(
    scene=dict(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        zaxis_title="UMAP 3",
    ),
    height=900,
)

fig.show()

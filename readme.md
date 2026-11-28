# ⚠️ DOCUMENTATION NOTICE

# **THIS README IS AI-GENERATED AND MAY CONTAIN INCORRECT OR OUTDATED INFORMATION.**

**The project is actively undergoing revision. Some implementation details, commands, configurations, and API descriptions may be inaccurate. Do not rely on this documentation as an authoritative reference until it has been reviewed and updated.**

# Semantic Photo Search

A semantic image-search system for indexing, embedding, and searching large personal photo collections. It supports text-to-image search, image-to-image search, and contrastive refinement of individual image embeddings.

The data pipeline is designed to handle large collections of mixed image formats, including JPEG and Canon RAW files (`.CR2`, `.CR3`), while keeping track of original files, capture relationships, extracted previews, and embedding status.

This write up is AI generated and is incorrect in a few places; this is undergoing revision

## Features

- Recursive photo inventory and EXIF metadata extraction
- Persistent file identifiers and capture grouping
- PostgreSQL as the source of truth for file metadata and processing state
- Resumable JPEG extraction from original images and RAW files
- 768-dimensional image embeddings using EmbeddingGemma2
- Sharded embedding generation to support interrupted runs
- Semantic text and image search
- Contrastive refinement using positive and negative concepts
- S3-backed image delivery through presigned URLs
- Flask API for the search frontend

## Architecture

```text
Original photo directories
          |
          v
   File inventory scan
          |
          v
    EXIF extraction
          |
          v
 PostgreSQL: files
          |
          v
    JPEG extraction
          |
          v
   JPEG representations
          |
          v
 EmbeddingGemma2 (768D)
          |
          v
  Embedding shards / DB
          |
          v
    Search API
          |
          +------> React frontend
          |
          +------> S3 image storage
```

PostgreSQL stores the inventory and processing metadata. JPEGs are generated as smaller representations of the original files so that embedding generation and image delivery do not require repeatedly decoding large RAW files.

## Technology

| Component | Technology |
|---|---|
| Language | Python |
| Database | PostgreSQL |
| Vector search | pgvector |
| Image embeddings | EmbeddingGemma2 |
| Embedding framework | Sentence Transformers |
| RAW processing | rawpy |
| Image processing | Pillow |
| Image storage | Amazon S3 |
| API | Flask |
| ML / optimization | PyTorch |
| Frontend | React + Vite |
| Deployment | Docker, AWS EC2 |

---

# Data pipeline

## 1. File inventory and metadata

The inventory stage scans source directories and records image files in PostgreSQL.

Supported image formats in the collection include:

- `.jpg`
- `.jpeg`
- `.png`
- `.cr2`
- `.cr3`

The inventory records file-level information such as:

- File path and filename
- Extension and file size
- Modification information
- Camera and lens metadata
- Image identifiers from EXIF
- Capture grouping information
- Processing and embedding state

The database uses persistent identifiers rather than relying on filenames as the identity of a file.

### Capture identifiers

Files can be associated with a shared `capture_key` when metadata indicates that they represent the same capture.

The identifier-generation logic prioritizes image-specific identifiers when available, with camera and timestamp information used as fallbacks.

Files without a reliable capture identifier retain their own database identity and are not forced into a capture group.

This allows multiple representations or copies of a photograph to be associated without treating their filesystem paths as their identity.

### Database

The primary inventory table is:

```sql
files
```

The pipeline uses this table as its source of truth for locating files and determining which processing steps remain incomplete.

A key distinction is:

- `id`: identity of an individual database file record
- `capture_key`: identifier shared by files belonging to the same capture
- `path`: current filesystem location
- `embedding`: generated image representation

The exact database schema is defined by the database setup used by the project.

## 2. JPEG extraction

The extraction stage creates JPEG representations from files that still need embeddings.

The database query selects records where:

```sql
embedding IS NULL
```

When multiple files share a `capture_key`, the extractor selects a representative file, prioritizing larger files. It can fall back to other paths in the same capture group if the selected file is unavailable or cannot be converted.

Files without a `capture_key` receive a deterministic output identifier based on their database ID:

```text
<capture_key>.jpg
file_<database_id>.jpg
```

### RAW handling

For `.CR2` and `.CR3` files:

1. Attempt to extract the embedded JPEG preview.
2. Use the preview if it is sufficiently large.
3. Otherwise, develop the RAW image using `rawpy`.
4. Apply orientation handling.
5. Resize the resulting image.

For ordinary image formats, Pillow is used to load and normalize the image, including EXIF orientation handling.

### Extraction configuration

The current `extract_missing_photos.py` implementation uses:

| Setting | Value |
|---|---:|
| Maximum image side | 1600 px |
| JPEG quality | 90 |
| Minimum ordinary image side | 512 px |
| Minimum RAW preview side | 1200 px |
| Worker threads | 4 |
| Result log | `extract_results.jsonl` |
| Output directory | `jpegs/` |

The output is written through a temporary file and renamed after successful conversion, avoiding partially written final JPEGs.

### Running extraction

Set the PostgreSQL connection string:

```powershell
$env:DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DATABASE"
```

Then run:

```bash
python extract_missing_photos.py
```

The script fetches pending candidates from PostgreSQL, checks its previous result log, and skips completed outputs that still exist.

### Resumability

Extraction results are appended to:

```text
extract_results.jsonl
```

Each result records information such as:

- Database ID
- Capture key
- Output identifier
- Extraction status
- Generated JPEG filename
- Original path used
- Conversion method
- Errors encountered

Possible statuses include:

- `ok`
- `failed`
- `unavailable`

The result log is used to avoid repeating completed work.

The script has two configuration switches:

```python
RETRY_FAILED = False
FRESH_START = False
```

- `RETRY_FAILED`: allows previously failed candidates to be attempted again.
- `FRESH_START`: clears the extraction result log and generated JPEG/temp files before starting.

**Warning:** `FRESH_START = True` deletes generated extraction outputs in the configured output directory. Do not enable it if those outputs need to be preserved.

## 3. Embedding generation

The embedding stage converts extracted JPEGs into semantic image vectors using EmbeddingGemma2.

The model produces 768-dimensional embeddings:

```text
image -> EmbeddingGemma2 -> 768-dimensional vector
```

Embeddings are normalized so cosine similarity can be used for retrieval.

The project also uses Matryoshka dimensions:

| Dimension selector | Dimensions |
|---:|---:|
| 0 | 768 |
| 1 | 512 |
| 2 | 256 |
| 3 | 128 |

Reduced-dimensional vectors are created by truncating the original vector and normalizing the truncated result.

This allows retrieval at different embedding sizes without running a separate image encoder for each dimension.

### Sharded generation

Embedding generation can be split into NPZ shards instead of requiring the entire collection to be processed in one run.

The shard files contain aligned arrays:

```text
shard_XXXXXXX.npz
    ids
    capture_keys
    vectors
```

The ID and capture-key arrays must remain aligned with the vector rows.

Sharding makes interrupted embedding jobs easier to resume and allows intermediate results to be inspected before merging.

**Important:** shard filenames based on candidate positions assume the candidate ordering remains stable. If the candidate set or ordering changes, old shards should not be reused blindly.

When intentionally regenerating all shards, clear the shard output directory first and run generation again. Preserve the old shards if they are the only copy of completed embedding work.

### Embedding persistence

The project has used NPZ files for intermediate and legacy embedding storage, and PostgreSQL/pgvector for the database-backed pipeline.

The database-backed workflow should use PostgreSQL's embedding state to select unembedded candidates rather than repeatedly processing every scanned file.

The exact command-line flags for the current embedding-generation script are intentionally not specified here; they should be taken from the current script's argument parser rather than guessed.

## 4. S3 image storage

Generated JPEGs can be uploaded to S3 for serving through the search application.

Current bucket configuration:

```text
Bucket: thumbnail-s3-semanticsearch
Region: ap-south-1
Prefix: jpegs/
```

The application expects image objects in this layout:

```text
s3://thumbnail-s3-semanticsearch/
    jpegs/
        <image-id>.jpg
```

The application uses Boto3 and the AWS credential chain. Configure AWS credentials through an IAM role or the normal AWS CLI credential configuration; credentials should not be hardcoded into the source code.

---

# Search application

## Overview

The Flask application exposes endpoints for:

- Health checks
- Text-based semantic search
- Image-based semantic search
- Pagination through search results
- Retrieving image files from S3
- Contrastive correction of image embeddings

The inspected `app.py` implementation is the NPZ-backed application. It loads the model and embedding arrays at startup, then performs similarity ranking in memory.

It is not yet the PostgreSQL-query-per-search implementation.

## Configuration

The inspected application uses the following settings:

| Setting | Value |
|---|---|
| Embedding model | EmbeddingGemma2 |
| Base embedding size | 768 |
| Supported dimensions | 768, 512, 256, 128 |
| Results per request | 24 |
| Maximum uploaded request size | 10 MB |
| S3 bucket | `thumbnail-s3-semanticsearch` |
| S3 prefix | `jpegs` |
| AWS region | `ap-south-1` |

The application reads:

```text
DATABASE_URL
S3_BUCKET
S3_PREFIX
AWS_DEFAULT_REGION
```

`DATABASE_URL` is required by the inspected application, although its search ranking still uses the loaded NPZ arrays.

The model is loaded once during application startup and warmed up with text and image inputs.

---

# API reference

Base URL examples below use:

```text
http://localhost:8000
```

Replace this with the deployed API address when using the remote server.

## `GET /health`

Checks whether the Flask application is responding.

### Request

```http
GET /health
```

### Response

```json
{
  "status": "ok"
}
```

HTTP status: `200`

## `GET /search`

Performs a text-based semantic search.

### Query parameters

| Parameter | Required | Description |
|---|---|---|
| `q` | Yes, unless using `search_id` | Search expression |
| `offset` | No | Result offset; defaults to `0` |
| `dim` | No | Embedding dimension selector; defaults to `0` |
| `search_id` | No | Reuses a previous image-search ranking |

### Search expressions

The search expression supports positive and negative concepts.

For example:

```text
human + outdoors - cage
```

This combines the text embeddings with positive or negative signs before normalizing the final query vector.

A query must contain at least one positive text term unless an image is also supplied.

### Example

```http
GET /search?q=human%20%2B%20outdoors%20-%20cage&dim=0&offset=0
```

### Response

```json
{
  "total": 1000,
  "results": [
    {
      "index": 12,
      "id": "example-image-id",
      "score": 0.823,
      "raw_path": "D:/Photos/example.jpg"
    }
  ]
}
```

The example values are illustrative.

| Field | Description |
|---|---|
| `total` | Number of indexed images in the ranking |
| `results` | Current page of results |
| `index` | Position of the image in the loaded embedding arrays |
| `id` | Image identifier used by the application |
| `score` | Similarity score |
| `raw_path` | Original path recorded in extraction metadata |

The application returns up to 24 results per request.

To retrieve the next page, increase `offset` by 24.

## `POST /search`

Performs an image-based search, optionally combined with a text query.

The endpoint accepts `multipart/form-data`.

### Form fields

| Field | Required | Description |
|---|---|---|
| `image` | For image search | Uploaded image |
| `q` | No | Optional text query |
| `offset` | No | Result offset |
| `dim` | No | Embedding dimension selector |

### Example using cURL

```bash
curl -X POST http://localhost:8000/search \
  -F "image=@photo.jpg" \
  -F "q=outdoors" \
  -F "offset=0"
```

### Example response

```json
{
  "total": 1000,
  "search_id": "generated-search-uuid",
  "results": [
    {
      "index": 12,
      "id": "example-image-id",
      "score": 0.823,
      "raw_path": "D:/Photos/example.jpg"
    }
  ]
}
```

The `search_id` identifies the cached ranking generated by the image search.

It can be reused to paginate through the same ranking without uploading the image again.

### Image-search pagination

```http
GET /search?search_id=<search-id>&offset=24
```

The search ID is held in the application's in-memory cache. It is temporary and may expire when the cache is replaced or the application restarts.

## Embedding dimensions

The `dim` parameter selects one of the supported dimensions:

| `dim` | Actual dimensions |
|---:|---:|
| `0` | 768 |
| `1` | 512 |
| `2` | 256 |
| `3` | 128 |

For example:

```http
GET /search?q=birds&dim=2
```

This selects the 256-dimensional representation.

The same dimension selector is supported by image search and contrastive correction.

## `GET /img/<image-id>`

Retrieves an image from S3.

### Request

```http
GET /img/<image-id>
```

The image ID must not contain path separators.

The endpoint constructs the S3 object key:

```text
jpegs/<image-id>.jpg
```

It generates a presigned S3 URL with a one-hour expiry and redirects the client to it.

### Behaviour

- `302`: Redirect to the presigned image URL
- `404`: Invalid image identifier/path
- `502`: Error generating the S3 URL

The S3 bucket does not need to be publicly readable for this endpoint to work, provided the application's AWS identity has permission to retrieve the object.

## `POST /correct`

Updates an image embedding using a contrastive correction.

The correction encourages an image to become more similar to a correct concept and less similar to a wrong concept, while preserving its original representation.

The endpoint accepts JSON.

### Request body

```json
{
  "index": 12,
  "wrong": "indoor",
  "correct": "outdoors",
  "margin": 0.15,
  "preserve_weight": 2.0,
  "lr": 0.01,
  "steps": 200,
  "search_dim": 0
}
```

### Parameters

| Parameter | Description |
|---|---|
| `index` | Embedding-array index of the image to update |
| `wrong` | Concept the image should become less similar to |
| `correct` | Concept the image should become more similar to |
| `margin` | Desired similarity separation |
| `preserve_weight` | Weight for preserving the original image representation |
| `lr` | Optimizer learning rate |
| `steps` | Number of optimization steps |
| `search_dim` | Dimension selector: 0–3; defaults to 0 |

The `wrong` and `correct` fields must be non-empty strings. They cannot contain `+` or `-`.

### Example using cURL

```bash
curl -X POST http://localhost:8000/correct \
  -H "Content-Type: application/json" \
  -d '{
    "index": 12,
    "wrong": "indoor",
    "correct": "outdoors",
    "margin": 0.15,
    "preserve_weight": 2.0,
    "lr": 0.01,
    "steps": 200,
    "search_dim": 0
  }'
```

### Successful response

```json
{
  "success": true,
  "index": 12,
  "wrong": "indoor",
  "correct": "outdoors"
}
```

### Error response

```json
{
  "error": "Invalid dimension"
}
```

Invalid request data returns HTTP `400`.

**Important:** `index` is an index into the application's loaded embedding arrays, not a PostgreSQL `files.id` or a `capture_key`.

In the inspected implementation, the correction updates the in-memory contrastive embedding representation and its memory-mapped `.npy` backing file. It does not update a PostgreSQL vector row.

---

# Running the application

The application expects the model files and legacy embedding/metadata artifacts to be available at the paths configured in `app.py`.

The inspected version uses:

```text
app.py
embeddings.npz
contrastive_embeddings.npy
convert_results.jsonl
```

It also expects the locally downloaded EmbeddingGemma2 model directory and AWS credentials for S3 access.

### Local execution

Install the project's Python dependencies:

```bash
pip install -r requirements.txt
```

Set the required environment variables, then start Flask:

```bash
python app.py
```

The inspected application listens on:

```text
0.0.0.0:8000
```

It uses an ad-hoc SSL context in its direct development-server entry point. For deployment, use the repository's Docker/Gunicorn configuration and terminate HTTPS at the appropriate deployment layer.

### Docker deployment

The repository contains a `Dockerfile` and `docker-compose.yaml`.

Build and start the service with:

```bash
docker compose up --build -d
```

Inspect logs:

```bash
docker compose logs -f
```

Stop the service:

```bash
docker compose down
```

The model directory, application artifacts, environment variables, and AWS permissions must be available to the container.

---

# Current implementation notes

There are two related parts of the project that should not be confused:

1. **Database-backed pipeline:** PostgreSQL tracks the source files, capture relationships, and embedding-processing state. Extraction candidates are selected from the database.
2. **Inspected Flask API:** The uploaded `app.py` still loads `embeddings.npz`, contrastive vectors, and JSONL metadata into memory and ranks those arrays.

The API's `/correct` operation therefore modifies the legacy contrastive embedding store rather than a PostgreSQL vector.

A future PostgreSQL-backed search API should use the database as the source of truth for retrieval and corrections, rather than relying on the positional index of an in-memory NPZ array.

## Dataset scale

The collection processed by the pipeline has included:

- More than 400,000 scanned image files
- More than 8 TB of source data
- JPEG, PNG, and Canon RAW formats

These are collection/inventory scale figures, not a claim that every scanned file has a successfully extracted JPEG or an embedding.

## Repository structure

The repository has included the following pipeline and application files:

```text
PythonSemanticSearch/
│
├── app.py
├── imageScan.py
├── imageScanPersistent.py
├── metadataExtraction.py
├── jpegExtraction.py
├── extract_missing_photos.py
│
├── checkExtraction.py
├── checkShards.py
├── embeddingTest.py
├── queryTest.py
│
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
│
├── jpegs/
├── shards/
│
├── embeddings.npz
├── contrastive_embeddings.npy
└── convert_results.jsonl
```

The exact contents of the repository may change as the pipeline moves from NPZ-based intermediate artifacts toward PostgreSQL-backed embedding storage.
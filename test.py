# import cv2

# cap = cv2.VideoCapture("rtsp://localhost:8554/drone")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     cv2.imshow("DJI Feed", frame)

#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import os
import numpy as np
import psycopg
from collections import Counter


def check_embedding_matches(
    npz_path="embeddings.npz",
    database_url=os.getenv("DATABASE_URL")
):
    if not database_url:
        raise ValueError("Set DATABASE_URL")

    # Read filename + size from NPZ IDs
    data = np.load(npz_path, allow_pickle=True)

    embedding_pairs = []
    invalid_ids = 0

    for value in data["ids"]:
        parts = str(value).rsplit("__", 1)

        if len(parts) != 2:
            invalid_ids += 1
            continue

        filename, size = parts

        try:
            embedding_pairs.append((filename, int(size)))
        except ValueError:
            invalid_ids += 1

    # Read filename + size from DB
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT filename, size FROM files")
            db_pairs = cur.fetchall()

    embedding_set = set(embedding_pairs)
    db_set = set(db_pairs)

    matched = embedding_set & db_set
    missing_in_db = embedding_set - db_set
    missing_in_npz = db_set - embedding_set

    # Duplicate detection
    duplicate_db = {
        pair: count
        for pair, count in Counter(db_pairs).items()
        if count > 1
    }

    duplicate_npz = {
        pair: count
        for pair, count in Counter(embedding_pairs).items()
        if count > 1
    }

    print("\n=== Embedding / Database Diff ===")
    print(f"NPZ IDs:                  {len(data['ids']):,}")
    print(f"Invalid NPZ IDs:          {invalid_ids:,}")
    print(f"Unique NPZ pairs:         {len(embedding_set):,}")
    print(f"Database files:           {len(db_pairs):,}")
    print(f"Unique DB pairs:          {len(db_set):,}")
    print(f"Matched pairs:            {len(matched):,}")
    print(f"NPZ pairs missing in DB:  {len(missing_in_db):,}")
    print(f"DB pairs missing in NPZ:  {len(missing_in_npz):,}")
    print(f"Duplicate DB pairs:       {len(duplicate_db):,}")
    print(f"Duplicate NPZ pairs:      {len(duplicate_npz):,}")

    return {
        "matched": matched,
        "missing_in_db": missing_in_db,
        "missing_in_npz": missing_in_npz,
        "duplicate_db": duplicate_db,
        "duplicate_npz": duplicate_npz,
    }


result = check_embedding_matches()
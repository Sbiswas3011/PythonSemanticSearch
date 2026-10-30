import os
import hashlib
import time
from pathlib import Path
from collections import Counter
from contextlib import closing

import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv
from tqdm import tqdm


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

SOURCES = ["C:/", "D:/", "E:/", "F:/", "G:/", "H:/", "I:/"]
SOURCES = ["E:/"]

RAW_EXTS = {".cr2", ".cr3"}
EXTS = RAW_EXTS | {".jpg", ".jpeg", ".png"}

SKIP_DIRS = {
    "$recycle.bin",
    "system volume information",
}

MIN_NONRAW_BYTES = 100_000

HASH_CHUNK_SIZE = 4 * 1024 * 1024
DB_BATCH_SIZE = 500

# Set True to rehash every file, even when size and mtime are unchanged.
FORCE_REHASH = False


# ---------------------------------------------------------
# FILE HELPERS
# ---------------------------------------------------------

def drive_of(path):
    return os.path.splitdrive(path)[0].upper()


def path_key(path):
    # Windows paths are case-insensitive.
    return os.path.normcase(os.path.normpath(path))


def hash_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def iter_files(root, errors):
    stack = [os.path.normpath(str(root))]

    while stack:
        directory = stack.pop()

        try:
            entries = os.scandir(directory)
        except OSError:
            errors["directory_errors"] += 1
            continue

        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name.lower() in SKIP_DIRS:
                            continue

                        # Avoid junctions and directory loops.
                        is_junction = getattr(entry, "is_junction", None)
                        if is_junction and is_junction():
                            continue

                        stack.append(entry.path)

                    elif entry.is_file(follow_symlinks=False):
                        yield entry

                except OSError:
                    errors["entry_errors"] += 1


# ---------------------------------------------------------
# DATABASE HELPERS
# ---------------------------------------------------------

def insert_event(cur, run_id, event, **kwargs):
    cur.execute(
        """
        INSERT INTO scan_events (
            run_id, event, old_sha256, new_sha256,
            old_path, new_path, data
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            run_id,
            event,
            kwargs.get("old_sha256"),
            kwargs.get("new_sha256"),
            kwargs.get("old_path"),
            kwargs.get("new_path"),
            Jsonb(kwargs.get("data", {})),
        ),
    )


def flush_batch(cur, batch):
    if not batch:
        return

    cur.executemany(
        """
        INSERT INTO photo_paths (
            path_key, path, drive, stem, ext,
            sha256, size, mtime_ns,
            status, first_seen, last_seen, last_seen_run_id
        )
        VALUES (
            %(path_key)s, %(path)s, %(drive)s, %(stem)s, %(ext)s,
            %(sha256)s, %(size)s, %(mtime_ns)s,
            'present', NOW(), NOW(), %(run_id)s
        )
        ON CONFLICT (path_key)
        DO UPDATE SET
            path = EXCLUDED.path,
            drive = EXCLUDED.drive,
            stem = EXCLUDED.stem,
            ext = EXCLUDED.ext,
            sha256 = EXCLUDED.sha256,
            size = EXCLUDED.size,
            mtime_ns = EXCLUDED.mtime_ns,
            status = 'present',
            last_seen = NOW(),
            last_seen_run_id = EXCLUDED.last_seen_run_id,
            missing_since = NULL
        """,
        batch,
    )

    batch.clear()


# ---------------------------------------------------------
# MAIN SCANNER
# ---------------------------------------------------------

def scan_photos(sources=SOURCES):

    started = time.time()
    errors = Counter()
    counts = Counter()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:

            # Start scan run.
            cur.execute(
                "INSERT INTO scan_runs DEFAULT VALUES RETURNING id"
            )
            run_id = cur.fetchone()[0]

            conn.commit()

            # Load known paths and hashes.
            print("Loading existing database records...")

            cur.execute(
                """
                SELECT path_key, path, drive, sha256,
                       size, mtime_ns, status
                FROM photo_paths
                """
            )

            existing = {}

            for row in cur.fetchall():
                existing[row[0]] = {
                    "path": row[1],
                    "drive": row[2],
                    "sha256": row[3].strip(),
                    "size": row[4],
                    "mtime_ns": row[5],
                    "status": row[6],
                }

            cur.execute("SELECT sha256 FROM photos")
            known_hashes = {row[0].strip() for row in cur.fetchall()}

            first_run = not existing

            print(
                f"Loaded {len(existing):,} paths and "
                f"{len(known_hashes):,} unique file hashes."
            )

            # Drives whose scans completed successfully.
            scanned_drives = set()

            # Track every path found during this run.
            found_keys = set()

            # Buffer database writes.
            batch = []

            for source in sources:
                root = Path(source)
                drive = root.drive.upper()

                if not root.exists():
                    tqdm.write(f"[SKIP] {source} not found; leaving unchanged.")
                    continue

                drive_start = time.time()
                drive_errors_before = sum(errors.values())
                found_count = 0
                processed_count = 0

                tqdm.write(f"\n[START] Scanning {source}")

                with tqdm(
                    desc=source,
                    unit=" files",
                    leave=True,
                ) as bar:

                    for entry in iter_files(root, errors):

                        ext = Path(entry.name).suffix.lower()

                        if ext not in EXTS:
                            continue

                        try:
                            stat = entry.stat(follow_symlinks=False)
                        except OSError as e:
                            errors["stat_errors"] += 1
                            continue

                        size = stat.st_size

                        if size == 0:
                            counts["zero_bytes"] += 1
                            continue

                        if ext not in RAW_EXTS and size < MIN_NONRAW_BYTES:
                            counts["small_nonraw"] += 1
                            continue

                        path = os.path.normpath(entry.path)
                        key = path_key(path)

                        if key in found_keys:
                            continue

                        found_keys.add(key)
                        found_count += 1
                        bar.update(1)

                        mtime_ns = stat.st_mtime_ns
                        old = existing.get(key)

                        # Check whether the file is unchanged.
                        unchanged = (
                            old is not None
                            and old["size"] == size
                            and old["mtime_ns"] == mtime_ns
                            and not FORCE_REHASH
                        )

                        if unchanged:
                            sha256 = old["sha256"]
                            counts["hash_skipped"] += 1

                        else:
                            try:
                                sha256 = hash_file(path)
                            except OSError as e:
                                errors["hash_errors"] += 1
                                found_keys.discard(key)

                                tqdm.write(f"[HASH ERROR] {path}: {e}")
                                continue

                            counts["hashed"] += 1

                        # Check whether this content existed before this discovery.
                        was_known = sha256 in known_hashes

                        # Insert new content if necessary.
                        if not was_known:
                            cur.execute(
                                """
                                INSERT INTO photos (sha256, size)
                                VALUES (%s, %s)
                                ON CONFLICT (sha256) DO NOTHING
                                """,
                                (sha256, size),
                            )

                            known_hashes.add(sha256)
                            counts["new_content"] += 1

                        # Detect modified files.
                        if old is not None and old["sha256"] != sha256:
                            counts["modified"] += 1

                            tqdm.write(
                                f"[MODIFIED] {path}: "
                                f"{old['sha256'][:12]} -> {sha256[:12]}"
                            )

                            if not first_run:
                                insert_event(
                                    cur,
                                    run_id,
                                    "modified",
                                    old_sha256=old["sha256"],
                                    new_sha256=sha256,
                                    old_path=old["path"],
                                    new_path=path,
                                    data={
                                        "old_size": old["size"],
                                        "new_size": size,
                                    },
                                )

                        # Detect new paths.
                        elif old is None:
                            counts["new_paths"] += 1

                            if was_known and not first_run:
                                tqdm.write(f"[PATH ADDED] {path}")

                                insert_event(
                                    cur,
                                    run_id,
                                    "path_added",
                                    new_sha256=sha256,
                                    new_path=path,
                                )

                        # ALWAYS queue the path, whether new, modified or unchanged.
                        batch.append({
                            "path_key": key,
                            "path": path,
                            "drive": drive,
                            "stem": Path(entry.name).stem.lower(),
                            "ext": ext,
                            "sha256": sha256,
                            "size": size,
                            "mtime_ns": mtime_ns,
                            "run_id": run_id,
                        })

                        processed_count += 1

                        # Flush database batch.
                        if len(batch) >= DB_BATCH_SIZE:
                            batch_size = len(batch)

                            flush_batch(cur, batch)
                            conn.commit()

                            tqdm.write(
                                f"[DB COMMIT] {drive} | "
                                f"{batch_size:,} paths committed | "
                                f"Processed: {processed_count:,}"
                            )

                # Flush remaining records after finishing this drive.
                if batch:
                    batch_size = len(batch)

                    flush_batch(cur, batch)
                    conn.commit()

                    tqdm.write(
                        f"[DB COMMIT] Final batch: {batch_size:,} paths"
                    )

                tqdm.write(
                    f"[DRIVE COMPLETE] {drive} | "
                    f"Files: {found_count:,} | "
                    f"Time: {time.time() - drive_start:.1f}s | "
                    f"Errors: {sum(errors.values()) - drive_errors_before:,}"
                )
            # Flush anything remaining.
            flush_batch(cur, batch)
            conn.commit()

            # Mark paths missing only on successfully scanned drives.
            for drive in scanned_drives:

                cur.execute(
                    """
                    UPDATE photo_paths
                    SET
                        status = 'missing',
                        missing_since = COALESCE(missing_since, NOW())
                    WHERE drive = %s
                      AND status = 'present'
                      AND (
                          last_seen_run_id IS DISTINCT FROM %s
                      )
                    RETURNING path, sha256
                    """,
                    (drive, run_id),
                )

                missing_rows = cur.fetchall()

                for old_path, old_sha in missing_rows:
                    counts["missing"] += 1

                    if not first_run:
                        insert_event(
                            cur,
                            run_id,
                            "missing",
                            old_sha256=old_sha.strip(),
                            old_path=old_path,
                        )

            # Count duplicate content records.
            cur.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT sha256
                    FROM photo_paths
                    WHERE status = 'present'
                    GROUP BY sha256
                    HAVING COUNT(*) > 1
                ) duplicates
                """
            )
            duplicate_groups = cur.fetchone()[0]

            # Finish run.
            counts["paths_found"] = len(found_keys)
            counts["duplicate_groups"] = duplicate_groups
            counts["filesystem_errors"] = sum(errors.values())

            cur.execute(
                """
                UPDATE scan_runs
                SET
                    finished_at = NOW(),
                    status = 'completed',
                    scanned_drives = %s,
                    counts = %s
                WHERE id = %s
                """,
                (
                    sorted(scanned_drives),
                    Jsonb(dict(counts)),
                    run_id,
                ),
            )

            conn.commit()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print(f"Scan run: {run_id}")
    print(f"Drives scanned: {', '.join(sorted(scanned_drives))}")
    print(f"Paths found: {counts['paths_found']:,}")
    print(f"Files hashed: {counts['hashed']:,}")
    print(f"Hashes skipped: {counts['hash_skipped']:,}")
    print(f"New content: {counts['new_content']:,}")
    print(f"New paths: {counts['new_paths']:,}")
    print(f"Modified files: {counts['modified']:,}")
    print(f"Missing paths: {counts['missing']:,}")
    print(f"Duplicate groups: {counts['duplicate_groups']:,}")
    print(f"Filesystem errors: {counts['filesystem_errors']:,}")
    print(f"Total time: {time.time() - started:.1f}s")
    print("=" * 55)

def clear_database():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                TRUNCATE TABLE
                    scan_events,
                    photo_paths,
                    photos,
                    scan_runs
                RESTART IDENTITY CASCADE;
            """)

    print("Database cleared successfully.")

if __name__ == "__main__":
    clear_database()
    scan_photos()
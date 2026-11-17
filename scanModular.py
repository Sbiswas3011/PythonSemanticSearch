import os
import re
import subprocess
import time
import uuid
from collections import Counter

import psycopg
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]

SOURCE = "I:/"
OUT_FILE = "C:/Users/JAYANTA/Desktop/PythonCases/ids.txt"
ERR_FILE = "C:/Users/JAYANTA/Desktop/PythonCases/errors.txt"

BATCH_SIZE = 500
IDLE_WAIT = 2          # seconds to wait when the file has no new data
NS = uuid.UUID("6f1c0d2e-0000-4000-8000-000000000001")  # never change this once used

FIELDS = ["$Directory/$FileName", "$Model", "$SerialNumber", "$ImageUniqueID",
          "$LensSerialNumber", "$SubSecDateTimeOriginal", "$OriginalDocumentID",
          "$DocumentID", "$InstanceID"]
CMD = ["exiftool", "-r",
       "-ext", "jpg", "-ext", "jpeg", "-ext", "cr2", "-ext", "cr3",
       "-i", "$RECYCLE.BIN", "-i", "System Volume Information",
       "-q", "-f", "-p", "|".join(FIELDS), SOURCE]


# ---------------------------------------------------------
# PARSING
# ---------------------------------------------------------

def clean(v):
    v = v.strip()
    return None if v in ("", "-") else v


def build_row(line, run_id):
    parts = line.rstrip("\r").rsplit("|", 8)    # path may itself contain '|'
    if len(parts) != 9:
        return None
    path, _model, serial, uid, _lens, subsec_dt, _orig, _doc, inst = parts
    path = path.strip()
    if not path:
        return None

    serial, uid, subsec_dt, inst = map(clean, (serial, uid, subsec_dt, inst))
    if uid and re.fullmatch(r"0+", uid):
        uid = None

    subsec_key = None
    if serial and subsec_dt:
        dt = re.sub(r"(Z|[+-]\d\d:\d\d)$", "", subsec_dt)   # drop timezone offset
        if re.search(r"\.\d+$", dt):                         # must include subseconds
            subsec_key = f"{serial}|{dt}"

    if subsec_key:
        capture_key = "subsec:" + subsec_key
    elif uid:
        capture_key = "uuid:" + uid
    else:
        capture_key = None

    return {
        "path": path,
        "shot_id": uuid.uuid5(NS, capture_key) if capture_key else None,
        "capture_key": capture_key,
        "cam_uuid": uid,
        "subsec_key": subsec_key,
        "instance_id": inst,
        "run_id": run_id,
    }


# ---------------------------------------------------------
# TAIL THE OUTPUT FILE
# ---------------------------------------------------------

def tail_lines(proc):
    """Yield complete lines as exiftool writes them. Yields None when idle.
    Returns only after exiftool has exited AND the file is fully read."""
    buf = ""
    with open(OUT_FILE, "r", encoding="utf-8", errors="replace", newline="") as f:
        while True:
            exited = proc.poll() is not None     # check BEFORE reading (avoids a race)
            chunk = f.read(1 << 20)
            if chunk:
                buf += chunk
                *lines, buf = buf.split("\n")    # last piece may be a half-written line
                yield from lines
            elif exited:
                break                            # exited and nothing left to read
            else:
                yield None                       # idle: caller can flush its batch
                time.sleep(IDLE_WAIT)
        if buf.strip():
            yield buf                            # final line with no trailing newline


# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

def flush(cur, batch, run_id, first_run, counts):
    if not batch:
        return
    rows = list(batch.values())

    cur.execute("SELECT path, instance_id, status FROM files WHERE path = ANY(%s)",
                ([r["path"] for r in rows],))
    old = {p: (inst, st) for p, inst, st in cur.fetchall()}

    shots = {r["shot_id"]: r for r in rows if r["shot_id"]}.values()
    cur.executemany(
        """INSERT INTO shots (shot_id, capture_key, cam_uuid, subsec_key)
           VALUES (%(shot_id)s, %(capture_key)s, %(cam_uuid)s, %(subsec_key)s)
           ON CONFLICT (capture_key) DO UPDATE SET
               cam_uuid   = COALESCE(shots.cam_uuid,   EXCLUDED.cam_uuid),
               subsec_key = COALESCE(shots.subsec_key, EXCLUDED.subsec_key)""",
        list(shots))

    events = []
    for r in rows:
        o = old.get(r["path"])
        if o is None:
            ev = "new"
        elif o[0] and r["instance_id"] and o[0] != r["instance_id"]:
            ev = "modified"
        elif o[1] == "missing":
            ev = "reappeared"
        else:
            ev = None
        if ev:
            counts[ev] += 1
            if not (first_run and ev == "new"):
                events.append((run_id, ev, r["shot_id"], None, r["path"]))

    if events:
        cur.executemany(
            """INSERT INTO scan_events (run_id, event, shot_id, old_path, new_path)
               VALUES (%s, %s, %s, %s, %s)""", events)

    cur.executemany(
        """INSERT INTO files (path, shot_id, instance_id, status,
                              first_seen, last_seen, last_seen_run_id)
           VALUES (%(path)s, %(shot_id)s, %(instance_id)s, 'present',
                   NOW(), NOW(), %(run_id)s)
           ON CONFLICT (path) DO UPDATE SET
               shot_id = EXCLUDED.shot_id,
               instance_id = EXCLUDED.instance_id,
               status = 'present',
               last_seen = NOW(),
               last_seen_run_id = EXCLUDED.last_seen_run_id,
               missing_since = NULL""",
        rows)
    batch.clear()


def mark_missing_and_moved(cur, run_id, counts):
    cur.execute(
        """UPDATE files
           SET status = 'missing', missing_since = COALESCE(missing_since, NOW())
           WHERE status = 'present'
             AND starts_with(path, %s)
             AND last_seen_run_id IS DISTINCT FROM %s
           RETURNING path, shot_id""",
        (SOURCE, run_id))
    missing = cur.fetchall()
    counts["missing"] = len(missing)
    if missing:
        cur.executemany(
            """INSERT INTO scan_events (run_id, event, shot_id, old_path)
               VALUES (%s, 'missing', %s, %s)""",
            [(run_id, sid, p) for p, sid in missing])

    # moved = a path went missing and a new path with the same shot + file name appeared
    cur.execute(
        r"""INSERT INTO scan_events (run_id, event, shot_id, old_path, new_path)
            SELECT %(r)s, 'moved', n.shot_id, m.old_path, n.new_path
            FROM scan_events n
            JOIN scan_events m
              ON m.run_id = n.run_id AND m.shot_id = n.shot_id
             AND lower(substring(m.old_path from '[^/]+$'))
               = lower(substring(n.new_path from '[^/]+$'))
            WHERE n.run_id = %(r)s AND n.event = 'new' AND m.event = 'missing'""",
        {"r": run_id})
    counts["moved"] = cur.rowcount


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    global SOURCE
    SOURCE = SOURCE.replace("\\", "/")
    if not SOURCE.endswith("/"):
        SOURCE += "/"
    if not os.path.exists(SOURCE):
        print(f"{SOURCE} not found, nothing scanned.")
        return

    counts = Counter()
    interrupted = False

    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT EXISTS (SELECT 1 FROM files)")
        first_run = not cur.fetchone()[0]
        cur.execute("INSERT INTO scan_runs DEFAULT VALUES RETURNING id")
        run_id = cur.fetchone()[0]
        conn.commit()

        out, err = open(OUT_FILE, "wb"), open(ERR_FILE, "wb")   # truncates old output
        proc = subprocess.Popen(CMD, stdout=out, stderr=err)
        out.close(); err.close()

        batch = {}
        try:
            with tqdm(unit=" rows") as bar:
                for line in tail_lines(proc):
                    if line is None:                        # idle: push what we have
                        flush(cur, batch, run_id, first_run, counts)
                        conn.commit()
                        continue
                    if not line.strip():
                        continue
                    row = build_row(line, run_id)
                    if row is None:
                        counts["bad_lines"] += 1
                        continue
                    batch[row["path"]] = row
                    counts["rows"] += 1
                    bar.update(1)
                    if len(batch) >= BATCH_SIZE:
                        flush(cur, batch, run_id, first_run, counts)
                        conn.commit()
        except KeyboardInterrupt:
            interrupted = True
            proc.terminate()
            print("\nInterrupted; saving what was read.")

        flush(cur, batch, run_id, first_run, counts)
        conn.commit()

        ok = not interrupted and counts["rows"] > 0
        if ok:
            mark_missing_and_moved(cur, run_id, counts)

        cur.execute("UPDATE scan_runs SET finished_at = NOW(), status = %s WHERE id = %s",
                    ("done" if ok else "failed", run_id))
        conn.commit()

    print(f"\nRun {run_id}: {'done' if ok else 'FAILED'} | exiftool exit code: {proc.returncode}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v:,}")
    if counts["rows"] == 0:
        print(f"No rows were read, so nothing was marked missing. Check {ERR_FILE}")


if __name__ == "__main__":
    main()
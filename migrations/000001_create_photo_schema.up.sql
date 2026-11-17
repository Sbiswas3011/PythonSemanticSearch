CREATE TABLE scan_runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running'   -- running | done | failed
);

CREATE TABLE shots (
    shot_id UUID PRIMARY KEY,                -- uuid5 of capture_key
    capture_key TEXT NOT NULL UNIQUE,        -- 'uuid:<cam_uuid>' or 'subsec:<subsec_key>'
    cam_uuid TEXT,                           -- Canon ImageUniqueID, NULL if absent
    subsec_key TEXT,                         -- serial|DateTimeOriginal|SubSecTimeOriginal
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE files (
    id BIGSERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    shot_id UUID REFERENCES shots(shot_id),  -- NULL if no usable key
    size BIGINT NOT NULL,
    mtime_ns BIGINT NOT NULL,
    fingerprint TEXT NOT NULL,               -- size + first/last 64 KB hash
    status TEXT NOT NULL DEFAULT 'present',  -- present | missing
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    missing_since TIMESTAMPTZ
);
CREATE INDEX idx_files_shot ON files(shot_id);
  
CREATE TABLE scan_events (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES scan_runs(id),
    event TEXT NOT NULL,                     -- new | modified | moved | duplicate | deleted
    shot_id UUID REFERENCES shots(shot_id),
    old_path TEXT,
    new_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_events_run ON scan_events(run_id);
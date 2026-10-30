CREATE TABLE photos (
    sha256 CHAR(64) PRIMARY KEY,
    size BIGINT NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE scan_runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    scanned_drives TEXT[] NOT NULL DEFAULT '{}',
    counts JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE photo_paths (
    id BIGSERIAL PRIMARY KEY,
    path_key TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    drive TEXT NOT NULL,
    stem TEXT NOT NULL,
    ext TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL REFERENCES photos(sha256),
    size BIGINT NOT NULL,
    mtime_ns BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'present',
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    missing_since TIMESTAMPTZ,
    last_seen_run_id BIGINT REFERENCES scan_runs(id)
);

CREATE INDEX idx_photo_paths_sha256 ON photo_paths(sha256);
CREATE INDEX idx_photo_paths_drive_status ON photo_paths(drive, status);

CREATE TABLE scan_events (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES scan_runs(id),
    event TEXT NOT NULL,
    old_sha256 CHAR(64),
    new_sha256 CHAR(64),
    old_path TEXT,
    new_path TEXT,
    data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_scan_events_run ON scan_events(run_id);
CREATE INDEX idx_scan_events_event ON scan_events(event);

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE files (
    id BIGSERIAL PRIMARY KEY,

    -- Filesystem
    drive_id TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    extension TEXT NOT NULL,
    size BIGINT NOT NULL,
    mtime_ns BIGINT NOT NULL,

    -- EXIF metadata
    model TEXT,
    serial_number TEXT,
    image_unique_id TEXT,
    lens_serial_number TEXT,
    subsec_datetime_original TEXT,
    original_document_id TEXT,
    document_id TEXT,
    instance_id TEXT,

    -- Capture identity
    capture_key UUID,

    -- EmbeddingGemma2
    embedding vector(768),

    -- Inventory
    status TEXT NOT NULL DEFAULT 'present',
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_files_capture
    ON files(capture_key);

CREATE INDEX idx_files_status
    ON files(status);

CREATE INDEX idx_files_drive
    ON files(drive_id);

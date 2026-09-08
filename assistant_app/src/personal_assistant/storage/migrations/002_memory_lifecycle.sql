-- M4: fields required by the trusted-memory lifecycle.
ALTER TABLE memory_candidates
    ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'context';

ALTER TABLE memory_candidates
    ADD COLUMN IF NOT EXISTS withdraw_deadline TIMESTAMPTZ;

ALTER TABLE memory_candidates
    ADD COLUMN IF NOT EXISTS embedding vector(1536);

CREATE INDEX IF NOT EXISTS memory_candidates_user_kind_status_idx
    ON memory_candidates (user_id, kind, status);

ALTER TABLE memory_candidates
    DROP CONSTRAINT IF EXISTS memory_candidates_status_check;

ALTER TABLE memory_candidates
    ADD CONSTRAINT memory_candidates_status_check
    CHECK (status IN ('CANDIDATE', 'CONFLICT', 'PUBLISHED', 'EXPIRED', 'REJECTED'));

ALTER TABLE memory_records
    DROP CONSTRAINT IF EXISTS memory_records_status_check;

ALTER TABLE memory_records
    ADD CONSTRAINT memory_records_status_check
    CHECK (status IN ('CANDIDATE', 'PUBLISHED', 'SUPERSEDED', 'EXPIRED', 'DELETED'));

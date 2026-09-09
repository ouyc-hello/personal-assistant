-- Compatibility migration for the existing PostgreSQL database.
-- The host database may have an older memory_candidates shape; add only the
-- fields required by the current business model, never drop user data.
ALTER TABLE memory_candidates
    ADD COLUMN IF NOT EXISTS memory_record_id UUID REFERENCES memory_records(id),
    ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'context',
    ADD COLUMN IF NOT EXISTS withdraw_deadline TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS embedding vector;

CREATE INDEX IF NOT EXISTS memory_candidates_user_kind_status_idx
    ON memory_candidates (user_id, kind, status);

-- Older deployments used these names. Keep them as source-compatible aliases
-- for readers outside this application; the application writes current names.
ALTER TABLE audit_events
    ADD COLUMN IF NOT EXISTS entity_type TEXT,
    ADD COLUMN IF NOT EXISTS entity_id TEXT,
    ADD COLUMN IF NOT EXISTS aggregate_type TEXT,
    ADD COLUMN IF NOT EXISTS aggregate_id TEXT,
    ADD COLUMN IF NOT EXISTS actor_type TEXT;

UPDATE audit_events
SET entity_type = COALESCE(entity_type, aggregate_type),
    entity_id = COALESCE(entity_id, aggregate_id)
WHERE entity_type IS NULL OR entity_id IS NULL;

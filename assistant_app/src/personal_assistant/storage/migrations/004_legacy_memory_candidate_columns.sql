-- Complete compatibility for databases created by the earlier memory SDK.
-- These columns are required because that schema made title, candidate_type and
-- version non-null even though the current service uses content and kind.
ALTER TABLE memory_candidates
    ADD COLUMN IF NOT EXISTS thread_id UUID,
    ADD COLUMN IF NOT EXISTS source_message_id UUID,
    ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS candidate_type TEXT NOT NULL DEFAULT 'context',
    ADD COLUMN IF NOT EXISTS sdk_memory_id TEXT,
    ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

UPDATE memory_candidates
SET title = COALESCE(NULLIF(title, ''), content),
    candidate_type = COALESCE(NULLIF(candidate_type, ''), kind);

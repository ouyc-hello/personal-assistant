-- M7: persist approval ownership even when a request has no thread.
ALTER TABLE approval_requests
    ADD COLUMN IF NOT EXISTS user_id TEXT;

UPDATE approval_requests AS request
SET user_id = thread.user_id
FROM threads AS thread
WHERE request.user_id IS NULL
  AND request.thread_id = thread.id;

-- Legacy requests without a thread have no recoverable owner. Keep them
-- inaccessible instead of allowing an arbitrary user to resolve them.
UPDATE approval_requests
SET user_id = 'legacy-unowned'
WHERE user_id IS NULL;

ALTER TABLE approval_requests
    ALTER COLUMN user_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS approval_requests_user_status_idx
    ON approval_requests (user_id, status);

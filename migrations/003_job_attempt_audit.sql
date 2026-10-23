CREATE TABLE job_attempts (
    token uuid PRIMARY KEY, job_id uuid NOT NULL REFERENCES jobs(id), attempt_number integer NOT NULL,
    agent_id text NOT NULL, state text NOT NULL CHECK(state IN ('leased','done','expired','retry','blocked')),
    claimed_at timestamptz NOT NULL DEFAULT clock_timestamp(), last_heartbeat_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    lease_until timestamptz NOT NULL, finished_at timestamptz, error text,
    UNIQUE(job_id,attempt_number)
);
CREATE INDEX job_attempts_job_idx ON job_attempts(job_id,attempt_number);
INSERT INTO job_attempts(token,job_id,attempt_number,agent_id,state,claimed_at,last_heartbeat_at,lease_until,finished_at,error)
SELECT token,id,attempts,agent_id,
    CASE WHEN state='done' THEN 'done' WHEN state='blocked' THEN 'blocked'
         WHEN state='queued' THEN 'retry' WHEN lease_until<=clock_timestamp() THEN 'expired' ELSE 'leased' END,
    updated_at,updated_at,coalesce(lease_until,updated_at),
    CASE WHEN state <> 'leased' OR lease_until<=clock_timestamp() THEN updated_at ELSE NULL END,error
FROM jobs WHERE token IS NOT NULL AND agent_id IS NOT NULL AND attempts>0;

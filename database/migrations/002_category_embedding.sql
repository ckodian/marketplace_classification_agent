CREATE TABLE embedding_jobs (
                                job_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                                embed_model  text NOT NULL,
                                filters      jsonb NOT NULL DEFAULT '{}',   -- {"branch": ..., "force": ...}
                                status       text NOT NULL DEFAULT 'queued'
                                    CHECK (status IN ('queued','running','completed','failed')),
                                total        int,
                                processed    int NOT NULL DEFAULT 0,
                                failed       int NOT NULL DEFAULT 0,
                                last_error   text,
                                created_at   timestamptz NOT NULL DEFAULT now(),
                                started_at   timestamptz,
                                finished_at  timestamptz
);
CREATE UNIQUE INDEX embedding_jobs_one_active
    ON embedding_jobs (embed_model) WHERE status IN ('queued','running');
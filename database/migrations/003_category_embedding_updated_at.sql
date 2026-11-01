ALTER TABLE category_embeddings
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();
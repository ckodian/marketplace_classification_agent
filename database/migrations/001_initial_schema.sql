-- Extensions
CREATE
EXTENSION IF NOT EXISTS vector;
CREATE
EXTENSION IF NOT EXISTS ltree;


-- updated_at trigger function
CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW IS DISTINCT FROM OLD THEN
        NEW.updated_at := now();
END IF;
RETURN NEW;
END;
$$;
LANGUAGE plpgsql;


-- ---------------------------------------------------------------------
-- 1. Category tree (your information architecture)
-- ---------------------------------------------------------------------
CREATE TABLE categories
(
    id                    text PRIMARY KEY,            -- id from the source taxonomy
    parent_id             text REFERENCES categories (id),
    path                  ltree       NOT NULL UNIQUE, -- e.g. 536.638.4515 (labels = ids)
    name                  text        NOT NULL,
    description           text,                        -- fill gaps with an LLM once; retrieval depends on it
    description_generated boolean     NOT NULL DEFAULT false,
    logo_url              text,
    depth                 int         NOT NULL,
    is_leaf               boolean     NOT NULL DEFAULT false,
    is_active             boolean     NOT NULL DEFAULT true,
    taxonomy_version      int         NOT NULL DEFAULT 1,
    created_at            timestamptz NOT NULL DEFAULT now(),
    updated_at            timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX categories_path_gist ON categories USING gist (path);
CREATE INDEX categories_parent_idx ON categories (parent_id);
CREATE INDEX categories_leaf_idx ON categories (is_leaf) WHERE is_active;
CREATE TRIGGER set_categories_updated_at
    BEFORE UPDATE
    ON categories
    FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at();

-- One row per category per embedding model, so a model switch never
-- silently mixes vectors from two models.
CREATE TABLE category_embeddings
(
    category_id      text        NOT NULL REFERENCES categories (id) ON DELETE CASCADE,
    embed_model      text        NOT NULL, -- e.g. 'nomic-embed-text'
    source_text      text        NOT NULL, -- 'path > name: description' exactly as embedded
    source_text_hash text        NOT NULL, -- re-embed only when this changes
    embedding        vector(768) NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (category_id, embed_model)
);
CREATE INDEX category_embeddings_hnsw
    ON category_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE TRIGGER set_category_embeddings_updated_at
    BEFORE UPDATE
    ON category_embeddings
    FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at();
-- ---------------------------------------------------------------------
-- 2. Products as the agent saw them
-- ---------------------------------------------------------------------
-- One row per distinct content version. content_hash covers only the
-- fields that affect classification (title, description, attributes,
-- image hashes), so price-only updates produce the same hash.
CREATE TABLE product_snapshots
(
    product_id      text        NOT NULL,
    content_hash    text        NOT NULL,
    base_product_id text,                         -- null when this is the base product
    title           text        NOT NULL,
    description     text,
    attributes      jsonb       NOT NULL DEFAULT '{}',
    price_amount    numeric(12, 2),
    price_currency  char(3),
    image_hashes    text[] NOT NULL DEFAULT '{}', -- join to image_captions
    payload         jsonb       NOT NULL,         -- raw input, for replay
    embed_model     text,
    embedding       vector(768),                  -- for "similar classified product" lookups
    source_event_id text,
    received_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (product_id, content_hash)
);
CREATE INDEX product_snapshots_base_idx ON product_snapshots (base_product_id);
CREATE INDEX product_snapshots_latest_idx ON product_snapshots (product_id, received_at DESC);
CREATE INDEX product_snapshots_hnsw
    ON product_snapshots USING hnsw (embedding vector_cosine_ops);

-- Vision output cached by image hash, so the same image is captioned once.
CREATE TABLE image_captions
(
    image_hash   text        NOT NULL,
    vision_model text        NOT NULL,
    image_url    text,
    caption      text        NOT NULL,
    attributes   jsonb       NOT NULL DEFAULT '{}', -- colour, material, product type...
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (image_hash, vision_model)
);

-- ---------------------------------------------------------------------
-- 3. Evaluation runs (grouping for model / prompt comparisons)
-- ---------------------------------------------------------------------
CREATE TABLE eval_runs
(
    eval_run_id     uuid PRIMARY KEY     DEFAULT gen_random_uuid(),
    name            text        NOT NULL, -- e.g. 'ollama-qwen7b-prompts-v1'
    provider        text        NOT NULL, -- ollama | anthropic | bedrock
    model           text        NOT NULL,
    embed_model     text        NOT NULL,
    prompt_versions jsonb       NOT NULL, -- {"classifier":"v1","critic":"v1"}
    metrics         jsonb,                -- filled by eval.py at the end
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz
);

-- ---------------------------------------------------------------------
-- 4. Agent audit trail
-- ---------------------------------------------------------------------
CREATE TABLE classification_runs
(
    run_id               uuid PRIMARY KEY     DEFAULT gen_random_uuid(),
    product_id           text        NOT NULL,
    content_hash         text        NOT NULL,
    trigger              text        NOT NULL CHECK (trigger IN ('ui', 'event', 'eval', 'backfill')),
    source_event_id      text,
    eval_run_id          uuid REFERENCES eval_runs (eval_run_id),
    provider             text        NOT NULL,
    model                text        NOT NULL,
    embed_model          text        NOT NULL,
    prompt_versions      jsonb       NOT NULL,
    candidates           jsonb       NOT NULL DEFAULT '[]', -- [{category_id, rank, similarity, source: vector|sibling|base}]
    final_category_id    text REFERENCES categories (id),
    confidence           numeric(4, 3) CHECK (confidence BETWEEN 0 AND 1),
    confidence_breakdown jsonb,                             -- {self, critic_agree, sibling_agree, retrieval}
    decision             text CHECK (decision IN
                                     ('auto_linked', 'review', 'unclassified', 'inherited', 'skipped', 'failed')),
    status               text        NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    error                text,
    latency_ms           int,
    input_tokens         int,
    output_tokens        int,
    created_at           timestamptz NOT NULL DEFAULT now(),
    completed_at         timestamptz,
    FOREIGN KEY (product_id, content_hash)
        REFERENCES product_snapshots (product_id, content_hash)
);
CREATE INDEX classification_runs_product_idx ON classification_runs (product_id, created_at DESC);
CREATE INDEX classification_runs_eval_idx ON classification_runs (eval_run_id);
CREATE INDEX classification_runs_decision_idx ON classification_runs (decision, created_at DESC);

-- One row per LLM step. Keeping these as rows (not a jsonb blob) makes
-- "did the critic fix or break the answer?" a plain SQL query.
CREATE TABLE classification_steps
(
    run_id          uuid        NOT NULL REFERENCES classification_runs (run_id) ON DELETE CASCADE,
    step_no         int         NOT NULL,                         -- 1 classifier, 2 critic, 3 classifier (revise)...
    role            text        NOT NULL CHECK (role IN ('classifier', 'critic')),
    category_id     text,                                         -- not FK: may be an invalid ID the model invented
    is_valid_choice boolean     NOT NULL DEFAULT true,            -- false when outside the candidate list
    self_confidence numeric(4, 3),
    verdict         text CHECK (verdict IN ('accept', 'revise')), -- critic only
    output          jsonb       NOT NULL,                         -- full structured output incl. rationale
    raw_response    text,                                         -- keep for debugging small models
    latency_ms      int,
    input_tokens    int,
    output_tokens   int,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, step_no)
);

-- ---------------------------------------------------------------------
-- 5. Destination: product-to-category links
-- ---------------------------------------------------------------------
-- One primary category per product. If your marketplace allows several,
-- make the PK (product_id, category_id) and add is_primary.
CREATE TABLE category_links
(
    product_id       text PRIMARY KEY,
    category_id      text        NOT NULL REFERENCES categories (id),
    confidence       numeric(4, 3),
    source           text        NOT NULL CHECK (source IN ('llm', 'inherited', 'manual')),
    run_id           uuid REFERENCES classification_runs (run_id),
    inherited_from   text, -- base or sibling product_id when source = inherited
    taxonomy_version int         NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX category_links_category_idx ON category_links (category_id);

-- ---------------------------------------------------------------------
-- 6. Human review
-- ---------------------------------------------------------------------
CREATE TABLE review_queue
(
    review_id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id                uuid        NOT NULL UNIQUE REFERENCES classification_runs (run_id),
    product_id            text        NOT NULL,
    suggested_category_id text REFERENCES categories (id),
    confidence            numeric(4, 3),
    reason                text        NOT NULL CHECK (reason IN
                                                      ('mid_confidence', 'low_confidence', 'critic_disagreed', 'validation_failed')),
    status                text        NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'accepted', 'overridden', 'dismissed')),
    resolved_category_id  text REFERENCES categories (id),
    resolved_by           text,
    resolved_at           timestamptz,
    created_at            timestamptz NOT NULL DEFAULT now()
);
-- At most one open review per product.
CREATE UNIQUE INDEX review_queue_one_open
    ON review_queue (product_id) WHERE status = 'open';

-- ---------------------------------------------------------------------
-- 7. Kafka plumbing
-- ---------------------------------------------------------------------
-- Idempotent consumption: insert here in the same transaction as the
-- link write; a duplicate event_id makes the transaction a no-op.
CREATE TABLE processed_events
(
    event_id     text PRIMARY KEY,
    topic        text        NOT NULL,
    partition    int         NOT NULL,
    "offset"     bigint      NOT NULL,
    product_id   text,
    outcome      text        NOT NULL, -- classified | inherited | skipped | dlq
    processed_at timestamptz NOT NULL DEFAULT now()
);

-- Optional: transactional outbox. Write result events here in the same
-- transaction as category_links, then a small relay publishes them to
-- Kafka. Avoids the DB-committed-but-event-lost dual-write problem.
-- Skip it if time is short and publish directly after commit.
CREATE TABLE event_outbox
(
    outbox_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    topic        text        NOT NULL,
    key          text        NOT NULL,
    payload      jsonb       NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    published_at timestamptz
);
CREATE INDEX event_outbox_unpublished ON event_outbox (outbox_id) WHERE published_at IS NULL;

-- ---------------------------------------------------------------------
-- 8. Gold set
-- ---------------------------------------------------------------------
CREATE TABLE eval_gold
(
    product_id              text PRIMARY KEY,
    payload                 jsonb       NOT NULL,         -- fixture fed to the pipeline
    expected_category_id    text        NOT NULL REFERENCES categories (id),
    acceptable_category_ids text[] NOT NULL DEFAULT '{}', -- genuinely dual-fit products
    difficulty              text CHECK (difficulty IN ('easy', 'ambiguous', 'poor_data', 'variant_trap')),
    notes                   text,
    labelled_by             text,
    created_at              timestamptz NOT NULL DEFAULT now()
);

-- =====================================================================
-- Reference queries
-- =====================================================================

-- Candidate retrieval: top 15 active leaves for a product embedding ($1)
-- SELECT c.id, c.path, c.name, 1 - (e.embedding <=> $1) AS similarity
-- FROM category_embeddings e
-- JOIN categories c ON c.id = e.category_id
-- WHERE e.embed_model = 'nomic-embed-text' AND c.is_leaf AND c.is_active
-- ORDER BY e.embedding <=> $1
-- LIMIT 15;

-- Categories already linked to siblings / the base product ($1 = base_product_id)
-- SELECT DISTINCT l.category_id
-- FROM category_links l
-- JOIN product_snapshots s ON s.product_id = l.product_id
-- WHERE s.base_product_id = $1 OR l.product_id = $1;

-- Hierarchical match depth between chosen ($1) and gold ($2) category
-- SELECT nlevel(lca(a.path, b.path)) AS shared_depth, nlevel(b.path) AS gold_depth
-- FROM categories a, categories b WHERE a.id = $1 AND b.id = $2;

-- Critic delta for one eval run ($1): first classifier pick vs final pick
-- SELECT
--   count(*) FILTER (WHERE first.category_id <> g.expected_category_id
--                      AND r.final_category_id = g.expected_category_id) AS critic_fixed,
--   count(*) FILTER (WHERE first.category_id = g.expected_category_id
--                      AND r.final_category_id <> g.expected_category_id) AS critic_broke
-- FROM classification_runs r
-- JOIN classification_steps first ON first.run_id = r.run_id AND first.step_no = 1
-- JOIN eval_gold g ON g.product_id = r.product_id
-- WHERE r.eval_run_id = $1;
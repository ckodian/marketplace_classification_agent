# Product Category Classifier (learn-agentic)

An agent that places marketplace products into an existing category tree. It reads a product's title, description, attributes and related variants, retrieves likely categories by semantic search, asks an LLM to choose one, has a second LLM critique that choice, and revises if needed. The result is a category, a confidence score and a full trace of how it was reached.

The prototype runs locally on Ollama. The MVP adds Kafka: product events arrive, the agent decides whether to classify, and high-confidence results become category links in the destination database.

> **Status.** Prototype pipeline works end to end: taxonomy import, category embeddings, retrieval, classifier, critic, revision loop, scoring, and a sample runner. Persistence of runs, evaluation, vision, Kafka and the UI are still to do. See [Remaining work](#remaining-work).

---

## Contents

- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Load the data](#load-the-data)
- [Run the API](#run-the-api)
- [Classify products](#classify-products)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Remaining work](#remaining-work)

---

## How it works

```mermaid
flowchart LR
  P["Product (API or event)"] --> CB["Context build"]
  CB --> RET["Retrieve candidates - pgvector top-k plus sibling categories"]
  RET --> CLS["Classifier LLM"]
  CLS --> CRT["Critic LLM"]
  CRT -- "revise" --> CLS
  CRT -- "accept" --> SC["Score"]
  SC --> DEC{"Decide"}
  DEC -- "0.85 or above" --> L["Auto-link"]
  DEC -- "0.60 to 0.85" --> R["Review"]
  DEC -- "below 0.60" --> U["Unclassified"]
```

1. **Context build.** The product is normalised into a `ProductContext`. Categories already linked to its base product or siblings are looked up, since variants almost always share a category.
2. **Retrieval.** The product text is embedded with `nomic-embed-text` (query prefix `search_query: `) and compared against every active leaf category's embedding. The top 15 leaves, plus any sibling categories, become the candidates. The LLM can only choose from these.
3. **Classifier.** Picks one candidate ID, with a confidence, a rationale, evidence and alternates. An ID outside the candidates is rejected and retried once.
4. **Critic.** States its own pick first, then accepts or revises the proposal. A revision goes back to the classifier with the critique. At most 2 rounds.
5. **Score and decide.** A weighted blend of the classifier's confidence, critic agreement, sibling agreement and retrieval similarity. The weights are placeholders until the gold set exists.

Every step is recorded with its output, validity, error and latency, and returned in the response.

### Key design decisions

| Decision | Why |
|---|---|
| Retrieval before the LLM | The LLM chooses from 15 candidates, not 5,500 categories. Smaller prompts, no invented categories. |
| One pipeline for API and Kafka | Both call `classify_product()`. Kafka is another entry point, not a second codebase. |
| Embeddings keyed by `(category_id, embed_model)` | Switching embedding models can never mix vectors from two models in one search. |
| Re-embedding driven by a hash of the source text | A run can stop at any point and resume. Renaming a category re-queues it and its descendants; nothing else moves. |
| Critic commits to its own pick before judging | Small models otherwise agree with whatever they're shown. |
| Importers never call an LLM | Data loading and enrichment fail in different ways and rerun at different rates. |

---

## Project layout

Files built so far. Paths under `src/` match the running app; adjust if yours differ.

```
learn-agentic/
├── database/
│   ├── scripts/
│   │   └── apply_migrations.sh           # applies new migrations once, each in one transaction
│   ├── migrations/
│   │   ├── 001_initial_schema.sql        # all tables (schema.sql)
│   │   └── 002_*.sql                     # trigger, constraints, added columns (see Setup)
│   └── seed/
│       └── google-product-taxonomy-en-US.csv
├── scripts/
│   ├── import_categories.py              # taxonomy CSV  -> categories
│   └── import_products.py                # Amazon 2023   -> product_snapshots (+ samples file)
├── samples/
│   ├── sample_products.json              # 13 hand-written test products
│   ├── run_samples.py                    # posts samples to the API, prints a summary
│   └── results/                          # one JSON file per run, for comparing changes
├── src/
│   ├── main.py                           # FastAPI app
│   ├── config.py                         # pydantic-settings
│   ├── database.py                       # async engine, session factory, get_db
│   ├── models/categories.py              # Category, CategoryEmbedding (SQLAlchemy)
│   ├── embedding/
│   │   ├── ollama_embedder.py            # batched /api/embed, nomic task prefixes
│   │   └── category_embedding_service.py # source text, pending selection, upsert
│   ├── agents/
│   │   ├── schemas.py                    # ProductIn, ProductContext, Candidate, outputs
│   │   ├── llm.py                        # OllamaChat.complete_json (structured output)
│   │   ├── retrieval.py                  # candidate retrieval
│   │   ├── agents.py                     # ClassifierAgent, CriticAgent, v0 prompts
│   │   └── pipeline.py                   # classify_product(): the full loop
│   └── routers/
│       ├── category_backend_router.py    # POST /categories/embed, GET /categories/embed/status
│       └── products.py                   # POST /products/classify
├── requirements.txt
└── .env
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12 | Use a virtualenv. See [Troubleshooting](#troubleshooting) for the interpreter mismatch we hit. |
| PostgreSQL | 16 | With the `pgvector` and `ltree` extensions. The `pgvector/pgvector:pg16` Docker image has both. |
| Ollama | Recent | Native install on macOS. Docker on a Mac can't use the GPU. |
| Docker | Any | For Postgres now, Kafka later. |
| Node.js | 20+ | Only for the Next.js UI (not built yet). |

Models:

```bash
ollama pull qwen2.5:7b          # classifier and critic
ollama pull nomic-embed-text    # embeddings, 768 dimensions
# later, for vision: ollama pull qwen2.5vl:7b
```

Ollama settings that matter here (set before `ollama serve`, or with `launchctl setenv` for the macOS app):

```bash
OLLAMA_KEEP_ALIVE=30m          # or -1; avoids reloading models between steps
OLLAMA_MAX_LOADED_MODELS=3     # chat, embedding, vision
OLLAMA_NUM_PARALLEL=1          # best single-request latency on a laptop
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
```

---

## Setup

### 1. Python environment

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt`:

```txt
fastapi
uvicorn[standard]
pydantic-settings
python-dotenv
sqlalchemy[asyncio]
asyncpg
sqlalchemy-utils
pgvector
httpx
psycopg[binary]
```

`asyncpg` is used by the app. `psycopg` is used by the import scripts, which are synchronous. Once it works, pin versions with `python -m pip freeze > requirements.lock.txt`.

### 2. Postgres

```bash
docker run -d --name pgvector -p 5533:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=pgvector \
  pgvector/pgvector:pg16
```

### 3. Environment

`.env` in the project root:

```env
ENV=development
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5533/pgvector
CORS_ORIGINS=["http://localhost:3000"]

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:7b
OLLAMA_DEFAULT_EMBEDDED=nomic-embed-text
OLLAMA_EMBEDDING_DIMENSION=768
OLLAMA_MAX_BATCH_EMBEDDING=64
TIMEOUT_EXTERNAL_HTTP_SECONDS=180
```

List settings (`CORS_ORIGINS`) must be JSON in `.env`.

### 4. Schema

Migrations live in `database/migrations/` and are applied by `database/scripts/apply_migrations.sh`, which runs each new file once, in order, and records it in `schema_migrations`.

```bash
# see what would run
PG_URL="$DATABASE_URL" database/scripts/apply_migrations.sh --dry-run

# apply
PG_URL="$DATABASE_URL" database/scripts/apply_migrations.sh

# or through the container (connects as $PGUSER, default postgres, to $PGDATABASE, default pgvector)
DB_CONTAINER=pgvector database/scripts/apply_migrations.sh
```

- Each file runs in one transaction together with its tracking row. If a statement fails, nothing from that file is kept and it isn't recorded, so you fix the file and rerun.
- Don't put `BEGIN`/`COMMIT` or `CREATE INDEX CONCURRENTLY` in a migration file; both conflict with the wrapping transaction.
- Name files with zero-padded numbers (`001_`, `002_` … `010_`) so lexical order is the intended order.
- Never edit an applied file. The script stores a checksum and warns if an applied file has changed; add a new migration instead.
- If the database was created by Docker's `docker-entrypoint-initdb.d`, record the files already in place without running them: `--mark-up-to 001_initial_schema.sql` (or `--mark-baseline` for all current files).
- `PG_URL` accepts the app's `postgresql+asyncpg://` URL.

The changes made after the initial schema go in `002_post_initial.sql`:

```sql
-- updated_at maintained by the database, and only when a row actually changes
CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW IS DISTINCT FROM OLD THEN
        NEW.updated_at := now();
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER set_categories_updated_at
    BEFORE UPDATE ON categories
    FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at();

ALTER TABLE categories
    ADD CONSTRAINT depth_matches_path CHECK (depth = nlevel(path));

ALTER TABLE category_embeddings
    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();
```

> To reset a dev database: `DROP SCHEMA public CASCADE; CREATE SCHEMA public;` then rerun the script. This also drops `schema_migrations`, so every file is applied again.

### Database tables

| Table | Holds |
|---|---|
| `categories` | The taxonomy. `path` is an `ltree` of category IDs, e.g. `1.2.3.7385`. |
| `category_embeddings` | One vector per category per embedding model, with the source text and its hash |
| `product_snapshots` | Products as the agent saw them, keyed by `(product_id, content_hash)` |
| `image_captions` | Vision output, cached by image hash (not used yet) |
| `classification_runs`, `classification_steps` | Audit trail of every run and every LLM step (not written yet) |
| `category_links` | Product → category, the destination of the MVP |
| `review_queue` | Mid and low confidence results for a human |
| `processed_events`, `event_outbox` | Kafka idempotency and reliable publishing |
| `eval_runs`, `eval_gold` | Evaluation runs and the labelled gold set |

---

## Load the data

Three steps, in order. Each is safe to rerun.

### 1. Import the category tree

Source: the Google Product Taxonomy, as a CSV with `id, parent_id, level, full_path, level_1 … level_7`.

```bash
# validate only, no database writes
python scripts/import_categories.py \
  --file database/seed/google-product-taxonomy-en-US.csv \
  --taxonomy-version 1 --dry-run

# import
python scripts/import_categories.py \
  --file database/seed/google-product-taxonomy-en-US.csv \
  --taxonomy-version 1 \
  --dsn "$DATABASE_URL"
```

What it does:

- Derives `parent_id`, `path`, `depth` and `is_leaf` from `full_path`, and cross-checks every column that states the hierarchy (`parent_id`, `level`, `level_1 … level_7`). All errors are reported at once, with line numbers.
- Upserts in one transaction, parents first. Reruns change nothing: expect `0 inserted, 0 updated, N unchanged`.
- Never overwrites a description generated later.
- Categories missing from the file are deactivated, never deleted. If that would deactivate more than 20% of the tree, it aborts unless you pass `--allow-mass-deactivate`.
- Accepts SQLAlchemy URLs (`postgresql+asyncpg://`) as well as plain ones.

Bump `--taxonomy-version` when loading a new release.

### 2. Embed the categories

Start the API (next section), then:

```bash
curl -s -X POST localhost:8000/categories/embed | python -m json.tool
curl -s localhost:8000/categories/embed/status | python -m json.tool
```

- `POST /categories/embed` starts a background run and returns `202` with the pending count. It returns `200 nothing_to_do` when everything is current, and `409` if a run is already going. Body (optional): `{"force": false, "batch_size": 64}`.
- `GET /categories/embed/status` returns active, embedded and pending counts read live from the database, plus the last run's result.

Each category is embedded from this text (built in one SQL expression):

```
Bird Cage Bird Baths
Animals & Pet Supplies > Pet Supplies > Bird Supplies > Bird Cage Accessories > Bird Cage Bird Baths
<description, when there is one>
```

A category is pending when it has no embedding for this model, or when the hash of its current text differs from the stored one. Kill the run at any time; the next `POST` resumes. Use `force: true` only when something outside the text changes, such as the task prefix. About 5,500 categories take a few minutes on a laptop.

Check the result:

```sql
SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding))
FROM category_embeddings WHERE embed_model = 'nomic-embed-text';

-- pgAdmin shows only the first line of multi-line text; this shows all of it
SELECT category_id, replace(source_text, chr(10), ' ⏎ ') FROM category_embeddings LIMIT 5;
```

### 3. Import products (optional, for realistic test data)

Source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) item metadata. Each item is a parent product (`parent_asin`); the dataset has no variant-level metadata, so the importer can generate synthetic variants.

```bash
# streams from the URL and stops after --limit items; no full download needed
python scripts/import_products.py \
  --source https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Pet_Supplies.jsonl.gz \
  --limit 500 --dry-run

python scripts/import_products.py \
  --source https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Pet_Supplies.jsonl.gz \
  --limit 500 \
  --export-samples samples/amazon_pet.json \
  --dsn "$DATABASE_URL"
```

| Option | Default | Effect |
|---|---|---|
| `--limit` | 500 | Parent products to import |
| `--skip` | 0 | Source lines to skip, to sample further into the file |
| `--variant-share` | 0.3 | Share of parents that get synthetic variants |
| `--variants-per-parent` | 2 | Variants per selected parent; `0` disables |
| `--export-samples` | – | Also write a file `run_samples.py` can use |

- Synthetic variants differ from their parent by one attribute (colour, size or pack size), point at it through `base_product_id`, and carry `"synthetic": true` in `payload`. The same parent always gets the same variants.
- Amazon's own category path is stored in `payload` and used as the `expected` hint in the samples file. It is never part of the product sent to the agent, because it's effectively the answer.
- `--source` also takes a local `.jsonl` or `.jsonl.gz`; the format is detected from the content, and an HTML error page is rejected with a clear message.
- Licence: Amazon Reviews 2023 is published for research use. Fine for this project; check the terms before using it anywhere commercial.

---

## Run the API

```bash
source venv/bin/activate
python -m uvicorn src.main:app --port 8000
```

Use `python -m uvicorn`, not bare `uvicorn`, so the app runs in the venv's interpreter. Leave out `--reload` during embedding runs: a reload kills the background task (it resumes on the next `POST`, but it's confusing).

| Route | Purpose |
|---|---|
| `GET /health` | Liveness |
| `POST /categories/embed` | Start embedding pending categories |
| `GET /categories/embed/status` | Embedding progress |
| `POST /products/classify` | Classify one product (synchronous, 5–60 s on Ollama) |
| `GET /docs` | Swagger UI (not in production) |

---

## Classify products

### One product

```bash
curl -s -X POST localhost:8000/products/classify \
  -H 'Content-Type: application/json' \
  -d '{
        "product_id": "p1",
        "title": "Stainless Steel Clip-On Water Dish for Parrot Cages",
        "description": "Rust-proof 300 ml cup with a screw clamp that fits cage bars.",
        "attributes": {"material": "stainless steel"}
      }' | python -m json.tool
```

Only `product_id` and `title` are required. Other fields: `description`, `attributes`, `price` (`{"amount", "currency"}`), `image_urls`, `base_product_id`, `sibling_ids`.

The response contains `final_category_id`, `final_category_path`, `confidence`, `confidence_breakdown`, `decision`, all `candidates` with similarity, and every `step` with its output, validity, error and latency.

### The sample set

```bash
python samples/run_samples.py                          # 13 hand-written products
python samples/run_samples.py --only s-008 s-011       # selected ones
python samples/run_samples.py --file samples/amazon_pet.json
```

For each product it prints the chosen path, the expected path, the top retrieval hit, confidence, decision, step count, revisions, invalid IDs and time. Full responses are saved to `samples/results/<timestamp>.json` so runs can be compared after a prompt or model change.

The hand-written set covers easy cases, close siblings, a variant, ambiguous products (smartwatch, laptop sleeve, exercise mat) and products with poor data. The `expected` values are hints, not verified labels.

How to read the results:

- **Top retrieval hit wrong** → a retrieval problem. Fix the source text or add descriptions before touching prompts.
- **Revisions and invalid IDs** show the agents at work. A critic that never revises on ambiguous cases is probably just agreeing.
- **Poor-data products** should land on `review` or `unclassified`. A confident `auto_linked` there means scoring is too generous.

### Results so far (v0 prompts, qwen2.5:7b)

13 samples: 6 auto-linked, 5 failed, 2 unclassified, about 9 s average. The failures were not checked to the root yet; the leading suspects are numeric category IDs returned as JSON numbers (rejected by the `str` fields) and the critic timing out. One encouraging case: for "$50 Store Gift Card", retrieval ranked "Blank ID Cards" first but the classifier chose "Gift Cards & Certificates".

---

## Configuration reference

| Variable | Example | Used by |
|---|---|---|
| `ENV` | `development` | Hides `/docs` in production |
| `DATABASE_URL` | `postgresql+asyncpg://…:5533/pgvector` | App; import scripts accept it too |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | JSON list. Avoid `["*"]` with credentials. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Embedder, chat |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:7b` | Classifier, critic |
| `OLLAMA_DEFAULT_EMBEDDED` | `nomic-embed-text` | Embeddings; part of the embedding key |
| `OLLAMA_EMBEDDING_DIMENSION` | `768` | Must match the model and `vector(768)` |
| `OLLAMA_MAX_BATCH_EMBEDDING` | `64` | Inputs per `/api/embed` call |
| `TIMEOUT_EXTERNAL_HTTP_SECONDS` | `180` | HTTP timeout for Ollama |

Changing the embedding model or its dimension means a new `vector(n)` column size and a full re-embed.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` for a package you installed | `uvicorn` on your PATH is the global one, not the venv's | `python -m pip install -r requirements.txt`, then `python -m uvicorn …` |
| `SettingsError: error parsing value for field "cors_origins"` | List settings are parsed as JSON | `CORS_ORIGINS=["http://localhost:3000"]` |
| `async_generator object get_db … is not a callable object` | `Depends(get_db())` | `Depends(get_db)`, without parentheses |
| `'_AsyncGeneratorContextManager' object is not an async iterator` | `get_db` decorated with `@asynccontextmanager` | Remove the decorator from FastAPI dependencies |
| `relation "categories" already exists` on a migration | A file was applied outside the script, or before it ran transactionally | Reset the dev schema and rerun `apply_migrations.sh`, or record it with `--mark-up-to` |
| Inserting vectors fails with asyncpg | pgvector type not registered | Register `pgvector.asyncpg.register_vector` on connect (below) |
| `BadGzipFile: … b'<!'` | The download saved an HTML error page | Use the URL from the dataset's project page; `curl -fL` |
| Critic step errors with `ReadTimeout('')` | Critic call exceeds the HTTP timeout | Raise the read timeout; shorten the critic prompt |
| A call never returns | Structured output stuck generating whitespace or repetition | Always set `num_predict` (below) |
| pgAdmin shows only the category name in `source_text` | The grid shows the first line only | `replace(source_text, chr(10), ' ⏎ ')` |

Registering pgvector with asyncpg:

```python
from pgvector.asyncpg import register_vector
from sqlalchemy import event

@event.listens_for(engine.sync_engine, "connect")
def _register_vector(dbapi_conn, _):
    dbapi_conn.run_async(register_vector)
```

Capping generation so every LLM call ends (in `OllamaChat.complete_json`):

```python
"options": {"temperature": 0, "num_predict": 600, "repeat_penalty": 1.1},
```

Finding out where Ollama spends the time: log `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count` and `eval_duration` from each response (all durations in nanoseconds). Check `ollama ps` shows `100% GPU`.

---

## Remaining work

In build order. Each item has a "done when" so progress is unambiguous.

### Now: make the prototype reliable

- [ ] **Stop runaway LLM calls.** Add `num_predict` and `repeat_penalty`; replay the critic request with `stream: true` to confirm the cause.
  *Done when:* no call runs longer than the timeout, and a capped reply fails as a clean `LLMError`.
- [ ] **Accept numeric IDs.** Add a `mode="before"` validator to `category_id`, `own_pick` and `suggested_category_id` that converts numbers to strings and strips an `id=` prefix.
  *Done when:* the 5 failed samples either succeed or fail for a different, logged reason.
- [ ] **Log failures in the pipeline.** The critic's `except LLMError` currently records the error silently.
- [ ] **Shrink prompts.** Compact JSON, shorter candidate descriptions, top 8 candidates for the critic plus the proposed one, shorter `rationale` and `detail` limits.
  *Done when:* critic calls finish in well under the timeout, and accuracy on the samples doesn't drop.
- [ ] **Compare LM Studio** as a second `LLMClient` implementation (OpenAI-compatible API, `response_format: json_schema`, `max_tokens`). Same model and quantisation as Ollama for a fair comparison.

### Persistence

- [ ] **Write `classification_runs` and `classification_steps`** for every run, including provider, model, prompt versions, candidates, latency and tokens.
- [ ] **Write `category_links`** for `auto_linked` results, and `review_queue` rows for `review` and `unclassified`.
  This also makes the variant path work: once a parent is linked, its variants see that category through `related_category_ids`.
- [ ] **Save the product snapshot** received through the API, with its `content_hash`.

### Vision attribute extraction

- [ ] Caption each image with a vision model (`qwen2.5vl:7b` on Ollama; Claude handles images natively) into a short description plus attributes (product type, colour, material).
- [ ] Cache in `image_captions` by image hash, so an image is captioned once. Replace the URL-hash placeholder in `import_products.py` with a hash of the downloaded bytes.
- [ ] Add the caption to `ProductContext.image_caption`, which is already included in the retrieval text and the prompts.
- [ ] Never send raw images into the critic loop; only the caption.

*Done when:* a product with a poor title and a good image classifies better with captioning than without, measured on the gold set.

### Evaluation

- [ ] **Gold set.** Label 50–100 products in `eval_gold` against the seeded taxonomy, using the Amazon category hint to narrow down. Include hard cases: ambiguous products, products that fit two categories (`acceptable_category_ids`), poor titles, and a variant whose base product is mislabelled.
- [ ] **Eval script** that runs the gold set under one `eval_runs` row and reports:
  - top-1 and top-3 accuracy
  - hierarchical accuracy (shared path depth with the gold category, via `ltree`'s `lca()`)
  - critic delta: how often the critic fixed a wrong answer versus broke a right one
  - calibration: accuracy per confidence bucket
  - auto-link precision at the chosen threshold
  - latency and tokens per product
- [ ] **Calibrate scoring.** Make the critic signal three-state (agree, disagree, missing). Replace raw similarity with rank or margin. Add gates (critic disagreed → at most `review`). Set the auto-link threshold where accuracy reaches the target.
- [ ] **Decide when to run the critic:** on every product, or only on uncertain ones. Measure both.

*Done when:* one table compares prompt versions and models on the same gold set.

### Event schema

- [ ] Define the input events, modelled on commercetools-style product messages:
  - `ProductCreated`, `ProductUpdated`, `ProductDeleted`, `VariantAdded`
  - `CategoryCreated`, `CategoryUpdated`, `CategoryDeleted` (taxonomy changes from the source system)
- [ ] Common envelope: `event_id`, `event_type`, `occurred_at`, `source`, `schema_version`, `payload`. Key: `base_product_id` for product events, so a product and its variants stay in order on one partition.
- [ ] Output events: `classification.completed` and `classification.review`, with `product_id`, `category_id`, `confidence`, `decision` and `run_id`.
- [ ] Validate events with Pydantic models; a schema registry can come later.

### Kafka setup

- [ ] Add Kafka to Docker Compose: `apache/kafka` in KRaft mode (no ZooKeeper).
- [ ] Topics:

  | Topic | Key | Purpose |
    |---|---|---|
  | `product.events` | `base_product_id` | Product changes in |
  | `category.events` | `category_id` | Taxonomy changes in |
  | `classification.completed` | `product_id` | Results out |
  | `classification.review` | `product_id` | Human review queue |
  | `product.events.dlq` | `product_id` | Messages that failed after retries |

- [ ] Python client: `confluent-kafka`. A test producer that sends events from sample files.

### MVP: event-driven classification

- [ ] **Consumer** with auto-commit off; commit offsets only after the database write.
- [ ] **Idempotency:** insert into `processed_events` in the same transaction as the link write; a duplicate `event_id` becomes a no-op.
- [ ] **Gate:** decide whether to classify at all.
  - `ProductCreated` → classify
  - `ProductUpdated` → classify only if `content_hash` changed (price-only updates skip)
  - `VariantAdded` with a high-confidence parent link → inherit the category, no LLM call (`source = 'inherited'`)
  - Existing `manual` link → never overwrite
- [ ] **Category events:** upsert the category with the same validation as the importer, then re-embed through the same hash-driven loop. Batch inside the consumer (up to ~50 events or ~1 s). Renames re-queue descendants automatically.
- [ ] **Outbox:** write result events to `event_outbox` in the same transaction as `category_links`; a relay publishes them to Kafka. Avoids a link committed without its event.
- [ ] **Retries and DLQ:** bounded retries, then `product.events.dlq`.
- [ ] **Single-category ingest API:** `PUT /categories/{id}` (idempotent; server derives `path`), embedding synchronously and reporting stale descendants.

*Done when:* an event on `product.events` produces a `category_links` row and a `classification.completed` message, and replaying the same event changes nothing.

### Cloud models

- [ ] `AnthropicProvider`: structured output through tool use.
- [ ] `BedrockProvider`: Converse API through boto3.
- [ ] Keep `nomic-embed-text` as the embedding model so retrieval stays identical, and rerun the eval for a like-for-like comparison.

### UI (Next.js)

- [ ] Upload page: form or pasted JSON, plus images.
- [ ] Trace view: candidates with similarity, classifier pick and rationale, critic verdict and issues, confidence breakdown.
- [ ] Review queue: accept or override; an override writes a `manual` link and can add the product to `eval_gold`.

### Production hardening (after the MVP)

- [ ] Replace in-memory embedding run tracking with an `embedding_jobs` table, and mark interrupted jobs on startup.
- [ ] Move `category_links` to its own schema or database to model the real destination.
- [ ] Category moves: rewrite the subtree's `path` in one `ltree` statement.
- [ ] Prompt files under `prompts/` with versions recorded on every run.
- [ ] Pin dependency versions.

---

## Data sources

- [Google Product Taxonomy](https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt): about 5,500 categories, 7 levels.
- [Shopify Standard Product Taxonomy](https://github.com/Shopify/product-taxonomy): larger, with attributes per category. A possible later upgrade.
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) (McAuley Lab): product metadata by category, research use.
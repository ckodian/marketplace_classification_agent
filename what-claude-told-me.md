The plan below is built to fit 8 hours: a working prototype by about hour 4.5, then the event-driven MVP, then the Ollama-to-Claude comparison. Two choices carry most of the quality: shortlist categories by retrieval before the LLM sees them, and treat already-classified siblings as strong evidence. With those in place, the LLM chooses from a short list of candidates instead of searching the whole tree.

1. Architecture
```mmd
flowchart LR
    subgraph ENTRY["Entry points"]
        UI["Next.js UI - upload, trace, review"]
        API["FastAPI"]
        PE[["Kafka topic: product.events"]]
        CON["Kafka consumer"]
    end

    subgraph PIPE["Classification pipeline - shared code path"]
        direction TB
        G["1. Gate - skip, inherit or classify"]
        CB["2. Context build - product, image caption, siblings"]
        RET["3. Retrieve candidates - pgvector top-k plus sibling categories"]
        CLS["4. Classifier LLM"]
        CRT["5. Critic LLM - accept or revise, max 2 loops"]
        SC["6. Confidence scoring"]
        DEC{"7. Decide"}
    end

    subgraph LLM["LLM providers"]
        OL["Ollama"]
        AN["Anthropic"]
        BR["Bedrock"]
    end

    subgraph DB["Postgres with pgvector"]
        T1[("categories and category_embeddings")]
        T2[("product_snapshots and processed_events")]
        T3[("classification_runs")]
        T4[("category_links")]
        T5[("review_queue")]
    end

    subgraph OUT["Kafka output topics"]
        OC[["classification.completed"]]
        ORV[["classification.review"]]
        DLQ[["product.events.dlq"]]
    end

    UI --> API
    API --> CB
    PE --> CON
    CON --> G

    G --> CB
    CB --> RET
    RET --> CLS
    CLS --> CRT
    CRT -- "revise" --> CLS
    CRT -- "accept" --> SC
    SC --> DEC

    CB <--> T2
    RET <--> T1
    SC --> T3

    CLS -.-> OL
    CLS -.-> AN
    CLS -.-> BR
    CRT -.-> OL
    CRT -.-> AN
    CRT -.-> BR

    DEC -- "confidence 0.85 or above" --> T4
    T4 --> OC
    DEC -- "0.60 to 0.85" --> T5
    DEC -- "below 0.60" --> T5
    T5 --> ORV
    CON -- "failures after retries" --> DLQ
```

The key point is that the prototype (UI upload) and the MVP (Kafka consumer) call the same classify(product_context) function. Kafka is just another entry point, which keeps your 8 hours from splitting into two codebases.

2. The agent pipeline

Step 1: Gate (MVP only). Decide whether a product needs classifying at all:

ProductCreated → classify
ProductUpdated → classify only if the hash of the classification-relevant fields changed (title, description, image URLs, key attributes). Price changes alone are skipped.
VariantAdded where the base product already has a high-confidence link → inherit the category and skip the LLM, writing source = inherited. This shortcut will cover a large share of marketplace volume.
If the product is already linked with status manual → never overwrite it.

You know commercetools’ message model, so mirroring its event shapes (ProductCreated, ProductPublished, and so on) in your payload schema gives you realistic events for free.

Step 2: Context build. Normalise everything into one ProductContext: title, description, attributes, price band, image caption, and the categories of any related products (base product, siblings, children).

Images: for the prototype, run a vision model (Ollama qwen2.5-vl or llava; Claude handles images natively) once to produce a text description plus extracted attributes, and cache it by image hash. Don’t send raw images into the critique loop.

Step 3: Candidate retrieval. Embed each leaf category as full path + name + description, for example “Home > Kitchen > Cookware > Frying Pans: …”.

Embed the product context and take the top 10–15 leaves by cosine similarity (pgvector).
Union in the categories of classified siblings and the base product. These are your strongest signal.
Keep the embedding model fixed, for example nomic-embed-text via Ollama, even when you switch the LLM to Claude. Changing embedding models means re-indexing and invalidates your comparison.

Step 4: Classifier. Use structured output: JSON schema with Ollama’s format, tool use with Claude.

{ category_id, confidence (0–1), rationale, evidence_used[], alternates: [{category_id, reason}] }

The prompt rule is to pick only from the candidate IDs and to prefer the deepest correct leaf. Validate the output: an ID not in the candidate list is a hard failure, which triggers one retry and then a fallback.

Step 5: Critic (reflection). A separate prompt, and ideally a separate persona, that receives the product, the candidates with their descriptions, and the classifier output. It checks:

whether the category is a leaf or too generic
whether a sibling or alternate fits better
whether the choice contradicts the classified siblings
whether the rationale relies on evidence that isn’t actually present (hallucinated attributes)

It returns {verdict: accept|revise, revised_category_id?, issues[], confidence}. Cap the loop at 2 revisions. More iterations rarely help and make debugging harder.

Step 6: Confidence scoring. Don’t trust the LLM’s self-reported number alone. Blend it with independent signals:

Signal	Why it helps
Classifier and critic agree	Two independent judgments matching is the strongest signal
Agreement with classified siblings	Variants usually share a category
Retrieval rank and similarity of the chosen category	Low similarity suggests a stretch
LLM self-score	Useful but over-confident, so give it a low weight

A simple weighted sum is fine for today. Calibrate the weights against the eval set later.

Step 7: Decide.

≥ 0.85: write the category_links row (status auto) and emit classification.completed.
0.60–0.85: write to review_queue and emit classification.review.
< 0.60: mark the product unclassified and route it to review.
3. Data model (Postgres)
   categories: id, parent_id, path (ltree gives you subtree queries cheaply), name, description, logo_url, is_leaf, version
   category_embeddings: category_id, model, vector, text_hash
   product_snapshots: product_id, base_product_id, payload jsonb, content_hash, image_caption
   classification_runs: the audit trail, with run_id, product_id, content_hash, model, prompt_version, candidates jsonb, classifier_out, critic_out, final_category, confidence, latency_ms, tokens, created_at. This table is also your eval data source.
   category_links: product_id, category_id, confidence, source (llm|inherited|manual), status, run_id. Put a unique constraint on product_id.
   processed_events: event_id primary key. This gives you idempotent consumption.
   eval_gold: product_id, expected_category_id, notes
4. Kafka design
   Topic	Key	Purpose
   product.events	base_product_id	Input. Keying by base product means variants are processed in order, so a sibling’s category exists before the next variant arrives.
   classification.completed	product_id	Downstream consumers (search, PIM)
   classification.review	product_id	Human-in-the-loop queue
   product.events.dlq	product_id	Poison messages and LLM failures after retries

Consumer rules:

Turn off auto-commit and commit after the database write.
Dedupe with processed_events in the same transaction as the link write.
Use a small worker pool. Ollama is the bottleneck, so concurrency of 1–2 locally is realistic.

Use apache/kafka in KRaft mode in Docker Compose (no ZooKeeper) and confluent-kafka as the Python client.

5. LLM provider abstraction

Define one interface and three implementations:

LLMProvider.complete_json(system, messages, schema) -> dict
LLMProvider.describe_image(bytes) -> str
Embedder.embed(texts) -> list[vector]   # keep separate, keep fixed
OllamaProvider (qwen2.5:7b/14b or llama3.1:8b). In my experience Qwen follows JSON schemas better, but test both.
AnthropicProvider (direct API)
BedrockProvider (Converse API through boto3; the tool use format differs slightly from the direct API)

Version your prompts as files, for example prompts/classifier_v1.md, and log prompt_version on every run. That is what makes before-and-after comparisons meaningful.

6. Evaluation

Build a gold set of 50–100 products labelled by you. Include deliberately hard cases: ambiguous items, products that genuinely sit in two categories, a variant whose base product is mislabelled, and items with poor titles.

Metrics, computed in one script that reads classification_runs:

Top-1 accuracy and top-3 (is the gold category among the chosen category plus alternates?)
Hierarchical accuracy: how many levels deep the chosen path matches the gold path. A wrong leaf under the right parent is a much smaller miss than the wrong department.
Critic delta: how often the critic fixed a wrong answer versus broke a right one. This tells you whether reflection is actually earning its cost.
Calibration: accuracy per confidence bucket. If 0.9 confidence is only 70% accurate, your thresholds are wrong.
Auto-link precision at your threshold. This is the business metric: a wrong auto-link is worse than a review.
Latency and cost per product.

Run the same eval against Ollama, Claude direct, and Bedrock, and compare the results in a table.

7. Next.js UI (keep it thin)
   Upload page: a form for title, description, images, price and related product IDs, or paste JSON.
   Result view: the candidate list with similarity scores, the classifier pick and rationale, the critic verdict and issues, the final decision and confidence breakdown.
   Review queue: accept or override. An override writes a manual link and optionally adds the item to eval_gold.

The trace view is the most valuable screen for learning, because it shows why the agent decided what it did.

8. Eight-hour schedule
   Time	Block	Done when
   0:00–0:45	Docker Compose (Kafka, Postgres with pgvector), schema, seed a category tree of 100–300 leaves, embed categories	SELECT of top-k similar categories for a text query works
   0:45–2:15	Provider abstraction (Ollama), context builder, retrieval, classifier with schema validation, CLI runner	python classify.py sample.json returns valid JSON
   2:15–3:15	Critic, revise loop, confidence scoring, classification_runs logging	Full trace is written to the database
   3:15–4:15	FastAPI endpoints and Next.js upload and trace page	Prototype done
   4:15–5:00	Gold set and eval script; Ollama baseline numbers	First metrics table
   5:00–6:30	Kafka producer (test event generator), consumer, gate, inheritance shortcut, link writer, result topics, DLQ, idempotency	Events in, links out, end to end
   6:30–7:15	Anthropic and Bedrock providers; rerun the eval	Comparison table
   7:15–8:00	Buffer: threshold tuning, review queue UI, demo	MVP done

If you fall behind, cut in this order: review queue UI → Bedrock (keep direct Anthropic) → image captioning (text only) → hierarchical metric.

Don’t cut the gold set or the run logging. Without them you can’t evaluate the critic, which is the whole point.

9. Pitfalls to watch for
   Small local models invent category IDs. Validate against the candidate list and fail closed.
   The critic agrees with everything. Small models defer to the first answer. Have the critic first state its own pick, then compare, and give it the alternates explicitly.
   Category descriptions are often empty in real trees. Your seed data needs decent descriptions, or retrieval quality collapses. Generating descriptions with an LLM once and storing them is a legitimate step.
   Tree drift. Store category.version on each link so that re-categorising after a taxonomy change becomes a query rather than archaeology.
   Ollama throughput. Expect a few seconds per call and three to five calls per product. Don’t flood the consumer with test events.

Three decisions to make before you start:

Orchestration: plain Python functions (better for learning, easier to debug) or LangGraph (closer to what you’d ship, more to learn today). For 8 hours, I’d go plain.
Seed category tree: a public one such as the Google Product Taxonomy (good size and depth) or something closer to your Contentful or commercetools work.
Whether to run the critic on every product or only on mid-confidence cases. Running it on everything gives cleaner eval data today. Running it selectively is how you’d control cost in production.

I can turn this into a doc you can keep open and tick through as you go. When you reach a block you’d like a second pair of eyes on, such as the prompts, the schema or the consumer commit logic, bring it here.
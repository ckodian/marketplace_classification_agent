"""Import product metadata from Amazon Reviews 2023 (McAuley Lab) into product_snapshots.

Each source item is a parent product (one parent_asin). The dataset has no
variant-level metadata, so the importer can also generate synthetic variants
for a share of parents: same product, one changed attribute (colour, size or
pack size), base_product_id pointing at the parent. These are flagged with
"synthetic": true in payload so they're never mistaken for real data.

Source: a local .jsonl / .jsonl.gz file, or a URL that is streamed and closed
after --limit items (no full download). Metadata files are named
meta_<Category>.jsonl(.gz), e.g. meta_Pet_Supplies.

Usage:
    # preview, no database
    python import_products.py --source meta_Pet_Supplies.jsonl.gz --limit 500 --dry-run

    # import and write a samples file for run_samples.py
    python import_products.py --source meta_Pet_Supplies.jsonl.gz --limit 500 \
        --export-samples samples/amazon_pet.json

Amazon's own `categories` field is kept in payload and used as the `expected`
hint in the samples file. It is never put in the product the agent sees:
it is effectively the answer.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import zlib
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from pathlib import Path

DEFAULT_CURRENCY = "USD"
MAX_TITLE = 500
MAX_DESCRIPTION = 4000
MAX_ATTRIBUTES = 12
MAX_ATTR_VALUE = 200
MAX_IMAGES = 3

# details keys that are noise for classification (rankings, dates, ids)
SKIP_DETAIL_KEYS = {
    "best sellers rank", "customer reviews", "date first available", "asin",
    "is discontinued by manufacturer", "upc", "item model number",
    "domestic shipping", "international shipping", "batteries required",
}

VARIANT_AXES: list[tuple[str, list[str]]] = [
    ("colour", ["Black", "Grey", "Blue", "Red", "White"]),
    ("size", ["Small", "Medium", "Large", "X-Large"]),
    ("pack_size", ["Pack of 2", "Pack of 4", "Pack of 6"]),
]


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="import_products", description=__doc__.split("\n\n")[0])
    p.add_argument("--source", required=True,
                   help="Path or http(s) URL to a meta_<Category>.jsonl or .jsonl.gz file.")
    p.add_argument("--limit", type=int, default=500,
                   help="Parent products to import (after filtering). Default 500.")
    p.add_argument("--skip", type=int, default=0,
                   help="Source lines to skip first, to sample from further into the file.")
    p.add_argument("--variant-share", type=float, default=0.3,
                   help="Share of parents that get synthetic variants, 0-1. Default 0.3.")
    p.add_argument("--variants-per-parent", type=int, default=2,
                   help="Synthetic variants per selected parent. 0 disables. Default 2.")
    p.add_argument("--export-samples", type=Path,
                   help="Also write a run_samples.py-compatible JSON file.")
    p.add_argument("--dsn", default=os.environ.get("DATABASE_URL"),
                   help="Postgres connection string. Defaults to $DATABASE_URL.")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse and print a summary; don't write to the database.")
    args = p.parse_args(argv)

    if not args.source.startswith(("http://", "https://")) and not Path(args.source).is_file():
        p.error(f"--source not found: {args.source}")
    if args.limit < 1:
        p.error("--limit must be at least 1")
    if not 0 <= args.variant_share <= 1:
        p.error("--variant-share must be between 0 and 1")
    if not args.dry_run and not args.dsn:
        p.error("--dsn or $DATABASE_URL is required unless --dry-run is set")
    if args.dsn:
        args.dsn = re.sub(r"^postgres(?:ql)?\+\w+://", "postgresql://", args.dsn)
    return args


# ---------------------------------------------------------------------------
# Reading the source
# ---------------------------------------------------------------------------

def _iter_url_lines(url: str) -> Iterator[str]:
    """Stream a remote file line by line; the caller stops early, which closes it."""
    import httpx

    gz = url.split("?")[0].endswith(".gz")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as res:
        res.raise_for_status()
        decomp = zlib.decompressobj(16 + zlib.MAX_WBITS) if gz else None
        buf = b""
        for chunk in res.iter_raw():
            buf += decomp.decompress(chunk) if decomp else chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                yield line.decode("utf-8")
        if buf:
            yield buf.decode("utf-8")


def iter_source(source: str) -> Iterator[str]:
    if source.startswith(("http://", "https://")):
        yield from _iter_url_lines(source)
        return
    # Decide by content, not extension: a failed download is often an HTML
    # error page saved under the .gz name.
    with open(source, "rb") as f:
        head = f.read(512)
    if head[:2] == b"\x1f\x8b":
        opener = gzip.open
    elif head.lstrip()[:1] in (b"<",):
        raise SystemExit(
            f"{source} is an HTML page, not product data. The download URL returned an "
            f"error page. First bytes: {head[:120]!r}"
        )
    else:
        opener = open
    with opener(source, "rt", encoding="utf-8") as f:
        yield from f


# ---------------------------------------------------------------------------
# Normalising one source item
# ---------------------------------------------------------------------------

def _text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = " ".join(str(v).strip() for v in value if v and str(v).strip())
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def _price(value) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    s = str(value).strip().replace("$", "").replace(",", "")
    if s.lower() in {"", "none", "null", "nan"}:
        return None
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None      # ranges like "12.99 - 19.99" are skipped
    return d.quantize(Decimal("0.01")) if d >= 0 else None


def _image_urls(images) -> list[str]:
    """Handles both shapes: list of {hi_res, large, ...} or {hi_res: [...], large: [...]}."""
    urls: list[str] = []
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                urls.append(img.get("hi_res") or img.get("large") or "")
    elif isinstance(images, dict):
        his = images.get("hi_res") or []
        larges = images.get("large") or []
        for i in range(max(len(his), len(larges))):
            hi = his[i] if i < len(his) else None
            lg = larges[i] if i < len(larges) else None
            urls.append(hi or lg or "")
    return [u for u in urls if isinstance(u, str) and u.startswith(("http://", "https://"))][:MAX_IMAGES]


def _attributes(details, store: str | None) -> dict:
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    if not isinstance(details, dict):
        details = {}

    attrs: dict = {}
    if store:
        attrs["brand"] = store[:MAX_ATTR_VALUE]
    for key, value in details.items():
        k = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
        if not k or str(key).strip().lower() in SKIP_DETAIL_KEYS or k in attrs:
            continue
        if isinstance(value, (int, float, bool)):
            attrs[k] = value
        elif isinstance(value, str) and value.strip():
            attrs[k] = value.strip()[:MAX_ATTR_VALUE]
        elif isinstance(value, list) and all(isinstance(v, str) for v in value) and value:
            attrs[k] = [v[:MAX_ATTR_VALUE] for v in value[:5]]
        if len(attrs) >= MAX_ATTRIBUTES:
            break
    return attrs


def normalise(item: dict) -> dict | None:
    """Return a product dict, or None if the item is too thin to classify."""
    pid = _text(item.get("parent_asin"))
    title = _text(item.get("title"))
    if not pid or not title:
        return None

    description = _text(item.get("description")) or _text(item.get("features"))
    if description:
        description = description[:MAX_DESCRIPTION]

    return {
        "product_id": pid,
        "base_product_id": None,
        "title": title[:MAX_TITLE],
        "description": description,
        "attributes": _attributes(item.get("details"), _text(item.get("store"))),
        "price": _price(item.get("price")),
        "image_urls": _image_urls(item.get("images")),
        "source_categories": [c for c in (item.get("categories") or []) if isinstance(c, str)],
        "main_category": _text(item.get("main_category")),
        "synthetic": False,
        "raw": item,
    }


# ---------------------------------------------------------------------------
# Synthetic variants
# ---------------------------------------------------------------------------

def _stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def make_variants(parent: dict, share: float, per_parent: int) -> list[dict]:
    """Deterministic: the same parent always gets the same variants."""
    if per_parent <= 0 or (_stable_int(parent["product_id"]) % 1000) / 1000 >= share:
        return []

    h = _stable_int(parent["product_id"] + ":axis")
    axis, options = VARIANT_AXES[h % len(VARIANT_AXES)]
    stated = set(parent["attributes"]) | ({"colour"} if "color" in parent["attributes"] else set())
    if axis in stated:   # parent already states it; try the next axis
        axis, options = VARIANT_AXES[(h + 1) % len(VARIANT_AXES)]
    start = h % len(options)
    chosen = [options[(start + i) % len(options)] for i in range(min(per_parent, len(options)))]

    variants = []
    for value in chosen:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        variants.append({
            **parent,
            "product_id": f"{parent['product_id']}-{slug}",
            "base_product_id": parent["product_id"],
            "title": f"{parent['title']} - {value}"[:MAX_TITLE],
            "description": None,                     # variants are usually thin
            "attributes": {**parent["attributes"], axis: value},
            "image_urls": parent["image_urls"][:1],
            "synthetic": True,
            "raw": None,
        })
    return variants


# ---------------------------------------------------------------------------
# Hashing and database rows
# ---------------------------------------------------------------------------

def image_hash(url: str) -> str:
    # Placeholder until images are downloaded: hash of the URL, not the bytes.
    return hashlib.sha256(url.encode()).hexdigest()


def content_hash(p: dict) -> str:
    """Only fields that affect classification. Price changes don't change it."""
    basis = {
        "title": p["title"],
        "description": p["description"],
        "attributes": p["attributes"],
        "images": sorted(image_hash(u) for u in p["image_urls"]),
    }
    return hashlib.sha256(
        json.dumps(basis, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def to_row(p: dict) -> dict:
    payload = {
        "source": "amazon-reviews-2023",
        "synthetic": p["synthetic"],
        "main_category": p["main_category"],
        "source_categories": p["source_categories"],   # label hint; never shown to the agent
        "image_urls": p["image_urls"],
        "item": p["raw"],
    }
    return {
        "product_id": p["product_id"],
        "content_hash": content_hash(p),
        "base_product_id": p["base_product_id"],
        "title": p["title"],
        "description": p["description"],
        "attributes": json.dumps(p["attributes"], ensure_ascii=False),
        "price_amount": p["price"],
        "price_currency": DEFAULT_CURRENCY if p["price"] is not None else None,
        "image_hashes": [image_hash(u) for u in p["image_urls"]],
        "payload": json.dumps(payload, ensure_ascii=False, default=str),
    }


INSERT_SQL = """
             INSERT INTO product_snapshots
             (product_id, content_hash, base_product_id, title, description, attributes,
              price_amount, price_currency, image_hashes, payload)
             VALUES
                 (%(product_id)s, %(content_hash)s, %(base_product_id)s, %(title)s, %(description)s,
                  %(attributes)s::jsonb, %(price_amount)s, %(price_currency)s, %(image_hashes)s,
                  %(payload)s::jsonb)
                 ON CONFLICT (product_id, content_hash) DO NOTHING
RETURNING product_id \
             """


def write_db(dsn: str, products: list[dict]) -> int:
    import psycopg

    rows = [to_row(p) for p in products]
    inserted = 0
    with psycopg.connect(dsn) as conn, conn.transaction(), conn.cursor() as cur:
        cur.executemany(INSERT_SQL, rows, returning=True)
        while True:
            inserted += len(cur.fetchall())
            if not cur.nextset():
                break
    return inserted


# ---------------------------------------------------------------------------
# Samples export (run_samples.py format)
# ---------------------------------------------------------------------------

def to_product_in(p: dict, siblings: list[str]) -> dict:
    out = {
        "product_id": p["product_id"],
        "title": p["title"],
        "description": p["description"],
        "attributes": p["attributes"],
        "image_urls": p["image_urls"],
        "base_product_id": p["base_product_id"],
        "sibling_ids": siblings[:50],
    }
    if p["price"] is not None:
        out["price"] = {"amount": str(p["price"]), "currency": DEFAULT_CURRENCY}
    return {k: v for k, v in out.items() if v not in (None, [], {})}


def export_samples(path: Path, products: list[dict]) -> None:
    children: dict[str, list[str]] = {}
    for p in products:
        if p["base_product_id"]:
            children.setdefault(p["base_product_id"], []).append(p["product_id"])

    samples = []
    # Parents first, so a parent can be linked before its variants are classified
    for p in sorted(products, key=lambda p: p["base_product_id"] is not None):
        siblings = [c for c in children.get(p["base_product_id"] or "", []) if c != p["product_id"]]
        expected = " > ".join(p["source_categories"]) or p["main_category"] or "unknown"
        samples.append({
            "case": "amazon_variant" if p["synthetic"] else "amazon",
            "expected": f"Amazon: {expected}",
            "product": to_product_in(p, siblings),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    parents: list[dict] = []
    seen: set[str] = set()
    read = skipped_thin = skipped_bad = 0

    for i, line in enumerate(iter_source(args.source)):
        if i < args.skip:
            continue
        line = line.strip()
        if not line:
            continue
        read += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            skipped_bad += 1
            continue
        p = normalise(item)
        if p is None or p["product_id"] in seen:
            skipped_thin += 1
            continue
        seen.add(p["product_id"])
        parents.append(p)
        if len(parents) >= args.limit:
            break

    variants = [v for p in parents for v in make_variants(p, args.variant_share, args.variants_per_parent)]
    products = parents + variants

    with_desc = sum(1 for p in parents if p["description"])
    with_price = sum(1 for p in parents if p["price"] is not None)
    with_cats = sum(1 for p in parents if p["source_categories"])
    print(f"read {read} lines: {len(parents)} parents, {len(variants)} synthetic variants, "
          f"{skipped_thin} skipped (thin or duplicate), {skipped_bad} unparseable")
    print(f"parents with description {with_desc}, price {with_price}, "
          f"Amazon categories {with_cats}")
    for p in (parents[:2] + variants[:1]):
        print(f"  {p['product_id']:<22} base={p['base_product_id'] or '-':<12} "
              f"{p['title'][:70]!r}  attrs={list(p['attributes'])[:5]}")

    if args.export_samples:
        export_samples(args.export_samples, products)
        print(f"wrote {len(products)} samples to {args.export_samples}")

    if args.dry_run:
        print("Dry run: nothing written to the database.")
        return 0

    inserted = write_db(args.dsn, products)
    print(f"product_snapshots: {inserted} inserted, {len(products) - inserted} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
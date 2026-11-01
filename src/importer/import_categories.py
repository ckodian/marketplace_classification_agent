"""Import a product taxonomy CSV into the categories table.

Expected CSV columns:
    id, full_path                              required
    parent_id, level, level_1..level_N,
    name, description                          optional

full_path uses " > " between segments, e.g. "Home & Garden > Kitchen & Dining > Cookware".
The category name is the last segment of full_path, which is also level_<level>
in files that have level columns. Every column that states the name or the
hierarchy (name, level, level_N, parent_id) is cross-checked against full_path.

The importer is pure CSV -> Postgres. It never calls an LLM. Descriptions and
embeddings are filled later by a separate enrichment script.

Usage:
    python import_categories.py --file data/taxonomy.csv --taxonomy-version 1 --dry-run
    python import_categories.py --file data/taxonomy.csv --taxonomy-version 1
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = {"id", "full_path"}
LEVEL_COLUMN = re.compile(r"^level_(\d+)$")
PATH_SEPARATOR = " > "
LTREE_LABEL = re.compile(r"^[A-Za-z0-9_-]{1,255}$")  # hyphens need Postgres 16+

# Abort if the file would deactivate more than this share of active categories.
MAX_DEACTIVATE_RATIO = 0.2


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CategoryRow:
    """One validated CSV row, before tree context is known."""
    line_no: int
    id: str
    name: str
    full_path: str
    segments: tuple[str, ...]
    parent_id: str | None          # as given in the CSV; checked against full_path
    description: str | None


@dataclass
class Node:
    """A category with everything derived from the tree."""
    id: str
    name: str
    description: str | None
    parent_id: str | None
    path: str                      # ltree as text, e.g. "536.638.4515"
    depth: int
    is_leaf: bool
    line_no: int
    csv_parent_id: str | None


class ImportError_(Exception):
    """Raised to abort the import and roll back the transaction."""


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="import_categories",
        description="Import a product taxonomy CSV into the categories table.",
    )
    parser.add_argument("--file", type=Path, required=True,
                        help="Path to the taxonomy CSV.")
    parser.add_argument("--taxonomy-version", type=int, required=True,
                        help="Version number written to every imported row.")
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL"),
                        help="Postgres connection string. Defaults to $DATABASE_URL.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and validate only; don't connect to the database.")
    parser.add_argument("--allow-mass-deactivate", action="store_true",
                        help=f"Allow deactivating more than {MAX_DEACTIVATE_RATIO:.0%} "
                             "of active categories.")

    args = parser.parse_args(argv)

    if not args.file.is_file():
        parser.error(f"--file not found: {args.file}")
    if args.taxonomy_version < 1:
        parser.error("--taxonomy-version must be 1 or higher")
    if not args.dry_run and not args.dsn:
        parser.error("--dsn or $DATABASE_URL is required unless --dry-run is set")
    return args


# ---------------------------------------------------------------------------
# Read and row-level validation
# ---------------------------------------------------------------------------

def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_name(
        rec: dict[str, str | None],
        segments: tuple[str, ...],
        level_columns: list[str],
        line: int,
) -> tuple[str, list[str]]:
    """Work out the category name and cross-check every source that states it.

    The name is the leaf segment. Depending on the file it may come from a
    'name' column, from level_<level> (level_1 for a root, level_5 for a
    depth-5 node), or only from full_path. Every source present must agree.
    """
    errors: list[str] = []
    depth = len(segments)
    leaf = segments[-1]

    # level column, if present, must match the number of path segments
    level_raw = rec.get("level")
    if level_raw is not None:
        if not level_raw.isdigit() or int(level_raw) != depth:
            errors.append(f"line {line}: level {level_raw!r} doesn't match "
                          f"{depth} segments in full_path")

    # level_1..level_N must spell out full_path, and deeper columns must be empty
    for col in level_columns:
        i = int(LEVEL_COLUMN.match(col).group(1))
        value = rec.get(col)
        expected = segments[i - 1] if i <= depth else None
        if value != expected:
            errors.append(f"line {line}: {col} is {value!r}, expected {expected!r} "
                          f"from full_path")

    explicit = rec.get("name")
    if explicit is not None and explicit != leaf:
        errors.append(f"line {line}: name {explicit!r} doesn't match "
                      f"last path segment {leaf!r}")

    return leaf, errors


def read_rows(path: Path) -> tuple[list[CategoryRow], list[str]]:
    rows: list[CategoryRow] = []
    errors: list[str] = []
    seen_ids: dict[str, int] = {}
    seen_paths: dict[tuple[str, ...], int] = {}

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        headers = {h.strip() for h in (reader.fieldnames or [])}
        missing = REQUIRED_COLUMNS - headers
        if missing:
            return [], [f"missing columns {sorted(missing)}; file has {sorted(headers)}"]

        # level_1 .. level_N columns, in numeric order (level_10 after level_9)
        level_columns = sorted(
            (h for h in headers if LEVEL_COLUMN.match(h)),
            key=lambda h: int(LEVEL_COLUMN.match(h).group(1)),
        )
        if "name" not in headers and not level_columns:
            print("note: no 'name' or level_N columns; using last full_path segment as name",
                  file=sys.stderr)

        for raw in reader:
            line = reader.line_num
            # k is None when a row has more fields than the header
            if None in raw:
                errors.append(f"line {line}: more fields than headers")
                continue
            rec = {k.strip(): _clean(v) for k, v in raw.items()}

            empty = [c for c in sorted(REQUIRED_COLUMNS) if not rec.get(c)]
            if empty:
                errors.append(f"line {line}: empty {', '.join(empty)}")
                continue

            cat_id, full_path = rec["id"], rec["full_path"]
            row_errors: list[str] = []

            if not LTREE_LABEL.match(cat_id):
                row_errors.append(f"line {line}: id {cat_id!r} is not a valid ltree label")

            segments = tuple(s.strip() for s in full_path.split(PATH_SEPARATOR))
            if any(not s for s in segments):
                row_errors.append(f"line {line}: empty segment in full_path {full_path!r}")
                errors.extend(row_errors)
                continue

            name, name_errors = _resolve_name(rec, segments, level_columns, line)
            row_errors.extend(name_errors)

            if cat_id in seen_ids:
                row_errors.append(f"line {line}: duplicate id {cat_id!r} "
                                  f"(first on line {seen_ids[cat_id]})")
            if segments in seen_paths:
                row_errors.append(f"line {line}: duplicate full_path {full_path!r} "
                                  f"(first on line {seen_paths[segments]})")

            if row_errors:
                errors.extend(row_errors)
                continue

            seen_ids[cat_id] = line
            seen_paths[segments] = line
            rows.append(CategoryRow(
                line_no=line,
                id=cat_id,
                name=name,
                full_path=full_path,
                segments=segments,
                parent_id=rec.get("parent_id"),
                description=rec.get("description"),
            ))

    return rows, errors


# ---------------------------------------------------------------------------
# Tree building and tree-level validation
# ---------------------------------------------------------------------------

def build_tree(rows: list[CategoryRow]) -> tuple[dict[str, Node], list[str]]:
    """Derive parent, ltree path, depth and leaf flag from full_path.

    Returns the nodes keyed by id and a list of orphan errors (a node whose
    ancestor path isn't in the file).
    """
    by_segments = {r.segments: r for r in rows}
    tree: dict[str, Node] = {}
    errors: list[str] = []

    for r in rows:
        ancestor_ids: list[str] = []
        missing_at: tuple[str, ...] | None = None
        for i in range(1, len(r.segments)):
            ancestor = by_segments.get(r.segments[:i])
            if ancestor is None:
                missing_at = r.segments[:i]
                break
            ancestor_ids.append(ancestor.id)

        if missing_at is not None:
            errors.append(f"line {r.line_no}: orphan {r.id!r}; ancestor "
                          f"{PATH_SEPARATOR.join(missing_at)!r} is not in the file")
            continue

        tree[r.id] = Node(
            id=r.id,
            name=r.name,
            description=r.description,
            parent_id=ancestor_ids[-1] if ancestor_ids else None,
            path=".".join([*ancestor_ids, r.id]),
            depth=len(r.segments),
            is_leaf=True,              # corrected in the second pass
            line_no=r.line_no,
            csv_parent_id=r.parent_id,
        )

    # Second pass: anything that is someone's parent is not a leaf.
    for node in tree.values():
        if node.parent_id in tree:
            tree[node.parent_id].is_leaf = False

    return tree, errors


def validate(tree: dict[str, Node]) -> list[str]:
    errors: list[str] = []

    if not tree:
        errors.append("no categories to import")
        return errors

    if not any(n.parent_id is None for n in tree.values()):
        errors.append("no root categories found")

    for n in tree.values():
        if n.csv_parent_id is not None and n.csv_parent_id != n.parent_id:
            errors.append(f"line {n.line_no}: CSV parent_id {n.csv_parent_id!r} doesn't "
                          f"match parent derived from full_path {n.parent_id!r}")
        if n.path.count(".") + 1 != n.depth:
            errors.append(f"line {n.line_no}: path {n.path!r} doesn't match depth {n.depth}")

    return errors


def summarise(tree: dict[str, Node]) -> str:
    by_depth = Counter(n.depth for n in tree.values())
    leaves = sum(n.is_leaf for n in tree.values())
    roots = sorted(n.name for n in tree.values() if n.parent_id is None)
    lines = [
        f"{len(tree)} categories, {leaves} leaves, {len(roots)} roots",
        "by depth: " + ", ".join(f"{d}: {c}" for d, c in sorted(by_depth.items())),
        "roots: " + ", ".join(roots[:10]) + (" ..." if len(roots) > 10 else ""),
        "samples:",
        ]
    deepest = sorted(tree.values(), key=lambda n: -n.depth)[:3]
    for n in deepest:
        lines.append(f"  {n.id}  parent={n.parent_id}  path={n.path}  "
                     f"depth={n.depth}  leaf={n.is_leaf}  name={n.name!r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

UPSERT_SQL = """
             INSERT INTO categories
                 (id, parent_id, path, name, description, depth, is_leaf, is_active, taxonomy_version)
             VALUES
                 (%(id)s, %(parent_id)s, %(path)s::ltree, %(name)s, %(description)s,
                  %(depth)s, %(is_leaf)s, true, %(version)s)
                 ON CONFLICT (id) DO UPDATE SET
                 parent_id        = EXCLUDED.parent_id,
                                         path             = EXCLUDED.path,
                                         name             = EXCLUDED.name,
                                         description      = COALESCE(EXCLUDED.description, categories.description),
                                         depth            = EXCLUDED.depth,
                                         is_leaf          = EXCLUDED.is_leaf,
                                         is_active        = true,
                                         taxonomy_version = EXCLUDED.taxonomy_version
                                     WHERE (categories.parent_id, categories.path, categories.name,
                                         categories.description, categories.depth,
                                         categories.is_leaf, categories.is_active, categories.taxonomy_version)
                                         IS DISTINCT FROM
                                         (EXCLUDED.parent_id, EXCLUDED.path, EXCLUDED.name,
                                         COALESCE(EXCLUDED.description, categories.description), EXCLUDED.depth,
                                         EXCLUDED.is_leaf, true, EXCLUDED.taxonomy_version)
                                         RETURNING (xmax = 0) AS inserted \
             """


def upsert(conn, tree: dict[str, Node], version: int) -> tuple[int, int, int]:
    """Insert or update every node, parents first. Returns (inserted, updated, unchanged)."""
    params = [
        {
            "id": n.id,
            "parent_id": n.parent_id,
            "path": n.path,
            "name": n.name,
            "description": n.description,
            "depth": n.depth,
            "is_leaf": n.is_leaf,
            "version": version,
        }
        for n in sorted(tree.values(), key=lambda n: n.depth)  # FK on parent_id
    ]

    inserted = updated = 0
    with conn.cursor() as cur:
        cur.executemany(UPSERT_SQL, params, returning=True)
        while True:
            for (was_insert,) in cur.fetchall():
                if was_insert:
                    inserted += 1
                else:
                    updated += 1
            if not cur.nextset():
                break

    unchanged = len(params) - inserted - updated
    return inserted, updated, unchanged


def deactivate_missing(conn, ids: set[str], allow_mass: bool) -> int:
    """Mark active categories that aren't in this import as inactive. Never deletes."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM categories WHERE is_active")
        (active,) = cur.fetchone()

        cur.execute(
            "SELECT count(*) FROM categories WHERE is_active AND NOT (id = ANY(%s))",
            (list(ids),),
        )
        (to_deactivate,) = cur.fetchone()

        if active and to_deactivate / active > MAX_DEACTIVATE_RATIO and not allow_mass:
            raise ImportError_(
                f"would deactivate {to_deactivate} of {active} active categories "
                f"({to_deactivate / active:.0%}); check the file or pass "
                f"--allow-mass-deactivate"
            )

        # taxonomy_version is left alone: it records the last version that
        # contained the category.
        cur.execute(
            "UPDATE categories SET is_active = false "
            "WHERE is_active AND NOT (id = ANY(%s))",
            (list(ids),),
        )
        return cur.rowcount


POST_CHECKS = {
    "depth doesn't match path":
        """SELECT id FROM categories
           WHERE depth <> nlevel(path)""",
    "root has a parent, or non-root has none":
        """SELECT id FROM categories
           WHERE (nlevel(path) = 1) <> (parent_id IS NULL)""",
    "path doesn't extend parent's path":
        """SELECT c.id FROM categories c
                                JOIN categories p ON p.id = c.parent_id
           WHERE subpath(c.path, 0, nlevel(c.path) - 1) <> p.path""",
    "path doesn't end with own id":
        """SELECT id FROM categories
           WHERE subpath(path, nlevel(path) - 1)::text <> id""",
    "active category under an inactive parent":
        """SELECT c.id FROM categories c
                                JOIN categories p ON p.id = c.parent_id
           WHERE c.is_active AND NOT p.is_active""",
    "is_leaf disagrees with active children":
        """SELECT c.id FROM categories c
           WHERE c.is_active
             AND c.is_leaf = EXISTS (
               SELECT 1 FROM categories k
               WHERE k.parent_id = c.id AND k.is_active)""",
}


def post_checks(conn, expected_active: int) -> list[str]:
    errors: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM categories WHERE is_active")
        (active,) = cur.fetchone()
        if active != expected_active:
            errors.append(f"expected {expected_active} active categories, found {active}")

        for label, sql in POST_CHECKS.items():
            cur.execute(sql + " LIMIT 5")
            bad = [r[0] for r in cur.fetchall()]
            if bad:
                errors.append(f"{label}: {', '.join(bad)}")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_errors(title: str, errors: list[str], limit: int = 50) -> None:
    print(f"\n{title} ({len(errors)}):", file=sys.stderr)
    for e in errors[:limit]:
        print(f"  {e}", file=sys.stderr)
    if len(errors) > limit:
        print(f"  ... {len(errors) - limit} more", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    rows, row_errors = read_rows(args.file)
    tree, tree_errors = build_tree(rows)
    errors = row_errors + tree_errors + validate(tree)

    print(summarise(tree) if tree else "no categories parsed")

    if errors:
        _print_errors("Validation failed", errors)
        return 1

    if args.dry_run:
        print("\nDry run: validation passed, nothing written.")
        return 0

    import psycopg  # imported here so --dry-run works without psycopg installed

    try:
        with psycopg.connect(args.dsn) as conn:
            with conn.transaction():
                inserted, updated, unchanged = upsert(conn, tree, args.taxonomy_version)
                deactivated = deactivate_missing(conn, set(tree), args.allow_mass_deactivate)
                check_errors = post_checks(conn, expected_active=len(tree))
                if check_errors:
                    _print_errors("Post-checks failed, rolling back", check_errors)
                    raise ImportError_("post-checks failed")
    except ImportError_ as e:
        print(f"\nImport aborted: {e}", file=sys.stderr)
        return 1

    print(f"\nImported taxonomy version {args.taxonomy_version}: "
          f"{inserted} inserted, {updated} updated, {unchanged} unchanged, "
          f"{deactivated} deactivated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
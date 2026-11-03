#!/usr/bin/env bash
# ------------------------------------------------------------------
# apply_migrations.sh
#
# Apply every SQL file in ../migrations/ (relative to this script)
# exactly once, in lexical order. Tracks state in `schema_migrations`.
#
# Each file runs in ONE transaction together with its tracking row:
# if any statement fails, nothing from that file is kept and it is
# not recorded, so fixing the file and rerunning is always safe.
# Don't put BEGIN/COMMIT or CREATE INDEX CONCURRENTLY in a migration
# file; both conflict with the wrapping transaction.
#
# Name files with zero-padded numbers (001_, 002_ ... 010_) so that
# lexical order is the order you mean.
#
# Why this exists: docker-compose mounts the migrations folder to
# /docker-entrypoint-initdb.d/, which only runs on FIRST database
# initialisation. Once the pgdata volume has data, new SQL files
# there are never applied automatically, so we apply them here.
#
# Usage:
#   ./apply_migrations.sh                             # auto-detect container
#   DB_CONTAINER=pgvector-db ./apply_migrations.sh
#   PG_URL=postgresql://postgres:postgres@localhost:5533/pgvector ./apply_migrations.sh
#   PG_URL="$DATABASE_URL" ./apply_migrations.sh      # +asyncpg URLs are accepted
#   ./apply_migrations.sh --dsn "$DATABASE_URL"       # same as PG_URL
#
# URL mode needs psql installed locally. Container mode uses the psql
# inside the container and connects as $PGUSER to $PGDATABASE, which
# default to the container's own POSTGRES_USER and POSTGRES_DB.
#
# --mark-baseline           Record every pending migration as applied
#   without running it. Use once if the DB was created via
#   docker-entrypoint-initdb.d and all current files are in place.
#
# --mark-up-to <filename>   Record migrations up to and including
#   <filename> as applied without running them, then apply the rest.
#   Example:
#     ./apply_migrations.sh --mark-up-to 001_initial_schema.sql
#
# --dry-run                 Show what would be applied; change nothing.
# ------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MIG_DIR="$SCRIPT_DIR/../migrations"
export LC_ALL=C   # byte-order sorting for the glob and the --mark-up-to comparison

MARK_BASELINE=0
MARK_UP_TO=""
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mark-baseline) MARK_BASELINE=1 ;;
    --mark-up-to)
      [ $# -ge 2 ] || { echo "❌ --mark-up-to needs a filename" >&2; exit 2; }
      MARK_UP_TO="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    --dsn)
      [ $# -ge 2 ] || { echo "❌ --dsn needs a connection URL" >&2; exit 2; }
      PG_URL="$2"; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "❌ Unknown option: $1 (see --help)" >&2
      exit 2 ;;
  esac
  shift
done

if [ ! -d "$MIG_DIR" ]; then
  echo "❌ Migrations directory not found: $MIG_DIR" >&2
  exit 1
fi
if [ -n "$MARK_UP_TO" ] && [ ! -f "$MIG_DIR/$MARK_UP_TO" ]; then
  echo "❌ --mark-up-to file not found in $MIG_DIR: $MARK_UP_TO" >&2
  exit 1
fi

# Choose how we talk to Postgres.
APPLIER=""
if [ -n "${PG_URL:-}" ]; then
  command -v psql >/dev/null 2>&1 || {
    echo "❌ A connection URL was given, but psql isn't installed on this machine." >&2
    echo "   Either install the client (macOS: brew install libpq && brew link --force libpq)" >&2
    echo "   or use the psql inside the container: DB_CONTAINER=<name> $0" >&2
    exit 1
  }
  # psql doesn't understand SQLAlchemy driver suffixes like +asyncpg
  PG_URL="$(printf '%s' "$PG_URL" | sed -E 's#^postgres(ql)?\+[a-z0-9_]+://#postgresql://#')"
  APPLIER="psql"
elif [ -n "${DB_CONTAINER:-}" ]; then
  APPLIER="docker"
else
  for name in agent-db pgvector-db pgvector-db_1 pgvector; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$name"; then
      DB_CONTAINER="$name"
      APPLIER="docker"
      break
    fi
  done
fi

# In a container, default to the user and database the container was
# created with (POSTGRES_USER / POSTGRES_DB), so nothing is guessed.
if [ "$APPLIER" = "docker" ]; then
  container_env() { docker exec "$DB_CONTAINER" printenv "$1" 2>/dev/null || true; }
  PGUSER="${PGUSER:-$(container_env POSTGRES_USER)}"
  PGUSER="${PGUSER:-postgres}"
  PGDATABASE="${PGDATABASE:-$(container_env POSTGRES_DB)}"
  PGDATABASE="${PGDATABASE:-$PGUSER}"
fi

if [ -z "$APPLIER" ]; then
  echo "❌ Could not find a way to reach Postgres." >&2
  echo "   Either:" >&2
  echo "     - export DB_CONTAINER=<docker container name>" >&2
  echo "     - export PG_URL=postgresql://user:pw@host:port/db" >&2
  exit 1
fi

# psql reading SQL from stdin, stopping on the first error.
# Extra arguments (e.g. -1 for a single transaction) are passed through.
psql_stdin() {
  if [ "$APPLIER" = "docker" ]; then
    docker exec -i "$DB_CONTAINER" psql -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1 -q "$@"
  else
    psql "$PG_URL" -v ON_ERROR_STOP=1 -q "$@"
  fi
}

query() {
  printf '%s\n' "$1" | psql_stdin -tA
}

sql_quote() {
  printf "'%s'" "${1//\'/\'\'}"
}

checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# Ensure tracking table exists (checksum added for older tables).
# A dry run creates nothing; with no table yet, every file is pending.
HAS_TABLE="$(query "SELECT to_regclass('schema_migrations') IS NOT NULL;")"
[ "$DRY_RUN" = "1" ] || printf '%s\n' "
SET client_min_messages = warning;
CREATE TABLE IF NOT EXISTS schema_migrations (
  filename   TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS checksum TEXT;
ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS baseline BOOLEAN NOT NULL DEFAULT false;
" | psql_stdin

echo "Applying migrations from $MIG_DIR"
[ "$APPLIER" = "docker" ] && echo "  via container: $DB_CONTAINER ($PGUSER@$PGDATABASE)" || echo "  via PG_URL"
[ "$DRY_RUN" = "1" ] && echo "  dry run: nothing will be changed"

applied_count=0
skipped_count=0
baseline_count=0
shopt -s nullglob
files=("$MIG_DIR"/*.sql)
if [ ${#files[@]} -eq 0 ]; then
  echo "No .sql files in $MIG_DIR"
  exit 0
fi

for file in "${files[@]}"; do
  name="$(basename "$file")"
  sum="$(checksum "$file")"
  qname="$(sql_quote "$name")"
  recorded=""
  if [ "$DRY_RUN" = "0" ] || [ "$HAS_TABLE" = "t" ]; then
    recorded="$(query "SELECT coalesce(to_jsonb(s)->>'checksum', '-') FROM schema_migrations s WHERE filename = $qname;")"
  fi

  if [ -n "$recorded" ]; then
    if [ "$recorded" != "-" ] && [ "$recorded" != "$sum" ]; then
      echo "  ⚠ $name (already applied, but the file has changed since; add a new migration instead)"
    else
      echo "  ✓ $name (already applied)"
    fi
    skipped_count=$((skipped_count + 1))
    continue
  fi

  if [ "$MARK_BASELINE" = "1" ] || { [ -n "$MARK_UP_TO" ] && [[ ! "$name" > "$MARK_UP_TO" ]]; }; then
    if [ "$DRY_RUN" = "0" ]; then
      query "INSERT INTO schema_migrations (filename, checksum, baseline)
             VALUES ($qname, $(sql_quote "$sum"), true) ON CONFLICT DO NOTHING;" >/dev/null
    fi
    echo "  • $name (baseline, recorded without running)"
    baseline_count=$((baseline_count + 1))
    continue
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "  → $name (would apply)"
    applied_count=$((applied_count + 1))
    continue
  fi

  echo "  → $name"
  # File and tracking row in one transaction: all of it or none of it.
  if ! {
        cat "$file"
        printf '\n;\nINSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s);\n' \
          "$qname" "$(sql_quote "$sum")"
      } | psql_stdin -1; then
    echo "❌ $name failed and was rolled back; nothing from it was kept." >&2
    echo "   Fix the file and rerun. Applied before the failure: $applied_count." >&2
    exit 1
  fi
  applied_count=$((applied_count + 1))
done

if [ "$DRY_RUN" = "1" ]; then
  echo "✅ Dry run. Would apply $applied_count, record $baseline_count as baseline, skip $skipped_count."
else
  echo "✅ Done. Applied $applied_count, baseline $baseline_count, skipped $skipped_count."
fi
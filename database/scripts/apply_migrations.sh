#!/usr/bin/env bash
# ------------------------------------------------------------------
# apply_migrations.sh
#
# Apply every SQL file in infra/supabase/migrations/ exactly once,
# in lexical order. Tracks state in `schema_migrations`.
#
# Why this exists: docker-compose mounts the migrations folder to
# /docker-entrypoint-initdb.d/, which only runs on FIRST database
# initialization. Once the pgdata volume has data, new SQL files
# there are never applied automatically — so we apply them here.
#
# Usage:
#   nexus/scripts/apply_migrations.sh                 # auto-detect
#   DB_CONTAINER=luminary-test-pg ./apply_migrations.sh
#   PG_URL=postgres://user:pw@host:port/db ./apply_migrations.sh
#
# First run on an existing DB: pass --mark-baseline to record every
# migration as already applied without running it. Use this once if
# your DB was created via docker-entrypoint-initdb.d (i.e. the early
# migrations are already in place).
#
# --mark-up-to <filename>   Record migrations up to and including
#   <filename> as baseline (without running them), then apply the
#   rest normally. Useful when the DB was initialised via
#   docker-entrypoint-initdb.d and only newer files need applying.
#   Example:
#     ./apply_migrations.sh --mark-up-to 005_calendar_meeting_prep.sql
# ------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MIG_DIR="$SCRIPT_DIR/../migrations"

MARK_BASELINE=0
MARK_UP_TO=""
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
  arg="${args[$i]}"
  case "$arg" in
    --mark-baseline) MARK_BASELINE=1 ;;
    --mark-up-to)
      i=$((i + 1))
      MARK_UP_TO="${args[$i]}"
      ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
  esac
  i=$((i + 1))
done

if [ ! -d "$MIG_DIR" ]; then
  echo "❌ Migrations directory not found: $MIG_DIR" >&2
  exit 1
fi

# Choose how we talk to Postgres.
APPLIER=""
if [ -n "${PG_URL:-}" ]; then
  command -v psql >/dev/null 2>&1 || { echo "❌ PG_URL set but psql not on PATH." >&2; exit 1; }
  APPLIER="psql"
elif [ -n "${DB_CONTAINER:-}" ]; then
  APPLIER="docker"
else
  for name in agent-db pgvector-db pgvector-db_1; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$name"; then
      DB_CONTAINER="$name"
      APPLIER="docker"
      break
    fi
  done
fi

if [ -z "$APPLIER" ]; then
  echo "❌ Could not find a way to reach Postgres." >&2
  echo "   Either:" >&2
  echo "     - export DB_CONTAINER=<docker container name>" >&2
  echo "     - export PG_URL=postgres://user:pw@host:port/db" >&2
  exit 1
fi

run_sql() {
  local sql="$1"
  if [ "$APPLIER" = "docker" ]; then
    echo "$sql" | docker exec -i "$DB_CONTAINER" psql -U user -d pgvector -v ON_ERROR_STOP=1 -q
  else
    psql "$PG_URL" -v ON_ERROR_STOP=1 -q -c "$sql"
  fi
}

apply_file() {
  local file="$1"
  if [ "$APPLIER" = "docker" ]; then
    docker exec -i "$DB_CONTAINER" psql -U user -d pgvector -v ON_ERROR_STOP=1 -q < "$file"
  else
    psql "$PG_URL" -v ON_ERROR_STOP=1 -q -f "$file"
  fi
}

query() {
  local sql="$1"
  if [ "$APPLIER" = "docker" ]; then
    docker exec -i "$DB_CONTAINER" psql -U user -d pgvector -tAq -c "$sql"
  else
    psql "$PG_URL" -tAq -c "$sql"
  fi
}

# Ensure tracking table exists.
run_sql "CREATE TABLE IF NOT EXISTS schema_migrations (
  filename TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);"

echo "Applying migrations from $MIG_DIR"
[ -n "${DB_CONTAINER:-}" ] && echo "  via container: $DB_CONTAINER" || echo "  via PG_URL"

applied_count=0
skipped_count=0
shopt -s nullglob
for file in "$MIG_DIR"/*.sql; do
  name="$(basename "$file")"
  exists=$(query "SELECT 1 FROM schema_migrations WHERE filename = '${name//\'/\'\'}' LIMIT 1;")
  if [ "$exists" = "1" ]; then
    echo "  ✓ $name (already applied)"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  if [ "$MARK_BASELINE" = "1" ]; then
    run_sql "INSERT INTO schema_migrations (filename) VALUES ('${name//\'/\'\'}') ON CONFLICT DO NOTHING;"
    echo "  • $name (baseline — recorded without running)"
    continue
  fi

  if [ -n "$MARK_UP_TO" ] && [[ ! "$name" > "$MARK_UP_TO" ]]; then
    run_sql "INSERT INTO schema_migrations (filename) VALUES ('${name//\'/\'\'}') ON CONFLICT DO NOTHING;"
    echo "  • $name (baseline — recorded without running)"
    continue
  fi

  echo "→ $name"
  apply_file "$file"
  run_sql "INSERT INTO schema_migrations (filename) VALUES ('${name//\'/\'\'}');"
  applied_count=$((applied_count + 1))
done

if [ "$MARK_BASELINE" = "1" ]; then
  echo "✅ Baseline recorded. Future runs will apply only new files."
else
  echo "✅ Done. Applied $applied_count, skipped $skipped_count."
fi

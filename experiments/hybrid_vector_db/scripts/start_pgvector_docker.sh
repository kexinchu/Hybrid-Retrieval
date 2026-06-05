#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${PGVECTOR_CONTAINER:-hybrid-pgvector}"
PGPORT="${PGPORT:-55432}"
PGPASSWORD="${PGPASSWORD:-postgres}"
PGDATABASE="${PGDATABASE:-hybrid_vector}"
IMAGE="${PGVECTOR_IMAGE:-pgvector/pgvector:pg16}"
DATA_DIR="${PGVECTOR_DATA_DIR:-$PWD/.pgvector-data}"

mkdir -p "$DATA_DIR"

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "container already running: $CONTAINER_NAME"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker start "$CONTAINER_NAME"
else
  docker run -d \
    --name "$CONTAINER_NAME" \
    -e POSTGRES_PASSWORD="$PGPASSWORD" \
    -e POSTGRES_DB="$PGDATABASE" \
    -p "$PGPORT:5432" \
    -v "$DATA_DIR:/var/lib/postgresql/data" \
    "$IMAGE"
fi

echo "waiting for PostgreSQL on port $PGPORT"
for _ in $(seq 1 60); do
  if docker exec "$CONTAINER_NAME" pg_isready -U postgres >/dev/null 2>&1; then
    echo "PostgreSQL is ready"
    exit 0
  fi
  sleep 1
done

echo "PostgreSQL did not become ready in time" >&2
exit 1

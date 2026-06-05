from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PgConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str

    @property
    def conninfo(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )


def pg_config_from_env() -> PgConfig:
    return PgConfig(
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=int(os.environ.get("PGPORT", "55432")),
        dbname=os.environ.get("PGDATABASE", "hybrid_vector"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )


def require_psycopg():
    try:
        import psycopg  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing psycopg. Install with: .venv/bin/python -m pip install 'psycopg[binary]'"
        ) from exc

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VECTOR_SQL = ROOT / "third_party/pgvector-sqlens/sql/vector.sql"
VERSIONED_VECTOR_SQL = ROOT / "third_party/pgvector-sqlens/sql/vector--0.8.2.sql"
UPGRADE_VECTOR_SQL = (
    ROOT / "third_party/pgvector-sqlens/sql/vector--0.8.1--0.8.2.sql"
)
VECTOR_C = ROOT / "third_party/pgvector-sqlens/src/vector.c"
SMOKE_SQL = (
    ROOT
    / "experiments/hybrid_vector_db/sql/pgvector_epoch_writer_permissions_smoke.sql"
)
MAKEFILE = ROOT / "third_party/pgvector-sqlens/Makefile"


def test_epoch_trigger_uses_narrow_security_definer_authority() -> None:
    vector_sql = VECTOR_SQL.read_text()
    upgrade_sql = UPGRADE_VECTOR_SQL.read_text()
    vector_c = VECTOR_C.read_text()

    assert VECTOR_SQL.read_text() == VERSIONED_VECTOR_SQL.read_text()
    assert (
        "CREATE FUNCTION vector_hnsw_fragment_epoch_bump_trigger() RETURNS trigger\n"
        "\tAS 'MODULE_PATHNAME' LANGUAGE C SECURITY DEFINER\n"
        "\tSET search_path = pg_catalog, pg_temp;"
    ) in vector_sql
    assert (
        "ALTER FUNCTION vector_hnsw_fragment_epoch_bump_trigger() SECURITY DEFINER;"
    ) in upgrade_sql
    assert "SET search_path = pg_catalog, pg_temp;" in upgrade_sql
    assert (
        "UPDATE public.pgvector_hnsw_fragment_epoch "
        '"\n\t\t"SET epoch = epoch + 1, updated_at = pg_catalog.now() "'
    ) in vector_c
    assert (
        "INSERT INTO public.pgvector_hnsw_fragment_epoch "
        "(heap_oid, epoch) VALUES ($1, 1)"
    ) not in vector_c
    assert (
        "EXECUTE FUNCTION public.vector_hnsw_fragment_epoch_bump_trigger()"
    ) not in vector_c
    assert "p.pronamespace = x.extnamespace" in vector_c


def test_writer_permission_smoke_covers_dml_hot_and_metadata_denial() -> None:
    smoke_sql = SMOKE_SQL.read_text()
    makefile = MAKEFILE.read_text()

    for statement in ("INSERT INTO", "UPDATE", "DELETE FROM"):
        assert statement in smoke_sql
    assert "pg_stat_get_tuples_hot_updated" in smoke_sql
    assert "before_epoch + 4" in smoke_sql
    assert "EXCEPTION WHEN insufficient_privilege" in smoke_sql
    assert "pgvector_epoch_writer_permissions_smoke.sql" in makefile

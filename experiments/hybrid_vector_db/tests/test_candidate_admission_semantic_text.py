from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HNSW_SOURCE = ROOT / "third_party/pgvector-sqlens/src/hnsw.c"
SMOKE_SQL = ROOT / "experiments/hybrid_vector_db/sql/pgvector_traversal_guidance_binding_smoke.sql"


def test_traversal_guided_guc_help_describes_candidate_admission() -> None:
    source = HNSW_SOURCE.read_text(encoding="utf-8")

    assert "native graph expansion and vector distance computation followed by" in source
    assert "predicate-aware result-heap admission and pre-heap TID suppression" in source
    assert "pre-distance filtering" not in source
    assert "bounded bridge expansion" not in source


def test_legacy_traversal_gucs_are_explicitly_deprecated() -> None:
    source = HNSW_SOURCE.read_text(encoding="utf-8")

    assert "Deprecated compatibility target for legacy traversal_guided search" in source
    assert "native candidate admission ignores this value and collects the hnsw.ef_search result batch" in source
    assert "Deprecated compatibility limit for legacy traversal_guided bridge hops" in source
    assert "native candidate admission ignores this value and does not bound predicate-miss graph expansion" in source
    assert "Deprecated compatibility limit for legacy traversal_guided bridge work" in source
    assert "native candidate admission ignores this value and does not use bridge-work fallback" in source


def test_binding_smoke_asserts_r6_semantics_without_legacy_bridge_claims() -> None:
    sql = SMOKE_SQL.read_text(encoding="utf-8")
    normalized_sql = " ".join(sql.replace("--", "").split())

    assert "native graph expansion and vector distance" in normalized_sql
    assert "predicate-aware result-heap admission" in normalized_sql
    assert "heap TIDs suppressed before the AM returns them" in normalized_sql
    assert "candidate admission never prunes or bounds distance-ordered graph expansion" in normalized_sql
    assert "membership is checked before vector distance" not in sql
    assert "(guided_profile->>'graph_expansion_pruned')::boolean" in sql
    assert "(guided_profile->>'distance_computations_pruned')::boolean" in sql
    assert "(guided_profile->>'pre_distance_membership_checks')::bigint <> 0" in sql
    assert "(guided_profile->>'distance_computations_avoided')::bigint <> 0" in sql
    assert "(guided_profile->>'miss_bridge_nodes')::bigint <= 0" not in sql
    assert "(guided_profile->>'miss_bridge_edges')::bigint <= 0" not in sql
    assert "(guided_profile->>'traversal_bridge_expanded')::bigint" not in sql

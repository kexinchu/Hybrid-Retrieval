-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.8.2'" to load this file. \quit

ALTER FUNCTION vector_hnsw_fragment_epoch_bump_trigger() SECURITY DEFINER;
ALTER FUNCTION vector_hnsw_fragment_epoch_bump_trigger()
	SET search_path = pg_catalog, pg_temp;

CREATE FUNCTION vector_hnsw_guidance_bind(regclass, text[], text) RETURNS boolean
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION vector_sqlens_build_id() RETURNS text
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION vector_hnsw_graph_fingerprint(regclass) RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;

COMMENT ON FUNCTION vector_hnsw_graph_fingerprint(regclass) IS
	'Canonical HNSW logical SHA-256 and physical-layout SHA-256. The logical digest covers format/options, canonical entrypoint identity, and every node ordered by its same-heap ordered heap-TID bundle, including level, version, exact vector bytes, and ordered per-level neighbor identities. The result also contains full sqlens-hnsw-bfs-locality-v1 statistics: zero-based deterministic BFS rank to physical index block/offset adjacency; same means delta 0, next means forward delta +1, nondecreasing counts block transitions with current >= previous, and page_runs counts maximal equal-block runs. rank_samples is capped at 256 with evenly spaced inclusive endpoints while full_statistics remains true.';

CREATE FUNCTION vector_hnsw_graph_compare(regclass, regclass) RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;

COMMENT ON FUNCTION vector_hnsw_graph_compare(regclass, regclass) IS
	'Compares canonical logical graph digests, physical-layout digests, entrypoint identity, exact node/heap-TID coverage, and heap identity for two HNSW indexes. It also returns source/clone-symmetric left_bfs_locality and right_bfs_locality objects under the sqlens-hnsw-bfs-locality-v1 metric contract.';

REVOKE ALL ON FUNCTION vector_hnsw_graph_fingerprint(regclass) FROM PUBLIC;
REVOKE ALL ON FUNCTION vector_hnsw_graph_compare(regclass, regclass) FROM PUBLIC;

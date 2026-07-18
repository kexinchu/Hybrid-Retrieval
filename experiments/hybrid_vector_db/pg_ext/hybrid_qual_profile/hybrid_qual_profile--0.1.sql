CREATE FUNCTION hybrid_qual_profile_reset() RETURNS void
AS 'MODULE_PATHNAME', 'hybrid_qual_profile_reset'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION hybrid_qual_profile_last() RETURNS text
AS 'MODULE_PATHNAME', 'hybrid_qual_profile_last'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

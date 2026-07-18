#include "postgres.h"

#include <math.h>

#include "access/genam.h"
#include "access/skey.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "bitutils.h"
#include "bitvec.h"
#include "catalog/index.h"
#include "catalog/namespace.h"
#include "catalog/pg_type.h"
#include "commands/trigger.h"
#include "common/shortest_dec.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "funcapi.h"
#include "halfutils.h"
#include "halfvec.h"
#include "hnsw.h"
#include "ivfflat.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "port.h"				/* for strtof() */
#include "executor/spi.h"
#include "sparsevec.h"
#include "utils/inval.h"
#include "utils/array.h"
#include "utils/float.h"
#include "utils/fmgrprotos.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/hsearch.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/varbit.h"
#include "vector.h"

extern text *cstring_to_text(const char *s);

typedef struct HnswMaterializeCandidate
{
	ItemPointerData tid;
	int			rank;
	int64		id;
	bool		visible;
} HnswMaterializeCandidate;

typedef struct HnswMaterializeProfile
{
	bool		valid;
	int64		candidates;
	int64		visible;
	int64		returned;
	int64		distanceRuns;
	int64		distinctPages;
	double		indexMs;
	double		fetchMs;
} HnswMaterializeProfile;

typedef struct HnswMetadataCacheKey
{
	Oid			heapOid;
	char		filterName[256];
} HnswMetadataCacheKey;

typedef struct HnswMetadataTidKey
{
	ItemPointerData tid;
} HnswMetadataTidKey;

typedef struct HnswMetadataTidEntry
{
	HnswMetadataTidKey key;
	int64		id;
} HnswMetadataTidEntry;

typedef struct HnswMetadataCacheEntry
{
	HnswMetadataCacheKey key;
	HTAB	   *tidHash;
	uint8	   *pageBits;
	uint8	   *bloomBits;
	int64		rows;
	int64		pageRows;
	int64		pages;
	int64		bloomRows;
	Size		pageBitBytes;
	Size		bloomBytes;
	uint64		bloomBitCount;
	double		buildMs;
	double		pageBuildMs;
	double		bloomBuildMs;
	int64		memoryBytes;
	uint64		lastUsed;
	bool		epochTracked;
	int64		buildEpoch;
	Oid			buildRelFileNode;
} HnswMetadataCacheEntry;

typedef struct HnswGuidanceDescriptorKey
{
	Oid			heapOid;
	char		signature[1024];
} HnswGuidanceDescriptorKey;

typedef struct HnswGuidanceDescriptorEntry
{
	HnswGuidanceDescriptorKey key;
	uint64		hits;
	uint64		lastUsed;
	HTAB	   *exactTidHash;
	int64		exactRows;
	int64		exactMemoryBytes;
	double		exactBuildMs;
	uint64		exactHits;
	int64		exactEpoch;
	Oid			exactRelFileNode;
} HnswGuidanceDescriptorEntry;

typedef struct HnswMetadataFilterProfile
{
	bool		valid;
	bool		cacheHit;
	const char *cacheKind;
	int64		cacheRows;
	int64		cachePages;
	int64		candidates;
	int64		cacheChecks;
	int64		cacheMatches;
	int64		returned;
	int64		cacheMemoryBytes;
	double		cacheBuildMs;
	double		searchMs;
} HnswMetadataFilterProfile;

#define HNSW_GUIDANCE_MAX_ATOMS 128

typedef enum HnswGuidanceKind
{
	HNSW_GUIDANCE_KIND_OFF,
	HNSW_GUIDANCE_KIND_EXACT,
	HNSW_GUIDANCE_KIND_PAGE,
	HNSW_GUIDANCE_KIND_BLOOM
} HnswGuidanceKind;

typedef struct HnswGuidanceAtom
{
	HnswMetadataCacheEntry *cache;
	HnswGuidanceKind kind;
	bool		negated;
	int			group;
} HnswGuidanceAtom;

typedef struct HnswActiveGuidance
{
	bool		active;
	HnswGuidanceKind kind;
	Oid			heapOid;
	int			atoms;
	int			groups;
	int			negatedAtoms;
	double		lastBuildMs;
	int64		lastCacheRows;
	int64		lastCachePages;
	int64		lastCacheMemoryBytes;
	int64		fragmentCacheHits;
	int64		fragmentCacheMisses;
	int64		fragmentStoreHits;
	int64		fragmentBuilds;
	bool		composedGuideHit;
	int64		composedGuideHits;
	int64		composedGuideMisses;
	HnswGuidanceAtom atom[HNSW_GUIDANCE_MAX_ATOMS];
	bool		composedExactActive;
	bool		composedExactHit;
	HTAB	   *composedExactTidHash;
	int64		composedExactRows;
	int64		composedExactMemoryBytes;
	double		composedExactBuildMs;
	bool		epochTracked;
	int64		relationEpoch;
	Oid			relationRelFileNode;
} HnswActiveGuidance;

static HnswMaterializeProfile hnsw_materialize_last_profile;
static HTAB *hnsw_metadata_caches = NULL;
static HTAB *hnsw_guidance_descriptors = NULL;
static HnswMetadataFilterProfile hnsw_metadata_filter_last_profile;
static HnswActiveGuidance hnsw_active_guidance;
static int	hnsw_metadata_cache_max_mb = 64;
static bool hnsw_guidance_compose_exact_or = false;
static bool hnsw_guidance_require_epoch = true;
static bool hnsw_fragment_store_ready = false;
static uint64 hnsw_metadata_cache_clock = 0;
static int64 hnsw_metadata_cache_evictions = 0;

static void HnswMetadataFreeCacheEntry(HnswMetadataCacheEntry *entry);
static void HnswGuidanceFreeDescriptorEntry(HnswGuidanceDescriptorEntry *entry);
static void HnswGuidanceDeactivate(void);

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM >= 170000
#include "parser/scansup.h"
#endif

#define STATE_DIMS(x) (ARR_DIMS(x)[0] - 1)
#define CreateStateDatums(dim) palloc(sizeof(Datum) * (dim + 1))

#if defined(USE_TARGET_CLONES) && !defined(__FMA__)
#define VECTOR_TARGET_CLONES __attribute__((target_clones("default", "fma")))
#else
#define VECTOR_TARGET_CLONES
#endif

#if PG_VERSION_NUM >= 180000
PG_MODULE_MAGIC_EXT(.name = "vector",.version = "0.8.2");
#else
PG_MODULE_MAGIC;
#endif

/*
 * Initialize index options and variables
 */
PGDLLEXPORT void _PG_init(void);
void
_PG_init(void)
{
	BitvecInit();
	HalfvecInit();
	HnswInit();
	IvfflatInit();

	DefineCustomIntVariable("hnsw.metadata_cache_max_mb",
							"Sets the backend-local memory budget for HNSW fragment summaries",
							"Page, Bloom, and exact guidance fragments are evicted with an LRU policy when this budget is exceeded.",
							&hnsw_metadata_cache_max_mb,
							64, 1, 1024 * 1024, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.guidance_compose_exact_or",
							 "Builds a composed exact TID set for OR guidance predicates",
							 "When off, OR predicates are evaluated from cached atom fragments only.",
							 &hnsw_guidance_compose_exact_or,
								 false, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.guidance_require_epoch",
								 "Requires relation-epoch tracking before guidance can hard-prune HNSW candidates",
								 "Disable only for read-only compatibility experiments; tracked epochs are required for update-safe pruning.",
								 &hnsw_guidance_require_epoch,
								 true, PGC_USERSET, 0, NULL, NULL, NULL);
}

static void
VectorHnswLastProfileToText(StringInfo output, const HnswScanProfile *profile)
{
	appendStringInfo(output,
					"{\"valid\":%s,"
					"\"total_scan_ms\":%.6f,"
					"\"hnsw_search_ms\":%.6f,"
					"\"heap_fetch_ms\":%.6f,"
					"\"vector_search_ms\":%.6f,"
					"\"visited_tuples\":" INT64_FORMAT ","
					"\"returned_tuples\":" INT64_FORMAT ","
					"\"distance_compute_count\":" INT64_FORMAT ","
					"\"page_access_batches\":" INT64_FORMAT ","
					"\"page_access_candidates\":" INT64_FORMAT ","
					"\"page_access_prefetches\":" INT64_FORMAT ","
					"\"page_access_distance_runs\":" INT64_FORMAT ","
					"\"page_access_distinct_pages\":" INT64_FORMAT ","
					"\"guidance_checks\":" INT64_FORMAT ","
					"\"guidance_matches\":" INT64_FORMAT ","
					"\"guidance_skips\":" INT64_FORMAT ","
					"\"index_page_neighbor_loads\":" INT64_FORMAT ","
					"\"index_page_neighbor_runs\":" INT64_FORMAT ","
					"\"index_page_neighbor_distinct_pages\":" INT64_FORMAT ","
					"\"index_page_element_loads\":" INT64_FORMAT ","
					"\"index_page_element_runs\":" INT64_FORMAT ","
					"\"index_page_element_distinct_pages\":" INT64_FORMAT ","
					"\"index_page_prefetches\":" INT64_FORMAT ","
					"\"blks_hit_before\":" INT64_FORMAT ","
					"\"blks_hit_after\":" INT64_FORMAT ","
					"\"blks_read_before\":" INT64_FORMAT ","
					"\"blks_read_after\":" INT64_FORMAT ","
					"\"idx_blks_hit\":" INT64_FORMAT ","
					"\"idx_blks_read\":" INT64_FORMAT ","
					"\"heap_blks_hit\":" INT64_FORMAT ","
					"\"heap_blks_read\":" INT64_FORMAT ","
					"\"topk_count\":%d,"
					"\"topk_ids\":[",
					profile->valid ? "true" : "false",
					profile->totalScanMs,
					profile->hnswSearchMs,
					profile->heapFetchMs,
					profile->vectorSearchMs,
					profile->visitedTuples,
					profile->returnedTuples,
					profile->distanceComputations,
					profile->pageAccessBatches,
					profile->pageAccessCandidates,
					profile->pageAccessPrefetches,
					profile->pageAccessDistanceRuns,
					profile->pageAccessDistinctPages,
					profile->guidanceChecks,
					profile->guidanceMatches,
					profile->guidanceSkips,
					profile->indexPageNeighborLoads,
					profile->indexPageNeighborRuns,
					profile->indexPageNeighborDistinctPages,
					profile->indexPageElementLoads,
					profile->indexPageElementRuns,
					profile->indexPageElementDistinctPages,
					profile->indexPagePrefetches,
					profile->blksHitBefore,
					profile->blksHitAfter,
					profile->blksReadBefore,
					profile->blksReadAfter,
					profile->idxBlksHit,
					profile->idxBlksRead,
					profile->heapBlksHit,
					profile->heapBlksRead,
					profile->topkTidCount);

	for (int i = 0; i < profile->topkTidCount; i++)
	{
		if (i > 0)
			appendStringInfoChar(output, ',');
		appendStringInfo(output, "\"(%u,%u)\"",
						 ItemPointerGetBlockNumber(&profile->topkTids[i]),
						 ItemPointerGetOffsetNumber(&profile->topkTids[i]));
	}

	appendStringInfoString(output, "]}");
}

static void
ResetHnswMetadataFilterProfile(void)
{
	hnsw_metadata_filter_last_profile.valid = false;
	hnsw_metadata_filter_last_profile.cacheHit = false;
	hnsw_metadata_filter_last_profile.cacheKind = "none";
	hnsw_metadata_filter_last_profile.cacheRows = 0;
	hnsw_metadata_filter_last_profile.cachePages = 0;
	hnsw_metadata_filter_last_profile.candidates = 0;
	hnsw_metadata_filter_last_profile.cacheChecks = 0;
	hnsw_metadata_filter_last_profile.cacheMatches = 0;
	hnsw_metadata_filter_last_profile.returned = 0;
	hnsw_metadata_filter_last_profile.cacheMemoryBytes = 0;
	hnsw_metadata_filter_last_profile.cacheBuildMs = 0;
	hnsw_metadata_filter_last_profile.searchMs = 0;
}

static void
InitHnswMetadataCaches(void)
{
	HASHCTL		ctl;

	if (hnsw_metadata_caches != NULL)
		return;

	MemSet(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(HnswMetadataCacheKey);
	ctl.entrysize = sizeof(HnswMetadataCacheEntry);
	ctl.hcxt = TopMemoryContext;
	hnsw_metadata_caches = hash_create("hnsw metadata filter caches",
									   16,
									   &ctl,
									   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

static void
InitHnswGuidanceDescriptors(void)
{
	HASHCTL		ctl;

	if (hnsw_guidance_descriptors != NULL)
		return;

	MemSet(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(HnswGuidanceDescriptorKey);
	ctl.entrysize = sizeof(HnswGuidanceDescriptorEntry);
	ctl.hcxt = TopMemoryContext;
	hnsw_guidance_descriptors = hash_create("hnsw composed guidance descriptors",
											64,
											&ctl,
												HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

static void
HnswGuidanceFreeDescriptorEntry(HnswGuidanceDescriptorEntry *entry)
{
	if (entry->exactTidHash != NULL)
	{
		hash_destroy(entry->exactTidHash);
		entry->exactTidHash = NULL;
	}
	entry->exactRows = 0;
	entry->exactMemoryBytes = 0;
	entry->exactBuildMs = 0;
	entry->exactHits = 0;
	entry->exactEpoch = 0;
	entry->exactRelFileNode = InvalidOid;
}

static void
HnswMetadataResetCaches(void)
{
	HASH_SEQ_STATUS status;
	HnswMetadataCacheEntry *entry;
	HnswGuidanceDescriptorEntry *descriptor;

	if (hnsw_metadata_caches != NULL)
	{
		hash_seq_init(&status, hnsw_metadata_caches);
		while ((entry = (HnswMetadataCacheEntry *) hash_seq_search(&status)) != NULL)
			HnswMetadataFreeCacheEntry(entry);
		hash_destroy(hnsw_metadata_caches);
		hnsw_metadata_caches = NULL;
	}

	if (hnsw_guidance_descriptors != NULL)
	{
		hash_seq_init(&status, hnsw_guidance_descriptors);
		while ((descriptor = (HnswGuidanceDescriptorEntry *) hash_seq_search(&status)) != NULL)
			HnswGuidanceFreeDescriptorEntry(descriptor);
		hash_destroy(hnsw_guidance_descriptors);
		hnsw_guidance_descriptors = NULL;
	}

	hnsw_metadata_cache_clock = 0;
	hnsw_metadata_cache_evictions = 0;
	HnswGuidanceDeactivate();
}

static int64
HnswMetadataEntryMemoryBytes(HnswMetadataCacheEntry *cache)
{
	int64		bytes = 0;

	if (cache->tidHash != NULL)
		bytes += (int64) cache->rows * (int64) sizeof(HnswMetadataTidEntry);
	bytes += (int64) cache->pageBitBytes;
	bytes += (int64) cache->bloomBytes;
	return bytes;
}

static void
HnswMetadataTouchCache(HnswMetadataCacheEntry *cache)
{
	cache->lastUsed = ++hnsw_metadata_cache_clock;
	cache->memoryBytes = HnswMetadataEntryMemoryBytes(cache);
}

static int64
HnswMetadataCacheTotalBytes(void)
{
	HASH_SEQ_STATUS status;
	HnswMetadataCacheEntry *entry;
	int64		total = 0;

	if (hnsw_metadata_caches == NULL)
		return 0;

	hash_seq_init(&status, hnsw_metadata_caches);
	while ((entry = (HnswMetadataCacheEntry *) hash_seq_search(&status)) != NULL)
		total += HnswMetadataEntryMemoryBytes(entry);

	return total;
}

static void
HnswMetadataCacheStats(int64 *entries, int64 *residentEntries, int64 *residentBytes, int64 *largestEntryBytes)
{
	HASH_SEQ_STATUS status;
	HnswMetadataCacheEntry *entry;

	*entries = 0;
	*residentEntries = 0;
	*residentBytes = 0;
	*largestEntryBytes = 0;

	if (hnsw_metadata_caches == NULL)
		return;

	hash_seq_init(&status, hnsw_metadata_caches);
	while ((entry = (HnswMetadataCacheEntry *) hash_seq_search(&status)) != NULL)
	{
		int64		bytes = HnswMetadataEntryMemoryBytes(entry);

		(*entries)++;
		if (bytes > 0)
		{
			(*residentEntries)++;
			*residentBytes += bytes;
			if (bytes > *largestEntryBytes)
				*largestEntryBytes = bytes;
		}
	}
}

static void
HnswGuidanceDescriptorStats(int64 *entries, int64 *hits, int64 *exactEntries, int64 *exactRows, int64 *exactBytes, int64 *exactHits)
{
	HASH_SEQ_STATUS status;
	HnswGuidanceDescriptorEntry *entry;

	*entries = 0;
	*hits = 0;
	*exactEntries = 0;
	*exactRows = 0;
	*exactBytes = 0;
	*exactHits = 0;

	if (hnsw_guidance_descriptors == NULL)
		return;

	hash_seq_init(&status, hnsw_guidance_descriptors);
	while ((entry = (HnswGuidanceDescriptorEntry *) hash_seq_search(&status)) != NULL)
	{
		(*entries)++;
		*hits += entry->hits;
		if (entry->exactTidHash != NULL)
		{
			(*exactEntries)++;
			*exactRows += entry->exactRows;
			*exactBytes += entry->exactMemoryBytes;
			*exactHits += entry->exactHits;
		}
	}
}

static void
HnswMetadataFreeCacheEntry(HnswMetadataCacheEntry *entry)
{
	if (entry->tidHash != NULL)
	{
		hash_destroy(entry->tidHash);
		entry->tidHash = NULL;
	}
	if (entry->pageBits != NULL)
	{
		pfree(entry->pageBits);
		entry->pageBits = NULL;
	}
	if (entry->bloomBits != NULL)
	{
		pfree(entry->bloomBits);
		entry->bloomBits = NULL;
	}

	entry->rows = 0;
	entry->pageRows = 0;
	entry->pages = 0;
	entry->bloomRows = 0;
	entry->pageBitBytes = 0;
	entry->bloomBytes = 0;
	entry->bloomBitCount = 0;
	entry->buildMs = 0;
	entry->pageBuildMs = 0;
	entry->bloomBuildMs = 0;
	entry->memoryBytes = 0;
	entry->lastUsed = 0;
	entry->epochTracked = false;
	entry->buildEpoch = 0;
	entry->buildRelFileNode = InvalidOid;
}

static void
HnswMetadataEvictIfNeeded(HnswMetadataCacheEntry *protected)
{
	int64		limit = (int64) hnsw_metadata_cache_max_mb * 1024L * 1024L;

	while (HnswMetadataCacheTotalBytes() > limit)
	{
		HASH_SEQ_STATUS status;
		HnswMetadataCacheEntry *entry;
		HnswMetadataCacheEntry *victim = NULL;

		hash_seq_init(&status, hnsw_metadata_caches);
		while ((entry = (HnswMetadataCacheEntry *) hash_seq_search(&status)) != NULL)
		{
			if (entry == protected || HnswMetadataEntryMemoryBytes(entry) == 0)
				continue;
			if (victim == NULL || entry->lastUsed < victim->lastUsed)
				victim = entry;
		}

		if (victim == NULL)
			break;

		HnswMetadataFreeCacheEntry(victim);
		hnsw_metadata_cache_evictions++;
	}
}

static const char *
HnswMetadataPredicateSql(const char *filterName)
{
	if (strncmp(filterName, "sql:", 4) == 0)
		return filterName + 4;
	if (strcmp(filterName, "helpful_ge20") == 0)
		return "helpful_vote >= 20";
	if (strcmp(filterName, "grocery_long500") == 0)
		return "main_category = 'Grocery' AND review_text_len >= 500";
	if (strcmp(filterName, "grocery_helpful") == 0)
		return "main_category = 'Grocery' AND helpful_vote >= 1";
	if (strcmp(filterName, "rating5_price_le10") == 0)
		return "has_price AND price <= 10 AND rating = 5";
	if (strcmp(filterName, "c4_mixed_5_p_le10_pop_high") == 0)
		return "rating = 5 AND has_price AND price <= 10 AND item_rating_number >= 1000";
	if (strcmp(filterName, "c4_mixed_5_p_le10_pop_mid") == 0)
		return "rating = 5 AND has_price AND price <= 10 AND item_rating_number >= 100 AND item_rating_number < 1000";
	if (strcmp(filterName, "c4_mixed_5_p_le10_pop_low") == 0)
		return "rating = 5 AND has_price AND price <= 10 AND item_rating_number < 100";
	if (strcmp(filterName, "c4_mixed_5_p_10_20_pop_high") == 0)
		return "rating = 5 AND has_price AND price > 10 AND price <= 20 AND item_rating_number >= 1000";
	if (strcmp(filterName, "c4_mixed_5_p_10_20_pop_mid") == 0)
		return "rating = 5 AND has_price AND price > 10 AND price <= 20 AND item_rating_number >= 100 AND item_rating_number < 1000";
	if (strcmp(filterName, "c4_mixed_5_p_10_20_pop_low") == 0)
		return "rating = 5 AND has_price AND price > 10 AND price <= 20 AND item_rating_number < 100";
	if (strcmp(filterName, "c4_mixed_5_p_20_50_pop_high") == 0)
		return "rating = 5 AND has_price AND price > 20 AND price <= 50 AND item_rating_number >= 1000";
	if (strcmp(filterName, "c4_mixed_5_p_20_50_pop_mid") == 0)
		return "rating = 5 AND has_price AND price > 20 AND price <= 50 AND item_rating_number >= 100 AND item_rating_number < 1000";
	if (strcmp(filterName, "c4_mixed_5_p_20_50_pop_low") == 0)
		return "rating = 5 AND has_price AND price > 20 AND price <= 50 AND item_rating_number < 100";
	if (strcmp(filterName, "c4_mixed_5_p_gt50_pop_high") == 0)
		return "rating = 5 AND has_price AND price > 50 AND item_rating_number >= 1000";
	if (strcmp(filterName, "c4_mixed_5_p_gt50_pop_mid") == 0)
		return "rating = 5 AND has_price AND price > 50 AND item_rating_number >= 100 AND item_rating_number < 1000";
	if (strcmp(filterName, "c4_mixed_5_p_gt50_pop_low") == 0)
		return "rating = 5 AND has_price AND price > 50 AND item_rating_number < 100";
	if (strcmp(filterName, "c4_mixed_5_p_missing_pop_high") == 0)
		return "rating = 5 AND NOT has_price AND item_rating_number >= 1000";
	if (strcmp(filterName, "c4_mixed_5_p_missing_pop_mid") == 0)
		return "rating = 5 AND NOT has_price AND item_rating_number >= 100 AND item_rating_number < 1000";
	if (strcmp(filterName, "c4_mixed_5_p_missing_pop_low") == 0)
		return "rating = 5 AND NOT has_price AND item_rating_number < 100";

	ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("unsupported metadata cache filter \"%s\"", filterName),
			 errhint("Supported filters: helpful_ge20, grocery_long500, grocery_helpful, rating5_price_le10.")));
	return NULL;
}

static HnswMetadataCacheEntry *
FindHnswMetadataCache(Oid heapOid, const char *filterName, bool *found)
{
	HnswMetadataCacheKey key;

	InitHnswMetadataCaches();
	MemSet(&key, 0, sizeof(key));
	key.heapOid = heapOid;
	strlcpy(key.filterName, filterName, sizeof(key.filterName));
	{
		HnswMetadataCacheEntry *entry = (HnswMetadataCacheEntry *) hash_search(hnsw_metadata_caches, &key, HASH_ENTER, found);

		if (!*found)
		{
			HnswMetadataCacheKey savedKey = entry->key;

			MemSet(entry, 0, sizeof(HnswMetadataCacheEntry));
			entry->key = savedKey;
		}

		return entry;
	}
}

static char *
HnswMetadataQualifiedSource(Oid heapOid, const char **tidColumn)
{
	Oid			namespaceOid = get_rel_namespace(heapOid);
	char	   *namespaceName = get_namespace_name(namespaceOid);
	char	   *relationName = get_rel_name(heapOid);
	char	   *metaRelationName;
	Oid			metaOid;

	if (namespaceName == NULL || relationName == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_TABLE),
				 errmsg("could not resolve heap relation %u", heapOid)));

	metaRelationName = psprintf("%s_guidance_meta", relationName);
	metaOid = get_relname_relid(metaRelationName, namespaceOid);
	if (OidIsValid(metaOid))
	{
		*tidColumn = "heap_tid";
		return quote_qualified_identifier(namespaceName, metaRelationName);
	}

	*tidColumn = "ctid";
	return quote_qualified_identifier(namespaceName, relationName);
}

static bool
HnswMetadataPageBitTest(HnswMetadataCacheEntry *cache, BlockNumber block)
{
	Size		byte = block / 8;
	uint8		mask = 1 << (block % 8);

	if (cache->pageBits == NULL || byte >= cache->pageBitBytes)
		return false;
	return (cache->pageBits[byte] & mask) != 0;
}

static bool
HnswMetadataPageBitSet(HnswMetadataCacheEntry *cache, BlockNumber block)
{
	Size		byte = block / 8;
	uint8		mask = 1 << (block % 8);

	if (byte >= cache->pageBitBytes)
	{
		Size		oldBytes = cache->pageBitBytes;
		Size		newBytes = oldBytes > 0 ? oldBytes : 1024;
		MemoryContext oldCtx;

		while (byte >= newBytes)
			newBytes *= 2;

		oldCtx = MemoryContextSwitchTo(TopMemoryContext);
		if (cache->pageBits == NULL)
			cache->pageBits = (uint8 *) palloc0(newBytes);
		else
		{
			cache->pageBits = (uint8 *) repalloc(cache->pageBits, newBytes);
			MemSet(cache->pageBits + oldBytes, 0, newBytes - oldBytes);
		}
		MemoryContextSwitchTo(oldCtx);
		cache->pageBitBytes = newBytes;
	}

	if ((cache->pageBits[byte] & mask) != 0)
		return false;

	cache->pageBits[byte] |= mask;
	return true;
}

static uint64
HnswMetadataMix64(uint64 x)
{
	x ^= x >> 30;
	x *= UINT64CONST(0xbf58476d1ce4e5b9);
	x ^= x >> 27;
	x *= UINT64CONST(0x94d049bb133111eb);
	x ^= x >> 31;
	return x;
}

static uint64
HnswMetadataTidHash64(ItemPointer tid)
{
	uint64		block = ItemPointerGetBlockNumber(tid);
	uint64		offset = ItemPointerGetOffsetNumber(tid);

	return HnswMetadataMix64((block << 16) ^ offset);
}

static void
HnswMetadataBloomSet(HnswMetadataCacheEntry *cache, ItemPointer tid)
{
	uint64		h1 = HnswMetadataTidHash64(tid);
	uint64		h2 = HnswMetadataMix64(h1 ^ UINT64CONST(0x9e3779b97f4a7c15));

	for (int i = 0; i < 7; i++)
	{
		uint64		bit = (h1 + i * h2) % cache->bloomBitCount;

		cache->bloomBits[bit / 8] |= 1 << (bit % 8);
	}
}

static bool
HnswMetadataBloomMayContain(HnswMetadataCacheEntry *cache, ItemPointer tid)
{
	uint64		h1;
	uint64		h2;

	if (cache->bloomBits == NULL || cache->bloomBitCount == 0)
		return false;

	h1 = HnswMetadataTidHash64(tid);
	h2 = HnswMetadataMix64(h1 ^ UINT64CONST(0x9e3779b97f4a7c15));

	for (int i = 0; i < 7; i++)
	{
		uint64		bit = (h1 + i * h2) % cache->bloomBitCount;

		if ((cache->bloomBits[bit / 8] & (1 << (bit % 8))) == 0)
			return false;
	}

	return true;
}

static int64
HnswMetadataCacheMemoryBytes(HnswMetadataCacheEntry *cache, HnswGuidanceKind kind)
{
	switch (kind)
	{
		case HNSW_GUIDANCE_KIND_EXACT:
			return (int64) cache->rows * (int64) sizeof(HnswMetadataTidEntry) + (int64) cache->pageBitBytes;
		case HNSW_GUIDANCE_KIND_PAGE:
			return (int64) cache->pageBitBytes;
		case HNSW_GUIDANCE_KIND_BLOOM:
			return (int64) cache->bloomBytes;
		default:
			return 0;
	}
}

static const char *
HnswGuidanceKindName(HnswGuidanceKind kind);

static void
HnswMetadataEnsureFragmentStore(void)
{
	int			spiStatus;

	if (hnsw_fragment_store_ready)
		return;

	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);

	spiStatus = SPI_execute(
		"CREATE TABLE IF NOT EXISTS public.pgvector_hnsw_fragment_store ("
		"heap_oid oid NOT NULL,"
		"filter_name text NOT NULL,"
		"kind text NOT NULL,"
		"rows bigint NOT NULL,"
		"pages bigint NOT NULL,"
		"bloom_bit_count bigint NOT NULL,"
		"payload bytea NOT NULL,"
		"built_at timestamptz NOT NULL DEFAULT now(),"
		"PRIMARY KEY (heap_oid, filter_name, kind)"
		")",
		false, 0);
	if (spiStatus != SPI_OK_UTILITY)
		elog(ERROR, "SPI_execute failed: %d", spiStatus);

	spiStatus = SPI_execute(
		"CREATE TABLE IF NOT EXISTS public.pgvector_hnsw_fragment_epoch ("
		"heap_oid oid PRIMARY KEY,"
		"epoch bigint NOT NULL DEFAULT 0,"
		"updated_at timestamptz NOT NULL DEFAULT now()"
		")",
		false, 0);
	if (spiStatus != SPI_OK_UTILITY)
		elog(ERROR, "SPI_execute failed: %d", spiStatus);

	spiStatus = SPI_execute(
		"ALTER TABLE public.pgvector_hnsw_fragment_store "
		"ADD COLUMN IF NOT EXISTS build_epoch bigint NOT NULL DEFAULT 0, "
		"ADD COLUMN IF NOT EXISTS relfilenode oid NOT NULL DEFAULT 0",
		false, 0);
	if (spiStatus != SPI_OK_UTILITY)
		elog(ERROR, "SPI_execute failed: %d", spiStatus);

	SPI_finish();
	hnsw_fragment_store_ready = true;
}

static bool
HnswMetadataGetRelationVersion(Oid heapOid, int64 *epoch, Oid *relFileNode)
{
	int			spiStatus;
	Oid			argTypes[1] = {OIDOID};
	Datum		values[1] = {ObjectIdGetDatum(heapOid)};
	char		nulls[1] = {' '};
	bool		isnull;
	bool		relFileNodeIsNull;
	bool		tracked = false;

	*epoch = 0;
	*relFileNode = InvalidOid;
	HnswMetadataEnsureFragmentStore();

	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);

	spiStatus = SPI_execute_with_args(
			"SELECT e.epoch, pg_catalog.pg_relation_filenode($1) "
			"FROM (SELECT 1) AS singleton "
			"LEFT JOIN public.pgvector_hnsw_fragment_epoch AS e ON e.heap_oid = $1",
			1, argTypes, values, nulls, true, 1);
	if (spiStatus != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);

	if (SPI_processed == 1)
	{
		Datum		epochDatum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
		Datum		relFileNodeDatum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &relFileNodeIsNull);

		if (!isnull)
		{
			*epoch = DatumGetInt64(epochDatum);
				tracked = true;
			}
		if (!relFileNodeIsNull)
			*relFileNode = DatumGetObjectId(relFileNodeDatum);
	}

	SPI_finish();
	if (!OidIsValid(*relFileNode))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("relation %u does not have physical storage", heapOid)));
	return tracked;
}

PG_FUNCTION_INFO_V1(vector_hnsw_fragment_epoch_bump_trigger);
Datum
vector_hnsw_fragment_epoch_bump_trigger(PG_FUNCTION_ARGS)
{
	TriggerData *triggerData;
	Oid			heapOid;
	Oid			argTypes[1] = {OIDOID};
	Datum		values[1];
	char		nulls[1] = {' '};
	int			spiStatus;

	if (!CALLED_AS_TRIGGER(fcinfo))
		ereport(ERROR,
				(errcode(ERRCODE_E_R_I_E_TRIGGER_PROTOCOL_VIOLATED),
				 errmsg("vector_hnsw_fragment_epoch_bump_trigger must be called as a trigger")));

	triggerData = (TriggerData *) fcinfo->context;
	heapOid = RelationGetRelid(triggerData->tg_relation);
	values[0] = ObjectIdGetDatum(heapOid);

	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);
	spiStatus = SPI_execute_with_args(
		"INSERT INTO public.pgvector_hnsw_fragment_epoch (heap_oid, epoch) VALUES ($1, 1) "
		"ON CONFLICT (heap_oid) DO UPDATE SET epoch = "
		"public.pgvector_hnsw_fragment_epoch.epoch + 1, updated_at = now()",
		1, argTypes, values, nulls, false, 0);
	if (spiStatus != SPI_OK_INSERT && spiStatus != SPI_OK_UPDATE)
		elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);
	SPI_finish();

	PG_RETURN_POINTER(NULL);
}

PG_FUNCTION_INFO_V1(vector_hnsw_fragment_tracking_enable);
Datum
vector_hnsw_fragment_tracking_enable(PG_FUNCTION_ARGS)
{
	Oid			heapOid = PG_GETARG_OID(0);
	Oid			argTypes[1] = {OIDOID};
	Datum		values[1] = {ObjectIdGetDatum(heapOid)};
	char		nulls[1] = {' '};
	int			spiStatus;
	bool		hasTrigger = false;
	char	   *relName;
	char	   *namespaceName;
	char	   *qualifiedName;
	StringInfoData sql;
	int64		epoch;
	bool		isnull;

	relName = get_rel_name(heapOid);
	if (relName == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_TABLE),
				 errmsg("relation with OID %u does not exist", heapOid)));
	namespaceName = get_namespace_name(get_rel_namespace(heapOid));
	qualifiedName = quote_qualified_identifier(namespaceName, relName);
	HnswMetadataEnsureFragmentStore();

	initStringInfo(&sql);
	appendStringInfo(&sql, "LOCK TABLE %s IN SHARE ROW EXCLUSIVE MODE", qualifiedName);
	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);
	spiStatus = SPI_execute(sql.data, false, 0);
	if (spiStatus != SPI_OK_UTILITY)
		elog(ERROR, "SPI_execute failed: %d", spiStatus);
	SPI_finish();
	pfree(sql.data);

	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);
	spiStatus = SPI_execute_with_args(
			"INSERT INTO public.pgvector_hnsw_fragment_epoch (heap_oid, epoch) VALUES ($1, 0) "
			"ON CONFLICT (heap_oid) DO UPDATE SET heap_oid = EXCLUDED.heap_oid "
			"RETURNING epoch",
			1, argTypes, values, nulls, false, 0);
	if ((spiStatus != SPI_OK_INSERT_RETURNING && spiStatus != SPI_OK_UPDATE_RETURNING) || SPI_processed != 1)
		elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);
	epoch = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	if (isnull)
		elog(ERROR, "fragment epoch is null for relation %u", heapOid);

	spiStatus = SPI_execute_with_args(
		"SELECT 1 FROM pg_catalog.pg_trigger "
		"WHERE tgrelid = $1 AND tgname = 'pgvector_hnsw_fragment_epoch' AND NOT tgisinternal",
		1, argTypes, values, nulls, true, 1);
	if (spiStatus != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);
	hasTrigger = SPI_processed == 1;
	SPI_finish();

	if (!hasTrigger)
	{
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "CREATE TRIGGER pgvector_hnsw_fragment_epoch "
						 "AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON %s "
						 "FOR EACH STATEMENT EXECUTE FUNCTION public.vector_hnsw_fragment_epoch_bump_trigger()",
						 qualifiedName);
		spiStatus = SPI_connect();
		if (spiStatus != SPI_OK_CONNECT)
			elog(ERROR, "SPI_connect failed: %d", spiStatus);
		spiStatus = SPI_execute(sql.data, false, 0);
		if (spiStatus != SPI_OK_UTILITY)
			elog(ERROR, "SPI_execute failed: %d", spiStatus);
		SPI_finish();
		pfree(sql.data);
	}

	CommandCounterIncrement();
	pfree(qualifiedName);
	PG_RETURN_INT64(epoch);
}

static void
HnswMetadataCurrentCacheVersion(Oid heapOid, bool *tracked, int64 *epoch, Oid *relFileNode)
{
	*tracked = HnswMetadataGetRelationVersion(heapOid, epoch, relFileNode);
	if (hnsw_guidance_require_epoch && !*tracked)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("fragment epoch tracking is not enabled for relation %u", heapOid),
				 errhint("Call vector_hnsw_fragment_tracking_enable(%u::regclass) before activating guidance.", heapOid)));
}

static bool
HnswMetadataCacheVersionMatches(HnswMetadataCacheEntry *cache, bool tracked, int64 epoch, Oid relFileNode)
{
	if (cache->buildRelFileNode != relFileNode)
		return false;
	if (tracked)
		return cache->epochTracked && cache->buildEpoch == epoch;
	return !cache->epochTracked;
}

static void
HnswMetadataStampCacheVersion(HnswMetadataCacheEntry *cache, bool tracked, int64 epoch, Oid relFileNode)
{
	cache->epochTracked = tracked;
	cache->buildEpoch = epoch;
	cache->buildRelFileNode = relFileNode;
}

static bool
HnswMetadataLoadFragmentStore(Oid heapOid, const char *filterName, HnswGuidanceKind kind,
								  HnswMetadataCacheEntry *cache, bool tracked, int64 epoch, Oid relFileNode)
{
	int			spiStatus;
	Oid			argTypes[5] = {OIDOID, TEXTOID, TEXTOID, INT8OID, OIDOID};
	Datum		values[5];
	char		nulls[5] = {' ', ' ', ' ', ' ', ' '};
	bool		isnull;
	bool		loaded = false;
	const char *kindName = HnswGuidanceKindName(kind);

	HnswMetadataEnsureFragmentStore();

	values[0] = ObjectIdGetDatum(heapOid);
	values[1] = CStringGetTextDatum(filterName);
	values[2] = CStringGetTextDatum(kindName);
	values[3] = Int64GetDatum(epoch);
	values[4] = ObjectIdGetDatum(relFileNode);

	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);

	spiStatus = SPI_execute_with_args(
		"SELECT rows, pages, bloom_bit_count, payload "
		"FROM public.pgvector_hnsw_fragment_store "
		"WHERE heap_oid = $1 AND filter_name = $2 AND kind = $3 "
		"AND build_epoch = $4 AND relfilenode = $5",
		5, argTypes, values, nulls, true, 1);
	if (spiStatus != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);

	if (SPI_processed == 1)
	{
		HeapTuple	tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		payloadDatum;
		bytea	   *payload;
		Size		payloadBytes;
		MemoryContext oldCtx;

		payloadDatum = SPI_getbinval(tuple, tupdesc, 4, &isnull);
		if (!isnull)
		{
			payload = DatumGetByteaPP(payloadDatum);
			payloadBytes = VARSIZE_ANY_EXHDR(payload);

			oldCtx = MemoryContextSwitchTo(TopMemoryContext);
			if (kind == HNSW_GUIDANCE_KIND_PAGE)
			{
				if (cache->pageBits != NULL)
					pfree(cache->pageBits);
				cache->pageBits = (uint8 *) palloc(payloadBytes);
				memcpy(cache->pageBits, VARDATA_ANY(payload), payloadBytes);
				cache->pageBitBytes = payloadBytes;
				cache->pageRows = DatumGetInt64(SPI_getbinval(tuple, tupdesc, 1, &isnull));
				cache->pages = DatumGetInt64(SPI_getbinval(tuple, tupdesc, 2, &isnull));
				cache->pageBuildMs = 0;
				loaded = true;
			}
			else if (kind == HNSW_GUIDANCE_KIND_BLOOM)
			{
				if (cache->bloomBits != NULL)
					pfree(cache->bloomBits);
				cache->bloomBits = (uint8 *) palloc(payloadBytes);
				memcpy(cache->bloomBits, VARDATA_ANY(payload), payloadBytes);
				cache->bloomBytes = payloadBytes;
				cache->bloomRows = DatumGetInt64(SPI_getbinval(tuple, tupdesc, 1, &isnull));
				cache->bloomBitCount = DatumGetInt64(SPI_getbinval(tuple, tupdesc, 3, &isnull));
				cache->bloomBuildMs = 0;
				loaded = true;
			}
			MemoryContextSwitchTo(oldCtx);
		}
	}

	SPI_finish();
	if (loaded)
	{
		HnswMetadataStampCacheVersion(cache, tracked, epoch, relFileNode);
		HnswMetadataTouchCache(cache);
	}
	return loaded;
}

static void
HnswMetadataSaveFragmentStore(Oid heapOid, const char *filterName, HnswGuidanceKind kind, HnswMetadataCacheEntry *cache)
{
	int			spiStatus;
	Oid			argTypes[9] = {OIDOID, TEXTOID, TEXTOID, INT8OID, INT8OID, INT8OID, BYTEAOID, INT8OID, OIDOID};
	Datum		values[9];
	char		nulls[9] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};
	const char *kindName = HnswGuidanceKindName(kind);
	uint8	   *bits = NULL;
	Size		bytes = 0;
	int64		rows = 0;
	int64		pages = 0;
	int64		bloomBitCount = 0;
	bytea	   *payload;

	if (kind == HNSW_GUIDANCE_KIND_PAGE)
	{
		bits = cache->pageBits;
		bytes = cache->pageBitBytes;
		rows = cache->pageRows;
		pages = cache->pages;
	}
	else if (kind == HNSW_GUIDANCE_KIND_BLOOM)
	{
		bits = cache->bloomBits;
		bytes = cache->bloomBytes;
		rows = cache->bloomRows;
		bloomBitCount = cache->bloomBitCount;
	}

	if (bits == NULL || bytes == 0)
		return;

	HnswMetadataEnsureFragmentStore();

	payload = (bytea *) palloc(VARHDRSZ + bytes);
	SET_VARSIZE(payload, VARHDRSZ + bytes);
	memcpy(VARDATA(payload), bits, bytes);

	values[0] = ObjectIdGetDatum(heapOid);
	values[1] = CStringGetTextDatum(filterName);
	values[2] = CStringGetTextDatum(kindName);
	values[3] = Int64GetDatum(rows);
	values[4] = Int64GetDatum(pages);
	values[5] = Int64GetDatum(bloomBitCount);
	values[6] = PointerGetDatum(payload);
	values[7] = Int64GetDatum(cache->buildEpoch);
	values[8] = ObjectIdGetDatum(cache->buildRelFileNode);

	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);

	spiStatus = SPI_execute_with_args(
		"INSERT INTO public.pgvector_hnsw_fragment_store "
		"(heap_oid, filter_name, kind, rows, pages, bloom_bit_count, payload, build_epoch, relfilenode) "
		"VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) "
		"ON CONFLICT (heap_oid, filter_name, kind) DO UPDATE SET "
		"rows = EXCLUDED.rows,"
		"pages = EXCLUDED.pages,"
		"bloom_bit_count = EXCLUDED.bloom_bit_count,"
		"payload = EXCLUDED.payload,"
		"build_epoch = EXCLUDED.build_epoch,"
		"relfilenode = EXCLUDED.relfilenode,"
		"built_at = now()",
		9, argTypes, values, nulls, false, 0);
	if (spiStatus != SPI_OK_INSERT && spiStatus != SPI_OK_UPDATE && spiStatus != SPI_OK_INSERT_RETURNING)
	{
		/* INSERT ... ON CONFLICT reports SPI_OK_INSERT on supported versions. */
		if (spiStatus != SPI_OK_UTILITY)
			elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);
	}

	SPI_finish();
	pfree(payload);
}

static HnswMetadataCacheEntry *
BuildHnswMetadataCache(Oid heapOid, const char *filterName)
{
	bool		found;
	HnswMetadataCacheEntry *cache;
	HASHCTL		ctl;
	StringInfoData sql;
	char	   *qualifiedName;
	const char *tidColumn;
	int			spiStatus;
	const char *predicate;
	MemoryContext oldCtx;
	instr_time	start;
	instr_time	elapsed;
	bool		populatePageBits;

	cache = FindHnswMetadataCache(heapOid, filterName, &found);
	if (found && cache->tidHash != NULL)
		return cache;
	if (!found || cache->tidHash == NULL)
	{
		cache->tidHash = NULL;
		cache->rows = 0;
		cache->buildMs = 0;
		if (!found)
		{
			cache->pageBits = NULL;
			cache->bloomBits = NULL;
			cache->pageRows = 0;
			cache->pages = 0;
			cache->bloomRows = 0;
			cache->pageBitBytes = 0;
			cache->bloomBytes = 0;
			cache->bloomBitCount = 0;
			cache->pageBuildMs = 0;
			cache->bloomBuildMs = 0;
		}
		else if (cache->pageBits == NULL)
		{
			cache->pageRows = 0;
			cache->pages = 0;
			cache->pageBitBytes = 0;
			cache->pageBuildMs = 0;
		}
	}
	populatePageBits = cache->pageBits == NULL;

	predicate = HnswMetadataPredicateSql(filterName);
	qualifiedName = HnswMetadataQualifiedSource(heapOid, &tidColumn);

	oldCtx = MemoryContextSwitchTo(TopMemoryContext);
	MemSet(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(HnswMetadataTidKey);
	ctl.entrysize = sizeof(HnswMetadataTidEntry);
	ctl.hcxt = TopMemoryContext;
	cache->tidHash = hash_create("hnsw metadata passing tids",
								 1024,
								 &ctl,
								 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	MemoryContextSwitchTo(oldCtx);

	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT %s, id FROM %s WHERE %s", tidColumn, qualifiedName, predicate);

	INSTR_TIME_SET_CURRENT(start);
	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);
	spiStatus = SPI_execute(sql.data, true, 0);
	if (spiStatus != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute failed: %d", spiStatus);

	for (uint64 i = 0; i < SPI_processed; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		bool		isnull;
		Datum		ctidDatum;
		Datum		idDatum;
		ItemPointer ctid;
		HnswMetadataTidKey tidKey;
		HnswMetadataTidEntry *entry;
		bool		tidFound;
		BlockNumber block;

		ctidDatum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
		if (isnull)
			continue;
		ctid = (ItemPointer) DatumGetPointer(ctidDatum);
		block = ItemPointerGetBlockNumber(ctid);
		if (populatePageBits)
		{
			if (HnswMetadataPageBitSet(cache, block))
				cache->pages++;
			cache->pageRows++;
		}

		idDatum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
		if (isnull)
			continue;

		tidKey.tid = *ctid;
		entry = (HnswMetadataTidEntry *) hash_search(cache->tidHash, &tidKey, HASH_ENTER, &tidFound);
		entry->id = DatumGetInt64(idDatum);
		if (!tidFound)
			cache->rows++;
	}

	SPI_finish();
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	cache->buildMs = INSTR_TIME_GET_MILLISEC(elapsed);

	pfree(sql.data);
	return cache;
}

static HnswMetadataCacheEntry *
BuildHnswMetadataPageCache(Oid heapOid, const char *filterName)
{
	bool		found;
	HnswMetadataCacheEntry *cache;
	StringInfoData sql;
	char	   *qualifiedName;
	const char *tidColumn;
	int			spiStatus;
	const char *predicate;
	instr_time	start;
	instr_time	elapsed;

	cache = FindHnswMetadataCache(heapOid, filterName, &found);
	if (found && cache->pageBits != NULL)
		return cache;
	if (!found)
	{
		cache->tidHash = NULL;
		cache->pageBits = NULL;
		cache->bloomBits = NULL;
		cache->rows = 0;
		cache->pageRows = 0;
		cache->pages = 0;
		cache->bloomRows = 0;
		cache->pageBitBytes = 0;
		cache->bloomBytes = 0;
		cache->bloomBitCount = 0;
		cache->buildMs = 0;
		cache->pageBuildMs = 0;
		cache->bloomBuildMs = 0;
	}
	else
	{
		cache->pageBits = NULL;
		cache->pageRows = 0;
		cache->pages = 0;
		cache->pageBitBytes = 0;
		cache->pageBuildMs = 0;
	}

	predicate = HnswMetadataPredicateSql(filterName);
	qualifiedName = HnswMetadataQualifiedSource(heapOid, &tidColumn);

	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT %s FROM %s WHERE %s", tidColumn, qualifiedName, predicate);

	INSTR_TIME_SET_CURRENT(start);
	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);
	spiStatus = SPI_execute(sql.data, true, 0);
	if (spiStatus != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute failed: %d", spiStatus);

	for (uint64 i = 0; i < SPI_processed; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		bool		isnull;
		Datum		ctidDatum;
		ItemPointer ctid;
		BlockNumber block;

		ctidDatum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
		if (isnull)
			continue;
		ctid = (ItemPointer) DatumGetPointer(ctidDatum);
		block = ItemPointerGetBlockNumber(ctid);
		if (HnswMetadataPageBitSet(cache, block))
			cache->pages++;
		cache->pageRows++;
	}

	SPI_finish();
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	cache->pageBuildMs = INSTR_TIME_GET_MILLISEC(elapsed);

	pfree(sql.data);
	return cache;
}

static HnswMetadataCacheEntry *
BuildHnswMetadataBloomCache(Oid heapOid, const char *filterName)
{
	bool		found;
	HnswMetadataCacheEntry *cache;
	StringInfoData sql;
	char	   *qualifiedName;
	const char *tidColumn;
	int			spiStatus;
	const char *predicate;
	MemoryContext oldCtx;
	instr_time	start;
	instr_time	elapsed;

	cache = FindHnswMetadataCache(heapOid, filterName, &found);
	if (found && cache->bloomBits != NULL)
		return cache;
	if (!found)
	{
		cache->tidHash = NULL;
		cache->pageBits = NULL;
		cache->bloomBits = NULL;
		cache->rows = 0;
		cache->pageRows = 0;
		cache->pages = 0;
		cache->bloomRows = 0;
		cache->pageBitBytes = 0;
		cache->bloomBytes = 0;
		cache->bloomBitCount = 0;
		cache->buildMs = 0;
		cache->pageBuildMs = 0;
		cache->bloomBuildMs = 0;
	}
	else
	{
		cache->bloomBits = NULL;
		cache->bloomRows = 0;
		cache->bloomBytes = 0;
		cache->bloomBitCount = 0;
		cache->bloomBuildMs = 0;
	}

	predicate = HnswMetadataPredicateSql(filterName);
	qualifiedName = HnswMetadataQualifiedSource(heapOid, &tidColumn);

	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT %s FROM %s WHERE %s", tidColumn, qualifiedName, predicate);

	INSTR_TIME_SET_CURRENT(start);
	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);
	spiStatus = SPI_execute(sql.data, true, 0);
	if (spiStatus != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute failed: %d", spiStatus);

	cache->bloomRows = SPI_processed;
	cache->bloomBitCount = Max((uint64) 1024, (uint64) cache->bloomRows * 10);
	cache->bloomBytes = (cache->bloomBitCount + 7) / 8;

	oldCtx = MemoryContextSwitchTo(TopMemoryContext);
	cache->bloomBits = (uint8 *) palloc0(cache->bloomBytes);
	MemoryContextSwitchTo(oldCtx);

	for (uint64 i = 0; i < SPI_processed; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		bool		isnull;
		Datum		ctidDatum;
		ItemPointer ctid;

		ctidDatum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
		if (isnull)
			continue;
		ctid = (ItemPointer) DatumGetPointer(ctidDatum);
		HnswMetadataBloomSet(cache, ctid);
	}

	SPI_finish();
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	cache->bloomBuildMs = INSTR_TIME_GET_MILLISEC(elapsed);

	pfree(sql.data);
	return cache;
}

static HnswMetadataCacheEntry *
GetHnswMetadataCache(Oid heapOid, const char *filterName, bool buildIfMissing, bool *cacheHit, bool *storeHit)
{
	bool		found;
	bool		tracked;
	int64		epoch;
	Oid			relFileNode;
	HnswMetadataCacheEntry *cache;

	if (storeHit != NULL)
		*storeHit = false;
	HnswMetadataCurrentCacheVersion(heapOid, &tracked, &epoch, &relFileNode);
	cache = FindHnswMetadataCache(heapOid, filterName, &found);
	if (found && cache->tidHash != NULL)
	{
		if (HnswMetadataCacheVersionMatches(cache, tracked, epoch, relFileNode))
		{
			HnswMetadataTouchCache(cache);
			if (cacheHit != NULL)
				*cacheHit = true;
			return cache;
		}
		HnswMetadataFreeCacheEntry(cache);
	}

	if (!buildIfMissing)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("metadata cache for filter \"%s\" has not been built", filterName)));

	if (cacheHit != NULL)
		*cacheHit = false;
	cache = BuildHnswMetadataCache(heapOid, filterName);
	HnswMetadataStampCacheVersion(cache, tracked, epoch, relFileNode);
	HnswMetadataTouchCache(cache);
	HnswMetadataEvictIfNeeded(cache);
	return cache;
}

static HnswMetadataCacheEntry *
GetHnswMetadataPageCache(Oid heapOid, const char *filterName, bool buildIfMissing, bool *cacheHit, bool *storeHit)
{
	bool		found;
	bool		tracked;
	int64		epoch;
	Oid			relFileNode;
	HnswMetadataCacheEntry *cache;

	if (storeHit != NULL)
		*storeHit = false;
	HnswMetadataCurrentCacheVersion(heapOid, &tracked, &epoch, &relFileNode);
	cache = FindHnswMetadataCache(heapOid, filterName, &found);
	if (found && cache->pageBits != NULL)
	{
		if (HnswMetadataCacheVersionMatches(cache, tracked, epoch, relFileNode))
		{
			HnswMetadataTouchCache(cache);
			if (cacheHit != NULL)
				*cacheHit = true;
			return cache;
		}
		HnswMetadataFreeCacheEntry(cache);
	}

	if (!buildIfMissing)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("metadata page cache for filter \"%s\" has not been built", filterName)));

	if (cacheHit != NULL)
		*cacheHit = false;
	if (HnswMetadataLoadFragmentStore(heapOid, filterName, HNSW_GUIDANCE_KIND_PAGE,
									   cache, tracked, epoch, relFileNode))
	{
		if (storeHit != NULL)
			*storeHit = true;
		HnswMetadataEvictIfNeeded(cache);
		return cache;
	}

	cache = BuildHnswMetadataPageCache(heapOid, filterName);
	HnswMetadataStampCacheVersion(cache, tracked, epoch, relFileNode);
	HnswMetadataTouchCache(cache);
	HnswMetadataSaveFragmentStore(heapOid, filterName, HNSW_GUIDANCE_KIND_PAGE, cache);
	HnswMetadataEvictIfNeeded(cache);
	return cache;
}

static HnswMetadataCacheEntry *
GetHnswMetadataBloomCache(Oid heapOid, const char *filterName, bool buildIfMissing, bool *cacheHit, bool *storeHit)
{
	bool		found;
	bool		tracked;
	int64		epoch;
	Oid			relFileNode;
	HnswMetadataCacheEntry *cache;

	if (storeHit != NULL)
		*storeHit = false;
	HnswMetadataCurrentCacheVersion(heapOid, &tracked, &epoch, &relFileNode);
	cache = FindHnswMetadataCache(heapOid, filterName, &found);
	if (found && cache->bloomBits != NULL)
	{
		if (HnswMetadataCacheVersionMatches(cache, tracked, epoch, relFileNode))
		{
			HnswMetadataTouchCache(cache);
			if (cacheHit != NULL)
				*cacheHit = true;
			return cache;
		}
		HnswMetadataFreeCacheEntry(cache);
	}

	if (!buildIfMissing)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("metadata bloom cache for filter \"%s\" has not been built", filterName)));

	if (cacheHit != NULL)
		*cacheHit = false;
	if (HnswMetadataLoadFragmentStore(heapOid, filterName, HNSW_GUIDANCE_KIND_BLOOM,
									   cache, tracked, epoch, relFileNode))
	{
		if (storeHit != NULL)
			*storeHit = true;
		HnswMetadataEvictIfNeeded(cache);
		return cache;
	}

	cache = BuildHnswMetadataBloomCache(heapOid, filterName);
	HnswMetadataStampCacheVersion(cache, tracked, epoch, relFileNode);
	HnswMetadataTouchCache(cache);
	HnswMetadataSaveFragmentStore(heapOid, filterName, HNSW_GUIDANCE_KIND_BLOOM, cache);
	HnswMetadataEvictIfNeeded(cache);
	return cache;
}

static const char *
HnswGuidanceKindName(HnswGuidanceKind kind)
{
	switch (kind)
	{
		case HNSW_GUIDANCE_KIND_EXACT:
			return "exact";
		case HNSW_GUIDANCE_KIND_PAGE:
			return "page";
		case HNSW_GUIDANCE_KIND_BLOOM:
			return "bloom";
		default:
			return "off";
	}
}

static HnswGuidanceKind
HnswGuidanceKindFromText(const char *kindName)
{
	if (pg_strcasecmp(kindName, "exact") == 0 || pg_strcasecmp(kindName, "tid") == 0)
		return HNSW_GUIDANCE_KIND_EXACT;
	if (pg_strcasecmp(kindName, "page") == 0)
		return HNSW_GUIDANCE_KIND_PAGE;
	if (pg_strcasecmp(kindName, "bloom") == 0)
		return HNSW_GUIDANCE_KIND_BLOOM;

	ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("unsupported HNSW guidance kind \"%s\"", kindName),
			 errhint("Supported kinds: exact, page, bloom.")));
	return HNSW_GUIDANCE_KIND_OFF;
}

static const char *
HnswGuidanceParseAtomKind(const char *atomName, HnswGuidanceKind defaultKind, HnswGuidanceKind *kind)
{
	const char *colon = strchr(atomName, ':');

	*kind = defaultKind;
	if (colon == NULL)
		return atomName;

	if (pg_strncasecmp(atomName, "exact", colon - atomName) == 0 ||
		pg_strncasecmp(atomName, "tid", colon - atomName) == 0)
		*kind = HNSW_GUIDANCE_KIND_EXACT;
	else if (pg_strncasecmp(atomName, "page", colon - atomName) == 0)
		*kind = HNSW_GUIDANCE_KIND_PAGE;
	else if (pg_strncasecmp(atomName, "bloom", colon - atomName) == 0)
		*kind = HNSW_GUIDANCE_KIND_BLOOM;
	else
		return atomName;

	if (*(colon + 1) == '\0')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("empty HNSW guidance atom after kind prefix")));

	return colon + 1;
}

static bool
HnswGuidanceCacheAllowsTid(HnswMetadataCacheEntry *cache, HnswGuidanceKind kind, ItemPointer tid)
{
	switch (kind)
	{
		case HNSW_GUIDANCE_KIND_EXACT:
		{
			HnswMetadataTidKey tidKey;

			if (cache->pageBits != NULL && !HnswMetadataPageBitTest(cache, ItemPointerGetBlockNumber(tid)))
				return false;

			tidKey.tid = *tid;
			return hash_search(cache->tidHash, &tidKey, HASH_FIND, NULL) != NULL;
		}
		case HNSW_GUIDANCE_KIND_PAGE:
			return HnswMetadataPageBitTest(cache, ItemPointerGetBlockNumber(tid));
		case HNSW_GUIDANCE_KIND_BLOOM:
			return HnswMetadataBloomMayContain(cache, tid);
		default:
			return true;
	}
}

static bool
HnswGuidanceCanComposeExactOr(HnswActiveGuidance *guidance)
{
	if (guidance->atoms < 2 || guidance->groups < 2 || guidance->negatedAtoms > 0)
		return false;

	for (int group = 0; group < guidance->groups; group++)
	{
		int			groupAtoms = 0;

		for (int i = 0; i < guidance->atoms; i++)
		{
			HnswGuidanceAtom *atom = &guidance->atom[i];

			if (atom->group != group)
				continue;

			if (atom->kind != HNSW_GUIDANCE_KIND_EXACT || atom->cache == NULL || atom->cache->tidHash == NULL)
				return false;

			groupAtoms++;
		}

		if (groupAtoms != 1)
			return false;
	}

	return true;
}

static void
HnswGuidanceUseComposedExact(HnswActiveGuidance *guidance, HnswGuidanceDescriptorEntry *descriptor, bool cacheHit)
{
	guidance->composedExactActive = true;
	guidance->composedExactHit = cacheHit;
	guidance->composedExactTidHash = descriptor->exactTidHash;
	guidance->composedExactRows = descriptor->exactRows;
	guidance->composedExactMemoryBytes = descriptor->exactMemoryBytes;
	guidance->composedExactBuildMs = cacheHit ? 0 : descriptor->exactBuildMs;
	if (!cacheHit)
		guidance->lastBuildMs += descriptor->exactBuildMs;
}

static void
HnswGuidanceBuildComposedExactOr(HnswActiveGuidance *guidance, HnswGuidanceDescriptorEntry *descriptor)
{
	HASHCTL		ctl;
	instr_time	start;
	instr_time	elapsed;
	if (!HnswGuidanceCanComposeExactOr(guidance))
		return;

	if (descriptor->exactTidHash != NULL &&
		(descriptor->exactEpoch != guidance->relationEpoch ||
		 descriptor->exactRelFileNode != guidance->relationRelFileNode))
		HnswGuidanceFreeDescriptorEntry(descriptor);

	if (descriptor->exactTidHash != NULL)
	{
		descriptor->exactHits++;
		HnswGuidanceUseComposedExact(guidance, descriptor, true);
		return;
	}

	MemSet(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(HnswMetadataTidKey);
	ctl.entrysize = sizeof(HnswMetadataTidEntry);
	ctl.hcxt = TopMemoryContext;

	descriptor->exactTidHash = hash_create("hnsw composed exact guidance tids",
										   1024,
										   &ctl,
										   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	INSTR_TIME_SET_CURRENT(start);
	for (int i = 0; i < guidance->atoms; i++)
	{
		HASH_SEQ_STATUS status;
		HnswMetadataTidEntry *source;

		hash_seq_init(&status, guidance->atom[i].cache->tidHash);
		while ((source = (HnswMetadataTidEntry *) hash_seq_search(&status)) != NULL)
		{
			HnswMetadataTidEntry *entry;
			bool		found;

			entry = (HnswMetadataTidEntry *) hash_search(descriptor->exactTidHash, &source->key, HASH_ENTER, &found);
			if (!found)
			{
				entry->id = source->id;
				descriptor->exactRows++;
			}
		}
	}
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);

	descriptor->exactBuildMs = INSTR_TIME_GET_MILLISEC(elapsed);
	descriptor->exactMemoryBytes = descriptor->exactRows * (int64) sizeof(HnswMetadataTidEntry);
	descriptor->exactHits = 0;
	descriptor->exactEpoch = guidance->relationEpoch;
	descriptor->exactRelFileNode = guidance->relationRelFileNode;
	HnswGuidanceUseComposedExact(guidance, descriptor, false);

	/* Touch metadata caches after the potentially long merge to keep LRU roughly query-local. */
	for (int i = 0; i < guidance->atoms; i++)
	{
		if (guidance->atom[i].cache != NULL)
			HnswMetadataTouchCache(guidance->atom[i].cache);
	}
}

bool
HnswGuidanceIsActive(void)
{
	return hnsw_active_guidance.active;
}

bool
HnswGuidanceIsActiveForHeap(Oid heapOid)
{
	return hnsw_active_guidance.active &&
		OidIsValid(heapOid) &&
			hnsw_active_guidance.heapOid == heapOid;
}

bool
HnswGuidancePrepareForScan(Oid heapOid)
{
	bool		tracked;
	int64		epoch;
	Oid			relFileNode;

	if (!HnswGuidanceIsActiveForHeap(heapOid))
		return false;

	tracked = HnswMetadataGetRelationVersion(heapOid, &epoch, &relFileNode);
	if ((hnsw_guidance_require_epoch && !tracked) ||
		tracked != hnsw_active_guidance.epochTracked ||
		(tracked && epoch != hnsw_active_guidance.relationEpoch) ||
		relFileNode != hnsw_active_guidance.relationRelFileNode)
	{
		/* A stale guide must never remove candidates from a newer table version. */
		HnswGuidanceDeactivate();
		return false;
	}

	return true;
}

bool
HnswGuidanceAllowsTid(ItemPointer tid)
{
	if (!hnsw_active_guidance.active)
		return true;

	if (hnsw_active_guidance.composedExactActive && hnsw_active_guidance.composedExactTidHash != NULL)
	{
		HnswMetadataTidKey tidKey;

		tidKey.tid = *tid;
		return hash_search(hnsw_active_guidance.composedExactTidHash, &tidKey, HASH_FIND, NULL) != NULL;
	}

	for (int group = 0; group < hnsw_active_guidance.groups; group++)
	{
		bool		groupMatches = true;

		for (int i = 0; i < hnsw_active_guidance.atoms; i++)
		{
			HnswGuidanceAtom *atom = &hnsw_active_guidance.atom[i];
			bool		matches;

			if (atom->group != group)
				continue;

			matches = HnswGuidanceCacheAllowsTid(atom->cache, atom->kind, tid);

			if (atom->negated)
				matches = !matches;
			if (!matches)
			{
				groupMatches = false;
				break;
			}
		}

		if (groupMatches)
			return true;
	}

	return false;
}

static void
HnswGuidanceDeactivate(void)
{
	MemSet(&hnsw_active_guidance, 0, sizeof(hnsw_active_guidance));
}

PG_FUNCTION_INFO_V1(vector_hnsw_guidance_reset);
Datum
vector_hnsw_guidance_reset(PG_FUNCTION_ARGS)
{
	HnswGuidanceDeactivate();
	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(vector_hnsw_guidance_activate);
Datum
vector_hnsw_guidance_activate(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	ArrayType  *filterArray = PG_GETARG_ARRAYTYPE_P(1);
	text	   *kindText = PG_GETARG_TEXT_PP(2);
	char	   *kindName = text_to_cstring(kindText);
	HnswGuidanceKind kind = HnswGuidanceKindFromText(kindName);
	Oid			heapOid = IndexGetRelation(indexOid, false);
	Datum	   *filterDatums;
	bool	   *filterNulls;
	int			filterCount;
	HnswActiveGuidance nextGuidance;
	StringInfoData signature;
	HnswGuidanceDescriptorKey descriptorKey;
	HnswGuidanceDescriptorEntry *descriptor;
	bool		descriptorFound;
	bool		epochTracked;
	bool		finalEpochTracked;
	int64		relationEpoch;
	int64		finalRelationEpoch;
	Oid			relationRelFileNode;
	Oid			finalRelationRelFileNode;

	deconstruct_array(filterArray, TEXTOID, -1, false, 'i', &filterDatums, &filterNulls, &filterCount);
	if (filterCount < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("at least one guidance atom is required")));
	initStringInfo(&signature);
	appendStringInfo(&signature, "kind=%s", HnswGuidanceKindName(kind));
	for (int i = 0; i < filterCount; i++)
	{
		char	   *rawName;

		if (filterNulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("guidance atom names cannot be null")));
		rawName = text_to_cstring(DatumGetTextPP(filterDatums[i]));
		appendStringInfo(&signature, "|%s", rawName);
		}

	HnswMetadataCurrentCacheVersion(heapOid, &epochTracked, &relationEpoch, &relationRelFileNode);

	InitHnswGuidanceDescriptors();
	MemSet(&descriptorKey, 0, sizeof(descriptorKey));
	descriptorKey.heapOid = heapOid;
	strlcpy(descriptorKey.signature, signature.data, sizeof(descriptorKey.signature));
	descriptor = (HnswGuidanceDescriptorEntry *) hash_search(hnsw_guidance_descriptors, &descriptorKey, HASH_ENTER, &descriptorFound);
	if (!descriptorFound)
	{
		HnswGuidanceDescriptorKey savedKey = descriptor->key;

		MemSet(descriptor, 0, sizeof(HnswGuidanceDescriptorEntry));
		descriptor->key = savedKey;
	}
	else
		descriptor->hits++;
	descriptor->lastUsed = ++hnsw_metadata_cache_clock;

	MemSet(&nextGuidance, 0, sizeof(nextGuidance));
	nextGuidance.kind = kind;
	nextGuidance.heapOid = heapOid;
	nextGuidance.groups = 1;
	nextGuidance.composedGuideHit = descriptorFound;
	nextGuidance.composedGuideHits = descriptorFound ? 1 : 0;
	nextGuidance.composedGuideMisses = descriptorFound ? 0 : 1;
	nextGuidance.epochTracked = epochTracked;
	nextGuidance.relationEpoch = relationEpoch;
	nextGuidance.relationRelFileNode = relationRelFileNode;

	for (int i = 0; i < filterCount; i++)
	{
		char	   *filterName;
		bool		negated = false;
		bool		cacheHit = false;
		bool		storeHit = false;
		HnswMetadataCacheEntry *cache = NULL;
		HnswGuidanceKind atomKind;
		int			atomIndex;

		if (filterNulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("guidance atom names cannot be null")));

		filterName = text_to_cstring(DatumGetTextPP(filterDatums[i]));
		if (strcmp(filterName, "|") == 0 || pg_strcasecmp(filterName, "OR") == 0)
		{
			if (nextGuidance.atoms == 0 || nextGuidance.atom[nextGuidance.atoms - 1].group != nextGuidance.groups - 1)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("empty HNSW guidance OR group")));
			nextGuidance.groups++;
			continue;
		}

		if (filterName[0] == '!')
		{
			negated = true;
			filterName++;
		}

		filterName = (char *) HnswGuidanceParseAtomKind(filterName, kind, &atomKind);
		if (negated && atomKind != HNSW_GUIDANCE_KIND_EXACT)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("negated guidance atoms require exact kind"),
					 errhint("Page and bloom guidance can have false positives, so NOT would be unsafe.")));

		switch (atomKind)
		{
			case HNSW_GUIDANCE_KIND_EXACT:
				cache = GetHnswMetadataCache(heapOid, filterName, true, &cacheHit, &storeHit);
				nextGuidance.lastCacheRows += cache->rows;
				nextGuidance.lastCacheMemoryBytes += HnswMetadataCacheMemoryBytes(cache, atomKind);
				nextGuidance.lastBuildMs += cacheHit ? 0 : cache->buildMs;
				break;
			case HNSW_GUIDANCE_KIND_PAGE:
				cache = GetHnswMetadataPageCache(heapOid, filterName, true, &cacheHit, &storeHit);
				nextGuidance.lastCacheRows += cache->pageRows;
				nextGuidance.lastCachePages += cache->pages;
				nextGuidance.lastCacheMemoryBytes += HnswMetadataCacheMemoryBytes(cache, atomKind);
				nextGuidance.lastBuildMs += (cacheHit || storeHit) ? 0 : cache->pageBuildMs;
				break;
			case HNSW_GUIDANCE_KIND_BLOOM:
				cache = GetHnswMetadataBloomCache(heapOid, filterName, true, &cacheHit, &storeHit);
				nextGuidance.lastCacheRows += cache->bloomRows;
				nextGuidance.lastCacheMemoryBytes += HnswMetadataCacheMemoryBytes(cache, atomKind);
				nextGuidance.lastBuildMs += (cacheHit || storeHit) ? 0 : cache->bloomBuildMs;
				break;
			default:
				break;
		}
		if (cacheHit)
			nextGuidance.fragmentCacheHits++;
		else
		{
			nextGuidance.fragmentCacheMisses++;
			if (storeHit)
				nextGuidance.fragmentStoreHits++;
			else
				nextGuidance.fragmentBuilds++;
		}

		if (nextGuidance.atoms >= HNSW_GUIDANCE_MAX_ATOMS)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("too many guidance atoms"),
					 errhint("Maximum supported atoms: %d.", HNSW_GUIDANCE_MAX_ATOMS)));

		atomIndex = nextGuidance.atoms++;
		nextGuidance.atom[atomIndex].cache = cache;
		nextGuidance.atom[atomIndex].kind = atomKind;
		nextGuidance.atom[atomIndex].negated = negated;
		nextGuidance.atom[atomIndex].group = nextGuidance.groups - 1;
		if (negated)
			nextGuidance.negatedAtoms++;
	}

	if (nextGuidance.atoms == 0 || nextGuidance.atom[nextGuidance.atoms - 1].group != nextGuidance.groups - 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("empty HNSW guidance OR group")));

	if (hnsw_guidance_compose_exact_or)
		HnswGuidanceBuildComposedExactOr(&nextGuidance, descriptor);

	HnswMetadataCurrentCacheVersion(heapOid, &finalEpochTracked, &finalRelationEpoch,
									&finalRelationRelFileNode);
	if (finalEpochTracked != nextGuidance.epochTracked ||
		(finalEpochTracked && finalRelationEpoch != nextGuidance.relationEpoch) ||
		finalRelationRelFileNode != nextGuidance.relationRelFileNode)
		ereport(ERROR,
				(errcode(ERRCODE_T_R_SERIALIZATION_FAILURE),
				 errmsg("relation changed while HNSW guidance was being activated"),
				 errhint("Retry vector_hnsw_guidance_activate().")));

	nextGuidance.active = true;
	hnsw_active_guidance = nextGuidance;
	PG_RETURN_INT32(nextGuidance.atoms);
}

PG_FUNCTION_INFO_V1(vector_hnsw_guidance_profile);
Datum
vector_hnsw_guidance_profile(PG_FUNCTION_ARGS)
{
	StringInfoData output;

	initStringInfo(&output);
	appendStringInfo(&output,
					 "{\"active\":%s,"
					 "\"kind\":\"%s\","
						 "\"heap_oid\":%u,"
						 "\"epoch_tracked\":%s,"
						 "\"relation_epoch\":" INT64_FORMAT ","
						 "\"relation_relfilenode\":%u,"
						 "\"atoms\":%d,"
					 "\"groups\":%d,"
					 "\"negated_atoms\":%d,"
					 "\"last_cache_build_ms\":%.6f,"
					 "\"last_cache_rows\":" INT64_FORMAT ","
					 "\"last_cache_pages\":" INT64_FORMAT ","
					 "\"last_cache_memory_bytes\":" INT64_FORMAT ","
					 "\"fragment_cache_hits\":" INT64_FORMAT ","
					 "\"fragment_cache_misses\":" INT64_FORMAT ","
					 "\"fragment_store_hits\":" INT64_FORMAT ","
						 "\"fragment_builds\":" INT64_FORMAT ","
						 "\"composed_guide_hit\":%s,"
						 "\"composed_guide_hits\":" INT64_FORMAT ","
						 "\"composed_guide_misses\":" INT64_FORMAT ","
						 "\"composed_exact_active\":%s,"
						 "\"composed_exact_hit\":%s,"
						 "\"composed_exact_rows\":" INT64_FORMAT ","
						 "\"composed_exact_memory_bytes\":" INT64_FORMAT ","
						 "\"composed_exact_build_ms\":%.6f}",
					 hnsw_active_guidance.active ? "true" : "false",
						 HnswGuidanceKindName(hnsw_active_guidance.kind),
						 hnsw_active_guidance.heapOid,
						 hnsw_active_guidance.epochTracked ? "true" : "false",
						 hnsw_active_guidance.relationEpoch,
						 hnsw_active_guidance.relationRelFileNode,
						 hnsw_active_guidance.atoms,
					 hnsw_active_guidance.groups,
					 hnsw_active_guidance.negatedAtoms,
					 hnsw_active_guidance.lastBuildMs,
					 hnsw_active_guidance.lastCacheRows,
					 hnsw_active_guidance.lastCachePages,
					 hnsw_active_guidance.lastCacheMemoryBytes,
					 hnsw_active_guidance.fragmentCacheHits,
					 hnsw_active_guidance.fragmentCacheMisses,
					 hnsw_active_guidance.fragmentStoreHits,
					 hnsw_active_guidance.fragmentBuilds,
						 hnsw_active_guidance.composedGuideHit ? "true" : "false",
						 hnsw_active_guidance.composedGuideHits,
						 hnsw_active_guidance.composedGuideMisses,
						 hnsw_active_guidance.composedExactActive ? "true" : "false",
						 hnsw_active_guidance.composedExactHit ? "true" : "false",
						 hnsw_active_guidance.composedExactRows,
						 hnsw_active_guidance.composedExactMemoryBytes,
						 hnsw_active_guidance.composedExactBuildMs);

	PG_RETURN_TEXT_P(cstring_to_text(output.data));
}

PG_FUNCTION_INFO_V1(vector_hnsw_last_scan_profile);
Datum
vector_hnsw_last_scan_profile(PG_FUNCTION_ARGS)
{
	HnswScanProfile profile;
	StringInfoData output;

	HnswGetLastScanProfile(&profile);
	initStringInfo(&output);
	VectorHnswLastProfileToText(&output, &profile);
	PG_RETURN_TEXT_P(cstring_to_text(output.data));
}

PG_FUNCTION_INFO_V1(vector_hnsw_reset_scan_profile);
Datum
vector_hnsw_reset_scan_profile(PG_FUNCTION_ARGS)
{
	HnswResetScanProfile();
	PG_RETURN_VOID();
}

static int
CompareMaterializeCandidatesByPage(const void *a, const void *b)
{
	const HnswMaterializeCandidate *ca = (const HnswMaterializeCandidate *) a;
	const HnswMaterializeCandidate *cb = (const HnswMaterializeCandidate *) b;
	BlockNumber ba = ItemPointerGetBlockNumber(&ca->tid);
	BlockNumber bb = ItemPointerGetBlockNumber(&cb->tid);
	OffsetNumber oa = ItemPointerGetOffsetNumber(&ca->tid);
	OffsetNumber ob = ItemPointerGetOffsetNumber(&cb->tid);

	if (ba < bb)
		return -1;
	if (ba > bb)
		return 1;
	if (oa < ob)
		return -1;
	if (oa > ob)
		return 1;
	return 0;
}

static int
CompareMaterializeCandidatesByRank(const void *a, const void *b)
{
	const HnswMaterializeCandidate *ca = (const HnswMaterializeCandidate *) a;
	const HnswMaterializeCandidate *cb = (const HnswMaterializeCandidate *) b;

	if (ca->rank < cb->rank)
		return -1;
	if (ca->rank > cb->rank)
		return 1;
	return 0;
}

static void
ResetHnswMaterializeProfile(void)
{
	hnsw_materialize_last_profile.valid = false;
	hnsw_materialize_last_profile.candidates = 0;
	hnsw_materialize_last_profile.visible = 0;
	hnsw_materialize_last_profile.returned = 0;
	hnsw_materialize_last_profile.distanceRuns = 0;
	hnsw_materialize_last_profile.distinctPages = 0;
	hnsw_materialize_last_profile.indexMs = 0;
	hnsw_materialize_last_profile.fetchMs = 0;
}

PG_FUNCTION_INFO_V1(vector_hnsw_page_materialize_profile);
Datum
vector_hnsw_page_materialize_profile(PG_FUNCTION_ARGS)
{
	StringInfoData output;
	HnswMaterializeProfile *profile = &hnsw_materialize_last_profile;

	initStringInfo(&output);
	appendStringInfo(&output,
					 "{\"valid\":%s,"
					 "\"candidates\":" INT64_FORMAT ","
					 "\"visible\":" INT64_FORMAT ","
					 "\"returned\":" INT64_FORMAT ","
					 "\"distance_order_page_runs\":" INT64_FORMAT ","
					 "\"distinct_heap_pages\":" INT64_FORMAT ","
					 "\"index_ms\":%.6f,"
					 "\"page_fetch_ms\":%.6f}",
					 profile->valid ? "true" : "false",
					 profile->candidates,
					 profile->visible,
					 profile->returned,
					 profile->distanceRuns,
					 profile->distinctPages,
					 profile->indexMs,
					 profile->fetchMs);

	PG_RETURN_TEXT_P(cstring_to_text(output.data));
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_cache_build);
Datum
vector_hnsw_metadata_cache_build(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	text	   *filterText = PG_GETARG_TEXT_PP(1);
	char	   *filterName = text_to_cstring(filterText);
	Oid			heapOid = IndexGetRelation(indexOid, false);
	HnswMetadataCacheEntry *cache;
	bool		cacheHit;

	ResetHnswMetadataFilterProfile();
	cache = GetHnswMetadataCache(heapOid, filterName, true, &cacheHit, NULL);
	hnsw_metadata_filter_last_profile.valid = true;
	hnsw_metadata_filter_last_profile.cacheHit = cacheHit;
	hnsw_metadata_filter_last_profile.cacheKind = "exact";
	hnsw_metadata_filter_last_profile.cacheRows = cache->rows;
	hnsw_metadata_filter_last_profile.cacheMemoryBytes = HnswMetadataCacheMemoryBytes(cache, HNSW_GUIDANCE_KIND_EXACT);
	hnsw_metadata_filter_last_profile.cacheBuildMs = cacheHit ? 0 : cache->buildMs;

	PG_RETURN_INT64(cache->rows);
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_page_cache_build);
Datum
vector_hnsw_metadata_page_cache_build(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	text	   *filterText = PG_GETARG_TEXT_PP(1);
	char	   *filterName = text_to_cstring(filterText);
	Oid			heapOid = IndexGetRelation(indexOid, false);
	HnswMetadataCacheEntry *cache;
	bool		cacheHit;

	ResetHnswMetadataFilterProfile();
	cache = GetHnswMetadataPageCache(heapOid, filterName, true, &cacheHit, NULL);
	hnsw_metadata_filter_last_profile.valid = true;
	hnsw_metadata_filter_last_profile.cacheHit = cacheHit;
	hnsw_metadata_filter_last_profile.cacheKind = "page";
	hnsw_metadata_filter_last_profile.cacheRows = cache->pageRows;
	hnsw_metadata_filter_last_profile.cachePages = cache->pages;
	hnsw_metadata_filter_last_profile.cacheMemoryBytes = HnswMetadataCacheMemoryBytes(cache, HNSW_GUIDANCE_KIND_PAGE);
	hnsw_metadata_filter_last_profile.cacheBuildMs = cacheHit ? 0 : cache->pageBuildMs;

	PG_RETURN_INT64(cache->pageRows);
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_bloom_cache_build);
Datum
vector_hnsw_metadata_bloom_cache_build(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	text	   *filterText = PG_GETARG_TEXT_PP(1);
	char	   *filterName = text_to_cstring(filterText);
	Oid			heapOid = IndexGetRelation(indexOid, false);
	HnswMetadataCacheEntry *cache;
	bool		cacheHit;

	ResetHnswMetadataFilterProfile();
	cache = GetHnswMetadataBloomCache(heapOid, filterName, true, &cacheHit, NULL);
	hnsw_metadata_filter_last_profile.valid = true;
	hnsw_metadata_filter_last_profile.cacheHit = cacheHit;
	hnsw_metadata_filter_last_profile.cacheKind = "bloom";
	hnsw_metadata_filter_last_profile.cacheRows = cache->bloomRows;
	hnsw_metadata_filter_last_profile.cacheMemoryBytes = HnswMetadataCacheMemoryBytes(cache, HNSW_GUIDANCE_KIND_BLOOM);
	hnsw_metadata_filter_last_profile.cacheBuildMs = cacheHit ? 0 : cache->bloomBuildMs;

	PG_RETURN_INT64(cache->bloomRows);
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_filter_profile);
Datum
vector_hnsw_metadata_filter_profile(PG_FUNCTION_ARGS)
{
	StringInfoData output;
	HnswMetadataFilterProfile *profile = &hnsw_metadata_filter_last_profile;

	initStringInfo(&output);
	appendStringInfo(&output,
					 "{\"valid\":%s,"
					 "\"cache_hit\":%s,"
					 "\"cache_kind\":\"%s\","
					 "\"cache_rows\":" INT64_FORMAT ","
					 "\"cache_pages\":" INT64_FORMAT ","
					 "\"candidates\":" INT64_FORMAT ","
					 "\"cache_checks\":" INT64_FORMAT ","
					 "\"cache_matches\":" INT64_FORMAT ","
					 "\"returned\":" INT64_FORMAT ","
					 "\"cache_memory_bytes\":" INT64_FORMAT ","
					 "\"cache_build_ms\":%.6f,"
					 "\"search_ms\":%.6f}",
					 profile->valid ? "true" : "false",
					 profile->cacheHit ? "true" : "false",
					 profile->cacheKind != NULL ? profile->cacheKind : "none",
					 profile->cacheRows,
					 profile->cachePages,
					 profile->candidates,
					 profile->cacheChecks,
					 profile->cacheMatches,
					 profile->returned,
					 profile->cacheMemoryBytes,
					 profile->cacheBuildMs,
					 profile->searchMs);

	PG_RETURN_TEXT_P(cstring_to_text(output.data));
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_cache_profile);
Datum
vector_hnsw_metadata_cache_profile(PG_FUNCTION_ARGS)
{
	int64		entries;
	int64		residentEntries;
	int64		residentBytes;
	int64		largestEntryBytes;
	int64		descriptorEntries;
	int64		descriptorHits;
	int64		descriptorExactEntries;
	int64		descriptorExactRows;
	int64		descriptorExactBytes;
	int64		descriptorExactHits;
	int64		budgetBytes = (int64) hnsw_metadata_cache_max_mb * 1024L * 1024L;
	StringInfoData output;

	HnswMetadataCacheStats(&entries, &residentEntries, &residentBytes, &largestEntryBytes);
	HnswGuidanceDescriptorStats(&descriptorEntries, &descriptorHits, &descriptorExactEntries, &descriptorExactRows, &descriptorExactBytes, &descriptorExactHits);

	initStringInfo(&output);
	appendStringInfo(&output,
					 "{\"entries\":" INT64_FORMAT ","
					 "\"resident_entries\":" INT64_FORMAT ","
					 "\"resident_bytes\":" INT64_FORMAT ","
					 "\"largest_entry_bytes\":" INT64_FORMAT ","
						 "\"evictions\":" INT64_FORMAT ","
						 "\"composed_guide_entries\":" INT64_FORMAT ","
						 "\"composed_guide_hits\":" INT64_FORMAT ","
						 "\"composed_exact_entries\":" INT64_FORMAT ","
						 "\"composed_exact_rows\":" INT64_FORMAT ","
						 "\"composed_exact_bytes\":" INT64_FORMAT ","
						 "\"composed_exact_hits\":" INT64_FORMAT ","
						 "\"budget_mb\":%d,"
						 "\"budget_bytes\":" INT64_FORMAT "}",
					 entries,
					 residentEntries,
					 residentBytes,
					 largestEntryBytes,
						 hnsw_metadata_cache_evictions,
						 descriptorEntries,
						 descriptorHits,
						 descriptorExactEntries,
						 descriptorExactRows,
						 descriptorExactBytes,
						 descriptorExactHits,
						 hnsw_metadata_cache_max_mb,
						 budgetBytes);

	PG_RETURN_TEXT_P(cstring_to_text(output.data));
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_cache_reset);
Datum
vector_hnsw_metadata_cache_reset(PG_FUNCTION_ARGS)
{
	HnswMetadataResetCaches();
	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_filter_search);
Datum
vector_hnsw_metadata_filter_search(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	Oid			indexOid = PG_GETARG_OID(0);
	Datum		query = PG_GETARG_DATUM(1);
	int32		k = PG_GETARG_INT32(2);
	int32		candidateLimit = PG_GETARG_INT32(3);
	text	   *filterText = PG_GETARG_TEXT_PP(4);
	char	   *filterName = text_to_cstring(filterText);
	Relation	indexRel;
	Relation	heapRel;
	Oid			heapOid;
	ScanKeyData orderby;
	IndexScanDesc scan;
	Snapshot	snapshot;
	HnswMetadataCacheEntry *cache;
	bool		cacheHit;
	int			oldPageAccess;
	int			candidates = 0;
	int			returned = 0;
	instr_time	start;
	instr_time	elapsed;

	if (k < 1)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("k must be at least 1")));
	if (candidateLimit < k)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("candidate limit must be at least k")));

	InitMaterializedSRF(fcinfo, MAT_SRF_USE_EXPECTED_DESC | MAT_SRF_BLESS);
	tupdesc = rsinfo->setDesc;
	tupstore = rsinfo->setResult;

	ResetHnswMetadataFilterProfile();

	indexRel = index_open(indexOid, AccessShareLock);
	heapOid = IndexGetRelation(indexOid, false);
	heapRel = table_open(heapOid, AccessShareLock);
	cache = GetHnswMetadataCache(heapOid, filterName, true, &cacheHit, NULL);

	hnsw_metadata_filter_last_profile.cacheHit = cacheHit;
	hnsw_metadata_filter_last_profile.cacheRows = cache->rows;
	hnsw_metadata_filter_last_profile.cacheMemoryBytes = HnswMetadataCacheMemoryBytes(cache, HNSW_GUIDANCE_KIND_EXACT);
	hnsw_metadata_filter_last_profile.cacheBuildMs = cacheHit ? 0 : cache->buildMs;

	snapshot = GetActiveSnapshot();
	oldPageAccess = hnsw_page_access;
	hnsw_page_access = HNSW_PAGE_ACCESS_OFF;

	MemSet(&orderby, 0, sizeof(ScanKeyData));
	orderby.sk_attno = 1;
	orderby.sk_strategy = InvalidStrategy;
	orderby.sk_subtype = InvalidOid;
	orderby.sk_collation = InvalidOid;
	orderby.sk_argument = query;
	scan = index_beginscan(heapRel, indexRel, snapshot, 0, 1);

	INSTR_TIME_SET_CURRENT(start);
	index_rescan(scan, NULL, 0, &orderby, 1);
	while (candidates < candidateLimit && returned < k)
	{
		ItemPointer tid = index_getnext_tid(scan, ForwardScanDirection);
		HnswMetadataTidKey tidKey;
		HnswMetadataTidEntry *entry;
		Datum		values[3];
		bool		nulls[3] = {false, false, false};
		char		ctidbuf[64];

		if (tid == NULL)
			break;

		candidates++;
		hnsw_metadata_filter_last_profile.cacheChecks++;
		tidKey.tid = *tid;
		entry = (HnswMetadataTidEntry *) hash_search(cache->tidHash, &tidKey, HASH_FIND, NULL);
		if (entry == NULL)
			continue;

		hnsw_metadata_filter_last_profile.cacheMatches++;
		snprintf(ctidbuf, sizeof(ctidbuf), "(%u,%u)",
				 ItemPointerGetBlockNumber(tid),
				 ItemPointerGetOffsetNumber(tid));
		values[0] = Int32GetDatum(candidates);
		values[1] = Int64GetDatum(entry->id);
		values[2] = CStringGetTextDatum(ctidbuf);
		tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		returned++;
	}
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);

	index_endscan(scan);
	hnsw_page_access = oldPageAccess;
	table_close(heapRel, AccessShareLock);
	index_close(indexRel, AccessShareLock);

	hnsw_metadata_filter_last_profile.valid = true;
	hnsw_metadata_filter_last_profile.candidates = candidates;
	hnsw_metadata_filter_last_profile.returned = returned;
	hnsw_metadata_filter_last_profile.searchMs = INSTR_TIME_GET_MILLISEC(elapsed);

	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_page_filter_candidates);
Datum
vector_hnsw_metadata_page_filter_candidates(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	Oid			indexOid = PG_GETARG_OID(0);
	Datum		query = PG_GETARG_DATUM(1);
	int32		candidateLimit = PG_GETARG_INT32(2);
	text	   *filterText = PG_GETARG_TEXT_PP(3);
	char	   *filterName = text_to_cstring(filterText);
	Relation	indexRel;
	Relation	heapRel;
	Oid			heapOid;
	ScanKeyData orderby;
	IndexScanDesc scan;
	Snapshot	snapshot;
	HnswMetadataCacheEntry *cache;
	bool		cacheHit;
	int			oldPageAccess;
	int			candidates = 0;
	int			returned = 0;
	instr_time	start;
	instr_time	elapsed;

	if (candidateLimit < 1)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("candidate limit must be at least 1")));

	InitMaterializedSRF(fcinfo, MAT_SRF_USE_EXPECTED_DESC | MAT_SRF_BLESS);
	tupdesc = rsinfo->setDesc;
	tupstore = rsinfo->setResult;

	ResetHnswMetadataFilterProfile();

	indexRel = index_open(indexOid, AccessShareLock);
	heapOid = IndexGetRelation(indexOid, false);
	heapRel = table_open(heapOid, AccessShareLock);
	cache = GetHnswMetadataPageCache(heapOid, filterName, true, &cacheHit, NULL);

	hnsw_metadata_filter_last_profile.cacheHit = cacheHit;
	hnsw_metadata_filter_last_profile.cacheKind = "page";
	hnsw_metadata_filter_last_profile.cacheRows = cache->pageRows;
	hnsw_metadata_filter_last_profile.cachePages = cache->pages;
	hnsw_metadata_filter_last_profile.cacheMemoryBytes = HnswMetadataCacheMemoryBytes(cache, HNSW_GUIDANCE_KIND_PAGE);
	hnsw_metadata_filter_last_profile.cacheBuildMs = cacheHit ? 0 : cache->pageBuildMs;

	snapshot = GetActiveSnapshot();
	oldPageAccess = hnsw_page_access;
	hnsw_page_access = HNSW_PAGE_ACCESS_OFF;

	MemSet(&orderby, 0, sizeof(ScanKeyData));
	orderby.sk_attno = 1;
	orderby.sk_strategy = InvalidStrategy;
	orderby.sk_subtype = InvalidOid;
	orderby.sk_collation = InvalidOid;
	orderby.sk_argument = query;
	scan = index_beginscan(heapRel, indexRel, snapshot, 0, 1);

	INSTR_TIME_SET_CURRENT(start);
	index_rescan(scan, NULL, 0, &orderby, 1);
	while (candidates < candidateLimit)
	{
		ItemPointer tid = index_getnext_tid(scan, ForwardScanDirection);
		BlockNumber block;
		Datum		values[2];
		bool		nulls[2] = {false, false};
		char		ctidbuf[64];

		if (tid == NULL)
			break;

		candidates++;
		hnsw_metadata_filter_last_profile.cacheChecks++;
		block = ItemPointerGetBlockNumber(tid);
		if (!HnswMetadataPageBitTest(cache, block))
			continue;

		hnsw_metadata_filter_last_profile.cacheMatches++;
		snprintf(ctidbuf, sizeof(ctidbuf), "(%u,%u)",
				 ItemPointerGetBlockNumber(tid),
				 ItemPointerGetOffsetNumber(tid));
		values[0] = Int32GetDatum(candidates);
		values[1] = CStringGetTextDatum(ctidbuf);
		tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		returned++;
	}
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);

	index_endscan(scan);
	hnsw_page_access = oldPageAccess;
	table_close(heapRel, AccessShareLock);
	index_close(indexRel, AccessShareLock);

	hnsw_metadata_filter_last_profile.valid = true;
	hnsw_metadata_filter_last_profile.candidates = candidates;
	hnsw_metadata_filter_last_profile.returned = returned;
	hnsw_metadata_filter_last_profile.searchMs = INSTR_TIME_GET_MILLISEC(elapsed);

	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_bloom_filter_candidates);
Datum
vector_hnsw_metadata_bloom_filter_candidates(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	Oid			indexOid = PG_GETARG_OID(0);
	Datum		query = PG_GETARG_DATUM(1);
	int32		candidateLimit = PG_GETARG_INT32(2);
	text	   *filterText = PG_GETARG_TEXT_PP(3);
	char	   *filterName = text_to_cstring(filterText);
	Relation	indexRel;
	Relation	heapRel;
	Oid			heapOid;
	ScanKeyData orderby;
	IndexScanDesc scan;
	Snapshot	snapshot;
	HnswMetadataCacheEntry *cache;
	bool		cacheHit;
	int			oldPageAccess;
	int			candidates = 0;
	int			returned = 0;
	instr_time	start;
	instr_time	elapsed;

	if (candidateLimit < 1)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("candidate limit must be at least 1")));

	InitMaterializedSRF(fcinfo, MAT_SRF_USE_EXPECTED_DESC | MAT_SRF_BLESS);
	tupdesc = rsinfo->setDesc;
	tupstore = rsinfo->setResult;

	ResetHnswMetadataFilterProfile();

	indexRel = index_open(indexOid, AccessShareLock);
	heapOid = IndexGetRelation(indexOid, false);
	heapRel = table_open(heapOid, AccessShareLock);
	cache = GetHnswMetadataBloomCache(heapOid, filterName, true, &cacheHit, NULL);

	hnsw_metadata_filter_last_profile.cacheHit = cacheHit;
	hnsw_metadata_filter_last_profile.cacheKind = "bloom";
	hnsw_metadata_filter_last_profile.cacheRows = cache->bloomRows;
	hnsw_metadata_filter_last_profile.cacheMemoryBytes = HnswMetadataCacheMemoryBytes(cache, HNSW_GUIDANCE_KIND_BLOOM);
	hnsw_metadata_filter_last_profile.cacheBuildMs = cacheHit ? 0 : cache->bloomBuildMs;

	snapshot = GetActiveSnapshot();
	oldPageAccess = hnsw_page_access;
	hnsw_page_access = HNSW_PAGE_ACCESS_OFF;

	MemSet(&orderby, 0, sizeof(ScanKeyData));
	orderby.sk_attno = 1;
	orderby.sk_strategy = InvalidStrategy;
	orderby.sk_subtype = InvalidOid;
	orderby.sk_collation = InvalidOid;
	orderby.sk_argument = query;
	scan = index_beginscan(heapRel, indexRel, snapshot, 0, 1);

	INSTR_TIME_SET_CURRENT(start);
	index_rescan(scan, NULL, 0, &orderby, 1);
	while (candidates < candidateLimit)
	{
		ItemPointer tid = index_getnext_tid(scan, ForwardScanDirection);
		Datum		values[2];
		bool		nulls[2] = {false, false};
		char		ctidbuf[64];

		if (tid == NULL)
			break;

		candidates++;
		hnsw_metadata_filter_last_profile.cacheChecks++;
		if (!HnswMetadataBloomMayContain(cache, tid))
			continue;

		hnsw_metadata_filter_last_profile.cacheMatches++;
		snprintf(ctidbuf, sizeof(ctidbuf), "(%u,%u)",
				 ItemPointerGetBlockNumber(tid),
				 ItemPointerGetOffsetNumber(tid));
		values[0] = Int32GetDatum(candidates);
		values[1] = CStringGetTextDatum(ctidbuf);
		tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		returned++;
	}
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);

	index_endscan(scan);
	hnsw_page_access = oldPageAccess;
	table_close(heapRel, AccessShareLock);
	index_close(indexRel, AccessShareLock);

	hnsw_metadata_filter_last_profile.valid = true;
	hnsw_metadata_filter_last_profile.candidates = candidates;
	hnsw_metadata_filter_last_profile.returned = returned;
	hnsw_metadata_filter_last_profile.searchMs = INSTR_TIME_GET_MILLISEC(elapsed);

	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(vector_hnsw_metadata_bloom_filter_candidates_limited);
Datum
vector_hnsw_metadata_bloom_filter_candidates_limited(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	Oid			indexOid = PG_GETARG_OID(0);
	Datum		query = PG_GETARG_DATUM(1);
	int32		candidateLimit = PG_GETARG_INT32(2);
	int32		matchLimit = PG_GETARG_INT32(3);
	text	   *filterText = PG_GETARG_TEXT_PP(4);
	char	   *filterName = text_to_cstring(filterText);
	Relation	indexRel;
	Relation	heapRel;
	Oid			heapOid;
	ScanKeyData orderby;
	IndexScanDesc scan;
	Snapshot	snapshot;
	HnswMetadataCacheEntry *cache;
	bool		cacheHit;
	int			oldPageAccess;
	int			candidates = 0;
	int			returned = 0;
	instr_time	start;
	instr_time	elapsed;

	if (candidateLimit < 1)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("candidate limit must be at least 1")));
	if (matchLimit < 1)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("match limit must be at least 1")));

	InitMaterializedSRF(fcinfo, MAT_SRF_USE_EXPECTED_DESC | MAT_SRF_BLESS);
	tupdesc = rsinfo->setDesc;
	tupstore = rsinfo->setResult;

	ResetHnswMetadataFilterProfile();

	indexRel = index_open(indexOid, AccessShareLock);
	heapOid = IndexGetRelation(indexOid, false);
	heapRel = table_open(heapOid, AccessShareLock);
	cache = GetHnswMetadataBloomCache(heapOid, filterName, true, &cacheHit, NULL);

	hnsw_metadata_filter_last_profile.cacheHit = cacheHit;
	hnsw_metadata_filter_last_profile.cacheKind = "bloom_limited";
	hnsw_metadata_filter_last_profile.cacheRows = cache->bloomRows;
	hnsw_metadata_filter_last_profile.cacheMemoryBytes = HnswMetadataCacheMemoryBytes(cache, HNSW_GUIDANCE_KIND_BLOOM);
	hnsw_metadata_filter_last_profile.cacheBuildMs = cacheHit ? 0 : cache->bloomBuildMs;

	snapshot = GetActiveSnapshot();
	oldPageAccess = hnsw_page_access;
	hnsw_page_access = HNSW_PAGE_ACCESS_OFF;

	MemSet(&orderby, 0, sizeof(ScanKeyData));
	orderby.sk_attno = 1;
	orderby.sk_strategy = InvalidStrategy;
	orderby.sk_subtype = InvalidOid;
	orderby.sk_collation = InvalidOid;
	orderby.sk_argument = query;
	scan = index_beginscan(heapRel, indexRel, snapshot, 0, 1);

	INSTR_TIME_SET_CURRENT(start);
	index_rescan(scan, NULL, 0, &orderby, 1);
	while (candidates < candidateLimit && returned < matchLimit)
	{
		ItemPointer tid = index_getnext_tid(scan, ForwardScanDirection);
		Datum		values[2];
		bool		nulls[2] = {false, false};
		char		ctidbuf[64];

		if (tid == NULL)
			break;

		candidates++;
		hnsw_metadata_filter_last_profile.cacheChecks++;
		if (!HnswMetadataBloomMayContain(cache, tid))
			continue;

		hnsw_metadata_filter_last_profile.cacheMatches++;
		snprintf(ctidbuf, sizeof(ctidbuf), "(%u,%u)",
				 ItemPointerGetBlockNumber(tid),
				 ItemPointerGetOffsetNumber(tid));
		values[0] = Int32GetDatum(candidates);
		values[1] = CStringGetTextDatum(ctidbuf);
		tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		returned++;
	}
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);

	index_endscan(scan);
	hnsw_page_access = oldPageAccess;
	table_close(heapRel, AccessShareLock);
	index_close(indexRel, AccessShareLock);

	hnsw_metadata_filter_last_profile.valid = true;
	hnsw_metadata_filter_last_profile.candidates = candidates;
	hnsw_metadata_filter_last_profile.returned = returned;
	hnsw_metadata_filter_last_profile.searchMs = INSTR_TIME_GET_MILLISEC(elapsed);

	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(vector_hnsw_page_materialize);
Datum
vector_hnsw_page_materialize(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Datum		query = PG_GETARG_DATUM(1);
	int32		k = PG_GETARG_INT32(2);
	int32		candidateLimit = PG_GETARG_INT32(3);
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	Relation	indexRel;
	Relation	heapRel;
	Oid			heapOid;
	AttrNumber	idAttnum;
	ScanKeyData orderby;
	IndexScanDesc scan;
	Snapshot	snapshot;
	HnswMaterializeCandidate *items;
	int			count = 0;
	int			returned = 0;
	BlockNumber previousBlock = InvalidBlockNumber;
	BlockNumber previousPageBlock = InvalidBlockNumber;
	TupleTableSlot *slot;
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	int			oldPageAccess;
	instr_time	start;
	instr_time	elapsed;

	if (k < 1)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("k must be at least 1")));
	if (candidateLimit < k)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("candidate limit must be at least k")));

	InitMaterializedSRF(fcinfo, MAT_SRF_USE_EXPECTED_DESC | MAT_SRF_BLESS);
	tupdesc = rsinfo->setDesc;
	tupstore = rsinfo->setResult;

	ResetHnswMaterializeProfile();

	indexRel = index_open(indexOid, AccessShareLock);
	heapOid = IndexGetRelation(indexOid, false);
	heapRel = table_open(heapOid, AccessShareLock);
	idAttnum = get_attnum(heapOid, "id");
	if (idAttnum == InvalidAttrNumber)
		ereport(ERROR, (errcode(ERRCODE_UNDEFINED_COLUMN),
						errmsg("heap relation \"%s\" does not have an id column",
							   RelationGetRelationName(heapRel))));

	items = (HnswMaterializeCandidate *) palloc0(sizeof(HnswMaterializeCandidate) * candidateLimit);
	snapshot = GetActiveSnapshot();

	/*
	 * Collect candidate TIDs from the HNSW AM without letting the regular
	 * executor fetch heap tuples one-by-one. Disable HNSW page access here so
	 * the collected rank order is the standard distance order.
	 */
	oldPageAccess = hnsw_page_access;
	hnsw_page_access = HNSW_PAGE_ACCESS_OFF;

	MemSet(&orderby, 0, sizeof(ScanKeyData));
	orderby.sk_attno = 1;
	orderby.sk_strategy = InvalidStrategy;
	orderby.sk_subtype = InvalidOid;
	orderby.sk_collation = InvalidOid;
	orderby.sk_argument = query;
	scan = index_beginscan(heapRel, indexRel, snapshot, 0, 1);

	INSTR_TIME_SET_CURRENT(start);
	index_rescan(scan, NULL, 0, &orderby, 1);
	while (count < candidateLimit)
	{
		ItemPointer tid = index_getnext_tid(scan, ForwardScanDirection);
		BlockNumber block;

		if (tid == NULL)
			break;

		items[count].tid = *tid;
		items[count].rank = count;
		block = ItemPointerGetBlockNumber(&items[count].tid);
		if (count == 0 || block != previousBlock)
			hnsw_materialize_last_profile.distanceRuns++;
		previousBlock = block;
		count++;
	}
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	hnsw_materialize_last_profile.indexMs = INSTR_TIME_GET_MILLISEC(elapsed);

	index_endscan(scan);
	hnsw_page_access = oldPageAccess;

	qsort(items, count, sizeof(HnswMaterializeCandidate), CompareMaterializeCandidatesByPage);
	slot = table_slot_create(heapRel, NULL);

	INSTR_TIME_SET_CURRENT(start);
	for (int i = 0; i < count; i++)
	{
		BlockNumber block = ItemPointerGetBlockNumber(&items[i].tid);

		if (i == 0 || block != previousPageBlock)
			hnsw_materialize_last_profile.distinctPages++;
		previousPageBlock = block;

		if (table_tuple_fetch_row_version(heapRel, &items[i].tid, snapshot, slot))
		{
			bool		isnull;
			Datum		idDatum = slot_getattr(slot, idAttnum, &isnull);

			if (!isnull)
			{
				items[i].id = DatumGetInt64(idDatum);
				items[i].visible = true;
				hnsw_materialize_last_profile.visible++;
			}

			ExecClearTuple(slot);
		}
	}
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	hnsw_materialize_last_profile.fetchMs = INSTR_TIME_GET_MILLISEC(elapsed);
	ExecDropSingleTupleTableSlot(slot);

	qsort(items, count, sizeof(HnswMaterializeCandidate), CompareMaterializeCandidatesByRank);
	for (int i = 0; i < count && returned < k; i++)
	{
		Datum		values[3];
		bool		nulls[3] = {false, false, false};
		char		ctidbuf[64];

		if (!items[i].visible)
			continue;

		snprintf(ctidbuf, sizeof(ctidbuf), "(%u,%u)",
				 ItemPointerGetBlockNumber(&items[i].tid),
				 ItemPointerGetOffsetNumber(&items[i].tid));
		values[0] = Int32GetDatum(items[i].rank + 1);
		values[1] = Int64GetDatum(items[i].id);
		values[2] = CStringGetTextDatum(ctidbuf);
		tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		returned++;
	}

	hnsw_materialize_last_profile.valid = true;
	hnsw_materialize_last_profile.candidates = count;
	hnsw_materialize_last_profile.returned = returned;

	table_close(heapRel, AccessShareLock);
	index_close(indexRel, AccessShareLock);

	PG_RETURN_VOID();
}

/*
 * Ensure same dimensions
 */
static inline void
CheckDims(Vector * a, Vector * b)
{
	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", a->dim, b->dim)));
}

/*
 * Ensure expected dimensions
 */
static inline void
CheckExpectedDim(int32 typmod, int dim)
{
	if (typmod != -1 && typmod != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected %d dimensions, not %d", typmod, dim)));
}

/*
 * Ensure valid dimensions
 */
static inline void
CheckDim(int dim)
{
	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector must have at least 1 dimension")));

	if (dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("vector cannot have more than %d dimensions", VECTOR_MAX_DIM)));
}

/*
 * Ensure finite element
 */
static inline void
CheckElement(float value)
{
	if (isnan(value))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("NaN not allowed in vector")));

	if (isinf(value))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("infinite value not allowed in vector")));
}

/*
 * Allocate and initialize a new vector
 */
Vector *
InitVector(int dim)
{
	Vector	   *result;
	int			size;

	size = VECTOR_SIZE(dim);
	result = (Vector *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = dim;

	return result;
}

#if PG_VERSION_NUM >= 170000
#define vector_isspace(ch) scanner_isspace(ch)
#else
static inline bool
vector_isspace(char ch)
{
	if (ch == ' ' ||
		ch == '\t' ||
		ch == '\n' ||
		ch == '\r' ||
		ch == '\v' ||
		ch == '\f')
		return true;
	return false;
}
#endif

/*
 * Check state array
 */
static float8 *
CheckStateArray(ArrayType *statearray, const char *caller)
{
	if (ARR_NDIM(statearray) != 1 ||
		ARR_DIMS(statearray)[0] < 1 ||
		ARR_HASNULL(statearray) ||
		ARR_ELEMTYPE(statearray) != FLOAT8OID)
		elog(ERROR, "%s: expected state array", caller);
	return (float8 *) ARR_DATA_PTR(statearray);
}

/*
 * Convert textual representation to internal representation
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_in);
Datum
vector_in(PG_FUNCTION_ARGS)
{
	char	   *lit = PG_GETARG_CSTRING(0);
	int32		typmod = PG_GETARG_INT32(2);
	float		x[VECTOR_MAX_DIM];
	int			dim = 0;
	char	   *pt = lit;
	Vector	   *result;

	while (vector_isspace(*pt))
		pt++;

	if (*pt != '[')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for type vector: \"%s\"", lit),
				 errdetail("Vector contents must start with \"[\".")));

	pt++;

	while (vector_isspace(*pt))
		pt++;

	if (*pt == ']')
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector must have at least 1 dimension")));

	for (;;)
	{
		float		val;
		char	   *stringEnd;

		if (dim == VECTOR_MAX_DIM)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("vector cannot have more than %d dimensions", VECTOR_MAX_DIM)));

		while (vector_isspace(*pt))
			pt++;

		/* Check for empty string like float4in */
		if (*pt == '\0')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type vector: \"%s\"", lit)));

		errno = 0;

		/* Use strtof like float4in to avoid a double-rounding problem */
		/* Postgres sets LC_NUMERIC to C on startup */
		val = strtof(pt, &stringEnd);

		if (stringEnd == pt)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type vector: \"%s\"", lit)));

		/* Check for range error like float4in */
		if (errno == ERANGE && isinf(val))
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("\"%s\" is out of range for type vector", pnstrdup(pt, stringEnd - pt))));

		CheckElement(val);
		x[dim++] = val;

		pt = stringEnd;

		while (vector_isspace(*pt))
			pt++;

		if (*pt == ',')
			pt++;
		else if (*pt == ']')
		{
			pt++;
			break;
		}
		else
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type vector: \"%s\"", lit)));
	}

	/* Only whitespace is allowed after the closing brace */
	while (vector_isspace(*pt))
		pt++;

	if (*pt != '\0')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for type vector: \"%s\"", lit),
				 errdetail("Junk after closing right brace.")));

	CheckDim(dim);
	CheckExpectedDim(typmod, dim);

	result = InitVector(dim);
	for (int i = 0; i < dim; i++)
		result->x[i] = x[i];

	PG_RETURN_POINTER(result);
}

#define AppendChar(ptr, c) (*(ptr)++ = (c))
#define AppendFloat(ptr, f) ((ptr) += float_to_shortest_decimal_bufn((f), (ptr)))

/*
 * Convert internal representation to textual representation
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_out);
Datum
vector_out(PG_FUNCTION_ARGS)
{
	Vector	   *vector = PG_GETARG_VECTOR_P(0);
	int			dim = vector->dim;
	char	   *buf;
	char	   *ptr;

	/*
	 * Need:
	 *
	 * dim * (FLOAT_SHORTEST_DECIMAL_LEN - 1) bytes for
	 * float_to_shortest_decimal_bufn
	 *
	 * dim - 1 bytes for separator
	 *
	 * 3 bytes for [, ], and \0
	 */
	buf = (char *) palloc(FLOAT_SHORTEST_DECIMAL_LEN * dim + 2);
	ptr = buf;

	AppendChar(ptr, '[');

	for (int i = 0; i < dim; i++)
	{
		if (i > 0)
			AppendChar(ptr, ',');

		AppendFloat(ptr, vector->x[i]);
	}

	AppendChar(ptr, ']');
	*ptr = '\0';

	PG_FREE_IF_COPY(vector, 0);
	PG_RETURN_CSTRING(buf);
}

/*
 * Print vector - useful for debugging
 */
void
PrintVector(char *msg, Vector * vector)
{
	char	   *out = DatumGetPointer(DirectFunctionCall1(vector_out, PointerGetDatum(vector)));

	elog(INFO, "%s = %s", msg, out);
	pfree(out);
}

/*
 * Convert type modifier
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_typmod_in);
Datum
vector_typmod_in(PG_FUNCTION_ARGS)
{
	ArrayType  *ta = PG_GETARG_ARRAYTYPE_P(0);
	int32	   *tl;
	int			n;

	tl = ArrayGetIntegerTypmods(ta, &n);

	if (n != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid type modifier")));

	if (*tl < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dimensions for type vector must be at least 1")));

	if (*tl > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dimensions for type vector cannot exceed %d", VECTOR_MAX_DIM)));

	PG_RETURN_INT32(*tl);
}

/*
 * Convert external binary representation to internal representation
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_recv);
Datum
vector_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);
	int32		typmod = PG_GETARG_INT32(2);
	Vector	   *result;
	int16		dim;
	int16		unused;

	dim = pq_getmsgint(buf, sizeof(int16));
	unused = pq_getmsgint(buf, sizeof(int16));

	CheckDim(dim);
	CheckExpectedDim(typmod, dim);

	if (unused != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected unused to be 0, not %d", unused)));

	result = InitVector(dim);
	for (int i = 0; i < dim; i++)
	{
		result->x[i] = pq_getmsgfloat4(buf);
		CheckElement(result->x[i]);
	}

	PG_RETURN_POINTER(result);
}

/*
 * Convert internal representation to the external binary representation
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_send);
Datum
vector_send(PG_FUNCTION_ARGS)
{
	Vector	   *vec = PG_GETARG_VECTOR_P(0);
	StringInfoData buf;

	pq_begintypsend(&buf);
	pq_sendint(&buf, vec->dim, sizeof(int16));
	pq_sendint(&buf, vec->unused, sizeof(int16));
	for (int i = 0; i < vec->dim; i++)
		pq_sendfloat4(&buf, vec->x[i]);

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*
 * Convert vector to vector
 * This is needed to check the type modifier
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector);
Datum
vector(PG_FUNCTION_ARGS)
{
	Vector	   *vec = PG_GETARG_VECTOR_P(0);
	int32		typmod = PG_GETARG_INT32(1);

	CheckExpectedDim(typmod, vec->dim);

	PG_RETURN_POINTER(vec);
}

/*
 * Convert array to vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(array_to_vector);
Datum
array_to_vector(PG_FUNCTION_ARGS)
{
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
	int32		typmod = PG_GETARG_INT32(1);
	Vector	   *result;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	Datum	   *elemsp;
	int			nelemsp;

	if (ARR_NDIM(array) > 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("array must be 1-D")));

	if (ARR_HASNULL(array) && array_contains_nulls(array))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not contain nulls")));

	get_typlenbyvalalign(ARR_ELEMTYPE(array), &typlen, &typbyval, &typalign);
	deconstruct_array(array, ARR_ELEMTYPE(array), typlen, typbyval, typalign, &elemsp, NULL, &nelemsp);

	CheckDim(nelemsp);
	CheckExpectedDim(typmod, nelemsp);

	result = InitVector(nelemsp);

	if (ARR_ELEMTYPE(array) == INT4OID)
	{
		for (int i = 0; i < nelemsp; i++)
			result->x[i] = DatumGetInt32(elemsp[i]);
	}
	else if (ARR_ELEMTYPE(array) == FLOAT8OID)
	{
		for (int i = 0; i < nelemsp; i++)
			result->x[i] = DatumGetFloat8(elemsp[i]);
	}
	else if (ARR_ELEMTYPE(array) == FLOAT4OID)
	{
		for (int i = 0; i < nelemsp; i++)
			result->x[i] = DatumGetFloat4(elemsp[i]);
	}
	else if (ARR_ELEMTYPE(array) == NUMERICOID)
	{
		for (int i = 0; i < nelemsp; i++)
			result->x[i] = DatumGetFloat4(DirectFunctionCall1(numeric_float4, elemsp[i]));
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("unsupported array type")));
	}

	/*
	 * Free allocation from deconstruct_array. Do not free individual elements
	 * when pass-by-reference since they point to original array.
	 */
	pfree(elemsp);

	/* Check elements */
	for (int i = 0; i < result->dim; i++)
		CheckElement(result->x[i]);

	PG_RETURN_POINTER(result);
}

/*
 * Convert vector to float4[]
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_to_float4);
Datum
vector_to_float4(PG_FUNCTION_ARGS)
{
	Vector	   *vec = PG_GETARG_VECTOR_P(0);
	Datum	   *datums;
	ArrayType  *result;

	datums = (Datum *) palloc(sizeof(Datum) * vec->dim);

	for (int i = 0; i < vec->dim; i++)
		datums[i] = Float4GetDatum(vec->x[i]);

	/* Use TYPALIGN_INT for float4 */
	result = construct_array(datums, vec->dim, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT);

	pfree(datums);

	PG_RETURN_POINTER(result);
}

/*
 * Convert half vector to vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(halfvec_to_vector);
Datum
halfvec_to_vector(PG_FUNCTION_ARGS)
{
	HalfVector *vec = PG_GETARG_HALFVEC_P(0);
	int32		typmod = PG_GETARG_INT32(1);
	Vector	   *result;

	CheckDim(vec->dim);
	CheckExpectedDim(typmod, vec->dim);

	result = InitVector(vec->dim);

	for (int i = 0; i < vec->dim; i++)
		result->x[i] = HalfToFloat4(vec->x[i]);

	PG_RETURN_POINTER(result);
}

VECTOR_TARGET_CLONES static float
VectorL2SquaredDistance(int dim, float *ax, float *bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		float		diff = ax[i] - bx[i];

		distance += diff * diff;
	}

	return distance;
}

/*
 * Get the L2 distance between vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(l2_distance);
Datum
l2_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8(sqrt((double) VectorL2SquaredDistance(a->dim, a->x, b->x)));
}

/*
 * Get the L2 squared distance between vectors
 * This saves a sqrt calculation
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_l2_squared_distance);
Datum
vector_l2_squared_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8((double) VectorL2SquaredDistance(a->dim, a->x, b->x));
}

VECTOR_TARGET_CLONES static float
VectorInnerProduct(int dim, float *ax, float *bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
		distance += ax[i] * bx[i];

	return distance;
}

/*
 * Get the inner product of two vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(inner_product);
Datum
inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8((double) VectorInnerProduct(a->dim, a->x, b->x));
}

/*
 * Get the negative inner product of two vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_negative_inner_product);
Datum
vector_negative_inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8((double) -VectorInnerProduct(a->dim, a->x, b->x));
}

VECTOR_TARGET_CLONES static double
VectorCosineSimilarity(int dim, float *ax, float *bx)
{
	float		similarity = 0.0;
	float		norma = 0.0;
	float		normb = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		similarity += ax[i] * bx[i];
		norma += ax[i] * ax[i];
		normb += bx[i] * bx[i];
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b) */
	return (double) similarity / sqrt((double) norma * (double) normb);
}

/*
 * Get the cosine distance between two vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(cosine_distance);
Datum
cosine_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	double		similarity;

	CheckDims(a, b);

	similarity = VectorCosineSimilarity(a->dim, a->x, b->x);

#ifdef _MSC_VER
	/* /fp:fast may not propagate NaN */
	if (isnan(similarity))
		PG_RETURN_FLOAT8(NAN);
#endif

	/* Keep in range */
	if (similarity > 1)
		similarity = 1.0;
	else if (similarity < -1)
		similarity = -1.0;

	PG_RETURN_FLOAT8(1.0 - similarity);
}

/*
 * Get the distance for spherical k-means
 * Currently uses angular distance since needs to satisfy triangle inequality
 * Assumes inputs are unit vectors (skips norm)
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_spherical_distance);
Datum
vector_spherical_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	double		distance;

	CheckDims(a, b);

	distance = (double) VectorInnerProduct(a->dim, a->x, b->x);

	/* Prevent NaN with acos with loss of precision */
	if (distance > 1)
		distance = 1;
	else if (distance < -1)
		distance = -1;

	PG_RETURN_FLOAT8(acos(distance) / M_PI);
}

/* Does not require FMA, but keep logic simple */
VECTOR_TARGET_CLONES static float
VectorL1Distance(int dim, float *ax, float *bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
		distance += fabsf(ax[i] - bx[i]);

	return distance;
}

/*
 * Get the L1 distance between two vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(l1_distance);
Datum
l1_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	CheckDims(a, b);

	PG_RETURN_FLOAT8((double) VectorL1Distance(a->dim, a->x, b->x));
}

/*
 * Get the dimensions of a vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_dims);
Datum
vector_dims(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);

	PG_RETURN_INT32(a->dim);
}

/*
 * Get the L2 norm of a vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_norm);
Datum
vector_norm(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	float	   *ax = a->x;
	double		norm = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < a->dim; i++)
		norm += (double) ax[i] * (double) ax[i];

	PG_RETURN_FLOAT8(sqrt(norm));
}

/*
 * Normalize a vector with the L2 norm
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(l2_normalize);
Datum
l2_normalize(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	float	   *ax = a->x;
	double		norm = 0;
	Vector	   *result;
	float	   *rx;

	result = InitVector(a->dim);
	rx = result->x;

	/* Auto-vectorized */
	for (int i = 0; i < a->dim; i++)
		norm += (double) ax[i] * (double) ax[i];

	norm = sqrt(norm);

	/* Return zero vector for zero norm */
	if (norm > 0)
	{
		for (int i = 0; i < a->dim; i++)
			rx[i] = ax[i] / norm;

		/* Check for overflow */
		for (int i = 0; i < a->dim; i++)
		{
			if (isinf(rx[i]))
				float_overflow_error();
		}
	}

	PG_RETURN_POINTER(result);
}

/*
 * Add vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_add);
Datum
vector_add(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	float	   *ax = a->x;
	float	   *bx = b->x;
	Vector	   *result;
	float	   *rx;

	CheckDims(a, b);

	result = InitVector(a->dim);
	rx = result->x;

	/* Auto-vectorized */
	for (int i = 0, imax = a->dim; i < imax; i++)
		rx[i] = ax[i] + bx[i];

	/* Check for overflow */
	for (int i = 0, imax = a->dim; i < imax; i++)
	{
		if (isinf(rx[i]))
			float_overflow_error();
	}

	PG_RETURN_POINTER(result);
}

/*
 * Subtract vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_sub);
Datum
vector_sub(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	float	   *ax = a->x;
	float	   *bx = b->x;
	Vector	   *result;
	float	   *rx;

	CheckDims(a, b);

	result = InitVector(a->dim);
	rx = result->x;

	/* Auto-vectorized */
	for (int i = 0, imax = a->dim; i < imax; i++)
		rx[i] = ax[i] - bx[i];

	/* Check for overflow */
	for (int i = 0, imax = a->dim; i < imax; i++)
	{
		if (isinf(rx[i]))
			float_overflow_error();
	}

	PG_RETURN_POINTER(result);
}

/*
 * Multiply vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_mul);
Datum
vector_mul(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	float	   *ax = a->x;
	float	   *bx = b->x;
	Vector	   *result;
	float	   *rx;

	CheckDims(a, b);

	result = InitVector(a->dim);
	rx = result->x;

	/* Auto-vectorized */
	for (int i = 0, imax = a->dim; i < imax; i++)
		rx[i] = ax[i] * bx[i];

	/* Check for overflow and underflow */
	for (int i = 0, imax = a->dim; i < imax; i++)
	{
		if (isinf(rx[i]))
			float_overflow_error();

		if (rx[i] == 0 && !(ax[i] == 0 || bx[i] == 0))
			float_underflow_error();
	}

	PG_RETURN_POINTER(result);
}

/*
 * Concatenate vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_concat);
Datum
vector_concat(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);
	Vector	   *result;
	int			dim = a->dim + b->dim;

	CheckDim(dim);
	result = InitVector(dim);

	/* Auto-vectorized */
	for (int i = 0, imax = a->dim; i < imax; i++)
		result->x[i] = a->x[i];

	/* Auto-vectorized */
	for (int i = 0, imax = b->dim, start = a->dim; i < imax; i++)
		result->x[i + start] = b->x[i];

	PG_RETURN_POINTER(result);
}

/*
 * Quantize a vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(binary_quantize);
Datum
binary_quantize(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	float	   *ax = a->x;
	VarBit	   *result = InitBitVector(a->dim);
	unsigned char *rx = VARBITS(result);
	int			i = 0;
	int			count = (a->dim / 8) * 8;

	/* Auto-vectorized */
	for (; i < count; i += 8)
	{
		unsigned char result_byte = 0;

		for (int j = 0; j < 8; j++)
			result_byte |= (ax[i + j] > 0) << (7 - j);

		rx[i / 8] = result_byte;
	}

	for (; i < a->dim; i++)
		rx[i / 8] |= (ax[i] > 0) << (7 - (i % 8));

	PG_RETURN_VARBIT_P(result);
}

/*
 * Get a subvector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(subvector);
Datum
subvector(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	int32		start = PG_GETARG_INT32(1);
	int32		count = PG_GETARG_INT32(2);
	int32		end;
	float	   *ax = a->x;
	Vector	   *result;
	int			dim;

	if (count < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector must have at least 1 dimension")));

	/*
	 * Check if (start + count > a->dim), avoiding integer overflow. a->dim
	 * and count are both positive, so a->dim - count won't overflow.
	 */
	if (start > a->dim - count)
		end = a->dim + 1;
	else
		end = start + count;

	/* Indexing starts at 1, like substring */
	if (start < 1)
		start = 1;
	else if (start > a->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector must have at least 1 dimension")));

	dim = end - start;
	CheckDim(dim);
	result = InitVector(dim);

	for (int i = 0; i < dim; i++)
		result->x[i] = ax[start - 1 + i];

	PG_RETURN_POINTER(result);
}

/*
 * Internal helper to compare vectors
 */
int
vector_cmp_internal(Vector * a, Vector * b)
{
	int			dim = Min(a->dim, b->dim);

	/* Check values before dimensions to be consistent with Postgres arrays */
	for (int i = 0; i < dim; i++)
	{
		if (a->x[i] < b->x[i])
			return -1;

		if (a->x[i] > b->x[i])
			return 1;
	}

	if (a->dim < b->dim)
		return -1;

	if (a->dim > b->dim)
		return 1;

	return 0;
}

/*
 * Less than
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_lt);
Datum
vector_lt(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) < 0);
}

/*
 * Less than or equal
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_le);
Datum
vector_le(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) <= 0);
}

/*
 * Equal
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_eq);
Datum
vector_eq(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) == 0);
}

/*
 * Not equal
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_ne);
Datum
vector_ne(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) != 0);
}

/*
 * Greater than or equal
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_ge);
Datum
vector_ge(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) >= 0);
}

/*
 * Greater than
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_gt);
Datum
vector_gt(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_BOOL(vector_cmp_internal(a, b) > 0);
}

/*
 * Compare vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_cmp);
Datum
vector_cmp(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_VECTOR_P(0);
	Vector	   *b = PG_GETARG_VECTOR_P(1);

	PG_RETURN_INT32(vector_cmp_internal(a, b));
}

/*
 * Accumulate vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_accum);
Datum
vector_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *statearray = PG_GETARG_ARRAYTYPE_P(0);
	Vector	   *newval = PG_GETARG_VECTOR_P(1);
	float8	   *statevalues;
	int16		dim;
	bool		newarr;
	float8		n;
	Datum	   *statedatums;
	float	   *x = newval->x;
	ArrayType  *result;

	/* Check array before using */
	statevalues = CheckStateArray(statearray, "vector_accum");
	dim = STATE_DIMS(statearray);
	newarr = dim == 0;

	if (newarr)
		dim = newval->dim;
	else
		CheckExpectedDim(dim, newval->dim);

	n = statevalues[0] + 1.0;

	statedatums = CreateStateDatums(dim);
	statedatums[0] = Float8GetDatum(n);

	if (newarr)
	{
		for (int i = 0; i < dim; i++)
			statedatums[i + 1] = Float8GetDatum((double) x[i]);
	}
	else
	{
		for (int i = 0; i < dim; i++)
		{
			double		v = statevalues[i + 1] + x[i];

			/* Check for overflow */
			if (isinf(v))
				float_overflow_error();

			statedatums[i + 1] = Float8GetDatum(v);
		}
	}

	/* Use float8 array like float4_accum */
	result = construct_array(statedatums, dim + 1,
							 FLOAT8OID,
							 sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);

	pfree(statedatums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * Combine vectors or half vectors (also used for halfvec_combine)
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_combine);
Datum
vector_combine(PG_FUNCTION_ARGS)
{
	/* Must also update parameters of halfvec_combine if modifying */
	ArrayType  *statearray1 = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *statearray2 = PG_GETARG_ARRAYTYPE_P(1);
	float8	   *statevalues1;
	float8	   *statevalues2;
	float8		n;
	float8		n1;
	float8		n2;
	int16		dim;
	Datum	   *statedatums;
	ArrayType  *result;

	/* Check arrays before using */
	statevalues1 = CheckStateArray(statearray1, "vector_combine");
	statevalues2 = CheckStateArray(statearray2, "vector_combine");

	n1 = statevalues1[0];
	n2 = statevalues2[0];

	if (n1 == 0.0)
	{
		n = n2;
		dim = STATE_DIMS(statearray2);
		statedatums = CreateStateDatums(dim);
		for (int i = 1; i <= dim; i++)
			statedatums[i] = Float8GetDatum(statevalues2[i]);
	}
	else if (n2 == 0.0)
	{
		n = n1;
		dim = STATE_DIMS(statearray1);
		statedatums = CreateStateDatums(dim);
		for (int i = 1; i <= dim; i++)
			statedatums[i] = Float8GetDatum(statevalues1[i]);
	}
	else
	{
		n = n1 + n2;
		dim = STATE_DIMS(statearray1);
		CheckExpectedDim(dim, STATE_DIMS(statearray2));
		statedatums = CreateStateDatums(dim);
		for (int i = 1; i <= dim; i++)
		{
			double		v = statevalues1[i] + statevalues2[i];

			/* Check for overflow */
			if (isinf(v))
				float_overflow_error();

			statedatums[i] = Float8GetDatum(v);
		}
	}

	statedatums[0] = Float8GetDatum(n);

	result = construct_array(statedatums, dim + 1,
							 FLOAT8OID,
							 sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);

	pfree(statedatums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * Average vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_avg);
Datum
vector_avg(PG_FUNCTION_ARGS)
{
	ArrayType  *statearray = PG_GETARG_ARRAYTYPE_P(0);
	float8	   *statevalues;
	float8		n;
	uint16		dim;
	Vector	   *result;

	/* Check array before using */
	statevalues = CheckStateArray(statearray, "vector_avg");
	n = statevalues[0];

	/* SQL defines AVG of no values to be NULL */
	if (n == 0.0)
		PG_RETURN_NULL();

	/* Create vector */
	dim = STATE_DIMS(statearray);
	CheckDim(dim);
	result = InitVector(dim);
	for (int i = 0; i < dim; i++)
	{
		result->x[i] = statevalues[i + 1] / n;
		CheckElement(result->x[i]);
	}

	PG_RETURN_POINTER(result);
}

/*
 * Convert sparse vector to dense vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sparsevec_to_vector);
Datum
sparsevec_to_vector(PG_FUNCTION_ARGS)
{
	SparseVector *svec = PG_GETARG_SPARSEVEC_P(0);
	int32		typmod = PG_GETARG_INT32(1);
	Vector	   *result;
	int			dim = svec->dim;
	float	   *values = SPARSEVEC_VALUES(svec);

	CheckDim(dim);
	CheckExpectedDim(typmod, dim);

	result = InitVector(dim);
	for (int i = 0; i < svec->nnz; i++)
		result->x[svec->indices[i]] = values[i];

	PG_RETURN_POINTER(result);
}

#include "postgres.h"

#include <math.h>

#include "access/amapi.h"
#include "access/genam.h"
#include "access/heapam.h"
#include "access/skey.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "bitutils.h"
#include "bitvec.h"
#include "catalog/index.h"
#include "catalog/namespace.h"
#include "catalog/pg_class.h"
#include "catalog/pg_type.h"
#include "commands/trigger.h"
#include "common/hashfn.h"
#include "common/shortest_dec.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "funcapi.h"
#include "halfutils.h"
#include "halfvec.h"
#include "hnsw.h"
#include "ivfflat.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/params.h"
#include "optimizer/clauses.h"
#include "optimizer/optimizer.h"
#include "port.h"				/* for strtof() */
#include "executor/spi.h"
#include "parser/analyze.h"
#include "parser/parser.h"
#include "sparsevec.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/inval.h"
#include "utils/array.h"
#include "utils/float.h"
#include "utils/fmgrprotos.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/guc.h"
#include "utils/hsearch.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"
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
	double		benefitPerByte;
	uint64		uses;
	bool		adaptiveManaged;
} HnswMetadataCacheEntry;

typedef enum HnswAdaptiveState
{
	HNSW_ADAPTIVE_MISSING,
	HNSW_ADAPTIVE_PROBING,
	HNSW_ADAPTIVE_PAGE,
	HNSW_ADAPTIVE_BLOOM,
	HNSW_ADAPTIVE_EXACT,
	HNSW_ADAPTIVE_STALE
} HnswAdaptiveState;

typedef struct HnswGuidanceDescriptorKey
{
	Oid			heapOid;
	uint32		signatureBytes;
	uint64		signatureHash1;
	uint64		signatureHash2;
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
	HnswAdaptiveState adaptiveState;
	uint64		adaptiveRequests;
	uint64		adaptiveCycleRequests;
	uint64		adaptiveProbes;
	uint64		adaptiveCycleProbes;
	uint64		adaptiveUses;
	uint64		adaptiveAdmissions;
	int64		adaptiveProbeCandidates;
	int64		adaptiveProbeChecks;
	int64		adaptiveProbeSkips;
	double		adaptiveProbeHeapFetchMs;
	double		adaptiveProbeTotalMs;
	double		adaptivePageSkipRate;
	double		adaptiveBenefitPerByte;
	int64		adaptiveBytes;
	bool		adaptiveRefinePending;
	bool		adaptiveEpochTracked;
	int64		adaptiveEpoch;
	Oid			adaptiveRelFileNode;
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
	Oid			indexOid;
	Oid			heapOid;
	uint64		generation;
	uint32		signatureBytes;
	uint64		signatureHash1;
	uint64		signatureHash2;
	bool		statementBound;
	int64		bindingAttempts;
	int64		bindingMatches;
	int64		bindingMismatches;
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
	bool		adaptive;
	HnswGuidanceDescriptorEntry *adaptiveDescriptor;
	Expr	   *predicateExpr;
	MemoryContext predicateContext;
} HnswActiveGuidance;

typedef struct HnswGuidancePlanBinding
{
	struct HnswGuidancePlanBinding *next;
	QueryDesc  *queryDesc;
	IndexScanState *indexState;
	ExecProcNodeMtd underlyingExecProcNode;
	IndexScanDesc scan;
	IndexScan  *plan;
	uint64		frameId;
	uint64		guideGeneration;
	int			planNodeId;
	Index		scanrelid;
	Oid			indexOid;
	Oid			heapOid;
	HnswPlannerProofBypassReason precheckReason;
} HnswGuidancePlanBinding;

struct HnswScanGuidance
{
	IndexScanDesc scan;
	uint64		frameId;
	uint64		generation;
	int			planNodeId;
	Oid			indexOid;
	Oid			heapOid;
	bool		estimatedSkipRateValid;
	double		estimatedSkipRate;
	HnswActiveGuidance guide;
};

typedef struct HnswAdaptiveProbe
{
	HnswGuidanceDescriptorEntry *descriptor;
	Oid			heapOid;
	bool		epochTracked;
	int64		epoch;
	Oid			relFileNode;
} HnswAdaptiveProbe;

typedef struct HnswAdaptiveProfile
{
	int64		requests;
	int64		probes;
	int64		admissions;
	int64		rejections;
	int64		pageBuilds;
	int64		bloomBuilds;
	int64		refinements;
	int64		staleBypasses;
	int64		evictions;
	int64		bytes;
	double		score;
	int64		checks;
	int64		skips;
} HnswAdaptiveProfile;

static HnswMaterializeProfile hnsw_materialize_last_profile;
static HTAB *hnsw_metadata_caches = NULL;
static HTAB *hnsw_guidance_descriptors = NULL;
static HnswMetadataFilterProfile hnsw_metadata_filter_last_profile;
static HnswActiveGuidance hnsw_active_guidance;
static int	hnsw_metadata_cache_max_mb = 64;
static bool hnsw_guidance_compose_exact_or = false;
static bool hnsw_guidance_require_epoch = true;
static int	hnsw_d3_probe_requests = 2;
static double hnsw_d3_min_benefit_per_byte = 0;
static int	hnsw_d3_max_fragment_mb = 16;
static double hnsw_d3_page_min_skip_rate = 0.05;
static bool hnsw_fragment_store_ready = false;
static uint64 hnsw_metadata_cache_clock = 0;
static int64 hnsw_metadata_cache_evictions = 0;
static HnswAdaptiveProbe hnsw_adaptive_probe;
static HnswAdaptiveProfile hnsw_adaptive_profile;
static HnswGuidanceDescriptorEntry *hnsw_last_adaptive_descriptor = NULL;
static int64 hnsw_binding_attempts = 0;
static int64 hnsw_binding_matches = 0;
static int64 hnsw_binding_mismatches = 0;
static int64 hnsw_binding_scan_checks = 0;
static int64 hnsw_binding_scan_matches = 0;
static int64 hnsw_binding_scan_bypasses = 0;
static int64 hnsw_planner_proof_attempts = 0;
static int64 hnsw_planner_proof_successes = 0;
static int64 hnsw_planner_proof_failures = 0;
static HnswPlannerProofBypassReason hnsw_planner_proof_last_reason = HNSW_PROOF_BYPASS_NONE;
static int hnsw_planner_proof_last_plan_node_id = 0;
static Oid hnsw_planner_proof_last_index_oid = InvalidOid;
static Oid hnsw_planner_proof_last_heap_oid = InvalidOid;
static uint64 hnsw_planner_proof_last_generation = 0;
static uint64 hnsw_guidance_generation = 0;
static uint64 hnsw_executor_frame_generation = 0;
static ExecutorStart_hook_type previous_executor_start_hook = NULL;
static ExecutorEnd_hook_type previous_executor_end_hook = NULL;

/* QueryDesc pointers are used only while their executor frame is live. */
#define HNSW_EXECUTOR_BINDING_STACK_MAX 64
typedef struct HnswExecutorBindingFrame
{
	QueryDesc  *queryDesc;
	HnswGuidancePlanBinding *planBindings;
	SubTransactionId subid;
	uint64		frameId;
	bool		bindingSeen;
	bool		bindingMatched;
	uint64		boundGuideGeneration;
	Oid			boundIndexOid;
	Oid			boundHeapOid;
	uint32		boundSignatureBytes;
	uint64		boundSignatureHash1;
	uint64		boundSignatureHash2;
} HnswExecutorBindingFrame;

static HnswExecutorBindingFrame hnsw_executor_binding_stack[HNSW_EXECUTOR_BINDING_STACK_MAX];
static int hnsw_executor_binding_depth = 0;
static HnswGuidancePlanBinding *hnsw_executing_plan_binding = NULL;

static void HnswMetadataFreeCacheEntry(HnswMetadataCacheEntry *entry);
static void HnswGuidanceFreeDescriptorEntry(HnswGuidanceDescriptorEntry *entry);
static void HnswGuidanceDeactivate(void);
static const char *HnswPlannerProofBypassReasonName(HnswPlannerProofBypassReason reason);

static void
HnswExecutorBindingRefreshCompatibilityFlag(void)
{
	HnswExecutorBindingFrame *frame;

	hnsw_active_guidance.statementBound = false;
	if (hnsw_executor_binding_depth <= 0 || !hnsw_active_guidance.active)
		return;

	frame = &hnsw_executor_binding_stack[hnsw_executor_binding_depth - 1];
	hnsw_active_guidance.statementBound = frame->bindingMatched &&
		frame->boundGuideGeneration == hnsw_active_guidance.generation &&
		frame->boundIndexOid == hnsw_active_guidance.indexOid &&
		frame->boundHeapOid == hnsw_active_guidance.heapOid;
}

static HnswExecutorBindingFrame *
HnswExecutorBindingFindFrame(uint64 frameId, QueryDesc *queryDesc)
{
	for (int i = hnsw_executor_binding_depth - 1; i >= 0; i--)
	{
		HnswExecutorBindingFrame *frame = &hnsw_executor_binding_stack[i];

		if (frame->frameId == frameId && frame->queryDesc == queryDesc)
			return frame;
	}

	return NULL;
}

static void
HnswExecutorBindingReset(void)
{
	hnsw_executor_binding_depth = 0;
	hnsw_executing_plan_binding = NULL;
	hnsw_active_guidance.statementBound = false;
	MemSet(hnsw_executor_binding_stack, 0, sizeof(hnsw_executor_binding_stack));
}

static void
HnswExecutorBindingRestoreFrame(int frameIndex)
{
	if (frameIndex < 0 || frameIndex >= hnsw_executor_binding_depth)
	{
		HnswExecutorBindingReset();
		return;
	}

	hnsw_executor_binding_depth = frameIndex;
	for (int i = frameIndex; i < HNSW_EXECUTOR_BINDING_STACK_MAX; i++)
		MemSet(&hnsw_executor_binding_stack[i], 0, sizeof(HnswExecutorBindingFrame));
	HnswExecutorBindingRefreshCompatibilityFlag();
}

static void
HnswExecutorBindingAbortTo(QueryDesc *queryDesc)
{
	int			frameIndex;

	for (frameIndex = hnsw_executor_binding_depth - 1; frameIndex >= 0; frameIndex--)
	{
		if (hnsw_executor_binding_stack[frameIndex].queryDesc == queryDesc)
		{
			HnswExecutorBindingRestoreFrame(frameIndex);
			return;
		}
	}

	/* The executor frame was already discarded by a transaction callback. */
	HnswExecutorBindingReset();
}

static void
HnswExecutorBindingPop(QueryDesc *queryDesc)
{
	if (hnsw_executor_binding_depth > 0 &&
		hnsw_executor_binding_stack[hnsw_executor_binding_depth - 1].queryDesc == queryDesc)
	{
		HnswExecutorBindingRestoreFrame(hnsw_executor_binding_depth - 1);
		return;
	}

	/* Keep a malformed or partially unwound hook chain from leaking binding. */
	HnswExecutorBindingAbortTo(queryDesc);
}

static void
HnswExecutorBindingXactCallback(XactEvent event, void *arg)
{
	(void) arg;

	if (event == XACT_EVENT_ABORT || event == XACT_EVENT_PARALLEL_ABORT)
		hnsw_fragment_store_ready = false;

	if (event == XACT_EVENT_COMMIT || event == XACT_EVENT_PARALLEL_COMMIT ||
		event == XACT_EVENT_ABORT || event == XACT_EVENT_PARALLEL_ABORT)
		HnswExecutorBindingReset();
}

static void
HnswExecutorBindingSubXactCallback(SubXactEvent event,
									SubTransactionId mySubid,
									SubTransactionId parentSubid,
									void *arg)
{
	int				frameIndex;

	(void) arg;

	if (event == SUBXACT_EVENT_ABORT_SUB)
	{
		hnsw_fragment_store_ready = false;
		for (frameIndex = 0; frameIndex < hnsw_executor_binding_depth; frameIndex++)
		{
			if (hnsw_executor_binding_stack[frameIndex].subid == mySubid)
			{
				HnswExecutorBindingRestoreFrame(frameIndex);
				return;
			}
		}
	}
	else if (event == SUBXACT_EVENT_COMMIT_SUB)
	{
		for (frameIndex = 0; frameIndex < hnsw_executor_binding_depth; frameIndex++)
		{
			if (hnsw_executor_binding_stack[frameIndex].subid == mySubid)
				hnsw_executor_binding_stack[frameIndex].subid = parentSubid;
		}
	}
}

static void
VectorExecutorStart(QueryDesc *queryDesc, int eflags)
{
	HnswExecutorBindingFrame *frame;
	uint64		frameId;

	if (hnsw_executor_binding_depth >= HNSW_EXECUTOR_BINDING_STACK_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("HNSW executor binding stack overflow"),
				 errdetail("Nested executor depth exceeded the fixed limit of %d.",
						   HNSW_EXECUTOR_BINDING_STACK_MAX)));

	frame = &hnsw_executor_binding_stack[hnsw_executor_binding_depth++];
	MemSet(frame, 0, sizeof(*frame));
	frame->queryDesc = queryDesc;
	frame->subid = GetCurrentSubTransactionId();
	frame->frameId = ++hnsw_executor_frame_generation;
	if (frame->frameId == 0)
		frame->frameId = ++hnsw_executor_frame_generation;
	frameId = frame->frameId;

	/* Every QueryDesc owns an independent binding identity. */
	HnswExecutorBindingRefreshCompatibilityFlag();
	PG_TRY();
	{
		if (previous_executor_start_hook != NULL)
			previous_executor_start_hook(queryDesc, eflags);
		else
			standard_ExecutorStart(queryDesc, eflags);

		/* Register plan nodes; their IndexScanDesc objects are created lazily. */
		HnswGuidanceRegisterExecutorScans(queryDesc, frameId);
	}
	PG_CATCH();
	{
		HnswExecutorBindingAbortTo(queryDesc);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

static void
VectorExecutorEnd(QueryDesc *queryDesc)
{
	PG_TRY();
	{
		if (previous_executor_end_hook != NULL)
			previous_executor_end_hook(queryDesc);
		else
			standard_ExecutorEnd(queryDesc);
	}
	PG_CATCH();
	{
		HnswExecutorBindingAbortTo(queryDesc);
		PG_RE_THROW();
	}
	PG_END_TRY();
	HnswExecutorBindingPop(queryDesc);
}

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

	previous_executor_start_hook = ExecutorStart_hook;
	ExecutorStart_hook = VectorExecutorStart;
	previous_executor_end_hook = ExecutorEnd_hook;
	ExecutorEnd_hook = VectorExecutorEnd;
	RegisterXactCallback(HnswExecutorBindingXactCallback, NULL);
	RegisterSubXactCallback(HnswExecutorBindingSubXactCallback, NULL);

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
								 "Requires relation-epoch tracking for standalone guidance cache operations",
								 "Hard and pre-distance guidance always require a valid invalidation trigger, even when this setting is off.",
								 &hnsw_guidance_require_epoch,
								 true, PGC_SUSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.d3_probe_requests",
							"Sets the number of stock scans recorded before adaptive fragment admission",
							"Adaptive requests stay inactive while their descriptor collects this many scan observations.",
							&hnsw_d3_probe_requests,
							2, 1, 1000, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomRealVariable("hnsw.d3_min_benefit_per_byte",
							 "Sets the minimum estimated heap-fetch milliseconds saved per fragment byte",
							 "Bloom refinement requires a positive score at or above this value; the first page admission is the documented probe exception.",
							 &hnsw_d3_min_benefit_per_byte,
							 0, 0, DBL_MAX, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.d3_max_fragment_mb",
							"Sets the per-adaptive-admission fragment size cap",
							"An adaptive page or bloom activation is rejected when all selected atoms exceed this cap.",
							&hnsw_d3_max_fragment_mb,
							16, 1, 1024 * 1024, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomRealVariable("hnsw.d3_page_min_skip_rate",
							 "Sets the page-guidance skip rate below which adaptive refinement uses Bloom",
							 "The following activation builds Bloom after a page scan records a lower skip rate.",
							 &hnsw_d3_page_min_skip_rate,
							 0.05, 0, 1, PGC_USERSET, 0, NULL, NULL, NULL);
}

static const char *
HnswTraversalFinalPathName(HnswTraversalFinalPath path)
{
	switch (path)
	{
		case HNSW_TRAVERSAL_PATH_STOCK:
			return "stock";
		case HNSW_TRAVERSAL_PATH_VALIDATION_ONLY:
			return "validation_only";
		case HNSW_TRAVERSAL_PATH_LEGACY_GUIDED:
			return "legacy_guided";
		case HNSW_TRAVERSAL_PATH_GUIDED:
			return "guided";
		case HNSW_TRAVERSAL_PATH_STOCK_BYPASS:
			return "stock_bypass";
		case HNSW_TRAVERSAL_PATH_FRESH_STOCK_FALLBACK:
			return "fresh_stock_fallback";
	}
	return "unknown";
}

static const char *
HnswTraversalStockBypassReasonName(HnswTraversalStockBypassReason reason)
{
	switch (reason)
	{
		case HNSW_TRAVERSAL_BYPASS_NONE:
			return "none";
		case HNSW_TRAVERSAL_BYPASS_NO_PROVEN_GUIDE:
			return "no_proven_guide";
		case HNSW_TRAVERSAL_BYPASS_SKIP_ESTIMATE_UNAVAILABLE:
			return "skip_estimate_unavailable";
		case HNSW_TRAVERSAL_BYPASS_LOW_ESTIMATED_SKIP_RATE:
			return "low_estimated_skip_rate";
		case HNSW_TRAVERSAL_BYPASS_ITERATIVE_SCAN:
			return "iterative_scan";
	}
	return "unknown";
}

static const char *
HnswTraversalFallbackReasonName(HnswTraversalFallbackReason reason)
{
	switch (reason)
	{
		case HNSW_TRAVERSAL_FALLBACK_NONE:
			return "none";
		case HNSW_TRAVERSAL_FALLBACK_INSUFFICIENT_MATCHES:
			return "insufficient_guided_matches";
		case HNSW_TRAVERSAL_FALLBACK_BRIDGE_HOPS:
			return "bridge_hop_budget";
		case HNSW_TRAVERSAL_FALLBACK_BRIDGE_WORK:
			return "bridge_work_budget";
		case HNSW_TRAVERSAL_FALLBACK_MAX_SCAN_TUPLES:
			return "max_scan_tuples";
		case HNSW_TRAVERSAL_FALLBACK_MEMORY_LIMIT:
			return "memory_limit";
		case HNSW_TRAVERSAL_FALLBACK_INVALID_NEIGHBOR:
			return "invalid_neighbor_version";
	}
	return "unknown";
}

static const char *
HnswIterativeScanModeName(HnswIterativeScanMode mode)
{
	switch (mode)
	{
		case HNSW_ITERATIVE_SCAN_OFF:
			return "off";
		case HNSW_ITERATIVE_SCAN_RELAXED:
			return "relaxed_order";
		case HNSW_ITERATIVE_SCAN_STRICT:
			return "strict_order";
	}
	return "unknown";
}

static const char *
HnswFilterStrategyModeName(HnswFilterStrategyMode mode)
{
	switch (mode)
	{
		case HNSW_FILTER_STRATEGY_OFF:
			return "off";
		case HNSW_FILTER_STRATEGY_ACORN1:
			return "acorn1";
		case HNSW_FILTER_STRATEGY_GUIDED_COLLECT:
			return "guided_collect";
		case HNSW_FILTER_STRATEGY_TRAVERSAL_GUIDED:
			return "traversal_guided";
		case HNSW_FILTER_STRATEGY_SAFE_GUIDED:
			return "safe_guided";
	}
	return "unknown";
}

static bool
HnswTraversalNetDistanceSavedAvailable(HnswTraversalFinalPath path)
{
	return path == HNSW_TRAVERSAL_PATH_STOCK ||
		path == HNSW_TRAVERSAL_PATH_VALIDATION_ONLY ||
		path == HNSW_TRAVERSAL_PATH_STOCK_BYPASS ||
		path == HNSW_TRAVERSAL_PATH_FRESH_STOCK_FALLBACK;
}

static int64
HnswTraversalNetDistanceSaved(const HnswScanProfile *profile)
{
	if (profile->traversalFinalPath ==
		HNSW_TRAVERSAL_PATH_FRESH_STOCK_FALLBACK)
		return -profile->traversal.guidedPhaseDistanceComputations;
	return 0;
}

static void
VectorHnswLastProfileToText(StringInfo output, const HnswScanProfile *profile)
{
	appendStringInfo(output,
					"{\"valid\":%s,"
					"\"profile_semantics_version\":7,"
					"\"total_scan_ms\":%.6f,"
					"\"hnsw_search_ms\":%.6f,"
					"\"heap_fetch_ms\":%.6f,"
					"\"vector_search_ms\":%.6f,"
					"\"hnsw_am_callback_ms\":%.6f,"
					"\"executor_residual_ms\":%.6f,"
					"\"heap_fetch_ms_is_residual_proxy\":true,"
					"\"visited_tuples\":" INT64_FORMAT ","
					"\"returned_tuples\":" INT64_FORMAT ","
					"\"graph_elements_visited\":" INT64_FORMAT ","
					"\"raw_index_tids_returned\":" INT64_FORMAT ","
					"\"distance_compute_count\":" INT64_FORMAT ","
					"\"page_access_batches\":" INT64_FORMAT ","
					"\"page_access_candidates\":" INT64_FORMAT ","
					"\"page_access_prefetches\":" INT64_FORMAT ","
					"\"page_access_distance_runs\":" INT64_FORMAT ","
					"\"page_access_distinct_pages\":" INT64_FORMAT ","
						"\"guidance_checks\":" INT64_FORMAT ","
						"\"guidance_matches\":" INT64_FORMAT ","
						"\"guidance_skips\":" INT64_FORMAT ","
						"\"traversal_expanded_nodes\":" INT64_FORMAT ","
						"\"traversal_neighbors_examined\":" INT64_FORMAT ","
						"\"traversal_guidance_checks\":" INT64_FORMAT ","
						"\"traversal_guidance_matches\":" INT64_FORMAT ","
						"\"traversal_guidance_misses\":" INT64_FORMAT ","
						"\"neighbor_expansion_guidance_checks\":" INT64_FORMAT ","
						"\"neighbor_expansion_guidance_matches\":" INT64_FORMAT ","
						"\"neighbor_expansion_guidance_misses\":" INT64_FORMAT ","
						"\"traversal_matching_expanded\":" INT64_FORMAT ","
						"\"traversal_bridge_expanded\":" INT64_FORMAT ","
						"\"traversal_candidate_admissions\":" INT64_FORMAT ","
						"\"traversal_result_admissions\":" INT64_FORMAT ","
						"\"traversal_guided_admissions\":" INT64_FORMAT ","
						"\"traversal_guided_suppressions\":" INT64_FORMAT ","
						"\"traversal_heap_tids_suppressed\":" INT64_FORMAT ","
						"\"traversal_stop_deferrals\":" INT64_FORMAT ","
						"\"traversal_discarded_pushes\":" INT64_FORMAT ","
						"\"traversal_discarded_pops\":" INT64_FORMAT ","
						"\"traversal_initial_batches\":" INT64_FORMAT ","
						"\"traversal_resume_batches\":" INT64_FORMAT ","
						"\"traversal_strict_order_drops\":" INT64_FORMAT ","
						"\"traversal_stock_terminations\":" INT64_FORMAT ","
						"\"traversal_max_scan_terminations\":" INT64_FORMAT ","
						"\"traversal_exhausted_terminations\":" INT64_FORMAT ","
						"\"index_page_neighbor_loads\":" INT64_FORMAT ","
					"\"index_page_neighbor_runs\":" INT64_FORMAT ","
					"\"index_page_distinct_counts_exact\":%s,"
					"\"index_page_distinct_page_limit\":%d,"
					"\"index_page_distinct_scope\":\"sum_of_scan_local_unique_pages\","
					"\"index_page_profile_scope\":\"search_readbuffer_sequence_all_metapage_entry_neighbor_candidate_element\","
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
					"\"heap_blks_scope\":\"executor_buffer_delta_residual_after_index_delta\","
					"\"heap_blks_are_exact_heap_io\":false,"
					"\"topk_count\":%d,"
					"\"topk_ids\":[",
					profile->valid ? "true" : "false",
					profile->totalScanMs,
					profile->hnswSearchMs,
					profile->heapFetchMs,
					profile->vectorSearchMs,
					profile->hnswSearchMs,
					profile->heapFetchMs,
					profile->visitedTuples,
					profile->returnedTuples,
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
						profile->traversal.expandedNodes,
						profile->traversal.neighborsExamined,
						profile->traversal.guidanceChecks,
						profile->traversal.guidanceMatches,
						profile->traversal.guidanceMisses,
						profile->traversal.neighborGuidanceChecks,
						profile->traversal.neighborGuidanceMatches,
						profile->traversal.neighborGuidanceMisses,
						profile->traversal.matchingExpanded,
						profile->traversal.bridgeExpanded,
						profile->traversal.candidateAdmissions,
						profile->traversal.resultAdmissions,
						profile->traversal.guidedAdmissions,
						profile->traversal.guidedSuppressions,
						profile->traversal.heapTidsSuppressed,
						profile->traversal.stopDeferrals,
						profile->traversal.discardedPushes,
						profile->traversal.discardedPops,
						profile->traversal.initialBatches,
						profile->traversal.resumeBatches,
						profile->traversal.strictOrderDrops,
						profile->traversal.stockTerminations,
						profile->traversal.maxScanTerminations,
						profile->traversal.exhaustedTerminations,
						profile->indexPageNeighborLoads,
					profile->indexPageNeighborRuns,
					profile->indexPageDistinctCountsExact ? "true" : "false",
					HNSW_INDEX_PAGE_UNIQUE_LIMIT,
					profile->indexPageDistinctCountsExact ?
						profile->indexPageNeighborDistinctPages : -1,
					profile->indexPageElementLoads,
					profile->indexPageElementRuns,
					profile->indexPageDistinctCountsExact ?
						profile->indexPageElementDistinctPages : -1,
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

	appendStringInfo(output,
					"],\"index_page_loads\":" INT64_FORMAT
					",\"index_page_runs\":" INT64_FORMAT
					",\"index_page_distinct_pages\":" INT64_FORMAT
					",\"index_page_last_block\":%u"
					",\"index_page_distinct_pages_exact\":%s"
					",\"heap_tid_returns\":" INT64_FORMAT
					",\"heap_tid_page_runs\":" INT64_FORMAT
					",\"heap_tid_distinct_pages\":" INT64_FORMAT
					",\"heap_tid_distinct_pages_exact\":%s"
					",\"heap_tid_sequence_scope\":\"sum_of_scan_local_actual_xs_heaptid_return_sequences\""
					",\"heap_validation_guidance_checks\":" INT64_FORMAT
					",\"heap_validation_guidance_matches\":" INT64_FORMAT
					",\"heap_validation_guidance_skips\":" INT64_FORMAT
					",\"pre_distance_membership_checks\":" INT64_FORMAT
						",\"pre_distance_membership_matches\":" INT64_FORMAT
						",\"pre_distance_membership_misses\":" INT64_FORMAT
						",\"distance_computations_avoided_attempted\":" INT64_FORMAT
							",\"distance_computations_avoided\":" INT64_FORMAT
							",\"distance_computations_avoided_scope\":\"guided_path_local_only\""
						",\"miss_bridge_nodes\":" INT64_FORMAT
						",\"miss_bridge_edges\":" INT64_FORMAT
						",\"miss_bridge_max_hops\":" INT64_FORMAT
						",\"bridge_pending_at_termination\":" INT64_FORMAT
						",\"guided_expanded_nodes\":" INT64_FORMAT
						",\"guided_attempt_expanded_nodes\":" INT64_FORMAT
						",\"guided_phase_distance_computations\":" INT64_FORMAT
						",\"guided_attempt_distance_computations\":" INT64_FORMAT
						",\"stock_phase_expanded_nodes\":" INT64_FORMAT
					",\"stock_phase_distance_computations\":" INT64_FORMAT
					",\"stock_bypass_requests\":" INT64_FORMAT
					",\"stock_bypass_reason\":\"%s\""
					",\"fallback_requests\":" INT64_FORMAT
					",\"fallback_reason\":\"%s\""
						",\"fallback_stock_expanded_nodes\":" INT64_FORMAT
						",\"fallback_stock_distance_computations\":" INT64_FORMAT
						",\"net_distance_saved_available\":%s"
						",\"net_distance_saved\":" INT64_FORMAT
						",\"traversal_estimated_skip_rate_valid\":%s"
						",\"traversal_estimated_skip_rate\":%.6f"
						",\"iterative_scan\":\"%s\""
						",\"filter_strategy\":\"%s\""
						",\"traversal_guidance_scope\":\"candidate_admission_and_validation\""
						",\"graph_expansion_pruned\":false"
						",\"distance_computations_pruned\":false"
						",\"final_path\":\"%s\""
					",\"planner_proof_attempted\":%s"
					",\"planner_proof_succeeded\":%s"
					",\"planner_proof_bypass_reason\":\"%s\""
					",\"planner_proof_plan_node_id\":%d"
					",\"planner_proof_index_oid\":%u"
					",\"planner_proof_heap_oid\":%u"
					",\"planner_proof_generation\":" INT64_FORMAT
					",\"planner_proof_guide_generation\":" INT64_FORMAT
					",\"planner_proof_count\":%d"
					",\"planner_proofs_truncated\":%s"
					",\"planner_proofs\":[",
					profile->indexPageLoads,
					profile->indexPageRuns,
					profile->indexPageDistinctPagesExact ?
						profile->indexPageDistinctPages : -1,
					profile->indexPageLastBlock,
					profile->indexPageDistinctPagesExact ? "true" : "false",
					profile->heapTidReturns,
					profile->heapTidPageRuns,
					profile->heapTidDistinctPagesExact ?
						profile->heapTidDistinctPages : -1,
					profile->heapTidDistinctPagesExact ? "true" : "false",
					profile->guidanceChecks,
					profile->guidanceMatches,
					profile->guidanceSkips,
						profile->traversal.preDistanceChecks,
						profile->traversal.preDistanceMatches,
						profile->traversal.preDistanceMisses,
							profile->traversal.attemptedDistanceComputationsAvoided,
							profile->traversal.distanceComputationsAvoided,
						profile->traversal.missBridgeNodes,
						profile->traversal.missBridgeEdges,
						profile->traversal.maxMissBridgeHops,
						profile->traversal.bridgePendingAtTermination,
						profile->traversal.guidedExpandedNodes,
						profile->traversal.guidedExpandedNodes,
						profile->traversal.guidedPhaseDistanceComputations,
						profile->traversal.guidedPhaseDistanceComputations,
						profile->traversal.stockPhaseExpandedNodes,
					profile->traversal.stockPhaseDistanceComputations,
					profile->traversal.stockBypassRequests,
					HnswTraversalStockBypassReasonName(
						profile->traversalStockBypassReason),
					profile->traversal.fallbackRequests,
					HnswTraversalFallbackReasonName(
						profile->traversalFallbackReason),
						profile->traversal.fallbackStockExpandedNodes,
						profile->traversal.fallbackStockDistanceComputations,
						HnswTraversalNetDistanceSavedAvailable(
							profile->traversalFinalPath) ? "true" : "false",
						HnswTraversalNetDistanceSaved(profile),
						profile->traversalEstimatedSkipRateValid ? "true" : "false",
						profile->traversalEstimatedSkipRate,
						HnswIterativeScanModeName(profile->iterativeScan),
						HnswFilterStrategyModeName(profile->filterStrategy),
						HnswTraversalFinalPathName(profile->traversalFinalPath),
					profile->plannerProof.attempted ? "true" : "false",
					profile->plannerProof.succeeded ? "true" : "false",
					HnswPlannerProofBypassReasonName(profile->plannerProof.bypassReason),
					profile->plannerProof.planNodeId,
					profile->plannerProof.indexOid,
					profile->plannerProof.heapOid,
					(int64) profile->plannerProof.guideGeneration,
					(int64) profile->plannerProof.guideGeneration,
					profile->plannerProofCount,
					profile->plannerProofsTruncated ? "true" : "false");

	for (int i = 0; i < profile->plannerProofCount; i++)
	{
		const HnswPlannerProofOutcome *proof = &profile->plannerProofs[i];

		if (i > 0)
			appendStringInfoChar(output, ',');
		appendStringInfo(output,
						 "{\"attempted\":%s,\"succeeded\":%s,"
						 "\"bypass_reason\":\"%s\",\"plan_node_id\":%d,"
						 "\"index_oid\":%u,\"heap_oid\":%u,"
						 "\"guide_generation\":" INT64_FORMAT "}",
						 proof->attempted ? "true" : "false",
						 proof->succeeded ? "true" : "false",
						 HnswPlannerProofBypassReasonName(proof->bypassReason),
						 proof->planNodeId,
						 proof->indexOid,
						 proof->heapOid,
						 (int64) proof->guideGeneration);
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

	/* Drop active references before destroying their backing payloads. */
	HnswGuidanceDeactivate();

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
	MemSet(&hnsw_adaptive_probe, 0, sizeof(hnsw_adaptive_probe));
	MemSet(&hnsw_adaptive_profile, 0, sizeof(hnsw_adaptive_profile));
	hnsw_last_adaptive_descriptor = NULL;
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

static bool
HnswMetadataCacheEntryIsActive(HnswMetadataCacheEntry *entry)
{
	if (!hnsw_active_guidance.active)
		return false;

	for (int i = 0; i < hnsw_active_guidance.atoms; i++)
	{
		if (hnsw_active_guidance.atom[i].cache == entry)
			return true;
	}

	return false;
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
HnswMetadataAdaptiveCacheStats(int64 *entries, int64 *bytes, int64 *uses,
						   double *score)
{
	HASH_SEQ_STATUS status;
	HnswMetadataCacheEntry *entry;

	*entries = 0;
	*bytes = 0;
	*uses = 0;
	*score = 0;
	if (hnsw_metadata_caches == NULL)
		return;

	hash_seq_init(&status, hnsw_metadata_caches);
	while ((entry = (HnswMetadataCacheEntry *) hash_seq_search(&status)) != NULL)
	{
		if (!entry->adaptiveManaged)
			continue;
		(*entries)++;
		*bytes += HnswMetadataEntryMemoryBytes(entry);
		*uses += (int64) entry->uses;
		*score += entry->benefitPerByte;
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
	entry->benefitPerByte = 0;
	entry->uses = 0;
	entry->adaptiveManaged = false;
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
			if (entry == protected || HnswMetadataCacheEntryIsActive(entry) ||
				HnswMetadataEntryMemoryBytes(entry) == 0)
				continue;
			if (victim == NULL ||
				entry->benefitPerByte < victim->benefitPerByte ||
				(entry->benefitPerByte == victim->benefitPerByte && entry->lastUsed < victim->lastUsed))
				victim = entry;
		}

		if (victim == NULL)
			break;

		if (victim->adaptiveManaged)
			hnsw_adaptive_profile.evictions++;
		HnswMetadataFreeCacheEntry(victim);
		hnsw_metadata_cache_evictions++;
	}
}

static bool
HnswMetadataHasOnlyAllowedStaticArrayCasts(const char *predicate)
{
	const char *cast = predicate;

	while ((cast = strstr(cast, "::")) != NULL)
	{
		const char *before = cast;
		const char *typeName = cast + 2;
		Size		length = 0;

		while (before > predicate && isspace((unsigned char) before[-1]))
			before--;
		if (before == predicate || before[-1] != ']')
			return false;

		while (typeName[length] != '\0' &&
				(isalnum((unsigned char) typeName[length]) ||
				 typeName[length] == '_' || typeName[length] == '[' || typeName[length] == ']'))
			length++;

		if (!((length == strlen("int[]") && pg_strncasecmp(typeName, "int[]", length) == 0) ||
			  (length == strlen("bigint[]") && pg_strncasecmp(typeName, "bigint[]", length) == 0) ||
			  (length == strlen("text[]") && pg_strncasecmp(typeName, "text[]", length) == 0)))
			return false;
		cast = typeName + length;
	}

	return true;
}

static void
HnswMetadataValidateSqlPredicate(const char *predicate)
{
	char	   *lower = pstrdup(predicate);
	char	   *padded;
	const char *unsafeTokens[] = {
		" select ", " from ", " join ", " exists ", " with ",
		" union ", " current_user ", " session_user ", " current_",
		" localtime", " localtimestamp"
	};

	for (char *cursor = lower; *cursor != '\0'; cursor++)
		*cursor = pg_tolower((unsigned char) *cursor);
	padded = psprintf(" %s ", lower);

	if (strchr(predicate, ';') != NULL || strchr(predicate, '(') != NULL ||
		strchr(predicate, ')') != NULL || !HnswMetadataHasOnlyAllowedStaticArrayCasts(predicate) ||
		strchr(predicate, '.') != NULL)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("HNSW guidance requires a row-local immutable predicate"),
				 errhint("Use only unqualified columns, constants, comparisons, AND, OR, and NOT.")));

	for (int i = 0; i < lengthof(unsafeTokens); i++)
	{
		if (strstr(padded, unsafeTokens[i]) != NULL)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("HNSW guidance predicate has an untracked dependency"),
					 errdetail("Unsupported token: %s", unsafeTokens[i]),
					 errhint("Keep joins, RLS, ACL, temporal, and volatile checks in PostgreSQL's final recheck.")));
	}

	pfree(padded);
	pfree(lower);
}

static const char *
HnswMetadataPredicateSql(const char *filterName)
{
	if (strncmp(filterName, "sql:", 4) == 0)
	{
		HnswMetadataValidateSqlPredicate(filterName + 4);
		return filterName + 4;
	}
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
	Size		filterNameBytes = strlen(filterName);

	if (filterNameBytes >= sizeof(key.filterName))
		ereport(ERROR,
				(errcode(ERRCODE_NAME_TOO_LONG),
				 errmsg("HNSW guidance atom is too long"),
				 errdetail("Guidance atoms must be shorter than %zu bytes.", sizeof(key.filterName))));

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

	if (namespaceName == NULL || relationName == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_TABLE),
				 errmsg("could not resolve heap relation %u", heapOid)));

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

/*
 * PostgreSQL stores an index TID at the root of a HOT chain, while SELECT
 * ctid returns the currently visible chain member.  Candidate guidance is
 * applied to the index TID, so exact and Bloom fragments must include both.
 */
static void
HnswMetadataExpandHotRoots(Oid heapOid, HnswMetadataCacheEntry *cache,
							HnswGuidanceKind kind)
{
	Relation	heapRel;
	Snapshot	snapshot;
	BlockNumber blocks;

	Assert(kind == HNSW_GUIDANCE_KIND_EXACT || kind == HNSW_GUIDANCE_KIND_BLOOM);
	if (cache->pageBits == NULL)
		return;
	snapshot = GetActiveSnapshot();
	if (snapshot == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("HNSW guidance fragment construction requires an active snapshot")));

	heapRel = table_open(heapOid, AccessShareLock);
	blocks = RelationGetNumberOfBlocks(heapRel);
	for (BlockNumber block = 0; block < blocks; block++)
	{
		Buffer		buffer;
		Page		page;
		OffsetNumber maxOffset;

		if (!HnswMetadataPageBitTest(cache, block))
			continue;

		buffer = ReadBuffer(heapRel, block);
		LockBuffer(buffer, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buffer);
		maxOffset = PageGetMaxOffsetNumber(page);

		for (OffsetNumber offset = FirstOffsetNumber; offset <= maxOffset; offset++)
		{
			ItemId		linePointer = PageGetItemId(page, offset);
			ItemPointerData rootTid;
			ItemPointerData visibleTid;
			HeapTupleData visibleTuple;
			bool		matches = false;

			if (ItemIdIsNormal(linePointer))
			{
				HeapTupleHeader header = (HeapTupleHeader) PageGetItem(page, linePointer);

				if (HeapTupleHeaderIsHeapOnly(header) ||
					!HeapTupleHeaderIsHotUpdated(header))
					continue;
			}
			else if (!ItemIdIsRedirected(linePointer))
				continue;

			ItemPointerSet(&rootTid, block, offset);
			visibleTid = rootTid;
			if (!heap_hot_search_buffer(&visibleTid, heapRel, buffer, snapshot,
									&visibleTuple, NULL, true))
				continue;

			if (kind == HNSW_GUIDANCE_KIND_EXACT)
			{
				HnswMetadataTidKey visibleKey;
				HnswMetadataTidEntry *visibleEntry;

				visibleKey.tid = visibleTid;
				visibleEntry = (HnswMetadataTidEntry *) hash_search(cache->tidHash,
															  &visibleKey, HASH_FIND, NULL);
				if (visibleEntry != NULL)
					matches = true;
			}
			else
				matches = HnswMetadataBloomMayContain(cache, &visibleTid);

			if (matches && kind == HNSW_GUIDANCE_KIND_EXACT)
			{
				HnswMetadataTidKey rootKey;
				bool		found;

				rootKey.tid = rootTid;
				hash_search(cache->tidHash, &rootKey, HASH_ENTER, &found);
				if (!found)
				{
					cache->rows++;
				}
			}
			else if (matches)
				HnswMetadataBloomSet(cache, &rootTid);
		}

		UnlockReleaseBuffer(buffer);
	}
	table_close(heapRel, AccessShareLock);
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
HnswAdaptiveStateName(HnswAdaptiveState state)
{
	switch (state)
	{
		case HNSW_ADAPTIVE_MISSING:
			return "missing";
		case HNSW_ADAPTIVE_PROBING:
			return "probing";
		case HNSW_ADAPTIVE_PAGE:
			return "page";
		case HNSW_ADAPTIVE_BLOOM:
			return "bloom";
		case HNSW_ADAPTIVE_EXACT:
			return "exact";
		case HNSW_ADAPTIVE_STALE:
			return "stale";
	}

	return "missing";
}

static bool
HnswAdaptiveDescriptorVersionMatches(HnswGuidanceDescriptorEntry *descriptor,
								 bool tracked, int64 epoch, Oid relFileNode)
{
	return descriptor->adaptiveState != HNSW_ADAPTIVE_MISSING &&
		descriptor->adaptiveEpochTracked == tracked &&
		(!tracked || descriptor->adaptiveEpoch == epoch) &&
		descriptor->adaptiveRelFileNode == relFileNode;
}

static void
HnswAdaptiveBeginProbeCycle(HnswGuidanceDescriptorEntry *descriptor,
							bool tracked, int64 epoch, Oid relFileNode)
{
	descriptor->adaptiveState = HNSW_ADAPTIVE_PROBING;
	descriptor->adaptiveCycleRequests = 0;
	descriptor->adaptiveCycleProbes = 0;
	descriptor->adaptiveProbeCandidates = 0;
	descriptor->adaptiveProbeChecks = 0;
	descriptor->adaptiveProbeSkips = 0;
	descriptor->adaptiveProbeHeapFetchMs = 0;
	descriptor->adaptiveProbeTotalMs = 0;
	descriptor->adaptivePageSkipRate = 0;
	descriptor->adaptiveBenefitPerByte = 0;
	descriptor->adaptiveBytes = 0;
	descriptor->adaptiveRefinePending = false;
	descriptor->adaptiveEpochTracked = tracked;
	descriptor->adaptiveEpoch = epoch;
	descriptor->adaptiveRelFileNode = relFileNode;
}

static void
HnswAdaptiveMarkStale(HnswGuidanceDescriptorEntry *descriptor)
{
	if (descriptor == NULL)
		return;

	descriptor->adaptiveState = HNSW_ADAPTIVE_STALE;
	descriptor->adaptiveRefinePending = false;
	hnsw_adaptive_profile.staleBypasses++;
}

static int64
HnswAdaptiveFragmentLimitBytes(void)
{
	return (int64) hnsw_d3_max_fragment_mb * 1024L * 1024L;
}

static double
HnswAdaptiveEstimateBloomSkipRate(Oid heapOid, int64 matchingRows,
						  double pageSkipRate)
{
	Relation	heapRel;
	double		totalRows;
	double		selectivitySkipRate = 0;

	heapRel = table_open(heapOid, AccessShareLock);
	totalRows = heapRel->rd_rel->reltuples;
	table_close(heapRel, AccessShareLock);
	if (totalRows > 0)
		selectivitySkipRate = Max(0, 1.0 - ((double) matchingRows / totalRows));

	return Max(pageSkipRate, selectivitySkipRate);
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
		"format_version integer NOT NULL DEFAULT 3,"
		"built_at timestamptz NOT NULL DEFAULT pg_catalog.now(),"
		"PRIMARY KEY (heap_oid, filter_name, kind)"
		")",
		false, 0);
	if (spiStatus != SPI_OK_UTILITY)
		elog(ERROR, "SPI_execute failed: %d", spiStatus);

	spiStatus = SPI_execute(
		"CREATE TABLE IF NOT EXISTS public.pgvector_hnsw_fragment_epoch ("
		"heap_oid oid PRIMARY KEY,"
		"epoch bigint NOT NULL DEFAULT 0,"
		"updated_at timestamptz NOT NULL DEFAULT pg_catalog.now()"
		")",
		false, 0);
	if (spiStatus != SPI_OK_UTILITY)
		elog(ERROR, "SPI_execute failed: %d", spiStatus);

	spiStatus = SPI_execute(
		"ALTER TABLE public.pgvector_hnsw_fragment_store "
		"ADD COLUMN IF NOT EXISTS build_epoch bigint NOT NULL DEFAULT 0, "
		"ADD COLUMN IF NOT EXISTS relfilenode oid NOT NULL DEFAULT 0, "
		"ADD COLUMN IF NOT EXISTS format_version integer NOT NULL DEFAULT 1",
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
	bool		triggerIsNull;
	bool		validTrigger = false;
	bool		tracked = false;

	*epoch = 0;
	*relFileNode = InvalidOid;
	HnswMetadataEnsureFragmentStore();

	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);

	spiStatus = SPI_execute_with_args(
				"SELECT e.epoch, pg_catalog.pg_relation_filenode($1), "
				"EXISTS (SELECT 1 FROM pg_catalog.pg_trigger AS t "
				"WHERE t.tgrelid = $1 "
				"AND t.tgname = 'pgvector_hnsw_fragment_epoch' "
				"AND NOT t.tgisinternal "
				"AND t.tgenabled IN ('O', 'A') "
				"AND EXISTS (SELECT 1 "
				"FROM pg_catalog.pg_proc AS p "
				"JOIN pg_catalog.pg_depend AS d ON d.objid = p.oid "
				"AND d.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass "
				"AND d.refclassid = 'pg_catalog.pg_extension'::pg_catalog.regclass "
				"AND d.deptype = 'e' "
				"JOIN pg_catalog.pg_extension AS x ON x.oid = d.refobjid "
				"WHERE p.oid = t.tgfoid "
				"AND p.proname = 'vector_hnsw_fragment_epoch_bump_trigger' "
				"AND p.pronamespace = x.extnamespace "
				"AND p.pronargs = 0 "
				"AND p.prorettype = 'pg_catalog.trigger'::pg_catalog.regtype "
				"AND x.extname = 'vector') "
				"AND t.tgnargs = 0 "
				"AND t.tgqual IS NULL "
				"AND t.tgattr::text = '' "
				"AND t.tgtype = 60) "
				"FROM (SELECT 1) AS singleton "
			"LEFT JOIN public.pgvector_hnsw_fragment_epoch AS e ON e.heap_oid = $1",
			1, argTypes, values, nulls, true, 1);
	if (spiStatus != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);

	if (SPI_processed == 1)
	{
		Datum		epochDatum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
		Datum		relFileNodeDatum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &relFileNodeIsNull);
		Datum		triggerDatum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3, &triggerIsNull);

		if (!triggerIsNull)
			validTrigger = DatumGetBool(triggerDatum);
		if (!isnull && validTrigger)
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
	if (!TRIGGER_FIRED_FOR_STATEMENT(triggerData->tg_event) ||
		!(TRIGGER_FIRED_BY_INSERT(triggerData->tg_event) ||
		  TRIGGER_FIRED_BY_UPDATE(triggerData->tg_event) ||
		  TRIGGER_FIRED_BY_DELETE(triggerData->tg_event) ||
		  TRIGGER_FIRED_BY_TRUNCATE(triggerData->tg_event)))
		ereport(ERROR,
				(errcode(ERRCODE_E_R_I_E_TRIGGER_PROTOCOL_VIOLATED),
				 errmsg("vector_hnsw_fragment_epoch_bump_trigger requires a statement-level data-change trigger")));
	heapOid = RelationGetRelid(triggerData->tg_relation);
	values[0] = ObjectIdGetDatum(heapOid);

	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);
	spiStatus = SPI_execute_with_args(
		"UPDATE public.pgvector_hnsw_fragment_epoch "
		"SET epoch = epoch + 1, updated_at = pg_catalog.now() "
		"WHERE heap_oid = $1",
		1, argTypes, values, nulls, false, 0);
	if (spiStatus != SPI_OK_UPDATE)
		elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);
	if (SPI_processed != 1)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("fragment epoch tracking is not registered for relation %u", heapOid),
				 errhint("Call vector_hnsw_fragment_tracking_enable() before modifying the relation.")));
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
	bool		validTrigger = false;
	char	   *relName;
	char	   *namespaceName;
	char	   *qualifiedName;
	char	   *functionNamespaceName;
	char	   *qualifiedTriggerFunction;
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
	functionNamespaceName = get_namespace_name(get_func_namespace(fcinfo->flinfo->fn_oid));
	if (functionNamespaceName == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_SCHEMA),
				 errmsg("could not resolve the vector extension schema")));
	qualifiedTriggerFunction = quote_qualified_identifier(functionNamespaceName,
		"vector_hnsw_fragment_epoch_bump_trigger");
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
			"SELECT "
			"EXISTS (SELECT 1 FROM pg_catalog.pg_trigger AS t "
			"WHERE t.tgrelid = $1 "
			"AND t.tgname = 'pgvector_hnsw_fragment_epoch' "
			"AND NOT t.tgisinternal), "
			"EXISTS (SELECT 1 FROM pg_catalog.pg_trigger AS t "
			"WHERE t.tgrelid = $1 "
			"AND t.tgname = 'pgvector_hnsw_fragment_epoch' "
			"AND NOT t.tgisinternal "
			"AND t.tgenabled IN ('O', 'A') "
			"AND EXISTS (SELECT 1 "
			"FROM pg_catalog.pg_proc AS p "
			"JOIN pg_catalog.pg_depend AS d ON d.objid = p.oid "
			"AND d.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass "
			"AND d.refclassid = 'pg_catalog.pg_extension'::pg_catalog.regclass "
			"AND d.deptype = 'e' "
			"JOIN pg_catalog.pg_extension AS x ON x.oid = d.refobjid "
			"WHERE p.oid = t.tgfoid "
			"AND p.proname = 'vector_hnsw_fragment_epoch_bump_trigger' "
			"AND p.pronamespace = x.extnamespace "
			"AND p.pronargs = 0 "
			"AND p.prorettype = 'pg_catalog.trigger'::pg_catalog.regtype "
			"AND x.extname = 'vector') "
			"AND t.tgnargs = 0 "
			"AND t.tgqual IS NULL "
			"AND t.tgattr::text = '' "
			"AND t.tgtype = 60)",
			1, argTypes, values, nulls, true, 1);
	if (spiStatus != SPI_OK_SELECT)
		elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);
	if (SPI_processed == 1)
	{
		Datum		hasTriggerDatum;
		Datum		validTriggerDatum;
		bool		hasTriggerIsNull;
		bool		validTriggerIsNull;

		hasTriggerDatum = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc, 1, &hasTriggerIsNull);
		validTriggerDatum = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc, 2, &validTriggerIsNull);
		hasTrigger = !hasTriggerIsNull && DatumGetBool(hasTriggerDatum);
		validTrigger = !validTriggerIsNull && DatumGetBool(validTriggerDatum);
	}
	SPI_finish();

	if (!validTrigger)
	{
		if (hasTrigger)
		{
			initStringInfo(&sql);
			appendStringInfo(&sql,
				"DROP TRIGGER pgvector_hnsw_fragment_epoch ON %s",
				qualifiedName);
			spiStatus = SPI_connect();
			if (spiStatus != SPI_OK_CONNECT)
				elog(ERROR, "SPI_connect failed: %d", spiStatus);
			spiStatus = SPI_execute(sql.data, false, 0);
			if (spiStatus != SPI_OK_UTILITY)
				elog(ERROR, "SPI_execute failed: %d", spiStatus);
			SPI_finish();
			pfree(sql.data);
			CommandCounterIncrement();
		}

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "CREATE TRIGGER pgvector_hnsw_fragment_epoch "
						 "AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON %s "
						 "FOR EACH STATEMENT EXECUTE FUNCTION %s()",
						 qualifiedName, qualifiedTriggerFunction);
		spiStatus = SPI_connect();
		if (spiStatus != SPI_OK_CONNECT)
			elog(ERROR, "SPI_connect failed: %d", spiStatus);
		spiStatus = SPI_execute(sql.data, false, 0);
		if (spiStatus != SPI_OK_UTILITY)
			elog(ERROR, "SPI_execute failed: %d", spiStatus);
		SPI_finish();
		pfree(sql.data);
		CommandCounterIncrement();

		/* Writes may have escaped invalidation while the trigger was unsafe. */
		spiStatus = SPI_connect();
		if (spiStatus != SPI_OK_CONNECT)
			elog(ERROR, "SPI_connect failed: %d", spiStatus);
		spiStatus = SPI_execute_with_args(
			"UPDATE public.pgvector_hnsw_fragment_epoch "
			"SET epoch = epoch + 1, updated_at = pg_catalog.now() "
			"WHERE heap_oid = $1 RETURNING epoch",
			1, argTypes, values, nulls, false, 1);
		if (spiStatus != SPI_OK_UPDATE_RETURNING || SPI_processed != 1)
			elog(ERROR, "SPI_execute_with_args failed: %d", spiStatus);
		epoch = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc, 1, &isnull));
		if (isnull)
			elog(ERROR, "fragment epoch is null for relation %u", heapOid);
		SPI_finish();
	}

	CommandCounterIncrement();
	pfree(qualifiedName);
	pfree(qualifiedTriggerFunction);
	PG_RETURN_INT64(epoch);
}

static void
HnswMetadataCurrentCacheVersion(Oid heapOid, bool *tracked, int64 *epoch, Oid *relFileNode)
{
	Relation	heapRel = table_open(heapOid, AccessShareLock);

	if (heapRel->rd_rel->relrowsecurity || heapRel->rd_rel->relforcerowsecurity)
	{
		table_close(heapRel, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("HNSW hard guidance is not supported directly on an RLS relation"),
				 errhint("Keep RLS in PostgreSQL's final recheck and guide only with a row-local superset on a non-RLS vector heap.")));
	}
	table_close(heapRel, AccessShareLock);

	*tracked = HnswMetadataGetRelationVersion(heapOid, epoch, relFileNode);
	if (hnsw_guidance_require_epoch && !*tracked)
		ereport(ERROR,
					(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
					 errmsg("valid fragment epoch tracking is not enabled for relation %u", heapOid),
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

static void
HnswMetadataVerifyBuildVersion(Oid heapOid, HnswMetadataCacheEntry *cache,
							   bool tracked, int64 epoch, Oid relFileNode)
{
	bool		finalTracked;
	int64		finalEpoch;
	Oid			finalRelFileNode;

	finalTracked = HnswMetadataGetRelationVersion(heapOid, &finalEpoch, &finalRelFileNode);
	if (finalTracked != tracked || (tracked && finalEpoch != epoch) ||
		finalRelFileNode != relFileNode)
	{
		HnswMetadataFreeCacheEntry(cache);
		ereport(ERROR,
				(errcode(ERRCODE_T_R_SERIALIZATION_FAILURE),
				 errmsg("relation changed while an HNSW guidance fragment was being built"),
				 errhint("Retry the guidance activation.")));
	}
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
		"AND build_epoch = $4 AND relfilenode = $5 AND format_version = 3",
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
	Oid			argTypes[10] = {OIDOID, TEXTOID, TEXTOID, INT8OID, INT8OID, INT8OID, BYTEAOID, INT8OID, OIDOID, INT4OID};
	Datum		values[10];
	char		nulls[10] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};
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
	values[9] = Int32GetDatum(3);

	spiStatus = SPI_connect();
	if (spiStatus != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed: %d", spiStatus);

	spiStatus = SPI_execute_with_args(
		"INSERT INTO public.pgvector_hnsw_fragment_store "
		"(heap_oid, filter_name, kind, rows, pages, bloom_bit_count, payload, build_epoch, relfilenode, format_version) "
		"VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) "
		"ON CONFLICT (heap_oid, filter_name, kind) DO UPDATE SET "
		"rows = EXCLUDED.rows,"
		"pages = EXCLUDED.pages,"
		"bloom_bit_count = EXCLUDED.bloom_bit_count,"
		"payload = EXCLUDED.payload,"
		"build_epoch = EXCLUDED.build_epoch,"
		"relfilenode = EXCLUDED.relfilenode,"
		"format_version = EXCLUDED.format_version,"
		"built_at = pg_catalog.now() "
		"WHERE pgvector_hnsw_fragment_store.relfilenode <> EXCLUDED.relfilenode "
		"OR pgvector_hnsw_fragment_store.build_epoch <= EXCLUDED.build_epoch",
		10, argTypes, values, nulls, false, 0);
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
		HnswMetadataTidKey tidKey;
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

		tidKey.tid = *ctid;
		hash_search(cache->tidHash, &tidKey, HASH_ENTER, &tidFound);
		if (!tidFound)
			cache->rows++;
	}

	SPI_finish();
	HnswMetadataExpandHotRoots(heapOid, cache, HNSW_GUIDANCE_KIND_EXACT);
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
	bool		populatePageBits;

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
	populatePageBits = cache->pageBits == NULL;

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
		HnswMetadataBloomSet(cache, ctid);
	}

	SPI_finish();
	HnswMetadataExpandHotRoots(heapOid, cache, HNSW_GUIDANCE_KIND_BLOOM);
	if (populatePageBits && cache->pageBits != NULL)
	{
		pfree(cache->pageBits);
		cache->pageBits = NULL;
		cache->pageBitBytes = 0;
		cache->pageRows = 0;
		cache->pages = 0;
	}
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	cache->bloomBuildMs = INSTR_TIME_GET_MILLISEC(elapsed);

	pfree(sql.data);
	return cache;
}

static HnswMetadataCacheEntry *
GetHnswMetadataCache(Oid heapOid, const char *filterName, bool buildIfMissing,
					 bool evictIfNeeded, bool *cacheHit, bool *storeHit)
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
	HnswMetadataVerifyBuildVersion(heapOid, cache, tracked, epoch, relFileNode);
	HnswMetadataStampCacheVersion(cache, tracked, epoch, relFileNode);
	HnswMetadataTouchCache(cache);
	if (evictIfNeeded)
		HnswMetadataEvictIfNeeded(cache);
	return cache;
}

static HnswMetadataCacheEntry *
GetHnswMetadataPageCache(Oid heapOid, const char *filterName, bool buildIfMissing,
						 bool evictIfNeeded, bool *cacheHit, bool *storeHit)
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
		if (evictIfNeeded)
			HnswMetadataEvictIfNeeded(cache);
		return cache;
	}

	cache = BuildHnswMetadataPageCache(heapOid, filterName);
	HnswMetadataVerifyBuildVersion(heapOid, cache, tracked, epoch, relFileNode);
	HnswMetadataStampCacheVersion(cache, tracked, epoch, relFileNode);
	HnswMetadataTouchCache(cache);
	HnswMetadataSaveFragmentStore(heapOid, filterName, HNSW_GUIDANCE_KIND_PAGE, cache);
	if (evictIfNeeded)
		HnswMetadataEvictIfNeeded(cache);
	return cache;
}

static HnswMetadataCacheEntry *
GetHnswMetadataBloomCache(Oid heapOid, const char *filterName, bool buildIfMissing,
						  bool evictIfNeeded, bool *cacheHit, bool *storeHit)
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
		if (evictIfNeeded)
			HnswMetadataEvictIfNeeded(cache);
		return cache;
	}

	cache = BuildHnswMetadataBloomCache(heapOid, filterName);
	HnswMetadataVerifyBuildVersion(heapOid, cache, tracked, epoch, relFileNode);
	HnswMetadataStampCacheVersion(cache, tracked, epoch, relFileNode);
	HnswMetadataTouchCache(cache);
	HnswMetadataSaveFragmentStore(heapOid, filterName, HNSW_GUIDANCE_KIND_BLOOM, cache);
	if (evictIfNeeded)
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
			 errhint("Supported kinds: exact, page, bloom, adaptive.")));
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

#define HNSW_GUIDANCE_MAX_PREDICATE_BYTES (64 * 1024)

typedef struct HnswGuidancePredicateVarContext
{
	bool		valid;
} HnswGuidancePredicateVarContext;

static bool
HnswGuidancePredicateVarWalker(Node *node, HnswGuidancePredicateVarContext *context)
{
	if (node == NULL || !context->valid)
		return false;

	if (IsA(node, Var))
	{
		Var		   *var = (Var *) node;

		if (var->varno != 1 || var->varlevelsup != 0 ||
			!bms_is_empty(var->varnullingrels))
			context->valid = false;
		return false;
	}

	if (IsA(node, Query) || IsA(node, SubLink) || IsA(node, SubPlan) ||
		IsA(node, AlternativeSubPlan))
	{
		context->valid = false;
		return true;
	}

	return expression_tree_walker(node, HnswGuidancePredicateVarWalker, context);
}

static Expr *
HnswGuidanceBuildPredicate(Oid heapOid, Datum *filterDatums, bool *filterNulls,
						   int filterCount, HnswGuidanceKind defaultKind)
{
	StringInfoData dnf;
	StringInfoData sql;
	char	   *qualifiedName;
	const char *ignoredTidColumn;
	List	   *rawStatements;
	RawStmt    *rawStatement;
	Query	   *query;
	RangeTblEntry *rte;
	Expr	   *predicate;
	HnswGuidancePredicateVarContext varContext;
	int			groups = 1;
	int			atoms = 0;
	int			lastGroup = -1;
	bool		groupOpen = false;

	initStringInfo(&dnf);
	appendStringInfoChar(&dnf, '(');
	groupOpen = true;

	for (int i = 0; i < filterCount; i++)
	{
		char	   *filterName;
		const char *predicateSql;
		HnswGuidanceKind ignoredKind;
		bool		negated = false;

		if (filterNulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("guidance atom names cannot be null")));

		filterName = text_to_cstring(DatumGetTextPP(filterDatums[i]));
		if (strcmp(filterName, "|") == 0 || pg_strcasecmp(filterName, "OR") == 0)
		{
			if (atoms == 0 || lastGroup != groups - 1)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("empty HNSW guidance OR group")));
			appendStringInfoString(&dnf, ") OR (");
			groups++;
			groupOpen = true;
			continue;
		}

		if (filterName[0] == '!')
		{
			negated = true;
			filterName++;
		}
		filterName = (char *) HnswGuidanceParseAtomKind(filterName, defaultKind,
													&ignoredKind);
		predicateSql = HnswMetadataPredicateSql(filterName);

		if (lastGroup == groups - 1)
			appendStringInfoString(&dnf, " AND ");
		if (negated)
			appendStringInfo(&dnf, "NOT (%s)", predicateSql);
		else
			appendStringInfo(&dnf, "(%s)", predicateSql);
		atoms++;
		lastGroup = groups - 1;

		if (dnf.len > HNSW_GUIDANCE_MAX_PREDICATE_BYTES)
			return NULL;
	}

	if (atoms == 0 || lastGroup != groups - 1 || !groupOpen)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("empty HNSW guidance OR group")));
	appendStringInfoChar(&dnf, ')');

	qualifiedName = HnswMetadataQualifiedSource(heapOid, &ignoredTidColumn);
	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT 1 FROM %s AS sqlens_target WHERE %s",
					 qualifiedName, dnf.data);
	rawStatements = raw_parser(sql.data, RAW_PARSE_DEFAULT);
	if (list_length(rawStatements) != 1 || !IsA(linitial(rawStatements), RawStmt))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("HNSW guidance predicate did not parse as one statement")));
	rawStatement = (RawStmt *) linitial(rawStatements);
	query = parse_analyze_fixedparams(rawStatement, sql.data, NULL, 0, NULL);

	if (query->commandType != CMD_SELECT || query->hasSubLinks || query->hasAggs ||
		query->hasWindowFuncs || query->hasTargetSRFs || query->jointree == NULL ||
		query->jointree->quals == NULL || list_length(query->rtable) != 1 ||
		list_length(query->jointree->fromlist) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("HNSW guidance requires a row-local immutable predicate"),
				 errdetail("Subqueries, aggregates, window functions, and set-returning expressions are not supported.")));

	rte = (RangeTblEntry *) linitial(query->rtable);
	if (rte->rtekind != RTE_RELATION || rte->relid != heapOid)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("HNSW guidance predicate referenced a non-target relation")));

	predicate = (Expr *) query->jointree->quals;
	varContext.valid = true;
	(void) HnswGuidancePredicateVarWalker((Node *) predicate, &varContext);
	if (!varContext.valid || contain_agg_clause((Node *) predicate) ||
		contain_window_function((Node *) predicate) ||
		expression_returns_set((Node *) predicate) ||
		contain_mutable_functions((Node *) predicate))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("HNSW guidance requires a row-local immutable predicate"),
				 errdetail("Only immutable expressions over columns of relation %u are supported.", heapOid)));

	/* Match the constant-folded representation stored in finished plans. */
	predicate = (Expr *) eval_const_expressions(NULL, (Node *) predicate);
	return (Expr *) copyObject(predicate);
}

static void
HnswGuidancePersistPredicate(HnswActiveGuidance *guidance, Expr *predicate)
{
	MemoryContext oldContext;

	if (predicate == NULL)
		return;

	guidance->predicateContext = AllocSetContextCreate(TopMemoryContext,
													 "HNSW active guidance predicate",
													 ALLOCSET_SMALL_SIZES);
	oldContext = MemoryContextSwitchTo(guidance->predicateContext);
	guidance->predicateExpr = (Expr *) copyObject(predicate);
	MemoryContextSwitchTo(oldContext);
}

static uint64
HnswGuidanceNextGeneration(void)
{
	hnsw_guidance_generation++;
	if (hnsw_guidance_generation == 0)
		hnsw_guidance_generation++;
	return hnsw_guidance_generation;
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
			bool		found;

			hash_search(descriptor->exactTidHash, &source->key, HASH_ENTER, &found);
			if (!found)
				descriptor->exactRows++;
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

static void
HnswGuidanceValidateAdaptiveAtoms(Datum *filterDatums, bool *filterNulls, int filterCount)
{
	int		groups = 1;
	int		atoms = 0;
	int		lastGroup = -1;

	for (int i = 0; i < filterCount; i++)
	{
		char	   *filterName;
		HnswGuidanceKind ignoredKind;

		if (filterNulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("guidance atom names cannot be null")));

		filterName = text_to_cstring(DatumGetTextPP(filterDatums[i]));
		if (strcmp(filterName, "|") == 0 || pg_strcasecmp(filterName, "OR") == 0)
		{
			if (atoms == 0 || lastGroup != groups - 1)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("empty HNSW guidance OR group")));
			groups++;
			continue;
		}

		if (filterName[0] == '!')
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("adaptive HNSW guidance does not support negated atoms"),
					 errhint("Page and Bloom fragments are approximate supersets, so NOT would be unsafe.")));

		filterName = (char *) HnswGuidanceParseAtomKind(filterName,
				HNSW_GUIDANCE_KIND_PAGE, &ignoredKind);
		if (ignoredKind == HNSW_GUIDANCE_KIND_EXACT)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("adaptive HNSW guidance does not admit exact fragments"),
					 errhint("Composed exact in-memory guidance is intentionally disabled for adaptive admission.")));

		/* This validates the row-local predicate without constructing a payload. */
		(void) HnswMetadataPredicateSql(filterName);
		atoms++;
		lastGroup = groups - 1;
	}

	if (atoms == 0 || lastGroup != groups - 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("empty HNSW guidance OR group")));
}

static int
HnswGuidanceActivateAdaptive(Oid indexOid, Oid heapOid, Datum *filterDatums, bool *filterNulls,
							 int filterCount, HnswGuidanceDescriptorEntry *descriptor,
							 HnswGuidanceKind stage, bool epochTracked,
							 int64 relationEpoch, Oid relationRelFileNode,
							 Expr *predicate)
{
	HnswActiveGuidance nextGuidance;
	bool		finalEpochTracked;
	int64		finalRelationEpoch;
	Oid			finalRelationRelFileNode;
	int64		fragmentBytes = 0;
	int64		mostSelectiveRows = 0;

	Assert(stage == HNSW_GUIDANCE_KIND_PAGE || stage == HNSW_GUIDANCE_KIND_BLOOM);
	MemSet(&nextGuidance, 0, sizeof(nextGuidance));
	nextGuidance.kind = stage;
	nextGuidance.indexOid = indexOid;
	nextGuidance.heapOid = heapOid;
	nextGuidance.signatureBytes = descriptor->key.signatureBytes;
	nextGuidance.signatureHash1 = descriptor->key.signatureHash1;
	nextGuidance.signatureHash2 = descriptor->key.signatureHash2;
	nextGuidance.groups = 1;
	nextGuidance.epochTracked = epochTracked;
	nextGuidance.relationEpoch = relationEpoch;
	nextGuidance.relationRelFileNode = relationRelFileNode;
	nextGuidance.adaptive = true;
	nextGuidance.adaptiveDescriptor = descriptor;

	for (int i = 0; i < filterCount; i++)
	{
		char	   *filterName;
		HnswGuidanceKind ignoredKind;
		bool		cacheHit = false;
		bool		storeHit = false;
		HnswMetadataCacheEntry *cache;
		int		atomIndex;

		filterName = text_to_cstring(DatumGetTextPP(filterDatums[i]));
		if (strcmp(filterName, "|") == 0 || pg_strcasecmp(filterName, "OR") == 0)
		{
			nextGuidance.groups++;
			continue;
		}

		filterName = (char *) HnswGuidanceParseAtomKind(filterName,
				HNSW_GUIDANCE_KIND_PAGE, &ignoredKind);
		if (stage == HNSW_GUIDANCE_KIND_PAGE)
			cache = GetHnswMetadataPageCache(heapOid, filterName, true, false,
											 &cacheHit, &storeHit);
		else
			cache = GetHnswMetadataBloomCache(heapOid, filterName, true, false,
											  &cacheHit, &storeHit);

		if (nextGuidance.atoms >= HNSW_GUIDANCE_MAX_ATOMS)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("too many guidance atoms"),
					 errhint("Maximum supported atoms: %d.", HNSW_GUIDANCE_MAX_ATOMS)));

		atomIndex = nextGuidance.atoms++;
		nextGuidance.atom[atomIndex].cache = cache;
		nextGuidance.atom[atomIndex].kind = stage;
		nextGuidance.atom[atomIndex].group = nextGuidance.groups - 1;
		fragmentBytes += HnswMetadataCacheMemoryBytes(cache, stage);
		if (stage == HNSW_GUIDANCE_KIND_BLOOM &&
			(mostSelectiveRows == 0 || cache->bloomRows < mostSelectiveRows))
			mostSelectiveRows = cache->bloomRows;
		nextGuidance.lastCacheRows += stage == HNSW_GUIDANCE_KIND_PAGE ?
			cache->pageRows : cache->bloomRows;
		if (stage == HNSW_GUIDANCE_KIND_PAGE)
			nextGuidance.lastCachePages += cache->pages;
		nextGuidance.lastCacheMemoryBytes += HnswMetadataCacheMemoryBytes(cache, stage);
		nextGuidance.lastBuildMs += (cacheHit || storeHit) ? 0 :
			(stage == HNSW_GUIDANCE_KIND_PAGE ? cache->pageBuildMs : cache->bloomBuildMs);
		if (cacheHit)
			nextGuidance.fragmentCacheHits++;
		else
		{
			nextGuidance.fragmentCacheMisses++;
			if (storeHit)
				nextGuidance.fragmentStoreHits++;
			else
			{
				nextGuidance.fragmentBuilds++;
				if (stage == HNSW_GUIDANCE_KIND_PAGE)
					hnsw_adaptive_profile.pageBuilds++;
				else
					hnsw_adaptive_profile.bloomBuilds++;
			}
		}
	}

	if (fragmentBytes <= 0 || fragmentBytes > HnswAdaptiveFragmentLimitBytes())
	{
		hnsw_adaptive_profile.rejections++;
		descriptor->adaptiveBytes = fragmentBytes;
		if (stage == HNSW_GUIDANCE_KIND_PAGE)
			descriptor->adaptiveState = HNSW_ADAPTIVE_PROBING;
		else
			descriptor->adaptiveState = HNSW_ADAPTIVE_PAGE;
		descriptor->adaptiveRefinePending = false;
		hnsw_last_adaptive_descriptor = descriptor;
		return 0;
	}

	if (stage == HNSW_GUIDANCE_KIND_BLOOM)
	{
		double		averageProbeHeapFetchMs = descriptor->adaptiveCycleProbes > 0 ?
			descriptor->adaptiveProbeHeapFetchMs / descriptor->adaptiveCycleProbes : 0;
		double		averageProbeTotalMs = descriptor->adaptiveCycleProbes > 0 ?
			descriptor->adaptiveProbeTotalMs / descriptor->adaptiveCycleProbes : 0;
		double		estimatedSkipRate = HnswAdaptiveEstimateBloomSkipRate(heapOid,
			mostSelectiveRows, descriptor->adaptivePageSkipRate);
		double		estimatedScore =
			(averageProbeHeapFetchMs > 0 ? averageProbeHeapFetchMs : averageProbeTotalMs) *
			estimatedSkipRate / fragmentBytes;

		if (estimatedScore <= 0 || estimatedScore < hnsw_d3_min_benefit_per_byte)
		{
			hnsw_adaptive_profile.rejections++;
			descriptor->adaptiveState = HNSW_ADAPTIVE_PAGE;
			descriptor->adaptiveRefinePending = false;
			descriptor->adaptiveBenefitPerByte = estimatedScore;
			hnsw_last_adaptive_descriptor = descriptor;
			return 0;
		}
		descriptor->adaptiveBenefitPerByte = estimatedScore;
	}
	else
	{
		/* The first page fragment is the explicitly documented score-free probe
		 * exception. It is still gated by repeated requests and the size cap. */
		descriptor->adaptiveBenefitPerByte = 0;
	}

	HnswMetadataCurrentCacheVersion(heapOid, &finalEpochTracked, &finalRelationEpoch,
								&finalRelationRelFileNode);
	if (finalEpochTracked != nextGuidance.epochTracked ||
		(finalEpochTracked && finalRelationEpoch != nextGuidance.relationEpoch) ||
		finalRelationRelFileNode != nextGuidance.relationRelFileNode)
		ereport(ERROR,
				(errcode(ERRCODE_T_R_SERIALIZATION_FAILURE),
				 errmsg("relation changed while adaptive HNSW guidance was being activated"),
				 errhint("Retry vector_hnsw_guidance_activate().")));

	descriptor->adaptiveState = stage == HNSW_GUIDANCE_KIND_PAGE ?
		HNSW_ADAPTIVE_PAGE : HNSW_ADAPTIVE_BLOOM;
	descriptor->adaptiveRefinePending = false;
	descriptor->adaptiveBytes = fragmentBytes;
	descriptor->adaptiveAdmissions++;
	HnswGuidancePersistPredicate(&nextGuidance, predicate);
	nextGuidance.generation = HnswGuidanceNextGeneration();
	nextGuidance.active = true;
	hnsw_active_guidance = nextGuidance;
	hnsw_last_adaptive_descriptor = descriptor;
	hnsw_adaptive_profile.admissions++;
	hnsw_adaptive_profile.bytes = fragmentBytes;
	hnsw_adaptive_profile.score = descriptor->adaptiveBenefitPerByte;
	HnswMetadataEvictIfNeeded(NULL);
	return nextGuidance.atoms;
}

bool
HnswGuidanceIsActive(void)
{
	return hnsw_active_guidance.active;
}

static Oid
HnswGuidanceScanHeapOid(IndexScanDesc scan)
{
	if (scan == NULL)
		return InvalidOid;
	if (scan->heapRelation != NULL)
		return RelationGetRelid(scan->heapRelation);
	if (scan->indexRelation != NULL && scan->indexRelation->rd_index != NULL)
		return scan->indexRelation->rd_index->indrelid;
	return InvalidOid;
}

static bool
HnswGuidanceRelationHasRowSecurity(Oid heapOid)
{
	HeapTuple	classTuple;
	bool		hasRowSecurity = true;

	classTuple = SearchSysCache1(RELOID, ObjectIdGetDatum(heapOid));
	if (HeapTupleIsValid(classTuple))
	{
		Form_pg_class classForm = (Form_pg_class) GETSTRUCT(classTuple);

		hasRowSecurity = classForm->relrowsecurity || classForm->relforcerowsecurity;
		ReleaseSysCache(classTuple);
	}

	return hasRowSecurity;
}

static bool
HnswGuidanceQueryHasSecurityBarrier(QueryDesc *queryDesc)
{
	ListCell   *cell;

	foreach(cell, queryDesc->plannedstmt->rtable)
	{
		RangeTblEntry *rte = (RangeTblEntry *) lfirst(cell);

		if (rte->securityQuals != NIL ||
			(rte->rtekind == RTE_SUBQUERY && rte->security_barrier))
			return true;
	}

	return false;
}

typedef struct HnswGuidanceRegisterContext
{
	QueryDesc  *queryDesc;
	HnswExecutorBindingFrame *frame;
	uint64		frameId;
	uint64		guideGeneration;
	Oid			indexOid;
	Oid			heapOid;
} HnswGuidanceRegisterContext;

static HnswGuidancePlanBinding *
HnswGuidanceFindPlanBinding(PlanState *planState)
{
	for (int frameIndex = hnsw_executor_binding_depth - 1;
		 frameIndex >= 0; frameIndex--)
	{
		HnswGuidancePlanBinding *binding;

		for (binding = hnsw_executor_binding_stack[frameIndex].planBindings;
			 binding != NULL; binding = binding->next)
		{
			if (&binding->indexState->ss.ps == planState)
				return binding;
		}
	}

	return NULL;
}

static TupleTableSlot *
HnswGuidanceExecIndexScan(PlanState *planState)
{
	HnswGuidancePlanBinding *binding = HnswGuidanceFindPlanBinding(planState);
	HnswGuidancePlanBinding *previousBinding = hnsw_executing_plan_binding;
	TupleTableSlot *slot;

	if (binding == NULL || binding->underlyingExecProcNode == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("lost HNSW guidance plan registration")));

	hnsw_executing_plan_binding = binding;
	PG_TRY();
	{
		slot = binding->underlyingExecProcNode(planState);
	}
	PG_CATCH();
	{
		hnsw_executing_plan_binding = previousBinding;
		PG_RE_THROW();
	}
	PG_END_TRY();
	hnsw_executing_plan_binding = previousBinding;

	return slot;
}

static bool
HnswGuidanceRegisterPlanState(PlanState *planState, HnswGuidanceRegisterContext *context)
{
	if (planState == NULL)
		return false;

	if (IsA(planState, IndexScanState))
	{
		IndexScanState *indexState = (IndexScanState *) planState;
		IndexScan  *plan = (IndexScan *) planState->plan;

		if (plan->indexid == context->indexOid &&
			indexState->iss_RelationDesc != NULL &&
			indexState->iss_RelationDesc->rd_indam != NULL &&
			indexState->iss_RelationDesc->rd_indam->amgettuple == hnswgettuple)
		{
			HnswGuidancePlanBinding *binding;
			RangeTblEntry *rte = NULL;
			HnswPlannerProofBypassReason precheckReason = HNSW_PROOF_BYPASS_NONE;

			if (plan->scan.scanrelid > 0 &&
				plan->scan.scanrelid <= list_length(context->queryDesc->plannedstmt->rtable))
				rte = (RangeTblEntry *) list_nth(context->queryDesc->plannedstmt->rtable,
												 plan->scan.scanrelid - 1);

			if (rte == NULL || rte->rtekind != RTE_RELATION ||
				rte->relid != context->heapOid ||
				indexState->ss.ss_currentRelation == NULL ||
				RelationGetRelid(indexState->ss.ss_currentRelation) != context->heapOid ||
				indexState->iss_RelationDesc->rd_index == NULL ||
				indexState->iss_RelationDesc->rd_index->indrelid != context->heapOid)
				precheckReason = HNSW_PROOF_BYPASS_SCAN_IDENTITY;
			else if (context->queryDesc->plannedstmt->parallelModeNeeded ||
				context->queryDesc->estate->es_use_parallel_mode ||
				plan->scan.plan.parallel_aware)
				precheckReason = HNSW_PROOF_BYPASS_PARALLEL;
			else if (HnswGuidanceRelationHasRowSecurity(context->heapOid) ||
				HnswGuidanceQueryHasSecurityBarrier(context->queryDesc))
				precheckReason = HNSW_PROOF_BYPASS_RLS_SECURITY_BARRIER;

			binding = (HnswGuidancePlanBinding *) MemoryContextAllocZero(
				context->queryDesc->estate->es_query_cxt, sizeof(*binding));
			binding->next = context->frame->planBindings;
			binding->queryDesc = context->queryDesc;
			binding->indexState = indexState;
			binding->underlyingExecProcNode = indexState->ss.ps.ExecProcNodeReal;
			binding->plan = plan;
			binding->frameId = context->frameId;
			binding->guideGeneration = context->guideGeneration;
			binding->planNodeId = plan->scan.plan.plan_node_id;
			binding->scanrelid = plan->scan.scanrelid;
			binding->indexOid = context->indexOid;
			binding->heapOid = context->heapOid;
			binding->precheckReason = precheckReason;

			if (binding->underlyingExecProcNode != NULL)
			{
				context->frame->planBindings = binding;
				ExecSetExecProcNode(&indexState->ss.ps, HnswGuidanceExecIndexScan);
			}
		}
	}

	return planstate_tree_walker(planState, HnswGuidanceRegisterPlanState, context);
}

void
HnswGuidanceRegisterExecutorScans(QueryDesc *queryDesc, uint64 frameId)
{
	HnswGuidanceRegisterContext context;
	HnswExecutorBindingFrame *frame;

	if (!hnsw_active_guidance.active || queryDesc == NULL ||
		queryDesc->plannedstmt == NULL || queryDesc->planstate == NULL ||
		queryDesc->estate == NULL)
		return;
	frame = HnswExecutorBindingFindFrame(frameId, queryDesc);
	if (frame == NULL)
		return;

	context.queryDesc = queryDesc;
	context.frame = frame;
	context.frameId = frameId;
	context.guideGeneration = hnsw_active_guidance.generation;
	context.indexOid = hnsw_active_guidance.indexOid;
	context.heapOid = hnsw_active_guidance.heapOid;
	(void) HnswGuidanceRegisterPlanState(queryDesc->planstate, &context);
}

void
HnswGuidanceAttachCurrentPlan(IndexScanDesc scan)
{
	HnswGuidancePlanBinding *binding = hnsw_executing_plan_binding;
	HnswScanOpaque so;

	if (binding == NULL || scan == NULL || scan->opaque == NULL)
		return;
	so = (HnswScanOpaque) scan->opaque;
	so->guidancePlan = binding;

	if (binding->scan != NULL || scan->indexRelation == NULL ||
		RelationGetRelid(scan->indexRelation) != binding->indexOid ||
		HnswGuidanceScanHeapOid(scan) != binding->heapOid)
		return;

	binding->scan = scan;
}

typedef struct HnswGuidanceQualContext
{
	Index		scanrelid;
	ParamListInfo params;
	HnswPlannerProofBypassReason reason;
} HnswGuidanceQualContext;

static Node *
HnswGuidanceNormalizeQual(Node *node, HnswGuidanceQualContext *context)
{
	if (node == NULL || context->reason != HNSW_PROOF_BYPASS_NONE)
		return copyObject(node);

	if (IsA(node, Var))
	{
		Var		   *source = (Var *) node;
		Var		   *var;

		if (source->varno != context->scanrelid || source->varlevelsup != 0 ||
			!bms_is_empty(source->varnullingrels))
		{
			context->reason = HNSW_PROOF_BYPASS_NON_TARGET_VAR;
			return copyObject(node);
		}

		var = (Var *) copyObject(source);
		var->varno = 1;
		var->varnosyn = 1;
		return (Node *) var;
	}

	if (IsA(node, Param))
	{
		Param	   *param = (Param *) node;

		if (param->paramkind == PARAM_EXTERN)
		{
			ParamExternData workspace;
			ParamExternData *parameter;
			int16		typeLength;
			bool		typeByValue;
			Datum		value = (Datum) 0;

			/*
			 * A validation-only guide may bind a stable external parameter.  Hard
			 * traversal pruning cannot: its cached predicate must be equivalent
			 * to the complete scan predicate without execution-time substitution.
			 */
			if (hnsw_filter_strategy == HNSW_FILTER_STRATEGY_TRAVERSAL_GUIDED ||
				context->params == NULL || param->paramid <= 0 ||
				param->paramid > context->params->numParams)
			{
				context->reason = HNSW_PROOF_BYPASS_PARAM_EXTERN;
				return copyObject(node);
			}

			if (context->params->paramFetch != NULL)
				parameter = context->params->paramFetch(context->params,
												 param->paramid, false, &workspace);
			else
				parameter = &context->params->params[param->paramid - 1];

			if (parameter == NULL || !OidIsValid(parameter->ptype) ||
				parameter->ptype != param->paramtype)
			{
				context->reason = HNSW_PROOF_BYPASS_PARAM_EXTERN;
				return copyObject(node);
			}

			get_typlenbyval(param->paramtype, &typeLength, &typeByValue);
			if (!parameter->isnull)
				value = datumCopy(parameter->value, typeByValue, typeLength);
			return (Node *) makeConst(param->paramtype, param->paramtypmod,
									  param->paramcollid, typeLength, value,
									  parameter->isnull, typeByValue);
		}

		context->reason = HNSW_PROOF_BYPASS_PARAM_EXEC;
		return copyObject(node);
	}

	if (IsA(node, SubPlan) || IsA(node, AlternativeSubPlan) || IsA(node, SubLink) ||
		IsA(node, Aggref) || IsA(node, GroupingFunc) || IsA(node, WindowFunc) ||
		IsA(node, CurrentOfExpr) || IsA(node, NextValueExpr))
	{
		context->reason = HNSW_PROOF_BYPASS_UNSUPPORTED_QUAL;
		return copyObject(node);
	}

	return expression_tree_mutator(node, HnswGuidanceNormalizeQual, context);
}

static bool
HnswGuidanceStructuralSubset(List *predicateClauses, List *actualClauses)
{
	ListCell   *predicateCell;

	foreach(predicateCell, predicateClauses)
	{
		Node	   *predicate = (Node *) lfirst(predicateCell);
		ListCell   *actualCell;
		bool		found = false;

		foreach(actualCell, actualClauses)
		{
			if (equal(predicate, lfirst(actualCell)))
			{
				found = true;
				break;
			}
		}

		if (!found)
			return false;
	}

	return true;
}

static bool
HnswGuidanceEstimateSkipRate(HnswActiveGuidance *guide, double totalRows,
							 double *skipRate)
{
	double		matchingUpperBound = 0;

	if (guide == NULL || skipRate == NULL || totalRows <= 0 ||
		guide->atoms <= 0 || guide->groups <= 0)
		return false;

	if (guide->composedExactActive && guide->composedExactTidHash != NULL)
	{
		matchingUpperBound = guide->composedExactRows;
	}
	else
	{
		for (int group = 0; group < guide->groups; group++)
		{
			double		groupUpperBound = totalRows;
			bool		groupHasAtom = false;

			for (int i = 0; i < guide->atoms; i++)
			{
				HnswGuidanceAtom *atom = &guide->atom[i];
				double		atomUpperBound;

				if (atom->group != group)
					continue;
				if (atom->negated || atom->cache == NULL)
					return false;
				if (atom->kind == HNSW_GUIDANCE_KIND_EXACT)
					atomUpperBound = atom->cache->rows;
				else if (atom->kind == HNSW_GUIDANCE_KIND_BLOOM &&
						 atom->cache->bloomBits != NULL &&
						 atom->cache->bloomBitCount > 0)
				{
					/* Seven hashes over ten bits/item.  False positives only
					 * reduce benefit; they cannot remove a qualifying TID. */
					double		items = Min(totalRows,
						(double) atom->cache->bloomRows);
					double		fill = 1.0 - exp(-7.0 * items /
						(double) atom->cache->bloomBitCount);
					double		falsePositiveRate = pow(fill, 7.0);

					atomUpperBound = items +
						(totalRows - items) * falsePositiveRate;
				}
				else
					return false;
				groupHasAtom = true;
				groupUpperBound = Min(groupUpperBound,
										 atomUpperBound);
			}

			if (!groupHasAtom)
				return false;
			matchingUpperBound += groupUpperBound;
		}
	}

	matchingUpperBound = Min(totalRows, Max(0.0, matchingUpperBound));
	*skipRate = Max(0.0, Min(1.0, 1.0 - matchingUpperBound / totalRows));
	return true;
}

static HnswScanGuidance *
HnswGuidanceProofFailure(IndexScanDesc scan, HnswGuidancePlanBinding *binding,
						 HnswPlannerProofOutcome *proof,
						 HnswPlannerProofBypassReason reason)
{
	if (proof != NULL)
	{
		proof->succeeded = false;
		proof->bypassReason = reason;
		if (proof->attempted)
		{
			hnsw_planner_proof_failures++;
			hnsw_binding_scan_bypasses++;
		}
	}
	hnsw_planner_proof_last_reason = reason;
	hnsw_planner_proof_last_plan_node_id = proof != NULL ? proof->planNodeId : 0;
	hnsw_planner_proof_last_index_oid = proof != NULL ? proof->indexOid : InvalidOid;
	hnsw_planner_proof_last_heap_oid = proof != NULL ? proof->heapOid : InvalidOid;
	hnsw_planner_proof_last_generation = proof != NULL ? proof->guideGeneration : 0;
	return NULL;
}

bool
HnswGuidanceIsActiveForHeap(Oid heapOid)
{
	return hnsw_active_guidance.active &&
		hnsw_filter_strategy != HNSW_FILTER_STRATEGY_OFF &&
		hnsw_active_guidance.statementBound &&
		OidIsValid(heapOid) &&
			hnsw_active_guidance.heapOid == heapOid;
}

HnswScanGuidance *
HnswGuidancePrepareForScan(IndexScanDesc scan, void *planBinding,
						   HnswPlannerProofOutcome *proof)
{
	HnswGuidancePlanBinding *binding = (HnswGuidancePlanBinding *) planBinding;
	HnswExecutorBindingFrame *frame;
	HnswGuidanceQualContext qualContext;
	List	   *actualClauses = NIL;
	List	   *predicateClauses;
	ListCell   *cell;
	bool		tracked;
	int64		epoch;
	Oid			relFileNode;
	bool		implied;
	HnswScanGuidance *guidance;

	hnsw_binding_scan_checks++;
	if (proof != NULL)
	{
		MemSet(proof, 0, sizeof(*proof));
		proof->attempted = binding != NULL || hnsw_active_guidance.active;
		proof->bypassReason = HNSW_PROOF_BYPASS_SCAN_NOT_STARTED;
		proof->planNodeId = binding != NULL ? binding->planNodeId : 0;
		proof->indexOid = binding != NULL ? binding->indexOid :
			(scan != NULL && scan->indexRelation != NULL ?
			 RelationGetRelid(scan->indexRelation) : InvalidOid);
		proof->heapOid = binding != NULL ? binding->heapOid :
			(scan != NULL ? HnswGuidanceScanHeapOid(scan) : InvalidOid);
		proof->guideGeneration = binding != NULL ? binding->guideGeneration :
			(hnsw_active_guidance.active ? hnsw_active_guidance.generation : 0);
		if (proof->attempted)
			hnsw_planner_proof_attempts++;
	}

	if (binding == NULL)
		return HnswGuidanceProofFailure(scan, NULL, proof,
			hnsw_active_guidance.active ? HNSW_PROOF_BYPASS_NO_PLAN_REGISTRATION :
			HNSW_PROOF_BYPASS_NO_ACTIVE_GUIDE);
	if (binding->scan != scan || binding->plan == NULL ||
		binding->indexState == NULL || binding->indexState->iss_ScanDesc != scan ||
		binding->indexState->ss.ps.plan != &binding->plan->scan.plan ||
		binding->indexState->ss.ps.state != binding->queryDesc->estate ||
		binding->planNodeId != binding->indexState->ss.ps.plan->plan_node_id ||
		binding->indexOid != RelationGetRelid(scan->indexRelation) ||
		binding->heapOid != HnswGuidanceScanHeapOid(scan))
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_SCAN_IDENTITY);
	if (!hnsw_active_guidance.active)
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_NO_ACTIVE_GUIDE);
	if (hnsw_filter_strategy == HNSW_FILTER_STRATEGY_OFF)
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_STRATEGY_OFF);
	if (binding->guideGeneration != hnsw_active_guidance.generation ||
		binding->indexOid != hnsw_active_guidance.indexOid ||
		binding->heapOid != hnsw_active_guidance.heapOid)
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_LATE_GENERATION);
	if (binding->precheckReason != HNSW_PROOF_BYPASS_NONE)
		return HnswGuidanceProofFailure(scan, binding, proof, binding->precheckReason);
	frame = HnswExecutorBindingFindFrame(binding->frameId, binding->queryDesc);
	if (frame == NULL || !frame->bindingSeen || !frame->bindingMatched)
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_NO_STATEMENT_BINDING);
	if (frame->boundGuideGeneration != binding->guideGeneration ||
		frame->boundIndexOid != binding->indexOid ||
		frame->boundHeapOid != binding->heapOid ||
		frame->boundSignatureBytes != hnsw_active_guidance.signatureBytes ||
		frame->boundSignatureHash1 != hnsw_active_guidance.signatureHash1 ||
		frame->boundSignatureHash2 != hnsw_active_guidance.signatureHash2)
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_BINDING_IDENTITY);
	if (hnsw_active_guidance.predicateExpr == NULL)
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_PREDICATE_UNAVAILABLE);

	tracked = HnswMetadataGetRelationVersion(binding->heapOid, &epoch, &relFileNode);
	if (!tracked || !hnsw_active_guidance.epochTracked ||
		(tracked && epoch != hnsw_active_guidance.relationEpoch) ||
		relFileNode != hnsw_active_guidance.relationRelFileNode)
	{
		/* A stale guide must never remove candidates from a newer table version. */
		if (hnsw_active_guidance.adaptive)
		{
			HnswGuidanceDescriptorEntry *descriptor = hnsw_active_guidance.adaptiveDescriptor;

			for (int i = 0; i < hnsw_active_guidance.atoms; i++)
			{
				HnswMetadataCacheEntry *cache = hnsw_active_guidance.atom[i].cache;

				if (cache != NULL)
					HnswMetadataFreeCacheEntry(cache);
			}
			HnswAdaptiveMarkStale(descriptor);
			hnsw_last_adaptive_descriptor = descriptor;
		}
		HnswGuidanceDeactivate();
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_STALE_RELATION);
	}

	qualContext.scanrelid = binding->scanrelid;
	qualContext.params = binding->queryDesc->params;
	qualContext.reason = HNSW_PROOF_BYPASS_NONE;
	foreach(cell, binding->plan->scan.plan.qual)
	{
		Expr	   *normalized = (Expr *) HnswGuidanceNormalizeQual((Node *) lfirst(cell),
																 &qualContext);

		if (qualContext.reason != HNSW_PROOF_BYPASS_NONE)
			return HnswGuidanceProofFailure(scan, binding, proof, qualContext.reason);
		if (contain_agg_clause((Node *) normalized) ||
			contain_window_function((Node *) normalized) ||
			expression_returns_set((Node *) normalized) ||
			contain_mutable_functions_after_planning(normalized))
			return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_UNSUPPORTED_QUAL);
		normalized = (Expr *) eval_const_expressions(NULL, (Node *) normalized);
		actualClauses = list_concat(actualClauses, make_ands_implicit(normalized));
	}
	foreach(cell, binding->plan->indexqualorig)
	{
		Expr	   *normalized = (Expr *) HnswGuidanceNormalizeQual((Node *) lfirst(cell),
																 &qualContext);

		if (qualContext.reason != HNSW_PROOF_BYPASS_NONE)
			return HnswGuidanceProofFailure(scan, binding, proof, qualContext.reason);
		if (contain_agg_clause((Node *) normalized) ||
			contain_window_function((Node *) normalized) ||
			expression_returns_set((Node *) normalized) ||
			contain_mutable_functions_after_planning(normalized))
			return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_UNSUPPORTED_QUAL);
		normalized = (Expr *) eval_const_expressions(NULL, (Node *) normalized);
		actualClauses = list_concat(actualClauses, make_ands_implicit(normalized));
	}

	if (actualClauses == NIL)
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_NO_ACTUAL_QUALS);
	predicateClauses = make_ands_implicit((Expr *) copyObject(hnsw_active_guidance.predicateExpr));
	implied = predicate_implied_by(predicateClauses, actualClauses, false);
	if (!implied)
		implied = HnswGuidanceStructuralSubset(predicateClauses, actualClauses);
	if (!implied)
		return HnswGuidanceProofFailure(scan, binding, proof, HNSW_PROOF_BYPASS_PREDICATE_NOT_IMPLIED);
	/*
	 * Candidate admission needs actual => guide, not equivalence.  A tuple
	 * outside the guide cannot satisfy the SQL predicate, while its HNSW node
	 * remains on the native expansion frontier.  Excluding it from W can only
	 * defer the stock termination threshold; residual and join quals continue
	 * to run in the PostgreSQL executor.
	 */

	guidance = (HnswScanGuidance *) MemoryContextAllocZero(
		MemoryContextGetParent(((HnswScanOpaque) scan->opaque)->tmpCtx),
		sizeof(*guidance));
	guidance->scan = scan;
	guidance->frameId = binding->frameId;
	guidance->generation = binding->guideGeneration;
	guidance->planNodeId = binding->planNodeId;
	guidance->indexOid = binding->indexOid;
	guidance->heapOid = binding->heapOid;
	guidance->guide = hnsw_active_guidance;
	if (binding->indexState->ss.ss_currentRelation != NULL)
		guidance->estimatedSkipRateValid = HnswGuidanceEstimateSkipRate(
			&guidance->guide,
			binding->indexState->ss.ss_currentRelation->rd_rel->reltuples,
			&guidance->estimatedSkipRate);

	if (proof != NULL)
	{
		proof->attempted = true;
		proof->succeeded = true;
		proof->bypassReason = HNSW_PROOF_BYPASS_NONE;
	}
	hnsw_binding_scan_matches++;
	hnsw_planner_proof_successes++;
	hnsw_planner_proof_last_reason = HNSW_PROOF_BYPASS_NONE;
	hnsw_planner_proof_last_plan_node_id = binding->planNodeId;
	hnsw_planner_proof_last_index_oid = binding->indexOid;
	hnsw_planner_proof_last_heap_oid = binding->heapOid;
	hnsw_planner_proof_last_generation = binding->guideGeneration;
	return guidance;
}

bool
HnswGuidanceIsActiveForScan(HnswScanGuidance *guidance)
{
	return guidance != NULL && guidance->scan != NULL &&
		hnsw_filter_strategy != HNSW_FILTER_STRATEGY_OFF &&
		hnsw_active_guidance.active &&
		hnsw_active_guidance.generation == guidance->generation &&
		hnsw_active_guidance.indexOid == guidance->indexOid &&
		hnsw_active_guidance.heapOid == guidance->heapOid;
}

bool
HnswGuidanceGetEstimatedSkipRate(HnswScanGuidance *guidance, double *skipRate)
{
	if (!HnswGuidanceIsActiveForScan(guidance) ||
		!guidance->estimatedSkipRateValid || skipRate == NULL)
		return false;

	*skipRate = guidance->estimatedSkipRate;
	return true;
}

bool
HnswGuidanceAllowsTid(HnswScanGuidance *guidance, ItemPointer tid)
{
	HnswActiveGuidance *snapshot;

	if (!HnswGuidanceIsActiveForScan(guidance))
		return true;
	snapshot = &guidance->guide;

	if (snapshot->composedExactActive && snapshot->composedExactTidHash != NULL)
	{
		HnswMetadataTidKey tidKey;

		tidKey.tid = *tid;
		return hash_search(snapshot->composedExactTidHash, &tidKey, HASH_FIND, NULL) != NULL;
	}

	for (int group = 0; group < snapshot->groups; group++)
	{
		bool		groupMatches = true;

		for (int i = 0; i < snapshot->atoms; i++)
		{
			HnswGuidanceAtom *atom = &snapshot->atom[i];
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

void
HnswGuidanceEndScan(HnswScanGuidance *guidance)
{
	if (guidance != NULL)
		pfree(guidance);
}

static void
HnswGuidanceDeactivate(void)
{
	MemoryContext predicateContext = hnsw_active_guidance.predicateContext;

	(void) HnswGuidanceNextGeneration();
	MemSet(&hnsw_active_guidance, 0, sizeof(hnsw_active_guidance));
	MemSet(&hnsw_adaptive_probe, 0, sizeof(hnsw_adaptive_probe));
	if (predicateContext != NULL)
		MemoryContextDelete(predicateContext);
	HnswExecutorBindingRefreshCompatibilityFlag();
}

void
HnswGuidanceRecordScan(Oid heapOid, int64 candidates, int64 guidanceChecks,
						int64 guidanceSkips, double heapFetchMs, double totalScanMs)
{
	HnswGuidanceDescriptorEntry *descriptor = NULL;

	if (hnsw_adaptive_probe.descriptor != NULL &&
		hnsw_adaptive_probe.heapOid == heapOid)
	{
		descriptor = hnsw_adaptive_probe.descriptor;

		/* hnswendscan can run after the executor's outer portal has closed, so
		 * this path only writes backend-local observations. Activation validates
		 * the saved epoch/relfilenode before any payload can be reused. */
		descriptor->adaptiveProbes++;
		descriptor->adaptiveCycleProbes++;
		descriptor->adaptiveProbeCandidates += candidates;
		descriptor->adaptiveProbeChecks += guidanceChecks;
		descriptor->adaptiveProbeSkips += guidanceSkips;
		descriptor->adaptiveProbeHeapFetchMs += heapFetchMs;
		descriptor->adaptiveProbeTotalMs += totalScanMs;
		hnsw_adaptive_profile.probes++;
		hnsw_last_adaptive_descriptor = descriptor;
		MemSet(&hnsw_adaptive_probe, 0, sizeof(hnsw_adaptive_probe));
		return;
	}

	if (!hnsw_active_guidance.active || !hnsw_active_guidance.adaptive ||
		hnsw_active_guidance.heapOid != heapOid)
		return;

	descriptor = hnsw_active_guidance.adaptiveDescriptor;
	if (descriptor == NULL)
		return;

	{
		double		skipRate = guidanceChecks > 0 ? (double) guidanceSkips / guidanceChecks : 0;
		double		averageProbeHeapFetchMs = descriptor->adaptiveCycleProbes > 0 ?
			descriptor->adaptiveProbeHeapFetchMs / descriptor->adaptiveCycleProbes : 0;
		double		averageProbeTotalMs = descriptor->adaptiveCycleProbes > 0 ?
			descriptor->adaptiveProbeTotalMs / descriptor->adaptiveCycleProbes : 0;
		double		estimatedBenefitMs =
			(averageProbeHeapFetchMs > 0 ? averageProbeHeapFetchMs : averageProbeTotalMs) * skipRate;
		int64		bytes = 0;

		for (int i = 0; i < hnsw_active_guidance.atoms; i++)
			bytes += HnswMetadataCacheMemoryBytes(hnsw_active_guidance.atom[i].cache,
											 hnsw_active_guidance.atom[i].kind);

		descriptor->adaptiveUses++;
		descriptor->adaptiveBytes = bytes;
		descriptor->adaptiveBenefitPerByte = bytes > 0 ? estimatedBenefitMs / bytes : 0;
		hnsw_adaptive_profile.bytes = bytes;
		hnsw_adaptive_profile.score = descriptor->adaptiveBenefitPerByte;
		hnsw_adaptive_profile.checks += guidanceChecks;
		hnsw_adaptive_profile.skips += guidanceSkips;

		for (int i = 0; i < hnsw_active_guidance.atoms; i++)
		{
			HnswGuidanceAtom *atom = &hnsw_active_guidance.atom[i];
			int64		atomBytes = HnswMetadataCacheMemoryBytes(atom->cache, atom->kind);

			atom->cache->adaptiveManaged = true;
			atom->cache->uses++;
			atom->cache->benefitPerByte = atomBytes > 0 ? estimatedBenefitMs / atomBytes : 0;
			HnswMetadataTouchCache(atom->cache);
		}

		if (descriptor->adaptiveState == HNSW_ADAPTIVE_PAGE)
		{
			descriptor->adaptivePageSkipRate = skipRate;
			if (guidanceChecks > 0 && skipRate < hnsw_d3_page_min_skip_rate)
				descriptor->adaptiveRefinePending = true;
		}
	}

	hnsw_last_adaptive_descriptor = descriptor;
}

PG_FUNCTION_INFO_V1(vector_hnsw_guidance_reset);

PG_FUNCTION_INFO_V1(vector_sqlens_build_id);

Datum
vector_sqlens_build_id(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(SQLENS_BUILD_ID));
}

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
	bool		adaptive = pg_strcasecmp(kindName, "adaptive") == 0;
	HnswGuidanceKind kind = adaptive ? HNSW_GUIDANCE_KIND_PAGE : HnswGuidanceKindFromText(kindName);
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
	Expr	   *guidancePredicate;

	deconstruct_array(filterArray, TEXTOID, -1, false, 'i', &filterDatums, &filterNulls, &filterCount);
	if (filterCount < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("at least one guidance atom is required")));

	/* A failed replacement must leave guidance disabled, never pointing at a
	 * cache entry that stale-version handling may free below. */
	HnswGuidanceDeactivate();
	initStringInfo(&signature);
	appendStringInfo(&signature, "kind=%s", adaptive ? "adaptive" : HnswGuidanceKindName(kind));
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
	guidancePredicate = HnswGuidanceBuildPredicate(heapOid, filterDatums,
												 filterNulls, filterCount, kind);
	HnswMetadataCurrentCacheVersion(heapOid, &epochTracked, &relationEpoch, &relationRelFileNode);
	if (!epochTracked)
		ereport(ERROR,
					(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
					 errmsg("valid fragment epoch tracking is required for HNSW guidance on relation %u",
							heapOid),
					 errhint("Call vector_hnsw_fragment_tracking_enable(%u::regclass) before activating guidance.",
							heapOid)));

	InitHnswGuidanceDescriptors();
	MemSet(&descriptorKey, 0, sizeof(descriptorKey));
	descriptorKey.heapOid = heapOid;
	descriptorKey.signatureBytes = signature.len;
	descriptorKey.signatureHash1 = hash_any_extended((const unsigned char *) signature.data,
											 signature.len, UINT64CONST(0x534c656e735f4433));
	descriptorKey.signatureHash2 = hash_any_extended((const unsigned char *) signature.data,
											 signature.len, UINT64CONST(0xa91763f24b08d5ce));
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

	if (adaptive)
	{
		HnswGuidanceValidateAdaptiveAtoms(filterDatums, filterNulls, filterCount);
		hnsw_adaptive_profile.requests++;
		descriptor->adaptiveRequests++;
		if (!HnswAdaptiveDescriptorVersionMatches(descriptor, epochTracked,
				relationEpoch, relationRelFileNode))
		{
			if (descriptor->adaptiveState != HNSW_ADAPTIVE_MISSING)
				HnswAdaptiveMarkStale(descriptor);
			HnswAdaptiveBeginProbeCycle(descriptor, epochTracked,
				relationEpoch, relationRelFileNode);
		}

		descriptor->adaptiveCycleRequests++;
		hnsw_last_adaptive_descriptor = descriptor;
		if (descriptor->adaptiveCycleRequests <= hnsw_d3_probe_requests)
		{
			hnsw_adaptive_probe.descriptor = descriptor;
			hnsw_adaptive_probe.heapOid = heapOid;
			hnsw_adaptive_probe.epochTracked = epochTracked;
			hnsw_adaptive_probe.epoch = relationEpoch;
			hnsw_adaptive_probe.relFileNode = relationRelFileNode;
			PG_RETURN_INT32(0);
		}

		if (descriptor->adaptiveState == HNSW_ADAPTIVE_PROBING)
			PG_RETURN_INT32(HnswGuidanceActivateAdaptive(indexOid, heapOid, filterDatums,
				filterNulls, filterCount, descriptor, HNSW_GUIDANCE_KIND_PAGE,
				epochTracked, relationEpoch, relationRelFileNode, guidancePredicate));

		if (descriptor->adaptiveState == HNSW_ADAPTIVE_PAGE &&
			descriptor->adaptiveRefinePending)
		{
			hnsw_adaptive_profile.refinements++;
			PG_RETURN_INT32(HnswGuidanceActivateAdaptive(indexOid, heapOid, filterDatums,
				filterNulls, filterCount, descriptor, HNSW_GUIDANCE_KIND_BLOOM,
				epochTracked, relationEpoch, relationRelFileNode, guidancePredicate));
		}

		if (descriptor->adaptiveState == HNSW_ADAPTIVE_PAGE ||
			descriptor->adaptiveState == HNSW_ADAPTIVE_BLOOM)
			PG_RETURN_INT32(HnswGuidanceActivateAdaptive(indexOid, heapOid, filterDatums,
				filterNulls, filterCount, descriptor,
				descriptor->adaptiveState == HNSW_ADAPTIVE_PAGE ?
				HNSW_GUIDANCE_KIND_PAGE : HNSW_GUIDANCE_KIND_BLOOM,
				epochTracked, relationEpoch, relationRelFileNode, guidancePredicate));

		/* STALE is observable until the next request starts its probe cycle. */
		HnswAdaptiveBeginProbeCycle(descriptor, epochTracked, relationEpoch,
			relationRelFileNode);
		hnsw_adaptive_probe.descriptor = descriptor;
		hnsw_adaptive_probe.heapOid = heapOid;
		hnsw_adaptive_probe.epochTracked = epochTracked;
		hnsw_adaptive_probe.epoch = relationEpoch;
		hnsw_adaptive_probe.relFileNode = relationRelFileNode;
		PG_RETURN_INT32(0);
	}

	MemSet(&nextGuidance, 0, sizeof(nextGuidance));
	nextGuidance.kind = kind;
	nextGuidance.indexOid = indexOid;
	nextGuidance.heapOid = heapOid;
	nextGuidance.signatureBytes = descriptorKey.signatureBytes;
	nextGuidance.signatureHash1 = descriptorKey.signatureHash1;
	nextGuidance.signatureHash2 = descriptorKey.signatureHash2;
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
				cache = GetHnswMetadataCache(heapOid, filterName, true, false, &cacheHit, &storeHit);
				nextGuidance.lastCacheRows += cache->rows;
				nextGuidance.lastCacheMemoryBytes += HnswMetadataCacheMemoryBytes(cache, atomKind);
				nextGuidance.lastBuildMs += cacheHit ? 0 : cache->buildMs;
				break;
			case HNSW_GUIDANCE_KIND_PAGE:
				cache = GetHnswMetadataPageCache(heapOid, filterName, true, false, &cacheHit, &storeHit);
				nextGuidance.lastCacheRows += cache->pageRows;
				nextGuidance.lastCachePages += cache->pages;
				nextGuidance.lastCacheMemoryBytes += HnswMetadataCacheMemoryBytes(cache, atomKind);
				nextGuidance.lastBuildMs += (cacheHit || storeHit) ? 0 : cache->pageBuildMs;
				break;
			case HNSW_GUIDANCE_KIND_BLOOM:
				cache = GetHnswMetadataBloomCache(heapOid, filterName, true, false, &cacheHit, &storeHit);
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

	HnswGuidancePersistPredicate(&nextGuidance, guidancePredicate);
	nextGuidance.generation = HnswGuidanceNextGeneration();
	nextGuidance.active = true;
	hnsw_active_guidance = nextGuidance;
	/* Eviction can now protect every atom referenced by the new guide. */
	HnswMetadataEvictIfNeeded(NULL);
	PG_RETURN_INT32(nextGuidance.atoms);
}

PG_FUNCTION_INFO_V1(vector_hnsw_guidance_bind);

static const char *
HnswPlannerProofBypassReasonName(HnswPlannerProofBypassReason reason)
{
	switch (reason)
	{
		case HNSW_PROOF_BYPASS_NONE:
			return "none";
		case HNSW_PROOF_BYPASS_SCAN_NOT_STARTED:
			return "scan_not_started";
		case HNSW_PROOF_BYPASS_NO_PLAN_REGISTRATION:
			return "no_plan_registration";
		case HNSW_PROOF_BYPASS_SCAN_IDENTITY:
			return "scan_identity_mismatch";
		case HNSW_PROOF_BYPASS_NO_ACTIVE_GUIDE:
			return "no_active_guide";
		case HNSW_PROOF_BYPASS_LATE_GENERATION:
			return "late_guide_generation";
		case HNSW_PROOF_BYPASS_NO_STATEMENT_BINDING:
			return "no_statement_binding";
		case HNSW_PROOF_BYPASS_BINDING_IDENTITY:
			return "binding_identity_mismatch";
		case HNSW_PROOF_BYPASS_STRATEGY_OFF:
			return "filter_strategy_off";
		case HNSW_PROOF_BYPASS_PARALLEL:
			return "parallel_plan";
		case HNSW_PROOF_BYPASS_RLS_SECURITY_BARRIER:
			return "rls_or_security_barrier";
		case HNSW_PROOF_BYPASS_STALE_RELATION:
			return "stale_relation";
		case HNSW_PROOF_BYPASS_PREDICATE_UNAVAILABLE:
			return "predicate_unavailable";
		case HNSW_PROOF_BYPASS_NO_ACTUAL_QUALS:
			return "no_actual_quals";
		case HNSW_PROOF_BYPASS_PARAM_EXEC:
			return "param_exec";
		case HNSW_PROOF_BYPASS_PARAM_EXTERN:
			return "param_extern_unresolved";
		case HNSW_PROOF_BYPASS_NON_TARGET_VAR:
			return "non_target_var";
		case HNSW_PROOF_BYPASS_UNSUPPORTED_QUAL:
			return "unsupported_qual";
		case HNSW_PROOF_BYPASS_PREDICATE_NOT_IMPLIED:
			return "predicate_not_implied";
	}

	return "unknown";
}

Datum
vector_hnsw_guidance_bind(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	ArrayType  *filterArray = PG_GETARG_ARRAYTYPE_P(1);
	text	   *kindText = PG_GETARG_TEXT_PP(2);
	char	   *kindName = text_to_cstring(kindText);
	bool		adaptive = pg_strcasecmp(kindName, "adaptive") == 0;
	HnswGuidanceKind kind = adaptive ? HNSW_GUIDANCE_KIND_PAGE : HnswGuidanceKindFromText(kindName);
	Oid			heapOid = IndexGetRelation(indexOid, false);
	Datum	   *filterDatums;
	bool	   *filterNulls;
	int			filterCount;
	StringInfoData signature;
	uint32		signatureBytes;
	uint64		signatureHash1;
	uint64		signatureHash2;
	bool		matched;
	HnswExecutorBindingFrame *frame = hnsw_executor_binding_depth > 0 ?
		&hnsw_executor_binding_stack[hnsw_executor_binding_depth - 1] : NULL;

	deconstruct_array(filterArray, TEXTOID, -1, false, 'i',
					  &filterDatums, &filterNulls, &filterCount);
	if (filterCount < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("at least one guidance atom is required")));

	initStringInfo(&signature);
	appendStringInfo(&signature, "kind=%s",
					 adaptive ? "adaptive" : HnswGuidanceKindName(kind));
	for (int i = 0; i < filterCount; i++)
	{
		if (filterNulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("guidance atom names cannot be null")));
		appendStringInfo(&signature, "|%s",
						 text_to_cstring(DatumGetTextPP(filterDatums[i])));
	}
	signatureBytes = signature.len;
	signatureHash1 = hash_any_extended((const unsigned char *) signature.data,
										signature.len, UINT64CONST(0x534c656e735f4433));
	signatureHash2 = hash_any_extended((const unsigned char *) signature.data,
										signature.len, UINT64CONST(0xa91763f24b08d5ce));

	hnsw_binding_attempts++;
	matched = hnsw_active_guidance.active &&
		hnsw_active_guidance.indexOid == indexOid &&
		hnsw_active_guidance.heapOid == heapOid &&
		hnsw_active_guidance.signatureBytes == signatureBytes &&
		hnsw_active_guidance.signatureHash1 == signatureHash1 &&
		hnsw_active_guidance.signatureHash2 == signatureHash2;
	hnsw_active_guidance.bindingAttempts++;
	if (frame != NULL)
	{
		frame->bindingSeen = true;
		if (matched)
		{
			frame->bindingMatched = true;
			frame->boundGuideGeneration = hnsw_active_guidance.generation;
			frame->boundIndexOid = indexOid;
			frame->boundHeapOid = heapOid;
			frame->boundSignatureBytes = signatureBytes;
			frame->boundSignatureHash1 = signatureHash1;
			frame->boundSignatureHash2 = signatureHash2;
		}
	}
	if (matched)
	{
		hnsw_binding_matches++;
		hnsw_active_guidance.bindingMatches++;
	}
	else
	{
		hnsw_binding_mismatches++;
		hnsw_active_guidance.bindingMismatches++;
	}
	HnswExecutorBindingRefreshCompatibilityFlag();

	/* This is a correctness marker, not a SQL filter. A mismatch deliberately
	 * fails open to stock HNSW instead of changing query results. */
	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(vector_hnsw_guidance_profile);
Datum
vector_hnsw_guidance_profile(PG_FUNCTION_ARGS)
{
	StringInfoData output;
	HnswGuidanceDescriptorEntry *adaptiveDescriptor = hnsw_active_guidance.adaptive ?
		hnsw_active_guidance.adaptiveDescriptor : hnsw_last_adaptive_descriptor;
	const char *adaptiveState = adaptiveDescriptor != NULL ?
		HnswAdaptiveStateName(adaptiveDescriptor->adaptiveState) : "missing";

	initStringInfo(&output);
	appendStringInfo(&output,
						 "{\"active\":%s,"
						 "\"statement_bound\":%s,"
						 "\"effective_active\":%s,"
						 "\"binding_attempts\":" INT64_FORMAT ","
						 "\"binding_matches\":" INT64_FORMAT ","
						 "\"binding_mismatches\":" INT64_FORMAT ","
						 "\"binding_scan_checks\":" INT64_FORMAT ","
						 "\"binding_scan_matches\":" INT64_FORMAT ","
						 "\"binding_scan_bypasses\":" INT64_FORMAT ","
						 "\"planner_proof_attempts\":" INT64_FORMAT ","
						 "\"planner_proof_successes\":" INT64_FORMAT ","
						 "\"planner_proof_failures\":" INT64_FORMAT ","
						 "\"planner_proof_bypass_reason\":\"%s\","
						 "\"planner_proof_last_plan_node_id\":%d,"
						 "\"planner_proof_last_index_oid\":%u,"
						 "\"planner_proof_last_heap_oid\":%u,"
						 "\"planner_proof_last_generation\":" INT64_FORMAT ","
					 "\"kind\":\"%s\","
						 "\"index_oid\":%u,"
						 "\"heap_oid\":%u,"
						 "\"guide_generation\":" INT64_FORMAT ","
						 "\"predicate_available\":%s,"
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
						 "\"composed_exact_build_ms\":%.6f,"
						 "\"adaptive_state\":\"%s\","
						 "\"adaptive_requests\":" INT64_FORMAT ","
						 "\"adaptive_probes\":" INT64_FORMAT ","
						 "\"adaptive_admissions\":" INT64_FORMAT ","
						 "\"adaptive_rejections\":" INT64_FORMAT ","
						 "\"adaptive_page_builds\":" INT64_FORMAT ","
						 "\"adaptive_bloom_builds\":" INT64_FORMAT ","
						 "\"adaptive_refinements\":" INT64_FORMAT ","
						 "\"adaptive_stale_bypasses\":" INT64_FORMAT ","
						 "\"adaptive_evictions\":" INT64_FORMAT ","
						 "\"adaptive_bytes\":" INT64_FORMAT ","
						 "\"adaptive_score\":%.12g,"
						 "\"adaptive_checks\":" INT64_FORMAT ","
						 "\"adaptive_skips\":" INT64_FORMAT ","
						 "\"adaptive_uses\":" INT64_FORMAT ","
						 "\"adaptive_refine_pending\":%s}",
						 hnsw_active_guidance.active ? "true" : "false",
						 hnsw_active_guidance.statementBound ? "true" : "false",
						 HnswGuidanceIsActiveForHeap(hnsw_active_guidance.heapOid) ? "true" : "false",
						 hnsw_binding_attempts,
						 hnsw_binding_matches,
						 hnsw_binding_mismatches,
						 hnsw_binding_scan_checks,
						 hnsw_binding_scan_matches,
						 hnsw_binding_scan_bypasses,
						 hnsw_planner_proof_attempts,
						 hnsw_planner_proof_successes,
						 hnsw_planner_proof_failures,
						 HnswPlannerProofBypassReasonName(hnsw_planner_proof_last_reason),
						 hnsw_planner_proof_last_plan_node_id,
						 hnsw_planner_proof_last_index_oid,
						 hnsw_planner_proof_last_heap_oid,
						 (int64) hnsw_planner_proof_last_generation,
						 HnswGuidanceKindName(hnsw_active_guidance.kind),
						 hnsw_active_guidance.indexOid,
						 hnsw_active_guidance.heapOid,
						 (int64) hnsw_active_guidance.generation,
						 hnsw_active_guidance.predicateExpr != NULL ? "true" : "false",
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
						 hnsw_active_guidance.composedExactBuildMs,
						 adaptiveState,
						 hnsw_adaptive_profile.requests,
						 hnsw_adaptive_profile.probes,
						 hnsw_adaptive_profile.admissions,
						 hnsw_adaptive_profile.rejections,
						 hnsw_adaptive_profile.pageBuilds,
						 hnsw_adaptive_profile.bloomBuilds,
						 hnsw_adaptive_profile.refinements,
						 hnsw_adaptive_profile.staleBypasses,
						 hnsw_adaptive_profile.evictions,
						 adaptiveDescriptor != NULL ? adaptiveDescriptor->adaptiveBytes : 0,
						 adaptiveDescriptor != NULL ? adaptiveDescriptor->adaptiveBenefitPerByte : 0,
						 hnsw_adaptive_profile.checks,
						 hnsw_adaptive_profile.skips,
						 adaptiveDescriptor != NULL ? (int64) adaptiveDescriptor->adaptiveUses : 0,
						 adaptiveDescriptor != NULL && adaptiveDescriptor->adaptiveRefinePending ? "true" : "false");

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
	cache = GetHnswMetadataCache(heapOid, filterName, true, true, &cacheHit, NULL);
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
	cache = GetHnswMetadataPageCache(heapOid, filterName, true, true, &cacheHit, NULL);
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
	cache = GetHnswMetadataBloomCache(heapOid, filterName, true, true, &cacheHit, NULL);
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
	int64		adaptiveCacheEntries;
	int64		adaptiveCacheBytes;
	int64		adaptiveCacheUses;
	double		adaptiveCacheScore;
	int64		budgetBytes = (int64) hnsw_metadata_cache_max_mb * 1024L * 1024L;
	StringInfoData output;

	HnswMetadataCacheStats(&entries, &residentEntries, &residentBytes, &largestEntryBytes);
	HnswMetadataAdaptiveCacheStats(&adaptiveCacheEntries, &adaptiveCacheBytes,
		&adaptiveCacheUses, &adaptiveCacheScore);
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
						 "\"adaptive_cache_entries\":" INT64_FORMAT ","
						 "\"adaptive_bytes\":" INT64_FORMAT ","
						 "\"adaptive_uses\":" INT64_FORMAT ","
						 "\"adaptive_score\":%.12g,"
						 "\"adaptive_requests\":" INT64_FORMAT ","
						 "\"adaptive_probes\":" INT64_FORMAT ","
						 "\"adaptive_admissions\":" INT64_FORMAT ","
						 "\"adaptive_rejections\":" INT64_FORMAT ","
						 "\"adaptive_page_builds\":" INT64_FORMAT ","
						 "\"adaptive_bloom_builds\":" INT64_FORMAT ","
						 "\"adaptive_refinements\":" INT64_FORMAT ","
						 "\"adaptive_stale_bypasses\":" INT64_FORMAT ","
						 "\"adaptive_evictions\":" INT64_FORMAT ","
						 "\"adaptive_checks\":" INT64_FORMAT ","
						 "\"adaptive_skips\":" INT64_FORMAT ","
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
						 adaptiveCacheEntries,
						 adaptiveCacheBytes,
						 adaptiveCacheUses,
						 adaptiveCacheScore,
						 hnsw_adaptive_profile.requests,
						 hnsw_adaptive_profile.probes,
						 hnsw_adaptive_profile.admissions,
						 hnsw_adaptive_profile.rejections,
						 hnsw_adaptive_profile.pageBuilds,
						 hnsw_adaptive_profile.bloomBuilds,
						 hnsw_adaptive_profile.refinements,
						 hnsw_adaptive_profile.staleBypasses,
						 hnsw_adaptive_profile.evictions,
						 hnsw_adaptive_profile.checks,
						 hnsw_adaptive_profile.skips,
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

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("vector_hnsw_metadata_filter_search() is retired"),
			 errdetail("The legacy function returned cached IDs without PostgreSQL heap, MVCC, and predicate rechecks."),
			 errhint("Use a normal SELECT ... WHERE ... ORDER BY vector_distance query with validation-only safe_guided or planner-proven traversal_guided.")));

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
	cache = GetHnswMetadataCache(heapOid, filterName, true, true, &cacheHit, NULL);

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
		/* The retired function errors before this unreachable output path. */
		values[1] = Int64GetDatum(0);
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
	cache = GetHnswMetadataPageCache(heapOid, filterName, true, true, &cacheHit, NULL);

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
	cache = GetHnswMetadataBloomCache(heapOid, filterName, true, true, &cacheHit, NULL);

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
	cache = GetHnswMetadataBloomCache(heapOid, filterName, true, true, &cacheHit, NULL);

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

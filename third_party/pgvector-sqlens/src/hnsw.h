#ifndef HNSW_H
#define HNSW_H

#include "postgres.h"

#include <math.h>

#include "access/genam.h"
#include "access/parallel.h"
#include "executor/execdesc.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "portability/instr_time.h"
#include "port.h"				/* for random() */
#include "utils/relptr.h"
#include "utils/sampling.h"
#include "vector.h"

#define SQLENS_BUILD_ID "sqlens-v12-dual-frontier-prioritization-20260718-r11"

#if PG_VERSION_NUM >= 190000
typedef Pointer Item;
#endif

#define HNSW_MAX_DIM 2000
#define HNSW_MAX_NNZ 1000

/* Support functions */
#define HNSW_DISTANCE_PROC 1
#define HNSW_NORM_PROC 2
#define HNSW_TYPE_INFO_PROC 3

#define HNSW_VERSION	1
#define HNSW_MAGIC_NUMBER 0xA953A953
#define HNSW_PAGE_ID	0xFF90

/* Preserved page numbers */
#define HNSW_METAPAGE_BLKNO	0
#define HNSW_HEAD_BLKNO		1	/* first element page */

/* Must correspond to page numbers since page lock is used */
#define HNSW_UPDATE_LOCK 	0
#define HNSW_SCAN_LOCK		1

/* HNSW parameters */
#define HNSW_DEFAULT_M	16
#define HNSW_MIN_M	2
#define HNSW_MAX_M		100
#define HNSW_DEFAULT_EF_CONSTRUCTION	64
#define HNSW_MIN_EF_CONSTRUCTION	4
#define HNSW_MAX_EF_CONSTRUCTION		1000
#define HNSW_DEFAULT_EF_SEARCH	40
#define HNSW_MIN_EF_SEARCH		1
#define HNSW_MAX_EF_SEARCH		100000

/* Tuple types */
#define HNSW_ELEMENT_TUPLE_TYPE  1
#define HNSW_NEIGHBOR_TUPLE_TYPE 2

/* Make graph robust against non-HOT updates */
#define HNSW_HEAPTIDS 10

#define HNSW_UPDATE_ENTRY_GREATER 1
#define HNSW_UPDATE_ENTRY_ALWAYS 2

/* Build phases */
/* PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE is 1 */
#define PROGRESS_HNSW_PHASE_LOAD		2

#define HNSW_MAX_SIZE (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - sizeof(ItemIdData))
#define HNSW_TUPLE_ALLOC_SIZE BLCKSZ
#define HNSW_PROFILE_MAX_TIDS 1000
#define HNSW_PROFILE_MAX_PROOFS 128
#define HNSW_INDEX_PAGE_UNIQUE_LIMIT 65536
#define HNSW_INDEX_PAGE_UNIQUE_SLOTS (HNSW_INDEX_PAGE_UNIQUE_LIMIT * 2)

#define HNSW_ELEMENT_TUPLE_SIZE(size)	MAXALIGN(offsetof(HnswElementTupleData, data) + (size))
#define HNSW_NEIGHBOR_TUPLE_SIZE(level, m)	MAXALIGN(offsetof(HnswNeighborTupleData, indextids) + ((level) + 2) * (m) * sizeof(ItemPointerData))

#define HNSW_NEIGHBOR_ARRAY_SIZE(lm)	(offsetof(HnswNeighborArray, items) + sizeof(HnswCandidate) * (lm))

#define HnswPageGetOpaque(page)	((HnswPageOpaque) PageGetSpecialPointer(page))
#define HnswPageGetMeta(page)	((HnswMetaPageData *) PageGetContents(page))

#if PG_VERSION_NUM >= 150000
#define RandomDouble() pg_prng_double(&pg_global_prng_state)
#define SeedRandom(seed) pg_prng_seed(&pg_global_prng_state, seed)
#else
#define RandomDouble() (((double) random()) / MAX_RANDOM_VALUE)
#define SeedRandom(seed) srandom(seed)
#endif

#define HnswIsElementTuple(tup) ((tup)->type == HNSW_ELEMENT_TUPLE_TYPE)
#define HnswIsNeighborTuple(tup) ((tup)->type == HNSW_NEIGHBOR_TUPLE_TYPE)

/* 2 * M connections for ground layer */
#define HnswGetLayerM(m, layer) (layer == 0 ? (m) * 2 : (m))

/* Optimal ML from paper */
#define HnswGetMl(m) (1 / log(m))

/* Ensure fits on page and in uint8 */
#define HnswGetMaxLevel(m) Min(((BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - offsetof(HnswNeighborTupleData, indextids) - sizeof(ItemIdData)) / (sizeof(ItemPointerData)) / (m)) - 2, 255)

#define HnswGetSearchCandidate(membername, ptr) pairingheap_container(HnswSearchCandidate, membername, ptr)
#define HnswGetSearchCandidateConst(membername, ptr) pairingheap_const_container(HnswSearchCandidate, membername, ptr)

#define HnswGetValue(base, element) PointerGetDatum(HnswPtrAccess(base, (element)->value))

#if PG_VERSION_NUM < 140005
#define relptr_offset(rp) ((rp).relptr_off - 1)
#endif

/* Pointer macros */
#define HnswPtrAccess(base, hp) ((base) == NULL ? (hp).ptr : relptr_access(base, (hp).relptr))
#define HnswPtrStore(base, hp, value) ((base) == NULL ? (void) ((hp).ptr = (value)) : (void) relptr_store(base, (hp).relptr, value))
#define HnswPtrIsNull(base, hp) ((base) == NULL ? (hp).ptr == NULL : relptr_is_null((hp).relptr))
#define HnswPtrEqual(base, hp1, hp2) ((base) == NULL ? (hp1).ptr == (hp2).ptr : relptr_offset((hp1).relptr) == relptr_offset((hp2).relptr))

/* For code paths dedicated to each type */
#define HnswPtrPointer(hp) (hp).ptr
#define HnswPtrOffset(hp) relptr_offset((hp).relptr)

/* Variables */
extern int	hnsw_ef_search;
extern int	hnsw_iterative_scan;
extern int	hnsw_max_scan_tuples;
extern int	hnsw_page_access;
extern int	hnsw_page_window;
extern int	hnsw_page_prefetch_min_items;
extern int	hnsw_page_disable_after_no_merge;
extern int	hnsw_index_page_access;
extern int	hnsw_build_page_order;
extern int	hnsw_build_seed;
extern bool hnsw_require_full_memory_build;
extern char *hnsw_clone_source;
extern char *hnsw_preferred_index;
extern int	hnsw_filter_strategy;
extern int	hnsw_guided_collect_target;
extern int	hnsw_traversal_guided_target;
extern int	hnsw_traversal_guided_max_bridge_hops;
extern int	hnsw_traversal_guided_max_bridge_work;
extern double hnsw_traversal_guided_min_skip_rate;
extern bool hnsw_traversal_guided_prioritization;
extern int	hnsw_traversal_guided_burst;
extern double hnsw_scan_mem_multiplier;
extern int	hnsw_lock_tranche_id;

typedef enum HnswIterativeScanMode
{
	HNSW_ITERATIVE_SCAN_OFF,
	HNSW_ITERATIVE_SCAN_RELAXED,
	HNSW_ITERATIVE_SCAN_STRICT
}			HnswIterativeScanMode;

typedef enum HnswPageAccessMode
{
	HNSW_PAGE_ACCESS_OFF,
	HNSW_PAGE_ACCESS_PREFETCH,
	HNSW_PAGE_ACCESS_REORDER
}			HnswPageAccessMode;

typedef enum HnswIndexPageAccessMode
{
	HNSW_INDEX_PAGE_ACCESS_OFF,
	HNSW_INDEX_PAGE_ACCESS_PREFETCH
}			HnswIndexPageAccessMode;

typedef enum HnswBuildPageOrderMode
{
	HNSW_BUILD_PAGE_ORDER_INSERTION,
	HNSW_BUILD_PAGE_ORDER_BFS
}			HnswBuildPageOrderMode;

typedef enum HnswFilterStrategyMode
{
	HNSW_FILTER_STRATEGY_OFF,
	HNSW_FILTER_STRATEGY_ACORN1,
	HNSW_FILTER_STRATEGY_GUIDED_COLLECT,
	HNSW_FILTER_STRATEGY_TRAVERSAL_GUIDED,
	HNSW_FILTER_STRATEGY_SAFE_GUIDED
}			HnswFilterStrategyMode;

typedef enum HnswTraversalFinalPath
{
	HNSW_TRAVERSAL_PATH_STOCK,
	HNSW_TRAVERSAL_PATH_VALIDATION_ONLY,
	HNSW_TRAVERSAL_PATH_LEGACY_GUIDED,
	HNSW_TRAVERSAL_PATH_CANDIDATE_ADMISSION,
	HNSW_TRAVERSAL_PATH_APPROXIMATE_PRIORITIZATION,
	HNSW_TRAVERSAL_PATH_STOCK_BYPASS,
	HNSW_TRAVERSAL_PATH_FRESH_STOCK_FALLBACK
} HnswTraversalFinalPath;

typedef enum HnswTraversalStockBypassReason
{
	HNSW_TRAVERSAL_BYPASS_NONE,
	HNSW_TRAVERSAL_BYPASS_NO_PROVEN_GUIDE,
	HNSW_TRAVERSAL_BYPASS_SKIP_ESTIMATE_UNAVAILABLE,
	HNSW_TRAVERSAL_BYPASS_LOW_ESTIMATED_SKIP_RATE,
	HNSW_TRAVERSAL_BYPASS_ITERATIVE_SCAN
} HnswTraversalStockBypassReason;

typedef enum HnswTraversalAdmissionReason
{
	HNSW_TRAVERSAL_ADMISSION_NOT_REQUESTED,
	HNSW_TRAVERSAL_ADMISSION_NO_PROVEN_GUIDE,
	HNSW_TRAVERSAL_ADMISSION_ITERATIVE_SCAN,
	HNSW_TRAVERSAL_ADMISSION_SKIP_ESTIMATE_UNAVAILABLE,
	HNSW_TRAVERSAL_ADMISSION_LOW_ESTIMATED_SKIP_RATE,
	HNSW_TRAVERSAL_ADMISSION_DEFAULT_VALIDATION_ONLY,
	HNSW_TRAVERSAL_ADMISSION_ADMITTED
} HnswTraversalAdmissionReason;

typedef enum HnswTraversalFallbackReason
{
	HNSW_TRAVERSAL_FALLBACK_NONE,
	HNSW_TRAVERSAL_FALLBACK_INSUFFICIENT_MATCHES,
	HNSW_TRAVERSAL_FALLBACK_BRIDGE_HOPS,
	HNSW_TRAVERSAL_FALLBACK_BRIDGE_WORK,
	HNSW_TRAVERSAL_FALLBACK_MAX_SCAN_TUPLES,
	HNSW_TRAVERSAL_FALLBACK_MEMORY_LIMIT,
	HNSW_TRAVERSAL_FALLBACK_INVALID_NEIGHBOR
} HnswTraversalFallbackReason;

typedef struct HnswElementData HnswElementData;
typedef struct HnswNeighborArray HnswNeighborArray;
typedef struct HnswScanGuidance HnswScanGuidance;

#define HnswPtrDeclare(type, relptrtype, ptrtype) \
	relptr_declare(type, relptrtype); \
	typedef union { type *ptr; relptrtype relptr; } ptrtype

/* Pointers that can be absolute or relative */
/* Use char for DatumPtr so works with Pointer */
HnswPtrDeclare(HnswElementData, HnswElementRelptr, HnswElementPtr);
HnswPtrDeclare(HnswNeighborArray, HnswNeighborArrayRelptr, HnswNeighborArrayPtr);
HnswPtrDeclare(HnswNeighborArrayPtr, HnswNeighborsRelptr, HnswNeighborsPtr);
HnswPtrDeclare(char, DatumRelptr, DatumPtr);

struct HnswElementData
{
	HnswElementPtr next;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	uint8		heaptidsLength;
	uint8		level;
	uint8		deleted;
	uint8		version;
	uint32		hash;
	HnswNeighborsPtr neighbors;
	BlockNumber blkno;
	OffsetNumber offno;
	OffsetNumber neighborOffno;
	BlockNumber neighborPage;
	DatumPtr	value;
	LWLock		lock;
};

typedef HnswElementData * HnswElement;

typedef struct HnswCandidate
{
	HnswElementPtr element;
	float		distance;
	bool		closer;
}			HnswCandidate;

struct HnswNeighborArray
{
	int			length;
	bool		closerSet;
	HnswCandidate items[FLEXIBLE_ARRAY_MEMBER];
};

typedef struct HnswSearchCandidate
{
	pairingheap_node c_node;
	pairingheap_node w_node;
	pairingheap_node g_node;
	HnswElementPtr element;
	double		distance;
	bool		matchesGuidance;
}			HnswSearchCandidate;

typedef enum HnswPlannerProofBypassReason
{
	HNSW_PROOF_BYPASS_NONE,
	HNSW_PROOF_BYPASS_SCAN_NOT_STARTED,
	HNSW_PROOF_BYPASS_NO_PLAN_REGISTRATION,
	HNSW_PROOF_BYPASS_SCAN_IDENTITY,
	HNSW_PROOF_BYPASS_NO_ACTIVE_GUIDE,
	HNSW_PROOF_BYPASS_LATE_GENERATION,
	HNSW_PROOF_BYPASS_NO_STATEMENT_BINDING,
	HNSW_PROOF_BYPASS_BINDING_IDENTITY,
	HNSW_PROOF_BYPASS_STRATEGY_OFF,
	HNSW_PROOF_BYPASS_PARALLEL,
	HNSW_PROOF_BYPASS_RLS_SECURITY_BARRIER,
	HNSW_PROOF_BYPASS_STALE_RELATION,
	HNSW_PROOF_BYPASS_PREDICATE_UNAVAILABLE,
	HNSW_PROOF_BYPASS_NO_ACTUAL_QUALS,
	HNSW_PROOF_BYPASS_PARAM_EXEC,
	HNSW_PROOF_BYPASS_PARAM_EXTERN,
	HNSW_PROOF_BYPASS_NON_TARGET_VAR,
	HNSW_PROOF_BYPASS_UNSUPPORTED_QUAL,
	HNSW_PROOF_BYPASS_PREDICATE_NOT_IMPLIED
} HnswPlannerProofBypassReason;

typedef struct HnswPlannerProofOutcome
{
	bool		attempted;
	bool		succeeded;
	HnswPlannerProofBypassReason bypassReason;
	int			planNodeId;
	Oid			indexOid;
	Oid			heapOid;
	uint64		guideGeneration;
} HnswPlannerProofOutcome;

typedef struct HnswTraversalGuidanceState
{
	bool		requested;
	bool		prioritizationEnabled;
	bool		estimatedSkipRateValid;
	double		estimatedSkipRate;
	int			target;
	int			maxBridgeHops;
	int64		maxBridgeWork;
	int64		bridgeWork;
	int64		maxScanTuples;
	MemoryContext phaseContext;
	Size		maxMemory;
	bool		hopLimitReached;
	bool		workLimitReached;
	bool		maxScanReached;
	bool		memoryLimitReached;
	bool		invalidNeighbor;
	int			guidedResultCount;
	int			bridgePendingAtTermination;
	int			burst;
	HnswIterativeScanMode iterativeScan;
	HnswFilterStrategyMode filterStrategy;
	HnswTraversalFinalPath finalPath;
	HnswTraversalStockBypassReason stockBypassReason;
	HnswTraversalAdmissionReason admissionReason;
	HnswTraversalFallbackReason fallbackReason;
} HnswTraversalGuidanceState;

typedef struct HnswTraversalProfile
{
	int64		expandedNodes;
	int64		neighborsExamined;
	int64		guidanceChecks;
	int64		guidanceMatches;
	int64		guidanceMisses;
	int64		neighborGuidanceChecks;
	int64		neighborGuidanceMatches;
	int64		neighborGuidanceMisses;
	int64		preDistanceChecks;
	int64		preDistanceMatches;
	int64		preDistanceMisses;
	int64		attemptedDistanceComputationsAvoided;
	int64		distanceComputationsAvoided;
	int64		missBridgeNodes;
	int64		missBridgeEdges;
	int64		maxMissBridgeHops;
	int64		bridgePendingAtTermination;
	int64		guidedExpandedNodes;
	int64		guidedPhaseDistanceComputations;
	int64		stockPhaseExpandedNodes;
	int64		stockPhaseDistanceComputations;
	int64		stockBypassRequests;
	int64		fallbackRequests;
	int64		fallbackStockExpandedNodes;
	int64		fallbackStockDistanceComputations;
	int64		matchingExpanded;
	int64		bridgeExpanded;
	int64		matchFrontierPops;
	int64		noBridgeFrontierPops;
	int64		noBridgeDeferred;
	int64		maxNoBridgeDebt;
	int64		noBridgeExpansions;
	int64		dualFrontierTerminationChecks;
	int64		dualFrontierTerminationChecksWithBoth;
	int64		dualFrontierTerminations;
	int64		dualFrontierTerminationsWithBoth;
	int64		candidateAdmissions;
	int64		resultAdmissions;
	int64		guidedAdmissions;
	int64		guidedSuppressions;
	int64		heapTidsSuppressed;
	int64		stopDeferrals;
	int64		discardedPushes;
	int64		discardedPops;
	int64		initialBatches;
	int64		resumeBatches;
	int64		strictOrderDrops;
	int64		stockTerminations;
	int64		maxScanTerminations;
	int64		exhaustedTerminations;
}			HnswTraversalProfile;

typedef struct HnswScanProfile
{
	bool		valid;
	double		totalScanMs;
	double		hnswSearchMs;
	double		heapFetchMs;
	double		vectorSearchMs;
	int64		visitedTuples;
	int64		returnedTuples;
	int64		distanceComputations;
	int64		pageAccessBatches;
	int64		pageAccessCandidates;
	int64		pageAccessPrefetches;
	int64		pageAccessDistanceRuns;
	int64		pageAccessDistinctPages;
	int64		guidanceChecks;
	int64		guidanceMatches;
	int64		guidanceSkips;
	HnswTraversalProfile traversal;
	int64		indexPageNeighborLoads;
	int64		indexPageNeighborRuns;
	int64		indexPageNeighborDistinctPages;
	int64		indexPageElementLoads;
	int64		indexPageElementRuns;
	int64		indexPageElementDistinctPages;
	int64		indexPagePrefetches;
	bool		indexPageDistinctCountsExact;
	int64		indexPageLoads;
	int64		indexPageRuns;
	int64		indexPageDistinctPages;
	BlockNumber indexPageLastBlock;
	bool		indexPageDistinctPagesExact;
	int64		heapTidReturns;
	int64		heapTidPageRuns;
	int64		heapTidDistinctPages;
	bool		heapTidDistinctPagesExact;
	int64		blksHitBefore;
	int64		blksHitAfter;
	int64		blksReadBefore;
	int64		blksReadAfter;
	int64		idxBlksHit;
	int64		idxBlksRead;
	int64		heapBlksHit;
	int64		heapBlksRead;
	int			topkTidCount;
	ItemPointerData topkTids[HNSW_PROFILE_MAX_TIDS];
	HnswPlannerProofOutcome plannerProof;
	HnswTraversalFinalPath traversalFinalPath;
	HnswTraversalStockBypassReason traversalStockBypassReason;
	HnswTraversalAdmissionReason traversalAdmissionReason;
	HnswTraversalFallbackReason traversalFallbackReason;
	bool		traversalEstimatedSkipRateValid;
	double		traversalEstimatedSkipRate;
	int			traversalPrioritizationBurst;
	HnswIterativeScanMode iterativeScan;
	HnswFilterStrategyMode filterStrategy;
	int			plannerProofCount;
	bool		plannerProofsTruncated;
	HnswPlannerProofOutcome plannerProofs[HNSW_PROFILE_MAX_PROOFS];
}			HnswScanProfile;

typedef struct HnswIndexPageProfile
{
	int64		neighborLoads;
	int64		neighborRuns;
	int64		neighborDistinctPages;
	int64		elementLoads;
	int64		elementRuns;
	int64		elementDistinctPages;
	int64		prefetches;
	bool		distinctCountsExact;
	int64		loads;
	int64		runs;
	int64		distinctPages;
	BlockNumber lastBlock;
}			HnswIndexPageProfile;

typedef struct HnswIndexPageSet
{
	uint32	   *slots;
	int			count;
} HnswIndexPageSet;

typedef struct HnswIndexPageProfileState
{
	HnswIndexPageProfile profile;
	MemoryContext context;
	BlockNumber lastNeighborBlock;
	BlockNumber lastElementBlock;
	HnswIndexPageSet neighborPages;
	HnswIndexPageSet elementPages;
	HnswIndexPageSet pages;
} HnswIndexPageProfileState;

typedef struct HnswPageAccessItem
{
	ItemPointerData heaptid;
	double		distance;
	int			rank;
	bool		guidanceChecked;
}			HnswPageAccessItem;

/* HNSW index options */
typedef struct HnswOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* number of connections */
	int			efConstruction; /* size of dynamic candidate list */
}			HnswOptions;

typedef struct HnswGraph
{
	/* Graph state */
	slock_t		lock;
	HnswElementPtr head;
	double		indtuples;

	/* Entry state */
	LWLock		entryLock;
	LWLock		entryWaitLock;
	HnswElementPtr entryPoint;

	/* Allocations state */
	LWLock		allocatorLock;
	Size		memoryUsed;
	Size		memoryTotal;

	/* Flushed state */
	LWLock		flushLock;
	bool		flushed;
}			HnswGraph;

typedef struct HnswShared
{
	/* Immutable state */
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;

	/* Worker progress */
	ConditionVariable workersdonecv;

	/* Mutex for mutable state */
	slock_t		mutex;

	/* Mutable state */
	int			nparticipantsdone;
	double		reltuples;
	HnswGraph	graphData;
}			HnswShared;

#define ParallelTableScanFromHnswShared(shared) \
	(ParallelTableScanDesc) ((char *) (shared) + BUFFERALIGN(sizeof(HnswShared)))

typedef struct HnswLeader
{
	ParallelContext *pcxt;
	int			nparticipanttuplesorts;
	HnswShared *hnswshared;
	Snapshot	snapshot;
	char	   *hnswarea;
}			HnswLeader;

typedef struct HnswAllocator
{
	void	   *(*alloc) (Size size, void *state);
	void	   *state;
}			HnswAllocator;

typedef struct HnswTypeInfo
{
	int			maxDimensions;
	Datum		(*normalize) (PG_FUNCTION_ARGS);
	void		(*checkValue) (Pointer v);
}			HnswTypeInfo;

typedef struct HnswSupport
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
}			HnswSupport;

typedef struct HnswQuery
{
	Datum		value;
}			HnswQuery;

typedef struct HnswBuildState
{
	/* Info */
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	ForkNumber	forkNum;
	const		HnswTypeInfo *typeInfo;

	/* Settings */
	int			dimensions;
	int			m;
	int			efConstruction;

	/* Statistics */
	double		indtuples;
	double		reltuples;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	HnswGraph	graphData;
	HnswGraph  *graph;
	double		ml;
	int			maxLevel;

	/* Memory */
	MemoryContext graphCtx;
	MemoryContext tmpCtx;
	HnswAllocator allocator;

	/* Parallel builds */
	HnswLeader *hnswleader;
	HnswShared *hnswshared;
	char	   *hnswarea;
}			HnswBuildState;

typedef struct HnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint32		dimensions;
	uint16		m;
	uint16		efConstruction;
	BlockNumber entryBlkno;
	OffsetNumber entryOffno;
	int16		entryLevel;
	BlockNumber insertPage;
}			HnswMetaPageData;

typedef HnswMetaPageData * HnswMetaPage;

typedef struct HnswPageOpaqueData
{
	BlockNumber nextblkno;
	uint16		unused;
	uint16		page_id;		/* for identification of HNSW indexes */
}			HnswPageOpaqueData;

typedef HnswPageOpaqueData * HnswPageOpaque;

typedef struct HnswElementTupleData
{
	uint8		type;
	uint8		level;
	uint8		deleted;
	uint8		version;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	ItemPointerData neighbortid;
	uint16		unused;
	Vector		data;
}			HnswElementTupleData;

typedef HnswElementTupleData * HnswElementTuple;

typedef struct HnswNeighborTupleData
{
	uint8		type;
	uint8		version;
	uint16		count;
	ItemPointerData indextids[FLEXIBLE_ARRAY_MEMBER];
}			HnswNeighborTupleData;

typedef HnswNeighborTupleData * HnswNeighborTuple;

typedef union
{
	struct pointerhash_hash *pointers;
	struct offsethash_hash *offsets;
	struct tidhash_hash *tids;
}			visited_hash;

typedef union
{
	HnswElement element;
	ItemPointerData indextid;
}			HnswUnvisited;

typedef struct HnswScanOpaqueData
{
	const		HnswTypeInfo *typeInfo;
	bool		first;
	List	   *w;
	visited_hash v;
	pairingheap *discarded;
	HnswQuery	q;
	int			m;
	int64		tuples;
	instr_time	scanStart;
	instr_time	inIndexTime;
	double		vectorSearchMs;
	int64		returnedTuples;
	int64		distanceComputations;
	int64		blksHitBefore;
	int64		blksReadBefore;
	int64		idxBlksHit;
	int64		idxBlksRead;
	int			topkTidCount;
	ItemPointerData topkTids[HNSW_PROFILE_MAX_TIDS];
	double		previousDistance;
	Size		maxMemory;
	MemoryContext tmpCtx;
	MemoryContext profileCtx;
	HnswPageAccessItem *pageItems;
	int			pageItemCount;
	int			pageItemIndex;
	int			pageItemCapacity;
	int64		pageAccessBatches;
	int64		pageAccessCandidates;
	int64		pageAccessPrefetches;
	int64		pageAccessDistanceRuns;
	int64		pageAccessDistinctPages;
	int			pageAccessNoMergeBatches;
	bool		pageAccessDisabled;
	int64		guidanceChecks;
	int64		guidanceMatches;
	int64		guidanceSkips;
	int64		heapTidReturns;
	int64		heapTidPageRuns;
	int64		heapTidDistinctPages;
	BlockNumber heapTidLastBlock;
	HnswIndexPageSet heapTidPages;
	bool		heapTidDistinctPagesExact;
	HnswTraversalProfile traversal;
	HnswIndexPageProfileState indexPageProfile;
	void	   *guidancePlan;
	HnswScanGuidance *guidance;
	bool		guidanceDecided;
	HnswPlannerProofOutcome plannerProof;
	HnswTraversalGuidanceState traversalGuidance;
	int64		abandonedGuidedTuples;

	/* Support functions */
	HnswSupport support;
}			HnswScanOpaqueData;

typedef HnswScanOpaqueData * HnswScanOpaque;

typedef struct HnswVacuumState
{
	/* Info */
	Relation	index;
	IndexBulkDeleteResult *stats;
	IndexBulkDeleteCallback callback;
	void	   *callback_state;

	/* Settings */
	int			m;
	int			efConstruction;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	struct tidhash_hash *deleted;
	BufferAccessStrategy bas;
	HnswNeighborTuple ntup;
	HnswElementData highestPoint;

	/* Memory */
	MemoryContext tmpCtx;
}			HnswVacuumState;

/* Methods */
int			HnswGetM(Relation index);
int			HnswGetEfConstruction(Relation index);
FmgrInfo   *HnswOptionalProcInfo(Relation index, uint16 procnum);
void		HnswInitSupport(HnswSupport * support, Relation index);
Datum		HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value);
bool		HnswCheckNorm(HnswSupport * support, Datum value);
Buffer		HnswNewBuffer(Relation index, ForkNumber forkNum);
void		HnswInitPage(Buffer buf, Page page);
void		HnswInit(void);
List	   *HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples, int64 *distanceComputations, HnswTraversalProfile *traversalProfile, HnswScanGuidance *guidance, HnswTraversalGuidanceState *traversalGuidance, HnswIndexPageProfileState *indexPageProfile);
HnswElement HnswGetEntryPoint(Relation index);
void		HnswGetMetaPageInfo(Relation index, int *m, HnswElement * entryPoint);
void		HnswGetMetaPageInfoTracked(Relation index, int *m, HnswElement * entryPoint, HnswIndexPageProfileState *profile);
void	   *HnswAlloc(HnswAllocator * allocator, Size size);
HnswElement HnswInitElement(char *base, ItemPointer tid, int m, double ml, int maxLevel, HnswAllocator * alloc);
HnswElement HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno);
void		HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing);
HnswSearchCandidate *HnswEntryCandidate(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec);
HnswSearchCandidate *HnswEntryCandidateTracked(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, HnswIndexPageProfileState *profile);
void		HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building);
void		HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m);
void		HnswAddHeapTid(HnswElement element, ItemPointer heaptid);
HnswNeighborArray *HnswInitNeighborArray(int lm, HnswAllocator * allocator);
void		HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * alloc);
bool		HnswInsertTupleOnDisk(Relation index, HnswSupport * support, Datum value, ItemPointer heaptid, bool building);
void		HnswUpdateNeighborsOnDisk(Relation index, HnswSupport * support, HnswElement e, int m, bool checkExisting, bool building);
void		HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec);
void		HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance);
bool		HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support);
void		HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element);
void		HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support);
bool		HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc);
void		HnswInitLockTranche(void);
const		HnswTypeInfo *HnswGetTypeInfo(Relation index);
PGDLLEXPORT void HnswParallelBuildMain(dsm_segment *seg, shm_toc *toc);

/* Index access methods */
IndexBuildResult *hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		hnswbuildempty(Relation index);
void		HnswCloneGraph(HnswBuildState *buildstate);
bool		hnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
IndexBulkDeleteResult *hnswbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *hnswvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
IndexScanDesc hnswbeginscan(Relation index, int nkeys, int norderbys);
void		hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool		hnswgettuple(IndexScanDesc scan, ScanDirection dir);
void		hnswendscan(IndexScanDesc scan);
void		HnswResetScanProfile(void);
void		HnswGetLastScanProfile(HnswScanProfile *profile);
void		HnswInitIndexPageProfile(HnswIndexPageProfileState *state,
								 MemoryContext context);
void		HnswRecordHeapTid(HnswScanOpaque scan, ItemPointer tid);
bool		HnswGuidanceIsActive(void);
bool		HnswGuidanceIsActiveForHeap(Oid heapOid);
void		HnswGuidanceRegisterExecutorScans(QueryDesc *queryDesc, uint64 frameId);
void		HnswGuidanceAttachCurrentPlan(IndexScanDesc scan);
HnswScanGuidance *HnswGuidancePrepareForScan(IndexScanDesc scan, void *planBinding,
											 HnswPlannerProofOutcome *proof);
bool		HnswGuidanceIsActiveForScan(HnswScanGuidance *guidance);
bool		HnswGuidanceAllowsTid(HnswScanGuidance *guidance, ItemPointer tid);
bool		HnswGuidanceGetEstimatedSkipRate(HnswScanGuidance *guidance, double *skipRate);
void		HnswGuidanceEndScan(HnswScanGuidance *guidance);
void		HnswGuidanceRecordScan(Oid heapOid, int64 candidates,
							 int64 guidanceChecks, int64 guidanceSkips,
							 double heapFetchMs, double totalScanMs);

static inline HnswNeighborArray *
HnswGetNeighbors(char *base, HnswElement element, int lc)
{
	HnswNeighborArrayPtr *neighborList = HnswPtrAccess(base, element->neighbors);

	Assert(element->level >= lc);

	return HnswPtrAccess(base, neighborList[lc]);
}

/* Hash tables */
typedef struct TidHashEntry
{
	ItemPointerData tid;
	char		status;
}			TidHashEntry;

#define SH_PREFIX tidhash
#define SH_ELEMENT_TYPE TidHashEntry
#define SH_KEY_TYPE ItemPointerData
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct PointerHashEntry
{
	uintptr_t	ptr;
	char		status;
}			PointerHashEntry;

#define SH_PREFIX pointerhash
#define SH_ELEMENT_TYPE PointerHashEntry
#define SH_KEY_TYPE uintptr_t
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct OffsetHashEntry
{
	Size		offset;
	char		status;
}			OffsetHashEntry;

#define SH_PREFIX offsethash
#define SH_ELEMENT_TYPE OffsetHashEntry
#define SH_KEY_TYPE Size
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

#endif

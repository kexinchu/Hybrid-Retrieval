#include "postgres.h"

#include "access/genam.h"
#include "access/relscan.h"
#include "executor/instrument.h"
#include "hnsw.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "nodes/pg_list.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/relcache.h"
#include "utils/snapmgr.h"
#include "portability/instr_time.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

/*
 * Algorithm 5 from paper
 */
static HnswScanProfile hnsw_last_profile;

static Datum GetScanValue(IndexScanDesc scan);

static Oid
HnswScanHeapOid(IndexScanDesc scan)
{
	if (scan->heapRelation != NULL)
		return RelationGetRelid(scan->heapRelation);
	if (scan->indexRelation != NULL && scan->indexRelation->rd_index != NULL)
		return scan->indexRelation->rd_index->indrelid;
	return InvalidOid;
}

static int
ComparePageAccessItems(const void *a, const void *b)
{
	const HnswPageAccessItem *ia = (const HnswPageAccessItem *) a;
	const HnswPageAccessItem *ib = (const HnswPageAccessItem *) b;
	BlockNumber ba = ItemPointerGetBlockNumber(&ia->heaptid);
	BlockNumber bb = ItemPointerGetBlockNumber(&ib->heaptid);
	OffsetNumber oa = ItemPointerGetOffsetNumber(&ia->heaptid);
	OffsetNumber ob = ItemPointerGetOffsetNumber(&ib->heaptid);

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
ComparePageAccessItemsByRank(const void *a, const void *b)
{
	const HnswPageAccessItem *ia = (const HnswPageAccessItem *) a;
	const HnswPageAccessItem *ib = (const HnswPageAccessItem *) b;

	if (ia->rank < ib->rank)
		return -1;
	if (ia->rank > ib->rank)
		return 1;
	return 0;
}

static List *
RunScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;

	/* Get m and entry point */
	HnswGetMetaPageInfoTracked(index, &m, &entryPoint, &so->indexPageProfile);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidateTracked(base, entryPoint, q, index, support,
											 false, &so->indexPageProfile));
	so->distanceComputations++;

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		instr_time start;
		instr_time elapsed;

		INSTR_TIME_SET_CURRENT(start);
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL, &so->distanceComputations, &so->traversal, so->guidance, &so->traversalGuidance, &so->indexPageProfile);
		INSTR_TIME_SET_CURRENT(elapsed);
		INSTR_TIME_SUBTRACT(elapsed, start);
		so->vectorSearchMs += INSTR_TIME_GET_MILLISEC(elapsed);
		ep = w;
	}

	{
		instr_time start;
		instr_time elapsed;
		List	   *next;

		INSTR_TIME_SET_CURRENT(start);
		so->traversal.initialBatches++;
		next = HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m, false, NULL, &so->v,
									   hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples, &so->distanceComputations, &so->traversal, so->guidance, &so->traversalGuidance, &so->indexPageProfile);
		INSTR_TIME_SET_CURRENT(elapsed);
		INSTR_TIME_SUBTRACT(elapsed, start);
		so->vectorSearchMs += INSTR_TIME_GET_MILLISEC(elapsed);
		return next;
	}
}

static void
HnswInitializeTraversalGuidance(HnswScanOpaque so)
{
	HnswTraversalGuidanceState *state = &so->traversalGuidance;
	bool		guidanceActive;

	MemSet(state, 0, sizeof(*state));
	state->iterativeScan = (HnswIterativeScanMode) hnsw_iterative_scan;
	state->filterStrategy = (HnswFilterStrategyMode) hnsw_filter_strategy;
	state->finalPath = HNSW_TRAVERSAL_PATH_STOCK;
	state->admissionReason = HNSW_TRAVERSAL_ADMISSION_NOT_REQUESTED;
	guidanceActive = HnswGuidanceIsActiveForScan(so->guidance);

	if (hnsw_filter_strategy == HNSW_FILTER_STRATEGY_OFF)
		return;

	if (hnsw_filter_strategy == HNSW_FILTER_STRATEGY_SAFE_GUIDED)
	{
		state->finalPath = guidanceActive ?
			HNSW_TRAVERSAL_PATH_VALIDATION_ONLY :
			HNSW_TRAVERSAL_PATH_STOCK_BYPASS;
		if (!guidanceActive)
		{
			state->stockBypassReason = HNSW_TRAVERSAL_BYPASS_NO_PROVEN_GUIDE;
			so->traversal.stockBypassRequests++;
		}
		return;
	}
	if (hnsw_filter_strategy == HNSW_FILTER_STRATEGY_ACORN1 ||
		hnsw_filter_strategy == HNSW_FILTER_STRATEGY_GUIDED_COLLECT)
	{
		state->finalPath = guidanceActive ?
			HNSW_TRAVERSAL_PATH_LEGACY_GUIDED :
			HNSW_TRAVERSAL_PATH_STOCK_BYPASS;
		if (!guidanceActive)
		{
			state->stockBypassReason = HNSW_TRAVERSAL_BYPASS_NO_PROVEN_GUIDE;
			so->traversal.stockBypassRequests++;
		}
		return;
	}
	Assert(hnsw_filter_strategy == HNSW_FILTER_STRATEGY_TRAVERSAL_GUIDED);

	state->requested = true;
	/* The deprecated target GUC cannot change the layer-0 result contract. */
	state->target = hnsw_ef_search;
	state->maxBridgeHops = hnsw_traversal_guided_max_bridge_hops;
	state->maxBridgeWork = hnsw_traversal_guided_max_bridge_work;
	state->maxScanTuples = hnsw_max_scan_tuples;
	state->maxMemory = so->maxMemory;
	state->burst = hnsw_traversal_guided_burst;
	state->estimatedSkipRateValid = HnswGuidanceGetEstimatedSkipRate(
		so->guidance, &state->estimatedSkipRate);

	if (!guidanceActive)
	{
		state->stockBypassReason = HNSW_TRAVERSAL_BYPASS_NO_PROVEN_GUIDE;
		state->admissionReason = HNSW_TRAVERSAL_ADMISSION_NO_PROVEN_GUIDE;
	}
	else if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
	{
		state->stockBypassReason = HNSW_TRAVERSAL_BYPASS_ITERATIVE_SCAN;
		state->admissionReason = HNSW_TRAVERSAL_ADMISSION_ITERATIVE_SCAN;
	}
	else if (!state->estimatedSkipRateValid)
	{
		state->stockBypassReason = HNSW_TRAVERSAL_BYPASS_SKIP_ESTIMATE_UNAVAILABLE;
		state->admissionReason = HNSW_TRAVERSAL_ADMISSION_SKIP_ESTIMATE_UNAVAILABLE;
	}
	else if (state->estimatedSkipRate < hnsw_traversal_guided_min_skip_rate)
	{
		state->stockBypassReason = HNSW_TRAVERSAL_BYPASS_LOW_ESTIMATED_SKIP_RATE;
		state->admissionReason = HNSW_TRAVERSAL_ADMISSION_LOW_ESTIMATED_SKIP_RATE;
	}
	else
	{
		if (!hnsw_traversal_guided_prioritization)
		{
			state->finalPath = HNSW_TRAVERSAL_PATH_CANDIDATE_ADMISSION;
			state->admissionReason =
				HNSW_TRAVERSAL_ADMISSION_DEFAULT_VALIDATION_ONLY;
			return;
		}

		/*
		 * Both frontiers remain distance ordered internally.  The bounded pop
		 * schedule prioritizes MAYBE nodes without starving predicate-NO graph
		 * bridges; this is approximate ANN prioritization, not exact-safe graph
		 * pruning or a stock-equivalent traversal.
		 */
		state->prioritizationEnabled = true;
		state->finalPath = HNSW_TRAVERSAL_PATH_APPROXIMATE_PRIORITIZATION;
		state->admissionReason = HNSW_TRAVERSAL_ADMISSION_ADMITTED;
		return;
	}

	state->finalPath = HNSW_TRAVERSAL_PATH_STOCK_BYPASS;
	so->traversal.stockBypassRequests++;
}

static HnswTraversalFallbackReason
HnswTraversalGuidanceFallbackReason(HnswTraversalGuidanceState *state)
{
	if (state->invalidNeighbor)
		return HNSW_TRAVERSAL_FALLBACK_INVALID_NEIGHBOR;
	if (state->memoryLimitReached)
		return HNSW_TRAVERSAL_FALLBACK_MEMORY_LIMIT;
	if (state->maxScanReached)
		return HNSW_TRAVERSAL_FALLBACK_MAX_SCAN_TUPLES;
	if (state->workLimitReached)
		return HNSW_TRAVERSAL_FALLBACK_BRIDGE_WORK;
	if (state->hopLimitReached)
		return HNSW_TRAVERSAL_FALLBACK_BRIDGE_HOPS;
	return HNSW_TRAVERSAL_FALLBACK_INSUFFICIENT_MATCHES;
}

static List *
GetScanItems(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	HnswTraversalGuidanceState *state = &so->traversalGuidance;
	int64		distanceStart = so->distanceComputations;
	int64		expandedStart = so->traversal.expandedNodes;
	int64		avoidedStart = so->traversal.distanceComputationsAvoided;
	List	   *items;

	if (!state->prioritizationEnabled)
	{
		bool		continuingFallback =
			state->finalPath == HNSW_TRAVERSAL_PATH_FRESH_STOCK_FALLBACK;
		bool		guidedAdmission =
			state->finalPath == HNSW_TRAVERSAL_PATH_CANDIDATE_ADMISSION;

		if (continuingFallback)
			so->traversal.fallbackRequests++;
		items = RunScanItems(scan, GetScanValue(scan));
		if (state->requested)
		{
			int64		phaseDistance = so->distanceComputations - distanceStart;
			int64		phaseExpanded =
				so->traversal.expandedNodes - expandedStart;

			if (guidedAdmission)
			{
				so->traversal.guidedPhaseDistanceComputations += phaseDistance;
				so->traversal.guidedExpandedNodes += phaseExpanded;
			}
			else
			{
				so->traversal.stockPhaseDistanceComputations += phaseDistance;
				so->traversal.stockPhaseExpandedNodes += phaseExpanded;
			}
			if (continuingFallback)
			{
				so->traversal.fallbackStockDistanceComputations += phaseDistance;
				so->traversal.fallbackStockExpandedNodes += phaseExpanded;
			}
		}
		return items;
	}

	{
		MemoryContext guidedContext;
		MemoryContext oldContext;
		bool		guidedUsable;

		guidedContext = AllocSetContextCreate(so->tmpCtx,
			"HNSW traversal-guided attempt", ALLOCSET_DEFAULT_SIZES);
		state->phaseContext = guidedContext;
		oldContext = MemoryContextSwitchTo(guidedContext);
		items = RunScanItems(scan, GetScanValue(scan));
		MemoryContextSwitchTo(oldContext);

		so->traversal.guidedPhaseDistanceComputations +=
			so->distanceComputations - distanceStart;
		so->traversal.guidedExpandedNodes +=
			so->traversal.expandedNodes - expandedStart;
		guidedUsable = state->guidedResultCount >= state->target &&
			!state->invalidNeighbor;
		if (guidedUsable)
			return items;

		state->fallbackReason = HnswTraversalGuidanceFallbackReason(state);
		state->finalPath = HNSW_TRAVERSAL_PATH_FRESH_STOCK_FALLBACK;
		state->prioritizationEnabled = false;
		state->phaseContext = NULL;
		so->traversal.fallbackRequests++;
		/* Abandoned pre-distance skips are attempted work, not net savings. */
		so->traversal.distanceComputationsAvoided = avoidedStart;
		so->abandonedGuidedTuples += so->tuples;
		so->tuples = 0;

		MemoryContextDelete(guidedContext);
		so->w = NIL;
		MemSet(&so->v, 0, sizeof(so->v));
		so->discarded = NULL;
		MemSet(&so->q, 0, sizeof(so->q));

		distanceStart = so->distanceComputations;
		expandedStart = so->traversal.expandedNodes;
		items = RunScanItems(scan, GetScanValue(scan));
		so->traversal.stockPhaseDistanceComputations +=
			so->distanceComputations - distanceStart;
		so->traversal.stockPhaseExpandedNodes +=
			so->traversal.expandedNodes - expandedStart;
		so->traversal.fallbackStockDistanceComputations +=
			so->distanceComputations - distanceStart;
		so->traversal.fallbackStockExpandedNodes +=
			so->traversal.expandedNodes - expandedStart;
		return items;
	}
}

/*
 * Resume scan at ground level with discarded candidates
 */
static List *
ResumeScanItems(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	List	   *ep = NIL;
	char	   *base = NULL;
	int			batch_size = hnsw_ef_search;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));
		so->traversal.discardedPops++;

		ep = lappend(ep, sc);
	}

	{
		instr_time start;
		instr_time elapsed;
		List	   *next;

		INSTR_TIME_SET_CURRENT(start);
		so->traversal.resumeBatches++;
		next = HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false,
									  &so->tuples, &so->distanceComputations, &so->traversal, so->guidance, &so->traversalGuidance, &so->indexPageProfile);
		INSTR_TIME_SET_CURRENT(elapsed);
		INSTR_TIME_SUBTRACT(elapsed, start);
		so->vectorSearchMs += INSTR_TIME_GET_MILLISEC(elapsed);
		return next;
	}
}

void
HnswResetScanProfile(void)
{
	hnsw_last_profile.valid = false;
	hnsw_last_profile.totalScanMs = 0;
	hnsw_last_profile.hnswSearchMs = 0;
	hnsw_last_profile.heapFetchMs = 0;
	hnsw_last_profile.vectorSearchMs = 0;
	hnsw_last_profile.visitedTuples = 0;
	hnsw_last_profile.returnedTuples = 0;
	hnsw_last_profile.distanceComputations = 0;
	hnsw_last_profile.pageAccessBatches = 0;
	hnsw_last_profile.pageAccessCandidates = 0;
	hnsw_last_profile.pageAccessPrefetches = 0;
	hnsw_last_profile.pageAccessDistanceRuns = 0;
	hnsw_last_profile.pageAccessDistinctPages = 0;
	hnsw_last_profile.guidanceChecks = 0;
	hnsw_last_profile.guidanceMatches = 0;
	hnsw_last_profile.guidanceSkips = 0;
	MemSet(&hnsw_last_profile.traversal, 0, sizeof(HnswTraversalProfile));
	hnsw_last_profile.indexPageNeighborLoads = 0;
	hnsw_last_profile.indexPageNeighborRuns = 0;
	hnsw_last_profile.indexPageNeighborDistinctPages = 0;
	hnsw_last_profile.indexPageElementLoads = 0;
	hnsw_last_profile.indexPageElementRuns = 0;
	hnsw_last_profile.indexPageElementDistinctPages = 0;
	hnsw_last_profile.indexPagePrefetches = 0;
	hnsw_last_profile.indexPageLoads = 0;
	hnsw_last_profile.indexPageRuns = 0;
	hnsw_last_profile.indexPageDistinctPages = 0;
	hnsw_last_profile.indexPageLastBlock = InvalidBlockNumber;
	hnsw_last_profile.indexPageDistinctPagesExact = true;
	hnsw_last_profile.heapTidReturns = 0;
	hnsw_last_profile.heapTidPageRuns = 0;
	hnsw_last_profile.heapTidDistinctPages = 0;
	hnsw_last_profile.heapTidDistinctPagesExact = true;
	hnsw_last_profile.indexPageDistinctCountsExact = true;
	hnsw_last_profile.blksHitBefore = 0;
	hnsw_last_profile.blksHitAfter = 0;
	hnsw_last_profile.blksReadBefore = 0;
	hnsw_last_profile.blksReadAfter = 0;
	hnsw_last_profile.idxBlksHit = 0;
	hnsw_last_profile.idxBlksRead = 0;
	hnsw_last_profile.heapBlksHit = 0;
	hnsw_last_profile.heapBlksRead = 0;
	hnsw_last_profile.topkTidCount = 0;
	MemSet(&hnsw_last_profile.plannerProof, 0,
		   sizeof(hnsw_last_profile.plannerProof));
	hnsw_last_profile.plannerProof.bypassReason = HNSW_PROOF_BYPASS_SCAN_NOT_STARTED;
	hnsw_last_profile.traversalFinalPath = HNSW_TRAVERSAL_PATH_STOCK;
	hnsw_last_profile.traversalStockBypassReason = HNSW_TRAVERSAL_BYPASS_NONE;
	hnsw_last_profile.traversalAdmissionReason =
		HNSW_TRAVERSAL_ADMISSION_NOT_REQUESTED;
	hnsw_last_profile.traversalFallbackReason = HNSW_TRAVERSAL_FALLBACK_NONE;
	hnsw_last_profile.traversalEstimatedSkipRateValid = false;
	hnsw_last_profile.traversalEstimatedSkipRate = 0;
	hnsw_last_profile.traversalPrioritizationBurst =
		hnsw_traversal_guided_burst;
	hnsw_last_profile.iterativeScan = (HnswIterativeScanMode) hnsw_iterative_scan;
	hnsw_last_profile.filterStrategy = (HnswFilterStrategyMode) hnsw_filter_strategy;
	hnsw_last_profile.plannerProofCount = 0;
	hnsw_last_profile.plannerProofsTruncated = false;
	MemSet(hnsw_last_profile.plannerProofs, 0,
		   sizeof(hnsw_last_profile.plannerProofs));
}

void
HnswGetLastScanProfile(HnswScanProfile *profile)
{
	if (profile != NULL)
		*profile = hnsw_last_profile;
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

static bool
HnswGetNextHeapTid(IndexScanDesc scan, ItemPointerData *heaptid, double *distance)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		ItemPointer tid;

		CHECK_FOR_INTERRUPTS();

		if (list_length(so->w) == 0)
		{
			if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
				return false;

			/* Empty index */
			if (so->discarded == NULL)
				return false;

			/* Reached max number of tuples or memory limit */
			if (so->tuples >= hnsw_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
			{
				if (pairingheap_is_empty(so->discarded))
					return false;

				/* Return remaining tuples */
				so->w = lappend(so->w, HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
				so->traversal.discardedPops++;
			}
			else
			{
				/*
				 * Locking ensures when neighbors are read, the elements they
				 * reference will not be deleted (and replaced) during the
				 * iteration.
				 *
				 * Elements loaded into memory on previous iterations may have
				 * been deleted (and replaced), so when reading neighbors, the
				 * element version must be checked.
				 */
				LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

				so->w = ResumeScanItems(scan);

				UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
				ShowMemoryUsage(so);
#endif
			}

			if (list_length(so->w) == 0)
				return false;
		}

		sc = llast(so->w);
		element = HnswPtrAccess(base, sc->element);

		/* Move to next element if no valid heap TIDs */
		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);

			/* Mark memory as free for next iteration */
			if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
			{
				pfree(element);
				pfree(sc);
			}

			continue;
		}

		tid = &element->heaptids[--element->heaptidsLength];

		if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
		{
			if (sc->distance < so->previousDistance)
			{
				so->traversal.strictOrderDrops++;
				continue;
			}

			so->previousDistance = sc->distance;
		}

		*heaptid = *tid;
		if (distance != NULL)
			*distance = sc->distance;
		return true;
	}
}

static bool
HnswScanUsesGuidanceValidation(HnswScanOpaque so)
{
	if (so->traversalGuidance.requested)
		return (so->traversalGuidance.finalPath ==
				HNSW_TRAVERSAL_PATH_CANDIDATE_ADMISSION ||
			so->traversalGuidance.finalPath ==
				HNSW_TRAVERSAL_PATH_APPROXIMATE_PRIORITIZATION) &&
			HnswGuidanceIsActiveForScan(so->guidance);

	return HnswGuidanceIsActiveForScan(so->guidance);
}

static bool
HnswFillPageAccessBuffer(IndexScanDesc scan, int pageAccessMode)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	int			target = hnsw_page_window;
	BlockNumber previousBlock = InvalidBlockNumber;
	BlockNumber previousPrefetchBlock = InvalidBlockNumber;
	int64		distanceRuns = 0;
	int64		distinctPages = 0;
	int			rawPulls = 0;
	int			mergeablePages = 0;
	bool		exhausted = false;

	so->pageItemCount = 0;
	so->pageItemIndex = 0;

	if (target < 1)
		target = 1;

	if (so->pageItemCapacity < target)
	{
		if (so->pageItems == NULL)
			so->pageItems = palloc(sizeof(HnswPageAccessItem) * target);
		else
			so->pageItems = repalloc(so->pageItems, sizeof(HnswPageAccessItem) * target);
		so->pageItemCapacity = target;
	}

	while (so->pageItemCount < target && rawPulls < target)
	{
		HnswPageAccessItem *item = &so->pageItems[so->pageItemCount];
		BlockNumber block;

		CHECK_FOR_INTERRUPTS();

		if (!HnswGetNextHeapTid(scan, &item->heaptid, &item->distance))
		{
			exhausted = true;
			break;
		}
		rawPulls++;

		item->guidanceChecked = false;
		if (HnswScanUsesGuidanceValidation(so))
		{
			item->guidanceChecked = true;
			so->guidanceChecks++;
			so->traversal.guidanceChecks++;
			if (!HnswGuidanceAllowsTid(so->guidance, &item->heaptid))
			{
				so->guidanceSkips++;
				so->traversal.guidanceMisses++;
				so->traversal.guidedSuppressions++;
				so->traversal.heapTidsSuppressed++;
				continue;
			}
			so->guidanceMatches++;
			so->traversal.guidanceMatches++;
		}

		item->rank = so->pageItemCount;
		block = ItemPointerGetBlockNumber(&item->heaptid);
		if (so->pageItemCount == 0 || block != previousBlock)
			distanceRuns++;
		previousBlock = block;

		so->pageItemCount++;
	}

	if (so->pageItemCount == 0)
	{
		if (!exhausted)
			so->pageAccessDisabled = true;
		return false;
	}

	so->pageAccessBatches++;
	so->pageAccessCandidates += so->pageItemCount;
	so->pageAccessDistanceRuns += distanceRuns;

	qsort(so->pageItems, so->pageItemCount, sizeof(HnswPageAccessItem), ComparePageAccessItems);

	for (int i = 0; i < so->pageItemCount; i++)
	{
		BlockNumber block = ItemPointerGetBlockNumber(&so->pageItems[i].heaptid);

		if (i == 0 || block != previousPrefetchBlock)
		{
			int			runLength = 1;

			distinctPages++;

			while (i + runLength < so->pageItemCount &&
				   ItemPointerGetBlockNumber(&so->pageItems[i + runLength].heaptid) == block)
				runLength++;

			if (pageAccessMode == HNSW_PAGE_ACCESS_PREFETCH || pageAccessMode == HNSW_PAGE_ACCESS_REORDER)
			{
				if (runLength >= hnsw_page_prefetch_min_items)
				{
					mergeablePages++;
					PrefetchBuffer(scan->heapRelation, MAIN_FORKNUM, block);
					so->pageAccessPrefetches++;
				}
			}
		}

		previousPrefetchBlock = block;
	}

	so->pageAccessDistinctPages += distinctPages;
	if (hnsw_page_disable_after_no_merge > 0)
	{
		if (mergeablePages == 0)
			so->pageAccessNoMergeBatches++;
		else
			so->pageAccessNoMergeBatches = 0;

		if (so->pageAccessNoMergeBatches >= hnsw_page_disable_after_no_merge)
			so->pageAccessDisabled = true;
	}

	/*
	 * prefetch mode preserves the candidate distance order. reorder mode is
	 * experimental and returns TIDs in heap page order within the window.
	 */
	if (pageAccessMode == HNSW_PAGE_ACCESS_PREFETCH)
		qsort(so->pageItems, so->pageItemCount, sizeof(HnswPageAccessItem), ComparePageAccessItemsByRank);

	return true;
}

#if defined(HNSW_MEMORY)
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(HnswScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
										   "Hnsw scan temporary context",
										   0, 8 * 1024, 256 * 1024);
	so->profileCtx = AllocSetContextCreate(CurrentMemoryContext,
										   "Hnsw scan profile context",
										   ALLOCSET_DEFAULT_SIZES);
	HnswInitIndexPageProfile(&so->indexPageProfile, so->profileCtx);
	so->pageItems = NULL;
	so->pageItemCount = 0;
	so->pageItemIndex = 0;
	so->pageItemCapacity = 0;
	so->pageAccessBatches = 0;
	so->pageAccessCandidates = 0;
	so->pageAccessPrefetches = 0;
	so->pageAccessDistanceRuns = 0;
	so->pageAccessDistinctPages = 0;
	so->pageAccessNoMergeBatches = 0;
	so->pageAccessDisabled = false;
	so->guidanceChecks = 0;
	so->guidanceMatches = 0;
	so->guidanceSkips = 0;
	so->heapTidReturns = 0;
	so->heapTidPageRuns = 0;
	so->heapTidDistinctPages = 0;
	so->heapTidLastBlock = InvalidBlockNumber;
	MemSet(&so->heapTidPages, 0, sizeof(so->heapTidPages));
	so->heapTidDistinctPagesExact = true;
	MemSet(&so->traversal, 0, sizeof(HnswTraversalProfile));
	so->guidancePlan = NULL;
	so->guidance = NULL;
	so->guidanceDecided = false;
	MemSet(&so->plannerProof, 0, sizeof(so->plannerProof));
	MemSet(&so->traversalGuidance, 0, sizeof(so->traversalGuidance));
	so->traversalGuidance.finalPath = HNSW_TRAVERSAL_PATH_STOCK;
	so->abandonedGuidedTuples = 0;
	so->plannerProof.bypassReason = HNSW_PROOF_BYPASS_SCAN_NOT_STARTED;
	so->plannerProof.indexOid = RelationGetRelid(index);
	so->plannerProof.heapOid = index->rd_index != NULL ?
		index->rd_index->indrelid : InvalidOid;

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

	scan->opaque = so;
	HnswGuidanceAttachCurrentPlan(scan);

	return scan;
}

/*
 * Start or restart an index scan
 */
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->first = true;
	/* v and discarded are allocated in tmpCtx */
	so->v.tids = NULL;
	so->discarded = NULL;
	so->tuples = 0;
	INSTR_TIME_SET_CURRENT(so->scanStart);
	INSTR_TIME_SET_ZERO(so->inIndexTime);
	so->vectorSearchMs = 0;
	so->returnedTuples = 0;
	so->distanceComputations = 0;
	so->blksHitBefore = pgBufferUsage.shared_blks_hit;
	so->blksReadBefore = pgBufferUsage.shared_blks_read;
	so->idxBlksHit = 0;
	so->idxBlksRead = 0;
	so->topkTidCount = 0;
	so->previousDistance = -get_float8_infinity();
	if (so->traversalGuidance.phaseContext != NULL)
	{
		MemoryContextDelete(so->traversalGuidance.phaseContext);
		so->traversalGuidance.phaseContext = NULL;
	}
	MemoryContextReset(so->tmpCtx);
	MemoryContextReset(so->profileCtx);
	HnswInitIndexPageProfile(&so->indexPageProfile, so->profileCtx);
	so->pageItems = NULL;
	so->pageItemCount = 0;
	so->pageItemIndex = 0;
	so->pageItemCapacity = 0;
	so->pageAccessBatches = 0;
	so->pageAccessCandidates = 0;
	so->pageAccessPrefetches = 0;
	so->pageAccessDistanceRuns = 0;
	so->pageAccessDistinctPages = 0;
	so->pageAccessNoMergeBatches = 0;
	so->pageAccessDisabled = false;
	so->guidanceChecks = 0;
	so->guidanceMatches = 0;
	so->guidanceSkips = 0;
	so->heapTidReturns = 0;
	so->heapTidPageRuns = 0;
	so->heapTidDistinctPages = 0;
	so->heapTidLastBlock = InvalidBlockNumber;
	MemSet(&so->heapTidPages, 0, sizeof(so->heapTidPages));
	so->heapTidDistinctPagesExact = true;
	MemSet(&so->traversal, 0, sizeof(HnswTraversalProfile));
	so->abandonedGuidedTuples = 0;
	so->traversalGuidance.bridgeWork = 0;
	so->traversalGuidance.hopLimitReached = false;
	so->traversalGuidance.workLimitReached = false;
	so->traversalGuidance.maxScanReached = false;
	so->traversalGuidance.memoryLimitReached = false;
	so->traversalGuidance.invalidNeighbor = false;
	so->traversalGuidance.guidedResultCount = 0;
	so->traversalGuidance.bridgePendingAtTermination = 0;
	so->traversalGuidance.prioritizationEnabled =
		so->traversalGuidance.finalPath ==
		HNSW_TRAVERSAL_PATH_APPROXIMATE_PRIORITIZATION;
	HnswResetScanProfile();

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
static void
HnswAccumulateIndexCall(HnswScanOpaque so, int64 entryBlksHit, int64 entryBlksRead, instr_time entryTime)
{
	instr_time	exitTime;

	so->idxBlksHit += pgBufferUsage.shared_blks_hit - entryBlksHit;
	so->idxBlksRead += pgBufferUsage.shared_blks_read - entryBlksRead;
	INSTR_TIME_SET_CURRENT(exitTime);
	INSTR_TIME_ACCUM_DIFF(so->inIndexTime, exitTime, entryTime);
}

bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx;
	int64		entryBlksHit = pgBufferUsage.shared_blks_hit;
	int64		entryBlksRead = pgBufferUsage.shared_blks_read;
	instr_time	entryTime;

	INSTR_TIME_SET_CURRENT(entryTime);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first && !so->guidanceDecided)
	{
		/* This decision is final for the lifetime of this IndexScanDesc. */
		so->guidanceDecided = true;
		so->guidance = HnswGuidancePrepareForScan(scan, so->guidancePlan,
											 &so->plannerProof);
		HnswInitializeTraversalGuidance(so);
	}

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	if (so->first)
	{
		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->w = GetScanItems(scan);

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		ItemPointerData heaptid;
		int			pageAccessMode = hnsw_page_access;
		bool		guidanceChecked = false;

		CHECK_FOR_INTERRUPTS();

		if (pageAccessMode == HNSW_PAGE_ACCESS_REORDER &&
			(so->traversalGuidance.requested ||
			 hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT))
			pageAccessMode = HNSW_PAGE_ACCESS_PREFETCH;

		if (so->pageAccessDisabled)
			pageAccessMode = HNSW_PAGE_ACCESS_OFF;

		if (pageAccessMode != HNSW_PAGE_ACCESS_OFF && hnsw_page_window > 1)
		{
			if (so->pageItemIndex >= so->pageItemCount)
			{
				if (!HnswFillPageAccessBuffer(scan, pageAccessMode))
				{
					if (so->pageAccessDisabled)
						continue;
					break;
				}
			}

			{
				HnswPageAccessItem *item = &so->pageItems[so->pageItemIndex++];

				heaptid = item->heaptid;
				guidanceChecked = item->guidanceChecked;
			}
		}
		else if (!HnswGetNextHeapTid(scan, &heaptid, NULL))
			break;

		if (HnswScanUsesGuidanceValidation(so) && !guidanceChecked)
		{
			so->guidanceChecks++;
			so->traversal.guidanceChecks++;
			if (!HnswGuidanceAllowsTid(so->guidance, &heaptid))
			{
				so->guidanceSkips++;
				so->traversal.guidanceMisses++;
				so->traversal.guidedSuppressions++;
				so->traversal.heapTidsSuppressed++;
				continue;
			}
			so->guidanceMatches++;
			so->traversal.guidanceMatches++;
		}

		MemoryContextSwitchTo(oldCtx);

		if (so->topkTidCount < HNSW_PROFILE_MAX_TIDS)
			so->topkTids[so->topkTidCount++] = heaptid;
		so->returnedTuples++;
		HnswRecordHeapTid(so, &heaptid);

		scan->xs_heaptid = heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		HnswAccumulateIndexCall(so, entryBlksHit, entryBlksRead, entryTime);
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	HnswAccumulateIndexCall(so, entryBlksHit, entryBlksRead, entryTime);
	return false;
}

/*
 * End a scan and release resources
 */
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	if (so->traversalGuidance.requested &&
		so->traversalGuidance.finalPath == HNSW_TRAVERSAL_PATH_STOCK_BYPASS)
	{
		so->traversal.stockPhaseExpandedNodes = so->traversal.expandedNodes;
		so->traversal.stockPhaseDistanceComputations = so->distanceComputations;
	}

	if (!so->first)
	{
		HnswIndexPageProfile indexPageProfile;
		instr_time	now;
		instr_time	elapsed;
		double		totalScanMs;
		double		hnswSearchMs;
		double		heapFetchMs;
		int64		totalBlksHit;
		int64		totalBlksRead;

		indexPageProfile = so->indexPageProfile.profile;
		INSTR_TIME_SET_CURRENT(now);
		elapsed = now;
		INSTR_TIME_SUBTRACT(elapsed, so->scanStart);
		totalScanMs = INSTR_TIME_GET_MILLISEC(elapsed);
		hnswSearchMs = INSTR_TIME_GET_MILLISEC(so->inIndexTime);
		heapFetchMs = totalScanMs - hnswSearchMs;
		if (heapFetchMs < 0)
			heapFetchMs = 0;
		totalBlksHit = pgBufferUsage.shared_blks_hit - so->blksHitBefore;
		totalBlksRead = pgBufferUsage.shared_blks_read - so->blksReadBefore;

		hnsw_last_profile.valid = true;
		hnsw_last_profile.totalScanMs += totalScanMs;
		hnsw_last_profile.hnswSearchMs += hnswSearchMs;
		hnsw_last_profile.heapFetchMs += heapFetchMs;
		hnsw_last_profile.vectorSearchMs += so->vectorSearchMs;
		hnsw_last_profile.visitedTuples += so->tuples + so->abandonedGuidedTuples;
		hnsw_last_profile.returnedTuples += so->returnedTuples;
		hnsw_last_profile.distanceComputations += so->distanceComputations;
		hnsw_last_profile.pageAccessBatches += so->pageAccessBatches;
		hnsw_last_profile.pageAccessCandidates += so->pageAccessCandidates;
		hnsw_last_profile.pageAccessPrefetches += so->pageAccessPrefetches;
		hnsw_last_profile.pageAccessDistanceRuns += so->pageAccessDistanceRuns;
		hnsw_last_profile.pageAccessDistinctPages += so->pageAccessDistinctPages;
		hnsw_last_profile.guidanceChecks += so->guidanceChecks;
		hnsw_last_profile.guidanceMatches += so->guidanceMatches;
		hnsw_last_profile.guidanceSkips += so->guidanceSkips;
		hnsw_last_profile.traversal.expandedNodes += so->traversal.expandedNodes;
		hnsw_last_profile.traversal.neighborsExamined += so->traversal.neighborsExamined;
		hnsw_last_profile.traversal.guidanceChecks += so->traversal.guidanceChecks;
		hnsw_last_profile.traversal.guidanceMatches += so->traversal.guidanceMatches;
		hnsw_last_profile.traversal.guidanceMisses += so->traversal.guidanceMisses;
		hnsw_last_profile.traversal.neighborGuidanceChecks += so->traversal.neighborGuidanceChecks;
		hnsw_last_profile.traversal.neighborGuidanceMatches += so->traversal.neighborGuidanceMatches;
		hnsw_last_profile.traversal.neighborGuidanceMisses += so->traversal.neighborGuidanceMisses;
		hnsw_last_profile.traversal.preDistanceChecks += so->traversal.preDistanceChecks;
		hnsw_last_profile.traversal.preDistanceMatches += so->traversal.preDistanceMatches;
		hnsw_last_profile.traversal.preDistanceMisses += so->traversal.preDistanceMisses;
		hnsw_last_profile.traversal.attemptedDistanceComputationsAvoided +=
			so->traversal.attemptedDistanceComputationsAvoided;
		hnsw_last_profile.traversal.distanceComputationsAvoided += so->traversal.distanceComputationsAvoided;
		hnsw_last_profile.traversal.missBridgeNodes += so->traversal.missBridgeNodes;
		hnsw_last_profile.traversal.missBridgeEdges += so->traversal.missBridgeEdges;
		hnsw_last_profile.traversal.maxMissBridgeHops = Max(
			hnsw_last_profile.traversal.maxMissBridgeHops,
			so->traversal.maxMissBridgeHops);
		hnsw_last_profile.traversal.bridgePendingAtTermination +=
			so->traversal.bridgePendingAtTermination;
		hnsw_last_profile.traversal.guidedExpandedNodes += so->traversal.guidedExpandedNodes;
		hnsw_last_profile.traversal.guidedPhaseDistanceComputations += so->traversal.guidedPhaseDistanceComputations;
		hnsw_last_profile.traversal.stockPhaseExpandedNodes += so->traversal.stockPhaseExpandedNodes;
		hnsw_last_profile.traversal.stockPhaseDistanceComputations += so->traversal.stockPhaseDistanceComputations;
		hnsw_last_profile.traversal.stockBypassRequests += so->traversal.stockBypassRequests;
		hnsw_last_profile.traversal.fallbackRequests += so->traversal.fallbackRequests;
		hnsw_last_profile.traversal.fallbackStockExpandedNodes += so->traversal.fallbackStockExpandedNodes;
		hnsw_last_profile.traversal.fallbackStockDistanceComputations += so->traversal.fallbackStockDistanceComputations;
		hnsw_last_profile.traversal.matchingExpanded += so->traversal.matchingExpanded;
		hnsw_last_profile.traversal.bridgeExpanded += so->traversal.bridgeExpanded;
		hnsw_last_profile.traversal.matchFrontierPops += so->traversal.matchFrontierPops;
		hnsw_last_profile.traversal.noBridgeFrontierPops += so->traversal.noBridgeFrontierPops;
		hnsw_last_profile.traversal.noBridgeDeferred += so->traversal.noBridgeDeferred;
		hnsw_last_profile.traversal.maxNoBridgeDebt = Max(
			hnsw_last_profile.traversal.maxNoBridgeDebt,
			so->traversal.maxNoBridgeDebt);
		hnsw_last_profile.traversal.noBridgeExpansions += so->traversal.noBridgeExpansions;
		hnsw_last_profile.traversal.dualFrontierTerminationChecks +=
			so->traversal.dualFrontierTerminationChecks;
		hnsw_last_profile.traversal.dualFrontierTerminationChecksWithBoth +=
			so->traversal.dualFrontierTerminationChecksWithBoth;
		hnsw_last_profile.traversal.dualFrontierTerminations +=
			so->traversal.dualFrontierTerminations;
		hnsw_last_profile.traversal.dualFrontierTerminationsWithBoth +=
			so->traversal.dualFrontierTerminationsWithBoth;
		hnsw_last_profile.traversal.candidateAdmissions += so->traversal.candidateAdmissions;
		hnsw_last_profile.traversal.resultAdmissions += so->traversal.resultAdmissions;
		hnsw_last_profile.traversal.guidedAdmissions += so->traversal.guidedAdmissions;
		hnsw_last_profile.traversal.guidedSuppressions += so->traversal.guidedSuppressions;
		hnsw_last_profile.traversal.heapTidsSuppressed += so->traversal.heapTidsSuppressed;
		hnsw_last_profile.traversal.stopDeferrals += so->traversal.stopDeferrals;
		hnsw_last_profile.traversal.discardedPushes += so->traversal.discardedPushes;
		hnsw_last_profile.traversal.discardedPops += so->traversal.discardedPops;
		hnsw_last_profile.traversal.initialBatches += so->traversal.initialBatches;
		hnsw_last_profile.traversal.resumeBatches += so->traversal.resumeBatches;
		hnsw_last_profile.traversal.strictOrderDrops += so->traversal.strictOrderDrops;
		hnsw_last_profile.traversal.stockTerminations += so->traversal.stockTerminations;
		hnsw_last_profile.traversal.maxScanTerminations += so->traversal.maxScanTerminations;
		hnsw_last_profile.traversal.exhaustedTerminations += so->traversal.exhaustedTerminations;
		hnsw_last_profile.indexPageNeighborLoads += indexPageProfile.neighborLoads;
		hnsw_last_profile.indexPageNeighborRuns += indexPageProfile.neighborRuns;
		hnsw_last_profile.indexPageNeighborDistinctPages += indexPageProfile.neighborDistinctPages;
		hnsw_last_profile.indexPageElementLoads += indexPageProfile.elementLoads;
		hnsw_last_profile.indexPageElementRuns += indexPageProfile.elementRuns;
		hnsw_last_profile.indexPageElementDistinctPages += indexPageProfile.elementDistinctPages;
		hnsw_last_profile.indexPagePrefetches += indexPageProfile.prefetches;
		hnsw_last_profile.indexPageLoads += indexPageProfile.loads;
		hnsw_last_profile.indexPageRuns += indexPageProfile.runs;
		hnsw_last_profile.indexPageDistinctPages += indexPageProfile.distinctPages;
		hnsw_last_profile.indexPageLastBlock = indexPageProfile.lastBlock;
		if (!indexPageProfile.distinctCountsExact)
			hnsw_last_profile.indexPageDistinctPagesExact = false;
		hnsw_last_profile.heapTidReturns += so->heapTidReturns;
		hnsw_last_profile.heapTidPageRuns += so->heapTidPageRuns;
		hnsw_last_profile.heapTidDistinctPages += so->heapTidDistinctPages;
		hnsw_last_profile.heapTidDistinctPagesExact =
			hnsw_last_profile.heapTidDistinctPagesExact && so->heapTidDistinctPagesExact;
		if (!indexPageProfile.distinctCountsExact)
			hnsw_last_profile.indexPageDistinctCountsExact = false;
		hnsw_last_profile.blksHitBefore = so->blksHitBefore;
		hnsw_last_profile.blksHitAfter = pgBufferUsage.shared_blks_hit;
		hnsw_last_profile.blksReadBefore = so->blksReadBefore;
		hnsw_last_profile.blksReadAfter = pgBufferUsage.shared_blks_read;
		hnsw_last_profile.idxBlksHit += so->idxBlksHit;
		hnsw_last_profile.idxBlksRead += so->idxBlksRead;
		hnsw_last_profile.heapBlksHit += totalBlksHit - so->idxBlksHit;
		hnsw_last_profile.heapBlksRead += totalBlksRead - so->idxBlksRead;
		hnsw_last_profile.topkTidCount = Min(so->topkTidCount, HNSW_PROFILE_MAX_TIDS);
		memcpy(hnsw_last_profile.topkTids, so->topkTids, sizeof(ItemPointerData) * hnsw_last_profile.topkTidCount);

		/* Adaptive admission records the completed stock or guided scan here;
		 * fragment construction is intentionally confined to activation. */
		HnswGuidanceRecordScan(HnswScanHeapOid(scan),
							   so->tuples + so->abandonedGuidedTuples,
							   so->guidanceChecks, so->guidanceSkips,
							   heapFetchMs, totalScanMs);
	}

	HnswGuidanceEndScan(so->guidance);
	so->guidance = NULL;
	hnsw_last_profile.plannerProof = so->plannerProof;
	hnsw_last_profile.traversalFinalPath = so->traversalGuidance.finalPath;
	hnsw_last_profile.traversalStockBypassReason =
		so->traversalGuidance.stockBypassReason;
	hnsw_last_profile.traversalAdmissionReason =
		so->traversalGuidance.admissionReason;
	hnsw_last_profile.traversalFallbackReason =
		so->traversalGuidance.fallbackReason;
	hnsw_last_profile.traversalEstimatedSkipRateValid =
		so->traversalGuidance.estimatedSkipRateValid;
	hnsw_last_profile.traversalEstimatedSkipRate =
		so->traversalGuidance.estimatedSkipRate;
	hnsw_last_profile.traversalPrioritizationBurst =
		so->traversalGuidance.burst;
	hnsw_last_profile.iterativeScan = so->traversalGuidance.iterativeScan;
	hnsw_last_profile.filterStrategy = so->traversalGuidance.filterStrategy;
	if (hnsw_last_profile.plannerProofCount < HNSW_PROFILE_MAX_PROOFS)
		hnsw_last_profile.plannerProofs[hnsw_last_profile.plannerProofCount++] =
			so->plannerProof;
	else
		hnsw_last_profile.plannerProofsTruncated = true;
	MemoryContextDelete(so->tmpCtx);
	MemoryContextDelete(so->profileCtx);

	pfree(so);
	scan->opaque = NULL;
}

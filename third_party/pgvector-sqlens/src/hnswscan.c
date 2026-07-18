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
GetScanItems(IndexScanDesc scan, Datum value)
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
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));
	so->distanceComputations++;

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		instr_time start;
		instr_time elapsed;

		INSTR_TIME_SET_CURRENT(start);
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL, &so->distanceComputations);
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
		next = HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m, false, NULL, &so->v,
							   hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples, &so->distanceComputations);
		INSTR_TIME_SET_CURRENT(elapsed);
		INSTR_TIME_SUBTRACT(elapsed, start);
		so->vectorSearchMs += INSTR_TIME_GET_MILLISEC(elapsed);
		return next;
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

		ep = lappend(ep, sc);
	}

	{
		instr_time start;
		instr_time elapsed;
		List	   *next;

		INSTR_TIME_SET_CURRENT(start);
		next = HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false,
							  &so->tuples, &so->distanceComputations);
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
	hnsw_last_profile.indexPageNeighborLoads = 0;
	hnsw_last_profile.indexPageNeighborRuns = 0;
	hnsw_last_profile.indexPageNeighborDistinctPages = 0;
	hnsw_last_profile.indexPageElementLoads = 0;
	hnsw_last_profile.indexPageElementRuns = 0;
	hnsw_last_profile.indexPageElementDistinctPages = 0;
	hnsw_last_profile.indexPagePrefetches = 0;
	hnsw_last_profile.blksHitBefore = 0;
	hnsw_last_profile.blksHitAfter = 0;
	hnsw_last_profile.blksReadBefore = 0;
	hnsw_last_profile.blksReadAfter = 0;
	hnsw_last_profile.idxBlksHit = 0;
	hnsw_last_profile.idxBlksRead = 0;
	hnsw_last_profile.heapBlksHit = 0;
	hnsw_last_profile.heapBlksRead = 0;
	hnsw_last_profile.topkTidCount = 0;
	HnswResetIndexPageProfile();
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
				continue;

			so->previousDistance = sc->distance;
		}

		*heaptid = *tid;
		if (distance != NULL)
			*distance = sc->distance;
		return true;
	}
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
		if (HnswGuidanceIsActive())
		{
			item->guidanceChecked = true;
			so->guidanceChecks++;
			if (!HnswGuidanceAllowsTid(&item->heaptid))
			{
				so->guidanceSkips++;
				continue;
			}
			so->guidanceMatches++;
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

		/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

	scan->opaque = so;

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
	MemoryContextReset(so->tmpCtx);
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
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	int64		entryBlksHit = pgBufferUsage.shared_blks_hit;
	int64		entryBlksRead = pgBufferUsage.shared_blks_read;
	instr_time	entryTime;

	INSTR_TIME_SET_CURRENT(entryTime);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

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

		/* Get scan value */
		value = GetScanValue(scan);

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->w = GetScanItems(scan, value);

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

		if (pageAccessMode == HNSW_PAGE_ACCESS_REORDER && hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
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

		if (HnswGuidanceIsActive() && !guidanceChecked)
		{
			so->guidanceChecks++;
			if (!HnswGuidanceAllowsTid(&heaptid))
			{
				so->guidanceSkips++;
				continue;
			}
			so->guidanceMatches++;
		}

		MemoryContextSwitchTo(oldCtx);

		if (so->topkTidCount < HNSW_PROFILE_MAX_TIDS)
			so->topkTids[so->topkTidCount++] = heaptid;
		so->returnedTuples++;

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

	if (so->tuples > 0 || so->returnedTuples > 0)
	{
		HnswIndexPageProfile indexPageProfile;
		instr_time	now;
		instr_time	elapsed;
		double		totalScanMs;
		double		hnswSearchMs;
		double		heapFetchMs;
		int64		totalBlksHit;
		int64		totalBlksRead;

		HnswGetIndexPageProfile(&indexPageProfile);
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
		hnsw_last_profile.visitedTuples += so->tuples;
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
		hnsw_last_profile.indexPageNeighborLoads += indexPageProfile.neighborLoads;
		hnsw_last_profile.indexPageNeighborRuns += indexPageProfile.neighborRuns;
		hnsw_last_profile.indexPageNeighborDistinctPages += indexPageProfile.neighborDistinctPages;
		hnsw_last_profile.indexPageElementLoads += indexPageProfile.elementLoads;
		hnsw_last_profile.indexPageElementRuns += indexPageProfile.elementRuns;
		hnsw_last_profile.indexPageElementDistinctPages += indexPageProfile.elementDistinctPages;
		hnsw_last_profile.indexPagePrefetches += indexPageProfile.prefetches;
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
	}

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}

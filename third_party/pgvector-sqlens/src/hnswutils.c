#include "postgres.h"

#include <math.h>

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "common/hashfn.h"
#include "fmgr.h"
#include "hnsw.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "nodes/pg_list.h"
#include "port/atomics.h"
#include "sparsevec.h"
#include "storage/bufmgr.h"
#include "utils/datum.h"
#include "utils/memdebug.h"
#include "utils/rel.h"
#include "vector.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

static bool
HnswElementMatchesGuidance(HnswElement element, HnswScanGuidance *guidance,
						   HnswTraversalProfile *profile, bool preDistance)
{
	if (!HnswGuidanceIsActiveForScan(guidance))
		return true;

	if (profile != NULL)
	{
		profile->guidanceChecks++;
		profile->neighborGuidanceChecks++;
		if (preDistance)
			profile->preDistanceChecks++;
	}
	for (int i = 0; i < element->heaptidsLength; i++)
	{
		if (HnswGuidanceAllowsTid(guidance, &element->heaptids[i]))
		{
			if (profile != NULL)
			{
				profile->guidanceMatches++;
				profile->neighborGuidanceMatches++;
				if (preDistance)
					profile->preDistanceMatches++;
			}
			return true;
		}
	}

	if (profile != NULL)
	{
		profile->guidanceMisses++;
		profile->neighborGuidanceMisses++;
		if (preDistance)
			profile->preDistanceMisses++;
	}
	return false;
}

static int
CompareBlockNumbers(const void *a, const void *b)
{
	BlockNumber ba = *((const BlockNumber *) a);
	BlockNumber bb = *((const BlockNumber *) b);

	if (ba < bb)
		return -1;
	if (ba > bb)
		return 1;
	return 0;
}

static bool HnswIndexPageSetAdd(HnswIndexPageProfileState *state,
								HnswIndexPageSet *set, BlockNumber block);

void
HnswInitIndexPageProfile(HnswIndexPageProfileState *state,
						 MemoryContext context)
{
	MemSet(state, 0, sizeof(*state));
	state->context = context;
	state->lastNeighborBlock = InvalidBlockNumber;
	state->lastElementBlock = InvalidBlockNumber;
	state->profile.lastBlock = InvalidBlockNumber;
	state->profile.distinctCountsExact = true;
}

void
HnswRecordHeapTid(HnswScanOpaque scan, ItemPointer tid)
{
	HnswIndexPageProfileState state;
	BlockNumber block;

	if (scan == NULL || tid == NULL || !ItemPointerIsValid(tid))
		return;
	block = ItemPointerGetBlockNumber(tid);
	MemSet(&state, 0, sizeof(state));
	state.context = scan->profileCtx;
	state.profile.distinctCountsExact = true;
	scan->heapTidReturns++;
	if (block != scan->heapTidLastBlock)
		scan->heapTidPageRuns++;
	scan->heapTidLastBlock = block;
	if (HnswIndexPageSetAdd(&state, &scan->heapTidPages, block))
		scan->heapTidDistinctPages++;
	if (!state.profile.distinctCountsExact)
		scan->heapTidDistinctPagesExact = false;
}

static bool
HnswIndexPageSetAdd(HnswIndexPageProfileState *state, HnswIndexPageSet *set,
						BlockNumber block)
{
	uint32		value;
	uint32		slot;

	if (state == NULL || !BlockNumberIsValid(block))
		return false;
	if (set->slots == NULL)
		set->slots = MemoryContextAllocZero(state->context,
			sizeof(uint32) * HNSW_INDEX_PAGE_UNIQUE_SLOTS);

	value = ((uint32) block) + 1;
	slot = hash_uint32((uint32) block) & (HNSW_INDEX_PAGE_UNIQUE_SLOTS - 1);
	while (set->slots[slot] != 0)
	{
		if (set->slots[slot] == value)
			return false;
		slot = (slot + 1) & (HNSW_INDEX_PAGE_UNIQUE_SLOTS - 1);
	}

	if (set->count >= HNSW_INDEX_PAGE_UNIQUE_LIMIT)
	{
		state->profile.distinctCountsExact = false;
		return false;
	}

	set->slots[slot] = value;
	set->count++;
	return true;
}

static void
HnswRecordIndexNeighborPage(HnswIndexPageProfileState *state,
							BlockNumber block)
{
	state->profile.neighborLoads++;
	if (block != state->lastNeighborBlock)
	{
		state->profile.neighborRuns++;
	}
	state->lastNeighborBlock = block;
	if (HnswIndexPageSetAdd(state, &state->neighborPages, block))
		state->profile.neighborDistinctPages++;
}

static void
HnswRecordIndexPage(HnswIndexPageProfileState *state, BlockNumber block)
{
	if (state == NULL || !BlockNumberIsValid(block))
		return;
	state->profile.loads++;
	if (block != state->profile.lastBlock)
		state->profile.runs++;
	state->profile.lastBlock = block;
	if (HnswIndexPageSetAdd(state, &state->pages, block))
		state->profile.distinctPages++;
}

static void
HnswPrefetchUnvisitedIndexPages(Relation index, HnswUnvisited *unvisited,
								int unvisitedLength,
								HnswIndexPageProfileState *state)
{
	BlockNumber blocks[HNSW_MAX_M * 2];
	int			blockCount = 0;
	BlockNumber previousBlock = InvalidBlockNumber;

	if (unvisitedLength <= 0)
		return;

	for (int i = 0; i < unvisitedLength; i++)
	{
		ItemPointer indextid = &unvisited[i].indextid;
		BlockNumber block;

		if (!ItemPointerIsValid(indextid))
			continue;

		block = ItemPointerGetBlockNumber(indextid);
		state->profile.elementLoads++;
		if (block != state->lastElementBlock)
			state->profile.elementRuns++;
		state->lastElementBlock = block;
		if (HnswIndexPageSetAdd(state, &state->elementPages, block))
			state->profile.elementDistinctPages++;
		blocks[blockCount++] = block;
	}

	if (blockCount == 0)
		return;

	qsort(blocks, blockCount, sizeof(BlockNumber), CompareBlockNumbers);
	for (int i = 0; i < blockCount; i++)
	{
		BlockNumber block = blocks[i];

		if (i > 0 && block == previousBlock)
			continue;

		if (hnsw_index_page_access == HNSW_INDEX_PAGE_ACCESS_PREFETCH)
		{
			PrefetchBuffer(index, MAIN_FORKNUM, block);
			state->profile.prefetches++;
		}
		previousBlock = block;
	}
}

#if PG_VERSION_NUM < 170000
static inline uint64
murmurhash64(uint64 data)
{
	uint64		h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}
#endif

/* TID hash table */
static uint32
hash_tid(ItemPointerData tid)
{
	union
	{
		uint64		i;
		ItemPointerData tid;
	}			x;

	/* Initialize unused bytes */
	x.i = 0;
	x.tid = tid;

	return murmurhash64(x.i);
}

#define SH_PREFIX		tidhash
#define SH_ELEMENT_TYPE	TidHashEntry
#define SH_KEY_TYPE		ItemPointerData
#define	SH_KEY			tid
#define SH_HASH_KEY(tb, key)	hash_tid(key)
#define SH_EQUAL(tb, a, b)		ItemPointerEquals(&a, &b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Pointer hash table */
static uint32
hash_pointer(uintptr_t ptr)
{
#if SIZEOF_VOID_P == 8
	return murmurhash64((uint64) ptr);
#else
	return murmurhash32((uint32) ptr);
#endif
}

#define SH_PREFIX		pointerhash
#define SH_ELEMENT_TYPE	PointerHashEntry
#define SH_KEY_TYPE		uintptr_t
#define	SH_KEY			ptr
#define SH_HASH_KEY(tb, key)	hash_pointer(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Offset hash table */
static uint32
hash_offset(Size offset)
{
#if SIZEOF_SIZE_T == 8
	return murmurhash64((uint64) offset);
#else
	return murmurhash32((uint32) offset);
#endif
}

#define SH_PREFIX		offsethash
#define SH_ELEMENT_TYPE	OffsetHashEntry
#define SH_KEY_TYPE		Size
#define	SH_KEY			offset
#define SH_HASH_KEY(tb, key)	hash_offset(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/*
 * Get the max number of connections in an upper layer for each element in the index
 */
int
HnswGetM(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->m;

	return HNSW_DEFAULT_M;
}

/*
 * Get the size of the dynamic candidate list in the index
 */
int
HnswGetEfConstruction(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->efConstruction;

	return HNSW_DEFAULT_EF_CONSTRUCTION;
}

/*
 * Get proc
 */
FmgrInfo *
HnswOptionalProcInfo(Relation index, uint16 procnum)
{
	if (!OidIsValid(index_getprocid(index, 1, procnum)))
		return NULL;

	return index_getprocinfo(index, 1, procnum);
}

/*
 * Init support functions
 */
void
HnswInitSupport(HnswSupport * support, Relation index)
{
	support->procinfo = index_getprocinfo(index, 1, HNSW_DISTANCE_PROC);
	support->collation = index->rd_indcollation[0];
	support->normprocinfo = HnswOptionalProcInfo(index, HNSW_NORM_PROC);
}

/*
 * Normalize value
 */
Datum
HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value)
{
	return DirectFunctionCall1Coll(typeInfo->normalize, collation, value);
}

/*
 * Check if non-zero norm
 */
bool
HnswCheckNorm(HnswSupport * support, Datum value)
{
	return DatumGetFloat8(FunctionCall1Coll(support->normprocinfo, support->collation, value)) > 0;
}

/*
 * New buffer
 */
Buffer
HnswNewBuffer(Relation index, ForkNumber forkNum)
{
	Buffer		buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

/*
 * Init page
 */
void
HnswInitPage(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(HnswPageOpaqueData));
	HnswPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	HnswPageGetOpaque(page)->page_id = HNSW_PAGE_ID;
}

/*
 * Allocate a neighbor array
 */
HnswNeighborArray *
HnswInitNeighborArray(int lm, HnswAllocator * allocator)
{
	HnswNeighborArray *a = HnswAlloc(allocator, HNSW_NEIGHBOR_ARRAY_SIZE(lm));

	a->length = 0;
	a->closerSet = false;
	return a;
}

/*
 * Allocate neighbors
 */
void
HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * allocator)
{
	int			level = element->level;
	HnswNeighborArrayPtr *neighborList = (HnswNeighborArrayPtr *) HnswAlloc(allocator, sizeof(HnswNeighborArrayPtr) * (level + 1));

	HnswPtrStore(base, element->neighbors, neighborList);

	for (int lc = 0; lc <= level; lc++)
		HnswPtrStore(base, neighborList[lc], HnswInitNeighborArray(HnswGetLayerM(m, lc), allocator));
}

/*
 * Allocate memory from the allocator
 */
void *
HnswAlloc(HnswAllocator * allocator, Size size)
{
	if (allocator)
		return (*(allocator)->alloc) (size, (allocator)->state);

	return palloc(size);
}

/*
 * Allocate an element
 */
HnswElement
HnswInitElement(char *base, ItemPointer heaptid, int m, double ml, int maxLevel, HnswAllocator * allocator)
{
	HnswElement element = HnswAlloc(allocator, sizeof(HnswElementData));

	int			level = (int) (-log(RandomDouble()) * ml);

	/* Cap level */
	if (level > maxLevel)
		level = maxLevel;

	element->heaptidsLength = 0;
	HnswAddHeapTid(element, heaptid);

	element->level = level;
	element->deleted = 0;
	/* Start at one to make it easier to find issues */
	element->version = 1;

	HnswInitNeighbors(base, element, m, allocator);

	HnswPtrStore(base, element->value, (char *) NULL);

	return element;
}

/*
 * Add a heap TID to an element
 */
void
HnswAddHeapTid(HnswElement element, ItemPointer heaptid)
{
	element->heaptids[element->heaptidsLength++] = *heaptid;
}

/*
 * Allocate an element from block and offset numbers
 */
HnswElement
HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno)
{
	HnswElement element = palloc(sizeof(HnswElementData));
	char	   *base = NULL;

	element->blkno = blkno;
	element->offno = offno;
	HnswPtrStore(base, element->neighbors, (HnswNeighborArrayPtr *) NULL);
	HnswPtrStore(base, element->value, (char *) NULL);
	return element;
}

/*
 * Get the metapage info
 */
void
HnswGetMetaPageInfo(Relation index, int *m, HnswElement * entryPoint)
{
	HnswGetMetaPageInfoTracked(index, m, entryPoint, NULL);
}

void
HnswGetMetaPageInfoTracked(Relation index, int *m, HnswElement * entryPoint,
						   HnswIndexPageProfileState *profile)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	HnswRecordIndexPage(profile, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (unlikely(metap->magicNumber != HNSW_MAGIC_NUMBER))
		elog(ERROR, "hnsw index is not valid");

	if (m != NULL)
		*m = metap->m;

	if (entryPoint != NULL)
	{
		if (BlockNumberIsValid(metap->entryBlkno))
		{
			*entryPoint = HnswInitElementFromBlock(metap->entryBlkno, metap->entryOffno);
			(*entryPoint)->level = metap->entryLevel;
		}
		else
			*entryPoint = NULL;
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Get the entry point
 */
HnswElement
HnswGetEntryPoint(Relation index)
{
	HnswElement entryPoint;

	HnswGetMetaPageInfo(index, NULL, &entryPoint);

	return entryPoint;
}

/*
 * Update the metapage info
 */
static void
HnswUpdateMetaPageInfo(Page page, int updateEntry, HnswElement entryPoint, BlockNumber insertPage)
{
	HnswMetaPage metap = HnswPageGetMeta(page);

	if (updateEntry)
	{
		if (entryPoint == NULL)
		{
			metap->entryBlkno = InvalidBlockNumber;
			metap->entryOffno = InvalidOffsetNumber;
			metap->entryLevel = -1;
		}
		else if (entryPoint->level > metap->entryLevel || updateEntry == HNSW_UPDATE_ENTRY_ALWAYS)
		{
			metap->entryBlkno = entryPoint->blkno;
			metap->entryOffno = entryPoint->offno;
			metap->entryLevel = entryPoint->level;
		}
	}

	if (BlockNumberIsValid(insertPage))
		metap->insertPage = insertPage;
}

/*
 * Update the metapage
 */
void
HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;

	buf = ReadBufferExtended(index, forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	HnswUpdateMetaPageInfo(page, updateEntry, entryPoint, insertPage);

	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Form index value
 */
bool
HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support)
{
	/* Detoast once for all calls */
	Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Check value */
	if (typeInfo->checkValue != NULL)
		typeInfo->checkValue(DatumGetPointer(value));

	/* Normalize if needed */
	if (support->normprocinfo != NULL)
	{
		if (!HnswCheckNorm(support, value))
			return false;

		value = HnswNormValue(typeInfo, support->collation, value);
	}

	*out = value;

	return true;
}

/*
 * Set element tuple, except for neighbor info
 */
void
HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element)
{
	Pointer		valuePtr = HnswPtrAccess(base, element->value);

	etup->type = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level = element->level;
	etup->deleted = 0;
	etup->version = element->version;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (i < element->heaptidsLength)
			etup->heaptids[i] = element->heaptids[i];
		else
			ItemPointerSetInvalid(&etup->heaptids[i]);
	}
	memcpy(&etup->data, valuePtr, VARSIZE_ANY(valuePtr));
}

/*
 * Set neighbor tuple
 */
void
HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m)
{
	int			idx = 0;

	ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;

	for (int lc = e->level; lc >= 0; lc--)
	{
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);
		int			lm = HnswGetLayerM(m, lc);

		for (int i = 0; i < lm; i++)
		{
			ItemPointer indextid = &ntup->indextids[idx++];

			if (i < neighbors->length)
			{
				HnswCandidate *hc = &neighbors->items[i];
				HnswElement hce = HnswPtrAccess(base, hc->element);

				ItemPointerSet(indextid, hce->blkno, hce->offno);
			}
			else
				ItemPointerSetInvalid(indextid);
		}
	}

	ntup->count = idx;
	ntup->version = e->version;
}

/*
 * Load an element from a tuple
 */
void
HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec)
{
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;

	if (loadHeaptids)
	{
		for (int i = 0; i < HNSW_HEAPTIDS; i++)
		{
			/* Can stop at first invalid */
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			HnswAddHeapTid(element, &etup->heaptids[i]);
		}
	}

	if (loadVec)
	{
		char	   *base = NULL;
		Datum		value = datumCopy(PointerGetDatum(&etup->data), false, -1);

		HnswPtrStore(base, element->value, (char *) DatumGetPointer(value));
	}
}

/*
 * Calculate the distance between values
 */
static inline double
HnswGetDistance(Datum a, Datum b, HnswSupport * support)
{
	return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));
}

/*
 * Load an element and optionally get its distance from q
 */
static void
HnswLoadElementImpl(BlockNumber blkno, OffsetNumber offno, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance, HnswElement * element, HnswIndexPageProfileState *profile)
{
	Buffer		buf;
	Page		page;
	HnswElementTuple etup;

	/* Read vector */
	buf = ReadBuffer(index, blkno);
	HnswRecordIndexPage(profile, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));

	Assert(HnswIsElementTuple(etup));

	/* Calculate distance */
	if (distance != NULL)
	{
		if (DatumGetPointer(q->value) == NULL)
			*distance = 0;
		else
			*distance = HnswGetDistance(q->value, PointerGetDatum(&etup->data), support);
	}

	/* Load element */
	if (distance == NULL || maxDistance == NULL || *distance < *maxDistance)
	{
		if (*element == NULL)
			*element = HnswInitElementFromBlock(blkno, offno);

		HnswLoadElementFromTuple(*element, etup, true, loadVec);
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Load an element and optionally get its distance from q
 */
void
HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance)
{
	HnswLoadElementImpl(element->blkno, element->offno, distance, q, index, support, loadVec, maxDistance, &element, NULL);
}

/*
 * Get the distance for an element
 */
static double
GetElementDistance(char *base, HnswElement element, HnswQuery * q, HnswSupport * support)
{
	Datum		value = HnswGetValue(base, element);

	return HnswGetDistance(q->value, value, support);
}

/*
 * Allocate a search candidate
 */
static HnswSearchCandidate *
HnswInitSearchCandidate(char *base, HnswElement element, double distance)
{
	HnswSearchCandidate *sc = palloc(sizeof(HnswSearchCandidate));

	HnswPtrStore(base, sc->element, element);
	sc->distance = distance;
	sc->matchesGuidance = true;
	return sc;
}

/*
 * Create a candidate for the entry point
 */
HnswSearchCandidate *
HnswEntryCandidate(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec)
{
	return HnswEntryCandidateTracked(base, entryPoint, q, index, support, loadVec, NULL);
}

HnswSearchCandidate *
HnswEntryCandidateTracked(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, HnswIndexPageProfileState *profile)
{
	bool		inMemory = index == NULL;
	double		distance;

	if (inMemory)
		distance = GetElementDistance(base, entryPoint, q, support);
	else
		HnswLoadElementImpl(entryPoint->blkno, entryPoint->offno, &distance, q,
							index, support, loadVec, NULL, &entryPoint, profile);

	return HnswInitSearchCandidate(base, entryPoint, distance);
}

typedef struct HnswSearchHeapCompareContext
{
	char	   *base;
} HnswSearchHeapCompareContext;

static int CompareSeededElementKeys(HnswElement a, HnswElement b);

/*
 * Compare candidate distances
 */
static int
CompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	const HnswSearchCandidate *candidateA = HnswGetSearchCandidateConst(c_node, a);
	const HnswSearchCandidate *candidateB = HnswGetSearchCandidateConst(c_node, b);

	if (candidateA->distance < candidateB->distance)
		return 1;

	if (candidateA->distance > candidateB->distance)
		return -1;

	if (arg != NULL)
	{
		HnswSearchHeapCompareContext *context = arg;
		HnswElement elementA = HnswPtrAccess(context->base, candidateA->element);
		HnswElement elementB = HnswPtrAccess(context->base, candidateB->element);

		return -CompareSeededElementKeys(elementA, elementB);
	}

	return 0;
}

/*
 * Compare discarded candidate distances
 */
static int
CompareNearestDiscardedCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare candidate distances
 */
static int
CompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	const HnswSearchCandidate *candidateA = HnswGetSearchCandidateConst(w_node, a);
	const HnswSearchCandidate *candidateB = HnswGetSearchCandidateConst(w_node, b);

	if (candidateA->distance < candidateB->distance)
		return -1;

	if (candidateA->distance > candidateB->distance)
		return 1;

	if (arg != NULL)
	{
		HnswSearchHeapCompareContext *context = arg;
		HnswElement elementA = HnswPtrAccess(context->base, candidateA->element);
		HnswElement elementB = HnswPtrAccess(context->base, candidateB->element);

		return CompareSeededElementKeys(elementA, elementB);
	}

	return 0;
}

/*
 * Compare guided candidate distances
 */
static int
CompareFurthestGuidedCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(g_node, a)->distance < HnswGetSearchCandidateConst(g_node, b)->distance)
		return -1;

	if (HnswGetSearchCandidateConst(g_node, a)->distance > HnswGetSearchCandidateConst(g_node, b)->distance)
		return 1;

	return 0;
}

static inline bool CountElement(HnswElement skipElement, HnswElement e);

static inline void
HnswAddGuidedCandidate(pairingheap *G, List **guidedCandidates, List *wCandidates, pairingheap *discarded, int *glen, int ef, HnswSearchCandidate *sc, HnswElement element, HnswElement skipElement, HnswTraversalProfile *profile)
{
	if (G == NULL || !sc->matchesGuidance || !CountElement(skipElement, element))
		return;

	pairingheap_add(G, &sc->g_node);
	*guidedCandidates = lappend(*guidedCandidates, sc);
	(*glen)++;
	if (profile != NULL)
		profile->guidedAdmissions++;

	if (*glen > ef)
	{
		HnswSearchCandidate *evicted = HnswGetSearchCandidate(g_node, pairingheap_remove_first(G));

		*guidedCandidates = list_delete_ptr(*guidedCandidates, evicted);
		(*glen)--;

		/*
		 * A G-evicted candidate may already have left W.  Preserve it for a
		 * resume in that case; candidates still owned by W are moved below.
		 */
		if (discarded != NULL && !list_member_ptr(wCandidates, evicted))
		{
			pairingheap_add(discarded, &evicted->w_node);
			if (profile != NULL)
				profile->discardedPushes++;
		}
	}
}

/*
 * Init visited
 */
static inline void
InitVisited(char *base, visited_hash * v, bool inMemory, int ef, int m)
{
	if (!inMemory)
		v->tids = tidhash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else if (base != NULL)
		v->offsets = offsethash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else
		v->pointers = pointerhash_create(CurrentMemoryContext, ef * m * 2, NULL);
}

/*
 * Add to visited
 */
static inline void
AddToVisited(char *base, visited_hash * v, HnswElementPtr elementPtr, bool inMemory, bool *found)
{
	if (!inMemory)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);
		ItemPointerData indextid;

		ItemPointerSet(&indextid, element->blkno, element->offno);
		tidhash_insert(v->tids, indextid, found);
	}
	else if (base != NULL)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		offsethash_insert_hash(v->offsets, HnswPtrOffset(elementPtr), element->hash, found);
	}
	else
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		pointerhash_insert_hash(v->pointers, (uintptr_t) HnswPtrPointer(elementPtr), element->hash, found);
	}
}

/*
 * Count element towards ef
 */
static inline bool
CountElement(HnswElement skipElement, HnswElement e)
{
	if (skipElement == NULL)
		return true;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	/* Keep scan-build happy on Mac x86-64 */
	Assert(e);

	return e->heaptidsLength != 0;
}

/*
 * Load unvisited neighbors from memory
 */
static void
HnswLoadUnvisitedFromMemory(char *base, HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, int lc, HnswNeighborArray * localNeighborhood, Size neighborhoodSize)
{
	/* Get the neighborhood at layer lc */
	HnswNeighborArray *neighborhood = HnswGetNeighbors(base, element, lc);

	/* Copy neighborhood to local memory */
	LWLockAcquire(&element->lock, LW_SHARED);
	memcpy(localNeighborhood, neighborhood, neighborhoodSize);
	LWLockRelease(&element->lock);

	*unvisitedLength = 0;

	for (int i = 0; i < localNeighborhood->length; i++)
	{
		HnswCandidate *hc = &localNeighborhood->items[i];
		bool		found;

		AddToVisited(base, v, hc->element, true, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].element = HnswPtrAccess(base, hc->element);
	}
}

/*
 * Load neighbor index TIDs
 */
static bool
HnswLoadNeighborTidsTracked(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc, HnswIndexPageProfileState *profile)
{
	Buffer		buf;
	Page		page;
	HnswNeighborTuple ntup;
	int			start;

	buf = ReadBuffer(index, element->neighborPage);
	HnswRecordIndexPage(profile, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, element->neighborOffno));

	/*
	 * Ensure the neighbor tuple has not been deleted or replaced between
	 * index scan iterations
	 */
	if (ntup->version != element->version || ntup->count != (element->level + 2) * m)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	/* Copy to minimize lock time */
	start = (element->level - lc) * m;
	memcpy(indextids, ntup->indextids + start, lm * sizeof(ItemPointerData));

	UnlockReleaseBuffer(buf);
	return true;
}

bool
HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc)
{
	return HnswLoadNeighborTidsTracked(element, indextids, index, m, lm, lc, NULL);
}

/*
 * Load unvisited neighbors from disk
 */
static bool
HnswLoadUnvisitedFromDisk(HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, Relation index, int m, int lm, int lc, HnswIndexPageProfileState *profile)
{
	ItemPointerData indextids[HNSW_MAX_M * 2];

	*unvisitedLength = 0;

	if (!HnswLoadNeighborTidsTracked(element, indextids, index, m, lm, lc, profile))
		return false;

	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &indextids[i];
		bool		found;

		if (!ItemPointerIsValid(indextid))
			break;

		tidhash_insert(v->tids, *indextid, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].indextid = *indextid;
	}

	return true;
}

typedef struct HnswDualFrontier
{
	pairingheap *match;
	pairingheap *noBridge;
	int			noBridgePending;
	int			noBridgeDebt;
	int			burst;
	HnswTraversalProfile *profile;
} HnswDualFrontier;

static void
HnswDualFrontierInit(HnswDualFrontier *frontier, void *heapCompareArg,
						 int burst, HnswTraversalProfile *profile)
{
	Assert(burst >= 1);
	frontier->match = pairingheap_allocate(CompareNearestCandidates,
										  heapCompareArg);
	frontier->noBridge = pairingheap_allocate(CompareNearestCandidates,
											 heapCompareArg);
	frontier->noBridgePending = 0;
	frontier->noBridgeDebt = 0;
	frontier->burst = burst;
	frontier->profile = profile;
}

static bool
HnswDualFrontierIsEmpty(HnswDualFrontier *frontier)
{
	return pairingheap_is_empty(frontier->match) &&
		pairingheap_is_empty(frontier->noBridge);
}

static void
HnswDualFrontierAdd(HnswDualFrontier *frontier,
					HnswSearchCandidate *candidate)
{
	if (candidate->matchesGuidance)
		pairingheap_add(frontier->match, &candidate->c_node);
	else
	{
		pairingheap_add(frontier->noBridge, &candidate->c_node);
		frontier->noBridgePending++;
	}
}

static double
HnswDualFrontierMinDistance(HnswDualFrontier *frontier)
{
	HnswSearchCandidate *match = pairingheap_is_empty(frontier->match) ?
		NULL : HnswGetSearchCandidate(c_node, pairingheap_first(frontier->match));
	HnswSearchCandidate *noBridge = pairingheap_is_empty(frontier->noBridge) ?
		NULL : HnswGetSearchCandidate(c_node, pairingheap_first(frontier->noBridge));

	Assert(match != NULL || noBridge != NULL);
	if (match == NULL)
		return noBridge->distance;
	if (noBridge == NULL)
		return match->distance;
	return Min(match->distance, noBridge->distance);
}

static HnswSearchCandidate *
HnswDualFrontierPop(HnswDualFrontier *frontier)
{
	bool		hasMatch = !pairingheap_is_empty(frontier->match);
	bool		hasNoBridge = !pairingheap_is_empty(frontier->noBridge);

	Assert(hasMatch || hasNoBridge);
	if (hasMatch && (!hasNoBridge || frontier->noBridgeDebt < frontier->burst))
	{
		if (frontier->profile != NULL)
			frontier->profile->matchFrontierPops++;
		if (hasNoBridge)
		{
			frontier->noBridgeDebt++;
			if (frontier->profile != NULL)
			{
				frontier->profile->noBridgeDeferred++;
				frontier->profile->maxNoBridgeDebt = Max(
					frontier->profile->maxNoBridgeDebt,
					frontier->noBridgeDebt);
			}
		}
		else
			frontier->noBridgeDebt = 0;
		return HnswGetSearchCandidate(c_node,
			pairingheap_remove_first(frontier->match));
	}

	Assert(hasNoBridge);
	Assert(frontier->noBridgePending > 0);
	frontier->noBridgePending--;
	frontier->noBridgeDebt = 0;
	if (frontier->profile != NULL)
		frontier->profile->noBridgeFrontierPops++;
	return HnswGetSearchCandidate(c_node,
		pairingheap_remove_first(frontier->noBridge));
}

/*
 * Algorithm 2 from paper
 */
List *
HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples, int64 *distanceComputations, HnswTraversalProfile *traversalProfile, HnswScanGuidance *guidance, HnswTraversalGuidanceState *traversalGuidance, HnswIndexPageProfileState *indexPageProfile)
{
	HnswSearchHeapCompareContext heapCompareContext = {base};
	void	   *heapCompareArg = inserting && index == NULL &&
		hnsw_build_seed >= 0 ? &heapCompareContext : NULL;
	List	   *w = NIL;
	pairingheap *C = NULL;
	HnswDualFrontier dualFrontier = {0};
	pairingheap *W = NULL;
	pairingheap *G = NULL;
	List	   *guidedCandidates = NIL;
	List	   *wCandidates = NIL;
	int			wlen = 0;
	int			glen = 0;
	visited_hash vh;
	ListCell   *lc2;
	HnswNeighborArray *localNeighborhood = NULL;
	Size		neighborhoodSize = 0;
	int			lm = HnswGetLayerM(m, lc);
	HnswUnvisited *unvisited = NULL;
	int			unvisitedLength;
	bool		inMemory = index == NULL;
	bool		useAcorn1 = !inserting && lc == 0 && hnsw_filter_strategy == HNSW_FILTER_STRATEGY_ACORN1 && HnswGuidanceIsActiveForScan(guidance);
	bool		useGuidedCollect = !inserting && lc == 0 && hnsw_filter_strategy == HNSW_FILTER_STRATEGY_GUIDED_COLLECT && HnswGuidanceIsActiveForScan(guidance);
	bool		useTraversalPrioritization = !inserting && lc == 0 &&
		hnsw_filter_strategy == HNSW_FILTER_STRATEGY_TRAVERSAL_GUIDED &&
		traversalGuidance != NULL &&
		traversalGuidance->prioritizationEnabled &&
		traversalGuidance->finalPath ==
			HNSW_TRAVERSAL_PATH_APPROXIMATE_PRIORITIZATION &&
		HnswGuidanceIsActiveForScan(guidance);
	bool		useTraversalAdmission = !inserting && lc == 0 &&
		hnsw_filter_strategy == HNSW_FILTER_STRATEGY_TRAVERSAL_GUIDED &&
		traversalGuidance != NULL &&
		(traversalGuidance->finalPath ==
			HNSW_TRAVERSAL_PATH_CANDIDATE_ADMISSION ||
		 useTraversalPrioritization) &&
		HnswGuidanceIsActiveForScan(guidance);
	bool		useAdmissionGuidance = useAcorn1 || useTraversalAdmission;
	bool		trackTraversal = !inserting && lc == 0 && traversalProfile != NULL;
	bool		terminationRecorded = false;
	int			guidedTarget = hnsw_guided_collect_target;

	if (useTraversalPrioritization)
		HnswDualFrontierInit(&dualFrontier, heapCompareArg,
			traversalGuidance->burst, traversalProfile);
	else
		C = pairingheap_allocate(CompareNearestCandidates, heapCompareArg);
	W = pairingheap_allocate(CompareFurthestCandidates, heapCompareArg);
	unvisited = palloc(lm * sizeof(HnswUnvisited));

	if (useGuidedCollect)
	{
		if (guidedTarget > ef)
			guidedTarget = ef;
		G = pairingheap_allocate(CompareFurthestGuidedCandidates, NULL);
	}

	if (v == NULL)
	{
		v = &vh;
		initVisited = true;
	}

	if (initVisited)
	{
		InitVisited(base, v, inMemory, ef, m);

		if (discarded != NULL)
			*discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	}

	/* Create local memory for neighborhood if needed */
	if (inMemory)
	{
		neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		localNeighborhood = palloc(neighborhoodSize);
	}

	/* Add entry points to v, C, and W */
	foreach(lc2, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		bool		found;
		bool		matchesGuidance = true;

		if (initVisited)
		{
			AddToVisited(base, v, sc->element, inMemory, &found);

			/* OK to count elements instead of tuples */
			if (tuples != NULL)
				(*tuples)++;
		}

		if (useAdmissionGuidance || useGuidedCollect)
			matchesGuidance = HnswElementMatchesGuidance(HnswPtrAccess(base, sc->element), guidance, traversalProfile, false);
		sc->matchesGuidance = matchesGuidance;
		if (useTraversalPrioritization)
			HnswDualFrontierAdd(&dualFrontier, sc);
		else
			pairingheap_add(C, &sc->c_node);
		if (trackTraversal)
			traversalProfile->candidateAdmissions++;
		if (!useAdmissionGuidance || matchesGuidance)
		{
			pairingheap_add(W, &sc->w_node);
			if (trackTraversal)
			{
				traversalProfile->resultAdmissions++;
				if (useTraversalAdmission)
					traversalProfile->guidedAdmissions++;
			}
			if (useGuidedCollect)
				wCandidates = lappend(wCandidates, sc);
		}
		else if (trackTraversal && useTraversalAdmission)
		{
			traversalProfile->guidedSuppressions++;
			traversalProfile->heapTidsSuppressed +=
				HnswPtrAccess(base, sc->element)->heaptidsLength;
		}
		if (useGuidedCollect)
			HnswAddGuidedCandidate(G, &guidedCandidates, wCandidates, discarded != NULL ? *discarded : NULL,
							   &glen, ef, sc, HnswPtrAccess(base, sc->element), skipElement, traversalProfile);

		/*
		 * Do not count elements being deleted towards ef when vacuuming. It
		 * would be ideal to do this for inserts as well, but this could
		 * affect insert performance.
		 */
		if ((!useAdmissionGuidance || matchesGuidance) && CountElement(skipElement, HnswPtrAccess(base, sc->element)))
			wlen++;
	}

	while (useTraversalPrioritization ?
		   !HnswDualFrontierIsEmpty(&dualFrontier) :
		   !pairingheap_is_empty(C))
	{
		HnswSearchCandidate *c;
		HnswSearchCandidate *f = pairingheap_is_empty(W) ?
			NULL : HnswGetSearchCandidate(w_node, pairingheap_first(W));
		HnswElement cElement;

		CHECK_FOR_INTERRUPTS();

		if (useTraversalPrioritization)
		{
			if (f != NULL && wlen >= ef)
			{
				double		frontierMin =
					HnswDualFrontierMinDistance(&dualFrontier);

				if (trackTraversal)
				{
					traversalProfile->dualFrontierTerminationChecks++;
					if (!pairingheap_is_empty(dualFrontier.match) &&
						!pairingheap_is_empty(dualFrontier.noBridge))
						traversalProfile->dualFrontierTerminationChecksWithBoth++;
				}
				if (frontierMin > f->distance)
				{
					if (trackTraversal)
					{
						traversalProfile->stockTerminations++;
						traversalProfile->dualFrontierTerminations++;
						if (!pairingheap_is_empty(dualFrontier.match) &&
							!pairingheap_is_empty(dualFrontier.noBridge))
							traversalProfile->dualFrontierTerminationsWithBoth++;
					}
					terminationRecorded = true;
					break;
				}
			}
			c = HnswDualFrontierPop(&dualFrontier);
		}
		else
		{
			c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
			if (f != NULL && wlen >= ef && c->distance > f->distance)
			{
				if (!useGuidedCollect || glen >= guidedTarget)
				{
					if (trackTraversal)
						traversalProfile->stockTerminations++;
					terminationRecorded = true;
					break;
				}

				if (tuples != NULL && hnsw_max_scan_tuples > 0 && *tuples >= hnsw_max_scan_tuples)
				{
					if (trackTraversal)
						traversalProfile->maxScanTerminations++;
					terminationRecorded = true;
					break;
				}
				if (trackTraversal)
					traversalProfile->stopDeferrals++;
			}
		}

		cElement = HnswPtrAccess(base, c->element);
		if (trackTraversal)
		{
			traversalProfile->expandedNodes++;
			if (useAdmissionGuidance || useGuidedCollect)
			{
				if (c->matchesGuidance)
					traversalProfile->matchingExpanded++;
				else
				{
					traversalProfile->bridgeExpanded++;
					if (useTraversalPrioritization)
						traversalProfile->noBridgeExpansions++;
				}
			}
		}

		if (inMemory)
			HnswLoadUnvisitedFromMemory(base, cElement, unvisited, &unvisitedLength, v, lc, localNeighborhood, neighborhoodSize);
		else
		{
			if (!inserting)
					HnswRecordIndexNeighborPage(indexPageProfile,
						cElement->neighborPage);
			if (!HnswLoadUnvisitedFromDisk(cElement, unvisited,
					&unvisitedLength, v, index, m, lm, lc,
					indexPageProfile) && useTraversalPrioritization)
				traversalGuidance->invalidNeighbor = true;
			if (!inserting)
					HnswPrefetchUnvisitedIndexPages(index, unvisited,
						unvisitedLength, indexPageProfile);
		}

		/* OK to count elements instead of tuples */
		if (tuples != NULL)
			(*tuples) += unvisitedLength;
		if (trackTraversal)
			traversalProfile->neighborsExamined += unvisitedLength;
		if (trackTraversal && useTraversalAdmission && !c->matchesGuidance)
		{
			traversalProfile->missBridgeNodes++;
			traversalProfile->missBridgeEdges += unvisitedLength;
		}

		for (int i = 0; i < unvisitedLength; i++)
		{
			HnswElement eElement;
			HnswSearchCandidate *e;
			double		eDistance;
			bool		alwaysAdd = wlen < ef;
			bool		matchesGuidance = true;

			CHECK_FOR_INTERRUPTS();

			f = pairingheap_is_empty(W) ? NULL : HnswGetSearchCandidate(w_node, pairingheap_first(W));

			if (inMemory)
			{
				eElement = unvisited[i].element;
				eDistance = GetElementDistance(base, eElement, q, support);
				if (distanceComputations != NULL)
					(*distanceComputations)++;
			}
			else
			{
				ItemPointer indextid = &unvisited[i].indextid;
				BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
				OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);

				/* Avoid any allocations if not adding */
				eElement = NULL;
				HnswLoadElementImpl(blkno, offno, &eDistance, q, index, support, inserting, (alwaysAdd || discarded != NULL || f == NULL) ? NULL : &f->distance, &eElement, indexPageProfile);
				if (distanceComputations != NULL)
					(*distanceComputations)++;

				if (eElement == NULL)
					continue;
			}

			if (useAdmissionGuidance || useGuidedCollect)
				matchesGuidance = HnswElementMatchesGuidance(eElement, guidance, traversalProfile, false);

			if (useAdmissionGuidance && !matchesGuidance)
			{
				if (alwaysAdd || f == NULL || eDistance < f->distance)
				{
					e = HnswInitSearchCandidate(base, eElement, eDistance);
					e->matchesGuidance = false;
					if (useTraversalPrioritization)
						HnswDualFrontierAdd(&dualFrontier, e);
					else
						pairingheap_add(C, &e->c_node);
					if (trackTraversal)
					{
						traversalProfile->candidateAdmissions++;
						if (useTraversalAdmission)
						{
							traversalProfile->guidedSuppressions++;
							traversalProfile->heapTidsSuppressed +=
								eElement->heaptidsLength;
						}
					}
				}

				continue;
			}

			if (!((f == NULL || eDistance < f->distance) || alwaysAdd))
			{
				if (discarded != NULL || (useGuidedCollect && matchesGuidance))
				{
					/* Create a new candidate */
					e = HnswInitSearchCandidate(base, eElement, eDistance);
					e->matchesGuidance = matchesGuidance;
					if (useGuidedCollect)
						HnswAddGuidedCandidate(G, &guidedCandidates, wCandidates, discarded != NULL ? *discarded : NULL,
										   &glen, ef, e, eElement, skipElement, traversalProfile);
					if (discarded != NULL && !(useGuidedCollect && matchesGuidance))
					{
						pairingheap_add(*discarded, &e->w_node);
						if (trackTraversal)
							traversalProfile->discardedPushes++;
					}
				}

				continue;
			}

			/* Make robust to issues */
			if (eElement->level < lc)
				continue;

			/* Create a new candidate */
			e = HnswInitSearchCandidate(base, eElement, eDistance);
			e->matchesGuidance = matchesGuidance;
			if (useTraversalPrioritization)
				HnswDualFrontierAdd(&dualFrontier, e);
			else
				pairingheap_add(C, &e->c_node);
			pairingheap_add(W, &e->w_node);
			if (useGuidedCollect)
				wCandidates = lappend(wCandidates, e);
			if (trackTraversal)
			{
				traversalProfile->candidateAdmissions++;
				traversalProfile->resultAdmissions++;
				if (useTraversalAdmission)
					traversalProfile->guidedAdmissions++;
			}
			if (useGuidedCollect)
				HnswAddGuidedCandidate(G, &guidedCandidates, wCandidates, discarded != NULL ? *discarded : NULL,
									   &glen, ef, e, eElement, skipElement, traversalProfile);

			/*
			 * Do not count elements being deleted towards ef when vacuuming.
			 * It would be ideal to do this for inserts as well, but this
			 * could affect insert performance.
			 */
			if (CountElement(skipElement, eElement))
			{
				wlen++;

				/* No need to decrement wlen */
				if (wlen > ef)
				{
					HnswSearchCandidate *d = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

					if (useGuidedCollect)
						wCandidates = list_delete_ptr(wCandidates, d);

					if (discarded != NULL &&
						!(useGuidedCollect && d->matchesGuidance && list_member_ptr(guidedCandidates, d)))
					{
						pairingheap_add(*discarded, &d->w_node);
						if (trackTraversal)
							traversalProfile->discardedPushes++;
					}
				}
			}
		}
	}
	if (trackTraversal && !terminationRecorded)
		traversalProfile->exhaustedTerminations++;
	if (useTraversalPrioritization)
	{
		traversalGuidance->guidedResultCount = Min(wlen, ef);
		traversalGuidance->bridgePendingAtTermination =
			dualFrontier.noBridgePending;
		if (trackTraversal)
			traversalProfile->bridgePendingAtTermination +=
				dualFrontier.noBridgePending;
	}

	if (useGuidedCollect && G != NULL && !pairingheap_is_empty(G))
	{
		/*
		 * W owns the bridge frontier.  Remove every W node before either
		 * returning its G candidate or handing it to the resume heap.  A
		 * matching candidate still in G is returned, while every other W
		 * candidate remains available for iterative expansion.
		 */
		while (!pairingheap_is_empty(W))
		{
			HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

			wCandidates = list_delete_ptr(wCandidates, sc);
			if (discarded != NULL &&
				!(sc->matchesGuidance && list_member_ptr(guidedCandidates, sc)))
			{
				pairingheap_add(*discarded, &sc->w_node);
				if (trackTraversal)
					traversalProfile->discardedPushes++;
			}
		}

		while (!pairingheap_is_empty(G))
		{
			HnswSearchCandidate *sc = HnswGetSearchCandidate(g_node, pairingheap_remove_first(G));

			w = lappend(w, sc);
		}

		return w;
	}

	/* Add each element of W to w */
	while (!pairingheap_is_empty(W))
	{
		HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

		w = lappend(w, sc);
	}

	return w;
}

static int
CompareElementHeapTids(HnswElement a, HnswElement b)
{
	BlockNumber aBlock = ItemPointerGetBlockNumber(&a->heaptids[0]);
	BlockNumber bBlock = ItemPointerGetBlockNumber(&b->heaptids[0]);
	OffsetNumber aOffset = ItemPointerGetOffsetNumber(&a->heaptids[0]);
	OffsetNumber bOffset = ItemPointerGetOffsetNumber(&b->heaptids[0]);

	if (aBlock < bBlock)
		return -1;
	if (aBlock > bBlock)
		return 1;
	if (aOffset < bOffset)
		return -1;
	if (aOffset > bOffset)
		return 1;
	return 0;
}

static uint64
HashElementHeapTid(HnswElement element)
{
	BlockNumber block = ItemPointerGetBlockNumber(&element->heaptids[0]);
	OffsetNumber offset = ItemPointerGetOffsetNumber(&element->heaptids[0]);
	unsigned char bytes[6];

	/* Serialize explicitly so the tie order is independent of struct padding. */
	bytes[0] = (unsigned char) (block & 0xff);
	bytes[1] = (unsigned char) ((block >> 8) & 0xff);
	bytes[2] = (unsigned char) ((block >> 16) & 0xff);
	bytes[3] = (unsigned char) ((block >> 24) & 0xff);
	bytes[4] = (unsigned char) (offset & 0xff);
	bytes[5] = (unsigned char) ((offset >> 8) & 0xff);

	return hash_any_extended(bytes, sizeof(bytes), (uint64) (uint32) hnsw_build_seed);
}

static int
CompareSeededElementKeys(HnswElement a, HnswElement b)
{
	uint64		aHash = HashElementHeapTid(a);
	uint64		bHash = HashElementHeapTid(b);

	if (aHash < bHash)
		return -1;
	if (aHash > bHash)
		return 1;
	return CompareElementHeapTids(a, b);
}

/*
 * Compare candidate distances with a stable tie-breaker for reproducible
 * serial builds. Pointer addresses are process-local and can otherwise make
 * equal-distance candidates produce different logical graphs.
 */
static int
CompareCandidateDistances(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);
	HnswElement aElement;
	HnswElement bElement;
	int			keyCompare;

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (hnsw_build_seed >= 0)
	{
		aElement = HnswPtrPointer(hca->element);
		bElement = HnswPtrPointer(hcb->element);
		keyCompare = CompareSeededElementKeys(aElement, bElement);
		return -keyCompare;
	}

	if (HnswPtrPointer(hca->element) < HnswPtrPointer(hcb->element))
		return 1;

	if (HnswPtrPointer(hca->element) > HnswPtrPointer(hcb->element))
		return -1;

	return 0;
}

/*
 * Compare candidate distances with offset tie-breaker
 */
static int
CompareCandidateDistancesOffset(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrOffset(hca->element) < HnswPtrOffset(hcb->element))
		return 1;

	if (HnswPtrOffset(hca->element) > HnswPtrOffset(hcb->element))
		return -1;

	return 0;
}

/*
 * Check if an element is closer to q than any element from R
 */
static bool
CheckElementCloser(char *base, HnswCandidate * e, List *r, HnswSupport * support)
{
	HnswElement eElement = HnswPtrAccess(base, e->element);
	Datum		eValue = HnswGetValue(base, eElement);
	ListCell   *lc2;

	foreach(lc2, r)
	{
		HnswCandidate *ri = lfirst(lc2);
		HnswElement riElement = HnswPtrAccess(base, ri->element);
		Datum		riValue = HnswGetValue(base, riElement);
		float		distance = HnswGetDistance(eValue, riValue, support);

		if (distance <= e->distance)
			return false;
	}

	return true;
}

/*
 * Algorithm 4 from paper
 */
static List *
SelectNeighbors(char *base, List *c, int lm, HnswSupport * support, bool *closerSet, HnswCandidate * newCandidate, HnswCandidate * *pruned, bool sortCandidates)
{
	List	   *r = NIL;
	List	   *w = list_copy(c);
	HnswCandidate **wd;
	int			wdlen = 0;
	int			wdoff = 0;
	bool		mustCalculate = !(*closerSet);
	List	   *added = NIL;
	bool		removedAny = false;

	/* Seeded builds also need stable ordering when every candidate fits. */
	if (sortCandidates)
	{
		if (base == NULL)
			list_sort(w, CompareCandidateDistances);
		else
			list_sort(w, CompareCandidateDistancesOffset);
	}

	if (list_length(w) <= lm)
		return w;

	wd = palloc(sizeof(HnswCandidate *) * list_length(w));

	while (list_length(w) > 0 && list_length(r) < lm)
	{
		/* Assumes w is already ordered desc */
		HnswCandidate *e = llast(w);

		w = list_delete_last(w);

		/* Use previous state of r and wd to skip work when possible */
		if (mustCalculate)
			e->closer = CheckElementCloser(base, e, r, support);
		else if (list_length(added) > 0)
		{
			/* Keep Valgrind happy for in-memory, parallel builds */
			if (base != NULL)
				VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

			/*
			 * If the current candidate was closer, we only need to compare it
			 * with the other candidates that we have added.
			 */
			if (e->closer)
			{
				e->closer = CheckElementCloser(base, e, added, support);

				if (!e->closer)
					removedAny = true;
			}
			else
			{
				/*
				 * If we have removed any candidates from closer, a candidate
				 * that was not closer earlier might now be.
				 */
				if (removedAny)
				{
					e->closer = CheckElementCloser(base, e, r, support);
					if (e->closer)
						added = lappend(added, e);
				}
			}
		}
		else if (e == newCandidate)
		{
			e->closer = CheckElementCloser(base, e, r, support);
			if (e->closer)
				added = lappend(added, e);
		}

		/* Keep Valgrind happy for in-memory, parallel builds */
		if (base != NULL)
			VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

		if (e->closer)
			r = lappend(r, e);
		else
			wd[wdlen++] = e;
	}

	/* Cached value can only be used in future if sorted deterministically */
	*closerSet = sortCandidates;

	/* Keep pruned connections */
	while (wdoff < wdlen && list_length(r) < lm)
		r = lappend(r, wd[wdoff++]);

	/* Return pruned for update connections */
	if (pruned != NULL)
	{
		if (wdoff < wdlen)
			*pruned = wd[wdoff];
		else
			*pruned = linitial(w);
	}

	return r;
}

/*
 * Add connections
 */
static void
AddConnections(char *base, HnswElement element, List *neighbors, int lc)
{
	ListCell   *lc2;
	HnswNeighborArray *a = HnswGetNeighbors(base, element, lc);

	foreach(lc2, neighbors)
		a->items[a->length++] = *((HnswCandidate *) lfirst(lc2));
}

/*
 * Update connections
 */
void
HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support)
{
	HnswCandidate newHc;

	HnswPtrStore(base, newHc.element, newElement);
	newHc.distance = distance;

	if (neighbors->length < lm)
	{
		neighbors->items[neighbors->length++] = newHc;

		/* Track update */
		if (updateIdx != NULL)
			*updateIdx = -2;
	}
	else
	{
		/* Shrink connections */
		List	   *c = NIL;
		HnswCandidate *pruned = NULL;

		/* Add candidates */
		for (int i = 0; i < neighbors->length; i++)
			c = lappend(c, &neighbors->items[i]);
		c = lappend(c, &newHc);

		SelectNeighbors(base, c, lm, support, &neighbors->closerSet, &newHc, &pruned, true);

		/* Should not happen */
		if (pruned == NULL)
			return;

		/* Find and replace the pruned element */
		for (int i = 0; i < neighbors->length; i++)
		{
			if (HnswPtrEqual(base, neighbors->items[i].element, pruned->element))
			{
				neighbors->items[i] = newHc;

				/* Track update */
				if (updateIdx != NULL)
					*updateIdx = i;

				break;
			}
		}
	}
}

/*
 * Remove elements being deleted or skipped
 */
static List *
RemoveElements(char *base, List *w, HnswElement skipElement)
{
	ListCell   *lc2;
	List	   *w2 = NIL;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	foreach(lc2, w)
	{
		HnswCandidate *hc = (HnswCandidate *) lfirst(lc2);
		HnswElement hce = HnswPtrAccess(base, hc->element);

		/* Skip self for vacuuming update */
		if (skipElement != NULL && hce->blkno == skipElement->blkno && hce->offno == skipElement->offno)
			continue;

		if (hce->heaptidsLength != 0)
			w2 = lappend(w2, hc);
	}

	return w2;
}

/*
 * Precompute hash
 */
static void
PrecomputeHash(char *base, HnswElement element)
{
	HnswElementPtr ptr;

	HnswPtrStore(base, ptr, element);

	if (base == NULL)
		element->hash = hash_pointer((uintptr_t) HnswPtrPointer(ptr));
	else
		element->hash = hash_offset(HnswPtrOffset(ptr));
}

/*
 * Algorithm 1 from paper
 */
void
HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing)
{
	List	   *ep;
	List	   *w;
	int			level = element->level;
	int			entryLevel;
	HnswQuery	q;
	HnswElement skipElement = existing ? element : NULL;
	bool		inMemory = index == NULL;

	q.value = HnswGetValue(base, element);

	/* Precompute hash */
	if (inMemory)
		PrecomputeHash(base, element);

	/* No neighbors if no entry point */
	if (entryPoint == NULL)
		return;

	/* Get entry point and level */
	ep = list_make1(HnswEntryCandidate(base, entryPoint, &q, index, support, true));
	entryLevel = entryPoint->level;

	/* 1st phase: greedy search to insert level */
	for (int lc = entryLevel; lc >= level + 1; lc--)
	{
		w = HnswSearchLayer(base, &q, ep, 1, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL, NULL, NULL, NULL, NULL, NULL);
		ep = w;
	}

	if (level > entryLevel)
		level = entryLevel;

	/* Add one for existing element */
	if (existing)
		efConstruction++;

	/* 2nd phase */
	for (int lc = level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		List	   *neighbors;
		List	   *lw = NIL;
		ListCell   *lc2;

		w = HnswSearchLayer(base, &q, ep, efConstruction, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL, NULL, NULL, NULL, NULL, NULL);

		/* Convert search candidates to candidates */
		foreach(lc2, w)
		{
			HnswSearchCandidate *sc = lfirst(lc2);
			HnswCandidate *hc = palloc(sizeof(HnswCandidate));

			hc->element = sc->element;
			hc->distance = sc->distance;

			lw = lappend(lw, hc);
		}

		/* Elements being deleted or skipped can help with search */
		/* but should be removed before selecting neighbors */
		if (!inMemory)
			lw = RemoveElements(base, lw, skipElement);

		/* A seeded serial build uses heap TIDs to make every tie deterministic. */
		neighbors = SelectNeighbors(base, lw, lm, support, &HnswGetNeighbors(base, element, lc)->closerSet, NULL, NULL, hnsw_build_seed >= 0);

		AddConnections(base, element, neighbors, lc);

		ep = w;
	}
}

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum halfvec_l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum sparsevec_l2_normalize(PG_FUNCTION_ARGS);

static void
SparsevecCheckValue(Pointer v)
{
	SparseVector *vec = (SparseVector *) v;

	if (vec->nnz > HNSW_MAX_NNZ)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec cannot have more than %d non-zero elements for hnsw index", HNSW_MAX_NNZ)));
}

/*
 * Get type info
 */
const		HnswTypeInfo *
HnswGetTypeInfo(Relation index)
{
	FmgrInfo   *procinfo = HnswOptionalProcInfo(index, HNSW_TYPE_INFO_PROC);

	if (procinfo == NULL)
	{
		static const HnswTypeInfo typeInfo = {
			.maxDimensions = HNSW_MAX_DIM,
			.normalize = l2_normalize,
			.checkValue = NULL
		};

		return (&typeInfo);
	}
	else
		return (const HnswTypeInfo *) DatumGetPointer(FunctionCall0Coll(procinfo, InvalidOid));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_halfvec_support);
Datum
hnsw_halfvec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 2,
		.normalize = halfvec_l2_normalize,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_bit_support);
Datum
hnsw_bit_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 32,
		.normalize = NULL,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_sparsevec_support);
Datum
hnsw_sparsevec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = SPARSEVEC_MAX_DIM,
		.normalize = sparsevec_l2_normalize,
		.checkValue = SparsevecCheckValue
	};

	PG_RETURN_POINTER(&typeInfo);
}

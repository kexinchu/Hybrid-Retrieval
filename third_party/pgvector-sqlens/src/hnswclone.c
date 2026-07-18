#include "postgres.h"

#include "access/relation.h"
#include "catalog/index.h"
#include "catalog/namespace.h"
#include "catalog/pg_class_d.h"
#include "commands/defrem.h"
#include "commands/progress.h"
#include "commands/vacuum.h"
#include "common/cryptohash.h"
#include "common/sha2.h"
#include "fmgr.h"
#include "hnsw.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/nodes.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/builtins.h"
#include "utils/acl.h"
#include "utils/fmgrprotos.h"
#include "utils/hsearch.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/regproc.h"
#include "utils/rel.h"
#include "utils/relcache.h"

#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#else
#include "pgstat.h"
#endif

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#define HNSW_BFS_LOCALITY_SAMPLE_LIMIT 256

typedef struct HnswCloneTidKey
{
	BlockNumber blkno;
	OffsetNumber offno;
	uint16		padding;
} HnswCloneTidKey;

typedef struct HnswCloneElementEntry
{
	HnswCloneTidKey key;
	HnswElement element;
} HnswCloneElementEntry;

typedef struct HnswCloneNeighborEntry
{
	HnswCloneTidKey key;
	HnswElement owner;
	bool		tupleFound;
	bool		loaded;
} HnswCloneNeighborEntry;

typedef struct HnswDigest
{
	pg_cryptohash_ctx *ctx;
} HnswDigest;

typedef enum HnswDiskGraphPurpose
{
	HNSW_DISK_GRAPH_CLONE,
	HNSW_DISK_GRAPH_FINGERPRINT
} HnswDiskGraphPurpose;

typedef struct HnswBfsLocalitySample
{
	int64		rank;
	BlockNumber block;
	OffsetNumber offno;
} HnswBfsLocalitySample;

typedef struct HnswBfsLocality
{
	int64		graphNodes;
	int64		reachableNodes;
	int64		fallbackNodes;
	int64		sequenceNodes;
	int64		adjacentPairs;
	int64		sameBlockPairs;
	int64		nextBlockPairs;
	int64		sameOrNextPagePairs;
	int64		nondecreasingPairs;
	int64		backwardPairs;
	uint64		totalAbsBlockDelta;
	uint64		maxAbsBlockDelta;
	int64		pageRuns;
	int		sampleCount;
	HnswBfsLocalitySample samples[HNSW_BFS_LOCALITY_SAMPLE_LIMIT];
} HnswBfsLocality;

typedef struct HnswDiskGraph
{
	Relation	index;
	MemoryContext context;
	MemoryContext scratchContext;
	HnswGraph  *graph;
	HnswAllocator *allocator;
	HTAB	   *elements;
	HTAB	   *neighbors;
	HnswMetaPageData meta;
	BlockNumber blocks;
	int64		nodes;
	int64		heapTids;
	int64		tombstones;
	int		maxLevel;
	HnswDiskGraphPurpose purpose;
	bool		collectPhysicalDigest;
	bool		updateProgress;
	HnswDigest physical;
	uint8		physicalDigest[PG_SHA256_DIGEST_LENGTH];
} HnswDiskGraph;

typedef struct HnswGraphFingerprint
{
	Oid			heapOid;
	uint32		version;
	uint32		dimensions;
	uint16		m;
	uint16		efConstruction;
	int64		nodes;
	int64		heapTids;
	int		entryLevel;
	int		maxLevel;
	bool		hasEntry;
	uint8		entryIdentity[PG_SHA256_DIGEST_LENGTH];
	uint8		definitionDigest[PG_SHA256_DIGEST_LENGTH];
	uint8		tupleCoverageDigest[PG_SHA256_DIGEST_LENGTH];
	uint8		logicalDigest[PG_SHA256_DIGEST_LENGTH];
	uint8		physicalDigest[PG_SHA256_DIGEST_LENGTH];
	HnswBfsLocality bfsLocality;
	int64		tombstones;
} HnswGraphFingerprint;

static void
HnswDigestStart(HnswDigest *digest)
{
	digest->ctx = pg_cryptohash_create(PG_SHA256);
	if (digest->ctx == NULL || pg_cryptohash_init(digest->ctx) < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("could not initialize HNSW graph digest")));
}

static void
HnswDigestBytes(HnswDigest *digest, const void *data, Size len)
{
	if (pg_cryptohash_update(digest->ctx, (const uint8 *) data, len) < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("could not update HNSW graph digest"),
				 errdetail("%s", pg_cryptohash_error(digest->ctx))));
}

static void
HnswDigestUint8(HnswDigest *digest, uint8 value)
{
	HnswDigestBytes(digest, &value, sizeof(value));
}

static void
HnswDigestUint16(HnswDigest *digest, uint16 value)
{
	uint8		encoded[2];

	encoded[0] = (uint8) (value >> 8);
	encoded[1] = (uint8) value;
	HnswDigestBytes(digest, encoded, sizeof(encoded));
}

static void
HnswDigestUint32(HnswDigest *digest, uint32 value)
{
	uint8		encoded[4];

	encoded[0] = (uint8) (value >> 24);
	encoded[1] = (uint8) (value >> 16);
	encoded[2] = (uint8) (value >> 8);
	encoded[3] = (uint8) value;
	HnswDigestBytes(digest, encoded, sizeof(encoded));
}

static void
HnswDigestUint64(HnswDigest *digest, uint64 value)
{
	uint8		encoded[8];

	for (int i = 0; i < 8; i++)
		encoded[i] = (uint8) (value >> (56 - 8 * i));
	HnswDigestBytes(digest, encoded, sizeof(encoded));
}

static void
HnswDigestString(HnswDigest *digest, const char *value)
{
	Size		len = value == NULL ? 0 : strlen(value);

	if (len > PG_UINT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("HNSW index definition is too large to fingerprint")));
	HnswDigestUint32(digest, (uint32) len);
	if (len > 0)
		HnswDigestBytes(digest, value, len);
}

static void
HnswDigestFinal(HnswDigest *digest, uint8 output[PG_SHA256_DIGEST_LENGTH])
{
	if (pg_cryptohash_final(digest->ctx, output, PG_SHA256_DIGEST_LENGTH) < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("could not finalize HNSW graph digest"),
				 errdetail("%s", pg_cryptohash_error(digest->ctx))));
	pg_cryptohash_free(digest->ctx);
	digest->ctx = NULL;
}

static void
HnswDigestItemPointer(HnswDigest *digest, ItemPointer tid)
{
	HnswDigestUint32(digest, ItemPointerGetBlockNumber(tid));
	HnswDigestUint16(digest, ItemPointerGetOffsetNumber(tid));
}

static HnswCloneTidKey
HnswCloneTidKeyFromPointer(ItemPointer tid)
{
	HnswCloneTidKey key;

	key.blkno = ItemPointerGetBlockNumber(tid);
	key.offno = ItemPointerGetOffsetNumber(tid);
	key.padding = 0;
	return key;
}

static void
HnswCloneCorruption(Relation index, const char *detail)
{
	ereport(ERROR,
			(errcode(ERRCODE_INDEX_CORRUPTED),
			 errmsg("cannot clone corrupt HNSW index \"%s\"",
					RelationGetRelationName(index)),
			 errdetail_internal("%s", detail)));
}

static pg_attribute_noreturn() void
HnswDiskGraphMemoryError(HnswDiskGraph *disk, Size used)
{
	if (disk->purpose == HNSW_DISK_GRAPH_CLONE)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("source HNSW graph does not fit into maintenance_work_mem"),
				 errdetail("Exact graph cloning used %zu bytes after loading " INT64_FORMAT " graph nodes; the limit is %zu bytes.",
						   used, disk->nodes, disk->graph->memoryTotal),
				 errhint("Increase maintenance_work_mem and retry the clone build.")));

	ereport(ERROR,
			(errcode(ERRCODE_OUT_OF_MEMORY),
			 errmsg("HNSW graph fingerprint exceeds maintenance_work_mem"),
			 errdetail("Canonical graph proof used or reserved %zu bytes after loading " INT64_FORMAT " graph nodes; the limit is %zu bytes.",
					   used, disk->nodes, disk->graph->memoryTotal),
			 errhint("Increase maintenance_work_mem and retry the graph proof.")));
}

static void
HnswDiskGraphCheckMemory(HnswDiskGraph *disk)
{
	Size		used;

	if (disk->context == NULL || disk->graph->memoryTotal == 0)
		return;

	used = MemoryContextMemAllocated(disk->context, true);
	disk->graph->memoryUsed = used;
	if (used > disk->graph->memoryTotal)
		HnswDiskGraphMemoryError(disk, used);
}

static void
HnswDiskGraphReserveMemory(HnswDiskGraph *disk, Size bytes)
{
	Size		used;

	if (disk->context == NULL || disk->graph->memoryTotal == 0)
		return;
	used = MemoryContextMemAllocated(disk->context, true);
	if (bytes > disk->graph->memoryTotal ||
		used > disk->graph->memoryTotal - bytes)
		HnswDiskGraphMemoryError(disk,
							 bytes > SIZE_MAX - used ? SIZE_MAX : used + bytes);
}

static HTAB *
HnswCloneCreateHash(const char *name, Size entrySize, MemoryContext context)
{
	HASHCTL		ctl;

	MemSet(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(HnswCloneTidKey);
	ctl.entrysize = entrySize;
	ctl.hcxt = context;
	return hash_create(name, 1024, &ctl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

static void
HnswCloneValidatePage(HnswDiskGraph *disk, Page page, BlockNumber blkno)
{
	if (PageIsNew(page) ||
		PageGetSpecialSize(page) != MAXALIGN(sizeof(HnswPageOpaqueData)) ||
		HnswPageGetOpaque(page)->page_id != HNSW_PAGE_ID)
		HnswCloneCorruption(disk->index,
						psprintf("block %u is not a supported HNSW page", blkno));
}

static void
HnswCloneReadMeta(HnswDiskGraph *disk)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	disk->blocks = RelationGetNumberOfBlocks(disk->index);
	if (disk->blocks < 2)
		HnswCloneCorruption(disk->index, "the index has no graph page");

	buf = ReadBuffer(disk->index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	HnswCloneValidatePage(disk, page, HNSW_METAPAGE_BLKNO);
	metap = HnswPageGetMeta(page);

	if (metap->magicNumber != HNSW_MAGIC_NUMBER)
		HnswCloneCorruption(disk->index, "the metapage magic number is invalid");
	if (metap->version != HNSW_VERSION)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("cannot clone HNSW index \"%s\" with format version %u",
						RelationGetRelationName(disk->index), metap->version),
				 errdetail("SQLens supports exact cloning only for HNSW format version %u.",
						   HNSW_VERSION)));
	if (metap->m < HNSW_MIN_M || metap->m > HNSW_MAX_M ||
		metap->efConstruction < HNSW_MIN_EF_CONSTRUCTION ||
		metap->efConstruction > HNSW_MAX_EF_CONSTRUCTION ||
		metap->dimensions == 0)
		HnswCloneCorruption(disk->index, "the metapage options are invalid");

	disk->meta = *metap;
	UnlockReleaseBuffer(buf);

	if (disk->collectPhysicalDigest)
	{
		static const char domain[] = "SQLENS-HNSW-PHYSICAL-V1";

		HnswDigestStart(&disk->physical);
		HnswDigestBytes(&disk->physical, domain, sizeof(domain) - 1);
		HnswDigestUint32(&disk->physical, disk->meta.version);
		HnswDigestUint32(&disk->physical, disk->meta.dimensions);
		HnswDigestUint16(&disk->physical, disk->meta.m);
		HnswDigestUint16(&disk->physical, disk->meta.efConstruction);
		HnswDigestUint32(&disk->physical, disk->meta.entryBlkno);
		HnswDigestUint16(&disk->physical, disk->meta.entryOffno);
		HnswDigestUint16(&disk->physical, (uint16) disk->meta.entryLevel);
		HnswDigestUint32(&disk->physical, disk->meta.insertPage);
		HnswDigestUint32(&disk->physical, disk->blocks);
	}
}

static int
HnswCloneReadHeapTids(HnswDiskGraph *disk, HnswElementTuple etup,
						  HnswElement element)
{
	bool		seenInvalid = false;
	int		count = 0;

	for (int i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (!ItemPointerIsValid(&etup->heaptids[i]))
		{
			seenInvalid = true;
			continue;
		}
		if (seenInvalid)
			HnswCloneCorruption(disk->index,
							"an element has a hole in its ordered heap-TID bundle");
		HnswAddHeapTid(element, &etup->heaptids[i]);
		count++;
	}

	if (etup->deleted)
	{
		if (count != 0)
			HnswCloneCorruption(disk->index,
							"a tombstoned element retains live heap TIDs");
		disk->tombstones++;
		if (disk->purpose == HNSW_DISK_GRAPH_CLONE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("cannot clone HNSW index \"%s\" with tombstoned elements",
							RelationGetRelationName(disk->index)),
					 errdetail("Exact graph cloning requires every source element to have a live ordered heap-TID bundle and deleted = 0.")));
		return 0;
	}
	if (count == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("HNSW index \"%s\" has a live element without heap TIDs",
						RelationGetRelationName(disk->index))));
	return count;
}

static void
HnswCloneAppendElement(HnswDiskGraph *disk, HnswElement *tail,
					   HnswElementTuple etup, Size itemSize,
					   BlockNumber blkno, OffsetNumber offno)
{
	HnswElement element;
	HnswCloneElementEntry *elementEntry;
	HnswCloneNeighborEntry *neighborEntry;
	HnswCloneTidKey key;
	ItemPointerData sourceTid;
	Size		valueSize;
	Pointer		value;
	Pointer		valueCopy;
	bool		found;
	char	   *base = NULL;

	if (itemSize < offsetof(HnswElementTupleData, data) + VARHDRSZ)
		HnswCloneCorruption(disk->index, "an element tuple is truncated");
	if (etup->level > HnswGetMaxLevel(disk->meta.m))
		HnswCloneCorruption(disk->index, "an element level exceeds the format limit");

	if (!ItemPointerIsValid(&etup->neighbortid) ||
		ItemPointerGetBlockNumber(&etup->neighbortid) == HNSW_METAPAGE_BLKNO)
		HnswCloneCorruption(disk->index, "an element has an invalid neighbor-tuple pointer");

	element = (HnswElement) HnswAlloc(disk->allocator, sizeof(HnswElementData));
	MemSet(element, 0, sizeof(HnswElementData));
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->blkno = blkno;
	element->offno = offno;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;
	disk->heapTids += HnswCloneReadHeapTids(disk, etup, element);
	HnswInitNeighbors(NULL, element, disk->meta.m, disk->allocator);
	if (etup->deleted)
	{
		/* VACUUM zeroes a tombstone's varlena payload, including its header. */
		valueSize = VARHDRSZ;
		valueCopy = HnswAlloc(disk->allocator, valueSize);
		SET_VARSIZE(valueCopy, valueSize);
	}
	else
	{
		value = (Pointer) &etup->data;
		if (VARATT_IS_EXTENDED(value))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("cannot clone HNSW index \"%s\" with an extended on-disk vector",
							RelationGetRelationName(disk->index))));
		valueSize = VARSIZE_ANY(value);
		if (valueSize < VARHDRSZ || HNSW_ELEMENT_TUPLE_SIZE(valueSize) != itemSize)
			HnswCloneCorruption(disk->index, "an element tuple has an invalid vector size");
		valueCopy = HnswAlloc(disk->allocator, valueSize);
		memcpy(valueCopy, value, valueSize);
	}
	HnswPtrStore(base, element->value, (char *) valueCopy);
	HnswPtrStore(base, element->next, (HnswElement) NULL);

	if (*tail == NULL)
		HnswPtrStore(base, disk->graph->head, element);
	else
		HnswPtrStore(base, (*tail)->next, element);
	*tail = element;

	ItemPointerSet(&sourceTid, blkno, offno);
	key = HnswCloneTidKeyFromPointer(&sourceTid);
	elementEntry = hash_search(disk->elements, &key, HASH_ENTER, &found);
	if (found)
		HnswCloneCorruption(disk->index, "duplicate physical element pointer");
	elementEntry->element = element;

	key = HnswCloneTidKeyFromPointer(&etup->neighbortid);
	neighborEntry = hash_search(disk->neighbors, &key, HASH_ENTER, &found);
	if (!found)
	{
		neighborEntry->owner = NULL;
		neighborEntry->tupleFound = false;
		neighborEntry->loaded = false;
	}
	if (neighborEntry->owner != NULL)
		HnswCloneCorruption(disk->index,
						"multiple elements reference the same neighbor tuple");
	neighborEntry->owner = element;

	disk->nodes++;
	disk->maxLevel = Max(disk->maxLevel, (int) element->level);
}

static void
HnswCloneRecordNeighborTuple(HnswDiskGraph *disk, BlockNumber blkno,
							 OffsetNumber offno)
{
	HnswCloneNeighborEntry *entry;
	HnswCloneTidKey key;
	ItemPointerData tid;
	bool		found;

	ItemPointerSet(&tid, blkno, offno);
	key = HnswCloneTidKeyFromPointer(&tid);
	entry = hash_search(disk->neighbors, &key, HASH_ENTER, &found);
	if (!found)
	{
		entry->owner = NULL;
		entry->loaded = false;
		entry->tupleFound = false;
	}
	if (entry->tupleFound)
		HnswCloneCorruption(disk->index, "duplicate physical neighbor pointer");
	entry->tupleFound = true;
}

static void
HnswCloneLoadPassOne(HnswDiskGraph *disk)
{
	BlockNumber blkno = HNSW_HEAD_BLKNO;
	BlockNumber pages = 0;
	HnswElement tail = NULL;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		OffsetNumber maxoffno;
		BlockNumber nextblkno;

		CHECK_FOR_INTERRUPTS();
		if (blkno == HNSW_METAPAGE_BLKNO || blkno >= disk->blocks ||
			++pages > disk->blocks - 1)
			HnswCloneCorruption(disk->index, "the graph page chain is invalid");

		buf = ReadBuffer(disk->index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		HnswCloneValidatePage(disk, page, blkno);
		maxoffno = PageGetMaxOffsetNumber(page);
		nextblkno = HnswPageGetOpaque(page)->nextblkno;

		if (disk->collectPhysicalDigest)
		{
			HnswDigestUint32(&disk->physical, blkno);
			HnswDigestUint32(&disk->physical, nextblkno);
			HnswDigestUint16(&disk->physical, maxoffno);
			HnswDigestUint16(&disk->physical, ((PageHeader) page)->pd_lower);
			HnswDigestUint16(&disk->physical, ((PageHeader) page)->pd_upper);
			HnswDigestUint16(&disk->physical, ((PageHeader) page)->pd_special);
		}

		for (OffsetNumber offno = FirstOffsetNumber;
			 offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			ItemId		itemId = PageGetItemId(page, offno);
			Pointer		item;
			Size		itemSize;
			uint8		type;

			if (!ItemIdIsNormal(itemId))
				HnswCloneCorruption(disk->index,
								psprintf("block %u offset %u is not a normal tuple",
										 blkno, offno));
			item = PageGetItem(page, itemId);
			itemSize = ItemIdGetLength(itemId);
			if (itemSize < sizeof(uint8))
				HnswCloneCorruption(disk->index, "an index tuple is truncated");
			type = *((uint8 *) item);

			if (disk->collectPhysicalDigest)
			{
				HnswDigestUint16(&disk->physical, offno);
				HnswDigestUint16(&disk->physical, ItemIdGetOffset(itemId));
				HnswDigestUint8(&disk->physical, ItemIdGetFlags(itemId));
				HnswDigestUint32(&disk->physical, itemSize);
				HnswDigestBytes(&disk->physical, item, itemSize);
			}

			if (type == HNSW_ELEMENT_TUPLE_TYPE)
				HnswCloneAppendElement(disk, &tail,
								   (HnswElementTuple) item, itemSize,
								   blkno, offno);
			else if (type == HNSW_NEIGHBOR_TUPLE_TYPE)
			{
				if (itemSize < offsetof(HnswNeighborTupleData, indextids))
					HnswCloneCorruption(disk->index, "a neighbor tuple is truncated");
				HnswCloneRecordNeighborTuple(disk, blkno, offno);
			}
			else
				HnswCloneCorruption(disk->index, "an index tuple has an unsupported type");
		}

		UnlockReleaseBuffer(buf);
		if (BlockNumberIsValid(nextblkno) &&
			(nextblkno == HNSW_METAPAGE_BLKNO || nextblkno >= disk->blocks))
			HnswCloneCorruption(disk->index, "a graph page has an invalid next-page pointer");
		blkno = nextblkno;
			HnswDiskGraphCheckMemory(disk);
		if (disk->updateProgress)
			pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE,
									 disk->heapTids);
	}

	if (pages != disk->blocks - 1)
		HnswCloneCorruption(disk->index, "the graph page chain does not cover every index block");

	if (disk->collectPhysicalDigest)
		HnswDigestFinal(&disk->physical, disk->physicalDigest);
}

static void
HnswCloneValidateNeighborMap(HnswDiskGraph *disk)
{
	HASH_SEQ_STATUS scan;
	HnswCloneNeighborEntry *entry;
	long		count = 0;

	hash_seq_init(&scan, disk->neighbors);
	while ((entry = hash_seq_search(&scan)) != NULL)
	{
		if (entry->owner == NULL || !entry->tupleFound)
			HnswCloneCorruption(disk->index,
							"an element/neighbor tuple pair is missing or unresolved");
		count++;
	}
	if (count != disk->nodes)
		HnswCloneCorruption(disk->index,
						"the element and neighbor tuple counts differ");
}

static void
HnswCloneLoadNeighbor(HnswDiskGraph *disk, HnswCloneNeighborEntry *entry,
					  HnswNeighborTuple ntup, Size itemSize)
{
	HnswElement owner = entry->owner;
	int			expected = (owner->level + 2) * disk->meta.m;
	int			idx = 0;
	char	   *base = NULL;

	if (itemSize != HNSW_NEIGHBOR_TUPLE_SIZE(owner->level, disk->meta.m) ||
		ntup->count != expected)
		HnswCloneCorruption(disk->index,
						"a neighbor tuple has an invalid size or pointer count");
	if (ntup->version != owner->version)
		HnswCloneCorruption(disk->index,
						"an element and its neighbor tuple have different versions");

	for (int lc = owner->level; lc >= 0; lc--)
	{
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, owner, lc);
		int			lm = HnswGetLayerM(disk->meta.m, lc);
		bool		seenInvalid = false;

		for (int i = 0; i < lm; i++)
		{
			ItemPointer tid = &ntup->indextids[idx++];
			HnswCloneElementEntry *targetEntry;
			HnswCloneTidKey key;

			if (!ItemPointerIsValid(tid))
			{
				seenInvalid = true;
				continue;
			}
			if (seenInvalid)
				HnswCloneCorruption(disk->index,
								"a neighbor layer has a hole in its ordered pointers");

			key = HnswCloneTidKeyFromPointer(tid);
			targetEntry = hash_search(disk->elements, &key, HASH_FIND, NULL);
			if (targetEntry == NULL)
				HnswCloneCorruption(disk->index,
								"a neighbor pointer does not resolve to a source element");
			if (targetEntry->element == owner)
				HnswCloneCorruption(disk->index, "an element has a self edge");
			if (targetEntry->element->level < lc)
				HnswCloneCorruption(disk->index,
								"a neighbor pointer targets an element below its layer");

			HnswPtrStore(base, neighbors->items[neighbors->length].element,
						 targetEntry->element);
			neighbors->items[neighbors->length].distance = 0;
			neighbors->items[neighbors->length].closer = false;
			neighbors->length++;
		}
	}

	entry->loaded = true;
}

static void
HnswCloneLoadPassTwo(HnswDiskGraph *disk)
{
	BlockNumber blkno = HNSW_HEAD_BLKNO;
	BlockNumber pages = 0;
	int64		loaded = 0;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		OffsetNumber maxoffno;
		BlockNumber nextblkno;

		CHECK_FOR_INTERRUPTS();
		if (blkno == HNSW_METAPAGE_BLKNO || blkno >= disk->blocks ||
			++pages > disk->blocks - 1)
			HnswCloneCorruption(disk->index, "the graph page chain changed during loading");

		buf = ReadBuffer(disk->index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		HnswCloneValidatePage(disk, page, blkno);
		maxoffno = PageGetMaxOffsetNumber(page);
		nextblkno = HnswPageGetOpaque(page)->nextblkno;

		for (OffsetNumber offno = FirstOffsetNumber;
			 offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			ItemId		itemId = PageGetItemId(page, offno);
			Pointer		item;
			HnswCloneNeighborEntry *entry;
			HnswCloneTidKey key;
			ItemPointerData tid;

			if (!ItemIdIsNormal(itemId))
				HnswCloneCorruption(disk->index, "an index tuple changed during loading");
			item = PageGetItem(page, itemId);
			if (*((uint8 *) item) != HNSW_NEIGHBOR_TUPLE_TYPE)
				continue;

			ItemPointerSet(&tid, blkno, offno);
			key = HnswCloneTidKeyFromPointer(&tid);
			entry = hash_search(disk->neighbors, &key, HASH_FIND, NULL);
			if (entry == NULL || entry->loaded)
				HnswCloneCorruption(disk->index,
								"a neighbor tuple changed or was loaded more than once");
			HnswCloneLoadNeighbor(disk, entry, (HnswNeighborTuple) item,
							  ItemIdGetLength(itemId));
			loaded++;
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
	}

	if (pages != disk->blocks - 1 || loaded != disk->nodes)
		HnswCloneCorruption(disk->index,
						"the second graph pass did not resolve every neighbor tuple");
}

static void
HnswCloneResolveEntryPoint(HnswDiskGraph *disk)
{
	HnswCloneElementEntry *entry;
	HnswCloneTidKey key;
	ItemPointerData tid;
	char	   *base = NULL;

	if (disk->nodes == 0)
	{
		if (BlockNumberIsValid(disk->meta.entryBlkno) ||
			disk->meta.entryLevel != -1)
			HnswCloneCorruption(disk->index,
							"an empty graph has an entry point");
		HnswPtrStore(base, disk->graph->entryPoint, (HnswElement) NULL);
		return;
	}

	if (!BlockNumberIsValid(disk->meta.entryBlkno) ||
		!OffsetNumberIsValid(disk->meta.entryOffno))
		HnswCloneCorruption(disk->index, "a nonempty graph has no entry point");
	ItemPointerSet(&tid, disk->meta.entryBlkno, disk->meta.entryOffno);
	key = HnswCloneTidKeyFromPointer(&tid);
	entry = hash_search(disk->elements, &key, HASH_FIND, NULL);
	if (entry == NULL)
		HnswCloneCorruption(disk->index,
						"the entry point does not resolve to a source element");
	if (entry->element->level != disk->meta.entryLevel ||
		disk->meta.entryLevel != disk->maxLevel)
		HnswCloneCorruption(disk->index,
						"the entry point level is not the graph maximum");
	HnswPtrStore(base, disk->graph->entryPoint, entry->element);
}

static void
HnswLoadDiskGraph(HnswDiskGraph *disk)
{
	MemoryContext oldContext;
	MemoryContext hashContext;

	oldContext = MemoryContextSwitchTo(disk->context);
	disk->scratchContext = AllocSetContextCreate(disk->context,
											  "HNSW disk graph maps",
											  ALLOCSET_DEFAULT_SIZES);
	hashContext = MemoryContextSwitchTo(disk->scratchContext);
	disk->elements = HnswCloneCreateHash("HNSW clone source elements",
											 sizeof(HnswCloneElementEntry), disk->scratchContext);
	disk->neighbors = HnswCloneCreateHash("HNSW clone source neighbors",
										  sizeof(HnswCloneNeighborEntry), disk->scratchContext);
	MemoryContextSwitchTo(hashContext);
	disk->nodes = 0;
	disk->heapTids = 0;
	disk->tombstones = 0;
	disk->maxLevel = -1;

	HnswCloneReadMeta(disk);
	HnswCloneLoadPassOne(disk);
	HnswCloneValidateNeighborMap(disk);
	HnswCloneLoadPassTwo(disk);
	HnswCloneResolveEntryPoint(disk);
	MemoryContextDelete(disk->scratchContext);
	disk->scratchContext = NULL;
	disk->elements = NULL;
	disk->neighbors = NULL;
	HnswDiskGraphCheckMemory(disk);
	MemoryContextSwitchTo(oldContext);
}

typedef struct HnswBfsVisitedEntry
{
	HnswCloneTidKey key;
} HnswBfsVisitedEntry;

static bool
HnswBfsMarkVisited(HTAB *visited, HnswElement element)
{
	ItemPointerData tid;
	HnswCloneTidKey key;
	bool		found;

	ItemPointerSet(&tid, element->blkno, element->offno);
	key = HnswCloneTidKeyFromPointer(&tid);
	hash_search(visited, &key, HASH_ENTER, &found);
	return !found;
}

/*
 * Reproduce hnsw.build_page_order = bfs over the loaded graph.  The sequence
 * is zero-based, follows entry-point neighbors from high to low layer, and
 * appends the physical element-list order only for disconnected nodes.  The
 * full sequence is measured; only the rank-to-(block,offset) evidence is
 * bounded to keep proof JSON auditable at large graph sizes.  A same-page
 * pair has block delta 0, a next-page pair has forward block delta +1, and a
 * page run is a maximal run of equal physical index blocks.
 */
static void
HnswBuildBfsLocality(HnswDiskGraph *disk, HnswBfsLocality *result)
{
	MemoryContext scratchContext;
	MemoryContext oldContext;
	HASHCTL		ctl;
	HTAB		*visited;
	HnswElement *queue;
	HnswElementPtr iter;
	HnswElement entryPoint;
	int64		queueHead = 0;
	int64		queueCount = 0;
	int64		sampleTarget = 0;
	int		sampleIndex = 0;
	char	   *base = NULL;

	MemSet(result, 0, sizeof(*result));
	result->graphNodes = disk->nodes;
	if (disk->nodes == 0)
	{
		result->sequenceNodes = 0;
		return;
	}
	if ((uint64) disk->nodes > MaxAllocSize / sizeof(HnswElement))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("HNSW graph is too large for BFS locality proof")));

	HnswDiskGraphReserveMemory(disk,
							(Size) disk->nodes * sizeof(HnswElement));
	queue = palloc((Size) disk->nodes * sizeof(HnswElement));
	HnswDiskGraphCheckMemory(disk);

	oldContext = MemoryContextSwitchTo(disk->context);
	scratchContext = AllocSetContextCreate(disk->context,
										  "HNSW BFS locality proof",
										  ALLOCSET_DEFAULT_SIZES);
		MemoryContextSwitchTo(scratchContext);
	MemSet(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(HnswCloneTidKey);
	ctl.entrysize = sizeof(HnswBfsVisitedEntry);
	ctl.hcxt = scratchContext;
	visited = hash_create("HNSW BFS locality visited", 1024, &ctl,
						  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	entryPoint = HnswPtrAccess(base, disk->graph->entryPoint);
	if (entryPoint != NULL && HnswBfsMarkVisited(visited, entryPoint))
		queue[queueCount++] = entryPoint;
	while (queueHead < queueCount)
	{
		HnswElement element = queue[queueHead++];

		for (int lc = element->level; lc >= 0; lc--)
		{
			HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, lc);

			for (int i = 0; i < neighbors->length; i++)
			{
				HnswElement neighbor = HnswPtrAccess(base,
													 neighbors->items[i].element);

				if (neighbor != NULL && HnswBfsMarkVisited(visited, neighbor))
					queue[queueCount++] = neighbor;
			}
		}
	}
	result->reachableNodes = queueCount;

	/* Match the builder's deterministic disconnected-node completion rule. */
	iter = disk->graph->head;
	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);

		if (HnswBfsMarkVisited(visited, element))
			queue[queueCount++] = element;
		iter = element->next;
	}
	result->fallbackNodes = queueCount - result->reachableNodes;
	result->sequenceNodes = queueCount;
	if (queueCount != disk->nodes)
		HnswCloneCorruption(disk->index,
						"the BFS locality sequence does not cover every graph node");

	result->sampleCount = Min(queueCount,
							  (int64) HNSW_BFS_LOCALITY_SAMPLE_LIMIT);
	for (int64 rank = 0; rank < queueCount; rank++)
	{
		HnswElement element = queue[rank];
		BlockNumber previousBlock;
		uint64		blockDelta;

		if (result->sampleCount == 1)
			sampleTarget = 0;
		else if (result->sampleCount > 1)
			sampleTarget = ((int64) sampleIndex * (queueCount - 1)) /
				(result->sampleCount - 1);
		if (result->sampleCount > 0 && rank == sampleTarget)
		{
			result->samples[sampleIndex].rank = rank;
			result->samples[sampleIndex].block = element->blkno;
			result->samples[sampleIndex].offno = element->offno;
			sampleIndex++;
		}
		if (rank == 0)
		{
			result->pageRuns = 1;
			continue;
		}

		previousBlock = queue[rank - 1]->blkno;
		result->adjacentPairs++;
		if (element->blkno == previousBlock)
			result->sameBlockPairs++;
		if (element->blkno > previousBlock &&
			(element->blkno - previousBlock) == 1)
			result->nextBlockPairs++;
		if (element->blkno == previousBlock ||
			(element->blkno > previousBlock &&
			 (element->blkno - previousBlock) == 1))
			result->sameOrNextPagePairs++;
		if (element->blkno >= previousBlock)
			result->nondecreasingPairs++;
		else
			result->backwardPairs++;
		blockDelta = element->blkno >= previousBlock ?
			(uint64) (element->blkno - previousBlock) :
			(uint64) (previousBlock - element->blkno);
		result->totalAbsBlockDelta += blockDelta;
		result->maxAbsBlockDelta = Max(result->maxAbsBlockDelta, blockDelta);
		if (element->blkno != previousBlock)
			result->pageRuns++;
	}

	if (sampleIndex != result->sampleCount)
		HnswCloneCorruption(disk->index,
						"the BFS locality rank sample is incomplete");
	HnswDiskGraphCheckMemory(disk);
	hash_destroy(visited);
	MemoryContextSwitchTo(oldContext);
	MemoryContextDelete(scratchContext);
	pfree(queue);
}

static bool
HnswCloneIndexDefinitionsEqual(Relation source, Relation destination)
{
	Form_pg_index sourceIndex = source->rd_index;
	Form_pg_index destinationIndex = destination->rd_index;

	if (sourceIndex->indnatts != destinationIndex->indnatts ||
		sourceIndex->indnkeyatts != destinationIndex->indnkeyatts)
		return false;
	for (int i = 0; i < sourceIndex->indnatts; i++)
	{
		if (sourceIndex->indkey.values[i] != destinationIndex->indkey.values[i] ||
			source->rd_indcollation[i] != destination->rd_indcollation[i] ||
			source->rd_indoption[i] != destination->rd_indoption[i] ||
			get_index_column_opclass(RelationGetRelid(source), i + 1) !=
			get_index_column_opclass(RelationGetRelid(destination), i + 1))
			return false;
	}
	if (!equal(RelationGetIndexExpressions(source),
			   RelationGetIndexExpressions(destination)) ||
		!equal(RelationGetIndexPredicate(source),
			   RelationGetIndexPredicate(destination)))
		return false;
	return true;
}

static void
HnswValidateCloneSource(HnswBuildState *buildstate, Relation source)
{
	Relation	destination = buildstate->index;
	TupleDesc	sourceDesc;
	TupleDesc	destinationDesc;
	Form_pg_attribute sourceAttr;
	Form_pg_attribute destinationAttr;

	if (buildstate->indexInfo->ii_Concurrent)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hnsw.clone_source does not support CREATE INDEX CONCURRENTLY")));
	if (!hnsw_require_full_memory_build)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("hnsw.clone_source requires hnsw.require_full_memory_build = on")));
	if (hnsw_build_page_order != HNSW_BUILD_PAGE_ORDER_BFS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("hnsw.clone_source requires hnsw.build_page_order = bfs")));
	if (RelationGetRelid(source) == RelationGetRelid(destination))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("hnsw.clone_source cannot name the index being built")));
	if (source->rd_rel->relkind != RELKIND_INDEX || source->rd_index == NULL ||
		source->rd_rel->relam != destination->rd_rel->relam)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("hnsw.clone_source must name an HNSW index")));
	if (!source->rd_index->indisvalid || !source->rd_index->indisready ||
		!source->rd_index->indislive)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("source HNSW index \"%s\" is not valid and ready",
						RelationGetRelationName(source))));
	if (source->rd_index->indrelid != RelationGetRelid(buildstate->heap))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("source HNSW index \"%s\" is on a different heap",
						RelationGetRelationName(source))));
	if (!HnswCloneIndexDefinitionsEqual(source, destination))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("source and destination HNSW index definitions do not match"),
				 errdetail("The indexed column or expression, predicate, opclass, collation, or per-column options differ.")));
	if (HnswGetM(source) != buildstate->m ||
		HnswGetEfConstruction(source) != buildstate->efConstruction)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("source and destination HNSW build options do not match"),
				 errdetail("Both m and ef_construction must match exactly.")));

	sourceDesc = RelationGetDescr(source);
	destinationDesc = RelationGetDescr(destination);
	if (sourceDesc->natts != 1 || destinationDesc->natts != 1)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("exact HNSW graph cloning supports one index attribute")));
	sourceAttr = TupleDescAttr(sourceDesc, 0);
	destinationAttr = TupleDescAttr(destinationDesc, 0);
	if (sourceAttr->atttypid != destinationAttr->atttypid ||
		sourceAttr->atttypmod != destinationAttr->atttypmod ||
		sourceAttr->attcollation != destinationAttr->attcollation)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("source and destination HNSW index types or dimensions do not match")));
}

void
HnswCloneGraph(HnswBuildState *buildstate)
{
	Relation	source;
	Oid			sourceOid;
	List	   *names;
	RangeVar   *range;
	HnswDiskGraph disk;

	if (buildstate->heap == NULL || buildstate->forkNum != MAIN_FORKNUM)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hnsw.clone_source supports only a main-fork heap index build")));

	#if PG_VERSION_NUM >= 160000
	names = stringToQualifiedNameList(hnsw_clone_source, NULL);
#else
	names = stringToQualifiedNameList(hnsw_clone_source);
	#endif
	range = makeRangeVarFromNameList(names);
	sourceOid = RangeVarGetRelid(range, NoLock, false);
	if (!object_ownercheck(RelationRelationId, sourceOid, GetUserId()))
		aclcheck_error(ACLCHECK_NOT_OWNER, OBJECT_INDEX, get_rel_name(sourceOid));

	/* Hold the heap against DML/vacuum and the source against replacement. */
	LockRelationOid(RelationGetRelid(buildstate->heap), ShareLock);
	source = relation_open(sourceOid, ShareLock);
	HnswValidateCloneSource(buildstate, source);

	MemSet(&disk, 0, sizeof(disk));
	disk.index = source;
	disk.context = buildstate->graphCtx;
	disk.graph = buildstate->graph;
	disk.allocator = &buildstate->allocator;
	disk.purpose = HNSW_DISK_GRAPH_CLONE;
	disk.collectPhysicalDigest = false;
	disk.updateProgress = true;

	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_TOTAL,
							Max((int64) 0, (int64) source->rd_rel->reltuples));
	HnswLoadDiskGraph(&disk);

	if (disk.meta.dimensions != (uint32) buildstate->dimensions ||
		disk.meta.m != buildstate->m ||
		disk.meta.efConstruction != buildstate->efConstruction)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("source HNSW metapage does not match the destination definition")));

	buildstate->graph->indtuples = disk.heapTids;
	buildstate->indtuples = disk.heapTids;
	buildstate->reltuples = buildstate->heap->rd_rel->reltuples >= 0 ?
		buildstate->heap->rd_rel->reltuples : disk.heapTids;
	if (source->rd_index->indcheckxmin)
		buildstate->indexInfo->ii_BrokenHotChain = true;
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_TOTAL, disk.heapTids);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, disk.heapTids);

	/* Retain ShareLock to transaction end; the relcache reference is no longer needed. */
	relation_close(source, NoLock);
}

static int
HnswLogicalElementCompare(const void *left, const void *right)
{
	HnswElement a = *((const HnswElement *) left);
	HnswElement b = *((const HnswElement *) right);
	int			common = Min(a->heaptidsLength, b->heaptidsLength);

	if (a->heaptidsLength == 0 || b->heaptidsLength == 0)
	{
		if (a->heaptidsLength == 0 && b->heaptidsLength != 0)
			return 1;
		if (a->heaptidsLength != 0 && b->heaptidsLength == 0)
			return -1;
		if (a->blkno < b->blkno)
			return -1;
		if (a->blkno > b->blkno)
			return 1;
		if (a->offno < b->offno)
			return -1;
		if (a->offno > b->offno)
			return 1;
		return 0;
	}

	for (int i = 0; i < common; i++)
	{
		BlockNumber aBlock = ItemPointerGetBlockNumber(&a->heaptids[i]);
		BlockNumber bBlock = ItemPointerGetBlockNumber(&b->heaptids[i]);
		OffsetNumber aOffset = ItemPointerGetOffsetNumber(&a->heaptids[i]);
		OffsetNumber bOffset = ItemPointerGetOffsetNumber(&b->heaptids[i]);

		if (aBlock < bBlock)
			return -1;
		if (aBlock > bBlock)
			return 1;
		if (aOffset < bOffset)
			return -1;
		if (aOffset > bOffset)
			return 1;
	}
	if (a->heaptidsLength < b->heaptidsLength)
		return -1;
	if (a->heaptidsLength > b->heaptidsLength)
		return 1;
	return 0;
}

static void
HnswDigestElementIdentity(HnswDigest *digest, HnswElement element)
{
	if (element->heaptidsLength == 0)
	{
		HnswDigestUint8(digest, 1);
		HnswDigestUint32(digest, element->blkno);
		HnswDigestUint16(digest, element->offno);
	}
	else
	{
		HnswDigestUint8(digest, 0);
		HnswDigestUint8(digest, element->heaptidsLength);
		for (int i = 0; i < element->heaptidsLength; i++)
			HnswDigestItemPointer(digest, &element->heaptids[i]);
	}
}

static int
HnswHeapTidCompare(const void *left, const void *right)
{
	ItemPointer a = (ItemPointer) left;
	ItemPointer b = (ItemPointer) right;
	BlockNumber aBlock = ItemPointerGetBlockNumber(a);
	BlockNumber bBlock = ItemPointerGetBlockNumber(b);
	OffsetNumber aOffset = ItemPointerGetOffsetNumber(a);
	OffsetNumber bOffset = ItemPointerGetOffsetNumber(b);

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

static void
HnswBuildDefinitionDigest(HnswDiskGraph *disk,
						  HnswGraphFingerprint *result)
{
	static const char domain[] = "SQLENS-HNSW-DEFINITION-V1";
	Relation	index = disk->index;
	Form_pg_index definition = index->rd_index;
	TupleDesc	desc = RelationGetDescr(index);
	List	   *expressions = RelationGetIndexExpressions(index);
	List	   *predicate = RelationGetIndexPredicate(index);
	char	   *serialized;
	HnswDigest	digest;

	HnswDigestStart(&digest);
	HnswDigestBytes(&digest, domain, sizeof(domain) - 1);
	HnswDigestUint32(&digest, index->rd_rel->relam);
	HnswDigestUint16(&digest, definition->indnatts);
	HnswDigestUint16(&digest, definition->indnkeyatts);
	HnswDigestUint8(&digest, definition->indisunique ? 1 : 0);
#if PG_VERSION_NUM >= 150000
	HnswDigestUint8(&digest, definition->indnullsnotdistinct ? 1 : 0);
#else
	HnswDigestUint8(&digest, 0);
#endif
	for (int i = 0; i < definition->indnatts; i++)
	{
		Form_pg_attribute attr = TupleDescAttr(desc, i);

		HnswDigestUint16(&digest, (uint16) definition->indkey.values[i]);
		HnswDigestUint32(&digest, index->rd_indcollation[i]);
		HnswDigestUint16(&digest, index->rd_indoption[i]);
		HnswDigestUint32(&digest,
						 get_index_column_opclass(RelationGetRelid(index), i + 1));
		HnswDigestUint32(&digest, attr->atttypid);
		HnswDigestUint32(&digest, (uint32) attr->atttypmod);
		HnswDigestUint32(&digest, attr->attcollation);
	}
	serialized = nodeToString(expressions);
	HnswDiskGraphCheckMemory(disk);
	HnswDigestString(&digest, serialized);
	pfree(serialized);
	serialized = nodeToString(predicate);
	HnswDiskGraphCheckMemory(disk);
	HnswDigestString(&digest, serialized);
	pfree(serialized);
	HnswDigestUint16(&digest, HnswGetM(index));
	HnswDigestUint16(&digest, HnswGetEfConstruction(index));
	HnswDigestFinal(&digest, result->definitionDigest);
	HnswDiskGraphCheckMemory(disk);
}

static void
HnswBuildTupleCoverageDigest(HnswDiskGraph *disk,
							 HnswGraphFingerprint *result)
{
	static const char domain[] = "SQLENS-HNSW-TID-SET-V1";
	ItemPointerData *tids;
	HnswElementPtr iter;
	HnswDigest	digest;
	Size		arraySize;
	int64		idx = 0;
	char	   *base = NULL;

	if ((uint64) disk->heapTids > MaxAllocSize / sizeof(ItemPointerData))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("HNSW index has too many heap TIDs to fingerprint canonically")));
	arraySize = (Size) disk->heapTids * sizeof(ItemPointerData);
	HnswDiskGraphReserveMemory(disk, arraySize);
	tids = arraySize > 0 ? palloc(arraySize) : NULL;
	HnswDiskGraphCheckMemory(disk);

	iter = disk->graph->head;
	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);

		for (int i = 0; i < element->heaptidsLength; i++)
		{
			if (idx >= disk->heapTids)
				HnswCloneCorruption(disk->index,
								"the in-memory heap-TID count changed");
			tids[idx++] = element->heaptids[i];
		}
		iter = element->next;
	}
	if (idx != disk->heapTids)
		HnswCloneCorruption(disk->index, "the in-memory heap-TID count changed");
	if (disk->heapTids > 1)
		qsort(tids, disk->heapTids, sizeof(ItemPointerData), HnswHeapTidCompare);
	for (int64 i = 1; i < disk->heapTids; i++)
	{
		if (HnswHeapTidCompare(&tids[i - 1], &tids[i]) == 0)
			HnswCloneCorruption(disk->index,
							"a heap TID appears in more than one graph element");
	}

	HnswDigestStart(&digest);
	HnswDigestBytes(&digest, domain, sizeof(domain) - 1);
	HnswDigestUint64(&digest, disk->heapTids);
	for (int64 i = 0; i < disk->heapTids; i++)
		HnswDigestItemPointer(&digest, &tids[i]);
	HnswDigestFinal(&digest, result->tupleCoverageDigest);
	if (tids != NULL)
		pfree(tids);
}

static void
HnswElementIdentityDigest(HnswElement element,
						  uint8 output[PG_SHA256_DIGEST_LENGTH])
{
	static const char domain[] = "SQLENS-HNSW-NODE-IDENTITY-V1";
	HnswDigest	digest;

	HnswDigestStart(&digest);
	HnswDigestBytes(&digest, domain, sizeof(domain) - 1);
	HnswDigestElementIdentity(&digest, element);
	HnswDigestFinal(&digest, output);
}

static void
HnswBuildLogicalDigest(HnswDiskGraph *disk, HnswGraphFingerprint *result)
{
	static const char domain[] = "SQLENS-HNSW-LOGICAL-V1";
	HnswElement *elements;
	HnswElementPtr iter;
	HnswElement entryPoint;
	HnswDigest	digest;
	Size		arraySize;
	int64		idx = 0;
	char	   *base = NULL;

	if ((uint64) disk->nodes > MaxAllocSize / sizeof(HnswElement))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("HNSW graph has too many nodes to fingerprint canonically")));
	arraySize = (Size) disk->nodes * sizeof(HnswElement);
	HnswDiskGraphReserveMemory(disk, arraySize);
	elements = arraySize > 0 ? palloc(arraySize) : NULL;
	HnswDiskGraphCheckMemory(disk);
	iter = disk->graph->head;
	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);

		if (idx >= disk->nodes)
			HnswCloneCorruption(disk->index, "the in-memory element list is cyclic");
		elements[idx++] = element;
		iter = element->next;
	}
	if (idx != disk->nodes)
		HnswCloneCorruption(disk->index, "the in-memory element count changed");

	if (disk->nodes > 1)
		qsort(elements, disk->nodes, sizeof(HnswElement),
			  HnswLogicalElementCompare);
	for (int64 i = 1; i < disk->nodes; i++)
	{
		if (HnswLogicalElementCompare(&elements[i - 1], &elements[i]) == 0)
			HnswCloneCorruption(disk->index,
							"two graph nodes have the same ordered heap-TID identity");
	}

	HnswDigestStart(&digest);
	HnswDigestBytes(&digest, domain, sizeof(domain) - 1);
	HnswDigestUint32(&digest, disk->meta.version);
	HnswDigestUint32(&digest, disk->meta.dimensions);
	HnswDigestUint16(&digest, disk->meta.m);
	HnswDigestUint16(&digest, disk->meta.efConstruction);
	HnswDigestUint64(&digest, disk->nodes);
	HnswDigestUint64(&digest, disk->heapTids);

	entryPoint = HnswPtrAccess(base, disk->graph->entryPoint);
	HnswDigestUint8(&digest, entryPoint != NULL ? 1 : 0);
	if (entryPoint != NULL)
	{
		HnswDigestElementIdentity(&digest, entryPoint);
		HnswDigestUint8(&digest, entryPoint->level);
		HnswElementIdentityDigest(entryPoint, result->entryIdentity);
		result->hasEntry = true;
		result->entryLevel = entryPoint->level;
	}
	else
	{
		result->hasEntry = false;
		result->entryLevel = -1;
	}

	for (int64 i = 0; i < disk->nodes; i++)
	{
		HnswElement element = elements[i];
		Pointer		value = HnswPtrAccess(base, element->value);
		Size		valueSize = element->deleted ? 0 : VARSIZE_ANY(value);

		HnswDigestElementIdentity(&digest, element);
		HnswDigestUint8(&digest, element->level);
		HnswDigestUint8(&digest, element->deleted);
		HnswDigestUint8(&digest, element->version);
		HnswDigestUint32(&digest, valueSize);
		if (valueSize > 0)
			HnswDigestBytes(&digest, value, valueSize);

		for (int lc = element->level; lc >= 0; lc--)
		{
			HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, lc);

			HnswDigestUint8(&digest, lc);
			HnswDigestUint16(&digest, neighbors->length);
			for (int n = 0; n < neighbors->length; n++)
			{
				HnswElement neighbor = HnswPtrAccess(base,
											 neighbors->items[n].element);

				if (neighbor == NULL)
					HnswCloneCorruption(disk->index,
									"an in-memory neighbor is unresolved");
				HnswDigestElementIdentity(&digest, neighbor);
			}
		}
	}

	HnswDigestFinal(&digest, result->logicalDigest);
	if (elements != NULL)
		pfree(elements);
	HnswDiskGraphCheckMemory(disk);
}

static void
HnswValidateProofIndex(Relation index)
{
	Oid			hnswAmOid = get_index_am_oid("hnsw", false);

	if (index->rd_rel->relkind != RELKIND_INDEX || index->rd_index == NULL ||
		index->rd_rel->relam != hnswAmOid)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("relation \"%s\" is not an HNSW index",
						RelationGetRelationName(index))));
	if (!index->rd_index->indisvalid || !index->rd_index->indisready ||
		!index->rd_index->indislive)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("HNSW index \"%s\" is not valid and ready",
						RelationGetRelationName(index))));
}

static void
HnswFingerprintRelation(Relation index, HnswGraphFingerprint *result)
{
	MemoryContext context;
	MemoryContext oldContext;
	HnswGraph	graph;
	HnswDiskGraph disk;
	char	   *base = NULL;

	MemSet(result, 0, sizeof(*result));
	result->heapOid = index->rd_index->indrelid;
	context = AllocSetContextCreate(CurrentMemoryContext,
								"HNSW graph fingerprint",
								ALLOCSET_DEFAULT_SIZES);
	oldContext = MemoryContextSwitchTo(context);
	MemSet(&graph, 0, sizeof(graph));
	HnswPtrStore(base, graph.head, (HnswElement) NULL);
	HnswPtrStore(base, graph.entryPoint, (HnswElement) NULL);
	graph.memoryTotal = (Size) maintenance_work_mem * 1024L;

	MemSet(&disk, 0, sizeof(disk));
	disk.index = index;
	disk.context = context;
	disk.graph = &graph;
	disk.allocator = NULL;
	disk.purpose = HNSW_DISK_GRAPH_FINGERPRINT;
	disk.collectPhysicalDigest = true;
	HnswLoadDiskGraph(&disk);

	result->version = disk.meta.version;
	result->dimensions = disk.meta.dimensions;
	result->m = disk.meta.m;
	result->efConstruction = disk.meta.efConstruction;
	result->nodes = disk.nodes;
	result->heapTids = disk.heapTids;
	result->tombstones = disk.tombstones;
	result->maxLevel = disk.maxLevel;
	memcpy(result->physicalDigest, disk.physicalDigest,
		   sizeof(result->physicalDigest));
	HnswBuildDefinitionDigest(&disk, result);
	HnswBuildLogicalDigest(&disk, result);
	HnswBuildTupleCoverageDigest(&disk, result);
	HnswBuildBfsLocality(&disk, &result->bfsLocality);
	HnswDiskGraphCheckMemory(&disk);

	MemoryContextSwitchTo(oldContext);
	MemoryContextDelete(context);
}

static void
HnswDigestToHex(const uint8 digest[PG_SHA256_DIGEST_LENGTH],
				 char output[PG_SHA256_DIGEST_STRING_LENGTH])
{
	hex_encode((const char *) digest, PG_SHA256_DIGEST_LENGTH, output);
	output[PG_SHA256_DIGEST_STRING_LENGTH - 1] = '\0';
}

static void
HnswAppendBfsLocalityJson(StringInfo json, const char *field,
						  const HnswBfsLocality *locality)
{
	double		adjacentDenominator = locality->adjacentPairs > 0 ?
			(double) locality->adjacentPairs : 1.0;

	appendStringInfo(json,
					 "\"%s\":{"
					 "\"format\":\"sqlens-hnsw-bfs-locality-v1\","
					 "\"rank_base\":0,"
					 "\"graph_nodes\":" INT64_FORMAT ","
					 "\"reachable_nodes\":" INT64_FORMAT ","
					 "\"fallback_nodes\":" INT64_FORMAT ","
					 "\"sequence_nodes\":" INT64_FORMAT ","
					 "\"adjacent_pairs\":" INT64_FORMAT ","
					 "\"same_block_pairs\":" INT64_FORMAT ","
					 "\"next_block_pairs\":" INT64_FORMAT ","
					 "\"same_or_next_page_pairs\":" INT64_FORMAT ","
					 "\"nondecreasing_pairs\":" INT64_FORMAT ","
					 "\"backward_pairs\":" INT64_FORMAT ","
					 "\"total_abs_block_delta\":" UINT64_FORMAT ","
					 "\"max_abs_block_delta\":" UINT64_FORMAT ","
					 "\"page_runs\":" INT64_FORMAT ","
					 "\"same_block_ratio\":%.17g,"
					 "\"same_or_next_page_ratio\":%.17g,"
					 "\"nondecreasing_ratio\":%.17g,"
					 "\"full_statistics\":true,"
					 "\"sample_limit\":%d,"
					 "\"sample_count\":%d,"
					 "\"sample_truncated\":%s,"
					 "\"sample_strategy\":\"evenly_spaced_inclusive\","
					 "\"rank_samples\":[",
					 field, locality->graphNodes, locality->reachableNodes,
					 locality->fallbackNodes, locality->sequenceNodes,
					 locality->adjacentPairs, locality->sameBlockPairs,
					 locality->nextBlockPairs, locality->sameOrNextPagePairs,
					 locality->nondecreasingPairs, locality->backwardPairs,
					 locality->totalAbsBlockDelta, locality->maxAbsBlockDelta,
					 locality->pageRuns,
					 (double) locality->sameBlockPairs / adjacentDenominator,
					 (double) locality->sameOrNextPagePairs / adjacentDenominator,
					 (double) locality->nondecreasingPairs / adjacentDenominator,
					 HNSW_BFS_LOCALITY_SAMPLE_LIMIT, locality->sampleCount,
					 locality->sampleCount < locality->sequenceNodes ? "true" : "false");
	for (int i = 0; i < locality->sampleCount; i++)
	{
		if (i > 0)
			appendStringInfoChar(json, ',');
		appendStringInfo(json,
						 "{\"rank\":" INT64_FORMAT ",\"block\":%u,"
						 "\"offset\":%u}",
						 locality->samples[i].rank,
						 (uint32) locality->samples[i].block,
						 (uint32) locality->samples[i].offno);
	}
	appendStringInfoString(json, "]}");
}

static Jsonb *
HnswFingerprintToJsonb(const HnswGraphFingerprint *fingerprint)
{
	char		logical[PG_SHA256_DIGEST_STRING_LENGTH];
	char		physical[PG_SHA256_DIGEST_STRING_LENGTH];
	char		definition[PG_SHA256_DIGEST_STRING_LENGTH];
	char		coverage[PG_SHA256_DIGEST_STRING_LENGTH];
	char		entry[PG_SHA256_DIGEST_STRING_LENGTH];
	StringInfoData json;

	HnswDigestToHex(fingerprint->logicalDigest, logical);
	HnswDigestToHex(fingerprint->physicalDigest, physical);
	HnswDigestToHex(fingerprint->definitionDigest, definition);
	HnswDigestToHex(fingerprint->tupleCoverageDigest, coverage);
	if (fingerprint->hasEntry)
		HnswDigestToHex(fingerprint->entryIdentity, entry);
	initStringInfo(&json);
	appendStringInfo(&json,
						 "{\"format\":\"sqlens-hnsw-graph-v2\","
						 "\"definition_digest\":\"sha256:%s\","
						 "\"tuple_coverage_digest\":\"sha256:%s\","
						 "\"logical_digest\":\"sha256:%s\","
						 "\"physical_digest\":\"sha256:%s\","
						 "\"nodes\":" INT64_FORMAT ","
						 "\"heap_tids\":" INT64_FORMAT ","
						 "\"tombstones\":" INT64_FORMAT ","
					 "\"entry_identity\":%s,"
						 "\"entry_level\":%d,\"max_level\":%d,"
						 "\"version\":%u,\"dimensions\":%u,"
						 "\"m\":%u,\"ef_construction\":%u,",
						 definition, coverage, logical, physical,
						 fingerprint->nodes, fingerprint->heapTids,
						 fingerprint->tombstones,
					 fingerprint->hasEntry ? psprintf("\"sha256:%s\"", entry) : "null",
						 fingerprint->entryLevel, fingerprint->maxLevel,
						 fingerprint->version, fingerprint->dimensions,
						 fingerprint->m, fingerprint->efConstruction);
	HnswAppendBfsLocalityJson(&json, "bfs_locality", &fingerprint->bfsLocality);
	appendStringInfoChar(&json, '}');
	return DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
										 CStringGetDatum(json.data)));
}

PG_FUNCTION_INFO_V1(vector_hnsw_graph_fingerprint);

Datum
vector_hnsw_graph_fingerprint(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Oid			heapOid = IndexGetRelation(indexOid, false);
	Relation	index;
	HnswGraphFingerprint fingerprint;

	if (!object_ownercheck(RelationRelationId, indexOid, GetUserId()))
		aclcheck_error(ACLCHECK_NOT_OWNER, OBJECT_INDEX, get_rel_name(indexOid));
	LockRelationOid(heapOid, ShareLock);
	index = relation_open(indexOid, ShareLock);
	HnswValidateProofIndex(index);
	HnswFingerprintRelation(index, &fingerprint);
	relation_close(index, NoLock);
	PG_RETURN_JSONB_P(HnswFingerprintToJsonb(&fingerprint));
}

PG_FUNCTION_INFO_V1(vector_hnsw_graph_compare);

Datum
vector_hnsw_graph_compare(PG_FUNCTION_ARGS)
{
	Oid			leftOid = PG_GETARG_OID(0);
	Oid			rightOid = PG_GETARG_OID(1);
	Oid			leftHeap = IndexGetRelation(leftOid, false);
	Oid			rightHeap = IndexGetRelation(rightOid, false);
	Relation	left;
	Relation	right;
	HnswGraphFingerprint leftFingerprint;
	HnswGraphFingerprint rightFingerprint;
	bool		sameHeap;
	bool		logicalEqual;
	bool		physicalEqual;
	bool		definitionEqual;
	bool		entryEqual;
	bool		coverageEqual;
	char		leftLogical[PG_SHA256_DIGEST_STRING_LENGTH];
	char		rightLogical[PG_SHA256_DIGEST_STRING_LENGTH];
	char		leftPhysical[PG_SHA256_DIGEST_STRING_LENGTH];
	char		rightPhysical[PG_SHA256_DIGEST_STRING_LENGTH];
	char		leftDefinition[PG_SHA256_DIGEST_STRING_LENGTH];
	char		rightDefinition[PG_SHA256_DIGEST_STRING_LENGTH];
	char		leftCoverage[PG_SHA256_DIGEST_STRING_LENGTH];
	char		rightCoverage[PG_SHA256_DIGEST_STRING_LENGTH];
	StringInfoData json;

	if (!object_ownercheck(RelationRelationId, leftOid, GetUserId()))
		aclcheck_error(ACLCHECK_NOT_OWNER, OBJECT_INDEX, get_rel_name(leftOid));
	if (!object_ownercheck(RelationRelationId, rightOid, GetUserId()))
		aclcheck_error(ACLCHECK_NOT_OWNER, OBJECT_INDEX, get_rel_name(rightOid));

	if (leftHeap <= rightHeap)
	{
		LockRelationOid(leftHeap, ShareLock);
		if (rightHeap != leftHeap)
			LockRelationOid(rightHeap, ShareLock);
	}
	else
	{
		LockRelationOid(rightHeap, ShareLock);
		LockRelationOid(leftHeap, ShareLock);
	}
	left = relation_open(leftOid, ShareLock);
	right = relation_open(rightOid, ShareLock);
	HnswValidateProofIndex(left);
	HnswValidateProofIndex(right);
	HnswFingerprintRelation(left, &leftFingerprint);
	HnswFingerprintRelation(right, &rightFingerprint);

	sameHeap = leftHeap == rightHeap;
	logicalEqual = memcmp(leftFingerprint.logicalDigest,
						  rightFingerprint.logicalDigest,
						  PG_SHA256_DIGEST_LENGTH) == 0;
	physicalEqual = memcmp(leftFingerprint.physicalDigest,
						   rightFingerprint.physicalDigest,
						   PG_SHA256_DIGEST_LENGTH) == 0;
	entryEqual = leftFingerprint.hasEntry == rightFingerprint.hasEntry &&
		leftFingerprint.entryLevel == rightFingerprint.entryLevel &&
		(!leftFingerprint.hasEntry ||
			 memcmp(leftFingerprint.entryIdentity, rightFingerprint.entryIdentity,
					PG_SHA256_DIGEST_LENGTH) == 0);
	definitionEqual = memcmp(leftFingerprint.definitionDigest,
							 rightFingerprint.definitionDigest,
							 PG_SHA256_DIGEST_LENGTH) == 0;
	coverageEqual = memcmp(leftFingerprint.tupleCoverageDigest,
						   rightFingerprint.tupleCoverageDigest,
						   PG_SHA256_DIGEST_LENGTH) == 0;
	HnswDigestToHex(leftFingerprint.logicalDigest, leftLogical);
	HnswDigestToHex(rightFingerprint.logicalDigest, rightLogical);
	HnswDigestToHex(leftFingerprint.physicalDigest, leftPhysical);
	HnswDigestToHex(rightFingerprint.physicalDigest, rightPhysical);
	HnswDigestToHex(leftFingerprint.definitionDigest, leftDefinition);
	HnswDigestToHex(rightFingerprint.definitionDigest, rightDefinition);
	HnswDigestToHex(leftFingerprint.tupleCoverageDigest, leftCoverage);
	HnswDigestToHex(rightFingerprint.tupleCoverageDigest, rightCoverage);

	initStringInfo(&json);
	appendStringInfo(&json,
						 "{\"format\":\"sqlens-hnsw-compare-v2\","
						 "\"same_heap\":%s,\"logical_equal\":%s,"
						 "\"physical_equal\":%s,\"entry_equal\":%s,"
						 "\"definition_equal\":%s,"
						 "\"tuple_coverage_equal\":%s,"
						 "\"left_definition_digest\":\"sha256:%s\","
						 "\"right_definition_digest\":\"sha256:%s\","
						 "\"left_tuple_coverage_digest\":\"sha256:%s\","
						 "\"right_tuple_coverage_digest\":\"sha256:%s\","
						 "\"left_logical_digest\":\"sha256:%s\","
						 "\"right_logical_digest\":\"sha256:%s\","
						 "\"left_physical_digest\":\"sha256:%s\","
						 "\"right_physical_digest\":\"sha256:%s\",",
						 sameHeap ? "true" : "false",
					 logicalEqual ? "true" : "false",
						 physicalEqual ? "true" : "false",
						 entryEqual ? "true" : "false",
						 definitionEqual ? "true" : "false",
						 coverageEqual ? "true" : "false",
						 leftDefinition, rightDefinition,
						 leftCoverage, rightCoverage,
						 leftLogical, rightLogical, leftPhysical, rightPhysical);
	HnswAppendBfsLocalityJson(&json, "left_bfs_locality",
						  &leftFingerprint.bfsLocality);
	appendStringInfoChar(&json, ',');
	HnswAppendBfsLocalityJson(&json, "right_bfs_locality",
						  &rightFingerprint.bfsLocality);
	appendStringInfoChar(&json, '}');

	relation_close(right, NoLock);
	relation_close(left, NoLock);
	PG_RETURN_JSONB_P(DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												  CStringGetDatum(json.data))));
}

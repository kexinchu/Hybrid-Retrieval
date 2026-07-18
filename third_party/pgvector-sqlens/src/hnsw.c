#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>

#include "access/amapi.h"
#include "access/genam.h"
#include "access/relation.h"
#include "access/reloptions.h"
#include "catalog/index.h"
#include "catalog/pg_index.h"
#include "catalog/pg_class_d.h"
#include "commands/defrem.h"
#include "commands/progress.h"
#include "commands/vacuum.h"
#include "fmgr.h"
#include "hnsw.h"
#include "miscadmin.h"
#include "nodes/pg_list.h"
#include "utils/float.h"
#include "utils/builtins.h"
#include "utils/acl.h"
#include "utils/fmgrprotos.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/relcache.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"
#include "utils/plancache.h"
#include "utils/syscache.h"
#include "vector.h"

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

static const struct config_enum_entry hnsw_iterative_scan_options[] = {
	{"off", HNSW_ITERATIVE_SCAN_OFF, false},
	{"relaxed_order", HNSW_ITERATIVE_SCAN_RELAXED, false},
	{"strict_order", HNSW_ITERATIVE_SCAN_STRICT, false},
	{NULL, 0, false}
};

static const struct config_enum_entry hnsw_page_access_options[] = {
	{"off", HNSW_PAGE_ACCESS_OFF, false},
	{"prefetch", HNSW_PAGE_ACCESS_PREFETCH, false},
	{"reorder", HNSW_PAGE_ACCESS_REORDER, false},
	{NULL, 0, false}
};

static const struct config_enum_entry hnsw_index_page_access_options[] = {
	{"off", HNSW_INDEX_PAGE_ACCESS_OFF, false},
	{"prefetch", HNSW_INDEX_PAGE_ACCESS_PREFETCH, false},
	{NULL, 0, false}
};

static const struct config_enum_entry hnsw_build_page_order_options[] = {
	{"insertion", HNSW_BUILD_PAGE_ORDER_INSERTION, false},
	{"bfs", HNSW_BUILD_PAGE_ORDER_BFS, false},
	{NULL, 0, false}
};

static const struct config_enum_entry hnsw_filter_strategy_options[] = {
	{"off", HNSW_FILTER_STRATEGY_OFF, false},
	{"acorn1", HNSW_FILTER_STRATEGY_ACORN1, false},
	{"guided_collect", HNSW_FILTER_STRATEGY_GUIDED_COLLECT, false},
	{"traversal_guided", HNSW_FILTER_STRATEGY_TRAVERSAL_GUIDED, false},
	{"safe_guided", HNSW_FILTER_STRATEGY_SAFE_GUIDED, false},
	{NULL, 0, false}
};

int			hnsw_ef_search;
int			hnsw_iterative_scan;
int			hnsw_max_scan_tuples;
int			hnsw_page_access;
int			hnsw_page_window;
int			hnsw_page_prefetch_min_items;
int			hnsw_page_disable_after_no_merge;
int			hnsw_index_page_access;
int			hnsw_build_page_order;
int			hnsw_build_seed;
bool		hnsw_require_full_memory_build;
char	   *hnsw_clone_source;
char	   *hnsw_preferred_index;
int			hnsw_filter_strategy;
int			hnsw_guided_collect_target;
int			hnsw_traversal_guided_target;
int			hnsw_traversal_guided_max_bridge_hops;
int			hnsw_traversal_guided_max_bridge_work;
double		hnsw_traversal_guided_min_skip_rate;
double		hnsw_scan_mem_multiplier;
int			hnsw_lock_tranche_id;
static relopt_kind hnsw_relopt_kind;
static Oid	hnsw_preferred_index_oid = InvalidOid;

static void
HnswValidatePreferredIndex(Relation index, Oid expectedHeapOid)
{
	char	   *amName = index->rd_rel->relkind == RELKIND_INDEX ?
		get_am_name(index->rd_rel->relam) : NULL;
	bool		isHnsw = amName != NULL && strcmp(amName, "hnsw") == 0;

	if (index->rd_rel->relkind != RELKIND_INDEX || index->rd_index == NULL ||
		!isHnsw)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("hnsw.preferred_index must name an HNSW index"),
				 errdetail("Relation \"%s\" is not a valid HNSW index.",
						   RelationGetRelationName(index))));
	pfree(amName);
	if (!index->rd_index->indisvalid || !index->rd_index->indisready ||
		!index->rd_index->indislive)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("hnsw.preferred_index must name a valid and ready index"),
				 errdetail("HNSW index \"%s\" is invalid, not ready, or not live.",
						   RelationGetRelationName(index))));
	if (OidIsValid(expectedHeapOid) &&
		index->rd_index->indrelid != expectedHeapOid)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("hnsw.preferred_index is on a different heap"),
				 errdetail("Preferred index \"%s\" belongs to relation %u, but this HNSW path belongs to relation %u.",
						   RelationGetRelationName(index),
						   index->rd_index->indrelid, expectedHeapOid)));
}

static bool
HnswPreferredIndexCheck(char **newval, void **extra, GucSource source)
{
	Oid		   *parsedOid = guc_malloc(ERROR, sizeof(Oid));
	HeapTuple	relationTuple;
	HeapTuple	indexTuple;
	Form_pg_class relationForm;
	Form_pg_index indexForm;
	char	   *relationName;
	char	   *amName;
	bool		isHnsw;

	(void) source;
	*parsedOid = InvalidOid;
	if ((*newval)[0] == '\0')
	{
		*extra = parsedOid;
		return true;
	}

	*parsedOid = DatumGetObjectId(DirectFunctionCall1(
		regclassin, CStringGetDatum(*newval)));
	if (!object_ownercheck(RelationRelationId, *parsedOid, GetUserId()))
		aclcheck_error(ACLCHECK_NOT_OWNER, OBJECT_INDEX, get_rel_name(*parsedOid));

	relationTuple = SearchSysCache1(RELOID, ObjectIdGetDatum(*parsedOid));
	if (!HeapTupleIsValid(relationTuple))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_TABLE),
				 errmsg("relation with OID %u does not exist", *parsedOid)));
	relationForm = (Form_pg_class) GETSTRUCT(relationTuple);
	relationName = pstrdup(NameStr(relationForm->relname));
	amName = relationForm->relkind == RELKIND_INDEX ?
		get_am_name(relationForm->relam) : NULL;
	isHnsw = amName != NULL && strcmp(amName, "hnsw") == 0;
	ReleaseSysCache(relationTuple);
	if (!isHnsw)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("hnsw.preferred_index must name an HNSW index"),
				 errdetail("Relation \"%s\" is not a valid HNSW index.",
						   relationName)));
	pfree(amName);

	indexTuple = SearchSysCache1(INDEXRELID, ObjectIdGetDatum(*parsedOid));
	if (!HeapTupleIsValid(indexTuple))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("hnsw.preferred_index must name an HNSW index")));
	indexForm = (Form_pg_index) GETSTRUCT(indexTuple);
	if (!indexForm->indisvalid || !indexForm->indisready ||
		!indexForm->indislive)
	{
		ReleaseSysCache(indexTuple);
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("hnsw.preferred_index must name a valid and ready index"),
				 errdetail("HNSW index \"%s\" is invalid, not ready, or not live.",
						   relationName)));
	}
	ReleaseSysCache(indexTuple);
	pfree(relationName);
	*extra = parsedOid;
	return true;
}

static void
HnswPreferredIndexAssign(const char *newval, void *extra)
{
	Oid			newOid = newval[0] == '\0' || extra == NULL ?
		InvalidOid : *((Oid *) extra);

	if (newOid != hnsw_preferred_index_oid)
	{
		hnsw_preferred_index_oid = newOid;
		ResetPlanCache();
	}
}

/*
 * Assign a tranche ID for our LWLocks. This only needs to be done by one
 * backend, as the tranche ID is remembered in shared memory.
 *
 * This shared memory area is very small, so we just allocate it from the
 * "slop" that PostgreSQL reserves for small allocations like this. If
 * this grows bigger, we should use a shmem_request_hook and
 * RequestAddinShmemSpace() to pre-reserve space for this.
 */
void
HnswInitLockTranche(void)
{
	int		   *tranche_ids;
	bool		found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	tranche_ids = ShmemInitStruct("hnsw LWLock ids",
								  sizeof(int) * 1,
								  &found);
	if (!found)
	{
#if PG_VERSION_NUM >= 190000
		tranche_ids[0] = LWLockNewTrancheId("HnswBuild");
#else
		tranche_ids[0] = LWLockNewTrancheId();
#endif
	}
	hnsw_lock_tranche_id = tranche_ids[0];
	LWLockRelease(AddinShmemInitLock);

#if PG_VERSION_NUM < 190000
	/* Per-backend registration of the tranche ID */
	LWLockRegisterTranche(hnsw_lock_tranche_id, "HnswBuild");
#endif
}

/*
 * Initialize index options and variables
 */
void
HnswInit(void)
{
	if (!process_shared_preload_libraries_in_progress)
		HnswInitLockTranche();

	hnsw_relopt_kind = add_reloption_kind();
	add_int_reloption(hnsw_relopt_kind, "m", "Max number of connections",
					  HNSW_DEFAULT_M, HNSW_MIN_M, HNSW_MAX_M, AccessExclusiveLock);
	add_int_reloption(hnsw_relopt_kind, "ef_construction", "Size of the dynamic candidate list for construction",
					  HNSW_DEFAULT_EF_CONSTRUCTION, HNSW_MIN_EF_CONSTRUCTION, HNSW_MAX_EF_CONSTRUCTION, AccessExclusiveLock);

	DefineCustomIntVariable("hnsw.ef_search", "Sets the size of the dynamic candidate list for search",
							"Valid range is 1..10000.", &hnsw_ef_search,
							HNSW_DEFAULT_EF_SEARCH, HNSW_MIN_EF_SEARCH, HNSW_MAX_EF_SEARCH, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.iterative_scan", "Sets the mode for iterative scans",
							 NULL, &hnsw_iterative_scan,
							 HNSW_ITERATIVE_SCAN_OFF, hnsw_iterative_scan_options, PGC_USERSET, 0, NULL, NULL, NULL);

	/* This is approximate and does not affect the initial scan */
	DefineCustomIntVariable("hnsw.max_scan_tuples", "Sets the max number of tuples to visit for iterative scans",
							NULL, &hnsw_max_scan_tuples,
							20000, 1, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.page_access", "Sets heap-page-aware access mode for HNSW scans",
							 "prefetch preserves distance order; reorder is experimental and returns TIDs in heap page order within each window.",
							 &hnsw_page_access,
							 HNSW_PAGE_ACCESS_OFF, hnsw_page_access_options, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.page_window", "Sets the candidate window for heap-page-aware HNSW scans",
							"Valid range is 1..10000.", &hnsw_page_window,
							128, 1, 10000, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.page_prefetch_min_items", "Sets the minimum candidates per heap page before HNSW heap-page prefetch",
							"Prefetch is only issued for heap pages that have at least this many candidates in the current page window.",
							&hnsw_page_prefetch_min_items,
							2, 1, 10000, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.page_disable_after_no_merge", "Disables HNSW heap-page windows after consecutive windows without same-page candidates",
							"A value of 0 keeps heap-page windows enabled for the full scan.",
							&hnsw_page_disable_after_no_merge,
							2, 0, 10000, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.index_page_access", "Sets index-page-aware access mode for HNSW graph traversal",
							 "prefetch preserves HNSW traversal order and prefetches neighbor element index pages.",
							 &hnsw_index_page_access,
							 HNSW_INDEX_PAGE_ACCESS_OFF, hnsw_index_page_access_options, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.build_page_order", "Sets physical page order when flushing an in-memory HNSW graph",
							 "insertion preserves pgvector's default write order; bfs writes graph-neighbor BFS order from the entry point.",
							 &hnsw_build_page_order,
							 HNSW_BUILD_PAGE_ORDER_INSERTION, hnsw_build_page_order_options, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.build_seed", "Sets an optional deterministic seed and tie order for HNSW index builds",
							"-1 preserves stock pgvector behavior. Nonnegative seeds are experimental reproducibility controls, not a graph quality guarantee; physical graph determinism requires a serial full-memory build.",
							&hnsw_build_seed,
							-1, -1, PG_INT32_MAX, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.require_full_memory_build", "Rejects an HNSW build instead of switching to on-disk insertion",
							 "Use for physical-layout experiments that require every graph element to participate in the selected layout.",
							 &hnsw_require_full_memory_build,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomStringVariable("hnsw.clone_source", "Clones an existing same-heap HNSW logical graph",
							   "An empty value performs a normal build. Clone mode requires a non-concurrent full-memory BFS build and never scans the heap or constructs graph edges.",
							   &hnsw_clone_source,
							   "", PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomStringVariable("hnsw.preferred_index", "Restricts HNSW planning to one named index",
							   "An empty value preserves normal planning. This session-local experimental control makes other HNSW index paths infinitely costly so same-heap physical-layout experiments can select a proven source or clone index.",
							   &hnsw_preferred_index,
							   "", PGC_USERSET, 0, HnswPreferredIndexCheck,
							   HnswPreferredIndexAssign, NULL);

	DefineCustomEnumVariable("hnsw.filter_strategy", "Sets predicate-aware HNSW traversal strategy",
							 "off preserves pgvector behavior; safe_guided is validation-only and preserves stock graph traversal; traversal_guided performs planner-proven pre-distance filtering with bounded bridge expansion and fresh-stock fallback when iterative_scan is off, and otherwise bypasses to stock; acorn1 and guided_collect are experimental heuristic modes.",
							 &hnsw_filter_strategy,
							 HNSW_FILTER_STRATEGY_OFF, hnsw_filter_strategy_options, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.guided_collect_target", "Sets minimum guided candidates to collect before guided_collect can stop",
							"Only used when hnsw.filter_strategy = guided_collect. A smaller value is faster; a larger value improves filtered recall.",
							&hnsw_guided_collect_target,
							 100, 1, 1000000, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.traversal_guided_target", "Sets the minimum matching candidate batch for traversal_guided",
							 "A guided phase that cannot produce this many candidates before uncertainty is discarded and rerun through a fresh stock traversal.",
							 &hnsw_traversal_guided_target,
							 40, 1, 1000000, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.traversal_guided_max_bridge_hops", "Sets the maximum consecutive predicate-miss bridge hops for traversal_guided",
							 "Miss nodes may be expanded without vector distance only up to this level-0 hop bound.",
							 &hnsw_traversal_guided_max_bridge_hops,
							 2, 0, 64, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.traversal_guided_max_bridge_work", "Sets the maximum predicate-miss bridge work for traversal_guided",
							 "The bound counts expanded miss nodes and their newly discovered level-0 edges before a fresh-stock fallback.",
							 &hnsw_traversal_guided_max_bridge_work,
							 10000, 1, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomRealVariable("hnsw.traversal_guided_min_skip_rate", "Sets the minimum conservative skip-rate estimate for traversal_guided admission",
							 "Requests below this estimated benefit threshold bypass directly to stock traversal without membership checks.",
							 &hnsw_traversal_guided_min_skip_rate,
							 0.20, 0, 1, PGC_USERSET, 0, NULL, NULL, NULL);

	/* Same range as hash_mem_multiplier */
	DefineCustomRealVariable("hnsw.scan_mem_multiplier", "Sets the multiple of work_mem to use for iterative scans",
							 NULL, &hnsw_scan_mem_multiplier,
							 1, 1, 1000, PGC_USERSET, 0, NULL, NULL, NULL);

	MarkGUCPrefixReserved("hnsw");
}

/*
 * Get the name of index build phase
 */
static char *
hnswbuildphasename(int64 phasenum)
{
	switch (phasenum)
	{
		case PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE:
			return "initializing";
		case PROGRESS_HNSW_PHASE_LOAD:
			return "loading tuples";
		default:
			return NULL;
	}
}

/*
 * Estimate the cost of an index scan
 */
static void
hnswcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
				 Cost *indexStartupCost, Cost *indexTotalCost,
				 Selectivity *indexSelectivity, double *indexCorrelation,
				 double *indexPages)
{
	GenericCosts costs;
	int			m;
	double		ratio;
	double		startupPages;
	double		spc_seq_page_cost;
	Relation	index;
	Oid			preferredIndexOid = hnsw_preferred_index_oid;

	if (hnsw_preferred_index[0] != '\0')
	{
		Oid			pathHeapOid = IndexGetRelation(path->indexinfo->indexoid, false);
		Relation	preferredIndex;

		if (!OidIsValid(preferredIndexOid))
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("hnsw.preferred_index no longer resolves to an index")));
		preferredIndex = relation_open(preferredIndexOid, AccessShareLock);
		HnswValidatePreferredIndex(preferredIndex, pathHeapOid);
		relation_close(preferredIndex, AccessShareLock);
	}
	if (OidIsValid(preferredIndexOid) &&
		path->indexinfo->indexoid != preferredIndexOid)
	{
		*indexStartupCost = get_float8_infinity();
		*indexTotalCost = get_float8_infinity();
		*indexSelectivity = 0;
		*indexCorrelation = 0;
		*indexPages = 0;
		return;
	}

	/* Never use index without order */
	if (path->indexorderbys == NIL)
	{
		*indexStartupCost = get_float8_infinity();
		*indexTotalCost = get_float8_infinity();
		*indexSelectivity = 0;
		*indexCorrelation = 0;
		*indexPages = 0;
#if PG_VERSION_NUM >= 180000
		/* See "On disable_cost" thread on pgsql-hackers */
		path->path.disabled_nodes = 2;
#endif
		return;
	}

	MemSet(&costs, 0, sizeof(costs));

	genericcostestimate(root, path, loop_count, &costs);

	index = index_open(path->indexinfo->indexoid, NoLock);
	HnswGetMetaPageInfo(index, &m, NULL);
	index_close(index, NoLock);

	/*
	 * HNSW cost estimation follows a formula that accounts for the total
	 * number of tuples indexed combined with the parameters that most
	 * influence the duration of the index scan, namely: m - the number of
	 * tuples that are scanned in each step of the HNSW graph traversal
	 * ef_search - which influences the total number of steps taken at layer 0
	 *
	 * The source of the vector data can impact how many steps it takes to
	 * converge on the set of vectors to return to the executor. Currently, we
	 * use a hardcoded scaling factor (HNSWScanScalingFactor) to help
	 * influence that, but this could later become a configurable parameter
	 * based on the cost estimations.
	 *
	 * The tuple estimator formula is below:
	 *
	 * numIndexTuples = entryLevel * m + layer0TuplesMax * layer0Selectivity
	 *
	 * "entryLevel * m" represents the floor of tuples we need to scan to get
	 * to layer 0 (L0).
	 *
	 * "layer0TuplesMax" is the estimated total number of tuples we'd scan at
	 * L0 if we weren't discarding already visited tuples as part of the scan.
	 *
	 * "layer0Selectivity" estimates the percentage of tuples that are scanned
	 * at L0, accounting for previously visited tuples, multiplied by the
	 * "scalingFactor" (currently hardcoded).
	 */
	if (path->indexinfo->tuples > 0)
	{
		double		scalingFactor = 0.55;
		int			entryLevel = (int) (log(path->indexinfo->tuples) * HnswGetMl(m));
		int			layer0TuplesMax = HnswGetLayerM(m, 0) * hnsw_ef_search;
		double		layer0Selectivity = scalingFactor * log(path->indexinfo->tuples) / (log(m) * (1 + log(hnsw_ef_search)));

		ratio = (entryLevel * m + layer0TuplesMax * layer0Selectivity) / path->indexinfo->tuples;

		if (ratio > 1)
			ratio = 1;
	}
	else
		ratio = 1;

	get_tablespace_page_costs(path->indexinfo->reltablespace, NULL, &spc_seq_page_cost);

	/* Startup cost is cost before returning the first row */
	costs.indexStartupCost = costs.indexTotalCost * ratio;

	/* Adjust cost if needed since TOAST not included in seq scan cost */
	startupPages = costs.numIndexPages * ratio;
	if (startupPages > path->indexinfo->rel->pages && ratio < 0.5)
	{
		/* Change all page cost from random to sequential */
		costs.indexStartupCost -= startupPages * (costs.spc_random_page_cost - spc_seq_page_cost);

		/* Remove cost of extra pages */
		costs.indexStartupCost -= (startupPages - path->indexinfo->rel->pages) * spc_seq_page_cost;
	}

	*indexStartupCost = costs.indexStartupCost;
	*indexTotalCost = costs.indexTotalCost;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
	*indexPages = costs.numIndexPages;
}

/*
 * Parse and validate the reloptions
 */
static bytea *
hnswoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"m", RELOPT_TYPE_INT, offsetof(HnswOptions, m)},
		{"ef_construction", RELOPT_TYPE_INT, offsetof(HnswOptions, efConstruction)},
	};

	return (bytea *) build_reloptions(reloptions, validate,
									  hnsw_relopt_kind,
									  sizeof(HnswOptions),
									  tab, lengthof(tab));
}

/*
 * Validate catalog entries for the specified operator class
 */
static bool
hnswvalidate(Oid opclassoid)
{
	return true;
}

/*
 * Define index handler
 *
 * See https://www.postgresql.org/docs/current/index-api.html
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnswhandler);
Datum
hnswhandler(PG_FUNCTION_ARGS)
{
#if PG_VERSION_NUM >= 190000
	static const IndexAmRoutine amroutine = {
		.type = T_IndexAmRoutine,
		.amstrategies = 0,
		.amsupport = 3,
		.amoptsprocnum = 0,
		.amcanorder = false,
		.amcanorderbyop = true,
		.amcanhash = false,
		.amconsistentequality = false,
		.amconsistentordering = false,
		.amcanbackward = false,
		.amcanunique = false,
		.amcanmulticol = false,
		.amoptionalkey = true,
		.amsearcharray = false,
		.amsearchnulls = false,
		.amstorage = false,
		.amclusterable = false,
		.ampredlocks = false,
		.amcanparallel = false,
		.amcanbuildparallel = true,
		.amcaninclude = false,
		.amusemaintenanceworkmem = false,
		.amsummarizing = false,
		.amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL,
		.amkeytype = InvalidOid,

		.ambuild = hnswbuild,
		.ambuildempty = hnswbuildempty,
		.aminsert = hnswinsert,
		.aminsertcleanup = NULL,
		.ambulkdelete = hnswbulkdelete,
		.amvacuumcleanup = hnswvacuumcleanup,
		.amcanreturn = NULL,
		.amcostestimate = hnswcostestimate,
		.amgettreeheight = NULL,
		.amoptions = hnswoptions,
		.amproperty = NULL,
		.ambuildphasename = hnswbuildphasename,
		.amvalidate = hnswvalidate,
		.amadjustmembers = NULL,
		.ambeginscan = hnswbeginscan,
		.amrescan = hnswrescan,
		.amgettuple = hnswgettuple,
		.amgetbitmap = NULL,
		.amendscan = hnswendscan,
		.ammarkpos = NULL,
		.amrestrpos = NULL,
		.amestimateparallelscan = NULL,
		.aminitparallelscan = NULL,
		.amparallelrescan = NULL,
		.amtranslatestrategy = NULL,
		.amtranslatecmptype = NULL,
	};

	PG_RETURN_POINTER(&amroutine);
#else
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = 3;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
#if PG_VERSION_NUM >= 180000
	amroutine->amcanhash = false;
	amroutine->amconsistentequality = false;
	amroutine->amconsistentordering = false;
#endif
	amroutine->amcanbackward = false;	/* can change direction mid-scan */
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = false;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
#if PG_VERSION_NUM >= 170000
	amroutine->amcanbuildparallel = true;
#endif
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false; /* not used during VACUUM */
#if PG_VERSION_NUM >= 160000
	amroutine->amsummarizing = false;
#endif
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;

	/* Interface functions */
	amroutine->ambuild = hnswbuild;
	amroutine->ambuildempty = hnswbuildempty;
	amroutine->aminsert = hnswinsert;
#if PG_VERSION_NUM >= 170000
	amroutine->aminsertcleanup = NULL;
#endif
	amroutine->ambulkdelete = hnswbulkdelete;
	amroutine->amvacuumcleanup = hnswvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = hnswcostestimate;
#if PG_VERSION_NUM >= 180000
	amroutine->amgettreeheight = NULL;
#endif
	amroutine->amoptions = hnswoptions;
	amroutine->amproperty = NULL;	/* TODO AMPROP_DISTANCE_ORDERABLE */
	amroutine->ambuildphasename = hnswbuildphasename;
	amroutine->amvalidate = hnswvalidate;
#if PG_VERSION_NUM >= 140000
	amroutine->amadjustmembers = NULL;
#endif
	amroutine->ambeginscan = hnswbeginscan;
	amroutine->amrescan = hnswrescan;
	amroutine->amgettuple = hnswgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = hnswendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;

	/* Interface functions to support parallel index scans */
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

#if PG_VERSION_NUM >= 180000
	amroutine->amtranslatestrategy = NULL;
	amroutine->amtranslatecmptype = NULL;
#endif

	PG_RETURN_POINTER(amroutine);
#endif
}

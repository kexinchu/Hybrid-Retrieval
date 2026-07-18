#include "postgres.h"

#include "executor/executor.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "nodes/execnodes.h"
#include "portability/instr_time.h"
#include "utils/builtins.h"
#include "utils/rel.h"

PG_MODULE_MAGIC;

typedef struct QualProfileEntry
{
	ExprStateEvalFunc original_evalfunc;
	ExprState  *expr;
	int			node_tag;
	char		relname[NAMEDATALEN];
	double		total_ms;
	uint64		calls;
	uint64		true_count;
	uint64		false_count;
	uint64		null_count;
	struct QualProfileEntry *next;
} QualProfileEntry;

static ExecutorStart_hook_type previous_ExecutorStart = NULL;
static ExecutorRun_hook_type previous_ExecutorRun = NULL;
static QualProfileEntry *profile_entries = NULL;
static uint64 profile_query_count = 0;
static double profile_query_qual_ms = 0.0;
static uint64 profile_query_qual_calls = 0;
static uint64 profile_seen_plan_nodes = 0;
static uint64 profile_seen_qual_nodes = 0;

void		_PG_init(void);
void		_PG_fini(void);

PG_FUNCTION_INFO_V1(hybrid_qual_profile_reset);
PG_FUNCTION_INFO_V1(hybrid_qual_profile_last);

static QualProfileEntry *
FindEntry(ExprState *expr)
{
	QualProfileEntry *entry;

	for (entry = profile_entries; entry != NULL; entry = entry->next)
	{
		if (entry->expr == expr)
			return entry;
	}

	return NULL;
}

static Datum
ProfiledQualEval(ExprState *expression, ExprContext *econtext, bool *isNull)
{
	QualProfileEntry *entry = FindEntry(expression);
	instr_time	start;
	instr_time	elapsed;
	Datum		result;

	if (entry == NULL || entry->original_evalfunc == NULL)
		return (Datum) 0;

	INSTR_TIME_SET_CURRENT(start);
	result = entry->original_evalfunc(expression, econtext, isNull);
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);

	if (expression->evalfunc != ProfiledQualEval &&
		expression->evalfunc != entry->original_evalfunc)
		entry->original_evalfunc = expression->evalfunc;
	expression->evalfunc = ProfiledQualEval;

	entry->total_ms += INSTR_TIME_GET_MILLISEC(elapsed);
	entry->calls++;

	if (isNull != NULL && *isNull)
		entry->null_count++;
	else if (DatumGetBool(result))
		entry->true_count++;
	else
		entry->false_count++;

	return result;
}

static void
RegisterQual(PlanState *planstate)
{
	QualProfileEntry *entry;
	const char *relname = "";

	if (planstate == NULL || planstate->qual == NULL)
		return;

	profile_seen_qual_nodes++;

	if (FindEntry(planstate->qual) != NULL)
		return;

	entry = (QualProfileEntry *) MemoryContextAllocZero(TopMemoryContext, sizeof(QualProfileEntry));
	entry->expr = planstate->qual;
	entry->original_evalfunc = planstate->qual->evalfunc;
	entry->node_tag = (int) nodeTag(planstate);

	switch (nodeTag(planstate))
	{
		case T_SeqScanState:
		case T_SampleScanState:
		case T_IndexScanState:
		case T_IndexOnlyScanState:
		case T_BitmapHeapScanState:
		case T_TidScanState:
		case T_TidRangeScanState:
		case T_SubqueryScanState:
		case T_FunctionScanState:
		case T_TableFuncScanState:
		case T_ValuesScanState:
		case T_CteScanState:
		case T_NamedTuplestoreScanState:
		case T_WorkTableScanState:
		{
			ScanState  *scanstate = (ScanState *) planstate;

			if (scanstate->ss_currentRelation != NULL)
				relname = RelationGetRelationName(scanstate->ss_currentRelation);
			break;
		}
		default:
			break;
	}

	strlcpy(entry->relname, relname, NAMEDATALEN);
	entry->next = profile_entries;
	profile_entries = entry;

	planstate->qual->evalfunc = ProfiledQualEval;
}

static void RegisterPlanTree(PlanState *planstate);

static void
RegisterSubPlans(List *subplans)
{
	ListCell   *lc;

	foreach(lc, subplans)
	{
		SubPlanState *subplan = (SubPlanState *) lfirst(lc);

		if (subplan != NULL)
			RegisterPlanTree(subplan->planstate);
	}
}

static void
RegisterPlanTree(PlanState *planstate)
{
	if (planstate == NULL)
		return;

	profile_seen_plan_nodes++;

	RegisterQual(planstate);
	RegisterPlanTree(outerPlanState(planstate));
	RegisterPlanTree(innerPlanState(planstate));
	RegisterSubPlans(planstate->initPlan);
	RegisterSubPlans(planstate->subPlan);
}

static void
HybridQualExecutorStart(QueryDesc *queryDesc, int eflags)
{
	if (previous_ExecutorStart)
		previous_ExecutorStart(queryDesc, eflags);
	else
		standard_ExecutorStart(queryDesc, eflags);

	if (queryDesc != NULL && queryDesc->planstate != NULL)
	{
		profile_query_count++;
		RegisterPlanTree(queryDesc->planstate);
	}
}

static void
HybridQualExecutorRun(QueryDesc *queryDesc, ScanDirection direction, uint64 count, bool execute_once)
{
	if (queryDesc != NULL && queryDesc->planstate != NULL)
		RegisterPlanTree(queryDesc->planstate);

	if (previous_ExecutorRun)
		previous_ExecutorRun(queryDesc, direction, count, execute_once);
	else
		standard_ExecutorRun(queryDesc, direction, count, execute_once);
}

void
_PG_init(void)
{
	previous_ExecutorStart = ExecutorStart_hook;
	previous_ExecutorRun = ExecutorRun_hook;
	ExecutorStart_hook = HybridQualExecutorStart;
	ExecutorRun_hook = HybridQualExecutorRun;
}

void
_PG_fini(void)
{
	ExecutorStart_hook = previous_ExecutorStart;
	ExecutorRun_hook = previous_ExecutorRun;
}

Datum
hybrid_qual_profile_reset(PG_FUNCTION_ARGS)
{
	QualProfileEntry *entry = profile_entries;

	while (entry != NULL)
	{
		QualProfileEntry *next = entry->next;

		pfree(entry);
		entry = next;
	}

	profile_entries = NULL;
	profile_query_count = 0;
	profile_query_qual_ms = 0.0;
	profile_query_qual_calls = 0;
	profile_seen_plan_nodes = 0;
	profile_seen_qual_nodes = 0;

	PG_RETURN_VOID();
}

Datum
hybrid_qual_profile_last(PG_FUNCTION_ARGS)
{
	StringInfoData output;
	QualProfileEntry *entry;
	bool		first = true;

	profile_query_qual_ms = 0.0;
	profile_query_qual_calls = 0;

	for (entry = profile_entries; entry != NULL; entry = entry->next)
	{
		profile_query_qual_ms += entry->total_ms;
		profile_query_qual_calls += entry->calls;
	}

	initStringInfo(&output);
	appendStringInfo(&output,
					 "{\"query_count\":" UINT64_FORMAT
					 ",\"seen_plan_nodes\":" UINT64_FORMAT
					 ",\"seen_qual_nodes\":" UINT64_FORMAT
					 ",\"qual_ms\":%.6f"
					 ",\"qual_calls\":" UINT64_FORMAT
					 ",\"entries\":[",
					 profile_query_count,
					 profile_seen_plan_nodes,
					 profile_seen_qual_nodes,
					 profile_query_qual_ms,
					 profile_query_qual_calls);

	for (entry = profile_entries; entry != NULL; entry = entry->next)
	{
		if (!first)
			appendStringInfoChar(&output, ',');
		first = false;

		appendStringInfo(&output,
						 "{\"node_tag\":%d,"
						 "\"relname\":\"%s\","
						 "\"qual_ms\":%.6f,"
						 "\"calls\":" UINT64_FORMAT ","
						 "\"true\":" UINT64_FORMAT ","
						 "\"false\":" UINT64_FORMAT ","
						 "\"null\":" UINT64_FORMAT "}",
						 entry->node_tag,
						 entry->relname,
						 entry->total_ms,
						 entry->calls,
						 entry->true_count,
						 entry->false_count,
						 entry->null_count);
	}

	appendStringInfoString(&output, "]}");

	PG_RETURN_TEXT_P(cstring_to_text(output.data));
}

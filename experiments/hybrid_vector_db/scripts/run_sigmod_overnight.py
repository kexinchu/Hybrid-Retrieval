from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "hybrid_vector_db"
LOGS = RESULTS / "logs"


@dataclass
class Job:
    name: str
    cmd: list[str]
    log: Path
    outputs: list[Path] = field(default_factory=list)
    depends_on: list[Path] = field(default_factory=list)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_logged(job: Job, manifest: dict[str, object]) -> int:
    missing = [str(path) for path in job.depends_on if not path.exists()]
    if missing:
        manifest["jobs"][job.name] = {
            "status": "skipped",
            "missing": missing,
            "cmd": job.cmd,
            "log": str(job.log),
            "outputs": [str(path) for path in job.outputs],
        }
        print(f"SKIP {job.name}: missing dependencies {missing}", flush=True)
        return 0

    job.log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== RUN {job.name} ===", flush=True)
    print(shlex.join(job.cmd), flush=True)
    start = time.perf_counter()
    with job.log.open("w") as log:
        log.write("$ " + shlex.join(job.cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            job.cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
    elapsed = time.perf_counter() - start
    manifest["jobs"][job.name] = {
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_s": elapsed,
        "cmd": job.cmd,
        "log": str(job.log),
        "outputs": [str(path) for path in job.outputs],
    }
    print(f"=== DONE {job.name}: rc={rc} elapsed={elapsed:.1f}s ===", flush=True)
    return rc


def pg_smoke_cmd(py: str) -> list[str]:
    return [
        py,
        "-c",
        (
            "import sys, psycopg; "
            "sys.path.insert(0, 'experiments/hybrid_vector_db/scripts'); "
            "from common_pg import pg_config_from_env; "
            "cfg=pg_config_from_env(); "
            "conn=psycopg.connect(cfg.conninfo, autocommit=True); "
            "cur=conn.cursor(); "
            "cur.execute('select count(*) from amazon_grocery_reviews_10m_pgvector'); "
            "print('pg_rows', cur.fetchone()[0]); "
            "cur.execute(\"select to_regclass('amazon_grocery_reviews_10m_pgvector_samegraph_bfs')\"); "
            "print('bfs_table', cur.fetchone()[0]); "
            "cur.execute(\"select to_regclass('amazon_grocery_reviews_10m_pgvector_samegraph_bfs_hnsw')\"); "
            "print('bfs_index', cur.fetchone()[0]); "
            "cur.execute('select vector_hnsw_metadata_cache_profile()'); "
            "print('cache_profile_ok')"
        ),
    ]


def build_jobs(args: argparse.Namespace) -> list[Job]:
    py = sys.executable
    tag = args.tag
    logs = LOGS / f"sigmod_{tag}"
    q = args.queries
    r = args.repeats

    d123 = RESULTS / f"sigmod_d123_selectivity_q{q}r{r}_warmall_{tag}.csv"
    d123_table = d123.with_name(d123.stem + "_table.csv")
    d4 = RESULTS / f"sigmod_d4_calibration_q{q}r{r}_warmall_{tag}.csv"
    d4_cal = d4.with_name(d4.stem + "_calibration.csv")
    d4_table = d4.with_name(d4.stem + "_merged_table.csv")
    c4_memory = RESULTS / f"sigmod_c4_guidance_memory_q{args.c4_queries}_{tag}.csv"
    c4_memory_summary = c4_memory.with_name(c4_memory.stem + "_summary.json")
    c4_route = RESULTS / f"sigmod_c4_route_audit_q{args.c4_route_queries}r{args.c4_route_repeats}_{tag}.csv"
    c4_route_summary = c4_route.with_name(c4_route.stem + "_summary.json")
    summary_prefix = RESULTS / f"sigmod_summary_{tag}"

    jobs = [
        Job("pg_smoke", pg_smoke_cmd(py), logs / "pg_smoke.log"),
        Job(
            "d123_selectivity",
            [
                py,
                "experiments/hybrid_vector_db/scripts/pgvector_design1_design2_design3_selectivity_benchmark.py",
                "--out",
                str(d123),
                "--queries",
                str(q),
                "--repeats",
                str(r),
                "--warmup-all-queries",
                "--progress-queries",
                str(args.progress_queries),
                "--statement-timeout-ms",
                str(args.statement_timeout_ms),
            ],
            logs / "d123_selectivity.log",
            [d123, d123_table, d123.with_name(d123.stem + "_profile_summary.csv")],
        ),
        Job(
            "d4_calibration",
            [
                py,
                "experiments/hybrid_vector_db/scripts/pgvector_design4_calibration_selectivity_benchmark.py",
                "--out",
                str(d4),
                "--baseline-table",
                str(d123_table),
                "--queries",
                str(q),
                "--repeats",
                str(r),
                "--calibration-queries",
                str(args.calibration_queries),
                "--warmup-all-queries",
                "--progress-queries",
                str(args.progress_queries),
                "--statement-timeout-ms",
                str(args.statement_timeout_ms),
            ],
            logs / "d4_calibration.log",
            [d4, d4_cal, d4_table],
            [d123_table],
        ),
        Job(
            "c4_guidance_memory",
            [
                py,
                "experiments/hybrid_vector_db/scripts/pgvector_c4_guidance_memory_benchmark.py",
                "--out",
                str(c4_memory),
                "--queries",
                str(args.c4_queries),
                "--stream",
                "--progress-queries",
                str(args.c4_progress_queries),
                "--statement-timeout-ms",
                str(args.statement_timeout_ms),
            ],
            logs / "c4_guidance_memory.log",
            [c4_memory, c4_memory_summary, c4_memory.with_name(c4_memory.stem + "_fragments.csv")],
        ),
        Job(
            "c4_route_audit",
            [
                py,
                "experiments/hybrid_vector_db/scripts/pgvector_c4_route_audit_design3.py",
                "--out",
                str(c4_route),
                "--queries",
                str(args.c4_route_queries),
                "--repeats",
                str(args.c4_route_repeats),
                "--progress-every",
                str(args.c4_progress_queries),
                "--statement-timeout-ms",
                str(args.statement_timeout_ms),
            ],
            logs / "c4_route_audit.log",
            [c4_route, c4_route_summary, c4_route.with_name(c4_route.stem + "_predicates.csv")],
        ),
        Job(
            "sigmod_summary",
            [
                py,
                "experiments/hybrid_vector_db/scripts/sigmod_result_summary.py",
                "--d123-raw",
                str(d123),
                "--d4-raw",
                str(d4),
                "--c4-memory-summary",
                str(c4_memory_summary),
                "--c4-route-summary",
                str(c4_route_summary),
                "--out-prefix",
                str(summary_prefix),
            ],
            logs / "sigmod_summary.log",
            [
                summary_prefix.with_name(summary_prefix.name + "_d123_summary.csv"),
                summary_prefix.with_name(summary_prefix.name + "_d4_summary.csv"),
                summary_prefix.with_name(summary_prefix.name + "_manifest.json"),
            ],
        ),
    ]

    selected = set(args.jobs.split(",")) if args.jobs else None
    return [job for job in jobs if selected is None or job.name in selected]


def apply_profile(args: argparse.Namespace) -> None:
    if args.profile == "smoke":
        args.queries = args.queries or 2
        args.repeats = args.repeats or 1
        args.calibration_queries = args.calibration_queries or 2
        args.c4_queries = args.c4_queries or 5
        args.c4_route_queries = args.c4_route_queries or 5
        args.c4_route_repeats = args.c4_route_repeats or 1
        args.progress_queries = args.progress_queries or 1
        args.c4_progress_queries = args.c4_progress_queries or 1
    else:
        args.queries = args.queries or 100
        args.repeats = args.repeats or 10
        args.calibration_queries = args.calibration_queries or 12
        args.c4_queries = args.c4_queries or 400
        args.c4_route_queries = args.c4_route_queries or 100
        args.c4_route_repeats = args.c4_route_repeats or 10
        args.progress_queries = args.progress_queries or 10
        args.c4_progress_queries = args.c4_progress_queries or 25


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SIGMOD D1-D4 overnight experiment chain.")
    parser.add_argument("--profile", choices=["smoke", "overnight"], default="overnight")
    parser.add_argument("--tag", default=now_tag())
    parser.add_argument("--jobs", help="Comma-separated subset of: pg_smoke,d123_selectivity,d4_calibration,c4_guidance_memory,c4_route_audit,sigmod_summary")
    parser.add_argument("--queries", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--calibration-queries", type=int)
    parser.add_argument("--c4-queries", type=int)
    parser.add_argument("--c4-route-queries", type=int)
    parser.add_argument("--c4-route-repeats", type=int)
    parser.add_argument("--progress-queries", type=int)
    parser.add_argument("--c4-progress-queries", type=int)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    apply_profile(args)

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS / f"sigmod_overnight_manifest_{args.tag}.json"
    manifest: dict[str, object] = {
        "tag": args.tag,
        "profile": args.profile,
        "started_at": datetime.now().isoformat(),
        "root": str(ROOT),
        "jobs": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    failed = False
    for job in build_jobs(args):
        rc = run_logged(job, manifest)
        manifest["updated_at"] = datetime.now().isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if rc != 0:
            failed = True
            if not args.keep_going:
                break

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["status"] = "failed" if failed else "ok"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

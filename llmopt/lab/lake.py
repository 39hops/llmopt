"""Parquet lake over the lab's jsonl/file exhaust — QUERY layer, not a write format.

Doctrine (2026-08-08): jsonl stays the write format (append-only, RESULTS-cited);
the lake is regenerable EXHAUST under data/lake/ (gitignored, logs-doctrine class
2026-08-06: regenerate-don't-download). Never point a booked verdict at a lake
file — the evidence record is the jsonl/RESULTS line, frozen in place.

Tables (Parquet, snappy):
  runs        — jobs/<id>.{cmd,rc,pid} quads + jobs/<id>.log mtime. rc_raw keeps
                the literal string ("killed" stays "killed" — the checkpoint
                selection-effect, bit three times, means killed rows must stay
                visible); rc is the int cast or null. source_grade=exploration.
  results     — docs/results-index.jsonl, PK id. The `line` column is a BYTE-
                FRAGILE pointer into RESULTS.md — gen_results_index.py
                regeneration invalidates it; id is the key, always join on id.
                source_grade=ledger.
  result_edges— exploded {src_id, edge_type (links|amends|superseded_by), dst_id}.
  models      — data/catalog/models.jsonl if present; absent => empty table
                (the catalog is built by a concurrent agent — never fail on it).
  gates       — schema + append_gate() writer. device, n_seeds, weights_sha are
                REQUIRED NON-NULL: cross-device gate comparison is doctrine-
                forbidden and sigma never transports (RESULTS, precision-doctrine
                scope fences), so every aggregation must be groupable by device;
                a device-less gate row is unaggregatable poison and is refused
                (ValueError) at write time, not discovered at query time.

Scoring doctrine reminder for consumers: never score weights by weight distance
(RESULTS 6163 joint-perm closure) — the lake carries weights_sha as an IDENTITY
key only, never a similarity axis. Function-space metrics or nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_LAKE_DIR = Path("data/lake")

EDGE_TYPES = ("links", "amends", "superseded_by")

GATES_SCHEMA = pa.schema(
    [
        pa.field("device", pa.string(), nullable=False),
        pa.field("n_seeds", pa.int64(), nullable=False),
        pa.field("weights_sha", pa.string(), nullable=False),
        pa.field("paired_with", pa.string()),
        pa.field("gate_dict", pa.string()),  # json string
        pa.field("total", pa.int64()),
        pa.field("source_grade", pa.string()),
    ]
)

_GATE_REQUIRED = ("device", "n_seeds", "weights_sha")


def _write(table: pa.Table, lake_dir: Path, name: str) -> Path:
    lake_dir = Path(lake_dir)
    lake_dir.mkdir(parents=True, exist_ok=True)
    out = lake_dir / f"{name}.parquet"
    pq.write_table(table, out, compression="snappy")
    return out


def build_runs(jobs_dir: Path = Path("jobs"), lake_dir: Path = DEFAULT_LAKE_DIR) -> Path:
    """jobs/<id>.{cmd,rc,pid} + <id>.log mtime -> runs.parquet.

    rc_raw is the literal file content (stripped); "killed" survives as a
    string so killed runs never vanish from aggregates (stream-your-rows /
    checkpoint selection-effect doctrine). rc is int(rc_raw) or null.
    """
    jobs_dir = Path(jobs_dir)
    rows = []
    if jobs_dir.is_dir():
        for cmd_path in sorted(jobs_dir.glob("*.cmd")):
            run_id = cmd_path.stem
            rc_raw = None
            rc_path = jobs_dir / f"{run_id}.rc"
            if rc_path.exists():
                rc_raw = rc_path.read_text().strip()
            try:
                rc = int(rc_raw) if rc_raw is not None else None
            except ValueError:
                rc = None  # "killed" stays visible in rc_raw
            log_path = jobs_dir / f"{run_id}.log"
            mtime = log_path.stat().st_mtime if log_path.exists() else None
            rows.append(
                {
                    "run_id": run_id,
                    "cmd": cmd_path.read_text().strip(),
                    "rc_raw": rc_raw,
                    "rc": rc,
                    "log_path": str(log_path) if log_path.exists() else None,
                    "mtime": mtime,
                    "source_grade": "exploration",
                }
            )
    schema = pa.schema(
        [
            ("run_id", pa.string()),
            ("cmd", pa.string()),
            ("rc_raw", pa.string()),
            ("rc", pa.int64()),
            ("log_path", pa.string()),
            ("mtime", pa.float64()),
            ("source_grade", pa.string()),
        ]
    )
    return _write(pa.Table.from_pylist(rows, schema=schema), lake_dir, "runs")


def build_results(
    index_path: Path = Path("docs/results-index.jsonl"),
    lake_dir: Path = DEFAULT_LAKE_DIR,
) -> tuple[Path, Path]:
    """docs/results-index.jsonl -> results.parquet + result_edges.parquet.

    PK is id. `line` is documented byte-fragile (pointer into RESULTS.md,
    invalidated by gen_results_index.py regeneration) — join on id, never line.
    Edges (links|amends|superseded_by) explode into result_edges.
    """
    index_path = Path(index_path)
    rows, edges = [], []
    if index_path.exists():
        for raw in index_path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rows.append(
                {
                    "id": rec["id"],
                    "date": rec.get("date"),
                    "line": rec.get("line"),
                    "title": rec.get("title"),
                    "type": rec.get("type"),
                    "verdict": rec.get("verdict"),
                    "threads": rec.get("threads") or [],
                    "source_grade": "ledger",
                }
            )
            for edge_type in EDGE_TYPES:
                for dst in rec.get(edge_type) or []:
                    edges.append({"src_id": rec["id"], "edge_type": edge_type, "dst_id": dst})
    results_schema = pa.schema(
        [
            ("id", pa.string()),
            ("date", pa.string()),
            ("line", pa.int64()),
            ("title", pa.string()),
            ("type", pa.string()),
            ("verdict", pa.string()),
            ("threads", pa.list_(pa.string())),
            ("source_grade", pa.string()),
        ]
    )
    edges_schema = pa.schema(
        [("src_id", pa.string()), ("edge_type", pa.string()), ("dst_id", pa.string())]
    )
    p1 = _write(pa.Table.from_pylist(rows, schema=results_schema), lake_dir, "results")
    p2 = _write(pa.Table.from_pylist(edges, schema=edges_schema), lake_dir, "result_edges")
    return p1, p2


def build_models(
    catalog_path: Path = Path("data/catalog/models.jsonl"),
    lake_dir: Path = DEFAULT_LAKE_DIR,
) -> Path:
    """data/catalog/models.jsonl -> models.parquet; absent file => EMPTY table
    (the catalog is authored by a concurrent agent — absence is not an error)."""
    catalog_path = Path(catalog_path)
    rows = []
    if catalog_path.exists():
        for raw in catalog_path.read_text().splitlines():
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        table = pa.Table.from_pylist([], schema=pa.schema([("model_id", pa.string())]))
    return _write(table, lake_dir, "models")


def build_gates(lake_dir: Path = DEFAULT_LAKE_DIR) -> Path:
    """Materialize an empty gates.parquet with the pinned schema (idempotent;
    never clobbers an existing gates table)."""
    lake_dir = Path(lake_dir)
    out = lake_dir / "gates.parquet"
    if not out.exists():
        _write(pa.Table.from_pylist([], schema=GATES_SCHEMA), lake_dir, "gates")
    return out


def append_gate(row: dict, lake_dir: Path = DEFAULT_LAKE_DIR) -> Path:
    """Append one gate row. REFUSES (ValueError) rows missing/null in any of
    device, n_seeds, weights_sha — cross-device comparison is doctrine-forbidden
    and sigma never transports, so an un-deviced gate row can never be
    aggregated safely; fail at write time. gate_dict must be a json string."""
    for key in _GATE_REQUIRED:
        if row.get(key) is None:
            raise ValueError(
                f"gate row missing required non-null column {key!r} "
                "(device/n_seeds/weights_sha are doctrine-required: every gate "
                "aggregation must group by device; sigma never transports)"
            )
    clean = {
        "device": str(row["device"]),
        "n_seeds": int(row["n_seeds"]),
        "weights_sha": str(row["weights_sha"]),
        "paired_with": row.get("paired_with"),
        "gate_dict": row.get("gate_dict"),
        "total": row.get("total"),
        "source_grade": row.get("source_grade", "exploration"),
    }
    lake_dir = Path(lake_dir)
    out = build_gates(lake_dir)
    existing = pq.read_table(out)
    new = pa.Table.from_pylist([clean], schema=GATES_SCHEMA)
    pq.write_table(pa.concat_tables([existing, new]), out, compression="snappy")
    return out


def query(sql: str, lake_dir: Path = DEFAULT_LAKE_DIR):
    """Run duckdb SQL over the lake. Every *.parquet under lake_dir is exposed
    as a view named by its stem (runs, results, result_edges, models, gates).
    Returns a list of dict rows."""
    import duckdb

    lake_dir = Path(lake_dir)
    con = duckdb.connect()
    try:
        for p in sorted(lake_dir.glob("*.parquet")):
            # CREATE VIEW cannot be a prepared statement in duckdb; escape
            # the path as a SQL string literal instead of binding it.
            lit = str(p).replace("'", "''")
            con.execute(f"CREATE VIEW \"{p.stem}\" AS SELECT * FROM read_parquet('{lit}')")
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()

"""Read-only checkpoint inventory, dependency graph, conservative
classification, APFS physical-storage probe and family decision sheet
(rehabilitation phase 2E).

Nothing here writes inside checkpoints/. The tool walks the tree,
hashes every file, collects every reference the repository makes to
it, classifies each file, probes whether byte-identical copies share
physical extents, and writes machine-readable results plus a Markdown
sheet to an output directory (default logs/rehab-2e/<stamp>/, run
exhaust, untracked).

Classes (most protective wins):
  FROZEN     sha-locked in docs/receipts.lock.json, pinned in
             docs/EVIDENCE.md, cited by docs/RESULTS.md (full path,
             parent directory, basename, or a stem of eight or more
             characters), confirmed in checkpoints/MANIFEST.jsonl, a
             verified parent of a FROZEN merge output, a sidecar of a
             FROZEN file, or named like a killed-run receipt.
  ESSENTIAL  git-tracked, named by a full path literal in code, jobs,
             REPRODUCE.md, README.md, FINDINGS, BOARD, EVIDENCE prose,
             CODEMAP, handoffs or specs, a parent of an ESSENTIAL merge
             output, or a sidecar of an ESSENTIAL file.
  DUPLICATE  byte-identical to a FROZEN or ESSENTIAL file (the
             canonical copy is named), or to a lexicographically
             earlier uncited twin, and reached by nothing itself.
  CANDIDATE  reached by nothing the scanner knows, older than
             --age-days, not a sidecar, not under a dynamic prefix.
  UNKNOWN    everything else: a bare basename in code, a file under a
             dynamically constructed path prefix that names at least
             one literal character past "checkpoints/", a file named
             only by an uncited log, a symlink, a sidecar whose
             primary is missing, an unparsable JSON sidecar, a recent
             uncited file.

Generic loaders ("checkpoints/" + name, os.path.join("checkpoints",
x), f"checkpoints/{name}") reach every file by construction and
carry no information about any one of them; they are counted and
listed (summary.generic_loaders, graph.unresolved kind
generic-loader) but not applied per file. A log that names a
checkpoint path counts as a FROZEN reference only when that log is
itself sha-locked or cited by RESULTS; a mention in uncited log
exhaust leaves the file UNKNOWN. Cited paths absent from the tree are
listed (absent_cited) with the other trees they were found in.

Physical storage: for every duplicate group the first and middle
extents of each copy are read with fcntl F_LOG2PHYS_EXT; equal keys
mean the copies share blocks (APFS clones) and reclaim nothing when
one is removed; different keys mean distinct blocks. Files whose
content is unique cannot be checked for partial sharing this way and
their allocated size is reported as an estimate, never as verified.

Usage:
    .venv/bin/python scripts/ckpt_census.py [--out DIR] [--age-days 14]
        [--extra-tree NAME=PATH ...] [--remote-shas NAME=FILE ...]
        [--verify-baseline FILE]
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CKPT = "checkpoints"
CKPT_EXTS = (".pt", ".bin", ".npz", ".npy", ".safetensors", ".json",
             ".jsonl", ".sha", ".pth", ".ckpt", ".wsl", ".py")
BASENAME_EXTS = ("pt", "bin", "npz", "npy", "safetensors", "json", "jsonl",
                 "sha", "pth", "ckpt")
RECEIPT_GLOBS = ("*oomkilled*", "*interrupted*", "*poisoned*", "*_oom*")
STEM_MIN = 8
CLASSES = ("FROZEN", "ESSENTIAL", "DUPLICATE", "CANDIDATE", "UNKNOWN")
RANK = {c: i for i, c in enumerate(CLASSES)}

RESULTS_SOURCES = ("docs/RESULTS.md",)
ESSENTIAL_DOC_SOURCES = ("README.md", "REPRODUCE.md", "docs/FINDINGS.md",
                         "docs/BOARD.md", "docs/EVIDENCE.md",
                         "docs/CODEMAP.md", "docs/THEORY.md",
                         "docs/RIFF-LEDGER.md")
ESSENTIAL_DOC_GLOBS = ("docs/handoffs/*.md", "docs/superpowers/specs/*.md",
                       "docs/preregs/*.json")
CODE_DIRS = ("scratch", "scripts", "llmopt", "tests", "jobs")
CODE_EXTS = (".py", ".sh", ".cmd", ".ps1", ".toml", ".yaml", ".yml", ".txt")
LOG_TEXT_EXTS = (".log", ".md", ".txt", ".json", ".jsonl")
LOG_TEXT_MAX = 2 << 20

PATH_TOKEN = re.compile(r"checkpoints/[\w\-./{},*]+")
DYNAMIC = re.compile(
    r"checkpoints/[\w\-./]*(?:\{|%|\$|\"\s*\+|'\s*\+|\"\s*%|'\s*%|\.format|f\")")
DYNAMIC_PREFIX = re.compile(r"checkpoints/([\w\-./]*)")
# os.path.join("checkpoints", x), Path("checkpoints") / x, ROOT / "checkpoints" / x
JOIN_FORMS = re.compile(
    r"""join\(\s*["']checkpoints["']|Path\(\s*["']checkpoints["']\s*\)\s*/|/\s*["']checkpoints["']\s*/\s*[A-Za-z_]""")


# --------------------------------------------------------------- hashing

def sha256_file(p: Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def extent_key(p: Path):
    """(flags, contig, devoffset) of the first and middle logical
    extents, or None when the ioctl is unavailable. Two files with
    equal keys share physical blocks there (APFS clone)."""
    try:
        size = p.stat().st_size
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return None
    try:
        keys = []
        for off in (0, max(0, (size // 2) // 4096 * 4096)):
            # struct log2phys is copied packed (20 bytes): u32 flags,
            # off_t contigbytes (in: bytes queried; out: contiguous
            # bytes), off_t devoffset (in: logical offset; out: device
            # offset). The natural-alignment layout returns garbage.
            buf = struct.pack("=IqQ", 0, 1 << 20, off)
            try:
                out = fcntl.fcntl(fd, 65, buf)      # F_LOG2PHYS_EXT
            except OSError:
                return None
            flags, contig, devoff = struct.unpack("=IqQ", out)
            keys.append((off, devoff, contig))
        return tuple(keys)
    finally:
        os.close(fd)


# -------------------------------------------------------------- scanning

def git_tracked(repo: Path) -> set[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"],
                             capture_output=True, check=False).stdout
    except OSError:
        return set()
    return set(x for x in out.decode(errors="replace").split("\0") if x)


def walk_tree(repo: Path, tree: str = CKPT) -> dict[str, dict]:
    """rel (inside the tree) -> row. Symlinks are recorded, never
    followed or hashed."""
    rows: dict[str, dict] = {}
    root = repo / tree
    if not root.exists():
        return rows
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            p = Path(dp) / fn
            rel = str(p.relative_to(root))
            st = p.lstat()
            row = {"path": rel, "repo_path": f"{tree}/{rel}",
                   "bytes": st.st_size, "alloc": st.st_blocks * 512,
                   "mtime": st.st_mtime,
                   "birthtime": getattr(st, "st_birthtime", None),
                   "mode": oct(st.st_mode & 0o7777), "nlink": st.st_nlink,
                   "symlink": p.is_symlink(), "sha256": None}
            if row["symlink"]:
                try:
                    row["target"] = os.readlink(p)
                except OSError:
                    row["target"] = None
            elif p.is_file():
                try:
                    row["sha256"] = sha256_file(p)
                except OSError as e:
                    row["sha256"] = None
                    row["error"] = str(e)
            rows[rel] = row
    return rows


def _norm_token(tok: str) -> str:
    return re.sub(r"[.,:;)\]}'\"`/]+$", "", tok)


def scan_text_for_refs(text: str, source: str) -> list[dict]:
    """Every reference form in one text: full path tokens (globs and
    braces reduce to their parent directory), bare basenames with a
    checkpoint extension, and dynamic constructions."""
    refs = []
    for m in PATH_TOKEN.finditer(text):
        tok = _norm_token(m.group(0))
        if tok in ("checkpoints", ""):
            continue
        if any(c in tok for c in "{}*"):
            parent = str(Path(tok.split("{")[0].split("*")[0]).parent)
            if parent not in ("checkpoints", "."):
                refs.append({"kind": "dir", "target": parent,
                             "source": source})
            else:
                # brace or glob directly under checkpoints/: the DYNAMIC
                # pass below records the literal fragment, if any
                pass
            continue
        refs.append({"kind": "path", "target": tok, "source": source})
    for m in DYNAMIC.finditer(text):
        pm = DYNAMIC_PREFIX.match(m.group(0))
        frag = (pm.group(1) if pm else "").rstrip("/")
        if frag:
            refs.append({"kind": "dynamic-prefix",
                         "target": "checkpoints/" + frag, "source": source})
        else:
            refs.append({"kind": "generic-loader", "target": "checkpoints",
                         "source": source})
    if JOIN_FORMS.search(text):
        refs.append({"kind": "generic-loader", "target": "checkpoints",
                     "source": source})
    for b in set(re.findall(
            r"(?<![\w/])([\w][\w\-.]*\.(?:%s))\b" % "|".join(BASENAME_EXTS),
            text)):
        refs.append({"kind": "basename", "target": b, "source": source})
    return refs


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def log_citation_sets(repo: Path, lock: dict) -> tuple[set[str], set[str]]:
    """(locked log paths, RESULTS-cited log tokens) so a log mention
    of a checkpoint can be weighed by the log's own standing."""
    locked = {p for p, rec in lock.items()
              if p.startswith("logs/") and rec.get("sha256")}
    cited: set[str] = set()
    try:
        sys.path.insert(0, str(repo / "scripts"))
        import log_hygiene as lh
        cited = lh.build_citation_set(repo / "docs/RESULTS.md")
    except Exception:
        text = _read(repo / "docs/RESULTS.md")
        cited = {_norm_token(m.group(0)) for m in re.finditer(r"logs/[\w\-./]+", text)}
    return locked, cited


def _log_is_cited(rel: str, locked: set[str], cited: set[str]) -> str | None:
    if rel in locked:
        return "locked receipt"
    p = Path(rel)
    for anc in [p, *p.parents]:
        if str(anc) in ("logs", "."):
            break
        if str(anc) in cited:
            return "cited receipt"
    return None


def collect_references(repo: Path) -> dict[str, list[dict]]:
    """source group -> refs. Groups: results, docs, code, logs."""
    out: dict[str, list[dict]] = {"results": [], "docs": [], "code": [],
                                  "logs": []}
    for rel in RESULTS_SOURCES:
        text = _read(repo / rel)
        out["results"] += scan_text_for_refs(text, rel)
        # stems: a word of STEM_MIN+ chars that equals a file stem
        out["results"].append({"kind": "stems", "source": rel,
                               "words": set(re.findall(r"[\w\-]{%d,}" % STEM_MIN,
                                                       text))})
    for rel in ESSENTIAL_DOC_SOURCES:
        if (repo / rel).exists():
            out["docs"] += scan_text_for_refs(_read(repo / rel), rel)
    for g in ESSENTIAL_DOC_GLOBS:
        for p in sorted(repo.glob(g)):
            out["docs"] += scan_text_for_refs(_read(p),
                                              str(p.relative_to(repo)))
    for d in CODE_DIRS:
        base = repo / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if (p.suffix in CODE_EXTS and p.is_file()
                    and ".venv" not in p.parts and "__pycache__" not in p.parts
                    and ".lake" not in p.parts):
                out["code"] += scan_text_for_refs(_read(p),
                                                  str(p.relative_to(repo)))
    logs = repo / "logs"
    if logs.exists():
        for p in sorted(logs.rglob("*")):
            if (p.is_file() and p.suffix in LOG_TEXT_EXTS
                    and p.stat().st_size <= LOG_TEXT_MAX):
                refs = scan_text_for_refs(_read(p), str(p.relative_to(repo)))
                out["logs"] += [r for r in refs if r["kind"] != "basename"]
    return out


def load_lock(repo: Path) -> dict:
    p = repo / "docs/receipts.lock.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("receipts", {})
    except (OSError, ValueError):
        return {}


def load_evidence_pins(repo: Path) -> dict[str, str]:
    """path -> sha256 from the EVIDENCE.md pinned table."""
    pins = {}
    for line in _read(repo / "docs/EVIDENCE.md").splitlines():
        m = re.match(r"\|\s*`([^`]+)`\s*\|\s*([\w-]+)\s*\|\s*([0-9a-f]{64}|\(locked\)|-)\s*\|",
                     line)
        if m and m.group(1).startswith("checkpoints/"):
            pins[m.group(1)] = m.group(3)
    return pins


def load_manifest(repo: Path) -> dict[str, dict]:
    rows = {}
    p = repo / CKPT / "MANIFEST.jsonl"
    if p.exists():
        for line in _read(p).splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    rows[r["path"]] = r
                except (ValueError, KeyError):
                    continue
    return rows


def parse_json_sidecar(repo: Path, rel: str):
    """Parents named by a JSON sidecar under checkpoints/: every
    {path, sha256} pair found anywhere in the document."""
    try:
        doc = json.loads((repo / CKPT / rel).read_text())
    except (OSError, ValueError):
        return None
    found = []

    def visit(o):
        if isinstance(o, dict):
            if isinstance(o.get("path"), str) and o["path"].startswith("checkpoints/"):
                found.append({"path": o["path"], "sha256": o.get("sha256")})
            for v in o.values():
                visit(v)
        elif isinstance(o, list):
            for v in o:
                visit(v)
    visit(doc)
    out = doc.get("out") if isinstance(doc, dict) else None
    return {"parents": found, "out": out if isinstance(out, str) else None}


# -------------------------------------------------------- classification

def _family(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) > 1:
        return parts[0]
    m = re.match(r"([A-Za-z]+(?:[0-9]+m?)?)", parts[0])
    return "root:" + (m.group(1) if m else parts[0])


def _is_receipt_name(name: str) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, g) for g in RECEIPT_GLOBS)


def build(repo: Path, tracked: set[str] | None = None, age_days: float = 14.0,
          now: float | None = None, extra_trees: dict[str, Path] | None = None,
          remote_shas: dict[str, dict[str, str]] | None = None,
          probe_physical: bool = True) -> dict:
    repo = Path(repo)
    now = time.time() if now is None else now
    tracked = git_tracked(repo) if tracked is None else tracked
    inv = walk_tree(repo)
    refs = collect_references(repo)
    lock = load_lock(repo)
    pins = load_evidence_pins(repo)
    manifest = load_manifest(repo)

    by_repo_path = {r["repo_path"]: rel for rel, r in inv.items()}
    by_base = collections.defaultdict(list)
    by_stem = collections.defaultdict(list)
    for rel in inv:
        name = Path(rel).name
        by_base[name].append(rel)
        stem = name.split(".")[0]
        if len(stem) >= STEM_MIN:
            by_stem[stem].append(rel)

    # reasons: rel -> list of (class, reason); edges: graph
    hits: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    edges: list[dict] = []
    unresolved: list[dict] = []
    dynamic_prefixes: list[tuple[str, str]] = []
    generic_loaders: set[str] = set()
    mismatches: list[str] = []
    locked_logs, cited_logs = log_citation_sets(repo, lock)
    absent_targets: dict[str, set] = collections.defaultdict(set)

    def hit(rel, cls, reason, source, kind):
        hits[rel].append((cls, reason))
        edges.append({"file": f"{CKPT}/{rel}", "class": cls, "kind": kind,
                      "source": source})

    def apply_refs(group: str, cls: str, label: str):
        for r in refs[group]:
            k = r["kind"]
            if group == "logs":
                standing = _log_is_cited(r["source"], locked_logs, cited_logs)
                if standing is None:
                    cls, label = "UNKNOWN", "uncited log"
                else:
                    cls, label = "FROZEN", standing
            if k == "stems":
                for w in r["words"]:
                    for rel in by_stem.get(w, []):
                        hit(rel, cls, f"{label} stem {w!r} in {r['source']}",
                            r["source"], "stem")
                continue
            t = r["target"]
            if k == "path":
                rel = by_repo_path.get(t)
                if rel is not None:
                    hit(rel, cls, f"{label} path in {r['source']}",
                        r["source"], "path")
                elif t.startswith("checkpoints/"):
                    # a directory cited without a trailing slash, or a
                    # path that no longer exists
                    prefix = t + "/"
                    matched = False
                    for rel2 in inv:
                        if (CKPT + "/" + rel2).startswith(prefix):
                            hit(rel2, cls, f"{label} dir {t} in {r['source']}",
                                r["source"], "dir")
                            matched = True
                    if not matched and group != "logs":
                        unresolved.append({"kind": "absent-path", "target": t,
                                           "source": r["source"]})
                        absent_targets[t].add(r["source"])
            elif k == "dir":
                prefix = t + "/"
                for rel2 in inv:
                    if (CKPT + "/" + rel2).startswith(prefix):
                        hit(rel2, cls, f"{label} dir {t} in {r['source']}",
                            r["source"], "dir")
            elif k == "basename":
                for rel2 in by_base.get(t, []):
                    if group == "code":
                        hits[rel2].append(("UNKNOWN",
                                           f"basename {t!r} in {r['source']}"))
                        edges.append({"file": f"{CKPT}/{rel2}", "class": "UNKNOWN",
                                      "kind": "basename", "source": r["source"]})
                    else:
                        hit(rel2, cls, f"{label} basename {t!r} in {r['source']}",
                            r["source"], "basename")
            elif k == "dynamic-prefix":
                if group in ("code",):
                    dynamic_prefixes.append((t, r["source"]))
            elif k == "generic-loader":
                if group in ("code",):
                    generic_loaders.add(r["source"])

    apply_refs("results", "FROZEN", "RESULTS")
    apply_refs("docs", "ESSENTIAL", "docs")
    apply_refs("code", "ESSENTIAL", "code")
    apply_refs("logs", "UNKNOWN", "uncited log")
    for src in sorted(generic_loaders):
        unresolved.append({"kind": "generic-loader", "target": "checkpoints",
                           "source": src})

    for prefix, source in sorted(set(dynamic_prefixes)):
        unresolved.append({"kind": "dynamic-prefix", "target": prefix,
                           "source": source})
        for rel in inv:
            if (CKPT + "/" + rel).startswith(prefix):
                hits[rel].append(("UNKNOWN",
                                  f"dynamic path prefix {prefix!r} in {source}"))

    for rel, row in inv.items():
        rp = row["repo_path"]
        if rp in lock:
            ok = lock[rp].get("sha256") == row["sha256"]
            hit(rel, "FROZEN", "sha-locked in docs/receipts.lock.json"
                + ("" if ok else " [SHA MISMATCH]"), "docs/receipts.lock.json",
                "lock")
            if not ok:
                mismatches.append(rp)
        if rp in pins:
            want = pins[rp]
            ok = want in ("(locked)", "-") or want == row["sha256"]
            hit(rel, "FROZEN", "pinned in docs/EVIDENCE.md"
                + ("" if ok else " [SHA MISMATCH]"), "docs/EVIDENCE.md",
                "evidence-pin")
            if not ok:
                mismatches.append(rp)
        if rel in manifest:
            m = manifest[rel]
            ok = m.get("sha256") == row["sha256"]
            hit(rel, "FROZEN" if m.get("verdict") == "confirmed" else "ESSENTIAL",
                "row in checkpoints/MANIFEST.jsonl" + ("" if ok else " [SHA MISMATCH]"),
                "checkpoints/MANIFEST.jsonl", "manifest")
            if not ok:
                mismatches.append(rp)
        if rp in tracked:
            hit(rel, "ESSENTIAL", "git-tracked", "git", "tracked")
        if _is_receipt_name(Path(rel).name):
            hit(rel, "FROZEN", "killed-run receipt name pattern", "name",
                "receipt-name")
        if row["symlink"]:
            hits[rel].append(("UNKNOWN", "symlink (not followed)"))

    # sidecars: JSON sidecars with parents; .sha sidecars with a primary
    sidecar_of: dict[str, str] = {}
    parents_of: dict[str, list[dict]] = {}
    for rel in inv:
        if rel.endswith(".sha"):
            primary = None
            for ext in (".bin", ".pt", ".npz", ".npy", ".safetensors"):
                cand = rel[:-4] + ext
                if cand in inv:
                    primary = cand
                    break
            if primary is None:
                hits[rel].append(("UNKNOWN", "sha sidecar with no primary file"))
            else:
                sidecar_of[rel] = primary
        elif rel.endswith(".json") and not inv[rel]["symlink"]:
            parsed = parse_json_sidecar(repo, rel)
            if parsed is None:
                hits[rel].append(("UNKNOWN", "unparsable JSON sidecar"))
                continue
            primary = None
            if rel.endswith(".merge.json") and rel[:-len(".merge.json")] in inv:
                primary = rel[:-len(".merge.json")]
            elif parsed["out"] and by_repo_path.get(parsed["out"]):
                primary = by_repo_path[parsed["out"]]
            if primary:
                sidecar_of[rel] = primary
                parents_of[primary] = parsed["parents"]
            elif parsed["parents"]:
                parents_of[rel] = parsed["parents"]

    def best(rel) -> str:
        cs = [c for c, _ in hits.get(rel, [])]
        return min(cs, key=lambda c: RANK[c]) if cs else "NONE"

    # propagate: parents inherit the output's class (verified or not)
    changed = True
    rounds = 0
    while changed and rounds < 10:
        changed = False
        rounds += 1
        for out_rel, plist in parents_of.items():
            oc = best(out_rel)
            if oc not in ("FROZEN", "ESSENTIAL"):
                continue
            for par in plist:
                prel = by_repo_path.get(par["path"])
                if prel is None:
                    if not any(u["target"] == par["path"] and u["kind"] == "absent-parent"
                               for u in unresolved):
                        unresolved.append({"kind": "absent-parent",
                                           "target": par["path"],
                                           "source": f"{CKPT}/{out_rel}"})
                    continue
                ok = par.get("sha256") in (None, inv[prel]["sha256"])
                reason = f"parent of {oc} {CKPT}/{out_rel}" + ("" if ok else " [SHA MISMATCH]")
                if (oc, reason) not in hits[prel]:
                    hits[prel].append((oc, reason))
                    edges.append({"file": f"{CKPT}/{prel}", "class": oc,
                                  "kind": "parent-of", "source": f"{CKPT}/{out_rel}"})
                    if not ok and f"{CKPT}/{prel}" not in mismatches:
                        mismatches.append(f"{CKPT}/{prel}")
                    changed = True
        for sc, primary in sidecar_of.items():
            pc = best(primary)
            if pc in ("FROZEN", "ESSENTIAL"):
                reason = f"sidecar of {pc} {CKPT}/{primary}"
                if (pc, reason) not in hits[sc]:
                    hits[sc].append((pc, reason))
                    edges.append({"file": f"{CKPT}/{sc}", "class": pc,
                                  "kind": "sidecar-of", "source": f"{CKPT}/{primary}"})
                    changed = True

    # duplicates
    by_sha = collections.defaultdict(list)
    for rel, row in inv.items():
        if row["sha256"] and row["bytes"] > 0:
            by_sha[row["sha256"]].append(rel)

    def retained_rank(rel):
        kinds = {e["kind"] for e in edges if e["file"] == f"{CKPT}/{rel}"}
        order = ("lock", "evidence-pin", "manifest", "tracked", "path", "dir",
                 "basename", "stem", "parent-of", "sidecar-of")
        return (min((order.index(k) for k in kinds if k in order), default=99), rel)

    classification: dict[str, dict] = {}
    for rel, row in inv.items():
        b = best(rel)
        reasons = [r for _, r in hits.get(rel, [])]
        canonical = None
        if b in ("FROZEN", "ESSENTIAL"):
            cls = b
        elif b == "UNKNOWN":
            cls = "UNKNOWN"
        else:
            grp = by_sha.get(row["sha256"], [])
            retained = [o for o in grp if o != rel and best(o) in ("FROZEN", "ESSENTIAL")]
            if retained:
                canonical = min(retained, key=retained_rank)
                cls = "DUPLICATE"
                reasons.append(f"byte-identical to {b or 'retained'} "
                               f"{CKPT}/{canonical}")
            elif len(grp) > 1 and rel != min(grp):
                canonical = min(grp)
                cls = "DUPLICATE"
                reasons.append(f"byte-identical to uncited {CKPT}/{canonical} "
                               "(lexicographic first kept)")
            elif rel in sidecar_of or rel in parents_of:
                cls = "UNKNOWN"
                reasons.append("sidecar or merge output with no retained relation")
            elif now - row["mtime"] < age_days * 86400:
                cls = "UNKNOWN"
                reasons.append(f"uncited, recent (under {age_days:g} days)")
            elif row["sha256"] is None:
                cls = "UNKNOWN"
                reasons.append("unreadable")
            else:
                cls = "CANDIDATE"
                reasons.append("no reference found in RESULTS, lock, EVIDENCE, "
                               "MANIFEST, docs, code, jobs or small logs")
        classification[rel] = {"class": cls, "reasons": reasons,
                               "canonical": f"{CKPT}/{canonical}" if canonical else None,
                               "family": _family(rel)}

    # duplicate groups with physical probe
    groups = []
    for sha, rels in by_sha.items():
        if len(rels) < 2:
            continue
        rels = sorted(rels)
        keys = {}
        if probe_physical:
            for rel in rels:
                keys[rel] = extent_key(repo / CKPT / rel)
        if not probe_physical or any(v is None for v in keys.values()):
            physical = "unknown"
        elif len(set(keys.values())) == 1:
            physical = "clone-shared"
        elif len(set(keys.values())) == len(rels):
            physical = "distinct"
        else:
            physical = "mixed"
        retained = [r for r in rels if classification[r]["class"] in ("FROZEN", "ESSENTIAL")]
        canonical = (min(retained, key=retained_rank) if retained else min(rels))
        groups.append({"sha256": sha, "bytes": inv[rels[0]]["bytes"],
                       "paths": [f"{CKPT}/{r}" for r in rels],
                       "classes": {f"{CKPT}/{r}": classification[r]["class"] for r in rels},
                       "canonical": f"{CKPT}/{canonical}",
                       "all_retained": len(retained) == len(rels),
                       "physical": physical,
                       "extent_keys": {f"{CKPT}/{r}": (None if keys.get(r) is None else [list(k) for k in keys[r]]) for r in rels},
                       "redundant_bytes": inv[rels[0]]["bytes"] * (len(rels) - 1)})
    groups.sort(key=lambda g: -g["redundant_bytes"])

    # extra trees (independent copies) and remote sha lists
    copies: dict[str, dict[str, list[str]]] = {}
    sha_to_rel = {row["sha256"]: rel for rel, row in inv.items() if row["sha256"]}
    for name, path in (extra_trees or {}).items():
        other = walk_tree(Path(path).parent, Path(path).name) if Path(path).exists() else {}
        m: dict[str, list[str]] = collections.defaultdict(list)
        for orel, orow in other.items():
            if orow["sha256"] in sha_to_rel:
                m[sha_to_rel[orow["sha256"]]].append(orel)
        copies[name] = dict(m)
    for name, table in (remote_shas or {}).items():
        m = collections.defaultdict(list)
        for opath, osha in table.items():
            if osha in sha_to_rel:
                m[sha_to_rel[osha]].append(opath)
        copies[name] = dict(m)

    # cited paths absent from this tree: present in another tree?
    other_paths: dict[str, set[str]] = {}
    for name, path in (extra_trees or {}).items():
        base = Path(path)
        other_paths[name] = set()
        if base.exists():
            for dp, _, fns in os.walk(base):
                for fn in fns:
                    rel = os.path.relpath(os.path.join(dp, fn), base)
                    other_paths[name].add(f"{CKPT}/{rel}")
    for name, table in (remote_shas or {}).items():
        other_paths[name] = set(table)
    absent_cited = []
    for t, sources in sorted(absent_targets.items()):
        present = [n for n, ps in other_paths.items()
                   if t in ps or any(p.startswith(t + "/") for p in ps)]
        absent_cited.append({"target": t, "sources": sorted(sources),
                             "present_in": present})

    families = _families(inv, classification, groups, edges, copies, now)
    summary = _summary(inv, classification, groups, tracked, mismatches,
                       unresolved)
    summary["generic_loaders"] = len(generic_loaders)
    summary["absent_cited_targets"] = len(absent_cited)
    return {"inventory": inv, "classification": classification,
            "graph": {"edges": edges, "unresolved": unresolved},
            "duplicates": groups, "families": families, "copies": copies,
            "absent_cited": absent_cited, "summary": summary}


def _families(inv, classification, groups, edges, copies, now):
    fam: dict[str, dict] = {}
    producers: dict[str, set] = collections.defaultdict(set)
    for e in edges:
        if e["kind"] in ("path", "basename", "dir") and e["source"].split("/")[0] in CODE_DIRS:
            rel = e["file"][len(CKPT) + 1:]
            producers[_family(rel)].add(e["source"])
    dup_by_fam: dict[str, int] = collections.defaultdict(int)
    dup_verified: dict[str, int] = collections.defaultdict(int)
    for g in groups:
        for p in g["paths"]:
            rel = p[len(CKPT) + 1:]
            if classification[rel]["class"] == "DUPLICATE":
                dup_by_fam[_family(rel)] += inv[rel]["alloc"]
                if g["physical"] == "distinct":
                    dup_verified[_family(rel)] += inv[rel]["alloc"]
    for rel, row in inv.items():
        f = classification[rel]["family"]
        d = fam.setdefault(f, {"family": f, "files": 0, "bytes": 0, "alloc": 0,
                               "newest_mtime": 0.0, "classes": collections.Counter(),
                               "class_bytes": collections.Counter(),
                               "sources": collections.Counter()})
        d["files"] += 1
        d["bytes"] += row["bytes"]
        d["alloc"] += row["alloc"]
        d["newest_mtime"] = max(d["newest_mtime"], row["mtime"])
        d["classes"][classification[rel]["class"]] += 1
        d["class_bytes"][classification[rel]["class"]] += row["alloc"]
    for e in edges:
        rel = e["file"][len(CKPT) + 1:]
        fam[_family(rel)]["sources"][e["source"].split("/")[0] if "/" in e["source"] else e["source"]] += 1
    out = []
    for f, d in fam.items():
        c = d["classes"]
        cb = d["class_bytes"]
        n = d["files"]
        copies_here = {name: sum(1 for rel in m if _family(rel) == f)
                       for name, m in copies.items()}
        if c["FROZEN"] + c["ESSENTIAL"] == n:
            action, conf, benefit = "keep (evidence or required)", "high", 0
        elif c["CANDIDATE"] == n:
            action = ("decide: archive off-volume or delete after manual "
                      "review; nothing reaches these files")
            conf, benefit = "estimate", cb["CANDIDATE"]
        elif c["UNKNOWN"] > 0 and c["CANDIDATE"] == 0 and c["DUPLICATE"] == 0:
            action, conf, benefit = ("hold: references of an unresolved form "
                                     "(dynamic path, bare basename, recent)"), "none", 0
        else:
            action = ("mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be "
                      "removed after per-row review; CANDIDATE rows need a "
                      "decision; UNKNOWN rows hold")
            conf = "mixed"
            benefit = dup_verified[f] + cb["CANDIDATE"]
        out.append({"family": f, "files": n, "bytes": d["bytes"], "alloc": d["alloc"],
                    "newest": time.strftime("%Y-%m-%d", time.localtime(d["newest_mtime"])),
                    "age_days": round((now - d["newest_mtime"]) / 86400, 1),
                    "classes": dict(c), "class_alloc": dict(cb),
                    "reference_sources": dict(d["sources"]),
                    "producers_or_consumers": sorted(producers.get(f, [])),
                    "duplicate_alloc": dup_by_fam[f],
                    "duplicate_alloc_verified_distinct": dup_verified[f],
                    "independent_copies": copies_here,
                    "proposed_action": action,
                    "expected_benefit_bytes": benefit,
                    "benefit_confidence": conf})
    out.sort(key=lambda d: -d["alloc"])
    return out


def _summary(inv, classification, groups, tracked, mismatches, unresolved):
    s = {"files": len(inv), "bytes": sum(r["bytes"] for r in inv.values()),
         "alloc": sum(r["alloc"] for r in inv.values()),
         "tracked_files": sum(1 for r in inv.values() if r["repo_path"] in tracked),
         "tracked_bytes": sum(r["bytes"] for r in inv.values() if r["repo_path"] in tracked),
         "symlinks": sum(1 for r in inv.values() if r["symlink"]),
         "hardlinked": sum(1 for r in inv.values() if r["nlink"] > 1),
         "by_class": {}, "sha_mismatches": mismatches,
         "unresolved": collections.Counter(u["kind"] for u in unresolved),
         "duplicate_groups": len(groups),
         "redundant_bytes_logical": sum(g["redundant_bytes"] for g in groups),
         "redundant_bytes_distinct_blocks": sum(g["redundant_bytes"] for g in groups
                                                if g["physical"] == "distinct"),
         "redundant_bytes_clone_shared": sum(g["redundant_bytes"] for g in groups
                                             if g["physical"] == "clone-shared"),
         "redundant_bytes_unknown_physical": sum(g["redundant_bytes"] for g in groups
                                                 if g["physical"] in ("unknown", "mixed"))}
    for c in CLASSES:
        rels = [r for r, k in classification.items() if k["class"] == c]
        s["by_class"][c] = {"files": len(rels),
                            "bytes": sum(inv[r]["bytes"] for r in rels),
                            "alloc": sum(inv[r]["alloc"] for r in rels)}
    dup_distinct = 0
    for g in groups:
        if g["physical"] == "distinct":
            for p in g["paths"]:
                rel = p[len(CKPT) + 1:]
                if classification[rel]["class"] == "DUPLICATE":
                    dup_distinct += inv[rel]["alloc"]
    s["reclaim_verified_duplicate_distinct_blocks"] = dup_distinct
    s["reclaim_estimate_candidate_alloc"] = s["by_class"]["CANDIDATE"]["alloc"]
    s["reclaim_unknown_class_alloc"] = s["by_class"]["UNKNOWN"]["alloc"]
    return s


# --------------------------------------------------------------- baseline

def baseline(repo: Path, tree: str = CKPT) -> dict[str, dict]:
    return {f"{tree}/{rel}": {"sha256": r["sha256"], "bytes": r["bytes"]}
            for rel, r in walk_tree(Path(repo), tree).items()}


def compare_baseline(before: dict, after: dict) -> dict:
    changed = sorted(p for p in before if p in after and before[p] != after[p])
    missing = sorted(p for p in before if p not in after)
    added = sorted(p for p in after if p not in before)
    return {"changed": changed, "missing": missing, "added": added}


# ------------------------------------------------------------------ main

def _gb(b):
    return f"{b / 1e9:.2f}"


def write_outputs(result: dict, out: Path, repo: Path, base_before: dict,
                  base_after: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "inventory.jsonl", "w") as f:
        for rel, row in result["inventory"].items():
            f.write(json.dumps({**row, **result["classification"][rel]}) + "\n")
    (out / "graph.json").write_text(json.dumps(result["graph"], indent=1))
    (out / "duplicates.json").write_text(json.dumps(result["duplicates"], indent=1))
    (out / "families.json").write_text(json.dumps(result["families"], indent=1))
    (out / "copies.json").write_text(json.dumps(result["copies"], indent=1))
    (out / "absent_cited.json").write_text(json.dumps(result["absent_cited"], indent=1))
    summary = dict(result["summary"])
    summary["baseline_compare"] = compare_baseline(base_before, base_after)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "baseline_manifest.json").write_text(json.dumps(base_before, indent=0))
    s = summary
    lines = ["# Checkpoint decision sheet", "",
             f"Tree: {repo / CKPT}. Files {s['files']}, logical {_gb(s['bytes'])} GB, "
             f"allocated {_gb(s['alloc'])} GB, tracked {s['tracked_files']} files "
             f"({_gb(s['tracked_bytes'])} GB), symlinks {s['symlinks']}, "
             f"hard-linked {s['hardlinked']}.", "",
             "| class | files | logical GB | allocated GB |", "|---|---:|---:|---:|"]
    for c in CLASSES:
        v = s["by_class"][c]
        lines.append(f"| {c} | {v['files']} | {_gb(v['bytes'])} | {_gb(v['alloc'])} |")
    lines += ["", "Duplicate groups: %d; redundant logical %s GB; of which distinct "
              "blocks %s GB, clone-shared %s GB, physical unknown %s GB." % (
                  s["duplicate_groups"], _gb(s["redundant_bytes_logical"]),
                  _gb(s["redundant_bytes_distinct_blocks"]),
                  _gb(s["redundant_bytes_clone_shared"]),
                  _gb(s["redundant_bytes_unknown_physical"])),
              "", "Reclaim, verified (DUPLICATE class on distinct blocks): %s GB. "
              "Reclaim, estimate (CANDIDATE class allocated bytes, pending decision): "
              "%s GB. UNKNOWN class allocated: %s GB (no estimate)." % (
                  _gb(s["reclaim_verified_duplicate_distinct_blocks"]),
                  _gb(s["reclaim_estimate_candidate_alloc"]),
                  _gb(s["reclaim_unknown_class_alloc"])),
              "", f"SHA mismatches: {s['sha_mismatches'] or 'none'}. "
              f"Unresolved references: {dict(s['unresolved'])}. Generic loaders "
              f"(code that opens any name under checkpoints/): {s['generic_loaders']}; "
              f"cited paths absent from this tree: {s['absent_cited_targets']} "
              f"(absent_cited.json).",
              "", "## Families (by allocated size)", "",
              "| family | files | alloc GB | newest | FROZEN | ESSENTIAL | DUPLICATE | "
              "CANDIDATE | UNKNOWN | dup distinct GB | copies | benefit GB (conf) | action |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for fam in result["families"]:
        c = fam["classes"]
        cp = ",".join(f"{k}:{v}" for k, v in fam["independent_copies"].items() if v) or "-"
        lines.append("| %s | %d | %s | %s | %d | %d | %d | %d | %d | %s | %s | %s (%s) | %s |" % (
            fam["family"], fam["files"], _gb(fam["alloc"]), fam["newest"],
            c.get("FROZEN", 0), c.get("ESSENTIAL", 0), c.get("DUPLICATE", 0),
            c.get("CANDIDATE", 0), c.get("UNKNOWN", 0),
            _gb(fam["duplicate_alloc_verified_distinct"]), cp,
            _gb(fam["expected_benefit_bytes"]), fam["benefit_confidence"],
            fam["proposed_action"]))
    lines += ["", "## Largest duplicate groups", "",
              "| redundant GB | physical | canonical | copies | classes |", "|---:|---|---|---|---|"]
    for g in result["duplicates"][:40]:
        lines.append("| %s | %s | %s | %s | %s |" % (
            _gb(g["redundant_bytes"]), g["physical"], g["canonical"],
            "; ".join(p for p in g["paths"] if p != g["canonical"]),
            ",".join(sorted(set(g["classes"].values())))))
    bc = summary["baseline_compare"]
    lines += ["", "## Baseline check", "",
              f"changed {len(bc['changed'])}, missing {len(bc['missing'])}, "
              f"added {len(bc['added'])} (must all be 0 for a read-only run)."]
    (out / "SHEET.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out", default=None)
    ap.add_argument("--age-days", type=float, default=14.0)
    ap.add_argument("--extra-tree", action="append", default=[],
                    help="NAME=PATH of another checkpoints directory to hash "
                         "for independent copies")
    ap.add_argument("--remote-shas", action="append", default=[],
                    help="NAME=FILE of 'sha256  path' lines from another machine")
    ap.add_argument("--no-physical", action="store_true")
    ap.add_argument("--verify-baseline", default=None,
                    help="compare the tree against a previous baseline_manifest.json")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    out = Path(a.out) if a.out else repo / "logs" / "rehab-2e" / time.strftime("%Y%m%d-%H%M%S")
    t0 = time.time()
    if a.verify_baseline:
        before = json.loads(Path(a.verify_baseline).read_text())
        d = compare_baseline(before, baseline(repo))
        print(json.dumps(d, indent=1))
        return 0 if not any(d.values()) else 1
    base_before = baseline(repo)
    print(f"baseline: {len(base_before)} files hashed in {time.time() - t0:.0f}s", flush=True)
    extra = {}
    for item in a.extra_tree:
        name, path = item.split("=", 1)
        extra[name] = Path(path)
    remote = {}
    for item in a.remote_shas:
        name, path = item.split("=", 1)
        table = {}
        for line in Path(path).read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and re.match(r"^[0-9a-f]{64}$", parts[0]):
                table[parts[1].strip().lstrip("*")] = parts[0]
        remote[name] = table
    result = build(repo, age_days=a.age_days, extra_trees=extra, remote_shas=remote,
                   probe_physical=not a.no_physical)
    base_after = baseline(repo)
    write_outputs(result, out, repo, base_before, base_after)
    s = result["summary"]
    print(f"wrote {out}: {s['files']} files, {_gb(s['bytes'])} GB logical, "
          f"{_gb(s['alloc'])} GB allocated, {time.time() - t0:.0f}s")
    for c in CLASSES:
        v = s["by_class"][c]
        print(f"  {c:10s} {v['files']:5d} files {_gb(v['alloc']):>8s} GB")
    bc = compare_baseline(base_before, base_after)
    print(f"  baseline compare: {bc}")
    return 0 if not any(bc.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

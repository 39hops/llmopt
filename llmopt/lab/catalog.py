"""catalog.py — model-checkpoint catalog rows (logs-doctrine EXHAUST).

The catalog jsonl this feeds (data/catalog/models.jsonl) is EXHAUST,
regenerable at will — it is NEVER evidence for a verdict (logs
doctrine 2026-08-06: regenerate-don't-download; evidence stays in
checkpoints/MANIFEST.jsonl + the RESULTS entries that cite files).
Nothing here scores weights: arch fields are read from state-dict
SHAPES only, function-space doctrine untouched (RESULTS 6163
joint-perm closure — never score weights by weight distance).

Row schema (scan_checkpoint):
  path        repo-relative POSIX path
  sha256      hex digest or None (fast pass / --no-sha)
  bytes, mtime, ext
  arch        {vocab, d_model, layers, ffn, heads} for house-format
              state dicts (emb.weight / blocks.N.qkv / .gate), else
              None. Loaded torch.load(map_location='cpu', mmap=True)
              — map_location='meta' fails on plain dicts — and only
              .shape is touched, tensors stay on-disk pages.
  parent_ids  FILENAME-lineage list (axes: _s<seed>, stage tags
              _birth/_grown/_latent/_ep<k>/_3ep, gen<k>, size tags
              _19m/_110m/_200m/_400m/d<k>). A _grown file's parent
              is the matching _birth sibling when present; _epK's
              parent is _ep(K-1); genK's parent is gen(K-1) with the
              same surrounding axes. Name-derived hints only, not
              provenance claims.
  ep_marker   text of <path>.ep sidecar if present, else None
  cited       name appears in docs/RESULTS.md (caller greps once and
              passes the set in; cited files are frozen-in-place per
              CODEMAP doctrine — this flag is the read-side warning).
"""
from __future__ import annotations

import os
import posixpath
import re

from llmopt.lab.hash import CHUNK, sha256_file  # noqa: F401  (canonical home moved 2026-08-11)


# ---------------------------------------------------------------- arch


def read_arch(path: str):
    """House-format arch from state-dict shapes; None otherwise.

    Never loads tensor data into judgment — shapes only. heads is
    reported only when derivable (GQA qkv rows != 3*d_model leaves it
    None too; plain MHA qkv carries no head count in its shape).
    """
    try:
        import torch  # optional dep — catalog degrades to arch=None
    except Exception:
        return None
    try:
        # weights_only=True ONLY — never unpickle arbitrary objects on a
        # tree-wide scan; non-tensor pickles just report arch=None.
        sd = torch.load(path, map_location="cpu", mmap=True,
                        weights_only=True)
    except Exception:
        return None
    if not isinstance(sd, dict):
        return None
    # unwrap common {'model': state_dict} / {'state_dict': ...} nests
    for key in ("model", "state_dict", "model_state", "sd"):
        inner = sd.get(key)
        if isinstance(inner, dict) and any(
                hasattr(v, "shape") for v in inner.values()):
            sd = inner
            break
    if "emb.weight" not in sd or not hasattr(sd["emb.weight"], "shape"):
        return None
    vocab, d_model = (int(x) for x in sd["emb.weight"].shape[:2])
    layers = -1
    ffn = None
    heads = None
    for k, v in sd.items():
        m = re.match(r"blocks\.(\d+)\.", k)
        if m:
            layers = max(layers, int(m.group(1)))
        if not hasattr(v, "shape"):
            continue
        if k == "blocks.0.gate.weight":
            ffn = int(v.shape[0])
        elif k == "blocks.0.up.weight" and ffn is None:
            ffn = int(v.shape[0])
        elif k == "blocks.0.qkv.weight":
            rows = int(v.shape[0])
            if rows != 3 * d_model and d_model and rows > d_model:
                # GQA: rows = d + 2*kv_dim; head count still not in shapes
                pass
    return {
        "vocab": vocab,
        "d_model": d_model,
        "layers": layers + 1 if layers >= 0 else None,
        "ffn": ffn,
        "heads": heads,
    }


# ------------------------------------------------------------ lineage

_STAGES = ("birth", "grown", "latent")


def parent_ids(name: str, siblings) -> list:
    """Filename-lineage parents (basenames) present in `siblings`.

    Axes parsed: stage tags _birth/_grown/_latent, _ep<k>/_3ep,
    gen<k>, seeds _s<k>, size tags (_19m/_110m/... , d<k>) are left
    in place as identity, not stepped. Rules:
      _grown -> matching _birth        _latent -> _grown else _birth
      _ep<k> -> _ep<k-1>, _ep0 -> bare stem
      _<k>ep -> plain (no-ep) twin     gen<k> -> gen<k-1> (A/B/suffix
      variants fall back to the plain gen<k-1> file if the exact twin
      is absent).
    """
    sib = set(siblings)
    stem, ext = os.path.splitext(name)
    out = []

    def hit(cand_stem):
        cand = cand_stem + ext
        if cand in sib and cand != name:
            out.append(cand)
            return True
        return False

    if "_grown" in stem:
        hit(stem.replace("_grown", "_birth"))
    if "_latent" in stem:
        if not hit(stem.replace("_latent", "_grown")):
            hit(stem.replace("_latent", "_birth"))
    m = re.search(r"_ep(\d+)", stem)
    if m:
        k = int(m.group(1))
        if k > 0:
            hit(stem[:m.start()] + f"_ep{k - 1}" + stem[m.end():])
        # _ep0 -> bare-stem edge DROPPED (review C1: gallery19m mtimes
        # showed the bare stem can be the rolling LATEST file — the
        # inferred edge pointed a child at its own future). _ep0 rows
        # are parentless unless another axis names a parent.
    m = re.search(r"_(\d+)ep\b", stem)
    if m:  # _3ep style: parent is the no-ep twin
        hit(stem[:m.start()] + stem[m.end():])
    m = re.search(r"gen(\d+)", stem)
    if m:
        k = int(m.group(1))
        if k > 0:
            pre, post = stem[:m.start()], stem[m.end():]
            if not hit(pre + f"gen{k - 1}" + post):
                # variant suffix (gen9B_mps_s2) -> plain prior gen
                for cand in sorted(sib):
                    cs, ce = os.path.splitext(cand)
                    if ce == ext and cs.startswith(pre + f"gen{k - 1}") \
                            and re.fullmatch(r"[A-Za-z]?", cs[len(pre) + 3 + len(str(k - 1)):]):
                        if cand != name:
                            out.append(cand)
                            break
    return out


# ---------------------------------------------------------------- row


def scan_checkpoint(path: str, repo_root: str, cited_names,
                    siblings=None, want_sha: bool = True,
                    want_arch: bool = True) -> dict:
    st = os.stat(path)
    rel = posixpath.join(*os.path.relpath(path, repo_root).split(os.sep))
    name = os.path.basename(path)
    if siblings is None:
        siblings = os.listdir(os.path.dirname(path))
    ep = None
    if os.path.exists(path + ".ep"):
        # errors=: a binary/garbled .ep must degrade, not kill the scan
        with open(path + ".ep", errors="replace") as f:
            ep = f.read().strip()
    return {
        "path": rel,
        "sha256": sha256_file(path) if want_sha else None,
        "bytes": st.st_size,
        "mtime": st.st_mtime,
        "ext": os.path.splitext(name)[1],
        "arch": read_arch(path) if want_arch else None,
        "parent_ids": parent_ids(name, siblings),
        "ep_marker": ep,
        "cited": name in cited_names,
    }

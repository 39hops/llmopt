"""fold_book — the /fold-book chain: fold a draft, lint it, THEN book it,
with every stage gated on a real exit code and no partial fold ever
reaching the ledger.

Origin (2026-09-14): a fold script asserted mid-way, left the draft
unfolded, and the booking script appended the unfolded text to RESULTS
(reverted before commit). This chain makes that impossible:

  1. FOLD    the fold script runs against a COPY of the draft; only if it
             exits 0 AND rewrote the copy is the copy moved over the draft
             (a fold that changes nothing is an error unless --allow-nofold).
  2. LINT    scripts/claim_lint.py on the folded draft (with --prereg/--obs
             when given); rc must be 0.
  3. CHECKS  any extra check commands (--check "cmd ..."), each rc 0.
  4. BOOK    the booking script (the /book ritual as a script) runs only
             now. Its rc is the chain's rc.

Nothing here knows the ledger's semantics: booking scripts keep them.

Usage:
  .venv/bin/python scripts/fold_book.py --draft /tmp/x.md --fold /tmp/fold.py --book /tmp/book.py
        [--prereg docs/preregs/x.json --obs obs.json] [--check "cmd"]... [--allow-nofold] [--dry-run]
The fold script receives the draft path as its single argument.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable


def run(cmd, label):
    print(f"[fold-book] {label}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[fold-book] STOP: {label} exited {r.returncode}; nothing booked", flush=True)
        sys.exit(r.returncode or 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--fold", help="fold script (python); receives the draft path; may be omitted with --allow-nofold")
    ap.add_argument("--book", required=True, help="booking script (python), run only after fold + lint + checks pass")
    ap.add_argument("--prereg"); ap.add_argument("--obs")
    ap.add_argument("--check", action="append", default=[], help="extra shell command that must exit 0 (repeatable)")
    ap.add_argument("--allow-nofold", action="store_true", help="accept a fold that leaves the draft byte-identical (or no --fold)")
    ap.add_argument("--dry-run", action="store_true", help="run fold + lint + checks, skip booking")
    a = ap.parse_args(argv)
    draft = Path(a.draft).resolve()
    if not draft.is_file():
        print(f"[fold-book] STOP: draft {draft} missing"); sys.exit(2)
    before = draft.read_bytes()
    if a.fold:
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / draft.name
            shutil.copyfile(draft, copy)
            run([PY, a.fold, str(copy)], "FOLD")
            after = copy.read_bytes()
            if after == before and not a.allow_nofold:
                print("[fold-book] STOP: fold exited 0 but changed nothing (pass --allow-nofold if intended)"); sys.exit(3)
            shutil.copyfile(copy, draft)
    elif not a.allow_nofold:
        print("[fold-book] STOP: no --fold given; pass --allow-nofold to book an unfolded draft deliberately"); sys.exit(3)
    lint = [PY, str(ROOT / "scripts" / "claim_lint.py"), str(draft)]
    if a.prereg and a.obs:
        lint += ["--prereg", a.prereg, "--obs", a.obs]
    run(lint, "LINT")
    for i, c in enumerate(a.check):
        run(["bash", "-c", c], f"CHECK{i}")
    if a.dry_run:
        print("[fold-book] dry run: fold + lint + checks passed; booking skipped"); return 0
    run([PY, a.book], "BOOK")
    print("[fold-book] booked", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Source-invariant guard for ATOM-DIET-TRAJECTORY-1 (RESULTS L66546,
qualification item 1): the trajectory driver is the results-cited
ATOM-DIET-LADDER-1 driver plus snapshot emission, stream-digest
assertions, paths and a non-finite-loss abort. Every line of the ladder
recipe must survive verbatim; the only ladder lines allowed to be
absent are the enumerated gate / receipt / path lines; and no gate call
may exist anywhere in the trajectory driver.
"""
import difflib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "scratch" / "birth19m_atoms_ladder.py"
TRAJ = ROOT / "scratch" / "birth19m_atoms_traj.py"

# ladder lines that the sibling is allowed to drop (gate, receipt, its
# own paths and docstring), matched by stripped content
ALLOWED_DROPPED = {
    'OUT = Path(f"checkpoints/gallery19m_{LADDER_ARM}_s{SEED}.pt")',
    "if OUT.exists():",
    'raise SystemExit(f"REFUSING: {OUT} exists")',
    'RECEIPTS = Path("logs/atomladder1/arms.jsonl")',
    "torch.save(model.state_dict(), OUT)",
    'print(f"[ladder] saved {OUT} after {step} steps "',
    'f"({time.time()-t0:.0f}s)", flush=True)',
    "from llmopt.lab.gate import gate_eval",
    "from llmopt.lab.hash import git_sha",
    "model.eval()",
    "solves, valid = gate_eval(model, tok, dev)",
    "tot = sum(solves.values())",
    "RECEIPTS.parent.mkdir(parents=True, exist_ok=True)",
    'row = {"arm": LADDER_ARM, "seed": SEED, "steps": step,',
    '"solves": solves, "total": tot,',
    '"valid_pct": round(valid, 2), "device": dev,',
    '"n_atoms_shard": len(atoms),',
    '"atom_rows_per_epoch": atom_exposure,',
    '"code_commit": git_sha(short=True)}',
    'with RECEIPTS.open("a") as f:',
    'f.write(json.dumps(row) + "\\n")',
    'print(f"[ladder] GATE arm={LADDER_ARM} s{SEED} {solves} = "',
    'f"{tot}/120 @ {valid:.2f}%", flush=True)',
    "def encode_flagged(rows, tok):",
    '"""The trainer\'s text/encode/filter path (C.encode_with_levels',
    'verbatim) with an is_atom flag carried alongside."""',
    "triples = []",
    "for r in rows:",
    "t = f\"Current: {r['cur']}\\nHints: none\\nStep: {r['nxt']}\\n\"",
    "try:",
    "ids = tok.encode(t) + [tok.eos_id]",
    "except ValueError:",
    "continue",
    'if len(ids) <= int(os.environ.get("SEQ_CAP", "512")):',
    'triples.append((ids, int(r["level"]),',
    'r.get("source") == "atom-oneply"))',
    "triples.sort(key=lambda p: len(p[0]))   # stable, = enc.sort(key=len)",
    "enc = [p[0] for p in triples]",
    "is_atom = [p[2] for p in triples]",
    "return enc, is_atom",
    "for a, b in stream:",
    "stream = C.stock_epoch_stream(len(enc), ep)",
    "dropped = len(stream) - steps_per_epoch",
    "stream = stream[:steps_per_epoch]",
    "n_atom_rows = sum(1 for a, b in stream",
    "for j in range(a, b) if is_atom[j])",
    'print(f"[ladder] ep{ep}: {len(stream)} batches ({dropped} "',
    'f"dropped to match stock count), {n_atom_rows} atom "',
    'f"rows in stream", flush=True)',
    'print(f"[ladder] shard {SHARD}: {len(atoms)} rows after "',
    'print(f"[ladder] arm={LADDER_ARM} seed={SEED}: stock "',
    'print(f"[ladder] steps_total {steps_total} "',
    "dev = (\"mps\" if torch.backends.mps.is_available() else",
    '"cuda" if torch.cuda.is_available() else "cpu")',
    'LADDER_ARM = os.environ.get("ARM", "")',
    'SEED = int(os.environ["SEED"])',
    'assert LADDER_ARM in ("stock", "atoms"), LADDER_ARM',
}

# recipe lines that must appear verbatim in both files
RECIPE_SENTINELS = [
    "model = TM.build_model(len(tok.vocab), d=384, layers=8,",
    "heads=6, ffn=1536).to(dev)",
    "opt = torch.optim.AdamW(model.parameters(), lr=3e-4,",
    "weight_decay=0.01)",
    "sched = torch.optim.lr_scheduler.OneCycleLR(",
    "opt, max_lr=3e-4, total_steps=steps_total, pct_start=0.03)",
    "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)",
    "if sched.last_epoch < steps_total - 1:",
    "sched.step()",
    "opt.zero_grad()",
    "torch.manual_seed(SEED)",
    "C.assert_noop(enc_stock)    # precondition, fresh, in-process",
    "steps_total = EPOCHS * (len(enc_stock) // BS)",
    "assert steps_total == STEPS_TOTAL_PIN, steps_total",
    "logits = model(ids[:, :-1], mask[:, :-1])",
    "labels[mask[:, 1:] == 0] = -100",
    "loss = torch.nn.functional.cross_entropy(",
    "labels.reshape(-1), ignore_index=-100)",
    'SHARD = Path("data/micromodel_atoms_shard0.jsonl")',
    "STEPS_TOTAL_PIN = 15_420",
    "EPOCHS, BS = C.EPOCHS, C.BS",
    'os.environ["ARM"] = "off"       # frozen module import side-effects only',
    'os.environ["BIRTH_SEED"] = str(SEED)',
]


def _code_lines(path):
    src = path.read_text()
    src = re.sub(r'^""".*?"""', "", src, count=1, flags=re.S)   # drop module docstring
    return [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]


def test_ladder_recipe_survives_verbatim():
    ladder, traj = _code_lines(LADDER), _code_lines(TRAJ)
    traj_set = set(traj)
    dropped = [l for l in ladder if l not in traj_set]
    unexpected = [l for l in dropped if l not in ALLOWED_DROPPED]
    assert not unexpected, "ladder lines missing from the trajectory driver:\n  " + "\n  ".join(unexpected)


def test_recipe_sentinels_present_in_both():
    for path in (LADDER, TRAJ):
        lines = set(_code_lines(path))
        missing = [s for s in RECIPE_SENTINELS if s not in lines]
        assert not missing, f"{path.name} lacks recipe lines: {missing}"


def test_no_gate_inside_trajectory_driver():
    src = TRAJ.read_text()
    assert "gate_eval" not in src
    assert "llmopt.lab.gate" not in src


def test_training_loop_body_identical():
    """The for-loop body from `for a, b in stream:` through `opt.zero_grad()`
    is byte-identical between the two files."""
    def body(path):
        src = path.read_text()
        m = re.search(r"for a, b in stream:\n(.*?)opt\.zero_grad\(\)\n", src, flags=re.S)
        assert m, path
        return m.group(1)
    assert body(LADDER) == body(TRAJ)


def test_sequence_order_is_stable():
    d = list(difflib.unified_diff(_code_lines(LADDER), _code_lines(TRAJ), lineterm="", n=0))
    assert d, "files identical: emission was never added"

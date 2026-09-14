"""scripts/fold_book.py guards: a fold that fails never lets the booking
script run and never touches the draft; a fold that changes nothing stops
unless --allow-nofold; a lint failure stops before booking; the happy path
folds, lints and books in that order."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _write(p, text):
    p.write_text(text)
    return str(p)


def _run(args):
    return subprocess.run([PY, str(ROOT / "scripts" / "fold_book.py"), *args], capture_output=True, text=True, cwd=ROOT)


def test_failing_fold_books_nothing_and_leaves_draft(tmp_path):
    draft = tmp_path / "d.md"; draft.write_text("## NOTE T: alpha\n\nbody alpha\n")
    fold = _write(tmp_path / "fold.py", "import sys\nfrom pathlib import Path\np=Path(sys.argv[1]); s=p.read_text()\nassert s.count('nope')==1\np.write_text(s.replace('alpha','beta'))\n")
    marker = tmp_path / "booked"
    book = _write(tmp_path / "book.py", f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n")
    r = _run(["--draft", str(draft), "--fold", fold, "--book", book])
    assert r.returncode != 0 and "STOP: FOLD" in r.stdout
    assert not marker.exists() and draft.read_text().count("alpha") == 2


def test_nofold_stops_unless_allowed(tmp_path):
    draft = tmp_path / "d.md"; draft.write_text("## NOTE T: alpha\n\nbody\n")
    fold = _write(tmp_path / "fold.py", "import sys\n")
    marker = tmp_path / "booked"
    book = _write(tmp_path / "book.py", f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n")
    r = _run(["--draft", str(draft), "--fold", fold, "--book", book])
    assert r.returncode == 3 and not marker.exists()
    r = _run(["--draft", str(draft), "--fold", fold, "--book", book, "--allow-nofold"])
    assert r.returncode == 0 and marker.exists()


def test_lint_or_check_failure_stops_before_booking(tmp_path):
    draft = tmp_path / "d.md"; draft.write_text("## NOTE T: alpha\n\nbody\n")
    marker = tmp_path / "booked"
    book = _write(tmp_path / "book.py", f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n")
    r = _run(["--draft", str(draft), "--book", book, "--allow-nofold", "--check", "false"])
    assert r.returncode != 0 and "STOP: CHECK0" in r.stdout and not marker.exists()


def test_happy_path_folds_then_books(tmp_path):
    draft = tmp_path / "d.md"; draft.write_text("## NOTE T: alpha\n\nbody alpha\n")
    fold = _write(tmp_path / "fold.py", "import sys\nfrom pathlib import Path\np=Path(sys.argv[1]); s=p.read_text()\nassert s.count('body alpha')==1\np.write_text(s.replace('body alpha','body beta'))\n")
    marker = tmp_path / "booked"
    book = _write(tmp_path / "book.py", f"from pathlib import Path\nPath({str(marker)!r}).write_text(Path({str(draft)!r}).read_text())\n")
    r = _run(["--draft", str(draft), "--fold", fold, "--book", book, "--check", "true"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "body beta" in draft.read_text() and marker.read_text() == draft.read_text()
    assert r.stdout.index("FOLD") < r.stdout.index("LINT") < r.stdout.index("CHECK0") < r.stdout.index("BOOK")

"""Import-every-module smoke (spec §Phase 5 dead-code pass): every
module in llmopt/ must import, or fail only on a KNOWN optional
extra. Anything else (syntax rot, moved deps, circular imports) is a
hard failure — this is the guard that makes dead-code deletion and
future moves safe."""
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = ("mlx", "triton", "pyarrow", "duckdb", "transformers",
            "matplotlib", "peft", "datasets")

MODULES = sorted(
    str(p.relative_to(ROOT)).removesuffix(".py").replace("/", ".")
    for p in (ROOT / "llmopt").rglob("*.py")
    if "__pycache__" not in p.parts and p.name != "__main__.py"
    # vendored axiom files are verbatim CLIs (argparse at module
    # scope) guarded by their own source-identity tests — not ours
    and "vendor" not in p.parts
)


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    try:
        importlib.import_module(mod)
    except ImportError as e:
        if any(opt in str(e) for opt in OPTIONAL):
            pytest.skip(f"optional extra: {e}")
        raise

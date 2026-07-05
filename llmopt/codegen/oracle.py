"""Compiler-as-oracle: MSVC labels C programs so nothing is hand-annotated.

The chess-engine framing: the toolchain is the rules of the game. Every
label is produced, not judged — compile ok / exact diagnostics, program
stdout, optimized asm listing, and (bytes, mnemonic) pairs straight out
of dumpbin /disasm. A model's answer is scored by asking the toolchain
again, never by string proximity to a reference.

MSVC needs its environment (vcvars64); the oracle locates vcvarsall via
VSINSTALLDIR/common install paths and runs every tool through one
`cmd /c call vcvars && ...` invocation per request.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_VCVARS_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
]


def find_vcvars() -> str | None:
    for c in _VCVARS_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def _run_in_vcvars(cmdline: str, cwd: str, timeout: int = 60):
    vcvars = find_vcvars()
    if vcvars is None:
        raise RuntimeError("MSVC vcvars64.bat not found")
    return subprocess.run(
        f'cmd /c "call "{vcvars}" >nul 2>&1 && {cmdline}"',
        cwd=cwd, capture_output=True, text=True, timeout=timeout, shell=True,
    )


@dataclass(frozen=True)
class CompileResult:
    ok: bool
    diagnostics: list[str]     # e.g. ["C2065: 'y': undeclared identifier"]
    asm: str                   # cleaned /FA listing ("" if compile failed)
    stdout: str | None         # program output (None if not run / failed)
    encodings: list[tuple[str, str]]  # (hex bytes, mnemonic) per instruction


_DIAG_RE = re.compile(r"(?:error|warning) (C\d{4,5}): (.*)")
_DISASM_RE = re.compile(
    r"^\s+[0-9A-F]+:\s+((?:[0-9A-F]{2} )+)\s*([a-z].*?)\s*$", re.MULTILINE
)


def _clean_asm(listing: str) -> str:
    """Strip MSVC listing noise (comments, directives) down to labels and
    instructions — the part a model should predict."""
    keep = []
    for line in listing.splitlines():
        s = line.split(";")[0].rstrip()
        if not s:
            continue
        if any(s.lstrip().startswith(d) for d in
               ("TITLE", "INCLUDELIB", ".model", "OPTION", "include",
                "PUBLIC", "EXTRN", "_TEXT", "END", "PARA", "msvcjmc")):
            continue
        keep.append(s)
    return "\n".join(keep)


def compile_c(
    source: str, *, opt: str = "/O2", run: bool = False, stdin: str = "",
) -> CompileResult:
    """Compile (optionally run) a C translation unit; harvest all labels."""
    with tempfile.TemporaryDirectory() as td:
        Path(td, "p.c").write_text(source)
        r = _run_in_vcvars(f"cl /nologo {opt} /FAs p.c", td)
        diags = [f"{code}: {msg}" for code, msg in _DIAG_RE.findall(r.stdout + r.stderr)]
        ok = Path(td, "p.exe").exists()
        asm = _clean_asm(Path(td, "p.asm").read_text()) if Path(td, "p.asm").exists() else ""

        encodings: list[tuple[str, str]] = []
        if Path(td, "p.obj").exists():
            d = _run_in_vcvars("dumpbin /nologo /disasm p.obj", td)
            encodings = [
                (b.strip(), m) for b, m in _DISASM_RE.findall(d.stdout)
            ]

        stdout = None
        if run and ok:
            p = subprocess.run(
                [str(Path(td, "p.exe"))], input=stdin,
                capture_output=True, text=True, timeout=10,
            )
            stdout = p.stdout
        return CompileResult(ok, diags, asm, stdout, encodings)


def msvc_available() -> bool:
    return find_vcvars() is not None

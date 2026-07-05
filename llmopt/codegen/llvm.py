"""LLVM toolchain oracle: clang / llvm-mc / objdump (MSYS mingw64).

Faster sibling of oracle.py (no vcvars shell), plus the two tools MSVC
lacks: llvm-mc gives per-instruction byte encodings both directions, so
a model's *predicted* assembly can be scored by assembling it — the
toolchain judges semantics, not string distance.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_TOOL_DIRS = [r"C:\msys64\mingw64\bin", r"C:\msys64\ucrt64\bin", ""]


def _tool(name: str) -> str | None:
    for d in _TOOL_DIRS:
        p = Path(d, f"{name}.exe") if d else None
        if p and p.exists():
            return str(p)
        if not d:
            from shutil import which
            return which(name)
    return None


def llvm_available() -> bool:
    return all(_tool(t) for t in ("clang", "llvm-mc", "objdump"))


def _run(args: list[str], input_text: str | None = None, timeout: int = 60):
    return subprocess.run(
        args, input=input_text, capture_output=True, text=True, timeout=timeout
    )


@dataclass(frozen=True)
class ClangResult:
    ok: bool
    diagnostics: list[str]            # "error: use of undeclared identifier 'x'"
    asm: str                          # intel-syntax -S output, cleaned
    stdout: str | None
    encodings: list[tuple[str, str]]  # (hex bytes, intel mnemonic)


_DIAG_RE = re.compile(r"(error|warning): (.*)")
_OBJDUMP_RE = re.compile(
    r"^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)\s*\t?([a-z].*?)\s*(?:#.*)?$", re.MULTILINE
)


def _clean_asm(listing: str) -> str:
    """Keep labels + instructions; drop directives and comments."""
    keep = []
    for line in listing.splitlines():
        s = line.split("#")[0].rstrip()
        if not s or s.lstrip().startswith("."):
            continue
        keep.append(s)
    return "\n".join(keep)


def compile_c(
    source: str, *, opt: str = "-O2", run: bool = False, stdin: str = "",
) -> ClangResult:
    clang = _tool("clang")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td, "p.c")
        src.write_text(source)
        rs = _run([clang, "-S", "-masm=intel", opt,
                   "-o", str(Path(td, "p.s")), str(src)])
        diags = [f"{lvl}: {msg}" for lvl, msg in _DIAG_RE.findall(rs.stderr)]
        ok = rs.returncode == 0
        asm = _clean_asm(Path(td, "p.s").read_text()) if ok else ""

        encodings: list[tuple[str, str]] = []
        stdout = None
        if ok:
            _run([clang, "-c", opt, "-o", str(Path(td, "p.o")), str(src)])
            d_raw = _run([_tool("objdump"), "-d", "-M", "intel",
                          str(Path(td, "p.o"))])
            encodings = [
                (b.strip(), m.strip()) for b, m in _OBJDUMP_RE.findall(d_raw.stdout)
                if not m.startswith(("nop", "int3", "lea    0x0"))
            ]
            if run:
                _run([clang, opt, "-o", str(Path(td, "p.exe")), str(src)])
                p = _run([str(Path(td, "p.exe"))], input_text=stdin, timeout=10)
                stdout = p.stdout
        return ClangResult(ok, diags, asm, stdout, encodings)


def assemble(instruction: str) -> str | None:
    """One intel-syntax instruction -> canonical hex bytes via llvm-mc
    ('# encoding: [0xb8,0x05,...]'). None if it does not assemble — which
    is exactly how a hallucinated mnemonic scores as wrong."""
    r = _run(
        [_tool("llvm-mc"), "--x86-asm-syntax=intel", "-triple",
         "x86_64-pc-windows-msvc", "--show-encoding"],
        input_text=".intel_syntax noprefix\n" + instruction.strip() + "\n",
    )
    m = re.search(r"encoding: \[([^\]]+)\]", r.stdout)
    if not m or r.returncode != 0:
        return None
    return " ".join(x.strip().replace("0x", "") for x in m.group(1).split(","))


def norm_bytes(hexstr: str) -> str:
    """Canonicalize a hex byte string for comparison ('B8 05' == 'b805')."""
    s = re.sub(r"[^0-9a-fA-F]", "", hexstr).lower()
    return " ".join(s[i : i + 2] for i in range(0, len(s), 2))

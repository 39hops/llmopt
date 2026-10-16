"""Serialization contract for mathgen pin digests.

A dataset digest is the SHA-256 of the UTF-8 encoding of the records
``f"{kind}\\t{level}\\t{prompt}\\t{answer}"`` joined by ``"\\n"`` with a
trailing ``"\\n"``, in dataset order. Prompts and answers are taken
verbatim from the ``Problem`` (no normalisation), so the digest pins the
sequence, the kind and level cycling, the seed arithmetic and the
retry outcomes together. ``_expr`` is not part of the record: the
payload is reconstructable from the answer and is not comparable
across sympy versions.

``CONTRACT`` names the exact rule and is written into every pin file
so a reader never has to infer it from code.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

CONTRACT = (
    "sha256(utf8('\\n'.join(f'{kind}\\t{level}\\t{prompt}\\t{answer}' "
    "for each problem in order) + '\\n'))"
)


def problem_record(p) -> str:
    return f"{p.kind}\t{p.level}\t{p.prompt}\t{p.answer}"


def dataset_digest(problems: Iterable) -> str:
    text = "\n".join(problem_record(p) for p in problems) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

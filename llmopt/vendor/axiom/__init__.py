"""Vendored axiom-lab sources (verbatim + 3-line provenance headers).

Adopted 2026-08-11 at axiom sha b785601. Upstream code, not house
style — fixes land upstream first, then re-vendor; source-identity is
guarded by tests/test_vendor_axiom.py (the test_lab_adoption pattern).

CAUTION: ``divergence.py`` and ``classify_sample.py`` are scripts —
``divergence`` runs argparse at import time. Import ``nn_exact_ref``
for the FX-V1 integer reference and the AXNN container writer
(``write_axnn``); the fixture-side parser lives in the tests.
"""

"""Root llmopt import stays cheap and torch-free (spec §Phase 5)."""
import subprocess
import sys


def test_import_llmopt_is_torch_free():
    code = ("import sys; import llmopt; "
            "assert 'torch' not in sys.modules, 'torch leaked'; "
            "assert 'sympy' not in sys.modules, 'sympy leaked'; "
            "print('CLEAN')")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout


def test_lazy_attrs_resolve():
    import llmopt
    assert llmopt.RadixCache.__name__ == "RadixCache"
    assert callable(llmopt.find_ngram_continuation)
    assert callable(llmopt.allocate_bits)
    assert callable(llmopt.pareto_front)

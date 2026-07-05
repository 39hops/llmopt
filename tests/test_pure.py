"""Tests for the torch-free parts: n-gram lookup, radix cache, allocator."""

from llmopt.quantize.allocator import allocate_bits, pareto_front
from llmopt.decoding.prompt_lookup import find_ngram_continuation
from llmopt.cache.radix import RadixCache
from llmopt.quantize.sensitivity import LayerSensitivity


# ---- prompt lookup ----

def test_ngram_finds_repeated_trigram():
    # ... 1 2 3 9 9 ... 1 2 3  -> continuation after earlier "1 2 3" is [9, 9]
    ctx = [5, 1, 2, 3, 9, 9, 7, 8, 1, 2, 3]
    assert find_ngram_continuation(ctx, num_draft=2) == [9, 9]


def test_ngram_prefers_longest_match():
    # bigram "2 3" appears twice; trigram "1 2 3" once earlier -> trigram wins
    ctx = [1, 2, 3, 4, 0, 2, 3, 5, 1, 2, 3]
    assert find_ngram_continuation(ctx, max_ngram=3, num_draft=1) == [4]


def test_ngram_no_match_returns_empty():
    assert find_ngram_continuation([1, 2, 3, 4, 5], num_draft=4) == []


def test_ngram_most_recent_match_wins():
    ctx = [7, 1, 7, 2, 7]  # unigram "7": matches at 0 (->1) and 2 (->2); recent wins
    assert find_ngram_continuation(ctx, max_ngram=1, num_draft=1) == [2]


# ---- radix cache ----

def test_radix_miss_then_hit():
    c = RadixCache()
    assert c.match([1, 2, 3]) == (0, [])
    c.insert([1, 2, 3], lambda s, e: f"kv[{s}:{e}]")
    n, payloads = c.match([1, 2, 3, 4, 5])
    assert n == 3
    assert payloads == ["kv[0:3]"]


def test_radix_extends_prefix():
    c = RadixCache()
    c.insert([1, 2, 3], lambda s, e: (s, e))
    c.insert([1, 2, 3, 4, 5], lambda s, e: (s, e))
    n, payloads = c.match([1, 2, 3, 4, 5])
    assert n == 5
    assert payloads == [(0, 3), (3, 5)]
    assert c.cached_tokens == 5


def test_radix_divergent_suffix_no_false_match():
    c = RadixCache()
    c.insert([1, 2, 3], lambda s, e: "a")
    n, _ = c.match([1, 9, 9])
    assert n == 0  # edge [1,2,3] only partially matches; treated as miss


def test_radix_lru_eviction():
    c = RadixCache(max_tokens=6)
    c.insert([1, 1, 1], lambda s, e: "old")
    c.match([2, 2, 2])  # no-op
    c.insert([2, 2, 2], lambda s, e: "mid")
    c.match([1, 1, 1])  # refresh old -> mid is now LRU
    c.insert([3, 3, 3], lambda s, e: "new")  # forces eviction
    assert c.cached_tokens <= 6
    assert c.match([1, 1, 1])[0] == 3  # refreshed entry survived
    assert c.match([2, 2, 2])[0] == 0  # LRU victim gone


# ---- allocator ----

def _sens():
    # two layers, same size; A fragile at 2-bit, B tolerant
    return [
        LayerSensitivity("A", 2, 1.00, 100),
        LayerSensitivity("A", 4, 0.10, 100),
        LayerSensitivity("B", 2, 0.05, 100),
        LayerSensitivity("B", 4, 0.01, 100),
    ]


def test_allocator_upgrades_fragile_layer_first():
    a = allocate_bits(_sens(), kl_budget=0.2)
    assert a.bits_by_layer == {"A": 4, "B": 2}
    assert abs(a.est_delta_kl - 0.15) < 1e-9
    assert a.total_bits == 4 * 100 + 2 * 100


def test_allocator_loose_budget_stays_low_bit():
    a = allocate_bits(_sens(), kl_budget=10.0)
    assert a.bits_by_layer == {"A": 2, "B": 2}


def test_allocator_zero_budget_goes_fp16():
    a = allocate_bits(_sens(), kl_budget=0.0)
    assert a.bits_by_layer == {"A": 16, "B": 16}
    assert abs(a.est_delta_kl) < 1e-12
    assert a.avg_bits == 16.0


def test_pareto_front_monotone():
    pts = pareto_front(_sens(), kl_budgets=[0.0, 0.05, 0.2, 2.0])
    sizes = [a.total_bits for _, a in pts]
    assert sizes == sorted(sizes, reverse=True)  # tighter budget -> more bits

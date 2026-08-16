"""SCIENCE-INCIDENT FIXTURE: wrong_metric_population.

The regression half of GRADUATION. House rule (RIFF-LEDGER
2026-08-16): every auditor BLOCKER graduates TWICE — once into an
executable invariant, once into a fixture reproducing the original
failure. A documented law alone is PROMOTED, not GRADUATED.

ORIGINAL FAILURE, reproduced below with the real numbers.
STREAM-WDISTILL-0 EXEC1 measured operator error at expert 0 only
(n=1 of 256) and scored BAR 2 — a bar registered for the LAYER —
against it. The audit-repair then pooled operator error over all
256 experts. Both numbers were valid; both arms were admissible;
differencing them measures the POPULATION CHANGE, not the repair.
The class recurred hours after being documented, inside the
analysis plan written to price that very repair.

Each test names the reason code it expects, so the incident table's
status column can be derived from the suite rather than asserted.
"""
import pytest

from llmopt.lab.metrics import (POPULATION_MISMATCH, AGGREGATION_MISMATCH,
                                NOT_ADJUDICABLE, DescriptiveContrast,
                                Metric, MetricContractError, adjudicate,
                                cross_population_difference)

# The real receipt values (logs/streamwd/pass12_B1.jsonl exec1 row,
# logs/streamwd/pass12_B1_repair.jsonl repair row).
EXEC1_OPERATOR_E0 = Metric(
    value=0.7657, metric="operator_error", population="expert:0",
    aggregation="mean_of_ratios", provenance="exec1")
REPAIR_OPERATOR_LAYER = Metric(
    value=0.769457, metric="operator_error", population="experts:0:256",
    aggregation="pooled_ratio", provenance="repair")


def test_the_original_failure_is_refused():
    """Direction 1: the subtraction that started it all must refuse."""
    with pytest.raises(MetricContractError) as e:
        _ = REPAIR_OPERATOR_LAYER - EXEC1_OPERATOR_E0
    assert e.value.reason == POPULATION_MISMATCH


def test_deliberate_cross_population_look_is_allowed():
    """Direction 2: you may look, explicitly, and get a DIFFERENT type."""
    c = cross_population_difference(
        REPAIR_OPERATOR_LAYER, EXEC1_OPERATOR_E0,
        purpose="descriptive: how far apart are the two runs' operator "
                "columns, knowing the population changed")
    assert isinstance(c, DescriptiveContrast)
    assert c.adjudicable is False


def test_the_look_cannot_become_evidence():
    """Direction 3: and the look must not satisfy a registered bar."""
    c = cross_population_difference(
        REPAIR_OPERATOR_LAYER, EXEC1_OPERATOR_E0, purpose="descriptive")
    with pytest.raises(MetricContractError) as e:
        adjudicate(c, bar_value=0.10, direction="below")
    assert e.value.reason == NOT_ADJUDICABLE


def test_bar_refuses_a_metric_from_the_wrong_population():
    """BAR 2's actual defect: a layer bar scored on one expert."""
    with pytest.raises(MetricContractError) as e:
        adjudicate(EXEC1_OPERATOR_E0, bar_value=0.10, direction="below",
                   required_population="experts:0:256")
    assert e.value.reason == POPULATION_MISMATCH


def test_aggregation_mismatch_is_its_own_class():
    """Pooled ratio and mean-of-ratios estimate different quantities.

    The repair's spectral column was a mean of ratios reported under
    a pooled name — the third appearance of this class in one thread.
    """
    pooled = Metric(0.5, "spectral_error", "experts:0:256", "pooled_ratio")
    meaned = Metric(0.5, "spectral_error", "experts:0:256",
                    "mean_of_ratios")
    with pytest.raises(MetricContractError) as e:
        _ = pooled - meaned
    assert e.value.reason == AGGREGATION_MISMATCH


def test_the_invariant_is_not_merely_refusing_everything():
    """A same-population, same-aggregation comparison must WORK.

    An invariant that refuses all comparisons would pass every test
    above while being useless. This is the negative control.
    """
    a = Metric(0.813590, "frobenius", "experts:0:256", "pooled_ratio",
               provenance="D")
    b = Metric(0.810601, "frobenius", "experts:0:256", "pooled_ratio",
               provenance="C")
    assert (a - b).value == pytest.approx(0.002989, abs=1e-6)
    # BAR 1 as actually adjudicated: (D - C)/D against a 10% bar.
    assert b.relative_to(a) == pytest.approx(0.003674, abs=1e-5)
    assert adjudicate(Metric(b.relative_to(a), "frobenius_rel_gain",
                             "experts:0:256", "pooled_ratio"),
                      bar_value=0.10, direction="above") == "NO-FIRE"

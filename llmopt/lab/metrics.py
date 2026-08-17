"""Typed metrics: a number that knows what population it describes.

THE INCIDENT THIS EXISTS FOR (2026-08-16, STREAM-WDISTILL-0).
EXEC1 measured operator error at EXPERT 0 ONLY (n=1 of 256). The
audit-repair changed it to a pooled numerator/denominator over all
256 experts. Both numbers were individually valid; both producing
arms were admissible; and their DIFFERENCE was still scientifically
meaningless, because it measures the population change rather than
the repair. The class was caught twice in one day — once in the
receipt (BAR 2 scored on one expert against a bar registered for
the layer) and once, hours after being documented as a house law,
in the analysis prose that was supposed to price the repair.

The lesson that shaped this module: a repository can execute every
floating-point operation correctly and still subtract two numbers
drawn from different populations. So population compatibility is
not a property of either number — it is a RELATION between the two
things being contrasted, and it needs to be first-class.

WHAT THIS ENFORCES
  Metric arithmetic refuses across mismatched population,
  aggregation, or metric identity, raising MetricContractError with
  a reason code drawn from the same vocabulary the incident table
  keys on.
  You may still look at apples versus oranges — deliberately, via
  cross_population_difference() — but that returns a
  DescriptiveContrast, a DIFFERENT TYPE that carries
  adjudicable=False, and adjudicate() refuses it structurally.
  So the guarantee is compositional: you may look, you may not
  accidentally turn the look into evidence for a registered bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Reason codes are shared with the incident table and the bar
# schema, so a refusal and its regression fixture speak one
# vocabulary.
POPULATION_MISMATCH = "metric_population_mismatch"
AGGREGATION_MISMATCH = "metric_aggregation_mismatch"
METRIC_MISMATCH = "metric_identity_mismatch"
NOT_ADJUDICABLE = "contrast_not_adjudicable"
UNIT_MISMATCH = "metric_unit_mismatch"


class MetricContractError(TypeError):
    """Raised when a metric operation violates its contract.

    Carries `reason` so a caller (or a test) can assert on the CLASS
    of the refusal rather than on message text.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class Metric:
    """A scalar that knows its semantic type, population and aggregation.

    population  a string naming WHAT WAS MEASURED OVER, e.g.
                "expert:0" or "experts:0:256". Two metrics with
                different populations are not comparable.
    aggregation how the scalar was formed, e.g. "ratio",
                "pooled_ratio" (summed numerator/denominator),
                "mean_of_ratios". Pooling and averaging estimate
                different quantities; the repair's spectral column
                was a mean of ratios reported under a pooled name.
    """

    value: float
    metric: str
    population: str
    aggregation: str
    unit: str = "relative"
    provenance: str = ""

    def _require_comparable(self, other: "Metric") -> None:
        if not isinstance(other, Metric):
            raise MetricContractError(
                METRIC_MISMATCH, f"{type(other).__name__} is not a Metric")
        if self.metric != other.metric:
            raise MetricContractError(
                METRIC_MISMATCH, f"{self.metric!r} v {other.metric!r}")
        if self.population != other.population:
            raise MetricContractError(
                POPULATION_MISMATCH,
                f"{self.population!r} v {other.population!r} — use "
                "cross_population_difference() if the comparison is "
                "deliberately descriptive")
        if self.aggregation != other.aggregation:
            raise MetricContractError(
                AGGREGATION_MISMATCH,
                f"{self.aggregation!r} v {other.aggregation!r}")
        if self.unit != other.unit:
            raise MetricContractError(
                UNIT_MISMATCH, f"{self.unit!r} v {other.unit!r}")

    def __sub__(self, other: "Metric") -> "Metric":
        self._require_comparable(other)
        return Metric(self.value - other.value, self.metric,
                      self.population, self.aggregation, self.unit,
                      f"({self.provenance} - {other.provenance})")

    def relative_to(self, other: "Metric") -> float:
        """(other - self) / other — the house's 'X% below' form."""
        self._require_comparable(other)
        if other.value == 0:
            raise MetricContractError(METRIC_MISMATCH, "zero denominator")
        return (other.value - self.value) / other.value


@dataclass(frozen=True)
class DescriptiveContrast:
    """A deliberately cross-population comparison. Never adjudicable.

    Exists so that looking at apples versus oranges is possible but
    cannot be laundered into bar evidence: `adjudicable` is False by
    construction and adjudicate() refuses this type outright.
    """

    value: float
    left: Metric
    right: Metric
    purpose: str
    adjudicable: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.purpose:
            raise MetricContractError(
                NOT_ADJUDICABLE,
                "a cross-population contrast must state its purpose")


def cross_population_difference(a: Metric, b: Metric,
                                purpose: str) -> DescriptiveContrast:
    """Difference two metrics ACROSS populations, descriptively.

    The escape hatch is a distinct TYPE, not a flag on a normal
    metric, so nothing downstream can mistake it for evidence.
    """
    if a.metric != b.metric:
        raise MetricContractError(
            METRIC_MISMATCH, f"{a.metric!r} v {b.metric!r}")
    return DescriptiveContrast(a.value - b.value, a, b, purpose)


def adjudicate(observed, bar_value: float, direction: str = "below",
               required_population: str | None = None) -> str:
    """FIRE / NO-FIRE for a registered bar. Refuses non-evidence.

    Returns "FIRE" or "NO-FIRE"; raises MetricContractError when the
    input cannot bear a bar at all — a DescriptiveContrast, or a
    Metric whose population is not the one the bar registered.
    """
    if isinstance(observed, DescriptiveContrast):
        raise MetricContractError(
            NOT_ADJUDICABLE,
            "a DescriptiveContrast is descriptive by construction and "
            "cannot satisfy a registered bar")
    if not isinstance(observed, Metric):
        raise MetricContractError(
            METRIC_MISMATCH, f"{type(observed).__name__} is not a Metric")
    if (required_population is not None
            and observed.population != required_population):
        raise MetricContractError(
            POPULATION_MISMATCH,
            f"bar registered for {required_population!r}, observed "
            f"{observed.population!r}")
    if direction == "below":
        return "FIRE" if observed.value <= bar_value else "NO-FIRE"
    if direction == "above":
        return "FIRE" if observed.value >= bar_value else "NO-FIRE"
    raise MetricContractError(METRIC_MISMATCH, f"direction {direction!r}")

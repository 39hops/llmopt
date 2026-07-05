"""Expert offload with an LRU resident set.

MoE weights dwarf activations, but each token touches only k experts —
so most experts can live on slow storage (CPU RAM, disk) with a small
LRU-cached resident set on the fast device. Hit rate is the whole game:
routing is heavy-tailed in practice, so a resident set far smaller than
E catches most traffic.
"""

from __future__ import annotations

from collections import OrderedDict


class ExpertCache:
    """LRU resident set over a list of expert modules.

    get(i) returns expert i on ``device``, loading (and evicting the
    least-recently-used resident back to ``home``) as needed.
    """

    def __init__(self, experts, capacity: int, device="cpu", home="cpu"):
        assert capacity >= 1
        self.experts = experts
        self.capacity = capacity
        self.device = device
        self.home = home
        self.resident: OrderedDict[int, None] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, i: int):
        if i in self.resident:
            self.hits += 1
            self.resident.move_to_end(i)
        else:
            self.misses += 1
            if len(self.resident) >= self.capacity:
                evicted, _ = self.resident.popitem(last=False)
                self.experts[evicted].to(self.home)
            self.experts[i].to(self.device)
            self.resident[i] = None
        return self.experts[i]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

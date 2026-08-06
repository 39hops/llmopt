"""TENET D3: the budget accountant (spec 2026-08-05-tenet-battery.md
deliverable D3; fixes PINCER R8's booked instrument defect — "my
equal-budget fence was violated by my own design": the peeler's
154,641 sampled tokens rode ON TOP of the 27,053-token forward
budget instead of being traded against it, 5.7x for +1 solve).

Contract (loud-failure): every arm in a matched-budget comparison
draws from ONE BudgetAccountant. Tokens are debited BEFORE the
sample is used; an insufficient balance REFUSES the debit (the
caller must stop, not shrink), refusals are typed+counted, and the
terminal census carries the full per-payer ledger so a booking can
verify the fence arithmetically. An arm that never checks .ok() is
structurally unable to overspend only if it routes all sampling
through charge() — the alternation harness (R1b-micro) must route
EVERY wave through charge(), peeler and forward alike.

Usage:
    acct = BudgetAccountant(total=30_000)
    texts = charge(acct, "peeler", texts)   # [] if refused
    ...
    acct.census()  # terminal record dict; refusals + ledger

Self-test: .venv/bin/python scratch/tenet_d3_budget.py
Unit tests: tests/test_budget_accountant.py
"""


class BudgetAccountant:
    def __init__(self, total: int):
        if total <= 0:
            raise ValueError("budget must be positive")
        self.total = int(total)
        self.spent = 0
        self.ledger: dict[str, int] = {}
        self.refusals: dict[str, int] = {}

    def remaining(self) -> int:
        return self.total - self.spent

    def debit(self, n: int, who: str) -> bool:
        """True and charged, or False and NOTHING charged. A refusal
        is terminal for the payer's current wave — callers must not
        retry with a smaller n (that would turn the fence into a
        soft cap; the R8 lesson is that soft caps drift)."""
        if n < 0:
            raise ValueError(f"negative debit {n} from {who}")
        if self.spent + n > self.total:
            self.refusals[who] = self.refusals.get(who, 0) + 1
            return False
        self.spent += n
        self.ledger[who] = self.ledger.get(who, 0) + n
        return True

    def census(self) -> dict:
        assert self.spent == sum(self.ledger.values()), \
            "ledger does not sum to spent — accountant corrupted"
        return {"total": self.total, "spent": self.spent,
                "remaining": self.remaining(),
                "ledger": dict(self.ledger),
                "refusals": dict(self.refusals)}


def charge(acct: BudgetAccountant, who: str, texts, tok_counts=None):
    """Charge a sampled wave; return the texts, or [] if refused.
    Cost = total sampled tokens across ALL candidates in the wave
    (tok_counts if provided, else len(text) as a character proxy —
    pass real token counts in any registered arm)."""
    cost = (sum(tok_counts) if tok_counts is not None
            else sum(len(t) for t in texts))
    if not acct.debit(cost, who):
        return []
    return texts


if __name__ == "__main__":
    acct = BudgetAccountant(total=100)
    assert charge(acct, "fwd", ["aaaa", "bbbb"]) != []      # 8
    assert charge(acct, "peel", ["c" * 80]) != []           # 88
    assert charge(acct, "peel", ["d" * 20]) == []           # refused
    assert acct.remaining() == 12
    c = acct.census()
    assert c["ledger"] == {"fwd": 8, "peel": 80}
    assert c["refusals"] == {"peel": 1}
    print("[d3] self-test OK:", c)

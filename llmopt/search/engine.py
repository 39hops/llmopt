"""The measured-best engines, as one import (integration of the
2026-07-06..08 racing results — see docs/RESULTS.md).

    from llmopt.search.engine import solve
    result = solve(sp.Integral(x * sp.cos(x), x))

Default = the zero-neural-network configuration that scored 316/360
(87.8%) on the held-out races: rule-bigram Markov prior (top-3
pruning), width-2 beam, sampled verification (winning paths always
fully verified). The 337/360 record config additionally needs the
LoRA'd 0.5B for entropy-gated adaptive k — pass its score_fn via
`llm_score_fn` to enable it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import sympy as sp

from llmopt.search.derivation import SearchResult, State, beam_search
from llmopt.search.proposer import entropy_k, make_scoring_proposer

_PRIOR_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "markov_prior.json"
_POLICY_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "syndrome_policy.pt"


class SyndromePolicy:
    """State-aware rule ranker (the qLDPC framing, production-grade):
    rank kids by a net reading the state's features + rule-fire
    syndrome bits (free — the search already evaluated every rule to
    produce kids) + previous-rule one-hot. Adopted 2026-07-10 after
    the brain races: beats markov 98v96 at -36% wall on fresh mixed,
    ties fresh L5 (76/76), best pure arm on the L5-heavy router race
    (124 v 121), and solves the nested-trig class markov cannot reach
    at any propose_k. Trained by imitation + 2 DAgger rounds (round 3
    checkpoint; round 4's pure-L5 mix REGRESSED — see RESULTS)."""

    def __init__(self, net, payload):
        self.net = net
        self.payload = payload

    @classmethod
    def load(cls, path: Path = _POLICY_PATH) -> "SyndromePolicy":
        import torch
        p = torch.load(path, weights_only=False)
        net = torch.nn.Sequential(
            torch.nn.Linear(len(p["mu"]), 96), torch.nn.ReLU(),
            torch.nn.Linear(96, 96), torch.nn.ReLU(),
            torch.nn.Linear(96, len(p["vocab"])))
        net.load_state_dict(p["state_dict"])
        net.eval()
        return cls(net, p)

    def proposer(self):
        import torch

        from llmopt.search.features import featurize
        from llmopt.search.rules import INT_RULES

        p = self.payload
        vi = {r: i for i, r in enumerate(p["vocab"])}
        pidx = {r: i for i, r in enumerate(p["prevs"])}
        # training-time syndrome vocab, PINNED in the checkpoint:
        # reading the live INT_RULES breaks every trained net the
        # moment a rule is added (measured 2026-07-11: i_heurisch
        # grew the vector 36->37, tensor-shape crash in production)
        rule_names = p.get("synd_rules", [n for n, _ in INT_RULES])

        def prop(state: State, kids):
            if not kids:
                return kids
            prev = (state.history[-1].split("@")[0] if state.history
                    else "<start>")
            oh = [0.0] * len(p["prevs"])
            if prev in pidx:
                oh[pidx[prev]] = 1.0
            kid_rules = {lab.split("@")[0] for lab, _ in kids}
            synd = [1.0 if n in kid_rules else 0.0 for n in rule_names]
            feats = featurize(state.expr) + synd + oh
            x = (torch.tensor([feats], dtype=torch.float32)
                 - p["mu"]) / p["sd"]
            with torch.no_grad():
                logits = self.net(x)[0]

            def score(lab):
                r = lab.split("@")[0]
                # unknown rule (added after training) gets a fair
                # trial at the mean logit — the markov prior's
                # trial-mass lesson, repeated in its successor
                # (measured 2026-07-11: i_heurisch scored -50, cut
                # from top-3 on every policy-routed problem)
                return (logits[vi[r]].item() if r in vi
                        else logits.mean().item())

            return sorted(kids, key=lambda c: -score(c[0]))

        return prop


class MarkovPrior:
    """Rule-bigram counts from verifier-approved winning paths. The
    measured result behind it: ties the fine-tuned 0.5B at top-3
    offline, beats it in-search (293 vs 288) at zero inference cost."""

    def __init__(self, unigram: dict, bigram: dict):
        self.unigram = unigram
        self.bigram = bigram

    @classmethod
    def load(cls, path: Path = _PRIOR_PATH) -> "MarkovPrior":
        d = json.loads(Path(path).read_text())
        return cls(d["unigram"], d["bigram"])

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "MarkovPrior":
        unigram: dict = {}
        bigram: dict = {}
        prev = None
        for row in rows:
            chosen = row["moves"][row["answer"]].split("@")[0]
            unigram[chosen] = unigram.get(chosen, 0) + 1
            if prev is not None:
                bigram.setdefault(prev, {})
                bigram[prev][chosen] = bigram[prev].get(chosen, 0) + 1
            prev = chosen
        return cls(unigram, bigram)

    def proposer(self):
        # Unseen-rule trial mass: rules the prior has never seen score
        # 0.5*median OUTRIGHT (not 0.01*median). The 0.01 factor was
        # measured insufficient (2026-07-10 five-ply trace: i_log_power
        # fired every ply, scored 0.74 vs bigram 34+, never cracked
        # top-3). Full trial mass is safe because rules only fire on
        # matching structure — the fire itself is the evidence. Do NOT
        # fix this by re-mining the prior instead: measured twice to
        # regress (dilution; L5 89.6% -> 73.1%, and 478e269).
        med = sorted(self.unigram.values())[len(self.unigram) // 2] \
            if self.unigram else 1

        def prop(state: State, children):
            prev = (state.history[-1].split("@")[0]
                    if state.history else None)
            table = self.bigram.get(prev) if prev else None

            def s(name: str) -> float:
                r = name.split("@")[0]
                if r not in self.unigram:
                    return 0.5 * med
                return ((table.get(r, 0) if table else 0)
                        + 0.01 * self.unigram[r])

            return sorted(children, key=lambda c: -s(c[0]))

        return prop


def solve(expr: sp.Expr, *, budget: int = 200,
          prior: MarkovPrior | None = None,
          llm_score_fn: Callable | None = None,
          use_macros: bool = True, magic: bool = True,
          ply_hook: Callable | None = None) -> SearchResult:
    """Solve with the measured-best configuration.

    Without llm_score_fn: markov3 @ width 2 (334/360-class with the
    autopsy rules, zero NN). With it: hybrid markov-ranks +
    entropy-gated k at T=0.1 (349/360-class record config).
    magic=True prunes Risch-certified dead states (RESULTS: +1
    replicated, 71 cuts at int L4, provably zero false positives)."""
    state_filter = None
    if magic:
        from llmopt.search.magic import is_dead
        state_filter = lambda s: not is_dead(s)  # noqa: E731
    if llm_score_fn is not None:
        return beam_search(
            expr, width=2, max_plies=24, max_nodes=budget,
            proposer=make_scoring_proposer(llm_score_fn),
            propose_k=entropy_k(1, 3, temperature=0.1),
            use_macros=use_macros, verify_p=0.1,
            state_filter=state_filter, ply_hook=ply_hook)
    # width 3 since 2026-07-10 node-cost round 2: the verify fix
    # (doit(integrals=False)) made nodes cheap enough to widen —
    # raced L5 238/249 (95.6%, from 223) at 3.5x LESS wall; L3 +1,
    # L4 tied on 60-problem samples. Width 2 lost composition-fragile
    # chains whose every rule existed (9-problem residue -> 8 solved).
    # Brain: dispatcher-net routed (2026-07-10/11, "verified speed is
    # intelligence"). Route history, three OOS bands: threshold-5.5
    # on estimator cost beat both pures twice (141/150@167s,
    # 126/130@164s); dispatcher v2 (features + root rule-fire
    # syndromes -> which brain; disagreement-oversampled labels,
    # disagreement acc 0.883) then TIED the threshold's solves at 18%
    # less wall on band 3 (144/150 @ 344s vs 417s) — adopted by the
    # FA-Law tiebreak. Fallback chain: dispatcher -> threshold ->
    # policy-only -> markov, per available checkpoints; explicit
    # prior= forces markov.
    if prior is None and _POLICY_PATH.exists():
        pol = SyndromePolicy.load().proposer()
        proposer = pol
        import torch
        # v3 preferred (2026-07-11 race: 114/120 @ 370s vs markov
        # 114 @ 644s, policy 112 @ 594s, v2 112 @ 637s — ties best
        # solves at 43% less wall; only router trained on L6/L7 and
        # the current brains); v2 fallback for older checkouts
        disp_path = _POLICY_PATH.parent / "dispatcher_v3.pt"
        if not disp_path.exists():
            disp_path = _POLICY_PATH.parent / "dispatcher_v2.pt"
        est_path = _POLICY_PATH.parent / "magic_estimator_v5.pt"
        if disp_path.exists():
            from llmopt.search.features import featurize
            from llmopt.search.rules import INT_RULES
            dp = torch.load(disp_path, weights_only=False)
            dnet = torch.nn.Sequential(
                torch.nn.Linear(len(dp["mu"]), 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 1))
            dnet.load_state_dict(dp["state_dict"])
            dnet.eval()
            node = max(expr.atoms(sp.Integral), key=sp.count_ops,
                       default=None)
            # pinned training-time syndrome vocab (see SyndromePolicy).
            # TIMEBOXED rule probes: i_heurisch runs sp.integrate and
            # a monster root can hang it OUTSIDE any search alarm
            # (measured: routing froze 73min at 99.7% CPU before the
            # search began)
            from llmopt.search.derivation import _timeboxed
            by_name = dict(INT_RULES)
            synd = []
            for rname in dp.get("synd_rules", [n for n, _ in INT_RULES]):
                rule = by_name.get(rname)
                fired = (_timeboxed(rule, node, default=[])
                         if node is not None and rule else [])
                synd.append(1.0 if fired else 0.0)
            f = torch.tensor([featurize(expr) + synd],
                             dtype=torch.float32)
            with torch.no_grad():
                logit = dnet((f - dp["mu"]) / dp["sd"]).item()
            if logit <= 0:
                proposer = MarkovPrior.load().proposer()
        elif est_path.exists():
            from llmopt.search.features import featurize
            pay = torch.load(est_path, weights_only=False)
            f = torch.tensor([featurize(expr)], dtype=torch.float32)
            import sys
            sys.path.insert(0, str(_POLICY_PATH.parents[1] / "scripts"))
            from train_magic_estimator import Estimator
            est = Estimator(d_in=len(pay["mu"]))
            est.load_state_dict(pay["state_dict"])
            est.eval()
            with torch.no_grad():
                _, cost = est((f - pay["mu"]) / pay["sd"])
            if cost.item() > 5.5:
                proposer = MarkovPrior.load().proposer()
    else:
        proposer = (prior or MarkovPrior.load()).proposer()
    return beam_search(
        expr, width=3, max_plies=24, max_nodes=budget,
        proposer=proposer, propose_k=3,
        use_macros=use_macros, verify_p=0.1,
        state_filter=state_filter, ply_hook=ply_hook)

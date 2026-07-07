"""Markov bigram move-prior (Artin's question: 'can't Markov chains
predict certain things?'). Rule-name bigram from winning paths: after
rule R at the parent ply, which rule tends to win next? Zero neural
nets, a dictionary of counts — the embarrassingly-cheap control for
the O1 distillation question: if this matches the LLM proposer's move
accuracy, the 0.5B is mostly memorizing rule GRAMMAR, not reading
expressions.

  python scripts/markov_prior.py
Prints top-1/top-3 move accuracy on data/proposer_eval.jsonl next to
the measured baselines (LLM tuned: 67.3%/99.7%; base: 50.0%/95.7%).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict


def rule_of(label: str) -> str:
    return label.split("@")[0]


def main() -> None:
    train = [json.loads(l) for l in open("data/proposer_train.jsonl")]
    evals = [json.loads(l) for l in open("data/proposer_eval.jsonl")][:300]

    # Winning paths arrive as consecutive rows per problem; reconstruct
    # bigrams by pairing each row's chosen rule with the next row's.
    # Also learn the unconditional prior (for path starts / unseen).
    unigram: Counter = Counter()
    bigram: dict[str, Counter] = defaultdict(Counter)
    prev_rule = None
    for row in train:
        chosen = rule_of(row["moves"][row["answer"]])
        unigram[chosen] += 1
        # heuristic path-boundary: state length resets small at new roots;
        # conservative: only count a bigram when the previous row's chosen
        # move appears among this row's parent context is unknowable here,
        # so accept consecutive-row pairing (rows are path-ordered by
        # construction in gen_proposer_data/path_rows).
        if prev_rule is not None:
            bigram[prev_rule][chosen] += 1
        prev_rule = chosen

    def score(prev: str | None, labels: list[str]) -> list[float]:
        table = bigram.get(prev) if prev else None
        out = []
        for lab in labels:
            r = rule_of(lab)
            b = (table.get(r, 0) if table else 0)
            out.append(b + 0.01 * unigram.get(r, 0))
        return out

    # evaluate: eval rows are also path-ordered; track prev per row chain
    hits1 = hits3 = 0
    prev = None
    for row in evals:
        s = score(prev, row["moves"])
        order = sorted(range(len(s)), key=lambda i: -s[i])
        hits1 += row["answer"] == order[0]
        hits3 += row["answer"] in order[:3]
        prev = rule_of(row["moves"][row["answer"]])
    n = len(evals)
    print(f"markov bigram prior: top1={hits1 / n:.1%} top3={hits3 / n:.1%}")
    print("measured baselines:  LLM tuned 67.3%/99.7%; LLM base 50.0%/95.7%")
    print(f"(unigram alone top rule: {unigram.most_common(3)})")


if __name__ == "__main__":
    main()

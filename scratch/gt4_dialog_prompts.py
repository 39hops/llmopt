"""MOE-GT-4 corpus: the SECOND verbal corpus (dialogue/QA register).

Mirrors the gt3 prose corpus geometry exactly — 4 templates x 20
topics = 80 prompts — so corpus shape is not a confound; only the
verbal REGISTER changes (expository -> conversational). Topics are
DISJOINT from the 20 prose topics (registered fence: shared content
words could inflate Jaccard(prose, dialog) without a shared branch).
Zero symbolic content. Deterministic: literal lists + string-seeded
shuffle (house convention).

Writes checkpoints/gt4_dialog_prompts.json ({prompt, kind, level}
rows, kind="dialog", level=0 — the gt3_probe_arm0 corpus format).
"""

import json
import random

TEMPLATES = [
    "Write a short dialogue between two friends talking about {t}.",
    "Answer conversationally, as if chatting with a friend: "
    "what do you think about {t}?",
    "Write a brief back-and-forth conversation between two neighbors "
    "about {t}.",
    "Someone asks you over coffee about {t}. Give a friendly, "
    "chatty reply.",
]

TOPICS = [
    "planning a surprise birthday party",
    "whether to adopt a rescue dog",
    "moving to a new city for a job",
    "learning to play the guitar as an adult",
    "training for a first 5k race",
    "redecorating a small living room",
    "picking a book for a book club",
    "dealing with a noisy upstairs neighbor",
    "planning a weekend camping trip",
    "trying a new restaurant downtown",
    "keeping houseplants alive",
    "teaching a teenager to drive",
    "hosting relatives for the holidays",
    "choosing a wedding gift for a coworker",
    "starting a community garden plot",
    "switching to cycling for the daily commute",
    "organizing a neighborhood garage sale",
    "taking up watercolor painting",
    "finding a reliable babysitter",
    "planning a road trip along the coast",
]


def main():
    rows = [{"prompt": t.format(t=topic), "kind": "dialog", "level": 0}
            for t in TEMPLATES for topic in TOPICS]
    assert len(rows) == 80 and len({r["prompt"] for r in rows}) == 80
    prose = {r["prompt"]
             for r in json.load(open("checkpoints/gt3_prose_prompts.json"))}
    assert not prose & {r["prompt"] for r in rows}
    random.Random("gt4-dialog-0").shuffle(rows)
    out = "checkpoints/gt4_dialog_prompts.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

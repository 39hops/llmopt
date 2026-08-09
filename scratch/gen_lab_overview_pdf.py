"""LinkedIn Featured collateral: 3-page lab-overview PDF composed
ENTIRELY from committed docs/assets figures + ledger-verified
numbers (receipt-checked 2026-08-08 against RESULTS/FINDINGS/
README for the LinkedIn pass). Layout only — no generated imagery.

Usage: .venv/bin/python scratch/gen_lab_overview_pdf.py
Output: figs/2026-08-08/lab-overview.pdf (untracked, personal
collateral — figs/ convention).
"""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

OUT = Path("figs/2026-08-08/lab-overview.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = letter
M = 54  # margin
INK = (0.13, 0.12, 0.18)
DIM = (0.42, 0.41, 0.48)
ACC = (0.45, 0.30, 0.75)


def head(c, title, sub=None):
    c.setFillColorRGB(*ACC)
    c.rect(M, H - 58, 64, 4, fill=1, stroke=0)
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 19)
    c.drawString(M, H - 86, title)
    if sub:
        c.setFillColorRGB(*DIM)
        c.setFont("Helvetica", 10.5)
        c.drawString(M, H - 103, sub)


def para(c, x, y, width, lines, size=10, leading=14.5, color=INK,
         font="Helvetica"):
    c.setFont(font, size)
    c.setFillColorRGB(*color)
    import textwrap
    yy = y
    for block in lines:
        for ln in textwrap.wrap(block, width=width) or [""]:
            c.drawString(x, yy, ln)
            yy -= leading
        yy -= leading * 0.35
    return yy


def image(c, path, y_top, max_h, caption=None):
    img = ImageReader(path)
    iw, ih = img.getSize()
    w = W - 2 * M
    h = w * ih / iw
    if h > max_h:
        h, w = max_h, max_h * iw / ih
    x = (W - w) / 2
    y = y_top - h
    c.drawImage(img, x, y, w, h, preserveAspectRatio=True)
    if caption:
        para(c, M, y - 16, 95, [caption], size=8.5, leading=11,
             color=DIM, font="Helvetica-Oblique")
    return y


c = canvas.Canvas(str(OUT), pagesize=letter)

# ---- page 1: who/what + the crystal ----
head(c, "llmopt + axiom  |  two measured research labs",
     "Artin Azizi  -  github.com/39hops/llmopt")
y = para(c, M, H - 136, 95, [
    "Open research lab for closed-system mathematical models and "
    "mixture-of-experts behavior, with chess-engine-style "
    "evaluation discipline: pre-registered experiments, paired "
    "seeds, oracle-verified scoring, and honest nulls kept in an "
    "append-only ledger.",
    "Paired with axiom, an independent C++ derivation engine: the "
    "two labs verify each other's artifacts, so \"reproduced\" "
    "means a second implementation agrees, not a rerun of the "
    "same code.",
])
image(c, "docs/assets/neurons-19m-zoom.png", y - 8, y - 8 - 96,
      caption="One layer of a 0.5B model during a verified 2.4x "
              "reinforcement-learning capability climb, each line "
              "one neuron's movement drawn at 60x magnification: "
              "every neuron nudged, none uprooted. The entire "
              "climb wrote ~6% of one supervised run's weight "
              "movement (delta-W 4.0 vs 61-87), with "
              "representations near-identical before and after "
              "(CKA 0.9998). RL edits the policy, not the "
              "representation.")
c.showPage()

# ---- page 2: the MoE results ----
head(c, "Expert identity beats aggregate summaries",
     "30B-class mixture-of-experts, 120-item oracle-scored gates")
y1 = image(c, "docs/assets/gt1-crest-small-multiples.png",
           H - 122, 300,
           caption="Keeping the top 45.3% of experts by math "
                   "demand beats the full model on the math gate, "
                   "6/6 paired seeds (+14.7 pooled). The same "
                   "recipe inverts on a mechanics gate (-59 "
                   "pooled): domain-specific identity, not generic "
                   "sparsity. Regenerates from a committed script; "
                   "provenance footer in-image.")
y2 = image(c, "docs/assets/identity-crest-fresh-seeds.png",
           y1 - 58, y1 - 58 - 140,
           caption="Decomposition: swap-derived keep-set (+53), "
                   "deletion of 80 named interfering experts "
                   "(+55), rank-matched control (+28), each 3/3 "
                   "at fresh seeds. Capability follows WHICH "
                   "experts are kept, not how many.")
c.showPage()

# ---- page 3: cross-lab verification + the climb ----
head(c, "Two labs, one verification loop",
     "Every headline number below is booked in the public ledger")
y = para(c, M, H - 136, 95, [
    "360/360 on held-out, sympy-oracle-verified calculus (up from "
    "73.6%), every gain tied to a named component.",
    "Cross-lab replay: 50/50 trajectories token-identical between "
    "llmopt's Python engine and axiom's C++ leg.",
    "Certified row factory: 167/167 axiom-emitted training rows "
    "pass llmopt's independent oracle, schema-exact; emission "
    "audits 5-for-5 clean (latest: 0 contaminated rows in "
    "145,011).",
    "21,614/21,914 kernel-certified Lean 4 certificates (98.63%, "
        "closed failure taxonomy); neural-net "
    "interchange format (AXNN) at 20/20 cross-lab parity.",
    "Exactness work: integer and deterministic compute paths with "
    "digest and trajectory checks, so agreement is bit-level "
    "where the instrument allows it.",
])
image(c, "docs/assets/neurons-qwen-vs-19m.png", y - 10,
      y - 10 - 170,
      caption="Same lens, two diets: web-pretrained 0.5B (dense "
              "isotropic cloud) vs the closed-system 19M math "
              "native (sparse ring structure). You can see the "
              "training diet in the weights.")
para(c, M, 72, 95, [
    "Philosophy: small, fully owned training paths beat "
    "un-auditable scale for causal claims; nulls and "
    "between-threshold outcomes are first-class results.",
], size=9.5, color=DIM, font="Helvetica-Oblique")
c.save()
print(f"wrote {OUT}")

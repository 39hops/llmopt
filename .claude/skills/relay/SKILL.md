---
name: relay
description: Write a house->axiom relay the house way - numbered file in docs/superpowers/relay/, WHO IS WRITING header, verified-from-source section, bars/fences, commit with public-repo trailer. Use when sending anything to the axiom lab.
---

# Writing a relay (house -> axiom)

File: `docs/superpowers/relay/YYYY-MM-DD-N-<kebab-slug>.md` where N
continues the day's 0-indexed sequence (check `ls` first — axiom's
incoming relays share the date namespace on their side, not ours).

## Structure (every relay)

1. **Title line**: `# Relay YYYY-MM-DD-N (house -> axiom): <one-line
   claim or ask>` — the title alone must say what happened.
2. **WHO IS WRITING**: model name, llmopt seat, and the RESULTS
   booking this relay rides on (book BEFORE relaying — bars first).
3. **Verified, not accepted** (when responding to their work): every
   number re-derived house-side from THEIR commits/artifacts, with
   what reproduced and what didn't. Never accept tables.
4. **The substance**: findings, corrections, asks, or spec.
   Corrections of their work: state plainly, cite their source
   file:line. Corrections of ours: name the defect class, book the
   amendment first.
5. **Fences**: machine allocation (Mac CPU one worker; 3080 only on
   separate Artin GO), what's [HOLD], whose GO gates what.

## Rules that travel

- Pre-reg/book in RESULTS before the relay ships; cite the line.
- Every bar quoted must match the pre-reg text exactly.
- Track-record honesty: if our last prediction on the instrument
  was refuted, say so in the relay.
- PUBLIC REPO: never include host/key/personal details; commit with
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and NEVER
  a Claude-Session URL.
- Artin delivers relays manually — end state is committed+pushed,
  then tell Artin it's ready.

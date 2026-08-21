---
name: rung
description: Pre-register an experiment (bars, registered prior, fences), scaffold its driver, launch it on the named machine, and arm a watcher. The front half of the cycle; /book is the back half.
disable-model-invocation: true
---

# Running a rung (pre-reg -> driver -> launch -> watch)

`/book` handles a landed result. This handles everything before it.
Do all of these in order; do not launch anything before the pre-reg
is committed.

## 1. Check the ledger first

`.venv/bin/python scripts/results_query.py --thread <name>` and grep
RESULTS. The idea has often been run, nulled, or pre-registered
already. Say so if it has.

## 2. Write the pre-reg INTO docs/RESULTS.md, then commit it

Heading: `## PRE-REG <NAME>: <one-line question> (<date>, <machine>)`

Every pre-reg carries:
- **Instrument**: exact recipe, arch, seeds, diet, knobs. Enough that
  someone else could rebuild the arm from this paragraph alone.
- **BARS**: numbered, each a threshold that can FIRE or NO-FIRE, each
  written so the measured number is compared against a number that is
  already on the page. A bar you can reword after seeing data is not
  a bar.
- **REFUTED-IF**: what result kills the hypothesis. If you cannot
  write one, the rung is not falsifiable yet — fix that first.
- **REGISTERED PRIOR**: what the house predicts, on the record,
  before the run. Being wrong in public is the point; track record
  is an instrument.
- **FENCES**: device, seed count, family-only comparisons, diet
  confounds, window/GO limits, what books NOT-RUN if the wall hits.
- **A NO-OP PRECONDITION, whenever the rung rebuilds or wraps the
  model** (virtual tokens, vocab padding, patched forwards, adapter
  shells). Register a cell that reproduces the stock reading EXACTLY,
  cell for cell, and gate every treatment cell behind it. A
  perturbation control is not a substitute: SOFT-PROMPT-1 had one, it
  fired correctly, and the rung still cost a full run because the
  harness was already off before any prefix existed. Bit-identical
  weights do not imply an identical reading — the sampler sees the
  harness too.

Commit the pre-reg BEFORE the run fires. A pre-reg committed after
receipts exist is not a pre-registration.

## 3. Scaffold the driver in scratch/

FIRST: `grep scripts/INDEX.md` and `docs/CODEMAP.md` for an existing
driver or instrument. Do not rewrite what exists, and do not fork a
frozen family — extend the `llmopt/lab/` module instead. CODEMAP
gives each file's class (frozen evidence / adopted / disposable).

Then name it after the rung. NEW DRIVERS SOURCE THE HARNESS:
`. "$(dirname "$0")/lib/driver.sh"` (scratch/lib/driver.sh) — it
carries strict mode with pipefail, `llmopt_cd`, `cuda_preamble`,
`mark_done` (marker on success only), and `wait_for` (pgrep pattern
must never match the launcher). Frozen drivers are never retrofitted.
Requirements the harness encodes, learned the hard way:

- `set -eo pipefail`, always. `tee` reports ITS OWN exit code, so
  without pipefail a training process that dies mid-epoch leaves the
  driver running happily into its gate step against a checkpoint that
  was never written — and the job records success. That happened on
  2026-08-11 and rjob logged rc=0 for a completely failed run.
- **Stream partial results**: gate and print each arm as it lands, so
  a wall-kill still leaves bookable cells. A driver that only reports
  at the end loses everything it did.
- **Unique output paths per arm x seed.** Never write into a path a
  booked verdict cites.
- Write receipts under `logs/<rung>/`, one file per arm.

## 3.5 Qualify the software (`/qualify`) — BEFORE launch

If the driver is new or copy-modified and the run costs >10 min,
>1 GiB, or a full model sweep: run the `/qualify` ladder first
(static checks -> golden fixtures -> parity -> resource preflight
-> mechanism-complete smoke). A run must never be the first test
of its own code — three launches died this way on 2026-08-17
alone. Full suite green (redirected rc, never piped) and
generated docs current are part of "qualified".

## 4. Launch — the signatures that bite

Record `git rev-parse HEAD` at launch; it becomes the entry's
`code_commit` at booking.

**Mac** (jobs land in `jobs/<id>.rc`, `jobs/<id>.log`):
```bash
RJOB_LOCAL=1 .venv/bin/python scripts/rjob.py launch <id> 'bash scratch/<driver>.sh'
```
The command is ONE quoted string. No `--` separator (returns 127);
multiple bare args silently keep only the first token.

**3080** — sync the remote checkout FIRST, then launch:
```bash
scratch/wsl.sh run 'cd ~/code/llmopt && git pull --ff-only -q && mkdir -p logs/<rung> && echo synced'
scratch/wsl.sh launch 'bash scratch/<driver>.sh' logs/<rung>/driver.log logs/<rung>.DONE
```
Three positional args: command, logfile, marker. Pre-make the log
directory — the redirect opens before the script runs, so a missing
directory kills the job instantly and silently. The marker fires on
SUCCESS only.

Then verify it is actually alive — a launch that returns "launched"
has proved nothing:
```bash
scratch/wsl.sh run 'pgrep -af <driver> | grep -v pgrep; tail -1 logs/<rung>/*birth*.log'
```

## 5. Arm a watcher

```bash
until scratch/wsl.sh run 'ls logs/<rung>.DONE' 2>/dev/null; do sleep 180; done
```
as a background Bash call. Dump ALWAYS-READABLE artifacts
(qualification receipts, rc, markers, registered pre-read
censuses) on the same line so the notification carries them —
but SEALED treatment values are gated MECHANICALLY on the run's
rc/qualification inside the watcher command itself, and a mixed
log interleaving both classes is never tailed. Full ritual and
the visibility classes: the `/watch` skill (earned 2026-08-21,
EX6-MED run 2 — an unconditional dump unblinded a
qualification-failed run permanently).

**A watcher watches; it does not launch.** Before arming one, confirm
two things, or it will poll happily forever:

- **Something wrote the marker.** `wsl.sh launch` appends
  `&& echo DONE > <marker>` itself, but a driver run any other way
  must `touch` its own marker on success. On 2026-08-11 a watcher
  polled `logs/softprompt1.DONE` for two hours against a driver whose
  last line was a `tee` — the marker had no author.
- **Something launched the driver.** A chained driver
  (`while [ ! -f logs/<other>.DONE ]; do sleep 300; done`) still has
  to be started by someone. That same night the chain was written,
  committed, and never launched; the first rung finished, the marker
  appeared, and nothing was waiting on it. `pgrep -af <driver>` is
  the check, and "launched" in a tool result is not.

Chaining rung B on rung A's marker is fine — but launch B explicitly
at arm time, and give B its own marker on success.

## Machine rules

- The Mac is the lab's; the 3080 runs on Artin's schedule and needs a
  GO outside its window. Both are Artin's own computers on his home
  network; `wsl.sh` is the job runner between them.
- Never two 30B-class models resident on the Mac at once.
- Cross-device gate comparisons are forbidden. Arms that must be
  compared run on ONE machine, always.
- `[HOLD]` means it waits for an explicit GO, never for an inference
  from context.

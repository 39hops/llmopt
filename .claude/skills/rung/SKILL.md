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

Commit the pre-reg BEFORE the run fires. A pre-reg committed after
receipts exist is not a pre-registration.

## 3. Scaffold the driver in scratch/

Name it after the rung. Requirements learned the hard way:

- `set -e`, and if you use `tee`, remember it masks failures — either
  `set -o pipefail` or state in the pre-reg that every arm log must
  be checked for its success line before the floors are trusted.
- **Stream partial results**: gate and print each arm as it lands, so
  a wall-kill still leaves bookable cells. A driver that only reports
  at the end loses everything it did.
- **Unique output paths per arm x seed.** Never write into a path a
  booked verdict cites.
- Write receipts under `logs/<rung>/`, one file per arm.

## 4. Launch — the signatures that bite

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
as a background Bash call, with the receipt dump on the same line so
the notification carries the numbers.

## Machine rules

- The Mac is the lab's; the 3080 runs on Artin's schedule and needs a
  GO outside its window. Both are Artin's own computers on his home
  network; `wsl.sh` is the job runner between them.
- Never two 30B-class models resident on the Mac at once.
- Cross-device gate comparisons are forbidden. Arms that must be
  compared run on ONE machine, always.
- `[HOLD]` means it waits for an explicit GO, never for an inference
  from context.

#!/bin/bash
# EX5-LAYERMATCH-0 treatment driver (Mac): 8 arms x 3 fresh seeds,
# one gt7_run invocation per seed (one 30B load each), rows stream
# into logs/ex5/ex5.jsonl as they land so a wall-kill leaves
# bookable cells. Pre-reg: docs/preregs/ex5-layermatch-0.json.
# Runs ONLY after the seed-1001 qualification matched cell-exact.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd

# The frozen gt7_run.py appends unconditionally; a relaunch would
# silently double the booked rows (receipt-audit finding, added
# post-booking so the as-run driver sha in the observations doc
# stays truthful — the run that produced the booked rows predates
# this guard).
[ -f logs/ex5/ex5.jsonl ] && {
    echo "REFUSING: logs/ex5/ex5.jsonl exists (booked receipts)" >&2
    exit 1
}

ARMSET="ex1_full128,ex3_del_invp,ex5_del_rank0,ex5_del_rank1,ex5_del_rank2,ex5_del_layer0,ex5_del_layer1,ex5_del_layer2"

for SEED in 4001 5002 6003; do
    echo "[ex5] === seed $SEED ===" >&2
    ARMS="$ARMSET" SEED="$SEED" N_EVAL=120 PERPROB=1 \
        LOG=logs/ex5/ex5.jsonl \
        PERPROB_LOG=logs/ex5/ex5_perprob.jsonl \
        ANSWERS_LOG=logs/ex5/ex5_answers.jsonl \
        .venv/bin/python scratch/gt7_run.py
done
mark_done logs/ex5/ex5.DONE $?

#!/bin/bash
# P4 device leg (GO 2026-08-01): reproduce every pinned gravmoe
# trajectory sha on the 3080 box, all arms in parallel (CPU int64
# battery; 1 torch thread per arm, 15 arms / 16 cores). Verifies
# each FINAL sha against the Mac pins and prints a PASS/FAIL table.
set -u
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DISABLE_NATIVE_JIT=1
unset ANSWER_ONLY
mkdir -p logs/p4

# arm-name | env | expected FINAL trajectory sha (Mac pins)
ARMS=(
  "A0|COND=0|6fffa718f9c7b2c07f2196a4ce079a705517b229baa71b853896f0cda8128faf"
  "A1|COND=0 LN=1 LD=16|1ad5f466aa9dab17540cae26358c4ad50749e9ae8ea4ad1faa71f4b4d2ed3ec1"
  "A2|COND=0 LN=1 LD=4|23c154e6a31daef758230c3780d55082a9be871b734e0844008514760b652ff1"
  "A3|COND=0 LN=1 LD=1|300e61ad2cd621f8c7c2f89cbce3d5f2cafd7397233d9b05ed571a6fe9bd8cab"
  "CA0|COND=1|0ad9da9476d8be45a6ffe28cd95702edabf7d41124c36efa454bf22777f08c2d"
  "CA1|COND=1 LN=1 LD=16|a328188cc7f46ebf2cacea1d6b23bd5a0e0b7f7acc58ed61119018d0f9dc21bc"
  "CA2|COND=1 LN=1 LD=4|35396b368e2f42b781f39167696ae21e60b83c2d3c0e1d483608ea46548aaa76"
  "CA3|COND=1 LN=1 LD=1|4b98a6ef05bb45f7ad3a7e5a50f32bb63b736060aa64b7763369c2ef8dc476e0"
  "GA0|GATE=1 COND=1|2b29bd4aa29bc4fb4ac1ea76084dd4c88e7d93084f27945df0c48c07fae407b1"
  "GA2|GATE=1 COND=1 LN=1 LD=4|66d8f8799f85599b05ac5cf2dd44dd12e88010ba835304625f9d6bed5babb9fc"
  "GA3|GATE=1 COND=1 LN=1 LD=1|919b83476dc9b84e3791821bd884933f88a57e4e261cd7d59bfeab4960412ce0"
  "RB1|COND=1 QK=1|c6766da235cf0b76be20035b893cb41fd0a2f8dbbc6339c96e8527ce2cb3f65f"
  "RB3|COND=1 QK=1 TAU=1|6968b583f440405f7da819ece57f924ffb5fd59dcc228dcf27c1d349b2ecd29d"
  "RB1S16|COND=1 QK=1 SHIFT=16|14981553e6cbebe11f9625fc7b4405dd73ffb9fc5060d8161b57f076ce492ee4"
  "GRB1|GATE=1 COND=1 QK=1|1fcfd187873d980c7c082a56c0f380ce2c40a859eab1e8a0c9dcf6baa4853eca"
  "S1|GATE=1 COND=1 SS=1|e1b633a965171f16ca58d17fe8c597ffbe6362ae2dc6ed7b6f58ec0ed69c6087"
)

for a in "${ARMS[@]}"; do
  IFS='|' read -r name env sha <<< "$a"
  echo "launch $name ($env)"
  env $env $PY scratch/detbwd_gravmoe.py > "logs/p4/$name.log" 2>&1 &
done
wait

echo "=== P4 VERIFICATION TABLE ==="
fail=0
for a in "${ARMS[@]}"; do
  IFS='|' read -r name env sha <<< "$a"
  got=$(grep "FINAL trajectory sha" "logs/p4/$name.log" | awk '{print $NF}')
  if [[ "$got" == "$sha" ]]; then
    echo "PASS $name $got"
  else
    echo "FAIL $name expected $sha got ${got:-MISSING}"
    fail=1
  fi
done
[[ $fail == 0 ]] && echo "P4 DEVICE LEG: ALL ARMS SHA-IDENTICAL" \
                 || echo "P4 DEVICE LEG: MISMATCHES PRESENT"
exit $fail

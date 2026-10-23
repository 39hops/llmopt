import Std.Tactic.BVDecide

/-- The DFloat11-style split of a BF16 bit pattern: the 8 exponent bits
(positions 7..14) and the sign-plus-mantissa byte (sign at bit 7, the 7
mantissa bits below it), as scratch/bf16_lossless_bench.py splits them. -/
def splitExp (u : BitVec 16) : BitVec 8 := ((u >>> 7) &&& 0xFF#16).truncate 8
def splitSM (u : BitVec 16) : BitVec 8 := (((u >>> 8) &&& 0x80#16) ||| (u &&& 0x7F#16)).truncate 8

/-- The join the decoder performs. -/
def join (e sm : BitVec 8) : BitVec 16 :=
  (((sm.zeroExtend 16) &&& 0x80#16) <<< 8) ||| ((e.zeroExtend 16) <<< 7) ||| ((sm.zeroExtend 16) &&& 0x7F#16)

/-- Exact round trip: every 16-bit pattern is reconstructed bit for bit. -/
theorem join_split (u : BitVec 16) : join (splitExp u) (splitSM u) = u := by
  unfold join splitExp splitSM
  bv_decide

/-- The split is injective (two patterns with the same split are equal). -/
theorem split_injective (u v : BitVec 16)
    (he : splitExp u = splitExp v) (hs : splitSM u = splitSM v) : u = v :=
  calc u = join (splitExp u) (splitSM u) := (join_split u).symm
    _ = join (splitExp v) (splitSM v) := by rw [he, hs]
    _ = v := join_split v

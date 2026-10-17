/-
SpecDecoding.lean: small, model-free theorems for greedy speculative decoding.

(M) PROVEN IN LEAN (this file, no `sorry` in the main sections):
  A. Logit margin stability over exact reals: a margin m at the teacher argmax
     survives any per-logit perturbation of size < m/2 (strict) or <= m/2
     (weak); the m/2 bound is tight; the conclusion is shift-invariant.
  B. Greedy speculative decoding over an abstract deterministic teacher
     `T : List α → α`: every speculative step keeps the emitted stream equal
     to the greedy stream, emits at least one token, and any run of steps with
     arbitrary proposals reproduces `greedy T p n` on its first n tokens.
  C. Expected tokens per block: tail-sum identity, geometric closed form,
     break-even inequality, monotonicity in α, and E - 1 <= k α.
  D. Acceptance = agreement at a context; candidate-set rule; top-k contains
     the unique argmax.
  E. Fold/snapshot (rollback) law for recurrent state.
  F. The gated delta-rule factor (1 - β k kᵀ) is singular when β‖k‖² = 1
     (state step forgets the k-direction), and the Sherman-Morrison inverse
     otherwise.

ASSUMPTIONS THAT LIVE OUTSIDE LEAN (not proven here):
  * Floating-point arithmetic: all logits here are exact reals. Mapping a
    real kernel's fp16/bf16 deviation to a bound |z - z'| < m/2 is an
    empirical/numerical-analysis claim, not part of this file.
  * Kernel equivalence: that a batched verify block computes the same teacher
    function `T` as the one-token eager path is assumed (it is what the
    margin theorem in A is used to argue up to rounding).
  * Determinism of the teacher: `T` is a function; real decoders are only
    deterministic up to tie-breaking and nondeterministic reductions.
  * Measured constants (acceptance rates, per-call costs) enter only as
    hypotheses; no experimental number appears in this file.
  * The i.i.d. acceptance model behind E = (1 - α^(k+1))/(1 - α) is a
    modelling assumption; C proves algebra about E, not that it models data.
-/
import Mathlib

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false

namespace SpecDecoding

open Finset

/-! ## A. Logit margin stability -/

/-- Strict form: margin `m` at `i`, perturbation strictly below `m/2`. -/
theorem margin_stable {V : ℕ} (z z' : Fin V → ℝ) (i : Fin V) (m : ℝ) (hm : 0 < m)
    (hmargin : ∀ j, j ≠ i → z j + m ≤ z i) (hpert : ∀ j, |z j - z' j| < m / 2) :
    ∀ j, j ≠ i → z' j < z' i := by
  intro j hj
  have h1 := hmargin j hj
  have h2 := abs_lt.mp (hpert j)
  have h3 := abs_lt.mp (hpert i)
  linarith [h2.1, h2.2, h3.1, h3.2]

/-- Weak form: perturbation at most `m/2` keeps `i` a (possibly tied) maximizer. -/
theorem margin_stable_weak {V : ℕ} (z z' : Fin V → ℝ) (i : Fin V) (m : ℝ)
    (hmargin : ∀ j, j ≠ i → z j + m ≤ z i) (hpert : ∀ j, |z j - z' j| ≤ m / 2) :
    ∀ j, j ≠ i → z' j ≤ z' i := by
  intro j hj
  have h1 := hmargin j hj
  have h2 := abs_le.mp (hpert j)
  have h3 := abs_le.mp (hpert i)
  linarith [h2.1, h2.2, h3.1, h3.2]

/-- Tightness: if some `j₀ ≠ i` sits exactly `m` below `i`, any budget
`ε > m/2` admits a perturbation that dethrones `i` (ties or flips it). -/
theorem margin_tight {V : ℕ} (z : Fin V → ℝ) (i j₀ : Fin V) (m ε : ℝ) (hm : 0 < m)
    (hj₀ : j₀ ≠ i) (hj₀m : z j₀ + m = z i) (hε : m / 2 < ε) :
    ∃ z' : Fin V → ℝ, (∀ j, |z j - z' j| ≤ ε) ∧ ∃ j, j ≠ i ∧ z' i ≤ z' j := by
  have hε0 : 0 ≤ ε := by linarith
  refine ⟨fun j => if j = i then z j - ε else if j = j₀ then z j + ε else z j, ?_, j₀, hj₀, ?_⟩
  · intro j
    by_cases h : j = i
    · simp [h, abs_of_nonneg hε0]
    · by_cases h' : j = j₀
      · subst h'; simp [h, abs_of_nonneg hε0]
      · simp [h, h', hε0]
  · simp only [if_true, if_neg hj₀]
    linarith

/-- Shift invariance: adding a constant to every logit does not change the
unique-maximizer conclusion. -/
theorem argmax_shift_invariant {V : ℕ} (z' : Fin V → ℝ) (i : Fin V) (c : ℝ) :
    (∀ j, j ≠ i → z' j < z' i) ↔ (∀ j, j ≠ i → z' j + c < z' i + c) := by
  constructor
  · intro h j hj; linarith [h j hj]
  · intro h j hj; linarith [h j hj]

/-- Shift-aligned form: the perturbation only needs to be small after an
arbitrary common offset `c` (e.g. a softmax-irrelevant normalizer). -/
theorem margin_stable_shift {V : ℕ} (z z' : Fin V → ℝ) (i : Fin V) (m : ℝ) (hm : 0 < m)
    (hmargin : ∀ j, j ≠ i → z j + m ≤ z i)
    (hpert : ∃ c : ℝ, ∀ j, |z j + c - z' j| < m / 2) :
    ∀ j, j ≠ i → z' j < z' i := by
  obtain ⟨c, hc⟩ := hpert
  exact margin_stable (fun j => z j + c) z' i m hm
    (fun j hj => by have := hmargin j hj; linarith) hc

/-! ## B. Greedy speculative decoding exactness -/

section Greedy

variable {α : Type*} [DecidableEq α]

/-- `greedy T p n`: the first `n` tokens of plain greedy decoding from prompt `p`. -/
def greedy (T : List α → α) (p : List α) : ℕ → List α
  | 0 => []
  | n + 1 => greedy T p n ++ [T (p ++ greedy T p n)]

/-- `s` is a greedy continuation of `p`: every token is the teacher's choice. -/
def IsGreedy (T : List α → α) (p s : List α) : Prop :=
  ∀ i (h : i < s.length), s[i] = T (p ++ s.take i)

/-- Length of the longest accepted prefix of proposal `d` at context `ctx`. -/
def acceptLen (T : List α → α) : List α → List α → ℕ
  | _, [] => 0
  | ctx, x :: xs => if x = T ctx then acceptLen T (ctx ++ [x]) xs + 1 else 0

/-- One speculative step: keep the accepted prefix, then append the teacher's
token at the first rejection (or the bonus token after a full accept). -/
def specStep (T : List α → α) (p s d : List α) : List α :=
  s ++ d.take (acceptLen T (p ++ s) d) ++ [T (p ++ s ++ d.take (acceptLen T (p ++ s) d))]

/-- A run of speculative steps with arbitrary proposals, from an empty stream. -/
def specRun (T : List α → α) (p : List α) (ds : List (List α)) : List α :=
  ds.foldl (fun s d => specStep T p s d) []

theorem acceptLen_le (T : List α → α) : ∀ (ctx d : List α), acceptLen T ctx d ≤ d.length
  | _, [] => by simp [acceptLen]
  | ctx, x :: xs => by
      unfold acceptLen
      split_ifs
      · have := acceptLen_le T (ctx ++ [x]) xs; simp; omega
      · simp

/-- Every accepted draft token equals the teacher's token at its context. -/
theorem acceptLen_spec (T : List α → α) : ∀ (ctx d : List α) (j : ℕ)
    (hj : j < acceptLen T ctx d), ∃ h : j < d.length, d[j] = T (ctx ++ d.take j)
  | _, [], j, hj => by simp [acceptLen] at hj
  | ctx, x :: xs, j, hj => by
      unfold acceptLen at hj
      split_ifs at hj with hx
      · cases j with
        | zero => exact ⟨by simp, by simpa using hx⟩
        | succ j =>
          obtain ⟨h1, h2⟩ := acceptLen_spec T (ctx ++ [x]) xs j (by omega)
          refine ⟨by simp; omega, ?_⟩
          simpa [List.take_succ_cons] using h2
      · omega

/-- Maximality: the token right after the accepted prefix (if any) is rejected. -/
theorem acceptLen_maximal (T : List α → α) : ∀ (ctx d : List α)
    (h : acceptLen T ctx d < d.length),
    d[acceptLen T ctx d] ≠ T (ctx ++ d.take (acceptLen T ctx d))
  | _, [], h => by simp at h
  | ctx, x :: xs, h => by
      by_cases hx : x = T ctx
      · have e : acceptLen T ctx (x :: xs) = acceptLen T (ctx ++ [x]) xs + 1 := by
          simp [acceptLen, hx]
        have h' : acceptLen T (ctx ++ [x]) xs < xs.length := by
          rw [e] at h; simp at h; omega
        have ih := acceptLen_maximal T (ctx ++ [x]) xs h'
        simp only [e, List.getElem_cons_succ, List.take_succ_cons]
        simpa using ih
      · have e : acceptLen T ctx (x :: xs) = 0 := by simp [acceptLen, hx]
        simp only [e]
        simpa using hx

/-- Appending a block whose every token is the teacher's choice keeps greediness. -/
theorem isGreedy_append (T : List α → α) (p s e : List α) (hs : IsGreedy T p s)
    (he : ∀ j (h : j < e.length), e[j] = T (p ++ s ++ e.take j)) :
    IsGreedy T p (s ++ e) := by
  intro i hi
  by_cases hlt : i < s.length
  · rw [List.getElem_append_left hlt, List.take_append_of_le_length (by omega)]
    exact hs i hlt
  · have hle : s.length ≤ i := by omega
    rw [List.getElem_append_right hle]
    have hi' : i - s.length < e.length := by rw [List.length_append] at hi; omega
    have ht : (s ++ e).take i = s ++ e.take (i - s.length) := by
      rw [List.take_append, List.take_of_length_le (l := s) (by omega)]
    rw [he _ hi', ht, List.append_assoc]

theorem isGreedy_nil (T : List α → α) (p : List α) : IsGreedy T p [] := by
  intro i h; simp at h

/-- (i) A speculative step preserves greediness. -/
theorem specStep_isGreedy (T : List α → α) (p s d : List α) (hs : IsGreedy T p s) :
    IsGreedy T p (specStep T p s d) := by
  unfold specStep
  set a := acceptLen T (p ++ s) d
  have h1 : IsGreedy T p (s ++ d.take a) := by
    apply isGreedy_append T p s _ hs
    intro j hj
    have hja : j < a := by simp at hj; omega
    obtain ⟨hjd, hval⟩ := acceptLen_spec T (p ++ s) d j hja
    rw [List.getElem_take, hval, List.take_take, Nat.min_eq_left (by omega)]
  apply isGreedy_append T p _ _ h1
  intro j hj
  simp at hj
  subst hj
  simp

/-- (ii) A speculative step strictly increases the length. -/
theorem specStep_length (T : List α → α) (p s d : List α) :
    s.length < (specStep T p s d).length := by
  simp [specStep]

/-- (iii) Uniqueness: two greedy continuations of equal length are equal. -/
theorem isGreedy_unique (T : List α → α) (p s t : List α) (hs : IsGreedy T p s)
    (ht : IsGreedy T p t) (hlen : s.length = t.length) : s = t := by
  have key : ∀ n, ∀ i (h1 : i < s.length) (h2 : i < t.length), i < n → s[i] = t[i] := by
    intro n
    induction n with
    | zero => intro i _ _ h; omega
    | succ n ih =>
      intro i h1 h2 hi
      have htake : s.take i = t.take i := by
        apply List.ext_getElem
        · simp; omega
        · intro j hj1 hj2
          simp only [List.getElem_take]
          simp at hj1
          exact ih j (by omega) (by omega) (by omega)
      rw [hs i h1, ht i h2, htake]
  apply List.ext_getElem hlen
  intro i h1 h2
  exact key (i + 1) i h1 h2 (by omega)

theorem greedy_length (T : List α → α) (p : List α) (n : ℕ) : (greedy T p n).length = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [greedy, ih]

theorem greedy_isGreedy (T : List α → α) (p : List α) (n : ℕ) : IsGreedy T p (greedy T p n) := by
  induction n with
  | zero => exact isGreedy_nil T p
  | succ n ih =>
    show IsGreedy T p (greedy T p n ++ [T (p ++ greedy T p n)])
    apply isGreedy_append T p _ _ ih
    intro j hj
    simp at hj; subst hj; simp

theorem isGreedy_take (T : List α → α) (p s : List α) (hs : IsGreedy T p s) (n : ℕ) :
    IsGreedy T p (s.take n) := by
  intro i hi
  simp at hi
  rw [List.getElem_take, hs i (by omega), List.take_take, Nat.min_eq_left (by omega)]

/-- Every greedy continuation is a prefix of plain greedy decoding. -/
theorem isGreedy_eq_greedy (T : List α → α) (p s : List α) (hs : IsGreedy T p s) :
    s = greedy T p s.length :=
  isGreedy_unique T p _ _ hs (greedy_isGreedy T p _) (by rw [greedy_length])

theorem specRun_aux (T : List α → α) (p : List α) :
    ∀ (ds : List (List α)) (s : List α), IsGreedy T p s →
      IsGreedy T p (ds.foldl (fun s d => specStep T p s d) s) ∧
      s.length + ds.length ≤ (ds.foldl (fun s d => specStep T p s d) s).length
  | [], s, hs => ⟨hs, by simp⟩
  | d :: ds, s, hs => by
      obtain ⟨h1, h2⟩ := specRun_aux T p ds (specStep T p s d) (specStep_isGreedy T p s d hs)
      have := specStep_length T p s d
      refine ⟨by simpa using h1, ?_⟩
      simp only [List.foldl_cons, List.length_cons]
      omega

theorem specRun_isGreedy (T : List α → α) (p : List α) (ds : List (List α)) :
    IsGreedy T p (specRun T p ds) :=
  (specRun_aux T p ds [] (isGreedy_nil T p)).1

/-- Each step emits at least one token, whatever the proposals. -/
theorem specRun_length (T : List α → α) (p : List α) (ds : List (List α)) :
    ds.length ≤ (specRun T p ds).length := by
  have := (specRun_aux T p ds [] (isGreedy_nil T p)).2
  simpa [specRun] using this

/-- (iv) Exactness: any run of speculative steps with arbitrary proposals
emits exactly the greedy stream, token for token. -/
theorem specRun_take_eq_greedy (T : List α → α) (p : List α) (ds : List (List α)) (n : ℕ)
    (hn : n ≤ (specRun T p ds).length) :
    (specRun T p ds).take n = greedy T p n := by
  have h := isGreedy_eq_greedy T p _ (isGreedy_take T p _ (specRun_isGreedy T p ds) n)
  rw [h, List.length_take, Nat.min_eq_left hn]

end Greedy

/-! ## C. Expected emitted tokens per block -/

/-- (i) Tail-sum identity: for `A` on `{0..k}` with law `p`,
`E[A] + 1 = Σ_{j=0}^{k} P(A ≥ j)`. -/
theorem tail_sum (k : ℕ) (p : Fin (k + 1) → ℝ) (hsum : ∑ j, p j = 1) :
    ∑ j : Fin (k + 1), ((j : ℕ) : ℝ) * p j + 1 =
      ∑ j : Fin (k + 1), ∑ i, (if j ≤ i then p i else 0) := by
  rw [Finset.sum_comm]
  have hinner : ∀ i : Fin (k + 1),
      ∑ j : Fin (k + 1), (if j ≤ i then p i else 0) = (((i : ℕ) : ℝ) + 1) * p i := by
    intro i
    rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    have : (Finset.univ.filter (fun j : Fin (k + 1) => j ≤ i)) = Finset.Iic i := by
      ext j; simp
    rw [this, Fin.card_Iic]
    push_cast; ring
  simp_rw [hinner, add_mul, one_mul, Finset.sum_add_distrib, hsum]

/-- (ii) Geometric closed form of `E = Σ_{j=0}^{k} α^j`. -/
theorem geom_closed (α : ℝ) (hα : α ≠ 1) (k : ℕ) :
    ∑ j ∈ Finset.range (k + 1), α ^ j = (1 - α ^ (k + 1)) / (1 - α) := by
  rw [geom_sum_eq hα, ← neg_div_neg_eq]
  congr 1 <;> ring

/-- Expected tokens emitted per speculative block under per-token acceptance `α`. -/
noncomputable def expTokens (α : ℝ) (k : ℕ) : ℝ := (1 - α ^ (k + 1)) / (1 - α)

theorem expTokens_eq_sum (α : ℝ) (hα : α < 1) (k : ℕ) :
    expTokens α k = ∑ j ∈ Finset.range (k + 1), α ^ j :=
  (geom_closed α (ne_of_lt hα) k).symm

/-- (iii-a) Break-even: speculative throughput `E / (k c_d + c_v)` beats plain
throughput `1 / c_t` iff `E c_t > k c_d + c_v`. -/
theorem break_even (α : ℝ) (k : ℕ) (cd cv ct : ℝ) (hcd : 0 ≤ cd) (hcv : 0 < cv)
    (hct : 0 < ct) :
    1 / ct < expTokens α k / (k * cd + cv) ↔ k * cd + cv < expTokens α k * ct := by
  have hden : 0 < (k : ℝ) * cd + cv := by positivity
  rw [div_lt_div_iff₀ hct hden, one_mul]

/-- (iii-b) `E` is monotone in `α` on `[0,1)`. -/
theorem expTokens_mono (k : ℕ) (a b : ℝ) (ha : 0 ≤ a) (hab : a ≤ b) (hb : b < 1) :
    expTokens a k ≤ expTokens b k := by
  rw [expTokens_eq_sum a (by linarith) k, expTokens_eq_sum b hb k]
  exact Finset.sum_le_sum (fun j _ => pow_le_pow_left₀ ha hab j)

/-- (iv) `E - 1 ≤ k α`. -/
theorem expTokens_sub_one_le (α : ℝ) (h0 : 0 ≤ α) (h1 : α < 1) (k : ℕ) :
    expTokens α k - 1 ≤ k * α := by
  rw [expTokens_eq_sum α h1 k, Finset.sum_range_succ']
  simp only [pow_zero, add_sub_cancel_right]
  calc ∑ j ∈ Finset.range k, α ^ (j + 1) ≤ ∑ j ∈ Finset.range k, α :=
        Finset.sum_le_sum (fun j _ => pow_le_of_le_one h0 h1.le (by omega))
    _ = k * α := by simp

/-- (iv') The logged accepted/drafted ratio `(E - 1)/k` underestimates `α`. -/
theorem logged_ratio_le (α : ℝ) (h0 : 0 ≤ α) (h1 : α < 1) (k : ℕ) (hk : 1 ≤ k) :
    (expTokens α k - 1) / k ≤ α := by
  have hkpos : (0 : ℝ) < k := by exact_mod_cast hk
  rw [div_le_iff₀ hkpos, mul_comm]
  exact expTokens_sub_one_le α h0 h1 k

/-! ## D. Acceptance equals agreement; candidate sets; top-k -/

/-- `i` is the unique maximizer of `z`. -/
def IsUniqueArgmax {V : ℕ} (z : Fin V → ℝ) (i : Fin V) : Prop := ∀ j, j ≠ i → z j < z i

theorem uniqueArgmax_unique {V : ℕ} (z : Fin V → ℝ) (i i' : Fin V)
    (h : IsUniqueArgmax z i) (h' : IsUniqueArgmax z i') : i = i' := by
  by_contra hne
  have a := h i' (Ne.symm hne)
  have b := h' i hne
  linarith

/-- Greedy acceptance: the draft token is accepted iff it is the teacher token. -/
def accept {V : ℕ} (d istar : Fin V) : Prop := d = istar

theorem accept_iff {V : ℕ} (d istar : Fin V) : accept d istar ↔ d = istar := Iff.rfl

/-- If the draft emits its own unique argmax `jstar` and the teacher's unique
argmax is `istar`, acceptance is exactly argmax agreement. -/
theorem accept_iff_argmax_agree {V : ℕ} (zd zt : Fin V → ℝ) (d jstar istar : Fin V)
    (hd : d = jstar) (hj : IsUniqueArgmax zd jstar) (hi : IsUniqueArgmax zt istar) :
    accept d istar ↔ jstar = istar := by
  subst hd; exact Iff.rfl

/-- Candidate-set rule: accept iff the teacher argmax lies in `S`; on accept
the emitted token is the teacher argmax. -/
def candEmit {V : ℕ} (S : Finset (Fin V)) (istar : Fin V) : Option (Fin V) :=
  if istar ∈ S then some istar else none

theorem candEmit_correct {V : ℕ} (S : Finset (Fin V)) (istar t : Fin V)
    (h : candEmit S istar = some t) : t = istar := by
  unfold candEmit at h
  split_ifs at h
  · exact (Option.some.inj h).symm

theorem candEmit_isSome_iff {V : ℕ} (S : Finset (Fin V)) (istar : Fin V) :
    (candEmit S istar).isSome ↔ istar ∈ S := by
  unfold candEmit; split_ifs with h <;> simp [h]

/-- Counterexample: the argmax of `z` restricted to `S` can differ from the
teacher argmax when the teacher argmax is not in `S` (V = 2). -/
theorem restricted_argmax_wrong :
    ∃ (z : Fin 2 → ℝ) (S : Finset (Fin 2)) (istar j : Fin 2),
      IsUniqueArgmax z istar ∧ j ∈ S ∧ istar ∉ S ∧
      (∀ j' ∈ S, j' ≠ j → z j' < z j) ∧ j ≠ istar := by
  refine ⟨![0, 1], {0}, 1, 0, ?_, by simp, by simp, ?_, by decide⟩
  · intro j hj
    fin_cases j
    · simp
    · exact absurd rfl hj
  · intro j' hj' hne
    simp at hj'
    exact absurd hj' hne

/-- Top-k set: tokens with fewer than `k` strictly larger logits. -/
def topk {V : ℕ} (z : Fin V → ℝ) (k : ℕ) : Set (Fin V) :=
  {j | (Finset.univ.filter (fun i => z j < z i)).card < k}

theorem argmax_mem_topk {V : ℕ} (z : Fin V → ℝ) (i : Fin V) (hi : IsUniqueArgmax z i)
    (k : ℕ) (hk : 1 ≤ k) : i ∈ topk z k := by
  show (Finset.univ.filter (fun l => z i < z l)).card < k
  have : Finset.univ.filter (fun l => z i < z l) = ∅ := by
    ext l
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty, iff_false,
      not_lt]
    by_cases hl : l = i
    · rw [hl]
    · exact (hi l hl).le
  rw [this, Finset.card_empty]
  omega

theorem topk_mono {V : ℕ} (z : Fin V → ℝ) (k k' : ℕ) (h : k ≤ k') : topk z k ⊆ topk z k' :=
  fun j hj => lt_of_lt_of_le hj h

/-! ## E. Fold / snapshot (rollback) law for recurrent state -/

/-- Rollback law: the state after `xs ++ ys` is the state after `xs`, continued
on `ys`. So a snapshot taken after the accepted prefix `xs` is exactly the
state a non-speculative run would hold there; the rejected suffix is discarded
by restoring the snapshot, never by inverting the update. -/
theorem fold_snapshot {S β : Type*} (f : S → β → S) (s₀ : S) (xs ys : List β) :
    List.foldl f s₀ (xs ++ ys) = List.foldl f (List.foldl f s₀ xs) ys :=
  List.foldl_append

/-- Hybrid state: an appendable history (rolled back by `take`) paired with a
recurrent state (rolled back by snapshot). -/
def hstep {S β : Type*} (f : S → β → S) (st : List β × S) (x : β) : List β × S :=
  (st.1 ++ [x], f st.2 x)

theorem hfold_fst {S β : Type*} (f : S → β → S) :
    ∀ (xs : List β) (st : List β × S), (List.foldl (hstep f) st xs).1 = st.1 ++ xs
  | [], st => by simp
  | x :: xs, st => by
      rw [List.foldl_cons, hfold_fst f xs]
      simp [hstep]

theorem hfold_snd {S β : Type*} (f : S → β → S) :
    ∀ (xs : List β) (st : List β × S), (List.foldl (hstep f) st xs).2 = List.foldl f st.2 xs
  | [], st => by simp
  | x :: xs, st => by
      rw [List.foldl_cons, hfold_snd f xs]
      simp [hstep]

/-- Rollback of the hybrid state: the history rolls back by `take`, the
recurrent part by the snapshot taken after `xs`. -/
theorem hybrid_rollback {S β : Type*} (f : S → β → S) (s₀ : S) (xs ys : List β) :
    (List.foldl (hstep f) ([], s₀) (xs ++ ys)).1.take xs.length = xs ∧
    List.foldl (hstep f) ([], s₀) xs = (xs, List.foldl f s₀ xs) := by
  constructor
  · rw [hfold_fst]; simp
  · ext1
    · rw [hfold_fst]; simp
    · rw [hfold_snd]

/-! ## F. Gated delta step: non-invertibility and Sherman-Morrison -/

section Delta

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem kkT_mul_kkT (k : n → ℝ) :
    vecMulVec k k * vecMulVec k k = (k ⬝ᵥ k) • vecMulVec k k := by
  rw [vecMulVec_mul_vecMulVec, ← smul_vecMulVec]
  ext a b
  simp [vecMulVec_apply]
  ring

/-- `(1 - β k kᵀ) k = 0` when `β ‖k‖² = 1`. -/
theorem delta_factor_kills_k (k : n → ℝ) (β : ℝ) (hβ : β * (k ⬝ᵥ k) = 1) :
    (1 - β • vecMulVec k k) *ᵥ k = 0 := by
  rw [sub_mulVec, one_mulVec, smul_mulVec, vecMulVec_mulVec]
  ext a
  simp [hβ, ← mul_assoc]

/-- Non-injectivity of the factor: it maps both `0` and `k` to `0`. -/
theorem delta_factor_not_injective (k : n → ℝ) (β : ℝ) (hβ : β * (k ⬝ᵥ k) = 1) :
    ¬ Function.Injective (fun x => (1 - β • vecMulVec k k) *ᵥ x) := by
  intro hinj
  have hk0 : k ≠ 0 := by
    rintro rfl
    simp at hβ
  apply hk0
  apply hinj
  simp only [delta_factor_kills_k k β hβ, mulVec_zero]

/-- The gated delta step `S' = g S (1 - β k kᵀ) + β v kᵀ` cannot determine
`S`: adding any `w kᵀ` to `S` leaves `S'` unchanged. So rollback must restore
a snapshot (section E) rather than invert the step. -/
theorem delta_step_forgets {m : Type*} (S : Matrix m n ℝ) (w v : m → ℝ) (k : n → ℝ)
    (g β : ℝ) (hβ : β * (k ⬝ᵥ k) = 1) :
    g • (S + vecMulVec w k) * (1 - β • vecMulVec k k) + β • vecMulVec v k =
      g • S * (1 - β • vecMulVec k k) + β • vecMulVec v k := by
  have hz : vecMulVec w k * (1 - β • vecMulVec k k) = 0 := by
    rw [vecMulVec_mul]
    have : k ᵥ* (1 - β • vecMulVec k k) = 0 := by
      rw [vecMul_sub, vecMul_one, vecMul_smul, vecMul_vecMulVec, smul_smul,
        hβ, one_smul, sub_self]
    rw [this]
    ext a b; simp [vecMulVec_apply]
  rw [smul_add, Matrix.add_mul, Matrix.smul_mul g (vecMulVec w k), hz, smul_zero, add_zero]

/-- Sherman-Morrison: when `β ‖k‖² ≠ 1` the factor is invertible. -/
theorem delta_factor_sherman_morrison (k : n → ℝ) (β : ℝ) (hβ : β * (k ⬝ᵥ k) ≠ 1) :
    (1 - β • vecMulVec k k) * (1 + (β / (1 - β * (k ⬝ᵥ k))) • vecMulVec k k) = 1 := by
  have hd : 1 - β * (k ⬝ᵥ k) ≠ 0 := sub_ne_zero.mpr (Ne.symm hβ)
  set γ := β / (1 - β * (k ⬝ᵥ k)) with hγ
  have hcoef : γ - β - β * γ * (k ⬝ᵥ k) = 0 := by
    rw [hγ]; field_simp; ring
  rw [sub_mul, mul_add, mul_add, one_mul, mul_one, one_mul, smul_mul_smul_comm,
    kkT_mul_kkT, smul_smul]
  have : (1 : Matrix n n ℝ) + γ • vecMulVec k k -
      (β • vecMulVec k k + (β * γ * (k ⬝ᵥ k)) • vecMulVec k k) =
      1 + (γ - β - β * γ * (k ⬝ᵥ k)) • vecMulVec k k := by
    rw [sub_smul, sub_smul]; abel
  rw [this, hcoef, zero_smul, add_zero]

end Delta

end SpecDecoding

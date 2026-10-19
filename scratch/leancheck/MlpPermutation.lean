/-
SwiGLU intermediate-neuron permutation invariance (Phase C of the
2026-10-19 representation round).

A gated MLP with gate matrix `Wg : Matrix (Fin m) (Fin d) R`, up matrix
`Wu : Matrix (Fin m) (Fin d) R`, down matrix `Wd : Matrix (Fin d) (Fin m) R`
and any gate function `act : R → R` maps `x : Fin d → R` to
`Wd *ᵥ (fun j => act ((Wg *ᵥ x) j) * (Wu *ᵥ x) j)`.
For a permutation `σ` of the intermediate index set, with
`P = σ.toPEquiv.toMatrix` the permutation matrix, the model with
`Wg' = P * Wg`, `Wu' = P * Wu`, `Wd' = Wd * Pᵀ` computes the same function.
Stated over any commutative semiring (so in particular over ℝ, the
exact-arithmetic model of the floating-point computation) and for any
`act`, so SiLU needs no property. Finite dimensions only; nothing about
rounding is claimed here.
-/
import Mathlib.Data.Matrix.PEquiv
import Mathlib.Data.Matrix.Mul
import Mathlib.Analysis.SpecialFunctions.Exp

namespace MlpPermutation

open Matrix

variable {R : Type*} [CommSemiring R] {m d : Type*}

/-- The gated MLP as a function of its three matrices and the gate function. -/
def gatedMlp [Fintype m] [Fintype d] (act : R → R) (Wg Wu : Matrix m d R) (Wd : Matrix d m R)
    (x : d → R) : d → R :=
  Wd *ᵥ (fun j => act ((Wg *ᵥ x) j) * (Wu *ᵥ x) j)

/-- Row permutation of a matrix by `σ`: `(P * W) i = W (σ i)`. -/
theorem permMatrix_mul_eq_submatrix [Fintype m] [DecidableEq m] (σ : Equiv.Perm m) (W : Matrix m d R) :
    σ.toPEquiv.toMatrix * W = W.submatrix σ id :=
  PEquiv.toMatrix_toPEquiv_mul σ W

/-- Column permutation by the transpose: `(Wd * Pᵀ) i k = Wd i (σ k)`. -/
theorem mul_permMatrix_transpose_eq_submatrix [Fintype m] [DecidableEq m] (σ : Equiv.Perm m)
    (Wd : Matrix d m R) :
    Wd * (σ.toPEquiv.toMatrix)ᵀ = Wd.submatrix id σ := by
  rw [← PEquiv.toMatrix_symm, ← Equiv.toPEquiv_symm, PEquiv.mul_toMatrix_toPEquiv]
  simp

/-- `mulVec` through a row-permuted matrix is the permuted vector. -/
theorem submatrix_mulVec [Fintype d] (σ : Equiv.Perm m) (W : Matrix m d R) (x : d → R) :
    (W.submatrix σ id) *ᵥ x = fun i => (W *ᵥ x) (σ i) := by
  funext i
  simp [Matrix.mulVec, dotProduct, Matrix.submatrix]

/-- `mulVec` of a column-permuted matrix against a permuted vector is the original. -/
theorem submatrix_mulVec_comp [Fintype m] (σ : Equiv.Perm m) (Wd : Matrix d m R) (v : m → R) :
    (Wd.submatrix id σ) *ᵥ (fun k => v (σ k)) = Wd *ᵥ v := by
  funext i
  simp only [Matrix.mulVec, dotProduct, Matrix.submatrix_apply, id]
  exact Equiv.sum_comp σ (fun k => Wd i k * v k)

/-- Main identity: a consistent permutation of the intermediate neurons leaves
the gated MLP unchanged, for every input, every gate function and every
commutative semiring of coefficients. -/
theorem gatedMlp_perm_invariant [Fintype m] [Fintype d] [DecidableEq m] (act : R → R) (σ : Equiv.Perm m)
    (Wg Wu : Matrix m d R) (Wd : Matrix d m R) (x : d → R) :
    gatedMlp act (σ.toPEquiv.toMatrix * Wg) (σ.toPEquiv.toMatrix * Wu)
      (Wd * (σ.toPEquiv.toMatrix)ᵀ) x = gatedMlp act Wg Wu Wd x := by
  unfold gatedMlp
  rw [permMatrix_mul_eq_submatrix, permMatrix_mul_eq_submatrix,
    mul_permMatrix_transpose_eq_submatrix, submatrix_mulVec, submatrix_mulVec]
  exact submatrix_mulVec_comp σ Wd (fun j => act ((Wg *ᵥ x) j) * (Wu *ᵥ x) j)

/-- The same identity stated for the whole function (all inputs at once). -/
theorem gatedMlp_perm_invariant_fun [Fintype m] [Fintype d] [DecidableEq m] (act : R → R)
    (σ : Equiv.Perm m)
    (Wg Wu : Matrix m d R) (Wd : Matrix d m R) :
    gatedMlp act (σ.toPEquiv.toMatrix * Wg) (σ.toPEquiv.toMatrix * Wu)
      (Wd * (σ.toPEquiv.toMatrix)ᵀ) = gatedMlp act Wg Wu Wd := by
  funext x
  exact gatedMlp_perm_invariant act σ Wg Wu Wd x

/-- Over the reals with SiLU as the gate (the SwiGLU of the model). -/
noncomputable def silu (t : ℝ) : ℝ := t / (1 + Real.exp (-t))

theorem swiglu_perm_invariant [Fintype m] [Fintype d] [DecidableEq m] (σ : Equiv.Perm m)
    (Wg Wu : Matrix m d ℝ) (Wd : Matrix d m ℝ) :
    gatedMlp silu (σ.toPEquiv.toMatrix * Wg) (σ.toPEquiv.toMatrix * Wu)
      (Wd * (σ.toPEquiv.toMatrix)ᵀ) = gatedMlp silu Wg Wu Wd :=
  gatedMlp_perm_invariant_fun silu σ Wg Wu Wd

end MlpPermutation

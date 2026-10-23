import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul

/-!
QWEN-LOSSY-GEOMETRY-0, the algebra of an input-basis rotation.

For a dense linear map y = W x and any square H with Hᵀ H = 1 (an
orthogonal matrix; a sign-randomised normalised Hadamard block matrix
is one), storing W' = W Hᵀ and feeding x' = H x reproduces W x exactly
in the reals. The statement is about real arithmetic; floating-point
evaluation of the two sides differs by rounding (measured separately in
scratch/qwen_geometry_quant.py), and nothing here says anything about
what happens when W' is quantised or when a rotation is moved through a
nonlinearity, a normalisation or a recurrent update.

Checked with: lake env lean InputRotation.lean (leanprover/lean4:v4.33.0-rc2, Mathlib as pinned).
-/

open Matrix

/-- Input-basis change by an orthogonal matrix: (W Hᵀ)(H x) = W x. -/
theorem input_rotation_identity {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]
    (W : Matrix m n ℝ) (H : Matrix n n ℝ) (hH : Hᵀ * H = 1) (x : n → ℝ) :
    (W * Hᵀ).mulVec (H.mulVec x) = W.mulVec x := by
  rw [mulVec_mulVec, Matrix.mul_assoc, hH, Matrix.mul_one]

/-- The same for a batch of inputs held as the columns of X. -/
theorem input_rotation_identity_batch {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq n]
    (W : Matrix m n ℝ) (H : Matrix n n ℝ) (hH : Hᵀ * H = 1) (X : Matrix n k ℝ) :
    (W * Hᵀ) * (H * X) = W * X := by
  rw [Matrix.mul_assoc, ← Matrix.mul_assoc Hᵀ, hH, Matrix.one_mul]

/-- A sign flip composed with an orthogonal matrix is orthogonal: the
sign-randomised Hadamard of QuIP# and HIGGS satisfies the hypothesis. -/
theorem sign_orthogonal {n : Type*} [Fintype n] [DecidableEq n]
    (H : Matrix n n ℝ) (hH : Hᵀ * H = 1) (s : n → ℝ) (hs : ∀ i, s i * s i = 1) :
    (H * Matrix.diagonal s)ᵀ * (H * Matrix.diagonal s) = 1 := by
  rw [Matrix.transpose_mul, Matrix.diagonal_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc Hᵀ, hH,
    Matrix.one_mul, Matrix.diagonal_mul_diagonal]
  simp [hs]

#print axioms input_rotation_identity
#print axioms input_rotation_identity_batch
#print axioms sign_orthogonal

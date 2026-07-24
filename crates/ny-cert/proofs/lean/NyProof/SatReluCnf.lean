/-
  SatReluCnf.lean — the end-to-end CNF ↔ unsafe-region equivalence for the
  sat_relu gadget (the CNF-recovery route: decompile the compiled k-SAT net
  back to its source CNF and decide it exactly).

  The sat_relu benchmark compiles k-SAT into Gemm→ReLU→Gemm nets.  Modelled in
  REAL arithmetic (this file, building on `Crownproof.SatReluGadget`):

  * clause row `i` is `ReLU(Σ_{j∈neg_i} x_j − Σ_{j∈pos_i} x_j + (1 − |neg_i|))`
    where `pos_i`/`neg_i` are the clause's positive/negated literal variables
    (disjoint finsets);
  * `Y_0 = 1 − Σ_i row_i`, `Y_1 = Σ_j (x_j − ReLU(2x_j − 1)) = Σ_j bres x_j`;
  * the property box is `x ∈ [0,1]^n`, the UNSAFE region `{Y_0 ≥ 1 ∧ Y_1 ≤ 0}`.

  Main theorem `sat_iff_unsafe`: over a finite variable set `s` and a finite
  family of clauses whose variables all lie in `s`,

      (∃ boolean assignment satisfying every clause)
        ↔ (∃ x ∈ [0,1]^s with Y_0 x ≥ 1 ∧ Y_1 x ≤ 0).

  Both directions are proved constructively-shaped:
  * `exists_satisfying_of_unsafe` (soundness of UNSAT, contrapositive form
    `unsat_implies_safe`): any unsafe box point is forced boolean by
    `forces_boolean` (from `SatReluGadget`), and at boolean points each clause
    row is `0` iff the clause is satisfied and `1` iff falsified, so `Y_0 ≥ 1`
    (i.e. `Σ rows ≤ 0` with all rows `≥ 0`) means every clause is satisfied;
  * `boolPoint_unsafe_of_satisfies` (SAT witness): a satisfying assignment `σ`
    gives the 0/1 point `boolPoint σ` which lies in the box with
    `Y_0 = 1` and `Y_1 = 0` exactly.

  HONEST SCOPE.  This theorem is about the gadget's exact REAL-arithmetic
  semantics only.  What binds a concrete ONNX network to this gadget shape is
  the fail-closed bit-exact detector in
  `crates/ny-cli/src/commands/beta_crown/cnf_route.rs`: all gadget weights are
  small integers, exactly representable in f32, and any deviation from the
  shape falls through to the normal pipeline.  The f32 FORWARD evaluation of
  the network is NOT covered here — SAT witnesses are confirmed by a concrete
  forward pass and gated on the ONNX-Runtime trusted oracle downstream.
  Hypotheses are stated explicitly: variables of every clause must lie in the
  summed variable set `s`, and the box constraint is only assumed on `s`.
-/
import NyProof.SatReluGadget
import Mathlib.Tactic.Ring

namespace Crownproof

namespace SatRelu

variable {ι : Type*}

/-- A CNF clause over variables `ι`: a finite set of positive-literal
variables and a disjoint finite set of negated-literal variables. -/
structure Clause (ι : Type*) where
  /-- Variables occurring positively. -/
  pos : Finset ι
  /-- Variables occurring negated. -/
  neg : Finset ι
  /-- A variable occurs at most once per clause (matches DIMACS emission). -/
  disj : Disjoint pos neg

/-- `σ` satisfies clause `c`: some positive literal is true or some negated
literal's variable is false. -/
def Clause.satisfies (σ : ι → Bool) (c : Clause ι) : Prop :=
  (∃ j ∈ c.pos, σ j = true) ∨ (∃ j ∈ c.neg, σ j = false)

/-- The gadget's hidden clause row:
`ReLU(Σ_{j∈neg} x_j − Σ_{j∈pos} x_j + (1 − |neg|))`
(weight `+1` on negated literals, `−1` on positive ones, bias `1 − #negated`). -/
def clauseRow (c : Clause ι) (x : ι → ℝ) : ℝ :=
  relu (∑ j ∈ c.neg, x j - ∑ j ∈ c.pos, x j + (1 - (c.neg.card : ℝ)))

/-- The 0/1 corner point of the box corresponding to a boolean assignment. -/
def boolPoint (σ : ι → Bool) : ι → ℝ := fun j => if σ j then 1 else 0

/-- Gadget output `Y_0 = 1 − Σ_i row_i` over a finite family of clauses. -/
def Y0 {m : ℕ} (C : Fin m → Clause ι) (x : ι → ℝ) : ℝ :=
  1 - ∑ i, clauseRow (C i) x

/-- Gadget output `Y_1 = Σ_j (x_j − ReLU(2x_j − 1))` over the variable set. -/
def Y1 (s : Finset ι) (x : ι → ℝ) : ℝ :=
  ∑ j ∈ s, bres (x j)

/-- A clause row is nonnegative for every real input (trivial: it is a ReLU). -/
theorem clauseRow_nonneg (c : Clause ι) (x : ι → ℝ) : 0 ≤ clauseRow c x :=
  le_max_left 0 _

/-- Boolean corner points lie in the `[0,1]` box (at every coordinate). -/
theorem boolPoint_mem_box (σ : ι → Bool) (j : ι) :
    0 ≤ boolPoint σ j ∧ boolPoint σ j ≤ 1 := by
  unfold boolPoint
  split <;> norm_num

/-- The Booleanization residual vanishes at boolean corner points. -/
theorem bres_boolPoint (σ : ι → Bool) (j : ι) : bres (boolPoint σ j) = 0 := by
  unfold boolPoint
  split
  · exact (bres_eq_zero_iff zero_le_one le_rfl).mpr (Or.inr rfl)
  · exact (bres_eq_zero_iff le_rfl zero_le_one).mpr (Or.inl rfl)

/-- The clause row depends only on the values of `x` on the clause's
variables. -/
theorem clauseRow_congr (c : Clause ι) {x y : ι → ℝ}
    (hp : ∀ j ∈ c.pos, x j = y j) (hn : ∀ j ∈ c.neg, x j = y j) :
    clauseRow c x = clauseRow c y := by
  unfold clauseRow
  rw [Finset.sum_congr rfl hn, Finset.sum_congr rfl hp]

/-- **Clause row at a boolean point, satisfied case.**  If `σ` satisfies `c`,
the row's pre-activation is `≤ 0`, so the row is `0`. -/
theorem clauseRow_boolPoint_of_satisfies (c : Clause ι) {σ : ι → Bool}
    (h : c.satisfies σ) : clauseRow c (boolPoint σ) = 0 := by
  classical
  have hge0 : ∀ j : ι, 0 ≤ boolPoint σ j := fun j => (boolPoint_mem_box σ j).1
  have hle1 : ∀ j : ι, boolPoint σ j ≤ 1 := fun j => (boolPoint_mem_box σ j).2
  have hneg_le : ∑ j ∈ c.neg, boolPoint σ j ≤ (c.neg.card : ℝ) := by
    calc ∑ j ∈ c.neg, boolPoint σ j ≤ ∑ _j ∈ c.neg, (1 : ℝ) :=
          Finset.sum_le_sum fun j _ => hle1 j
      _ = (c.neg.card : ℝ) := by simp
  have hpos_ge : (0 : ℝ) ≤ ∑ j ∈ c.pos, boolPoint σ j :=
    Finset.sum_nonneg fun j _ => hge0 j
  unfold clauseRow relu
  apply max_eq_left
  rcases h with ⟨j0, hj0, hσ⟩ | ⟨j0, hj0, hσ⟩
  · -- A positive literal is true: the positive sum is ≥ 1.
    have h1 : (1 : ℝ) ≤ ∑ j ∈ c.pos, boolPoint σ j := by
      have hs := Finset.single_le_sum (f := boolPoint σ) (fun j _ => hge0 j) hj0
      simpa [boolPoint, hσ] using hs
    linarith
  · -- A negated literal is false: the negated sum is ≤ |neg| − 1.
    have hzero : boolPoint σ j0 = 0 := by simp [boolPoint, hσ]
    have herase : ∑ j ∈ c.neg.erase j0, boolPoint σ j = ∑ j ∈ c.neg, boolPoint σ j :=
      Finset.sum_erase _ hzero
    have herase_le :
        ∑ j ∈ c.neg.erase j0, boolPoint σ j ≤ ((c.neg.erase j0).card : ℝ) := by
      calc ∑ j ∈ c.neg.erase j0, boolPoint σ j ≤ ∑ _j ∈ c.neg.erase j0, (1 : ℝ) :=
            Finset.sum_le_sum fun j _ => hle1 j
        _ = ((c.neg.erase j0).card : ℝ) := by simp
    have hcard : ((c.neg.erase j0).card : ℝ) + 1 = (c.neg.card : ℝ) := by
      exact_mod_cast Finset.card_erase_add_one hj0
    linarith

/-- **Clause row at a boolean point, falsified case.**  If `σ` falsifies `c`,
every positive variable is false and every negated variable is true, so the
pre-activation is exactly `1` and the row is `1`. -/
theorem clauseRow_boolPoint_of_not_satisfies (c : Clause ι) {σ : ι → Bool}
    (h : ¬ c.satisfies σ) : clauseRow c (boolPoint σ) = 1 := by
  unfold Clause.satisfies at h
  push Not at h
  obtain ⟨hpos, hneg⟩ := h
  have hpos0 : ∑ j ∈ c.pos, boolPoint σ j = 0 :=
    Finset.sum_eq_zero fun j hj => by
      have hf := hpos j hj
      simp only [Bool.not_eq_true] at hf
      simp [boolPoint, hf]
  have hneg1 : ∑ j ∈ c.neg, boolPoint σ j = (c.neg.card : ℝ) := by
    have hall : ∀ j ∈ c.neg, boolPoint σ j = 1 := fun j hj => by
      have ht := hneg j hj
      simp only [Bool.not_eq_false] at ht
      simp [boolPoint, ht]
    rw [Finset.sum_congr rfl hall]
    simp
  have harith : (c.neg.card : ℝ) - 0 + (1 - (c.neg.card : ℝ)) = 1 := by ring
  unfold clauseRow relu
  rw [hpos0, hneg1, harith]
  exact max_eq_right zero_le_one

/-- On boolean points the clause row is `0` **iff** the clause is satisfied
(and it is `1` otherwise, by `clauseRow_boolPoint_of_not_satisfies`). -/
theorem clauseRow_boolPoint_eq_zero_iff (c : Clause ι) (σ : ι → Bool) :
    clauseRow c (boolPoint σ) = 0 ↔ c.satisfies σ := by
  constructor
  · intro h0
    by_contra hns
    rw [clauseRow_boolPoint_of_not_satisfies c hns] at h0
    exact one_ne_zero h0
  · exact clauseRow_boolPoint_of_satisfies c

/-- **SAT witness direction.**  A satisfying assignment's 0/1 corner point
lies in the box and is UNSAFE with the exact values `Y_0 = 1`, `Y_1 = 0`
(hence `Y_0 ≥ 1 ∧ Y_1 ≤ 0`). -/
theorem boolPoint_unsafe_of_satisfies (s : Finset ι) {m : ℕ}
    (C : Fin m → Clause ι) {σ : ι → Bool} (hsat : ∀ i, (C i).satisfies σ) :
    (∀ j ∈ s, 0 ≤ boolPoint σ j ∧ boolPoint σ j ≤ 1) ∧
      1 ≤ Y0 C (boolPoint σ) ∧ Y1 s (boolPoint σ) ≤ 0 := by
  refine ⟨fun j _ => boolPoint_mem_box σ j, ?_, ?_⟩
  · unfold Y0
    have hrows : ∑ i, clauseRow (C i) (boolPoint σ) = 0 :=
      Finset.sum_eq_zero fun i _ => clauseRow_boolPoint_of_satisfies (C i) (hsat i)
    rw [hrows]
    norm_num
  · unfold Y1
    exact le_of_eq (Finset.sum_eq_zero fun j _ => bres_boolPoint σ j)

/-- **Soundness-of-UNSAT direction, witness-extraction form.**  Any point of
the `[0,1]` box (on `s`) in the unsafe region `{Y_0 ≥ 1 ∧ Y_1 ≤ 0}` yields a
boolean assignment satisfying every clause — provided every clause's variables
lie in `s`. -/
theorem exists_satisfying_of_unsafe (s : Finset ι) {m : ℕ}
    (C : Fin m → Clause ι) (hsub : ∀ i, (C i).pos ⊆ s ∧ (C i).neg ⊆ s)
    {x : ι → ℝ} (hbox : ∀ j ∈ s, 0 ≤ x j ∧ x j ≤ 1)
    (hY0 : 1 ≤ Y0 C x) (hY1 : Y1 s x ≤ 0) :
    ∃ σ : ι → Bool, ∀ i, (C i).satisfies σ := by
  classical
  -- `Y_1 ≤ 0` forces every coordinate in `s` to be boolean.
  have hbool : ∀ j ∈ s, x j = 0 ∨ x j = 1 := forces_boolean s x hbox hY1
  -- Read the boolean assignment off the point.
  set σ : ι → Bool := fun j => if x j = 1 then true else false with hσdef
  have hagree : ∀ j ∈ s, x j = boolPoint σ j := by
    intro j hj
    rcases hbool j hj with h0 | h1
    · simp [boolPoint, hσdef, h0, zero_ne_one]
    · simp [boolPoint, hσdef, h1]
  -- Clause rows only read coordinates inside `s`, where `x` is the corner point.
  have hrows : ∀ i, clauseRow (C i) x = clauseRow (C i) (boolPoint σ) := fun i =>
    clauseRow_congr (C i) (fun j hj => hagree j ((hsub i).1 hj))
      (fun j hj => hagree j ((hsub i).2 hj))
  -- `Y_0 ≥ 1` means the row sum is ≤ 0; rows are ≥ 0, so every row is 0.
  have hsum_le : ∑ i, clauseRow (C i) (boolPoint σ) ≤ 0 := by
    have hx : ∑ i, clauseRow (C i) x ≤ 0 := by
      unfold Y0 at hY0; linarith
    calc ∑ i, clauseRow (C i) (boolPoint σ)
        = ∑ i, clauseRow (C i) x := Finset.sum_congr rfl fun i _ => (hrows i).symm
      _ ≤ 0 := hx
  have hnn : ∀ i ∈ Finset.univ, (0 : ℝ) ≤ clauseRow (C i) (boolPoint σ) :=
    fun i _ => clauseRow_nonneg _ _
  have hsum0 : ∑ i, clauseRow (C i) (boolPoint σ) = 0 :=
    le_antisymm hsum_le (Finset.sum_nonneg hnn)
  refine ⟨σ, fun i => ?_⟩
  exact (clauseRow_boolPoint_eq_zero_iff (C i) σ).mp
    ((Finset.sum_eq_zero_iff_of_nonneg hnn).mp hsum0 i (Finset.mem_univ i))

/-- **THE BRIDGE.**  Over a finite variable set `s` and a finite family of
clauses whose variables all lie in `s`: the CNF is satisfiable **iff** the
gadget's unsafe region meets the `[0,1]` box.  Exact equivalence in real
arithmetic — not a relaxation in either direction. -/
theorem sat_iff_unsafe (s : Finset ι) {m : ℕ} (C : Fin m → Clause ι)
    (hsub : ∀ i, (C i).pos ⊆ s ∧ (C i).neg ⊆ s) :
    (∃ σ : ι → Bool, ∀ i, (C i).satisfies σ) ↔
      ∃ x : ι → ℝ, (∀ j ∈ s, 0 ≤ x j ∧ x j ≤ 1) ∧ 1 ≤ Y0 C x ∧ Y1 s x ≤ 0 := by
  constructor
  · rintro ⟨σ, hsat⟩
    obtain ⟨hbox, hY0, hY1⟩ := boolPoint_unsafe_of_satisfies s C hsat
    exact ⟨boolPoint σ, hbox, hY0, hY1⟩
  · rintro ⟨x, hbox, hY0, hY1⟩
    exact exists_satisfying_of_unsafe s C hsub hbox hY0 hY1

/-- **Soundness of UNSAT** (what the DRAT-certified `s UNSATISFIABLE` verdict
buys): if no boolean assignment satisfies every clause, then no point of the
`[0,1]` box lies in the unsafe region — the network property is (real-)safe. -/
theorem unsat_implies_safe (s : Finset ι) {m : ℕ} (C : Fin m → Clause ι)
    (hsub : ∀ i, (C i).pos ⊆ s ∧ (C i).neg ⊆ s)
    (hunsat : ∀ σ : ι → Bool, ∃ i, ¬ (C i).satisfies σ) :
    ∀ x : ι → ℝ, (∀ j ∈ s, 0 ≤ x j ∧ x j ≤ 1) →
      ¬(1 ≤ Y0 C x ∧ Y1 s x ≤ 0) := by
  rintro x hbox ⟨hY0, hY1⟩
  obtain ⟨σ, hsat⟩ := exists_satisfying_of_unsafe s C hsub hbox hY0 hY1
  obtain ⟨i, hi⟩ := hunsat σ
  exact hi (hsat i)

end SatRelu

end Crownproof

/-! ## Trust-base check — the CNF bridge must reduce to the standard axioms
only (no `sorryAx`). -/

#print axioms Crownproof.SatRelu.clauseRow_nonneg
#print axioms Crownproof.SatRelu.clauseRow_boolPoint_of_satisfies
#print axioms Crownproof.SatRelu.clauseRow_boolPoint_of_not_satisfies
#print axioms Crownproof.SatRelu.clauseRow_boolPoint_eq_zero_iff
#print axioms Crownproof.SatRelu.boolPoint_unsafe_of_satisfies
#print axioms Crownproof.SatRelu.exists_satisfying_of_unsafe
#print axioms Crownproof.SatRelu.sat_iff_unsafe
#print axioms Crownproof.SatRelu.unsat_implies_safe

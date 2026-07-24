/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
SPDX-License-Identifier: Apache-2.0

THE sat_relu VERDICT BRIDGE  (Route A, certified chain composition).

Composes the two halves that already live in the corpus into ONE
per-instance-applicable theorem:

  * `RupChecker.checkRefutation_sound` (this wave):  a kernel-checked LRAT/RUP
    replay proves `¬ ∃ σ, satFormula σ F` for the DIMACS formula `F`;
  * `SatRelu.unsat_implies_safe` (`SatReluCnf.lean`):  CNF-unsat ⇒ no point of
    the `[0,1]` box reaches the gadget's unsafe region `{Y₀ ≥ 1 ∧ Y₁ ≤ 0}`.

The two speak different clause dialects: the RUP development uses
`Clause = List (ℕ × Bool)` (literal lists, straight from DIMACS), while the
gadget development uses `SatRelu.Clause` (disjoint `Finset`s of positive /
negated variables, matching the Gemm rows).  `clauseOf` translates
list-clauses to finset-clauses; disjointness is forced by taking
`pos := posVars \ negVars` — for detector-emitted DIMACS a variable never
occurs twice in a clause, so this is the identity translation, and in general
the direction proven (`satisfies ⇒ satClause`) is exactly the one the
UNSAT ⇒ safe pipeline needs (a would-be unsafe box point yields a satisfying
assignment of every finset-clause, hence of every list-clause — contradicting
the kernel-checked refutation).

`safe_of_unsat` is the instance-shaped verdict: instantiated by the
`lrat_to_lean`-emitted demo files with `F` the transcribed DIMACS literal,
`s = Finset.Icc 1 n` the gadget's variable set, the `hsub` side condition
discharged by `decide`, and `hunsat := checkRefutation_sound … (by decide)`.
-/
import NyProof.RupChecker
import NyProof.SatReluCnf

namespace Crownproof

namespace SatReluVerdict

open RupImport.RUP

/-- Variables occurring POSITIVELY in a list-clause. -/
def posVars (c : RupImport.RUP.Clause) : Finset ℕ :=
  (c.filterMap fun l => if l.2 then some l.1 else none).toFinset

/-- Variables occurring NEGATED in a list-clause. -/
def negVars (c : RupImport.RUP.Clause) : Finset ℕ :=
  (c.filterMap fun l => if l.2 then none else some l.1).toFinset

theorem mem_posVars {c : RupImport.RUP.Clause} {j : ℕ} :
    j ∈ posVars c ↔ (j, true) ∈ c := by
  unfold posVars
  rw [List.mem_toFinset, List.mem_filterMap]
  constructor
  · rintro ⟨⟨v, b⟩, hmem, heq⟩
    cases b
    · simp at heq
    · simp only [if_pos] at heq
      obtain rfl : v = j := by simpa using heq
      exact hmem
  · intro hmem
    exact ⟨(j, true), hmem, by simp⟩

theorem mem_negVars {c : RupImport.RUP.Clause} {j : ℕ} :
    j ∈ negVars c ↔ (j, false) ∈ c := by
  unfold negVars
  rw [List.mem_toFinset, List.mem_filterMap]
  constructor
  · rintro ⟨⟨v, b⟩, hmem, heq⟩
    cases b
    · obtain rfl : v = j := by simpa using heq
      exact hmem
    · simp at heq
  · intro hmem
    exact ⟨(j, false), hmem, by simp⟩

/-- Translate a list-clause to a gadget finset-clause.  Disjointness is forced
by subtracting `negVars` from the positive side; detector-emitted DIMACS never
repeats a variable inside a clause, so no positive literal is actually lost
(and soundness of the UNSAT direction below holds regardless). -/
def clauseOf (c : RupImport.RUP.Clause) : SatRelu.Clause ℕ where
  pos := posVars c \ negVars c
  neg := negVars c
  disj := Finset.sdiff_disjoint

/-- The one direction the verdict chain needs: an assignment satisfying the
finset-clause satisfies the original list-clause. -/
theorem satClause_of_satisfies (c : RupImport.RUP.Clause) (σ : ℕ → Bool)
    (h : (clauseOf c).satisfies σ) : satClause σ c := by
  rcases h with ⟨j, hj, hσ⟩ | ⟨j, hj, hσ⟩
  · exact ⟨(j, true), mem_posVars.mp (Finset.mem_sdiff.mp hj).1, hσ⟩
  · exact ⟨(j, false), mem_negVars.mp hj, hσ⟩

/-- The gadget clause family of a DIMACS formula, indexed by clause position. -/
def clausesOf (F : RupImport.RUP.Formula) : Fin F.length → SatRelu.Clause ℕ :=
  fun i => clauseOf (F.get i)

/-- **THE VERDICT.**  If the DIMACS formula is unsatisfiable (e.g. by
`RupChecker.checkRefutation_sound` on a kernel-replayed LRAT refutation), then
no point of the `[0,1]` box on `s` reaches the sat_relu gadget's unsafe region
`{Y₀ ≥ 1 ∧ Y₁ ≤ 0}` — the property is (real-arithmetic) SAFE. -/
theorem safe_of_unsat (F : RupImport.RUP.Formula) (s : Finset ℕ)
    (hsub : ∀ i, (clausesOf F i).pos ⊆ s ∧ (clausesOf F i).neg ⊆ s)
    (hunsat : ¬ ∃ σ : Assign, satFormula σ F) :
    ∀ x : ℕ → ℝ, (∀ j ∈ s, 0 ≤ x j ∧ x j ≤ 1) →
      ¬ (1 ≤ SatRelu.Y0 (clausesOf F) x ∧ SatRelu.Y1 s x ≤ 0) := by
  apply SatRelu.unsat_implies_safe s (clausesOf F) hsub
  intro σ
  by_contra hcon
  push Not at hcon
  apply hunsat
  refine ⟨σ, fun C hC => ?_⟩
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hC
  exact satClause_of_satisfies _ σ (hcon ⟨i, hi⟩)

end SatReluVerdict

end Crownproof

/-! ## Trust-base check -/

#print axioms Crownproof.SatReluVerdict.satClause_of_satisfies
#print axioms Crownproof.SatReluVerdict.safe_of_unsat

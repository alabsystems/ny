/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
SPDX-License-Identifier: Apache-2.0

EXECUTABLE LRAT/RUP REFUTATION CHECKER  (sat_relu Route A, certified core).

`AristotleLemmas.lean` (batch 5) proves the ABSTRACT per-step rule
`RupImport.RUP.rup_sound`: if clause `C` is RUP with respect to formula `F`
(negate `C`, unit-propagate clauses of `F`, reach a conflict) then `F` entails
`C`.  That theorem is stated over a Prop-level `IsRUP` witness — it says
nothing about how to CHECK a concrete LRAT artifact.

This file closes that gap with the corpus's `BabProof.lean` discipline: a
total, DECIDABLE checker over `List`/`Nat`/`Bool` data only

  * `RStep`            — one LRAT addition line: the derived clause plus its
                         unit-propagation hints (0-based indices into the
                         current clause database);
  * `checkHints`       — replays one hint chain: each hint clause must be
                         falsified (conflict — accept) or unit (assert the
                         surviving literal and continue);
  * `checkStep`        — starts the replay from the negation of the derived
                         clause (the RUP rule);
  * `checkRefutation`  — threads the growing clause database (original formula,
                         then each checked derived clause in order) and accepts
                         when a checked step derives the EMPTY clause;

and the soundness theorem

  * `checkRefutation_sound : checkRefutation F steps = true →
                             ¬ ∃ σ, satFormula σ F`

proven by constructing, for every accepted step, the `IsRUP` witness
(`checkHints_sound` turns the Bool replay into the `Relation.ReflTransGen
(Step F)` propagation chain plus the conflict clause) and composing
`RupImport.RUP.rup_sound` along the database growth
(`satFormula σ db → satFormula σ (db ++ [derived])`), until the empty clause —
satisfiable by no assignment — is derived.

The checker itself is pure structural recursion (no `Classical`, no `Real`,
no `Finset`): it reduces inside the kernel under plain `decide`.  As with
`BabProof.checkBabProof`, kernel reduction is only cheap for small objects;
the sat_relu Route A refutations (≤ ~120 vars, a few hundred LRAT lines) are
measured in `SatReluDemo_*.lean`.  `native_decide` is NOT used anywhere.

Trusted plumbing that remains OUTSIDE the kernel: parsing the DIMACS + LRAT
artifacts into the `Formula`/`List RStep` literals (done by
`ny-cert`'s `lrat_to_lean` binary) — a syntactic transcription, no reasoning.
-/
import NyProof.AristotleLemmas

namespace Crownproof

namespace RupChecker

open RupImport.RUP

/-- One imported LRAT addition line: the `clause` being derived and the list of
unit-propagation `hints`, each a 0-based index into the CURRENT clause database
(the original formula followed by all previously derived clauses, in order).
The `lrat_to_lean` importer resolves LRAT clause ids to database indices;
deletion lines are dropped (sound: propagation over a superset database only
has MORE clauses available, and every hint is looked up explicitly). -/
structure RStep where
  /-- The derived clause (empty for the final conflict step). -/
  clause : Clause
  /-- Unit-propagation hints: 0-based indices into the clause database. -/
  hints : List Nat
deriving Repr, DecidableEq

/-- Replay one RUP hint chain.  `L` is the list of asserted literals (the
negated derived clause plus everything propagated so far).  For each hint,
look up the database clause and keep the literals NOT falsified by `L`:

* none survive — the clause is falsified: CONFLICT, accept;
* exactly one survives and is not yet asserted — a unit: assert it, continue;
* exactly one survives but is already asserted — redundant hint: skip it;
* otherwise — the hint is not unit: reject.

Runs out of hints without a conflict: reject. -/
def checkHints (F : Formula) : List Lit → List Nat → Bool
  | _, [] => false
  | L, i :: is =>
    match F[i]? with
    | none => false
    | some D =>
      match D.filter (fun l => !decide (negLit l ∈ L)) with
      | [] => true
      | [l] => if l ∈ L then checkHints F L is else checkHints F (l :: L) is
      | _ :: _ :: _ => false

/-- Check one LRAT addition step against the database `F`: the RUP replay
starts from the negations of the derived clause's literals. -/
def checkStep (F : Formula) (s : RStep) : Bool :=
  checkHints F (s.clause.map negLit) s.hints

/-- Check a whole refutation.  The database starts as the original formula;
each checked step's clause joins the database; accept as soon as a checked
step derives the EMPTY clause.  An exhausted step list rejects. -/
def checkRefutation (F : Formula) : List RStep → Bool
  | [] => false
  | s :: ss =>
    checkStep F s && (s.clause.isEmpty || checkRefutation (F ++ [s.clause]) ss)

/-! ## Soundness -/

private theorem mem_of_getElem? {α : Type*} {l : List α} {i : Nat} {a : α}
    (h : l[i]? = some a) : a ∈ l := by
  obtain ⟨hi, hEq⟩ := List.getElem?_eq_some_iff.mp h
  exact hEq ▸ List.getElem_mem hi

/-- A successful `checkHints` replay is a genuine unit-propagation chain: it
yields the final asserted list `Lk` reachable from `L` by `Step F`, together
with the conflict clause of `F` falsified under `Lk`.  This is exactly the
body of the `IsRUP` witness. -/
theorem checkHints_sound (F : Formula) :
    ∀ (hs : List Nat) (L : List Lit), checkHints F L hs = true →
      ∃ Lk, Relation.ReflTransGen (Step F) L Lk ∧
        ∃ D ∈ F, ∀ l' ∈ D, negLit l' ∈ Lk := by
  intro hs
  induction hs with
  | nil => intro L h; simp [checkHints] at h
  | cons i is ih =>
    intro L h
    unfold checkHints at h
    split at h
    · exact absurd h (by simp)
    · rename_i D hD?
      have hDmem : D ∈ F := mem_of_getElem? hD?
      split at h
      · -- CONFLICT: every literal of `D` is falsified by `L`.
        rename_i hfilter
        refine ⟨L, Relation.ReflTransGen.refl, D, hDmem, fun l' hl' => ?_⟩
        have := List.filter_eq_nil_iff.mp hfilter l' hl'
        simpa using this
      · -- UNIT: exactly one literal `l` of `D` survives.
        rename_i l hfilter
        have hlfilter : l ∈ D.filter (fun l' => !decide (negLit l' ∈ L)) := by
          rw [hfilter]; exact List.mem_singleton_self l
        have hlD : l ∈ D := List.mem_of_mem_filter hlfilter
        have hlneg : negLit l ∉ L := by
          have := List.of_mem_filter hlfilter
          simpa using this
        split at h
        · -- Redundant hint (`l` already asserted): the chain does not move.
          exact ih L h
        · -- Genuine unit: `Step F L (l :: L)`, then recurse.
          rename_i hlL
          have hothers : ∀ l' ∈ D, l' ≠ l → negLit l' ∈ L := by
            intro l' hl' hne
            by_contra hcon
            have : l' ∈ D.filter (fun l'' => !decide (negLit l'' ∈ L)) :=
              List.mem_filter.mpr ⟨hl', by simpa using hcon⟩
            rw [hfilter] at this
            exact hne (List.mem_singleton.mp this)
          have hstep : RupImport.RUP.Step F L (l :: L) :=
            ⟨D, l, ⟨hDmem, hlD, hothers, hlL, hlneg⟩, rfl⟩
          obtain ⟨Lk, hchain, hconf⟩ := ih (l :: L) h
          exact ⟨Lk, Relation.ReflTransGen.head hstep hchain, hconf⟩
      · exact absurd h (by simp)

/-- A checked step's clause is RUP with respect to the database. -/
theorem checkStep_isRUP (F : Formula) (s : RStep)
    (h : checkStep F s = true) : IsRUP F s.clause :=
  checkHints_sound F s.hints (s.clause.map negLit) h

/-- A checked step's clause is ENTAILED by the database
(via `RupImport.RUP.rup_sound`). -/
theorem checkStep_entails (F : Formula) (s : RStep)
    (h : checkStep F s = true) :
    ∀ σ, satFormula σ F → satClause σ s.clause :=
  rup_sound F s.clause (checkStep_isRUP F s h)

private theorem satFormula_append_singleton (σ : Assign) (F : Formula)
    (C : Clause) (hF : satFormula σ F) (hC : satClause σ C) :
    satFormula σ (F ++ [C]) := by
  intro D hD
  rcases List.mem_append.mp hD with hmem | hmem
  · exact hF D hmem
  · rw [List.mem_singleton.mp hmem]; exact hC

/-- **SOUNDNESS.**  A checked refutation proves the formula unsatisfiable:
by induction on the step list, every satisfying assignment of the current
database would also satisfy each derived clause (`checkStep_entails`), hence
the grown database — until the empty clause, which no assignment satisfies. -/
theorem checkRefutation_sound (F : Formula) (steps : List RStep)
    (h : checkRefutation F steps = true) : ¬ ∃ σ, satFormula σ F := by
  induction steps generalizing F with
  | nil => simp [checkRefutation] at h
  | cons s ss ih =>
    rintro ⟨σ, hσ⟩
    simp only [checkRefutation, Bool.and_eq_true, Bool.or_eq_true] at h
    obtain ⟨hstep, hrest⟩ := h
    have hC : satClause σ s.clause := checkStep_entails F s hstep σ hσ
    rcases hrest with hemp | hrec
    · rw [List.isEmpty_iff.mp hemp] at hC
      obtain ⟨l, hl, -⟩ := hC
      exact absurd hl (List.not_mem_nil)
    · exact ih (F ++ [s.clause]) hrec
        ⟨σ, satFormula_append_singleton σ F s.clause hσ hC⟩

/-! ## End-to-end micro-demo (the `BabProof.tiny_checks` discipline).

`{x₁} ∧ {¬x₁}`: negating the empty clause asserts nothing; hint 0 (clause
`[x₁]`) is unit and asserts `x₁`; hint 1 (clause `[¬x₁]`) is then falsified —
conflict.  The full `decide` route is exhibited honestly on this tiny instance
here; realistic sat_relu instances are emitted by `lrat_to_lean` and measured
in `SatReluDemo_*.lean`. -/

/-- Micro formula `{x₁} ∧ {¬x₁}`. -/
def tinyF : Formula := [[(1, true)], [(1, false)]]

/-- Its one-step RUP refutation: derive the empty clause with hints `[0, 1]`. -/
def tinySteps : List RStep := [⟨[], [0, 1]⟩]

/-- The checker accepts — by KERNEL reduction (`decide`, not `native_decide`). -/
theorem tiny_checks : checkRefutation tinyF tinySteps = true := by decide

/-- …and therefore the micro formula is unsatisfiable. -/
theorem tiny_unsat : ¬ ∃ σ, satFormula σ tinyF :=
  checkRefutation_sound tinyF tinySteps tiny_checks

end RupChecker

end Crownproof

/-! ## Trust-base check — checker soundness must reduce to the standard
axioms only (no `sorryAx`, no `Lean.ofReduceBool`). -/

#print axioms Crownproof.RupChecker.checkHints_sound
#print axioms Crownproof.RupChecker.checkStep_isRUP
#print axioms Crownproof.RupChecker.checkStep_entails
#print axioms Crownproof.RupChecker.checkRefutation_sound
#print axioms Crownproof.RupChecker.tiny_checks
#print axioms Crownproof.RupChecker.tiny_unsat

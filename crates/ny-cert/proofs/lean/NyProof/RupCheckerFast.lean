/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
SPDX-License-Identifier: Apache-2.0

FAST EXECUTABLE LRAT/RUP REFUTATION CHECKER  (sat_relu Route A, scaled core).

`RupChecker.lean` proves the checker discipline end-to-end, but its data
representations are the naive ones: the clause database is a `List Clause`
walked by `List.get?` on every hint, and the asserted-literal set is a
`List Lit` scanned linearly per literal test.  Kernel `decide` on a realistic
refutation therefore costs O(steps × hints × db × |L|) list-cell reductions —
measured to top out between `p cnf 92 117` (≈10 s) and `p cnf 100 373`
(did not finish in ~15 min).

This file is the SAME checker with kernel-fast data representations.  The
interface is unchanged — it consumes the very same `Formula` and
`List RStep` literals that `lrat_to_lean` emits — but internally:

  * literals are Nat codes `2*v + polarity`; negation flips the low bit
    (`negCode`, plain kernel-accelerated arithmetic);
  * the asserted-literal set is a `Nat` BITMASK: membership is
    `Nat.testBit` (one GMP shift+and), insertion is `A ||| (1 <<< c)` —
    O(1) instead of the O(|L|) list scan;
  * the clause database is a binary TRIE keyed by 1-based index in
    binary-heap navigation (`Db`): hint lookup and append are O(log n)
    constructor steps instead of O(n) `List.get?` walks.

Everything is STRUCTURAL recursion (trie lookup recurses on the trie,
insertion on an explicit fuel argument — `WellFounded.fix` would get stuck in
the kernel), so the whole replay reduces under plain `decide`.
`native_decide` is NOT used anywhere.

Soundness (`checkRefutationFast_sound`) is proven DIRECTLY by the same
`IsRUP`-witness construction as `RupChecker.checkHints_sound`, carrying two
invariants through the replay:

  * `Corr A L` — bit `encLit l` of the mask is set iff `l` is in the
    (abstract) asserted list `L`;
  * `Db.All P db` — every clause stored in the trie is (the encoding of) a
    clause of the current formula; note soundness needs only MEMBERSHIP of
    stored clauses, never index-correctness of insertion, which is why the
    fuel-based `setK` (a no-op on fuel exhaustion) is fail-safe.

The final composition goes through `RupImport.RUP.rup_sound` exactly as in
`RupChecker.checkRefutation_sound`, so the axiom footprint is identical
(`propext`, `Classical.choice`, `Quot.sound` — checked at the bottom).
-/
import NyProof.RupChecker

namespace Crownproof

namespace RupCheckerFast

open RupImport.RUP
open RupChecker (RStep)

/-! ## Literal codes -/

/-- Encoded clause: each literal `(v, b)` as the Nat code `2*v + b`. -/
abbrev EClause := List Nat

/-- Literal code: `2*v + 1` for a positive literal `(v, true)`, `2*v` for a
negated one. -/
def encLit (l : Lit) : Nat := 2 * l.1 + (if l.2 then 1 else 0)

/-- Encode a clause literal-by-literal. -/
def encClause (C : Clause) : EClause := C.map encLit

/-- Negation on codes: flip the low bit (kernel-fast `%`/`±1` arithmetic). -/
def negCode (c : Nat) : Nat := if c % 2 = 1 then c - 1 else c + 1

theorem encLit_negLit (l : Lit) : encLit (negLit l) = negCode (encLit l) := by
  rcases l with ⟨v, b⟩
  cases b
  · show (2 * v + 1 : Nat) = negCode (2 * v + 0)
    unfold negCode
    split <;> omega
  · show (2 * v + 0 : Nat) = negCode (2 * v + 1)
    unfold negCode
    split <;> omega

theorem encLit_injective : Function.Injective encLit := by
  rintro ⟨v, b⟩ ⟨w, c⟩ h
  simp only [encLit] at h
  cases b <;> cases c <;> simp_all <;> omega

/-! ## The clause-database trie -/

/-- Clause database: a binary trie keyed by 1-based index in binary-heap
navigation (key 1 at the root; even keys descend left, odd keys right, key
halving each level).  Lookup and insertion are O(log n) kernel reductions. -/
inductive Db where
  | empty : Db
  | node (val : Option EClause) (l r : Db) : Db

namespace Db

/-- Lookup by 1-based key.  Structural in the trie (kernel-reducible). -/
def getK : Db → Nat → Option EClause
  | .empty, _ => none
  | .node v l r, k =>
    if k ≤ 1 then v
    else if k % 2 = 0 then getK l (k / 2) else getK r (k / 2)

/-- Insert at 1-based key.  Structural in the explicit `fuel` argument
(kernel-reducible; `WellFounded.fix` on the key would get stuck).  `fuel ≥ k`
always suffices; on fuel exhaustion the trie is returned UNCHANGED, which is
fail-safe: soundness (`All`, below) never depends on an insert landing. -/
def setK : Nat → Db → Nat → EClause → Db
  | 0, t, _, _ => t
  | fuel + 1, t, k, c =>
    match t with
    | .empty =>
      if k ≤ 1 then .node (some c) .empty .empty
      else if k % 2 = 0 then .node none (setK fuel .empty (k / 2) c) .empty
      else .node none .empty (setK fuel .empty (k / 2) c)
    | .node v l r =>
      if k ≤ 1 then .node (some c) l r
      else if k % 2 = 0 then .node v (setK fuel l (k / 2) c) r
      else .node v l (setK fuel r (k / 2) c)

/-- 0-based read (LRAT hints are 0-based database indices). -/
def get (t : Db) (i : Nat) : Option EClause := getK t (i + 1)

/-- 0-based write; fuel `i + 1` is always enough for key `i + 1`. -/
def set (t : Db) (i : Nat) (c : EClause) : Db := setK (i + 1) t (i + 1) c

/-- Every clause stored in the trie satisfies `P`. -/
def All (P : EClause → Prop) : Db → Prop
  | .empty => True
  | .node v l r => (∀ D, v = some D → P D) ∧ All P l ∧ All P r

theorem All.empty {P : EClause → Prop} : All P Db.empty := trivial

theorem All.imp {P Q : EClause → Prop} (h : ∀ D, P D → Q D) :
    ∀ {t : Db}, All P t → All Q t := by
  intro t
  induction t with
  | empty => intro _; trivial
  | node v l r ihl ihr =>
    rintro ⟨hv, hl, hr⟩
    exact ⟨fun D hD => h D (hv D hD), ihl hl, ihr hr⟩

theorem All.of_getK {P : EClause → Prop} :
    ∀ {t : Db}, All P t → ∀ {k : Nat} {D : EClause}, t.getK k = some D → P D := by
  intro t
  induction t with
  | empty => intro _ k D h; simp [getK] at h
  | node v l r ihl ihr =>
    rintro ⟨hv, hl, hr⟩ k D h
    unfold getK at h
    split at h
    · exact hv D h
    · split at h
      · exact ihl hl h
      · exact ihr hr h

/-- Trie reads only ever return stored clauses (0-based wrapper). -/
theorem All.of_get {P : EClause → Prop} {t : Db} (ht : All P t) {i : Nat}
    {D : EClause} (h : t.get i = some D) : P D :=
  ht.of_getK h

theorem All.setK {P : EClause → Prop} {c : EClause} (hc : P c) :
    ∀ (fuel : Nat) {t : Db}, All P t → ∀ (k : Nat), All P (Db.setK fuel t k c) := by
  intro fuel
  induction fuel with
  | zero => intro t ht _; exact ht
  | succ n ih =>
    intro t ht k
    cases t with
    | empty =>
      simp only [Db.setK]
      split
      · exact ⟨fun D hD => Option.some.inj hD ▸ hc, All.empty, All.empty⟩
      · split
        · exact ⟨fun D hD => by simp at hD, ih All.empty _, All.empty⟩
        · exact ⟨fun D hD => by simp at hD, All.empty, ih All.empty _⟩
    | node v l r =>
      obtain ⟨hv, hl, hr⟩ := ht
      simp only [Db.setK]
      split
      · exact ⟨fun D hD => Option.some.inj hD ▸ hc, hl, hr⟩
      · split
        · exact ⟨hv, ih hl _, hr⟩
        · exact ⟨hv, hl, ih hr _⟩

/-- Insertion preserves `All` (0-based wrapper). -/
theorem All.set {P : EClause → Prop} {c : EClause} (hc : P c) {t : Db}
    (ht : All P t) (i : Nat) : All P (t.set i c) :=
  All.setK hc (i + 1) ht (i + 1)

end Db

/-! ## The fast checker

The asserted-literal set is a `Nat` bitmask `A`: bit `encLit l` set ⇔ `l`
asserted.  Membership is `Nat.testBit` (kernel-accelerated GMP shift/and),
insertion is `A ||| (1 <<< c)`. -/

/-- Replay one RUP hint chain — the fast mirror of `RupChecker.checkHints`:
per hint, look the clause up in the trie, drop the literals falsified by the
mask; none survive ⇒ conflict (accept), one survives ⇒ unit (assert it or
skip if already asserted), else reject. -/
def checkHints (db : Db) : Nat → List Nat → Bool
  | _, [] => false
  | A, i :: is =>
    match db.get i with
    | none => false
    | some D =>
      match D.filter (fun c => !A.testBit (negCode c)) with
      | [] => true
      | [c] =>
        if A.testBit c then checkHints db A is
        else checkHints db (A ||| (1 <<< c)) is
      | _ :: _ :: _ => false

/-- Bitmask asserting the NEGATION of every literal of `C` (the RUP start
state for deriving `C`). -/
def initMask (C : Clause) : Nat :=
  C.foldl (fun A l => A ||| (1 <<< encLit (negLit l))) 0

/-- Check one LRAT addition step against the trie database. -/
def checkStep (db : Db) (s : RStep) : Bool :=
  checkHints db (initMask s.clause) s.hints

/-- Main loop: `len` is the current database size (= next free 0-based
index); each checked step's clause is inserted there. -/
def checkRefutationGo : Db → Nat → List RStep → Bool
  | _, _, [] => false
  | db, len, s :: ss =>
    checkStep db s &&
      (s.clause.isEmpty ||
        checkRefutationGo (db.set len (encClause s.clause)) (len + 1) ss)

/-- Load the original formula into the trie at indices `i, i+1, …`. -/
def dbOfAux : Db → Nat → List Clause → Db
  | db, _, [] => db
  | db, i, C :: Cs => dbOfAux (db.set i (encClause C)) (i + 1) Cs

/-- The trie holding the original formula (indices `0 .. F.length - 1`). -/
def dbOf (F : Formula) : Db := dbOfAux .empty 0 F

/-- **THE FAST CHECKER** — same Bool interface as
`RupChecker.checkRefutation`: it consumes the very same `Formula` and
`RStep` list emitted by `lrat_to_lean`. -/
def checkRefutationFast (F : Formula) (steps : List RStep) : Bool :=
  checkRefutationGo (dbOf F) F.length steps

/-! ## Soundness -/

/-- Mask/list correspondence: bit `encLit l` of `A` is set iff `l ∈ L`. -/
def Corr (A : Nat) (L : List Lit) : Prop :=
  ∀ l : Lit, A.testBit (encLit l) = true ↔ l ∈ L

theorem Corr.insert {A : Nat} {L : List Lit} (h : Corr A L) (l : Lit) :
    Corr (A ||| (1 <<< encLit l)) (l :: L) := by
  intro l'
  simp only [Nat.testBit_or, Nat.one_shiftLeft, Nat.testBit_two_pow,
    Bool.or_eq_true, decide_eq_true_eq, List.mem_cons, h l']
  constructor
  · rintro (hmem | heq)
    · exact Or.inr hmem
    · exact Or.inl (encLit_injective heq).symm
  · rintro (rfl | hmem)
    · exact Or.inr rfl
    · exact Or.inl hmem

/-- Under `Corr A L`, the fast falsified-literal filter over an encoded
clause is the encoding of `RupChecker.checkHints`' abstract filter. -/
theorem filter_encClause {A : Nat} {L : List Lit} (h : Corr A L) (C : Clause) :
    (encClause C).filter (fun c => !A.testBit (negCode c))
      = (C.filter (fun l => !decide (negLit l ∈ L))).map encLit := by
  unfold encClause
  rw [List.filter_map]
  congr 1
  apply List.filter_congr
  intro l _
  simp only [Function.comp_apply, ← encLit_negLit]
  cases hb : A.testBit (encLit (negLit l)) with
  | true => simp [(h (negLit l)).mp hb]
  | false =>
    have hnm : negLit l ∉ L := fun hm => by simp [(h (negLit l)).mpr hm] at hb
    simp [hnm]

private theorem map_encLit_eq_singleton {C : List Lit} {c : Nat}
    (h : C.map encLit = [c]) : ∃ l, C = [l] ∧ encLit l = c := by
  cases C with
  | nil => simp at h
  | cons a t =>
    cases t with
    | nil =>
      simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at h
      exact ⟨a, rfl, h⟩
    | cons b t' => simp at h

/-- A successful fast replay is a genuine unit-propagation chain — the same
`IsRUP`-witness construction as `RupChecker.checkHints_sound`, transported
through the mask (`Corr`) and trie (`Db.All`) invariants. -/
theorem checkHints_sound (F : Formula) (db : Db)
    (hdb : Db.All (fun D => ∃ C ∈ F, D = encClause C) db) :
    ∀ (hs : List Nat) (A : Nat) (L : List Lit), Corr A L →
      checkHints db A hs = true →
      ∃ Lk, Relation.ReflTransGen (Step F) L Lk ∧
        ∃ D ∈ F, ∀ l' ∈ D, negLit l' ∈ Lk := by
  intro hs
  induction hs with
  | nil => intro A L _ h; simp [checkHints] at h
  | cons i is ih =>
    intro A L hCorr h
    unfold checkHints at h
    split at h
    · exact absurd h (by simp)
    · rename_i D hD?
      obtain ⟨C, hCF, rfl⟩ := hdb.of_get hD?
      rw [filter_encClause hCorr C] at h
      split at h
      · -- CONFLICT: every literal of `C` is falsified by `L`.
        rename_i hfilter
        have hq : C.filter (fun l' => !decide (negLit l' ∈ L)) = [] :=
          List.map_eq_nil_iff.mp hfilter
        refine ⟨L, Relation.ReflTransGen.refl, C, hCF, fun l' hl' => ?_⟩
        have := List.filter_eq_nil_iff.mp hq l' hl'
        simpa using this
      · -- UNIT: exactly one literal of `C` survives.
        rename_i c hfilter
        obtain ⟨l, hfl, hcl⟩ := map_encLit_eq_singleton hfilter
        rw [← hcl] at h
        have hlfilter : l ∈ C.filter (fun l' => !decide (negLit l' ∈ L)) := by
          rw [hfl]; exact List.mem_singleton_self l
        have hlD : l ∈ C := List.mem_of_mem_filter hlfilter
        have hlneg : negLit l ∉ L := by
          have := List.of_mem_filter hlfilter
          simpa using this
        split at h
        · -- Redundant hint (`l` already asserted): the chain does not move.
          exact ih A L hCorr h
        · -- Genuine unit: `Step F L (l :: L)`, then recurse.
          rename_i hbit
          have hlL : l ∉ L := fun hm => hbit ((hCorr l).mpr hm)
          have hothers : ∀ l' ∈ C, l' ≠ l → negLit l' ∈ L := by
            intro l' hl' hne
            by_contra hcon
            have : l' ∈ C.filter (fun l'' => !decide (negLit l'' ∈ L)) :=
              List.mem_filter.mpr ⟨hl', by simpa using hcon⟩
            rw [hfl] at this
            exact hne (List.mem_singleton.mp this)
          have hstep : RupImport.RUP.Step F L (l :: L) :=
            ⟨C, l, ⟨hCF, hlD, hothers, hlL, hlneg⟩, rfl⟩
          obtain ⟨Lk, hchain, hconf⟩ := ih _ (l :: L) (hCorr.insert l) h
          exact ⟨Lk, Relation.ReflTransGen.head hstep hchain, hconf⟩
      · exact absurd h (by simp)

/-- `initMask C` corresponds exactly to the negated-clause start list of the
abstract RUP replay. -/
theorem corr_initMask (C : Clause) : Corr (initMask C) (C.map negLit) := by
  unfold initMask
  suffices h : ∀ (A : Nat) (l' : Lit),
      (C.foldl (fun A l => A ||| (1 <<< encLit (negLit l))) A).testBit
          (encLit l') = true
        ↔ l' ∈ C.map negLit ∨ A.testBit (encLit l') = true by
    intro l'
    rw [h 0 l']
    simp
  induction C with
  | nil => intro A l'; simp
  | cons a C ihC =>
    intro A l'
    simp only [List.foldl_cons, List.map_cons, List.mem_cons]
    rw [ihC, Nat.testBit_or, Nat.one_shiftLeft, Nat.testBit_two_pow]
    simp only [Bool.or_eq_true, decide_eq_true_eq]
    constructor
    · rintro (hmem | (hb | heq))
      · exact Or.inl (Or.inr hmem)
      · exact Or.inr hb
      · exact Or.inl (Or.inl (encLit_injective heq).symm)
    · rintro ((rfl | hmem) | hb)
      · exact Or.inr (Or.inr rfl)
      · exact Or.inl hmem
      · exact Or.inr (Or.inl hb)

/-- A checked step's clause is RUP with respect to the formula the trie
represents. -/
theorem checkStep_isRUP (F : Formula) (db : Db)
    (hdb : Db.All (fun D => ∃ C ∈ F, D = encClause C) db)
    (s : RStep) (h : checkStep db s = true) : IsRUP F s.clause :=
  checkHints_sound F db hdb s.hints (initMask s.clause) (s.clause.map negLit)
    (corr_initMask s.clause) h

private theorem satFormula_append_singleton (σ : Assign) (F : Formula)
    (C : Clause) (hF : satFormula σ F) (hC : satClause σ C) :
    satFormula σ (F ++ [C]) := by
  intro D hD
  rcases List.mem_append.mp hD with hmem | hmem
  · exact hF D hmem
  · rw [List.mem_singleton.mp hmem]; exact hC

/-- Soundness of the main loop: by induction on the step list, exactly as
`RupChecker.checkRefutation_sound`, threading the trie invariant. -/
theorem checkRefutationGo_sound :
    ∀ (steps : List RStep) (F : Formula) (db : Db) (len : Nat),
      Db.All (fun D => ∃ C ∈ F, D = encClause C) db →
      checkRefutationGo db len steps = true →
      ¬ ∃ σ, satFormula σ F := by
  intro steps
  induction steps with
  | nil => intro F db len _ h; simp [checkRefutationGo] at h
  | cons s ss ih =>
    intro F db len hdb h
    rintro ⟨σ, hσ⟩
    simp only [checkRefutationGo, Bool.and_eq_true, Bool.or_eq_true] at h
    obtain ⟨hstep, hrest⟩ := h
    have hC : satClause σ s.clause :=
      rup_sound F s.clause (checkStep_isRUP F db hdb s hstep) σ hσ
    rcases hrest with hemp | hrec
    · rw [List.isEmpty_iff.mp hemp] at hC
      obtain ⟨l, hl, -⟩ := hC
      exact absurd hl List.not_mem_nil
    · have hdb' : Db.All (fun D => ∃ C ∈ F ++ [s.clause], D = encClause C)
          (db.set len (encClause s.clause)) := by
        refine Db.All.set ⟨s.clause, by simp, rfl⟩ ?_ len
        exact hdb.imp fun D ⟨C, hCmem, hE⟩ =>
          ⟨C, List.mem_append_left _ hCmem, hE⟩
      exact ih (F ++ [s.clause]) _ (len + 1) hdb' hrec
        ⟨σ, satFormula_append_singleton σ F s.clause hσ hC⟩

/-- The loaded trie stores only (encodings of) clauses of `F`. -/
theorem dbOfAux_all (P : EClause → Prop) :
    ∀ (Cs : List Clause) (db : Db) (i : Nat),
      Db.All P db → (∀ C ∈ Cs, P (encClause C)) →
      Db.All P (dbOfAux db i Cs) := by
  intro Cs
  induction Cs with
  | nil => intro db i hdb _; exact hdb
  | cons C Cs ih =>
    intro db i hdb hCs
    exact ih _ _ (hdb.set (hCs C (by simp)) i)
      (fun C' h => hCs C' (List.mem_cons_of_mem _ h))

/-- **SOUNDNESS.**  A fast-checked refutation proves the formula
unsatisfiable — same statement shape as `RupChecker.checkRefutation_sound`,
so it composes with `SatReluVerdict.safe_of_unsat` unchanged. -/
theorem checkRefutationFast_sound (F : Formula) (steps : List RStep)
    (h : checkRefutationFast F steps = true) : ¬ ∃ σ, satFormula σ F :=
  checkRefutationGo_sound steps F (dbOf F) F.length
    (dbOfAux_all _ F .empty 0 Db.All.empty fun C hC => ⟨C, hC, rfl⟩) h

/-! ## End-to-end micro-demo (the `RupChecker.tiny_checks` discipline).

Same micro instance `{x₁} ∧ {¬x₁}` as `RupChecker`, replayed through the
fast representations by KERNEL reduction. -/

/-- Micro formula `{x₁} ∧ {¬x₁}`. -/
def tinyF : Formula := [[(1, true)], [(1, false)]]

/-- Its one-step RUP refutation: derive the empty clause with hints `[0, 1]`. -/
def tinySteps : List RStep := [⟨[], [0, 1]⟩]

/-- The fast checker accepts — by KERNEL reduction (`decide`, not
`native_decide`). -/
theorem tiny_checks : checkRefutationFast tinyF tinySteps = true := by decide

/-- …and therefore the micro formula is unsatisfiable. -/
theorem tiny_unsat : ¬ ∃ σ, satFormula σ tinyF :=
  checkRefutationFast_sound tinyF tinySteps tiny_checks

end RupCheckerFast

end Crownproof

/-! ## Trust-base check — fast-checker soundness must reduce to the standard
axioms only (no `sorryAx`, no `Lean.ofReduceBool`). -/

#print axioms Crownproof.RupCheckerFast.checkHints_sound
#print axioms Crownproof.RupCheckerFast.checkStep_isRUP
#print axioms Crownproof.RupCheckerFast.checkRefutationGo_sound
#print axioms Crownproof.RupCheckerFast.checkRefutationFast_sound
#print axioms Crownproof.RupCheckerFast.tiny_checks
#print axioms Crownproof.RupCheckerFast.tiny_unsat

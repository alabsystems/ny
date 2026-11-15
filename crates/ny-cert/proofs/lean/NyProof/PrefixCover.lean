/-
  Copyright 2026 Andrew Yates
  SPDX-License-Identifier: Apache-2.0

  NEW reconstruction, 2026-10-08. This is not the missing 10,738-byte source
  identified by SHA256 81ef034d... in the historical 2026-10-01 report.

  Structural coverage of a destination and its first-disagreement siblings.
  Closed phase predicates need only be exhaustive: they can overlap at zero.
  Every safety/bound conclusion explicitly assumes valid premises for EVERY
  complete member. There is no theorem about the floating point row evaluator,
  source authentication, capacity ledger, Rust machine code, or numeric pruning.

  The executable descriptor model below mirrors prefix_cover.rs's destination
  first, then siblings at parent.length + j + 1. Its correspondence theorem is
  between Lean definitions; authenticating the Rust implementation against that
  model is a separate obligation. No Clean sources or dependencies are copied.
-/
import Std

namespace NyProof.PrefixCover

universe u v

inductive Side where
  | off
  | on
  deriving DecidableEq, Repr, Inhabited

def Side.opposite : Side → Side
  | .off => .on
  | .on => .off

structure Literal where
  site : Nat
  side : Side
  deriving DecidableEq, Repr, Inhabited

def Literal.opposite (literal : Literal) : Literal :=
  { literal with side := literal.side.opposite }

@[simp] theorem opposite_involution (literal : Literal) :
    literal.opposite.opposite = literal := by
  cases literal with
  | mk site side => cases side <;> rfl

/-- The coverage premise is supplied explicitly, for the ORIGINAL predicates.
    It says nothing about a supplied numerical approximation to these sets. -/
structure PhasePredicates (Sample : Type u) where
  off : Nat → Sample → Prop
  on : Nat → Sample → Prop
  exhaustive : ∀ site sample, off site sample ∨ on site sample

def Holds {Sample : Type u} (pred : PhasePredicates Sample)
    (literal : Literal) (sample : Sample) : Prop :=
  match literal.side with
  | .off => pred.off literal.site sample
  | .on => pred.on literal.site sample

theorem holds_or_opposite {Sample : Type u} (pred : PhasePredicates Sample)
    (literal : Literal) (sample : Sample) :
    Holds pred literal sample ∨ Holds pred literal.opposite sample := by
  cases literal with
  | mk site side =>
    cases side with
    | off => exact pred.exhaustive site sample
    | on => exact (pred.exhaustive site sample).symm

def Fits {Sample : Type u} (pred : PhasePredicates Sample) :
    List Literal → Sample → Prop
  | [], _ => True
  | literal :: rest, sample => Holds pred literal sample ∧ Fits pred rest sample

theorem fits_append_iff {Sample : Type u} (pred : PhasePredicates Sample)
    (parent path : List Literal) (sample : Sample) :
    Fits pred (parent ++ path) sample ↔
      Fits pred parent sample ∧ Fits pred path sample := by
  induction parent with
  | nil => simp [Fits]
  | cons literal parent ih => simp [Fits, ih, and_assoc]

/-- Siblings are ordered by increasing first-disagreement position. -/
def siblings : List Literal → List (List Literal)
  | [] => []
  | literal :: rest =>
      [literal.opposite] :: (siblings rest).map (fun path => literal :: path)

/-- Exactly the destination first and then every shorter sibling. -/
def members (chosen : List Literal) : List (List Literal) :=
  chosen :: siblings chosen

def completeMembers (parent chosen : List Literal) : List (List Literal) :=
  (members chosen).map (fun path => parent ++ path)

theorem siblings_count (chosen : List Literal) :
    (siblings chosen).length = chosen.length := by
  induction chosen with
  | nil => rfl
  | cons literal rest ih => simp [siblings, ih]

theorem exact_descriptor_count (parent chosen : List Literal) :
    (completeMembers parent chosen).length = chosen.length + 1 := by
  simp [completeMembers, members, siblings_count]

theorem destination_first (parent chosen : List Literal) :
    (completeMembers parent chosen)[0]? = some (parent ++ chosen) := by
  rfl

/-- A sample either follows the entire chosen path, or stops at a first
    opposite phase. An overlapping sample may fit more than one descriptor. -/
theorem destination_or_first_disagreement {Sample : Type u}
    (pred : PhasePredicates Sample) (chosen : List Literal) (sample : Sample) :
    Fits pred chosen sample ∨
      ∃ sibling ∈ siblings chosen, Fits pred sibling sample := by
  induction chosen with
  | nil => exact Or.inl trivial
  | cons literal rest ih =>
    rcases holds_or_opposite pred literal sample with hchosen | hopposite
    · rcases ih with hrest | ⟨sibling, hmember, hfits⟩
      · exact Or.inl ⟨hchosen, hrest⟩
      · apply Or.inr
        refine ⟨literal :: sibling, ?_, hchosen, hfits⟩
        exact List.mem_cons.mpr (Or.inr (List.mem_map.mpr ⟨sibling, hmember, rfl⟩))
    · apply Or.inr
      exact ⟨[literal.opposite], List.mem_cons_self, hopposite, trivial⟩

theorem covers_original_parent {Sample : Type u} (pred : PhasePredicates Sample)
    (parent chosen : List Literal) (sample : Sample)
    (hparent : Fits pred parent sample) :
    ∃ member ∈ completeMembers parent chosen, Fits pred member sample := by
  rcases destination_or_first_disagreement pred chosen sample with hdest | ⟨s, hs, hf⟩
  · refine ⟨parent ++ chosen, ?_, (fits_append_iff pred parent chosen sample).mpr
        ⟨hparent, hdest⟩⟩
    exact List.mem_map.mpr ⟨chosen, List.mem_cons_self, rfl⟩
  · refine ⟨parent ++ s, ?_, (fits_append_iff pred parent s sample).mpr ⟨hparent, hf⟩⟩
    exact List.mem_map.mpr ⟨s, List.mem_cons_of_mem chosen hs, rfl⟩

theorem member_contained_in_parent {Sample : Type u} (pred : PhasePredicates Sample)
    (parent chosen member : List Literal) (sample : Sample)
    (hmember : member ∈ completeMembers parent chosen)
    (hfits : Fits pred member sample) : Fits pred parent sample := by
  rcases List.mem_map.mp hmember with ⟨path, _, heq⟩
  subst member
  exact ((fits_append_iff pred parent path sample).mp hfits).1

/-- A valid safety premise for the destination alone is insufficient. -/
theorem all_member_safety_transfer {Sample : Type u} (pred : PhasePredicates Sample)
    (parent chosen : List Literal) (safe : Sample → Prop)
    (hvalid : ∀ member ∈ completeMembers parent chosen,
      ∀ sample, Fits pred member sample → safe sample) :
    ∀ sample, Fits pred parent sample → safe sample := by
  intro sample hparent
  obtain ⟨member, hm, hf⟩ := covers_original_parent pred parent chosen sample hparent
  exact hvalid member hm sample hf

/-- The common bound must lie below EVERY valid member bound. This generic
    transitive relation does not assume a float, an imported certificate, or max. -/
theorem common_lower_bound {Sample : Type u} {Value : Type v}
    (pred : PhasePredicates Sample) (parent chosen : List Literal)
    (le : Value → Value → Prop)
    (trans : ∀ a b c, le a b → le b c → le a c)
    (score : Sample → Value) (bound : List Literal → Value) (common : Value)
    (hcommon : ∀ member ∈ completeMembers parent chosen, le common (bound member))
    (hvalid : ∀ member ∈ completeMembers parent chosen,
      ∀ sample, Fits pred member sample → le (bound member) (score sample)) :
    ∀ sample, Fits pred parent sample → le common (score sample) := by
  intro sample hparent
  obtain ⟨member, hm, hf⟩ := covers_original_parent pred parent chosen sample hparent
  exact trans common (bound member) (score sample)
    (hcommon member hm) (hvalid member hm sample hf)

theorem strict_safety_of_all_member_bounds {Sample : Type u}
    (pred : PhasePredicates Sample) (parent chosen : List Literal)
    (score : Sample → Int) (bound : List Literal → Int)
    (hpositive : ∀ member ∈ completeMembers parent chosen, 0 < bound member)
    (hvalid : ∀ member ∈ completeMembers parent chosen,
      ∀ sample, Fits pred member sample → bound member ≤ score sample) :
    ∀ sample, Fits pred parent sample → 0 < score sample := by
  apply all_member_safety_transfer pred parent chosen (fun sample => 0 < score sample)
  intro member hm sample hf
  exact Int.lt_of_lt_of_le (hpositive member hm) (hvalid member hm sample hf)

/- Executable descriptor correspondence model.

   Rust stores chosen = parent ++ chosen_path. Its sibling at j has depth
   parent.length+j+1, uses stored chosen values below the final coordinate, and
   uses opposite(stored chosen[parent.length+j]) at that final coordinate.
   The following total model uses getD only behind j < chosen.length. -/
def descriptorSibling (chosen : List Literal) (j : Nat) : List Literal :=
  chosen.take j ++ [(chosen.getD j default).opposite]

def rustSiblingView (parent chosen : List Literal) (j : Nat) : List Literal :=
  parent ++ descriptorSibling chosen j

theorem sibling_descriptor_correspondence (chosen : List Literal)
    (j : Nat) (hj : j < chosen.length) :
    (siblings chosen)[j]? = some (descriptorSibling chosen j) := by
  induction chosen generalizing j with
  | nil => exact (Nat.not_lt_zero j hj).elim
  | cons literal rest ih =>
    cases j with
    | zero => simp [siblings, descriptorSibling, List.getD_cons_zero]
    | succ j =>
      have hrest : j < rest.length := Nat.lt_of_succ_lt_succ hj
      change ((siblings rest).map (fun path => literal :: path))[j]? = _
      rw [List.getElem?_map, ih j hrest]
      simp [descriptorSibling, List.getD_cons_succ]

theorem canonical_member_order (parent chosen : List Literal)
    (j : Nat) (hj : j < chosen.length) :
    (completeMembers parent chosen)[j + 1]? = some (rustSiblingView parent chosen j) := by
  simp only [completeMembers, members, List.getElem?_map, List.getElem?_cons_succ,
    sibling_descriptor_correspondence chosen j hj]
  rfl

theorem actual_sibling_depth (parent chosen : List Literal)
    (j : Nat) (hj : j < chosen.length) :
    (rustSiblingView parent chosen j).length = parent.length + j + 1 := by
  simp [rustSiblingView, descriptorSibling,
    List.length_take_of_le (Nat.le_of_lt hj), Nat.add_assoc]

theorem sibling_prefix_stops_at_first_opposite (chosen : List Literal)
    (j : Nat) (hj : j < chosen.length) :
    (descriptorSibling chosen j).take j = chosen.take j := by
  apply List.take_left'
  exact List.length_take_of_le (Nat.le_of_lt hj)

theorem sibling_final_opposite (chosen : List Literal)
    (j : Nat) (hj : j < chosen.length) :
    (descriptorSibling chosen j)[j]? = some (chosen.getD j default).opposite := by
  have hlen : (chosen.take j).length = j := List.length_take_of_le (Nat.le_of_lt hj)
  unfold descriptorSibling
  rw [List.getElem?_append_right (by omega), hlen]
  simp

theorem sibling_has_no_later_phase (parent chosen : List Literal)
    (j i : Nat) (hj : j < chosen.length) (hi : j + 1 ≤ i) :
    (rustSiblingView parent chosen j)[parent.length + i]? = none := by
  apply List.getElem?_eq_none
  rw [actual_sibling_depth parent chosen j hj]
  simpa [Nat.add_assoc] using Nat.add_le_add_left hi parent.length

theorem sibling_literal_semantics {Sample : Type u} (pred : PhasePredicates Sample)
    (chosen : List Literal) (j : Nat) (sample : Sample) :
    Fits pred (descriptorSibling chosen j) sample ↔
      Fits pred (chosen.take j) sample ∧
        Holds pred (chosen.getD j default).opposite sample := by
  simp [descriptorSibling, fits_append_iff, Fits]

/- Overlap is allowed above. Uniqueness requires this additional premise;
   it is valid for binary phase assignments but false for closed zero domains. -/
def DisjointOpposites {Sample : Type u} (pred : PhasePredicates Sample) : Prop :=
  ∀ literal sample, ¬ (Holds pred literal sample ∧ Holds pred literal.opposite sample)

private theorem member_cons_shape (literal : Literal) (rest path : List Literal) :
    path ∈ members (literal :: rest) ↔
      path = [literal.opposite] ∨
        ∃ tail ∈ members rest, path = literal :: tail := by
  constructor
  · intro hm
    exact (List.mem_cons.mp hm).elim
      (fun hdest => Or.inr ⟨rest, List.mem_cons_self, hdest⟩)
      (fun hs => (List.mem_cons.mp hs).elim
        (fun hfirst => Or.inl hfirst)
        (fun hlater => (List.mem_map.mp hlater).elim
          (fun tail ht => Or.inr ⟨tail, List.mem_cons_of_mem rest ht.1, ht.2.symm⟩)))
  · intro hm
    exact hm.elim
      (fun hfirst => List.mem_cons.mpr (Or.inr (List.mem_cons.mpr (Or.inl hfirst))))
      (fun hex => hex.elim (fun tail ht => (List.mem_cons.mp ht.1).elim
        (fun hdest => List.mem_cons.mpr
          (Or.inl (ht.2.trans (congrArg (List.cons literal) hdest))))
        (fun hs => List.mem_cons.mpr (Or.inr (List.mem_cons.mpr (Or.inr
          (List.mem_map.mpr ⟨tail, hs, ht.2.symm⟩)))))))

theorem unique_binary_member {Sample : Type u} (pred : PhasePredicates Sample)
    (hdisjoint : DisjointOpposites pred) (chosen : List Literal)
    (sample : Sample) :
    ∀ (lhs rhs : List Literal), lhs ∈ members chosen → rhs ∈ members chosen →
      Fits pred lhs sample → Fits pred rhs sample → lhs = rhs := by
  induction chosen with
  | nil =>
    intro lhs rhs hl hr hfl hfr
    have hl' : lhs = [] := by simpa [members, siblings] using hl
    have hr' : rhs = [] := by simpa [members, siblings] using hr
    exact hl'.trans hr'.symm
  | cons literal rest ih =>
    intro lhs rhs hl hr hfl hfr
    apply Or.elim ((member_cons_shape literal rest lhs).mp hl)
    · intro heql
      subst lhs
      apply Or.elim ((member_cons_shape literal rest rhs).mp hr)
      · intro heqr
        exact heqr.symm
      · intro hex
        apply Exists.elim hex
        intro rt hrt
        have hfr' : Fits pred (literal :: rt) sample := hrt.2 ▸ hfr
        exact (hdisjoint literal sample ⟨hfr'.1, hfl.1⟩).elim
    · intro hex
      apply Exists.elim hex
      intro lt hlt
      have hfl' : Fits pred (literal :: lt) sample := hlt.2 ▸ hfl
      apply Or.elim ((member_cons_shape literal rest rhs).mp hr)
      · intro heqr
        subst rhs
        exact (hdisjoint literal sample ⟨hfl'.1, hfr.1⟩).elim
      · intro hext
        apply Exists.elim hext
        intro rt hrt
        have hfr' : Fits pred (literal :: rt) sample := hrt.2 ▸ hfr
        exact hlt.2.trans ((congrArg (List.cons literal)
          (ih lt rt hlt.1 hrt.1 hfl'.2 hfr'.2)).trans hrt.2.symm)

def binaryPredicates {Sample : Type u} (assignment : Nat → Sample → Bool) :
    PhasePredicates Sample where
  off := fun site sample => assignment site sample = false
  on := fun site sample => assignment site sample = true
  exhaustive := by
    intro site sample
    cases h : assignment site sample <;> simp [h]

theorem binary_opposites_disjoint {Sample : Type u}
    (assignment : Nat → Sample → Bool) : DisjointOpposites (binaryPredicates assignment) := by
  intro literal sample
  cases literal with
  | mk site side =>
    cases side <;> cases h : assignment site sample <;>
      simp [Holds, binaryPredicates, Literal.opposite, Side.opposite, h]

theorem exactly_one_binary_descriptor {Sample : Type u}
    (assignment : Nat → Sample → Bool) (chosen : List Literal) (sample : Sample) :
    ∃ member, (member ∈ members chosen ∧ Fits (binaryPredicates assignment) member sample) ∧
      ∀ other, (other ∈ members chosen ∧ Fits (binaryPredicates assignment) other sample) →
        other = member := by
  have hexists : ∃ member ∈ members chosen,
      Fits (binaryPredicates assignment) member sample := by
    rcases destination_or_first_disagreement (binaryPredicates assignment) chosen sample with
      hdest | ⟨sibling, hs, hf⟩
    · exact ⟨chosen, List.mem_cons_self, hdest⟩
    · exact ⟨sibling, List.mem_cons_of_mem chosen hs, hf⟩
  obtain ⟨member, hm, hf⟩ := hexists
  refine ⟨member, ⟨hm, hf⟩, ?_⟩
  intro other ho
  exact unique_binary_member (binaryPredicates assignment) (binary_opposites_disjoint assignment)
    chosen sample other member ho.1 hm ho.2 hf

/- Four independent closed controls. These use exact integers and descriptor
   computation, not numerical verifier results or a supplied pruning premise. -/
def integerPhases : PhasePredicates Int where
  off := fun _ sample => sample ≤ 0
  on := fun _ sample => 0 ≤ sample
  exhaustive := fun _ sample => Int.le_total sample 0

def onZero : Literal := ⟨0, .on⟩
def offOne : Literal := ⟨1, .off⟩

theorem closed_zero_overlap_control :
    Fits integerPhases [onZero] 0 ∧ Fits integerPhases [onZero.opposite] 0 := by
  simp [Fits, Holds, integerPhases, onZero, Literal.opposite, Side.opposite]

theorem descriptor_multiplicity_control :
    (completeMembers [] [onZero, offOne]).length = 3 ∧
      (completeMembers [] [onZero, offOne]) =
        [[onZero, offOne], [onZero.opposite], [onZero, offOne.opposite]] ∧
      ∀ member ∈ completeMembers [] [onZero, offOne], Fits integerPhases member 0 := by
  constructor
  · decide
  constructor
  · rfl
  · simp [completeMembers, members, siblings, Fits, Holds, integerPhases,
      onZero, offOne, Literal.opposite, Side.opposite]

def controlScore (phase : Bool) : Int := if phase then 4 else 1

theorem minimum_bound_control :
    (∀ phase, (1 : Int) ≤ controlScore phase) ∧
      controlScore false = 1 ∧ controlScore true = 4 := by
  decide

theorem maximum_member_bound_is_unsound_control :
    controlScore true = 4 ∧ ¬ (∀ phase, (4 : Int) ≤ controlScore phase) ∧
      0 < (if true then (4 : Int) else -1) ∧
      ¬ (∀ phase : Bool, 0 < (if phase then (4 : Int) else -1)) := by
  decide

-- Standalone diagnostic roster; captured output is evidence only after an
-- actual kernel run. The pinned overlay separately imports this module.
#print axioms covers_original_parent
#print axioms member_contained_in_parent
#print axioms exact_descriptor_count
#print axioms canonical_member_order
#print axioms sibling_descriptor_correspondence
#print axioms sibling_final_opposite
#print axioms sibling_has_no_later_phase
#print axioms all_member_safety_transfer
#print axioms common_lower_bound
#print axioms strict_safety_of_all_member_bounds
#print axioms exactly_one_binary_descriptor
#print axioms closed_zero_overlap_control
#print axioms descriptor_multiplicity_control
#print axioms minimum_bound_control
#print axioms maximum_member_bound_is_unsound_control

end NyProof.PrefixCover

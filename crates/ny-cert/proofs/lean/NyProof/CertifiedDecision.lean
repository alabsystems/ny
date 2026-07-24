/-
  CertifiedDecision.lean — the HEADLINE composition: soundness + completeness of the
  exact bisection verifier, in one kernel theorem, through the RUNNABLE checker object.

  `Complete.complete'` (completeness) proves: on a δ-robust instance there EXISTS a
  finite bisection depth at which every leaf's relaxed bound is strictly positive.
  `Bab.safe_on_path` / `Bab.babtree_sound` (soundness) prove: a `BabProof` tree whose
  decidable per-leaf checks pass entails safety on the whole root region.

  This file closes the gap between them with a CONSTRUCTION: `toBabProof` mirrors the
  depth-`d` full bisection into a concrete `Bab.BabProof` whose leaf certificates carry
  the verifier's EXACT rational bounds.  The main theorem `certified_decision` then
  states, for a coordinate-split exact relaxation and any δ-robust instance:

      ∃ d,  checkBabProof (toBabProof … B₀ d) = true          -- PRODUCED and CHECKED
            ∧  ∀ s, R.mem B₀ s → R.safe s                      -- and CORRECT

  where the safety conclusion is routed THROUGH the checker-side recursor
  (`Bab.safe_on_path` + `R.decides`), not re-proved abstractly — i.e. the very proof
  object the fielded verifier emits is what carries the whole-box verdict.

  Honest scope (state this prominently wherever cited):
  * δ-ROBUSTNESS is a hypothesis: `0 < δ ≤ R.trueMin B₀` (the property holds with a
    real margin).  On non-robust instances no bisection verifier terminates with a
    positive-leaf tree; completeness is only meaningful on the δ-robust class.
  * The relaxation is ABSTRACT with the width-error law as a hypothesis (the same
    modelling discipline as `Complete.lean`); the EXACTNESS hypothesis `hq_val` says
    the runnable rational bounds are exactly the relaxation's bounds — true for the
    exact-rational CROWN path, NOT for float paths.
  * Splits are coordinate bisections (`hmem_lo`/`hmem_hi` transport membership into
    the matching half-box) — the shape of NY's input-split BaB and of `Bab.BabProof`.
-/
import Crownproof.Complete
import Crownproof.BabProof

namespace Crownproof

namespace CertifiedDecision

open Complete Bab

variable {Box : Type*} {Sample : Type*} {Coord : Type*}
variable (R : Relaxation Box Sample)
variable (coord : Coord → Sample → ℚ)
variable (sc : Box → Coord) (sm : Box → ℚ)
variable (q : Box → QPair)

/-- Mirror the depth-`d` full bisection of a box into a concrete `BabProof`:
    leaves carry the box's exact rational bound `q B` as their certificate margin;
    splits record the box's split coordinate `sc B` and midpoint `sm B`. -/
def toBabProof (B : Box) : ℕ → BabProof Coord
  | 0     => .leaf ⟨q B⟩
  | d + 1 => .split (sc B) (sm B)
      (toBabProof (R.split B).1 d) (toBabProof (R.split B).2 d)

/-- The leaf interpretation for the recursor: a leaf is safe-on-its-path because the
    path is contained in SOME box whose relaxed bound is strictly positive — exactly
    what `R.decides` consumes.  (Antitone in the path by construction.) -/
def coveredLeafSafe (_lc : LeafCert) (path : Sample → Prop) : Prop :=
  ∃ C : Box, (∀ s, path s → R.mem C s) ∧ 0 < R.relaxedBound C

/-- **CHECKED.**  If every depth-`d` leaf box has a strictly positive relaxed bound,
    and the rational readout `q` exactly represents the bounds (`hq_val`) with positive
    denominators (`hq_den`), then the constructed tree passes the DECIDABLE checker. -/
theorem check_toBabProof
    (hq_den : ∀ B, 0 < (q B).2)
    (hq_val : ∀ B, ((toQ (q B) : ℚ) : ℝ) = R.relaxedBound B)
    (B : Box) (d : ℕ)
    (hpos : ∀ C ∈ leafBoxes R B d, 0 < R.relaxedBound C) :
    checkBabProof (toBabProof R sc sm q B d) = true := by
  induction d generalizing B with
  | zero =>
      have hposB : 0 < R.relaxedBound B := hpos B (by simp [leafBoxes])
      have hq : (0 : ℚ) < toQ (q B) := by
        have : ((0 : ℚ) : ℝ) < ((toQ (q B) : ℚ) : ℝ) := by
          rw [hq_val B]; exact_mod_cast hposB
        exact_mod_cast this
      -- 0 < num/den with 0 < den forces 0 < num.
      have hden : (0 : ℚ) < ((q B).2 : ℚ) := by exact_mod_cast hq_den B
      have hnum : (0 : ℚ) < ((q B).1 : ℚ) := by
        unfold toQ at hq
        rcases div_pos_iff.mp hq with ⟨hn, _⟩ | ⟨_, hd⟩
        · exact hn
        · linarith
      have hnumZ : (0 : ℤ) ≤ (q B).1 := by exact_mod_cast hnum.le
      simp only [toBabProof, checkBabProof, checkLeafCert,
        Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨hq_den B, hnumZ⟩
  | succ d ih =>
      have hlo : ∀ C ∈ leafBoxes R (R.split B).1 d, 0 < R.relaxedBound C := by
        intro C hC
        exact hpos C (by simp only [leafBoxes, List.mem_append]; exact Or.inl hC)
      have hhi : ∀ C ∈ leafBoxes R (R.split B).2 d, 0 < R.relaxedBound C := by
        intro C hC
        exact hpos C (by simp only [leafBoxes, List.mem_append]; exact Or.inr hC)
      simp only [toBabProof, checkBabProof, Bool.and_eq_true]
      exact ⟨ih _ hlo, ih _ hhi⟩

/-- **OBLIGATIONS.**  Along every root-to-leaf path of the constructed tree, the
    accumulated half-box cuts transport membership into the matching leaf box
    (`hmem_lo`/`hmem_hi`), so each leaf's `coveredLeafSafe` obligation holds with
    witness = its own leaf box. -/
theorem obligations_toBabProof
    (hmem_lo : ∀ B s, R.mem B s → coord (sc B) s ≤ sm B → R.mem (R.split B).1 s)
    (hmem_hi : ∀ B s, R.mem B s → sm B ≤ coord (sc B) s → R.mem (R.split B).2 s)
    (B : Box) (d : ℕ)
    (hpos : ∀ C ∈ leafBoxes R B d, 0 < R.relaxedBound C)
    (path : Sample → Prop) (hpath : ∀ s, path s → R.mem B s) :
    Obligations coord (coveredLeafSafe R) (toBabProof R sc sm q B d) path := by
  induction d generalizing B path with
  | zero =>
      exact ⟨B, hpath, hpos B (by simp [leafBoxes])⟩
  | succ d ih =>
      refine ⟨?_, ?_⟩
      · exact ih (R.split B).1
          (fun C hC => hpos C (by simp only [leafBoxes, List.mem_append]; exact Or.inl hC))
          (fun s => path s ∧ coord (sc B) s ≤ sm B)
          (fun s hs => hmem_lo B s (hpath s hs.1) hs.2)
      · exact ih (R.split B).2
          (fun C hC => hpos C (by simp only [leafBoxes, List.mem_append]; exact Or.inr hC))
          (fun s => path s ∧ sm B ≤ coord (sc B) s)
          (fun s hs => hmem_hi B s (hpath s hs.1) hs.2)

/--
**THE HEADLINE THEOREM — `certified_decision`.**

For a coordinate-split relaxation with EXACT rational bounds, on any δ-robust
instance (`0 < δ ≤ R.trueMin B₀`) the bisection verifier PRODUCES a concrete
`BabProof` that (a) PASSES the decidable recursive checker and (b) whose
checker-side soundness recursor delivers the whole-box verdict:

    ∃ d,  checkBabProof (toBabProof … B₀ d) = true  ∧  ∀ s ∈ B₀, safe s.

Soundness AND completeness of the fielded exact decision procedure, composed in
one kernel statement: completeness supplies the decisive depth (`complete'` via
`exists_decisive_depth`), the construction realises it as the runnable checker
object, and safety flows through `Bab.safe_on_path` + `R.decides` — the same
recursor that discharges the verifier's emitted certificates.
-/
theorem certified_decision
    (hq_den : ∀ B, 0 < (q B).2)
    (hq_val : ∀ B, ((toQ (q B) : ℚ) : ℝ) = R.relaxedBound B)
    (hmem_lo : ∀ B s, R.mem B s → coord (sc B) s ≤ sm B → R.mem (R.split B).1 s)
    (hmem_hi : ∀ B s, R.mem B s → sm B ≤ coord (sc B) s → R.mem (R.split B).2 s)
    (B₀ : Box) {δ : ℝ} (hδ : 0 < δ) (hmin : δ ≤ R.trueMin B₀) :
    ∃ d : ℕ,
      checkBabProof (toBabProof R sc sm q B₀ d) = true ∧
      ∀ s, R.mem B₀ s → R.safe s := by
  obtain ⟨d, hpos, -⟩ := Complete.complete' R B₀ hδ hmin
  refine ⟨d, check_toBabProof R sc sm q hq_den hq_val B₀ d hpos, ?_⟩
  exact safe_on_path coord R.safe (coveredLeafSafe R)
    (fun _lc path hob s hs => by
      obtain ⟨C, hsub, hp⟩ := hob
      exact R.decides C hp s (hsub s hs))
    (toBabProof R sc sm q B₀ d) (R.mem B₀)
    (obligations_toBabProof R coord sc sm q hmem_lo hmem_hi B₀ d hpos
      (R.mem B₀) (fun _ hs => hs))

end CertifiedDecision

end Crownproof

/-! ## Trust-base check — the composition must reduce to the standard axioms only. -/

#print axioms Crownproof.CertifiedDecision.check_toBabProof
#print axioms Crownproof.CertifiedDecision.obligations_toBabProof
#print axioms Crownproof.CertifiedDecision.certified_decision

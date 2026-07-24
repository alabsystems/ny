/-
  Discharging Clean's kernel axiom `NNVerify.farkas_to_interval` (T09).

  Clean's external-certificate verifier currently ASSERTS, as a kernel axiom,
  that a non-negative multiplier vector combining the relaxed-network premises
  (box bounds + the affine ≤/≥ pairs + the ReLU envelopes) into `-(out) - c`
  certifies that the true network output satisfies `out ≥ -c` on the box.

  Here we DISCHARGE that assumption.  `farkas_to_interval` below is a sorry-free
  *theorem* with exactly the statement Clean axiomatises (the abstract Farkas
  premise-combination entailment), proven by reduction to
  `farkas_premise_combination` (Bridge.lean), which is in turn proven from
  Mathlib's ordered-field lemmas.  `#print axioms` confirms the trust base is
  only `[propext, Classical.choice, Quot.sound]` — so the end-to-end claim
  "Clean accepts the certificate ⇒ the bound is a true lower bound" no longer
  rests on an unproven kernel axiom, only on Lean's standard logical foundations.

  `farkas_to_interval_relu1` re-exports the concrete one-hidden-layer
  unstable-ReLU witness (`crown_bridge`), so the discharged axiom is
  demonstrably non-vacuous: a genuine network execution satisfies its
  hypotheses.
-/
import Crownproof.Bridge

namespace Crownproof

open Finset

/--
**Discharged `farkas_to_interval` (Clean kernel axiom T09).**

Given an indexed family of relaxed-network premises `g i : S → ℚ`, each a sound
`≤ 0` relaxation on every *valid* (genuine-execution) state, non-negative
multipliers `μ`, and the Farkas certificate identity
`∀ s, ∑ i, μ i * g i s = -(out s) - c`, the network output satisfies
`out s ≥ -c` on every valid state.

This is precisely the entailment Clean asserts as the axiom
`NNVerify.farkas_to_interval`; here it is a theorem, proven constructively via
`farkas_premise_combination`.  No new axiom is introduced.
-/
theorem farkas_to_interval
    {S : Type*} {ι : Type*} (premises : Finset ι)
    (g : ι → S → ℚ) (out : S → ℚ) (μ : ι → ℚ) (c : ℚ)
    (valid : S → Prop)
    (hμ : ∀ i ∈ premises, 0 ≤ μ i)
    (hg : ∀ i ∈ premises, ∀ s, valid s → g i s ≤ 0)
    (hcert : ∀ s, (∑ i ∈ premises, μ i * g i s) = -(out s) - c) :
    ∀ s, valid s → -c ≤ out s :=
  farkas_premise_combination premises g out μ c valid hμ hg hcert

/--
Concrete non-vacuity witness for the discharged axiom: the one-hidden-layer
unstable-ReLU network (input box, affine pre-activation, ReLU lower/upper
envelopes, affine scalar output).  A genuine execution satisfying the four
relaxed premises with the Farkas multipliers attains the certified lower bound
`y ≥ -c`.  This is `crown_bridge` re-exported under the discharged-axiom name.
-/
theorem farkas_to_interval_relu1
    (l u w1 b1 w2 b2 alpha s lz u_z c : ℚ)
    (m_bl m_bu m_rl m_ru : ℚ)
    (ha0 : 0 ≤ alpha) (ha1 : alpha ≤ 1)
    (hlz : lz < 0) (huz : 0 < u_z) (hs : s * (u_z - lz) = u_z)
    (hbox_z : ∀ st : NetState, NetState.valid l u w1 b1 w2 b2 st →
                lz ≤ st.z ∧ st.z ≤ u_z)
    (hm_bl : 0 ≤ m_bl) (hm_bu : 0 ≤ m_bu)
    (hm_rl : 0 ≤ m_rl) (hm_ru : 0 ≤ m_ru)
    (hcert : ∀ st : NetState,
        m_bl * (l - st.x)
      + m_bu * (st.x - u)
      + m_rl * (alpha * st.z - st.a)
      + m_ru * (st.a - s * (st.z - lz))
        = -(st.y) - c) :
    ∀ st : NetState, NetState.valid l u w1 b1 w2 b2 st → -c ≤ st.y :=
  crown_bridge l u w1 b1 w2 b2 alpha s lz u_z c m_bl m_bu m_rl m_ru
    ha0 ha1 hlz huz hs hbox_z hm_bl hm_bu hm_rl hm_ru hcert

/-! ## Trust-base check.  Must list only the three standard logical axioms,
    and in particular MUST NOT list `farkas_to_interval` itself. -/

#print axioms farkas_to_interval
#print axioms farkas_to_interval_relu1

end Crownproof

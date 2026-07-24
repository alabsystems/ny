/-
  AxiomAudit.lean — REPRODUCIBLE soundness-trust-base audit for ny-cert's proof-carrying pipeline.

  Every theorem ny-cert's certificates depend on (the 6 cite-map grounding theorems, the
  entailment-checker + Certificate-Equivalence spec soundness, and the float-adequacy lemmas)
  must depend on AT MOST Lean's standard `[propext, Classical.choice, Quot.sound]` — no `sorry`,
  no extra axioms.  (A SUBSET is better than the full list: e.g. the Cersyve induction and
  `Bab.tiny_checks` use only `[propext, Quot.sound]` — fully constructive but for propext.)
  This file makes that machine-checkable and reproducible rather than memory-resident:

      lake exe cache get && lake env lean NyProof/AxiomAudit.lean

  The captured output is committed alongside as `AXIOM_AUDIT.txt`. Any divergence (a `sorryAx`,
  an unexpected axiom, an unknown constant) is a soundness regression in the cited base.

  See PROVENANCE.md ("Local lake-validation") for context. Toolchain: leanprover/lean4:v4.30.0.
-/
import Crownproof.Block2
import Crownproof.Deep
import Crownproof.DeepK
import Crownproof.Sbar
import Crownproof.Pow2Envelope
import Crownproof.CertChecker
import Crownproof.CertCheckerZ
import Crownproof.CertEquiv
import NyProof.FloatAdequacy
import Crownproof.BranchTree
import Crownproof.BabProof
import Crownproof.Complete
import NyProof.CertifiedDecision
import NyProof.CersyveInduction
import NyProof.SatReluGadget
import NyProof.SatReluCnf
import NyProof.MeanValueForm
import NyProof.MeanValueChain
import NyProof.AristotleLemmas
import NyProof.RupChecker
import NyProof.SatReluVerdict
import NyProof.SatReluDemo_v10c26
import NyProof.SatReluDemo_v92c117
import NyProof.RupCheckerFast
import NyProof.SatReluDemo_v100c373
import NyProof.SatReluDemo_v99c485
import NyProof.SatReluSweep.V90C449
import NyProof.CersyveInstance_DoubleIntegrator
import NyProof.CersyveInstance_Pendulum
import NyProof.CersyveInstance_Unicycle

-- The 6 cite-map grounding theorems (every CertifiedModuloCite rests on these):
#print axioms Crownproof.farkas_premise_combination
#print axioms Crownproof.crown_bridge
#print axioms Crownproof.crown_bridge_deepK
#print axioms Crownproof.sbar_support_sound
#print axioms Crownproof.pow2_tangent
#print axioms Crownproof.pow2_secant

-- Entailment-checker spec + kernel-runnable checker + Certificate-Equivalence soundness:
#print axioms Crownproof.CertChecker.checkEntailment_sound
#print axioms Crownproof.CertCheckerZ.checkEntailmentZ_sound
#print axioms Crownproof.cert_list_sound
#print axioms Crownproof.crown_cert_instance

-- Float adequacy R_float ⊑ R_real (Program-1 #3, residual-float TCB lemmas):
#print axioms FloatAdequacy.interval_outward_contains
#print axioms FloatAdequacy.affine_lower_adequate
#print axioms FloatAdequacy.affine_upper_adequate
#print axioms FloatAdequacy.box_adequate
#print axioms FloatAdequacy.relu_chord_upper_adequate
#print axioms FloatAdequacy.float_bound_implies_real

-- Completeness + BaB-recursor soundness — the `certified_decision` components
-- (the headline composition theorem composes these; auditing them
-- here puts the whole composition target under the same 3-axiom trust base):
#print axioms Crownproof.BoxTree.safe_of_leaves
#print axioms Crownproof.Bab.checkLeafCert_sound
#print axioms Crownproof.Bab.safe_on_path
#print axioms Crownproof.Bab.babtree_sound
#print axioms Crownproof.Bab.tiny_safe
#print axioms Crownproof.Complete.exists_decisive_depth
#print axioms Crownproof.Complete.box_safe_of_leaves
#print axioms Crownproof.Complete.complete
#print axioms Crownproof.Complete.complete'

-- THE COMPOSITION — `certified_decision` (soundness + completeness of the exact
-- bisection verifier in ONE kernel theorem, verdict routed through the runnable
-- checker object; see NyProof/CertifiedDecision.lean for the honest scope):
#print axioms Crownproof.CertifiedDecision.check_toBabProof
#print axioms Crownproof.CertifiedDecision.obligations_toBabProof
#print axioms Crownproof.CertifiedDecision.certified_decision

-- Certified-safe-forever for clamped neural control loops (cersyve): the induction
-- upgrading NY's one-step `inv`/`con` unsat verdicts to unbounded closed-loop safety
-- (see NyProof/CersyveInduction.lean for the honest surrogate-dynamics scope):
#print axioms Crownproof.Cersyve.safe_forever
#print axioms Crownproof.Cersyve.safe_at_every_step

-- sat_relu CNF-decompilation soundness core (Route A): the exact-arithmetic
-- Booleanization equivalence the detector relies on — `x − ReLU(2x−1) ≥ 0` on
-- [0,1], zero iff boolean, so `Y_1 ≤ 0` forces boolean inputs (see
-- NyProof/SatReluGadget.lean and docs/MEASURED_SAT_RELU.md):
#print axioms Crownproof.SatRelu.bres_nonneg
#print axioms Crownproof.SatRelu.bres_eq_zero_iff
#print axioms Crownproof.SatRelu.forces_boolean

-- sat_relu END-TO-END equivalence (NyProof/SatReluCnf.lean): the gadget's
-- unsafe region is nonempty IFF the recovered CNF is satisfiable — the theorem
-- behind Route A's UNSAT verdicts (unsat CNF ⇒ property safe) and SAT witnesses:
#print axioms Crownproof.SatRelu.clauseRow_nonneg
#print axioms Crownproof.SatRelu.clauseRow_boolPoint_of_satisfies
#print axioms Crownproof.SatRelu.clauseRow_boolPoint_of_not_satisfies
#print axioms Crownproof.SatRelu.clauseRow_boolPoint_eq_zero_iff
#print axioms Crownproof.SatRelu.boolPoint_unsafe_of_satisfies
#print axioms Crownproof.SatRelu.exists_satisfying_of_unsafe
#print axioms Crownproof.SatRelu.sat_iff_unsafe
#print axioms Crownproof.SatRelu.unsat_implies_safe

-- Mean-value/centered-form soundness core (NyProof/MeanValueForm.lean): the
-- 1-D scalar-shadow lemmas behind graph_ibp_f64_mvf.rs (piecewise MVT telescope
-- + centered-form enclosure; multivariate chain step + f64 rounding are the
-- documented remaining hypotheses):
#print axioms Crownproof.interval_convex_combination_mem
#print axioms Crownproof.interval_convex_combination_sum_mem
#print axioms Crownproof.interval_mul_mem
#print axioms Crownproof.relu_hull_mul_mem
#print axioms Crownproof.mvt_piece_bound
#print axioms Crownproof.piecewise_mvt_telescope
#print axioms Crownproof.piecewise_mvt_telescope_subinterval
#print axioms Crownproof.centered_form_enclosure

-- Multivariate CHAIN STEP (NyProof/MeanValueChain.lean): segment derivative
-- = Σ of coordinate partials; coordinatewise corner hulls; the centered form in
-- n dimensions, single-piece + piecewise + branch-fixed (the degenerate
-- identically-zero-ReLU-piece case found by adversarial review):
#print axioms Crownproof.fderiv_apply_eq_sum_partials
#print axioms Crownproof.segment_deriv
#print axioms Crownproof.sum_partials_mem_hull
#print axioms Crownproof.multivariate_centered_form
#print axioms Crownproof.multivariate_centered_form_of_convex
#print axioms Crownproof.piecewise_multivariate_centered_form
#print axioms Crownproof.piecewise_multivariate_centered_form_branch_fixed

-- Aristotle-proven, locally re-verified lemmas (NyProof/AristotleLemmas.lean):
-- finite zeros of a nonzero analytic function (the MVF finitely-many-breakpoints
-- core) and the composed quartic envelope (pow2_compose, torus tier):
#print axioms Crownproof.analytic_zeros_finite
#print axioms Crownproof.pow2_compose_envelope
#print axioms Crownproof.ReluPiecewise.relu_analytic_piecewise
#print axioms Crownproof.farkas_refutation_sound
#print axioms Crownproof.FarkasStrict.farkas_la_generic_unsat

-- Piecewise-analytic closure (Aristotle batch 3b): the MVF DAG-induction engine —
-- PA closed under linear combination and under ReLU (see AristotleLemmas.lean):
#print axioms Crownproof.PiecewiseAnalytic.piecewiseAnalytic_linear_comb
#print axioms Crownproof.PiecewiseAnalytic.piecewiseAnalytic_relu

-- Propositional resolution soundness (Aristotle batch 4): the refutation-import
-- core for the fully-certified sat_relu sweep (roadmap 12, second half):
#print axioms Crownproof.Resolution.PropResolution.resolution_sound
#print axioms Crownproof.Resolution.PropResolution.derivation_sat
#print axioms Crownproof.Resolution.PropResolution.refutation_sound

-- RUP soundness (Aristotle batch 5): the exact LRAT/DRUP per-step rule —
-- completes the certified-sweep import chain down to artifact parsing:
#print axioms Crownproof.RupImport.RUP.rup_sound

-- Kernel-runnable LRAT/RUP refutation checker + the fully-certified sat_relu
-- verdict chain (RupChecker.lean, SatReluVerdict.lean): checkRefutation = true
-- (kernel decide) → CNF unsat → no box point reaches the gadget unsafe region:
#print axioms Crownproof.RupChecker.checkHints_sound
#print axioms Crownproof.RupChecker.checkStep_isRUP
#print axioms Crownproof.RupChecker.checkStep_entails
#print axioms Crownproof.RupChecker.checkRefutation_sound
#print axioms Crownproof.SatReluVerdict.satClause_of_satisfies
#print axioms Crownproof.SatReluVerdict.safe_of_unsat

-- END-TO-END CERTIFIED REAL INSTANCES (ny → ay LRAT → lrat_to_lean → kernel):
#print axioms Crownproof.SatReluDemo_v10c26.check_ok
#print axioms Crownproof.SatReluDemo_v10c26.instance_safe
#print axioms Crownproof.SatReluDemo_v92c117.check_ok
#print axioms Crownproof.SatReluDemo_v92c117.instance_safe

-- Fast kernel checker (Nat-bitmask assertions + trie database; RupCheckerFast.lean)
-- extends the certified envelope to EVERY staged sat_relu UNSAT instance:
#print axioms Crownproof.RupCheckerFast.checkRefutationFast_sound
#print axioms Crownproof.SatReluDemo_v100c373.check_ok
#print axioms Crownproof.SatReluDemo_v100c373.instance_safe
#print axioms Crownproof.SatReluDemo_v99c485.check_ok
#print axioms Crownproof.SatReluDemo_v99c485.instance_safe

-- Sweep sentinel (largest instance: 90 vars, 449 clauses, 1160 LRAT lines).
-- THE FULL 49-INSTANCE SWEEP gates via `lake build NyProof.SatReluSweepAll`
-- (kept out of this audit's regeneration loop; its build prints the complete
-- per-instance axiom manifest — all on the 3-axiom base, zero failures):
#print axioms Crownproof.SatReluSweep_v90c449.check_ok
#print axioms Crownproof.SatReluSweep_v90c449.instance_safe

-- Cersyve graph->ClampedSystem binding (worked instance): the double_integrator
-- con/inv ONNX nets as exact-Q Net literals + a ClampedSystem instance whose
-- hinv/hcon are derived from the Clean-verified certs and composed to unbounded
-- safety (residual trust: cert box-universality + the step dynamics net that
-- isn't in the con/inv pair — named as explicit Lean hypotheses):
#print axioms Crownproof.CersyveInstance_DoubleIntegrator.double_integrator_safe_forever
#print axioms Crownproof.CersyveInstance_DoubleIntegrator.double_integrator_safe_forever_full
#print axioms Crownproof.CersyveInstance_Pendulum.pendulum_safe_forever
#print axioms Crownproof.CersyveInstance_Pendulum.pendulum_safe_forever_full
#print axioms Crownproof.CersyveInstance_Unicycle.unicycle_safe_forever
#print axioms Crownproof.CersyveInstance_Unicycle.unicycle_safe_forever_full

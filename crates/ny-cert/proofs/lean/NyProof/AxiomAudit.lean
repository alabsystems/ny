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
import NyProof.SignFusion
import NyProof.IntegerInterval

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

-- Sign-pair fusion (L1) + value-branch coverage (L2) — the two lemmas under the
-- fused-`Sign` / binarized-net BaB route (NyProof/SignFusion.lean;
-- docs/SIGN_COMPOSITE_FUSION_DESIGN_2026-07-27.md).  L1: `Sign→Add(c)→Sign` with
-- `0 < |c| < 1` is EXACTLY a two-valued step (both polarities, plus the guards that
-- make `0 < |c| < 1` necessary and a rounding-robust form).  L2: the closed/closed
-- value-fixing split is sound iff the children's AGREEMENT sets cover — the regions
-- covering is NOT enough (`three_clause_split_unsound` is the machine-checked
-- refutation of the informal argument), and the same split on the RAW three-valued
-- ONNX `Sign` is unsound (`sgn_closed_split_unsound`):
#print axioms Crownproof.SignFusion.sgn_three_valued
#print axioms Crownproof.SignFusion.hs_two_valued
#print axioms Crownproof.SignFusion.hs'_two_valued
#print axioms Crownproof.SignFusion.sign_pair_fusion
#print axioms Crownproof.SignFusion.sign_pair_fusion_neg
#print axioms Crownproof.SignFusion.sign_pair_fusion_rewrite
#print axioms Crownproof.SignFusion.sign_pair_fusion_of_signFaithful
#print axioms Crownproof.SignFusion.sign_pair_fusion_fails_of_one_le
#print axioms Crownproof.SignFusion.sign_pair_fusion_fails_of_zero_const
#print axioms Crownproof.SignFusion.sign_pair_fusion_fails_of_le_neg_one
#print axioms Crownproof.SignFusion.branch_sound_of_agreement_cover
#print axioms Crownproof.SignFusion.branch_sound_two
#print axioms Crownproof.SignFusion.three_clause_split_unsound
#print axioms Crownproof.SignFusion.twoValued_closed_split_sound
#print axioms Crownproof.SignFusion.hs_closed_split_sound
#print axioms Crownproof.SignFusion.hs'_closed_split_sound
#print axioms Crownproof.SignFusion.sgn_closed_split_unsound
#print axioms Crownproof.SignFusion.offcenter_closed_split_unsound
#print axioms Crownproof.SignFusion.plus_child_must_contain_breakpoint
#print axioms Crownproof.SignFusion.hs_fix_pos_on_closed_sound
#print axioms Crownproof.SignFusion.hs_fix_neg_on_open_sound
#print axioms Crownproof.SignFusion.hs_fix_neg_on_closed_unsound
#print axioms Crownproof.SignFusion.hs'_fix_neg_on_closed_sound
#print axioms Crownproof.SignFusion.hs'_fix_pos_on_closed_unsound
#print axioms Crownproof.SignFusion.frontierSound_root
#print axioms Crownproof.SignFusion.frontierSound_verdict
#print axioms Crownproof.SignFusion.frontierSound_refine
#print axioms Crownproof.SignFusion.frontierSound_refine_sign

-- exact INTEGER / LATTICE interval reasoning for binarized nets
-- (NyProof/IntegerInterval.lean).  Integrality is MEASURED on the three
-- traffic_signs_recognition_2023 artifacts, not assumed: layer 1 is NOT integral over the
-- continuous `Real` input box (`box_contains_nonintegral`), everything from the first
-- `Sign` onward IS (`post_sign_layer_even`).  The three claims the integer route rests on:
-- tightening `[l,u]` to `[ceil l, floor u]` is sound AND a contraction AND optimal;
-- stability on integers is exact and ASYMMETRIC (matching `hs_fix_neg_on_closed_unsound`,
-- with `hs_stable_neg_int_unsound` refuting the symmetric form ny currently ships); and
-- integer interval arithmetic for +/- and +-1-weighted sums is exact, not merely sound:
#print axioms Crownproof.IntegerInterval.isIntegral_intCast
#print axioms Crownproof.IntegerInterval.isIntegral_zero
#print axioms Crownproof.IntegerInterval.isIntegral_one
#print axioms Crownproof.IntegerInterval.isIntegral_neg_one
#print axioms Crownproof.IntegerInterval.OnLattice.isIntegral
#print axioms Crownproof.IntegerInterval.onLattice_one
#print axioms Crownproof.IntegerInterval.OnLattice.add
#print axioms Crownproof.IntegerInterval.OnLattice.sub
#print axioms Crownproof.IntegerInterval.OnLattice.neg
#print axioms Crownproof.IntegerInterval.intTighten_sound
#print axioms Crownproof.IntegerInterval.intTighten_contracts
#print axioms Crownproof.IntegerInterval.intTighten_set_eq
#print axioms Crownproof.IntegerInterval.intTighten_optimal
#print axioms Crownproof.IntegerInterval.intTighten_infeasible
#print axioms Crownproof.IntegerInterval.intTighten_idem
#print axioms Crownproof.IntegerInterval.lattice_tighten_sound
#print axioms Crownproof.IntegerInterval.lattice_lower_forces_nonneg
#print axioms Crownproof.IntegerInterval.lattice_lower_forces_nonneg_sharp
#print axioms Crownproof.IntegerInterval.lattice_upper_forces_nonpos
#print axioms Crownproof.IntegerInterval.isIntegral_add
#print axioms Crownproof.IntegerInterval.isIntegral_sub
#print axioms Crownproof.IntegerInterval.isIntegral_neg
#print axioms Crownproof.IntegerInterval.isIntegral_mul
#print axioms Crownproof.IntegerInterval.isIntegral_max
#print axioms Crownproof.IntegerInterval.onLattice_max
#print axioms Crownproof.IntegerInterval.int_interval_add_sound
#print axioms Crownproof.IntegerInterval.int_interval_add_exact
#print axioms Crownproof.IntegerInterval.int_interval_sub_sound
#print axioms Crownproof.IntegerInterval.pm1_mul
#print axioms Crownproof.IntegerInterval.pm1_weighted_sum
#print axioms Crownproof.IntegerInterval.pm1_weighted_sum_even
#print axioms Crownproof.IntegerInterval.pm1_range_exact
#print axioms Crownproof.IntegerInterval.hs_stable_pos_int
#print axioms Crownproof.IntegerInterval.hs_stable_neg_int
#print axioms Crownproof.IntegerInterval.hs_stable_neg_int_unsound
#print axioms Crownproof.IntegerInterval.hs_stable_neg_int_sharp
#print axioms Crownproof.IntegerInterval.hs_stable_pos_int_sharp
#print axioms Crownproof.IntegerInterval.hs_stable_int_complete
#print axioms Crownproof.IntegerInterval.hs'_stable_pos_int
#print axioms Crownproof.IntegerInterval.hs'_stable_neg_int
#print axioms Crownproof.IntegerInterval.hs'_stable_pos_int_unsound
#print axioms Crownproof.IntegerInterval.affine_sign_iff_int
#print axioms Crownproof.IntegerInterval.hs_affine_int
#print axioms Crownproof.IntegerInterval.affine_ne_zero_of_threshold_not_int
#print axioms Crownproof.IntegerInterval.affine_strictMono
#print axioms Crownproof.IntegerInterval.box_contains_nonintegral
#print axioms Crownproof.IntegerInterval.post_sign_layer_integral
#print axioms Crownproof.IntegerInterval.post_sign_layer_even
#print axioms Crownproof.IntegerInterval.softmax_cmp_iff
#print axioms Crownproof.IntegerInterval.straddling_bound_pins_zero

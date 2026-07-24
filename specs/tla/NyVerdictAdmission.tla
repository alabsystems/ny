---- MODULE NyVerdictAdmission ----
\* Verdict-admission theorem for ny's scored VNN-COMP pipeline
\* (docs/BEYOND_SOTA_PLAN.md roadmap item 14; mirrors the evidence-contract
\* style of ty's specs/TrustEngineAdmission.tla).
\*
\* WHAT IS MODELED (grounded in the Rust, revision of 2026-07-10):
\*   * the `ny beta-crown` dispatch (crates/ny-cli/src/commands/beta_crown/mod.rs):
\*       cnf_route -> (is_mip_only -> run_mip_only) -> run_bab_with_fallback,
\*     where run_bab_with_fallback contains the IBP fast-path, the
\*     cell-enumeration driver (cell_enum.rs), the BaB loop, and the MIP
\*     auto-escalation (dispatch.rs);
\*   * the CNF-recovery DRAT gate (cnf_route.rs `check_drat`): ay's UNSAT is
\*     NEVER trusted without a verified DRAT refutation — an unchecked
\*     refutation falls through to the normal pipeline;
\*   * the process-global GPU CROWN soundness gate
\*     (ny-propagate/src/sound_gpu_gate.rs): when engaged, only a backend with
\*     `provides_sound_gpu_crown()` (or the CPU f64+gamma_n*S path) may decide a
\*     verdict bound; competition mode can never disengage it
\*     (ProofOpts::sound_gpu_crown_required, mod.rs);
\*   * the vnncomp SAT admission gate (vnncomp.rs `gate_sat_with_trusted_oracle`):
\*     every internal sat is re-confirmed by a real ONNX-Runtime forward AND
\*     re-validated by the sound true-f64 interval forward before emission;
\*     any gate failure downgrades to the sound `unknown`;
\*   * lane availability vs. configuration: preset `general.complete_verifier:
\*     mip` on a binary built without `--features mip` (verify_with_mip stub
\*     bails -> vnncomp maps the error to sound `unknown`), and the preset
\*     `solver.mip.mip_solver: scip` pin on a binary without `mip-scip`
\*     (resolve_preset_mip_solver degrades to ay WITH A LOGGED WARNING).
\*
\* THE HISTORICAL BUG THIS GUARDS (BEYOND_SOTA_PLAN.md, "Build-drift landmine"):
\*   the sat_relu SCIP pin was config-only — no build enabled `mip-scip` and the
\*   shipped binary had MIP compiled out entirely, so the configured lane
\*   silently never ran. The Bug* constants below reintroduce that class of
\*   drift; with any of them TRUE the invariants MUST fail (see the
\*   *.bug_*.cfg configurations and the receipt).
\*
\* SCOPE / HONESTY: this is a finite-state MODEL theorem about the admission
\* logic, checked over ALL 2^9 build/config environments. It is NOT a proof
\* about the Rust; the correspondence of each transition to the cited code is
\* code-level trust (see docs/VERDICT_ADMISSION_SPEC.md).

EXTENDS TLC

CONSTANTS
  \* Fault-injection toggles. All FALSE = the shipped (fixed) pipeline.
  BugConfigOnlyPin,     \* historical SCIP pin: preset pins a solver whose
                        \* feature is compiled out and the degrade is SILENT
                        \* (no warn / no receipt fact)
  BugPhantomMipLane,    \* MIP feature compiled out but a verdict is still
                        \* claimed through the MIP lane
  BugUncheckedDrat,     \* cnf_route emits Verified on ay's UNSAT without a
                        \* checked DRAT refutation
  BugSkipOrtGate,       \* sat emitted without the ORT trusted-oracle
                        \* re-confirmation + true-f64 re-validation
  BugUnsoundGpuVerdict  \* gate engaged, yet the fast round-to-nearest f32
                        \* GPU CROWN backward decides the verdict bound

ASSUME BugConfigOnlyPin \in BOOLEAN
ASSUME BugPhantomMipLane \in BOOLEAN
ASSUME BugUncheckedDrat \in BOOLEAN
ASSUME BugSkipOrtGate \in BOOLEAN
ASSUME BugUnsoundGpuVerdict \in BOOLEAN

VARIABLES
  \* --- pipeline stage (program counter) ---
  pc,
  \* --- build/config environment: chosen nondeterministically in Init and
  \*     then frozen, so the check quantifies over every environment ---
  mip_compiled,          \* built with --features mip
  scip_compiled,         \* built with the historical (since-deleted)
                         \* --features mip-scip toggle; modeled to cover
                         \* the config-only-pin bug class
  ay_binary_present,     \* $NY_AY / PATH probe found the ay CDCL solver
  gpu_sound_capability,  \* a backend advertising provides_sound_gpu_crown()
  preset_loaded,         \* a preset file was loaded (preset_id known)
  preset_requires_mip,   \* preset general.complete_verifier: mip
  preset_pins_scip,      \* preset solver.mip.mip_solver: scip
  competition_mode,      \* ProofOpts::competition_mode (ny vnncomp scored path)
  allow_unsound_gpu,     \* --allow-unsound-gpu-crown (never honored in comp mode)
  seq_model,             \* model loads as Sequential (vs Graph/DAG)
  \* --- evidence facts accumulated by the run ---
  sound_gate_engaged,    \* set_sound_gpu_crown_required(..) value
  gpu_backward_used,     \* "unresolved" | "cpu_f64" | "gpu_sound" | "gpu_fast_f32"
  drat_checked,          \* `ay check drat` returned s VERIFIED
  witness_confirmed_inproc, \* concrete forward re-check inside ny
  ort_confirmed,         \* ONNX-Runtime trusted oracle reproduced the violation
  f64_revalidated,       \* true-f64 interval forward did not refute the witness
  cell_coverage_complete,\* cell_enum evaluated EVERY cell (no partial coverage)
  lane_degrade_logged,   \* the legacy-solver-pin -> ay degrade warning was emitted
  \* --- outcome ---
  verdict,               \* "none" | "unsat" | "sat" | "unknown" | "timeout"
  verdict_lane,          \* "none" | "cnf" | "cell" | "mip" | "bab"
  emitted                \* the scored RESULTS token was written

vars == <<pc,
  mip_compiled, scip_compiled, ay_binary_present, gpu_sound_capability,
  preset_loaded, preset_requires_mip, preset_pins_scip, competition_mode,
  allow_unsound_gpu, seq_model,
  sound_gate_engaged, gpu_backward_used, drat_checked,
  witness_confirmed_inproc, ort_confirmed, f64_revalidated,
  cell_coverage_complete, lane_degrade_logged,
  verdict, verdict_lane, emitted>>

env == <<mip_compiled, scip_compiled, ay_binary_present, gpu_sound_capability,
  preset_loaded, preset_requires_mip, preset_pins_scip, competition_mode,
  allow_unsound_gpu, seq_model>>

Stages == {"Init", "CnfRoute", "CellEnum", "MipOnly", "BabLoop",
           "GpuDispatch", "SatWitness", "Emit", "Done"}

Verdicts == {"none", "unsat", "sat", "unknown", "timeout"}

Lanes == {"none", "cnf", "cell", "mip", "bab"}

GpuRoutes == {"unresolved", "cpu_f64", "gpu_sound", "gpu_fast_f32"}

\* ------------------------------------------------------------------------
\* Init: pick an arbitrary build/config environment; evidence starts empty.
\* `sound_gate_engaged` mirrors ProofOpts::sound_gpu_crown_required():
\* competition_mode || !allow_unsound_gpu — applied via
\* ny_propagate::set_sound_gpu_crown_required BEFORE any dispatch (mod.rs:690).
\* ------------------------------------------------------------------------
Init ==
  /\ pc = "Init"
  /\ mip_compiled \in BOOLEAN
  /\ scip_compiled \in BOOLEAN
  /\ ay_binary_present \in BOOLEAN
  /\ gpu_sound_capability \in BOOLEAN
  /\ preset_loaded \in BOOLEAN
  /\ preset_requires_mip \in BOOLEAN
  /\ preset_pins_scip \in BOOLEAN
  /\ competition_mode \in BOOLEAN
  /\ allow_unsound_gpu \in BOOLEAN
  /\ seq_model \in BOOLEAN
  \* a preset lane request presupposes a loaded preset
  /\ preset_requires_mip => preset_loaded
  /\ preset_pins_scip => preset_loaded
  \* scip is only meaningful under the mip feature build
  /\ scip_compiled => mip_compiled
  /\ sound_gate_engaged = (competition_mode \/ ~allow_unsound_gpu)
  /\ gpu_backward_used = "unresolved"
  /\ drat_checked = FALSE
  /\ witness_confirmed_inproc = FALSE
  /\ ort_confirmed = FALSE
  /\ f64_revalidated = FALSE
  /\ cell_coverage_complete = FALSE
  /\ lane_degrade_logged = FALSE
  /\ verdict = "none"
  /\ verdict_lane = "none"
  /\ emitted = FALSE

StartDispatch ==
  /\ pc = "Init"
  /\ pc' = "CnfRoute"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

\* ------------------------------------------------------------------------
\* Stage: CnfRoute (cnf_route.rs try_cnf_recovery)
\* Runs FIRST, before the MIP/BaB fork (mod.rs:1123). Fail-closed: any doubt
\* (no gadget, no ay binary, timeout, unparseable output, DRAT check failure)
\* falls through to the normal pipeline with zero behavior change.
\* ------------------------------------------------------------------------

\* is_mip_only == complete_verifier == Mip && model is Sequential (mod.rs:1086)
NextAfterCnf == IF preset_requires_mip /\ seq_model THEN "MipOnly" ELSE "CellEnum"

CnfFallThrough ==
  /\ pc = "CnfRoute"
  /\ pc' = NextAfterCnf
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

\* ay solved UNSAT and `ay check drat` returned `s VERIFIED` (cnf_route.rs:134-152):
\* only then is Verified admitted. A failed/missing DRAT is CnfFallThrough.
CnfUnsatDratChecked ==
  /\ pc = "CnfRoute"
  /\ ay_binary_present
  /\ drat_checked' = TRUE
  /\ verdict' = "unsat"
  /\ verdict_lane' = "cnf"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, emitted>>

\* FAULT INJECTION (BugUncheckedDrat): trust ay's `s UNSATISFIABLE` on faith.
CnfUnsatUnchecked ==
  /\ BugUncheckedDrat
  /\ pc = "CnfRoute"
  /\ ay_binary_present
  /\ verdict' = "unsat"
  /\ verdict_lane' = "cnf"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, emitted>>

\* ay solved SAT: the boolean model is confirmed by a concrete in-process
\* forward (confirm_boolean_witness) BEFORE claiming, then re-confirmed by the
\* vnncomp ORT gate downstream (cnf_route.rs:163-165).
CnfSatCandidate ==
  /\ pc = "CnfRoute"
  /\ ay_binary_present
  /\ witness_confirmed_inproc' = TRUE
  /\ verdict' = "sat"
  /\ verdict_lane' = "cnf"
  /\ pc' = "SatWitness"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       ort_confirmed, f64_revalidated, cell_coverage_complete,
       lane_degrade_logged, emitted>>

\* ------------------------------------------------------------------------
\* Stage: CellEnum (cell_enum.rs try_cell_enumeration, called inside
\* run_bab_with_fallback BEFORE BaB, dispatch.rs:343). Graph models only;
\* Sequential (and any non-qualifying spec) falls through unchanged.
\* ------------------------------------------------------------------------

CellFallThrough ==
  /\ pc = "CellEnum"
  /\ pc' = "BabLoop"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

\* Every cell evaluated definitely-safe by the sound f64 interval forward
\* => UNSAT. Partial coverage can NEVER conclude unsat (cell_enum.rs header).
CellUnsatFullCoverage ==
  /\ pc = "CellEnum"
  /\ ~seq_model
  /\ cell_coverage_complete' = TRUE
  /\ verdict' = "unsat"
  /\ verdict_lane' = "cell"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       lane_degrade_logged, emitted>>

\* A definitely-violating cell: the representative is a concrete witness,
\* re-confirmed in-process, then by the vnncomp ORT gate downstream.
CellSatCandidate ==
  /\ pc = "CellEnum"
  /\ ~seq_model
  /\ witness_confirmed_inproc' = TRUE
  /\ verdict' = "sat"
  /\ verdict_lane' = "cell"
  /\ pc' = "SatWitness"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       ort_confirmed, f64_revalidated, cell_coverage_complete,
       lane_degrade_logged, emitted>>

\* Deadline hit with partial coverage => Timeout (never unsat).
CellTimeout ==
  /\ pc = "CellEnum"
  /\ ~seq_model
  /\ verdict' = "timeout"
  /\ verdict_lane' = "cell"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, emitted>>

\* ------------------------------------------------------------------------
\* Stage: MipOnly (dispatch.rs run_mip_only; also the BaB->MIP escalation
\* target). Lane availability is a BUILD fact, not a config fact.
\* ------------------------------------------------------------------------

\* Binary built without --features mip: the verify_with_mip stub bails
\* (mod.rs:63-87) and vnncomp's run_and_translate maps the error to the sound
\* `unknown` (vnncomp.rs:2027). The lane is LOST but no verdict is faked.
MipUnavailable ==
  /\ pc = "MipOnly"
  /\ ~mip_compiled
  /\ verdict' = "unknown"
  /\ verdict_lane' = "none"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, emitted>>

\* FAULT INJECTION (BugPhantomMipLane): the compiled-out lane still "decides".
MipPhantomLane ==
  /\ BugPhantomMipLane
  /\ pc = "MipOnly"
  /\ ~mip_compiled
  /\ verdict' = "unsat"
  /\ verdict_lane' = "mip"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, emitted>>

\* Solver-pin resolution (mod.rs resolve_preset_mip_solver): a legacy preset
\* solver pin (e.g. `scip`) degrades to ay WITH a warning — the fix for the
\* historical config-only pin. Under BugConfigOnlyPin the degrade
\* is silent (the historical behavior: config claimed a lane that never ran
\* and nothing recorded the substitution).
ScipDegradeLogged ==
  IF preset_pins_scip /\ ~scip_compiled
  THEN (~BugConfigOnlyPin) \/ lane_degrade_logged
  ELSE lane_degrade_logged

MipUnsat ==
  /\ pc = "MipOnly"
  /\ mip_compiled
  /\ lane_degrade_logged' = ScipDegradeLogged
  /\ verdict' = "unsat"
  /\ verdict_lane' = "mip"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, emitted>>

\* A MIP counterexample is re-validated in-process before it is claimed
\* (witness-revalidation discipline), then ORT-gated downstream.
MipSatCandidate ==
  /\ pc = "MipOnly"
  /\ mip_compiled
  /\ lane_degrade_logged' = ScipDegradeLogged
  /\ witness_confirmed_inproc' = TRUE
  /\ verdict' = "sat"
  /\ verdict_lane' = "mip"
  /\ pc' = "SatWitness"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       ort_confirmed, f64_revalidated, cell_coverage_complete, emitted>>

MipInconclusive ==
  /\ pc = "MipOnly"
  /\ mip_compiled
  /\ lane_degrade_logged' = ScipDegradeLogged
  /\ verdict' \in {"unknown", "timeout"}
  /\ verdict_lane' = "none"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, emitted>>

\* ------------------------------------------------------------------------
\* Stages: BabLoop + GpuDispatch (verify_standard / verify_relational via
\* BetaCrownVerifier; GPU routing via sound_gpu_gate.rs). The route is a
\* process-global decision, so it is resolved once before any verdict bound.
\* ------------------------------------------------------------------------

BabResolveGpuRoute ==
  /\ pc = "BabLoop"
  /\ gpu_backward_used = "unresolved"
  /\ pc' = "GpuDispatch"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

\* gpu_crown_backward_route (sound_gpu_gate.rs:165-185):
\*   gate engaged + sound GPU capability  -> the sound GPU-resident backward;
\*   gate engaged + no sound capability   -> None -> CPU f64+gamma_n*S fallback;
\*   gate NOT engaged                     -> the fast f32 path may decide
\*                                           (explicit user opt-out), or CPU.
GpuRouteSound ==
  /\ pc = "GpuDispatch"
  /\ sound_gate_engaged
  /\ gpu_sound_capability
  /\ gpu_backward_used' = "gpu_sound"
  /\ pc' = "BabLoop"
  /\ UNCHANGED <<env, sound_gate_engaged, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

GpuRouteCpuFallback ==
  /\ pc = "GpuDispatch"
  /\ sound_gate_engaged
  /\ ~gpu_sound_capability
  /\ gpu_backward_used' = "cpu_f64"
  /\ pc' = "BabLoop"
  /\ UNCHANGED <<env, sound_gate_engaged, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

GpuRouteUngated ==
  /\ pc = "GpuDispatch"
  /\ ~sound_gate_engaged
  /\ gpu_backward_used' \in {"gpu_fast_f32", "gpu_sound", "cpu_f64"}
  /\ pc' = "BabLoop"
  /\ UNCHANGED <<env, sound_gate_engaged, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

\* FAULT INJECTION (BugUnsoundGpuVerdict): the gate is engaged yet the fast
\* round-to-nearest f32 backward decides the verdict bound anyway.
GpuRouteGateBypassed ==
  /\ BugUnsoundGpuVerdict
  /\ pc = "GpuDispatch"
  /\ sound_gate_engaged
  /\ gpu_backward_used' = "gpu_fast_f32"
  /\ pc' = "BabLoop"
  /\ UNCHANGED <<env, sound_gate_engaged, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

\* BaB decides Verified (includes the IBP fast-path — CPU, sound — folded in).
BabUnsat ==
  /\ pc = "BabLoop"
  /\ gpu_backward_used # "unresolved"
  /\ verdict' = "unsat"
  /\ verdict_lane' = "bab"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, emitted>>

\* PGD / BaB counterexample: confirm_potential_violation re-checks it with a
\* concrete in-process forward before it becomes Violated (dispatch.rs:462).
BabSatCandidate ==
  /\ pc = "BabLoop"
  /\ gpu_backward_used # "unresolved"
  /\ witness_confirmed_inproc' = TRUE
  /\ verdict' = "sat"
  /\ verdict_lane' = "bab"
  /\ pc' = "SatWitness"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       ort_confirmed, f64_revalidated, cell_coverage_complete,
       lane_degrade_logged, emitted>>

\* BaB inconclusive + `--features mip` build: auto-escalate to the MIP
\* complete verifier (dispatch.rs:481-601). Only unknown/timeout escalate; a
\* decided verdict is never discarded (should_auto_escalate_to_mip).
BabEscalateToMip ==
  /\ pc = "BabLoop"
  /\ gpu_backward_used # "unresolved"
  /\ mip_compiled
  /\ pc' = "MipOnly"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

BabInconclusive ==
  /\ pc = "BabLoop"
  /\ gpu_backward_used # "unresolved"
  /\ verdict' \in {"unknown", "timeout"}
  /\ verdict_lane' = "none"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, emitted>>

\* ------------------------------------------------------------------------
\* Stage: SatWitness — the vnncomp trusted-oracle admission gate
\* (vnncomp.rs:2033-2281). EVERY internal sat passes here before scoring.
\* ------------------------------------------------------------------------

\* ORT reproduces the violation on a real ONNX-Runtime forward AND the sound
\* true-f64 interval forward does not refute the witness (this includes the
\* declared-bound-snap re-pass path, which requires BOTH gates again).
SatAdmitted ==
  /\ pc = "SatWitness"
  /\ ort_confirmed' = TRUE
  /\ f64_revalidated' = TRUE
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, cell_coverage_complete, lane_degrade_logged,
       verdict, verdict_lane, emitted>>

\* Any gate failure (ORT unavailable, ORT says SAFE and refinement finds
\* nothing, true-f64 definitively rejects and the snapped witness fails):
\* downgrade to the sound `unknown`. A sat is NEVER emitted unconfirmed.
SatDowngraded ==
  /\ pc = "SatWitness"
  /\ verdict' = "unknown"
  /\ verdict_lane' = "none"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, emitted>>

\* FAULT INJECTION (BugSkipOrtGate): the internal sat is scored directly.
SatGateSkipped ==
  /\ BugSkipOrtGate
  /\ pc = "SatWitness"
  /\ pc' = "Emit"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane,
       emitted>>

\* ------------------------------------------------------------------------
\* Stage: Emit — write the scored RESULTS token.
\* ------------------------------------------------------------------------
EmitVerdict ==
  /\ pc = "Emit"
  /\ emitted' = TRUE
  /\ pc' = "Done"
  /\ UNCHANGED <<env, sound_gate_engaged, gpu_backward_used, drat_checked,
       witness_confirmed_inproc, ort_confirmed, f64_revalidated,
       cell_coverage_complete, lane_degrade_logged, verdict, verdict_lane>>

Terminated ==
  /\ pc = "Done"
  /\ UNCHANGED vars

Next ==
  \/ StartDispatch
  \/ CnfFallThrough \/ CnfUnsatDratChecked \/ CnfUnsatUnchecked \/ CnfSatCandidate
  \/ CellFallThrough \/ CellUnsatFullCoverage \/ CellSatCandidate \/ CellTimeout
  \/ MipUnavailable \/ MipPhantomLane \/ MipUnsat \/ MipSatCandidate \/ MipInconclusive
  \/ BabResolveGpuRoute \/ GpuRouteSound \/ GpuRouteCpuFallback
  \/ GpuRouteUngated \/ GpuRouteGateBypassed
  \/ BabUnsat \/ BabSatCandidate \/ BabEscalateToMip \/ BabInconclusive
  \/ SatAdmitted \/ SatDowngraded \/ SatGateSkipped
  \/ EmitVerdict
  \/ Terminated

Spec == Init /\ [][Next]_vars

\* ==========================================================================
\* INVARIANTS — the admission theorem
\* ==========================================================================

TypeOK ==
  /\ pc \in Stages
  /\ mip_compiled \in BOOLEAN /\ scip_compiled \in BOOLEAN
  /\ ay_binary_present \in BOOLEAN /\ gpu_sound_capability \in BOOLEAN
  /\ preset_loaded \in BOOLEAN /\ preset_requires_mip \in BOOLEAN
  /\ preset_pins_scip \in BOOLEAN /\ competition_mode \in BOOLEAN
  /\ allow_unsound_gpu \in BOOLEAN /\ seq_model \in BOOLEAN
  /\ sound_gate_engaged \in BOOLEAN
  /\ gpu_backward_used \in GpuRoutes
  /\ drat_checked \in BOOLEAN /\ witness_confirmed_inproc \in BOOLEAN
  /\ ort_confirmed \in BOOLEAN /\ f64_revalidated \in BOOLEAN
  /\ cell_coverage_complete \in BOOLEAN /\ lane_degrade_logged \in BOOLEAN
  /\ verdict \in Verdicts
  /\ verdict_lane \in Lanes
  /\ emitted \in BOOLEAN

\* The gate value is exactly ProofOpts::sound_gpu_crown_required(); in
\* particular competition mode can NEVER disengage it (mod.rs:292-295).
GateContract ==
  /\ sound_gate_engaged = (competition_mode \/ ~allow_unsound_gpu)
  /\ competition_mode => sound_gate_engaged

\* THE ADMISSION THEOREM, unsat leg: `Verified was emitted` implies every
\* required gate for the deciding lane actually passed.
AdmissionSound ==
  (verdict = "unsat") =>
    /\ verdict_lane \in {"cnf", "cell", "mip", "bab"}
    \* the deciding lane's evidence obligations:
    /\ (verdict_lane = "cnf")  => (drat_checked /\ ay_binary_present)
    /\ (verdict_lane = "cell") => cell_coverage_complete
    /\ (verdict_lane = "mip")  => mip_compiled
    \* sound-path-only: with the gate engaged (default; always in competition
    \* mode) the unsound fast f32 GPU backward never decided the bound.
    /\ sound_gate_engaged => (gpu_backward_used # "gpu_fast_f32")

\* THE ADMISSION THEOREM, sat leg: a scored `sat` was ORT-re-confirmed,
\* true-f64 re-validated, and carried an in-process-confirmed witness.
SatGated ==
  (emitted /\ verdict = "sat") =>
    (ort_confirmed /\ f64_revalidated /\ witness_confirmed_inproc)

\* No silent lane loss (the SCIP-pin / build-drift theorem):
\*  (a) a lane whose feature is compiled out never claims a verdict;
\*  (b) a config-only solver pin that degrades MUST leave a visible record
\*      (the warn / receipt fact) on any verdict the lane decides.
NoSilentLaneLoss ==
  /\ (preset_requires_mip /\ ~mip_compiled) => (verdict_lane # "mip")
  /\ (verdict_lane = "mip" /\ preset_pins_scip /\ ~scip_compiled)
       => lane_degrade_logged

\* A verdict is only ever emitted with a definite result token.
EmitComplete ==
  emitted => (verdict \in {"unsat", "sat", "unknown", "timeout"})

====

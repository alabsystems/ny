// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core `BetaCrownConfig` struct, `Default` impl, presets, and domain helper methods.

use std::time::{Duration, Instant};

use ny_core::{nan_propagating_max, NyError, Result};
use serde::{Deserialize, Serialize};

use crate::pgd_attack::{
    AdamClippingParams, ExactVjpPgdConfig, OpenDisjunctAttackConfig, PgdAlphaMode, PgdConfig,
    PgdInitialization, PgdOptimizer, GAMA_LAMBDA_DEFAULT,
};
use crate::AlphaCrownConfig;

use super::cut_config::{CutEvictionPolicy, CutScoreWeights};
use super::defaults::*;
use super::phase_budget::PhaseBudgetConfig;
use super::AdaptiveOptConfig;
use crate::beta_crown::branching::BranchingHeuristic;

/// Authority requested from a verification run.
///
/// `CertificateExport` is the fail-closed LIBRARY default: a proof lane that
/// cannot be represented by the current external certificate format must
/// decline verdict authority. `VerdictOnly` is an explicit request to produce
/// only the runtime verdict.
///
/// #soundness-spine M2a: the CLI frontends no longer derive this from a
/// "competition mode". It is the engine-side switch of `--require-certificate`
/// alone (`ProofOpts::verification_artifact_authority`): both proof profiles
/// run `VerdictOnly` unless that flag is given, so a Proof-profile run no
/// longer refuses verdicts the lanes certified at runtime. The proof profile
/// never reaches this value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationArtifactAuthority {
    /// The caller may consume a runtime verdict without an exported certificate.
    VerdictOnly,
    /// The caller requested certificate emission for any `Verified` verdict.
    #[default]
    CertificateExport,
}

/// Configuration for β-CROWN branch-and-bound search.
///
/// This pre-1.0 configuration surface grows as new proof policies acquire
/// typed authority. Downstream Rust callers should construct it from
/// [`Self::default`] and override selected fields instead of using exhaustive
/// struct literals; serialized configurations remain backward-compatible
/// through `#[serde(default)]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BetaCrownConfig {
    /// Runtime proof-artifact authority resolved by the frontend.
    ///
    /// This is deliberately not configurable through presets or serialized
    /// solver configuration. Frontends must derive it from their typed proof
    /// request after loading every preset; deserialization therefore restores
    /// the fail-closed `CertificateExport` default.
    #[serde(skip)]
    pub verification_artifact_authority: VerificationArtifactAuthority,
    /// Receipt-owning, immutable per-run parameter authority.
    ///
    /// Scored frontends attach the exact [`ny_levers::LeverSet`] that produces
    /// the flight receipt. It is runtime-only and graph clones retain the same
    /// [`std::sync::Arc`], so downstream cache identity and migrated readers
    /// cannot silently reopen process environment. Flight-v3 intentionally
    /// omits some lower-layer metadata and does not serialize this field.
    /// Interactive callers retain `None` until they enter the compatibility
    /// freeze boundary.
    #[serde(skip)]
    pub frozen_levers: Option<std::sync::Arc<ny_levers::LeverSet>>,
    /// Shared original-query optional-work owner; clones never renew its attempt.
    #[serde(skip)]
    pub certified_root_cut_owner:
        Option<std::sync::Arc<super::super::certified_root_cut::QueryOwner>>,
    /// Set only by the frontend that loaded this graph with PreserveRaw.
    /// Deserialization cannot authorize a default-folded graph.
    #[serde(skip)]
    pub certified_root_cut_raw_source: bool,
    /// Maximum number of domains to explore before giving up.
    #[serde(default = "default_max_domains")]
    pub max_domains: usize,
    /// Maximum number of domains to store in the domain queue simultaneously.
    ///
    /// When the queue exceeds this limit, the lowest-priority domains are evicted
    /// to prevent unbounded memory growth during BaB on hard verification instances.
    /// Set to 0 to disable the cap (unbounded queue, original behavior).
    #[serde(default = "default_max_queue_size")]
    pub max_queue_size: usize,
    /// Estimated payload cap in bytes for supported graph BaB frontiers.
    ///
    /// Coverage includes ordinary and precomputed ReLU-split heaps plus scalar
    /// GPU `DomainList` ReLU- and input-split routes. Heap routes use
    /// `estimate_graph_domain_bytes`; `DomainList` routes recompute their
    /// shared live-row census after every add. This is a logical-payload
    /// estimate, not an allocator-capacity, process-RSS, or device-memory
    /// limit. Grouped-disjunctive `DomainList` rejects a nonzero cap until its
    /// row sidecar is fully censused.
    ///
    /// When the resident queue exceeds the cap, the lowest-priority (least
    /// promising) domains are discarded. The completeness latch prevents a
    /// drained queue from yielding `Verified`; a concrete counterexample can
    /// still refute the property. Surviving bounds are unchanged.
    ///
    /// `0` disables the cap and is the default, preserving uncapped queue
    /// semantics. A queue retains at least one domain for forward progress, so
    /// one oversized domain may exceed the requested estimate.
    ///
    /// Reference: #ml4acopf-bab-queue-mem
    #[serde(default)]
    pub max_queue_bytes: usize,
    /// Concrete engine timeout for the entire search.
    ///
    /// A zero duration expires immediately. Frontends that expose zero as an
    /// unbounded sentinel must translate it before constructing the verifier.
    #[serde(default = "default_timeout")]
    pub timeout: Duration,
    /// Maximum depth of the search tree (max number of splits).
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,

    /// #budget-abandonment FIX A. When the ITERATIVE root α-CROWN warmup blows
    /// its `phase_budget.initial_bounds_fraction` cap while the global BaB
    /// deadline is still live, fall back to a root enclosure the mandatory
    /// bootstrap sweep ALREADY produced instead of returning `Unknown` and
    /// abandoning the remaining budget.
    ///
    /// Measured at HEAD on 14_ieee_ml4acopf: the bootstrap ran 0 -> 121.0 s
    /// against a 114 s warmup cap and the row returned `Unknown` with 449 s of
    /// its 600 s budget unspent, BaB never started. The work was already in
    /// hand when it was thrown away: `compute_relu_split_initial_bounds`
    /// completes `compute_graph_bab_bootstrap` FIRST and only then calls
    /// `compute_graph_root_output_bounds`, whose `?` discarded
    /// `bootstrap.initial_node_bounds` along with the error.
    ///
    /// SOUND BY MONOTONICITY: the fallback enclosure is WIDER than the
    /// optimized one, and both `domain_is_verified_for_mode` and
    /// `domain_is_violation_for_mode` are harder to satisfy on a wider
    /// interval, so a looser root can mint neither an `unsat` nor a spurious
    /// violation — only more search.
    ///
    /// `false` (the shipped default, and the state of every yaml) is
    /// byte-identical to the behaviour before the key existed.
    #[serde(default)]
    pub initial_bounds_warmup_fallback: bool,

    /// #budget-abandonment FIX B. Number of EXTRA passes the disjunctive
    /// multi-clause loop may make over the clauses it failed to decide.
    ///
    /// The per-clause budget allocator is one pass and back-loads the budget
    /// (`verify/disjunctive.rs`: `remaining / remaining_bab`), so the FIRST
    /// clauses get the SMALLEST slice and the failures concentrate exactly
    /// there. Measured on 14_ieee_ml4acopf prop1 (56 OR-clauses): clause 1
    /// received 10 s and died; clause 56 received 442 s and used 2.1 s; the
    /// row ended with 470 s of 600 unspent.
    ///
    /// `0` (the shipped default) is today's single pass, byte-identical.
    #[serde(default)]
    pub disjunctive_clause_retry_passes: usize,

    /// #disjunctive-shared-bootstrap. Compute the graph BaB bootstrap
    /// (intermediate node bounds, root alpha state, identity-spec root output
    /// box) ONCE per serial multi-clause disjunction and hand every clause an
    /// immutable `Arc` of it instead of re-running it per clause.
    ///
    /// The serial disjunctive loop (`verify/disjunctive.rs`) calls the graph
    /// ReLU-split engine once per OR-clause, and every call re-runs
    /// `compute_graph_bab_bootstrap` + `compute_graph_root_output_bounds`
    /// from scratch. Measured 2026-09-12 on 14_ieee_ml4acopf prop4 (28
    /// clauses, 600 s): `Computing alpha-CROWN initial bounds (200
    /// iterations)` printed once PER CLAUSE and `[warmup-fallback]` fired on
    /// every clause at 6.3-7.5 s against a 5.0 s cap, so the row spent ~7 s x
    /// 28 recomputing the SAME intermediate bounds. The clauses differ only in
    /// their output constraint; the network's intermediate enclosures over the
    /// property's input box do not depend on it.
    ///
    /// SOUND BY CONSTRUCTION: the shared node bounds, root alpha and identity
    /// output box are valid enclosures of the same network over the same input
    /// box for every clause -- exactly what each per-clause bootstrap computed
    /// before. What is NEVER shared: a clause's own per-clause input box
    /// (`per_clause_input_bounds` non-empty => that clause re-bootstraps on its
    /// own box), INVPROP-conditioned node bounds (the cache is refused outright
    /// when `alpha_config.invprop.enabled`), and any state a clause's BaB
    /// mutates (split constraints, beta, cuts, the domain queue) -- those are
    /// built per clause from a clone of the shared map, as the existing
    /// precomputed lane already does. Each clause's root objective bound is
    /// the projection of the shared identity output box (what the per-clause
    /// path computes) INTERSECTED with one spec-guided CROWN pass run with the
    /// shared root alpha slopes -- two certified enclosures of one scalar --
    /// and its BaB children warm-start from those slopes, as the per-clause
    /// lane's do. What is NOT reproduced is the per-clause warmup's
    /// spec-guided early exit, which only ever stopped a warmup SOONER.
    ///
    /// Refusals are named in the `[shared-bootstrap]` line: `dark`,
    /// `sequential`, `route` (only ReLU-split without gpu_bab reads the
    /// precomputed slot), `invprop`, `single`, `expired`, `declined` (the
    /// shared root pass blew its cap and FIX A is not armed), `error`. Every
    /// refusal runs the per-clause bootstrap exactly as before.
    ///
    /// `false` (the shipped default, and the state of every yaml) is
    /// byte-identical to the per-clause bootstrap.
    #[serde(default)]
    pub disjunctive_shared_bootstrap: bool,

    /// #genbab-precomputed-lane. Let the PRECOMPUTED-BOUNDS BaB lane
    /// (`verify_graph_relu_split_with_bounds`) honour a configured GenBaB
    /// nonlinear branching heuristic, as every other BaB lane already does.
    ///
    /// THE DEFECT THIS EXISTS FOR. `relu_split_bounds.rs` is the only BaB lane
    /// that never reads `branching_heuristic`. It goes straight to
    /// `find_unstable_graph_neurons`, which scores ReLU nodes only. The
    /// sequential lane's `process_sequential_domains` instead checks
    /// `BranchingHeuristic::GenBaB(_)` FIRST and routes to
    /// `process_graph_domain_parallel`, which can split a nonlinear node.
    ///
    /// On a benchmark whose branching is `method: nonlinear` and whose ReLUs
    /// are all stable -- ml4acopf, whose real branching targets `MulBinary`
    /// power-flow products -- the precomputed lane therefore finds NOTHING to
    /// split, sets `unresolved_due_to_no_branch`, and returns. Measured on
    /// `14_ieee_ml4acopf prop1` clause 39: the lane printed
    /// `Found 3 ReLU nodes for branching`, performed ZERO splits, and returned
    /// in ~48 MICROSECONDS while holding a 562 s allowance; the row published
    /// `unknown` with 98.0 % of its 600 s budget unspent. The sequential lane's
    /// same clause split to the 10000 depth cap using 192 s.
    ///
    /// SOUND EITHER WAY: branching only decides which sound sub-box to bound
    /// next, never a verdict -- the same argument `configs/vnncomp26/
    /// ml4acopf_2024.yaml` already makes for `method: nonlinear` itself. The
    /// enqueue/verify/violation fold is not reimplemented here; this routes
    /// through the SAME `aggregate_bab_results` the sequential lane uses, so
    /// the two cannot disagree about what a domain result means.
    ///
    /// SCOPE OF A NEARBY DIVERGENCE — stated precisely, because a first pass
    /// got it wrong in the cautious direction and an overstated warning is its
    /// own kind of false claim. This lane has a SECOND divergence from the
    /// others: **its own inline child bounding never optimises beta or
    /// alpha.** Every other BaB lane bounds children through
    /// `evaluate_graph_child_bounds` (`graph/propagation/beta.rs`), whose gate
    /// is `beta_iterations > 0 && child.depth <= beta_max_depth` and which then
    /// runs the analytical or SPSA optimiser — `relu_split/child_eval.rs`,
    /// `gpu_bab/batched_gpu.rs`, `gpu_bab/kfsb.rs`,
    /// `multi_objective/domain_process.rs`, `batched/batched_single.rs`. This
    /// lane instead calls `propagate_crown_with_graph_beta`, which is a pure
    /// forward to `propagate_crown_with_graph_constraints`: ONE pass, zero
    /// optimiser iterations, carrying only the parent's inherited multipliers.
    /// It never reads `beta_max_depth` at all, and consumes `beta_iterations`
    /// once at the root as a bool.
    ///
    /// BUT IT DOES NOT APPLY TO THIS ROUTE. The GenBaB arm added here ends in
    /// `continue`, so the lane's inline child bounding is never reached on it.
    /// `process_graph_domain_parallel` bounds ITS children through
    /// `evaluate_graph_child_bounds` — the optimising evaluator, gated
    /// `beta_iterations > 0 && depth <= beta_max_depth` — and additionally
    /// optimises a no-unstable LEAF regardless of those knobs, using
    /// `max(beta_iterations, root_beta_iterations)` as the leaf budget
    /// (`multi_objective/domain_process.rs`). So with
    /// `configs/vnncomp25/ml4acopf_2024.yaml`'s `iterations: 50`, children on
    /// THIS route are optimised.
    ///
    /// The divergence still bites the lane's ReLU-split path, which is what a
    /// reader arming this key on some OTHER benchmark should know: there,
    /// fifty beta iterations per child really are discarded. Fixing that is a
    /// separate change to lines that this key does not touch.
    ///
    /// `false` (the shipped default) keeps the lane byte-identical.
    #[serde(default)]
    pub genbab_on_precomputed_lane: bool,
    /// Use α-CROWN optimization within each domain.
    #[serde(default = "default_use_alpha_crown")]
    pub use_alpha_crown: bool,
    /// Reuse forward-linear intermediate bounds in the final backward pass.
    ///
    /// Matches alpha-beta-CROWN `forward+backward` / `forward+crown`
    /// semantics: compute intermediate graph bounds with forward symbolic
    /// relaxation, then keep the final objective solve on backward CROWN.
    #[serde(default)]
    pub use_forward_bounds: bool,
    /// Optimize alpha slopes independently for each disjunct in OR-spec
    /// properties. Reference: alpha-beta-CROWN `optimize_disjuncts_separately`.
    #[serde(default)]
    pub optimize_disjuncts_separately: bool,
    /// Optional scheduling cap in seconds for the root alpha-CROWN warmup,
    /// including its intermediate-bound collection. It only narrows the
    /// existing deadline; expiry preserves the route's certified fallback or
    /// refusal behavior.
    /// A positive finite `NY_ROOT_ALPHA_CAP_SECS` overrides this setting.
    #[serde(default)]
    pub root_alpha_cap_secs: Option<f64>,
    /// Retain a completed, certified DAG-alpha artifact when the root warmup's
    /// local phase cap expires, so the multi-objective root can re-evaluate it
    /// under the still-live outer verifier deadline.
    ///
    /// This is scheduling state only: the retained alpha/reference map cannot
    /// grant a verdict, and every objective is evaluated by the normal certified
    /// root path. Default false keeps unqualified presets on their historical
    /// deadline mapping. An absent `NY_ROOT_ALPHA_PHASE_CHECKPOINT` inherits
    /// this setting, literal `1` force-arms, and every other present value
    /// force-disarms the typed policy.
    #[serde(default)]
    pub root_alpha_phase_checkpoint: bool,
    /// Number of additional exact full-`C` root margin-Adam iterations.
    ///
    /// Zero is the default-dark historical behavior. A positive value arms the
    /// score-bearing atomic root-`C` route directly (without arming the DAG
    /// identity pre-loop), re-evaluating the complete source-ordered `C` matrix
    /// after every proposal. The resource bound is deliberately small: at most
    /// [`ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS`] whole-`C` proposals may run.
    #[serde(default)]
    pub atomic_root_c_margin_iterations: usize,
    /// Intersect CROWN backward intermediate bounds with IBP bounds on supported
    /// routes, at additional time and memory cost.
    #[serde(default)]
    pub use_crown_ibp: bool,
    /// Frozen runtime decision for objective-chunk parallel waves.
    ///
    /// This is not preset-configurable: scored VNN-COMP derives it once from
    /// the receipt-owning LeverSet, while the interactive CLI resolves its
    /// legacy environment compatibility layer once before verification. False
    /// is the production one-frontier memory guard.
    #[serde(skip)]
    pub chunk_wave_parallel_enabled: bool,
    /// Maximum number of sequential CROWN-IBP partial nodes before falling back
    /// to plain IBP intermediate bounds.
    ///
    /// Counts only the layer outputs that would trigger an expensive partial
    /// CROWN pass, not every layer in the sequential network.
    #[serde(default)]
    pub max_crown_ibp_nodes: Option<usize>,
    /// Floor (seconds) for CROWN-IBP collectors' per-node time budget
    /// (#cgan-bn11-budget). A node whose share
    /// falls below the floor is skipped to sound IBP. `None` (default) keeps
    /// the built-in 2.0 s constant — behavior is unchanged when unset.
    #[serde(default)]
    pub crown_ibp_per_node_floor_secs: Option<f64>,
    /// #per-node-floor-dim-scale: scale the CROWN-IBP per-node floor by
    /// `min(1, raw_node_dims / 28,800)` before the below-floor comparison, so
    /// the constant calibrated for 28,800-row conv targets stops refusing
    /// 20/118/128-wide dense nodes whose whole backward costs milliseconds.
    /// Composes with `crown_ibp_per_node_floor_secs` (whatever floor resolves
    /// is scaled). `false` (default) keeps the constant floor EXACTLY.
    /// Scheduling only: an admitted target publishes CROWN-IBP instead of
    /// IBP, both sound, the former tighter-or-equal.
    #[serde(default)]
    pub crown_ibp_per_node_floor_dim_scaled: bool,
    /// #pwl-lookup (preset `bab.pwl_lookup_exact_range`): intersect the IBP
    /// box of every ReLU LOOKUP TABLE output — `SubConstant(b) → ReLU →
    /// Linear(W ∈ R^{1×K})` (ONNX `MatMul(relu, w[K])`) or `→ MulConstant(w)
    /// → ReduceSum(-1)` — with the table's EXACT 1-D range, evaluated at the
    /// interval endpoints and interior breakpoints in outward-rounded f64. The
    /// verifier stamps it onto the loaded graph
    /// (`GraphNetwork::set_pwl_lookup_exact_range`), where both the IBP
    /// fast-path forward and the node-bound collector read it.
    ///
    /// SOUND: a shrink-only intersection of two enclosures of the same value;
    /// an empty intersection keeps the plain box (fail closed). It can only
    /// tighten a bound, so `unknown -> unsat` is reachable and `unsat -> sat`
    /// is not. `false` (default) is byte-identical. MEASUREMENT target:
    /// ml4acopf_2024 prop3/prop4 on the six `-linear-*` nets (12 rows/year).
    #[serde(default)]
    pub pwl_lookup_exact_range: bool,
    /// #fwd-root-screen (V4, preset `bab.forward_root_screen`): before the
    /// disjunctive CROWN precheck, run ONE certified correlation-preserving
    /// zonotope forward pass over the root input box (`zono_core::dag`:
    /// explicit symbol IDs, ID-aligned products, rigorous Sigmoid/Sin/Cos/Floor
    /// and lookup-table bands, one fresh symbol per unstable ReLU, every
    /// relaxation interval intersected with ny's IBP box and C1's exact range,
    /// binary64 with explicit Higham charges and outward endpoints) and refute
    /// every clause its certified output box rules out.
    ///
    /// SOUND: the verdict is the UNION with the existing precheck — the screen
    /// only adds individually certified refutations and never replaces
    /// IBP/C1. Any unsupported op, NaN/Inf, byte cap or deadline declines (no
    /// clause refuted by it). `false` (default) is byte-identical: no work, no
    /// output. MEASUREMENT target: the four ml4acopf rows F1 verified with the
    /// ZONO replica (14 base p1, 118 lin-res p5, 300 lin-nonres/lin-res p105).
    #[serde(default)]
    pub forward_root_screen: bool,
    /// #fwd-root-screen live-state ceiling in MiB (preset
    /// `bab.forward_root_screen_max_mb`, default 8192, admissible
    /// `64..=1048576`; 0 is refused, never "unlimited"). The admitted budget is
    /// additionally clamped to half the process memory headroom.
    #[serde(default = "default_forward_root_screen_max_mb")]
    pub forward_root_screen_max_mb: u32,
    /// #vit-zono-root (VIBECHECK I1, preset `bab.vit_zono_root`): compute ONE
    /// certified forward-zonotope ROOT map over the input box
    /// (`zono_core::vit_root`, the `zono_core::dag` program with the vit op set:
    /// live-live MatMul with a certified remainder symbol, Exp / Reciprocal /
    /// Clip bands, ReduceSum / ReduceMean, the direct-Softmax pairwise box,
    /// BatchNorm with certified radii, Conv2d, the probed data movement) and
    /// intersect its per-node boxes SHRINK-ONLY into the alpha reference map
    /// the root CROWN / alpha-margin pass reads (also the CROWN-IBP collector's
    /// baseline on the `fix_interm_bounds: false` route).
    ///
    /// SOUND: every node box is a certified enclosure (binary64, explicit
    /// Higham charges, outward endpoints, explicit symbol IDs) intersected with
    /// ny's IBP box; a runtime self-check (program point semantics vs ny's
    /// pointwise forward, sampled containment) must pass before anything is
    /// published; merges are shrink-only and a disjoint element keeps the
    /// current value. Any refusal publishes nothing. `false` (default) is
    /// byte-identical: no plan is stamped, no work, no output. MEASUREMENT
    /// target: vit_2023 (27 ibp root rows, 44 pgd root rows abc proves).
    #[serde(default)]
    pub vit_zono_root: bool,
    /// #vit-zono-root live-state ceiling in MiB (preset
    /// `bab.vit_zono_root_max_mb`, default 8192, admissible `64..=1048576`; 0
    /// is refused, never "unlimited"). The admitted budget is additionally
    /// clamped to half the process memory headroom (no cgroup governor in
    /// competition).
    #[serde(default = "default_vit_zono_root_max_mb")]
    pub vit_zono_root_max_mb: u32,
    /// #vit-zono-root wall cap of one root-map computation in seconds (preset
    /// `bab.vit_zono_root_max_secs`, default 30, admissible `(0, 3600]`),
    /// additionally bounded by the collection deadline.
    #[serde(default = "default_vit_zono_root_max_secs")]
    pub vit_zono_root_max_secs: f32,
    /// #vit-collector-throughput P2 step 1 (typed `bab.bilinear_rowpar_kernel`):
    /// run the var×var `MatMul` / `BilinearCrown` McCormick backward (both the
    /// midpoint arm and the interpolated α arm) row-parallel, with each
    /// element's McCormick planes computed once per chunk of rows instead of
    /// once per row. BIT-IDENTICAL to the serial nest (rows are independent and
    /// every row keeps its exact operation sequence;
    /// `bilinear_rowpar_bit_identical_to_serial`). Throughput only. `false`
    /// (default) keeps the serial nest. Delivered to the kernels through
    /// `layers::collector_kernels::arm` when a `BetaCrownVerifier` is built.
    #[serde(default)]
    pub bilinear_rowpar_kernel: bool,
    /// #vit-collector-throughput P3 (typed `bab.linear_position_stack`): run a
    /// sequence Linear's certified f64 `A·W` as ONE product over every token
    /// position instead of one per position. Bit-identical on the CPU reducer;
    /// a stacked product large enough for the sound f64 engine is sound through
    /// the same summation-order-independent `gamma_n * S` certificate but not
    /// bit-identical. `false` (default) keeps the per-position loop.
    #[serde(default)]
    pub linear_position_stack: bool,
    /// #vit-collector-throughput P4 (typed `bab.reciprocal_unbounded_upper`):
    /// relax `1/x` on `[l, +inf)` (`l > 0` finite) by `0 <= y <= next_up(1/l)`
    /// instead of refusing, and refuse every remaining out-of-domain
    /// pre-activation with `UnsupportedOp` (a per-target IBP fallback in every
    /// collector) instead of `InvalidSpec` (which aborted vit_2023's pgd rows).
    /// `false` (default) keeps the historical guard and error class exactly.
    #[serde(default)]
    pub reciprocal_unbounded_upper: bool,
    /// #vit-batched-bilinear (ENGINE_PROGRAM V1; typed
    /// `bab.bilinear_gemm_kernel`): run the var×var `MatMul` / `BilinearCrown`
    /// McCormick backward (midpoint and α arms) as the batched, GEMM-shaped
    /// kernel (`layers/binary_ops/matmul/crown_gemm.rs`): every element's
    /// planes computed once per call, rows in parallel chunks and tiles,
    /// sign-selected rank-1 panel updates. BIT-IDENTICAL to the serial nest
    /// (every accumulator keeps the nest's operation sequence;
    /// `bilinear_gemm_bit_identical_to_serial`), so throughput only. Tried
    /// before `bilinear_rowpar_kernel` when both are set. `false` (default)
    /// keeps the historical choice. Delivered to the kernels through
    /// `layers::collector_kernels::arm` when a `BetaCrownVerifier` is built.
    #[serde(default)]
    pub bilinear_gemm_kernel: bool,
    /// #hull-root / #hull-bab (wave-6 C4): the certified single-hidden-layer
    /// per-neuron hull bound in the sequential relational lane, run after the
    /// upfront attack and before the same-LHS reduction. `Off` (default,
    /// every shipped yaml) is byte-identical to the pre-key tree. `Root`
    /// certifies the ideal per-neuron hull at the root with a closed form
    /// whose soundness does not depend on the LP that proposes its
    /// multipliers; `Bab` adds a ReLU-split tree over the concave neurons,
    /// certified per node the same way. Either can only turn a timeout into
    /// `unsat`, never flip a verdict. Observables: `[hull-root] armed ...`
    /// (`root`) and `[hull-bab] armed ...` (`bab`) on every armed run.
    #[serde(default)]
    pub single_hidden_hull: SingleHiddenHullMode,
    /// #graph-lane-search (ENGINE_PROGRAM §2 M7, preset `bab.graph_search`):
    /// which UNTRUSTED batched α/β search the graph β-CROWN multi-objective
    /// lane consults. `off` (default, every shipped yaml) never consults a
    /// backend and is byte-identical to the pre-key tree. `cpu_f32` sends the
    /// root (after the lane's own root objective pass) and each shared-executor
    /// wave's open children (at most a fixed count per wave) to the injected
    /// backend over the lane's self-checked `GraphSearchNet` copy, with the
    /// lane's certified boxes frozen; the lane then ingests each hint through a
    /// clamp gate (finite slopes in `[0, 1]` on neurons unstable in the
    /// domain's CERTIFIED boxes, finite multipliers `>= 0` on the domain's own
    /// split literals), certifies each domain ONCE with its existing evaluator
    /// (`SpecCrownRequest` at the root, the β-CROWN multi-objective pass on a
    /// child) and raises a row's LOWER bound only when the certified candidate
    /// is strictly better. The device never certifies; a refusal or a search
    /// bug leaves the lane's own bounds. Observables: `[bbe] phase=graph_bab
    /// stage=root …` / `stage=wave …` and, on a disjoint root row, the
    /// unconditional `[bbe-alarm] phase=graph_bab DISJOINT_SOUNDNESS_ALARM`.
    #[serde(default)]
    pub graph_search: GraphLaneSearch,
    /// #la-warm-frontier (preset `bab.la_warm_frontier`): the FRONTIER-COMPLETE
    /// lA warm start on DAG graphs. Armed, a capturing backward pass records,
    /// at every node a split can name, the rest of its backward state at that
    /// instant (every other pending block and the network-input channel, each
    /// with its certified coefficient error folded outward), and a child whose
    /// single-block seed the #warm-dag guard refuses RESUMES its backward pass
    /// from its branch node's block plus that whole frontier (validated against
    /// the child's rows and widths) instead of from the output. Residual /
    /// multi-consumer nets (yolo, vit, soundnessbench residual models) regain
    /// the warm start the guard took away. `false` (default, every shipped
    /// yaml) is BIT-IDENTICAL to the guard-only tree: nothing is captured and
    /// no seed site consults a frontier. Requires `enable_la_warm_start`.
    /// SOUND for either value: the resumed bound is a bound on the objective
    /// over the child's region (proof in
    /// `engine/graph/propagation/warm_frontier.rs`); a missing or invalid
    /// frontier takes the cold pass. Observable: one
    /// `[la-warm-frontier] lane=… armed=1 captured=… resumed=…` line per armed
    /// multi-objective / DomainList run.
    #[serde(default)]
    pub la_warm_frontier: bool,
    /// #kfsb-single-shot (preset `bab.kfsb_single_shot`): the graph kFSB
    /// selector (`gpu_bab/kfsb.rs::select_graph_branch_kfsb_in_gpu_batched`,
    /// shared by the DomainList ReLU-split lane and the conjunctive
    /// multi-objective sequential lane's aggregation-critical route) scores its
    /// candidates the way αβ-CROWN's kFSB and this crate's own wave-batched
    /// `kfsb_multi` selector do, instead of with two costly stand-ins:
    ///
    /// * PRESCORE: BaBSR seeded with the selector's OBJECTIVE row and stopped
    ///   once every unstable ReLU node is scored, instead of the legacy
    ///   identity seed over the WHOLE graph output (a 21125 x 21125 coefficient
    ///   walk per decision on TinyYOLO);
    /// * CHILD SIMULATION: ONE constrained CROWN pass per candidate child with
    ///   the parent's inherited α/β, instead of the full β(+α) optimizer
    ///   (`beta_iterations` + 1 passes per child at depth <= `beta_max_depth`).
    ///
    /// ADVISORY-ONLY either way: the selector only chooses WHICH ReLU to
    /// partition; the committed children are bounded by the unchanged pipeline,
    /// so the key cannot change the soundness of any published bound. `false`
    /// (default, every shipped yaml) is BIT-IDENTICAL to the historical scorer.
    /// Observable: one `[kfsb-single-shot] armed=1 …` line per armed decision.
    #[serde(default)]
    pub kfsb_single_shot: bool,
    /// #attack-concurrent (preset `attack.concurrent`): run the verifier's
    /// conjunctive / single-clause UPFRONT PGD (the graph and sequential
    /// relational lanes' `upfront_pgd_deadline` slice) on a sibling thread,
    /// in a private Rayon pool, CONCURRENTLY with the proof instead of before
    /// it. The proof decides -> the attack is cancelled through its
    /// `PgdConfig::cancel` token; the attack finds a counterexample -> the
    /// candidate goes through the scored path's unchanged sat pipeline
    /// (trusted-ORT gate, VNN-LIB 2.0 normalization) and, upheld, is published
    /// first-verdict-wins under the #sat-precedence lock, ending the process.
    ///
    /// The attack's window is stretched so it receives at least the
    /// CPU-seconds the serial slice gave it (`W = slice * (N + k_a) / k_a`),
    /// and the lane stands down — today's serial code verbatim — whenever its
    /// admission refuses (no scored publication target, the concurrent
    /// margin-row lane owns concurrency, a shared accelerator, a Sign net, the
    /// QDQ report-only latch, a stretched window that does not fit, a denied
    /// memory lease). `false` (default, every shipped yaml) is BIT-IDENTICAL:
    /// the site reads this one bool and runs the historical serial code.
    /// SOUNDNESS: scheduling only — the proof is the unchanged code on the
    /// unchanged deadlines, and a `sat` still passes the unchanged gate.
    /// Observable: `[attack-concurrent] admit site=... route=...` and
    /// `[attack-concurrent] done site=... outcome=...` on every armed site.
    #[serde(default)]
    pub attack_concurrent: bool,
    /// Opt-in original-tower CPU point/VJP advice for shared-input relational
    /// falsification. The CLI's early relational route projects this policy
    /// from its frozen preset through the same applicator as ordinary configs.
    /// Every proposal retains the original dual-source witness acceptance.
    #[serde(default)]
    pub relational_point_vjp: bool,
    /// #relusplitter-lp-refine (wave-6 C7, preset
    /// `bab.mip_refine_intermediates: {layers, budget_frac}`): at the
    /// multi-objective graph BaB root, after the bootstrap and before the root
    /// objective pass, refine the hidden pre-activation layers `1..=layers` of
    /// a pure Linear/ReLU chain with RIGOROUS per-neuron triangle-LP bounds
    /// (ny-mip `LpTightener`, Neumaier–Shcherbina or exact rational — never a
    /// float LP optimum), layer-progressively, loosest triangle first, inside
    /// `budget_frac` of the remaining root window; then intersect them
    /// SHRINK-ONLY into the stored boxes that the root bound and every BaB
    /// domain inherit. Vibecheck's recipe is `layers: 3` (its MILP is replaced
    /// by the certified LP). Layer 0 is affine in the input box and already
    /// exact, so it is never re-solved.
    ///
    /// SOUND: every published bound is a rigorous outer bound of an LP whose
    /// feasible set contains the true ReLU graph over sound boxes, and the
    /// publication is an intersection, so `unknown -> unsat` is reachable and
    /// `unsat -> sat` is not. `None` (default, every shipped yaml) attaches no
    /// refiner: byte-identical. Observables: `[mip-refine] armed ...` when the
    /// CLI attaches it and `[mip-refine] root ...` whenever the root runs it.
    #[serde(default)]
    pub mip_refine_intermediates: Option<MipRefineIntermediates>,
    /// #vgg-gpu-crown (preset `bab.gpu_patches_crown: {chunk_rows, max_mb,
    /// headroom_frac, gpu_tile_mb, precision, depth, screen, budget_frac,
    /// max_secs, min_perturbed}`): before the graph BaB, run a certified
    /// patches-mode CROWN root pass — CROWN intermediates for every ReLU
    /// layer, intersected shrink-only into certified one-step screen boxes,
    /// then the spec rows — whose coefficient products run on the sound GPU
    /// GEMM engine (cuBLAS f64 or f32 with the a-posteriori `gamma * |A||B|`
    /// certificate charged against the multiplied value's box), chunked
    /// under host AND device memory admission, every refusal a bit-identical
    /// CPU product. Built for `vggnet16_2022` spec15-17 (all 150528 pixels
    /// perturbed).
    ///
    /// SOUND for any value: it can only return "every objective certified"
    /// (`unknown -> unsat`), never `sat`; a refusal changes nothing. `None`
    /// (default, every shipped yaml) runs nothing and prints nothing.
    /// Observable: `[gpu-patches-crown] armed=1 layers= gpu_passes=
    /// cpu_fallback= secs= peak_mb= ...` on every armed run.
    #[serde(default)]
    pub gpu_patches_crown: Option<crate::gpu_patches_crown::GpuPatchesCrownConfig>,
    /// Explicit base cap (seconds) on CROWN-IBP collectors' per-node time
    /// budget (#cgan-bn11-budget). `None` derives an adaptive cap from
    /// the remaining collection budget (25%, clamped to 12–600 seconds).
    /// Explicit caps are dimension-scaled above 28,800 rows. Purely a
    /// time-vs-tightness policy knob: any value is sound.
    #[serde(default)]
    pub crown_ibp_per_node_cap_secs: Option<f64>,
    /// #target-cost-weight: price CROWN-IBP scheduling weights by the MACs of
    /// the backward walk each target will run (`rows x prefix`) rather than by
    /// its objective-row count.
    ///
    /// The shipped row-count weight is a cost proxy measured wrong by ~9.5x on
    /// cgan (secs/row spans 27x across the same targets), which is why one wide
    /// shallow node takes 54% of the collection window while the deep narrow
    /// targets that carry the value share 2.9%. `false` (default) keeps the
    /// historical weights EXACTLY. Scheduling only: it never changes what a
    /// completed bound is, and every declined target keeps its certified IBP
    /// fallback.
    #[serde(default)]
    pub crown_ibp_prefix_cost_weighted: bool,
    /// #target-cost-weight: maximum fraction of the remaining CROWN-IBP
    /// collection window one non-final target may be granted. `None` (default)
    /// applies no cap; values outside `(0, 1)` are ignored. Sound at any value
    /// — the share only bounds how long a walk may run.
    #[serde(default)]
    pub crown_ibp_max_target_share_frac: Option<f64>,
    /// Maximum number of backward layers/nodes for fixed-slope CROWN.
    ///
    /// When set, CROWN backward stops after propagating through this many
    /// output-adjacent layers or graph nodes and concretizes the remaining
    /// frontier with sound forward bounds. This mirrors alpha-beta-CROWN's
    /// fixed-intermediate-bound truncation during BaB refinement.
    #[serde(default)]
    pub crown_backward_layers: Option<usize>,
    /// α-CROWN configuration (if use_alpha_crown is true).
    #[serde(default)]
    pub alpha_config: AlphaCrownConfig,
    /// Branching heuristic for selecting neurons to split.
    #[serde(default)]
    pub branching_heuristic: BranchingHeuristic,
    /// Number of input dimensions to split per input-splitting step.
    ///
    /// Higher values create 2^k child domains per split. Default: 1.
    #[serde(default = "default_input_split_depth")]
    pub input_split_depth: usize,
    /// Coefficient clamp threshold for input-splitting SB scoring.
    /// Matches alpha-beta-CROWN `sb_coeff_thresh` (default: 1e-3).
    #[serde(default = "default_input_split_coeff_thresh")]
    pub input_split_coeff_thresh: f32,
    /// Per-sub-domain α refinement iterations in the input-split BaB loop.
    ///
    /// When > 0 AND `use_alpha_crown` is enabled, each sub-domain warm-starts
    /// from its parent's optimized alphas and re-optimizes them for this many
    /// SPSA iterations against the sub-domain's tighter box (with
    /// `fix_interm_bounds=true`, skipping the O(N²) intermediate CROWN pass).
    /// The refined alphas are saved onto both child domains. This produces much
    /// tighter per-domain bounds → fewer splits.
    ///
    /// Default 0 keeps the historical single frozen-alpha bound computation per
    /// domain (root-optimized alphas threaded in, zero gradient steps), and the
    /// warm-start branch is never taken. The reordered production path is
    /// unchanged. Eager grouped-disjunctive screening also retains the sound
    /// component-wise lower floors from its parent and clip recomputes, so it may
    /// finish an already jointly verified domain earlier even at this default.
    ///
    /// Matches alpha-beta-CROWN `solver.alpha-crown.input_split_alpha_iteration`
    /// (reference default 5, `input_split/bounding.py:90-179`).
    #[serde(default = "default_input_split_alpha_iteration")]
    pub input_split_alpha_iteration: usize,
    /// Learning rate for per-sub-domain α refinement (see
    /// `input_split_alpha_iteration`). Only used when that knob is > 0.
    ///
    /// Matches alpha-beta-CROWN `solver.alpha-crown.input_split_lr_alpha`
    /// (default 0.05).
    #[serde(default = "default_input_split_lr_alpha")]
    pub input_split_lr_alpha: f32,
    /// Bonus score for bounds that touch zero during `sb_sum` input split scoring.
    /// Matches alpha-beta-CROWN `touch_zero_score` (default: 0.0 = disabled).
    #[serde(default = "default_input_split_touch_zero_score")]
    pub input_split_touch_zero_score: f32,
    /// Margin weight for SB scoring: scales the per-spec verification margin.
    ///
    /// Uses lower bounds in lower-bound mode and upper bounds in upper-bound
    /// mode so the heuristic stays aligned with the active verification target.
    /// When positive, dimensions that help nearly verified specs score higher.
    /// Matches alpha-beta-CROWN `sb_margin_weight` (default: 1.0).
    ///
    /// Reference: alpha-beta-CROWN `branching_heuristics.py:84-89`
    #[serde(default = "default_input_split_sb_margin_weight")]
    pub input_split_sb_margin_weight: f32,
    /// Sum across specs instead of taking max for SB scoring.
    ///
    /// When true, dimension scores are summed across all specification rows.
    /// When false (default), the max across specs is used.
    /// Matches alpha-beta-CROWN `sb_sum` (default: false).
    ///
    /// Reference: alpha-beta-CROWN `branching_heuristics.py:79-81`
    #[serde(default)]
    pub input_split_sb_sum: bool,
    /// Primary spec index for SB scoring (use single spec instead of max/sum).
    ///
    /// When set, only the specified spec row is used for scoring.
    /// Matches alpha-beta-CROWN `sb_primary_spec` (default: None = all specs).
    ///
    /// Reference: alpha-beta-CROWN `branching_heuristics.py:91-93`
    #[serde(default)]
    pub input_split_sb_primary_spec: Option<usize>,
    /// Number of candidate neurons to evaluate for FSB/kFSB branching.
    ///
    /// Higher values can reduce domain count but increase per-domain overhead.
    /// Default: 8 (ny default).
    #[serde(default = "default_fsb_candidates")]
    pub fsb_candidates: usize,
    /// Reduce operation for kFSB branching (how to combine branch scores).
    /// - Min: conservative (worst-case) - default
    /// - Max: optimistic (best-case)
    /// - Mean: balanced (average)
    #[serde(default)]
    pub kfsb_reduce_op: KfsbReduceOp,
    /// Typed candidate override scoped to the wave-batched multi-objective
    /// kFSB selector. `None` inherits `fsb_candidates` for shipped-preset
    /// compatibility; unlike the old delivery route, setting it cannot retune
    /// ordinary FSB/kFSB branching.
    #[serde(default)]
    pub kfsb_multi_candidates: Option<usize>,
    /// Typed reduce policy scoped to the same wave selector. `None` inherits
    /// `kfsb_reduce_op`.
    #[serde(default)]
    pub kfsb_multi_reduce_op: Option<KfsbReduceOp>,
    /// Bounded, branch-specific depth-2 lookahead advice for the
    /// multi-objective graph kFSB lane.
    ///
    /// This policy is default-off. When armed, it only ranks which already
    /// constructed first-level split to commit; private child-bound maps and
    /// second-level scores never become proof authority. Phase 1 is
    /// lower-bound-only and fails closed during upper-bound verification so
    /// the historical raw-lower objective selection remains unchanged. An
    /// admitted typed budget has priority over legacy M27/f64 observers for
    /// that wave; their advice-only one-shots remain available for a later
    /// wave instead of suppressing typed selection or sharing its deadline.
    #[serde(default)]
    pub depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig,
    /// Learning rate for β parameter optimization.
    pub beta_lr: f32,
    /// Number of optimization iterations per domain (applies to both α and β).
    pub beta_iterations: usize,
    /// Minimum improvement to continue optimization.
    pub beta_tolerance: f32,
    /// Break after this many consecutive iterations without a better lower bound.
    ///
    /// Matches alpha-beta-CROWN `early_stop_patience=10`
    /// (`optimized_bounds.py:75-77`). Set to `usize::MAX` to disable the
    /// patience guard while keeping the gradient-convergence exit active.
    #[serde(default = "default_early_stop_patience")]
    pub early_stop_patience: usize,
    /// Number of β optimization iterations to run on the root domain before BaB.
    /// This amortizes the optimization cost across all descendant domains via warmup.
    /// Set to 0 to disable root-level optimization (default: 20).
    #[serde(default = "default_root_beta_iterations")]
    pub root_beta_iterations: usize,
    /// Maximum depth at which to run per-domain β optimization.
    /// Domains deeper than this use inherited β values without further optimization.
    /// Set to 0 to disable per-domain optimization entirely (rely only on root + warmup).
    /// Default: 3 (optimize root and first 3 levels of splits).
    #[serde(default = "default_beta_max_depth")]
    pub beta_max_depth: usize,
    /// Use analytical β gradients for DAG networks instead of SPSA.
    /// Analytical gradients are computed from the A matrices stored during
    /// CROWN backward propagation, which is ~3x faster than SPSA (1 pass vs 3 passes per iteration).
    /// Default: true (use analytical gradients when available).
    #[serde(default = "default_use_analytical_beta_gradients")]
    pub use_analytical_beta_gradients: bool,
    /// Learning rate for α parameter optimization (when use_alpha_crown is true).
    pub alpha_lr: f32,
    /// Use momentum for α updates (inherits from alpha_config.momentum if true).
    pub alpha_momentum: bool,
    /// Number of domains to process in parallel (batch size).
    /// Set to 1 for sequential processing, or higher for parallel.
    pub batch_size: usize,
    /// Maximum number of spec rows to handle in one root build batch.
    ///
    /// Mirrors alpha-beta-CROWN `solver.build_batch_size` and is primarily used
    /// by large graph input-split warmup on multi-spec properties such as
    /// nn4sys `mscn_2048*`, where a single dense spec-matrix build can consume
    /// the full timeout budget before BaB begins.
    ///
    /// `None` keeps the current all-specs-at-once behavior.
    #[serde(default)]
    pub build_batch_size: Option<usize>,
    /// Use parallel child domain creation (both branches computed in parallel).
    pub parallel_children: bool,
    /// Use adaptive learning rates (Adam-style optimizer).
    /// When enabled, learning rates are automatically adjusted per-parameter
    /// based on gradient history, improving convergence on diverse problems.
    pub use_adaptive: bool,
    /// Configuration for adaptive optimizer (when use_adaptive is true).
    pub adaptive_config: AdaptiveOptConfig,
    // --- GCP-CROWN: Cutting Plane Configuration ---
    /// Enable GCP-CROWN cutting planes.
    /// When enabled, verified subdomains generate cuts that tighten bounds
    /// for remaining domains.
    #[serde(default)]
    pub enable_cuts: bool,
    /// Convolution backward mode for CROWN propagation.
    /// Auto (default): Patches when cuts disabled, Matrix when cuts enabled.
    /// Reference: alpha-beta-CROWN `general.conv_mode` (`abcrown.py:228-231`).
    #[serde(default)]
    pub conv_mode: ConvMode,
    /// Maximum number of cutting planes to retain.
    /// More cuts = tighter bounds but more computation per domain.
    #[serde(default = "default_max_cuts")]
    pub max_cuts: usize,
    /// Minimum depth of verified domain to generate a cut.
    /// Deeper domains produce more specific cuts.
    #[serde(default = "default_min_cut_depth")]
    pub min_cut_depth: usize,
    /// Enable near-miss cut generation.
    /// When enabled, cuts are also generated from domains where the lower bound
    /// is within `near_miss_margin` of the threshold, even if not verified.
    /// This can help prune similar regions that are "almost verified".
    #[serde(default)]
    pub enable_near_miss_cuts: bool,
    /// Margin for near-miss cut generation (as fraction of threshold).
    /// Only used when `enable_near_miss_cuts` is true.
    /// Default: 0.1 (10% of threshold, or absolute 0.1 if threshold is 0)
    #[serde(default = "default_near_miss_margin")]
    pub near_miss_margin: f32,
    /// Enable proactive cut generation (BICCOS-lite).
    /// When enabled, cuts are generated for unstable ReLUs BEFORE BaB starts,
    /// rather than waiting for domains to verify. This helps on hard instances
    /// where no domains verify initially (chicken-and-egg problem).
    ///
    /// The proactive cuts encode pairwise neuron implications based on initial bounds.
    #[serde(default)]
    pub enable_proactive_cuts: bool,
    /// Maximum number of proactive cuts to generate.
    /// More cuts = potentially tighter bounds but more computation.
    #[serde(default = "default_max_proactive_cuts")]
    pub max_proactive_cuts: usize,
    /// Enable BICCOS constraint strengthening for verified-domain cuts.
    /// When enabled, cuts are strengthened by dropping low-influence constraints
    /// (based on neuron influence scores) and re-verifying the reduced domain.
    #[serde(default)]
    pub enable_biccos_constraint_strengthening: bool,
    /// Drop ratio for BICCOS constraint strengthening.
    /// Uses the quantile of influence scores; higher values drop more constraints.
    #[serde(default = "default_biccos_drop_ratio")]
    pub biccos_drop_ratio: f32,
    /// Optimization interval for cut lambda parameters (domains between updates).
    /// Lambda optimization runs every N domains explored during BaB.
    /// Lower values update more frequently (better convergence, more overhead).
    #[serde(default = "default_lambda_opt_interval")]
    pub lambda_opt_interval: usize,
    /// Learning rate for cut lambda Adam optimization.
    /// Overrides `adaptive_config.lr_lambda` specifically for the BaB lambda path.
    /// Default: 0.05 (matches alpha-beta-CROWN lambda LR).
    #[serde(default = "default_lambda_lr")]
    pub lambda_lr: f32,
    /// Cut eviction policy for bounded cut pools.
    #[serde(default)]
    pub cut_eviction_policy: CutEvictionPolicy,
    /// Iteration threshold for stale cuts.
    #[serde(default = "default_cut_stale_iters")]
    pub cut_stale_iters: usize,
    /// Iteration threshold for hard-stale cuts (aggressive eviction).
    #[serde(default = "default_cut_hard_stale_iters")]
    pub cut_hard_stale_iters: usize,
    /// Lambda threshold below which stale cuts are evicted.
    #[serde(default = "default_cut_lambda_min")]
    pub cut_lambda_min: f32,
    /// Maximum fraction of proactive cuts allowed in the pool.
    #[serde(default = "default_cut_proactive_fraction")]
    pub cut_proactive_fraction: f32,
    /// Scoring weights for utility-weighted eviction.
    #[serde(default)]
    pub cut_score_weights: CutScoreWeights,
    // --- BICCOS Cold-Start Gating ---
    /// Enable cold-start gating for BICCOS cut generation.
    /// When enabled, cuts are withheld until early BaB statistics indicate
    /// verified domains or bound gains are sufficient to make cuts effective.
    #[serde(default)]
    pub enable_biccos_cold_start: bool,
    /// Minimum verified domains before enabling cuts.
    #[serde(default = "default_biccos_min_verified")]
    pub biccos_min_verified: usize,
    /// Minimum verified domain rate (per iteration) before enabling cuts.
    #[serde(default = "default_biccos_min_verified_rate")]
    pub biccos_min_verified_rate: f32,
    /// Sliding window size for verified-rate computation.
    #[serde(default = "default_biccos_verified_rate_window")]
    pub biccos_verified_rate_window: usize,
    /// Minimum cuts generated (e.g., via MTS) before enabling cuts.
    #[serde(default = "default_biccos_min_cuts")]
    pub biccos_min_cuts: usize,
    /// Minimum average bound gain per split before enabling cuts.
    #[serde(default = "default_biccos_min_bound_gain")]
    pub biccos_min_bound_gain: f32,
    /// Sliding window size for bound-gain computation.
    #[serde(default = "default_biccos_bound_gain_window")]
    pub biccos_bound_gain_window: usize,
    /// Maximum number of cold-start iterations before declaring exhaustion.
    #[serde(default = "default_biccos_cold_max_iters")]
    pub biccos_cold_max_iters: usize,
    /// Maximum number of iterations to keep cut generation enabled.
    #[serde(default = "default_biccos_cut_window")]
    pub biccos_cut_window: usize,
    /// Minimum cut yield before disabling new cut generation.
    #[serde(default = "default_biccos_min_cut_yield")]
    pub biccos_min_cut_yield: f32,
    /// Sliding window size for cut-yield computation.
    #[serde(default = "default_biccos_cut_yield_window")]
    pub biccos_cut_yield_window: usize,
    /// Number of low-yield windows before disabling cut generation.
    #[serde(default = "default_biccos_cut_yield_patience")]
    pub biccos_cut_yield_patience: usize,
    // --- Property Direction ---
    /// Verify upper bound instead of lower bound.
    ///
    /// When `false` (default): verifies output > threshold (lower_bound > threshold)
    /// When `true`: verifies output < threshold (upper_bound < threshold)
    ///
    /// Use `true` for VNNLIB constraints like `Y >= c` (unsafe region), where
    /// proving safety requires proving the upper bound is below the threshold.
    #[serde(default)]
    pub verify_upper_bound: bool,
    // --- PGD Attack for Counterexample Finding ---
    /// Enable PGD attack to find concrete counterexamples.
    /// When enabled and verification is inconclusive, PGD attack is run to
    /// try to find a concrete input that violates the property.
    #[serde(default)]
    pub enable_pgd_attack: bool,
    /// Number of random restarts for PGD attack.
    #[serde(default = "default_pgd_restarts")]
    pub pgd_restarts: usize,
    /// Number of gradient steps per restart.
    #[serde(default = "default_pgd_steps")]
    pub pgd_steps: usize,
    /// Restart PGD when a projected step leaves the point unchanged.
    ///
    /// Reference: alpha-beta-CROWN `pgd_restart_when_stuck` in
    /// `abcrown_all_params.yaml:279` and `acasxu.yaml:20`.
    #[serde(default)]
    pub pgd_restart_when_stuck: bool,
    /// PGD initialization strategy.
    ///
    /// `Uniform` (default): standard random sampling.
    /// `Osi`: Output Specification Initialization for diverse restarts.
    /// Reference: alpha-beta-CROWN `attack_mode: diversed_PGD` →
    /// `initialization = "osi"` in `attack_interface.py:29-35`.
    #[serde(default)]
    pub pgd_initialization: PgdInitialization,
    /// Number of OSI initialization steps. Default: 20.
    #[serde(default = "default_pgd_osi_steps")]
    pub pgd_osi_steps: usize,
    /// Enable the GAMA guidance loss for PGD attack steps.
    ///
    /// Set by `attack_mode: diversed_GAMA_PGD` (together with OSI
    /// initialization). Relational-target attack steps then ascend
    /// `softmax_margin + λ·‖P − softmax‖²` with λ annealed linearly from
    /// [`GAMA_LAMBDA_DEFAULT`] to 0 (Sriramanan et al., NeurIPS 2020) —
    /// designed for adversarially-trained networks whose raw-margin gradients
    /// are masked. Attack-only: candidates are re-validated before any `sat`,
    /// so this cannot affect soundness.
    /// Reference: alpha-beta-CROWN `attack_mode: diversed_GAMA_PGD` →
    /// `GAMA_loss=True` in `attack_interface.py:29-35`.
    #[serde(default)]
    pub pgd_gama: bool,
    /// PGD optimizer strategy.
    #[serde(default)]
    pub pgd_optimizer: PgdOptimizer,
    /// PGD alpha/step-size policy.
    #[serde(default)]
    pub pgd_alpha_mode: PgdAlphaMode,
    /// Per-step exponential decay for the PGD/Adam learning rate.
    ///
    /// Maps to `AdamClippingParams::lr_decay`. Lower values shrink the step
    /// size faster across PGD steps. Pure attack tuning: PGD only searches for
    /// counterexamples that are re-validated before being emitted as `sat`,
    /// so this cannot affect soundness.
    ///
    /// Reference: alpha-beta-CROWN `attack.pgd_lr_decay` /
    /// `attack_pgd.py:255` (`ExponentialLR(opt, lr_decay)`).
    #[serde(default = "default_pgd_lr_decay")]
    pub pgd_lr_decay: f32,
    /// Straight-through-estimator surrogate gradient for Sign layers during
    /// ATTACK gradient estimation (#surrogate-sign, preset key
    /// `attack.surrogate_sign_gradient`). Maps to
    /// `PgdConfig::surrogate_sign_gradient`. Attack-only: candidates are
    /// re-validated before any `sat`, so this cannot affect soundness.
    #[serde(default)]
    pub pgd_surrogate_sign_gradient: bool,
    /// Dense deterministic grid sweep over low-effective-dimension input
    /// boxes as a pre-PGD attack phase (#dense-sweep, preset key
    /// `attack.dense_low_dim_sweep`). Maps to
    /// `PgdConfig::dense_low_dim_sweep`. Attack-only.
    #[serde(default)]
    pub pgd_dense_low_dim_sweep: bool,
    /// Maximum number of nonzero-width input dims for the dense sweep to run
    /// (#dense-sweep). Default: 3.
    #[serde(default = "default_pgd_dense_sweep_max_dims")]
    pub pgd_dense_sweep_max_dims: usize,
    /// Total forward-evaluation budget for the dense sweep (#dense-sweep).
    /// Default: 32768 (a 128×128 initial grid plus refinement for 2 dims).
    #[serde(default = "default_pgd_dense_sweep_points")]
    pub pgd_dense_sweep_points: usize,
    /// Number of PGD restarts seeded at box corners (#attack-vertex-geometry,
    /// preset key `attack.corner_seed_restarts`): restart 0 at the lower
    /// corner, restart 1 at the upper corner, restarts `2..N` at random box
    /// vertices. Maps to `PgdConfig::corner_seed_restarts`. Default 0 keeps
    /// the historical seeding byte-for-byte. Attack-only: candidates are
    /// re-validated before any `sat`, so this cannot affect soundness.
    #[serde(default)]
    pub pgd_corner_seed_restarts: usize,
    /// Root-pruned, per-disjunct, many-restart attack stage
    /// (#attack-open-disjuncts, preset keys `attack.open_disjunct_*`). Read
    /// by the multi-clause disjunction route after its disjunctive attack
    /// phase. Default `restrict: false`: the stage does not exist and the
    /// route is byte-identical. Attack-only: every candidate passes the
    /// unchanged confirmation funnel and the trusted-ORT gate, so this cannot
    /// affect soundness.
    #[serde(default)]
    pub open_disjunct_attack: OpenDisjunctAttackConfig,
    /// Per-run exact-VJP graph attack policy.
    ///
    /// Runtime-only: scored VNN-COMP supplies it from the frozen entry-time
    /// LeverSet, while interactive preset fields are applied through the
    /// frontend. Keeping it out of generic config serialization avoids adding
    /// a second, ungoverned delivery surface.
    #[serde(skip)]
    pub pgd_exact_vjp: ExactVjpPgdConfig,
    // --- Relaxed Clipping (Clip-and-Verify) ---
    /// Enable relaxed clipping to tighten input bounds using CROWN constraints.
    ///
    /// When enabled, after each input split, the child domain's input bounds are
    /// tightened using closed-form 1D updates based on the CROWN linear constraints.
    /// This can significantly reduce the number of domains needed for verification.
    ///
    /// Requires `branching_heuristic` to be `InputSplit` to have effect.
    ///
    /// Reference: Wei et al., "Clip and Verify" (arXiv:2512.11087)
    #[serde(default)]
    pub enable_relaxed_clip: bool,
    /// Number of relaxed clipping iterations per split.
    ///
    /// More iterations can produce tighter bounds but with diminishing returns.
    /// Typical values: 1-3 iterations.
    /// Default: 1 (matches baseline)
    #[serde(default = "default_relaxed_clip_iterations")]
    pub relaxed_clip_iterations: usize,
    /// Input-domain clipping algorithm type.
    ///
    /// - `Relaxed` (default): Closed-form 1D updates, fast but axis-aligned.
    /// - `Complete`: LP-optimal via Lagrangian dual coordinate ascent, tighter
    ///   bounds by accounting for cross-constraint dependencies.
    ///
    /// Only has effect when `enable_relaxed_clip` is true.
    ///
    /// Reference: alpha-beta-CROWN `clip_input_domain.clip_type`
    #[serde(default)]
    pub input_clip_type: InputClipType,
    /// Default-off exact-domain Clip-and-Verify for batch-stack-unsafe
    /// multi-clause input splitting.
    ///
    /// Unlike the legacy post-split clip, this route computes one
    /// non-domain-stacked full-spec CROWN result on the exact popped domain,
    /// clips that same domain before choosing its split, drops the fresh planes,
    /// and queues children without any parent planes. It is deliberately
    /// restricted by [`Self::validate`] to the
    /// reordered + IBP-enhanced relaxed-clip route; the graph dispatcher further
    /// requires a batch-stack-unsafe graph. No shipped preset enables it.
    #[serde(default)]
    pub input_split_fresh_domain_clip: bool,
    /// Neuron selection ratio for complete clipping.
    ///
    /// **UNIMPLEMENTED**: Config is parsed and stored for preset compatibility
    /// but no engine code reads this field yet.
    ///
    /// When `input_clip_type == Complete`, intended to control what fraction of
    /// unstable neurons receive constrained concretization (LP-optimal bounds).
    /// - `-1.0` (default): Disabled, apply to all neurons.
    /// - `[0.0, 1.0]`: Apply to that fraction of unstable neurons per layer,
    ///   selected by largest uncertainty (upper - lower gap).
    ///
    /// Reference: alpha-beta-CROWN `clip_neuron_selection_value`
    #[serde(default = "default_clip_neuron_selection_ratio")]
    pub clip_neuron_selection_ratio: f32,
    // --- Intermediate Domain Clipping (clip_interm_domain) ---
    /// Enable intermediate domain clipping to tighten intermediate neuron bounds.
    ///
    /// When enabled, uses split-derived linear constraints in input space to tighten
    /// intermediate (pre-activation) bounds via constrained concretization. This can
    /// significantly reduce the number of unstable neurons and improve verification.
    ///
    /// Requires ReLU branching (not input split) and split history with constraints.
    ///
    /// Reference: alpha-beta-CROWN `clip_interm_domain` feature
    /// (`complete_verifier/domain_clipper.py`)
    #[serde(default)]
    pub enable_clip_interm_domain: bool,
    /// Number of objective neurons per layer to tighten with clip_interm_domain.
    ///
    /// Higher values tighten more neurons but increase computation.
    /// Default: 3. Scored presets may opt into alpha-beta-CROWN's larger
    /// `clip_topk_objective` value explicitly.
    #[serde(default = "default_clip_interm_topk")]
    pub clip_interm_topk: usize,
    /// Apply clip_interm_domain during α-CROWN optimization passes.
    ///
    /// Baseline: clip_n_verify.clip_interm_domain.clip_in_alpha_crown.
    /// Default: false (maintains existing behavior unless explicitly enabled).
    #[serde(default)]
    pub clip_in_alpha_crown: bool,
    /// Prune infeasible domains detected during activation-space clipping.
    ///
    /// Baseline: clip_n_verify.prune (domain_clipper).
    /// Default: false. The certificate-backed batched production path
    /// currently quarantines pruning and rejects `true` during validation;
    /// Complete Clip may tighten bounds but cannot mint infeasibility yet.
    #[serde(default)]
    pub clip_interm_prune: bool,
    /// Use final-layer constraints when pruning clipped domains.
    ///
    /// Baseline: clip_n_verify.final_layer (domain_clipper).
    /// Default: false.
    #[serde(default)]
    pub clip_interm_use_final_layer: bool,
    // --- Batched Intermediate Transfer (interm_transfer) ---
    /// Enable static intermediate bound transfer in batched domains.
    ///
    /// When enabled, batched domain storage captures static intermediate bounds
    /// and unstable masks for reuse across domain batches.
    #[serde(default)]
    pub enable_interm_transfer: bool,
    // --- Root intermediate CROWN CUDA factory fallback ---
    /// Allow root intermediate CROWN passes to obtain a sound CUDA engine from
    /// the process factory when the caller did not provide a usable local
    /// engine.
    ///
    /// Default false keeps existing configurations unchanged. The exact
    /// `NY_ROOT_INTERM_CUDA_FACTORY` environment override remains available:
    /// absent inherits this typed value, `1` forces on, and every other present
    /// value forces off.
    #[serde(default)]
    pub root_interm_cuda_factory: bool,
    // --- Post-root multi-objective CUDA factory engine handoff ---
    /// Allow multi-objective graph BaB to reuse an already-materialized sound
    /// CUDA engine after root evaluation when the caller supplied no engine.
    ///
    /// The handoff never initializes or waits for the process factory. It also
    /// requires the selected engine to advertise deadline-safe support for the
    /// complete post-root execution surface (including generic GEMM), plus
    /// verdict-grade GPU CROWN soundness and cooperative cancellation. Default
    /// false preserves the historical post-root sequential route. The exact
    /// `NY_MO_CUDA_FACTORY_ENGINE_HANDOFF` override is resolved after root:
    /// absent inherits this value, literal `1` enables, and every other present
    /// value disables.
    #[serde(default)]
    pub mo_cuda_factory_engine_handoff: bool,
    // --- Post-root bounded CUDA β shared-executor activation ---
    /// Allow multi-objective graph BaB to enter its shared executor through a
    /// local deadline-aware CPU GEMM facade when no caller/post-root engine is
    /// available and an already-materialized sound CUDA backend exposes only
    /// the audited call-local bounded β-CROWN surface.
    ///
    /// CUDA is never handed to generic shared-executor work. Default false
    /// preserves the historical sequential route. The exact
    /// `NY_MO_CUDA_BOUNDED_SHARED_EXECUTOR` override is resolved after root:
    /// absent inherits this value, literal `1` enables, and every other present
    /// value disables.
    #[serde(default)]
    pub mo_cuda_bounded_shared_executor: bool,
    // --- Multi-objective containment fence (#vit-bank-gate) ---
    /// Arm the downgrade-only containment fence on the multi-objective graph
    /// BaB (`beta_crown::engine::graph::multi_objective::containment_fence_lane`).
    ///
    /// When true, the root's node and objective bounds are compared against a
    /// sound f64 point walk at deterministic samples of the root box, and every
    /// child a shared-executor wave returns is compared, on its certified
    /// objective LOWER bounds, at every sample proved inside it. Any
    /// separation returns `Unknown` instead of letting the run publish; the
    /// fence never tightens, closes, or proves anything. It refuses to arm
    /// (and prints why) under cuts, conflict clauses, interm-domain or relaxed
    /// clipping, and on conjunctive runs, whose bounds are property-conditioned.
    ///
    /// Default false: every shipped preset is byte-identical. The vit_2023
    /// bank gate is the intended first consumer (preset key
    /// `bab.mo_containment_fence`).
    ///
    /// #vit-margin-alpha-complex (P8): when armed the fence also (a) checks a
    /// ROOT-terminal `Verified` on the certified objective LOWER bounds it
    /// rests on — not only the output enclosure — and FAILS CLOSED (returns
    /// `Unknown`) when that authority could not be checked at all, and (b)
    /// refuses a `Verified` whose BaB rested on any emptiness claim (an
    /// infeasible child pair, sibling or NoUnstable domain, a construction-time
    /// empty half-space), since an emptiness claim carries no bound to fence.
    #[serde(default)]
    pub mo_containment_fence: bool,
    /// #vit-margin-alpha-complex (P8): adjudicate a ROOT-terminal
    /// multi-objective `Verified` against an INDEPENDENT forward of the
    /// original model (ONNX Runtime; ny-cli attaches the oracle only when this
    /// key is set): at the fence's deterministic samples, every certified
    /// objective lower bound must sit at or below the ORT objective value.
    /// A separation, or an armed adjudication that could evaluate no sample,
    /// returns `Unknown`. Inert unless `mo_containment_fence` is also armed.
    /// Downgrade-only. Default false: byte-identical (preset key
    /// `bab.mo_root_ort_adjudication`).
    #[serde(default)]
    pub mo_root_ort_adjudication: bool,
    /// #vit-v3-bankable: the multi-objective COVER CERTIFICATION system on the
    /// disjunctive graph multi-objective BaB (the lane vit_2023 takes; preset
    /// key `bab.mo_cover_certify`, governed by `NY_MO_COVER_CERTIFY`):
    ///
    /// * `off` (default, every shipped yaml): byte-identical to the historical
    ///   lane — no frame, no replay, no line; its `Verified` publishes through
    ///   the counted `BetaCrownVerdict` legacy door;
    /// * `ledger`: a per-ROW path completeness ledger (`certify::graph`)
    ///   records every close with the region it closes (bound closes per
    ///   row, emptiness closes only with a certified slack). A `Verified` whose
    ///   rows' leaves do not cover the root is WITHDRAWN (`Unknown`); a proved
    ///   frame deposits one cover per row, so the instance door can mint an
    ///   `UnsatToken`; an unbuildable frame keeps the legacy door;
    /// * `replay`: `ledger` plus a CERTIFY-ONCE replay of every witness leaf
    ///   below the root with no reused lane state (no lA warm start, no
    ///   inherited node bounds, no parent flooring, no device batch; the
    ///   lane's final β only as a hint). `Verified` stands only when the cover
    ///   is proved AND every witness re-certifies; anything else is withdrawn.
    ///
    /// Downgrade-only in both armed modes: the system never tightens a bound,
    /// closes a domain, or creates a verdict.
    #[serde(default)]
    pub mo_cover_certify: MoCoverCertify,
    // --- Root dense-head intermediate CROWN (#cifar-head-crown) ---
    /// Run one bounded heuristic-slope CROWN backward for each dense-fed ReLU
    /// pre-activation at the graph root and shrink-intersect the certified box
    /// into the frozen intermediate bounds inherited by BaB.
    ///
    /// This is intentionally scoped structurally (Linear/Gemm -> ReLU), rather
    /// than by exporter-specific node names. Default false keeps every benchmark
    /// unchanged unless a measured preset opts in. `NY_ROOT_CROWN_INTERM=1/0`
    /// remains an explicit force-on/force-off override for A/B and rollback.
    #[serde(default)]
    pub root_crown_interm_dense_head: bool,
    /// Wall-clock cap in seconds for the one-time root dense-head CROWN pass.
    /// Zero skips the pass. A global verifier deadline always takes precedence.
    #[serde(default = "default_root_crown_interm_max_secs")]
    pub root_crown_interm_max_secs: u64,
    /// Maximum number of elements in a selected dense-head pre-activation.
    /// Zero selects no targets. `NY_ROOT_CROWN_INTERM_MAXDIM` overrides for A/B.
    #[serde(default = "default_root_crown_interm_max_dim")]
    pub root_crown_interm_max_dim: usize,
    // --- Root sparse crossing-row intermediate CROWN (#root-sparse-interm-crown) ---
    /// Tighten structurally eligible convolutional ReLU pre-activations at the
    /// graph root by seeding only their widest crossing rows into the certified
    /// sound GPU CROWN fold, then shrink-intersecting the resulting bounds.
    ///
    /// Dense/Gemm heads are deliberately excluded because
    /// `root_crown_interm_dense_head` handles them with a full identity seed.
    /// Default false keeps unmeasured presets byte-identical. The
    /// `NY_ROOT_SPARSE_INTERM_CROWN=1/0` override is retained for sealed A/B and
    /// emergency rollback.
    #[serde(default)]
    pub root_sparse_interm_crown: bool,
    /// One-time wall-clock cap. Zero skips the pass; the verifier deadline wins.
    #[serde(default = "default_root_sparse_interm_crown_max_secs")]
    pub root_sparse_interm_crown_max_secs: u64,
    /// Maximum flattened pre-activation width admitted structurally.
    #[serde(default = "default_root_sparse_interm_crown_max_dim")]
    pub root_sparse_interm_crown_max_dim: usize,
    /// Maximum widest crossing rows seeded per target.
    #[serde(default = "default_root_sparse_interm_crown_max_rows")]
    pub root_sparse_interm_crown_max_rows: usize,
    /// Maximum eligible targets, processed deepest-first.
    #[serde(default = "default_root_sparse_interm_crown_max_targets")]
    pub root_sparse_interm_crown_max_targets: usize,
    // --- Row-chunked comprehensive sound-GPU root intermediate sweep ---
    /// Run the comprehensive all-target sound-GPU root intermediate sweep, which
    /// injects identity rows at every eligible ReLU pre-activation depth in ONE
    /// atomic backend transaction and shrink-intersects the result.
    ///
    /// DELIVERY (this key exists because of `measured_gate_delivery.rs`): the
    /// scored entry point exports no `NY_*` variables, so an env-only gate
    /// is dead in competition no matter what it measured. The measurement below
    /// was taken through `NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN=1`; this typed key
    /// is how it actually reaches a scored run. The env lever is retained as a
    /// force-on/force-off override for A/B and rollback.
    ///
    /// MEASURED (cifar100 resnet_medium, official 100 s budget, scored config):
    /// with `root_comprehensive_gpu_interm_chunks` > 1 the root objective census
    /// goes from 0/99 to 92/99 on the best row and 8 of 14 sampled timeout rows
    /// gain, worst-objective margin -18.42 -> -1.09. No row converted yet.
    /// Default false keeps unmeasured presets byte-identical.
    #[serde(default)]
    pub root_comprehensive_gpu_interm: bool,
    /// `#bab-floor`: BaB's guaranteed share of the multi-objective root window,
    /// subtracted BEFORE any root phase sizes itself.
    ///
    /// DELIVERY: the scored entry point exports no `NY_*`, so the env
    /// lever these three were measured through is dead in competition. These
    /// typed keys are how a search result reaches a scored run.
    ///
    /// `None` (the default) means no arbitration exists and every root phase
    /// keeps the deadline it has today, so an unmeasured preset stays
    /// byte-identical. `NY_BAB_RESERVE_FRAC` overrides in BOTH directions.
    #[serde(default)]
    pub root_bab_reserve_frac: Option<f64>,
    /// `#bab-floor`: the root objective pass's share, subtracted after
    /// `root_bab_reserve_frac`. Read only when that reservation is armed —
    /// without a share of its own the pass starves and the BaB reserve behind
    /// it is unreachable. `NY_ROOT_SPEC_FRAC` overrides.
    #[serde(default)]
    pub root_spec_frac: Option<f64>,
    /// `#bab-floor`: the bootstrap ascent's share, min-composed onto
    /// `root_alpha_cap_secs`. That cap is a FIXED 40 s — 51% of the BaB slice
    /// at 100 s and 4% at 1200 s — so it is the one claimant whose cost does
    /// not scale with the window. `NY_ROOT_ALPHA_FRAC` overrides.
    #[serde(default)]
    pub root_alpha_frac: Option<f64>,
    /// How many DISJOINT row windows the comprehensive sweep may accumulate.
    ///
    /// The sweep is hard-capped in rows-per-target by device memory (the backend
    /// declines beyond its class budget), which is far below the coverage the root
    /// needs. Running the same bounded sweep repeatedly over disjoint windows
    /// keeps peak device memory at one chunk while coverage grows with the time
    /// available. Sound by construction: every sweep is atomic over its own
    /// window, every commit is a shrink-only intersect, and all windows are cut
    /// from one frozen transcript so they stay disjoint even as earlier chunks
    /// tighten the live bounds. `1` is byte-identical to the historical single
    /// sweep. `NY_INTERM_ROW_CHUNKS` overrides for A/B.
    #[serde(default = "default_root_comprehensive_gpu_interm_chunks")]
    pub root_comprehensive_gpu_interm_chunks: usize,
    // --- Post-C-matrix survivor Patch-CROWN (#post-c-survivor) ---
    /// After the cheap sound multi-objective root C-matrix candidate, rerun at
    /// most 16 still-unverified rows through one resource-bounded generic full
    /// DAG Patch-CROWN backward and sound-intersect the result.
    ///
    /// This is an experimental, default-OFF root-only policy. The CLI resolves
    /// the exact `NY_ROOT_POST_C_SURVIVOR=1` opt-in once into this typed field;
    /// the solver never reads or mutates process-global environment state.
    #[serde(default)]
    pub root_post_c_survivor: bool,
    // --- β/α-ascent graft for the multi-objective dense-spec lane (#mo-beta-graft) ---
    /// Run the wide GPU segment-lane β/α ascent to OPTIMIZE the split
    /// multipliers, then EVALUATE through the tight dense-spec batched
    /// primitive with those multipliers folded in, taking the elementwise
    /// tightest of {dense-spec bound, ascended wide bound}.
    ///
    /// Motivation (metaroom 6cnn, measured): the wide segment lane's base
    /// relaxation is 2.4-2.9x LOOSER than the dense-spec primitive, so
    /// routing conv chains onto it wholesale (NY_BAB_CHAIN_WIDE=1) REPLACES a
    /// tight bound with a loose-ascended one; but the ascent's optimized β
    /// genuinely tightens the dense bound when folded back in.
    ///
    /// SOUND: both inputs are valid bounds on the same spec rows over the
    /// same subdomain (the ascended bound folds the SAME β entries — built by
    /// `with_constraint` from this domain's split history, values only moved
    /// under a β>=0 clamp — that the dense backward's `apply_beta_contribution`
    /// folds), so the per-row intersection encloses the true range.
    ///
    /// Default: false. Env override: `NY_MO_BETA_GRAFT=1` forces on,
    /// `NY_MO_BETA_GRAFT=0` forces off (A/B).
    #[serde(default)]
    pub mo_beta_graft: bool,
    /// Admit pure Linear/ReLU chains (FC networks) to the wide GPU stacker
    /// ROUTING (#chain-wide, S2 of the GPU/batched child-eval design): lifts
    /// the conv-only gates — the multi-objective GPU single-pass admission
    /// (`has_conv_layers`) and the wide entry's `EntryGraphNotConv` decline —
    /// so the #mo-beta-graft composition can reach the wide seam on FC graphs.
    ///
    /// SOUNDNESS SHAPE: the wide-chain BASE relaxation is measured 2.4-2.9x
    /// LOOSER than the dense node-by-node backward, so on FC graphs the wide
    /// bound is only ever CAPTURED by a graft call and composed
    /// elementwise-tightest with the dense bound — never returned as a bound
    /// replacement (the `EntryGraphNotConv` lift requires the per-call
    /// `graft_pure_chain` marker). Arm together with `mo_beta_graft`
    /// (`bab.beta_graft`); armed alone this key changes no FC bound.
    ///
    /// Default: false (routing byte-identical). Preset key: `bab.chain_wide`.
    /// The historical dark env lever `NY_BAB_CHAIN_WIDE=1` additionally arms
    /// the extractor's pure-chain admission (conv-chain replacement mode).
    #[serde(default)]
    pub bab_chain_wide: bool,
    /// Arm the multi-objective wave-batched kFSB selector by default (env
    /// NY_MO_KFSB overrides either way). Set true only for benchmarks measured
    /// Pareto (CIFAR-100).
    ///
    /// A/B evidence (cifar100, live run): 9/9 instances strictly better bound
    /// trajectory, 0 verdict regressions, wall never extended. Scoped to the
    /// CIFAR-100 presets only so every other lane (relational, ACAS Xu,
    /// relusplitter, …) stays byte-identical. Kill switch: `NY_MO_KFSB=0`
    /// force-off, `NY_MO_KFSB=1` force-on; unset falls back to this field.
    #[serde(default)]
    pub use_kfsb_multi_branching: bool,
    /// Reuse a strictly-authorized scalar lower certificate from the historical
    /// wave-kFSB child simulations when it proves a committed child complete.
    ///
    /// Default false keeps every unqualified lane on the advisory-only path.
    /// An absent `NY_MO_KFSB_CERT_REUSE` inherits this typed setting, literal
    /// `1` force-arms it, and every other present byte string force-disarms it
    /// (including malformed and non-Unicode values).
    #[serde(default)]
    pub kfsb_cert_reuse: bool,
    /// Entry-time resolved certificate-reuse policy supplied by a scored
    /// frontend. `None` preserves the interactive environment-over-config
    /// resolver; `Some` is immutable runtime authority and must not reopen the
    /// process environment.
    #[serde(skip)]
    pub kfsb_cert_reuse_runtime: Option<bool>,
    /// Use the aggregation-critical objective row for the full graph kFSB
    /// scorer on auto-selected, high-dimensional conjunctive workloads.
    ///
    /// This is a runtime routing decision rather than a reusable preset knob:
    /// the CLI sets it only when model-aware `auto` branching resolves to kFSB
    /// for the high-dimensional/large-ReLU class. The selector additionally
    /// requires conjunctive aggregation. Default false preserves the cheaper
    /// historical intercept ranking for every explicit-kFSB, MIP-fallback, and
    /// disjunctive lane.
    #[serde(default)]
    pub use_multi_objective_critical_kfsb: bool,
    // --- Multi-Depth ReLU Splitting ---
    /// Max ReLU neurons to split per domain. Default: 1 (single-neuron).
    /// NY-only defensive expansion cap; it is not alpha-beta-CROWN's
    /// `max_split_depth` semantic.
    #[serde(default = "default_max_relu_split_depth")]
    pub max_relu_split_depth: usize,
    /// Queue fraction triggering multi-depth. Default: 0.5.
    /// Reference: alpha-beta-CROWN `min_batch_size_ratio`.
    #[serde(default = "default_min_batch_fill_ratio")]
    pub min_batch_fill_ratio: f32,
    // --- lA Warm-Start (GPU BaB) ---
    /// Enable lA warm-start in the GPU BaB backward pass.
    ///
    /// When enabled (default), child domains inherit cached linear bound
    /// coefficients (lA) from their parent's backward pass and use them to
    /// seed the next backward pass at the branch point instead of recomputing
    /// from the output node. This skips redundant backward computation for
    /// layers above the branch point.
    ///
    /// Disable with `--no-la-warm-start` for A/B benchmarking.
    ///
    /// # Reference
    /// alpha-beta-CROWN: `complete_verifier/tensor_storage.py` (all_lAs)
    #[serde(default = "default_enable_la_warm_start")]
    pub enable_la_warm_start: bool,
    // --- IBP Enhancement for Input Split ---
    /// Enable IBP-enhanced CROWN in the input-split BaB loop.
    ///
    /// When enabled, each domain (root and children) is first evaluated with IBP.
    /// Domains verified by IBP alone skip the expensive CROWN backward pass.
    /// Domains that remain undecided still run fresh spec-guided CROWN, but
    /// the current-domain IBP intermediates are threaded in as reference bounds
    /// so each nonlinear layer can keep the tighter of `{fresh CROWN, IBP}`.
    ///
    /// Reference name: alpha-beta-CROWN `bab.branching.input_split.ibp_enhancement`
    /// (`alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/input_split/bounding.py:131-136`)
    ///
    /// Default: false (no IBP enhancement).
    #[serde(default)]
    pub input_split_ibp_enhancement: bool,

    /// Admit the bounded proof-only nonnegative conic closure for the two-row
    /// conjunctive shape `[1, 0] <= +0 AND [0, -1] <= -0` during graph input
    /// splitting. Derived weighted rows are never added to the `VnnLibSpec`,
    /// attack paths, witness paths, or ReLU-split searches.
    ///
    /// Admission additionally requires verdict-only artifact authority because
    /// the external certificate format cannot currently encode the derived
    /// rows' provenance.
    ///
    /// Default: false (historical objective set and call path).
    #[serde(default)]
    pub input_split_conic_objective: bool,

    /// Maximum deferred-rebound tranche for an authenticated affine-conic
    /// reordered input-split search.
    ///
    /// Smaller tranches return newly bounded domains to the priority heap
    /// sooner, discard less completed work at a deadline, and bound the live
    /// `LinearBounds` matrices carried by the popped domain set. This setting
    /// has no effect outside the exact affine-conic route. Must be at least 1.
    #[serde(default = "default_input_split_conic_queue_refresh_batch_size")]
    pub input_split_conic_queue_refresh_batch_size: usize,

    /// Decompose the canonical exactly-two-singleton disjunction into
    /// sequential single-objective DomainList searches.
    ///
    /// Default false. This route is an explicit category treatment, not a
    /// shape-wide heuristic: the same AST occurs in unrelated cGAN and
    /// ml4acopf properties whose grouped search policy must remain unchanged.
    #[serde(default)]
    pub input_split_independent_singleton_disjunction: bool,

    /// `#lowdim-prover` at the DomainList input-split entry: on a graph whose
    /// input dimension is at most `lowdim::MAX_INPUT_DIM`, try the sound
    /// forward-linear lane first with half of this call's BaB slice. It only
    /// ever returns a proof of `lower(objective · y) > threshold`; anything
    /// else falls through to the unchanged driver with the time that remains.
    ///
    /// Default false: a category opts in by its typed preset key
    /// (`bab.branching.input_split.lowdim_prover`) after its own measurement.
    /// The linearizenn per-clause route arms the lane independently of this
    /// key, so it must NOT also set it (that would try the lane twice).
    #[serde(default)]
    pub input_split_lowdim_prover: bool,

    /// Fraction of the call's budget the `#lowdim-prover` lane may spend before
    /// falling through (read through [`Self::lowdim_prover_share`], which
    /// clamps it to [0.05, 0.95]; non-finite reads as 0.5). Default 0.5. A
    /// category whose fallback lane never proves the rows the lane is armed for
    /// (tllverifybench: every open row is a fallback timeout) can give it more;
    /// on sat rows the lane stops on a candidate within a second, so the share
    /// only matters where the lane is making progress.
    #[serde(default = "default_lowdim_prover_share")]
    pub input_split_lowdim_prover_share: f32,

    /// `#lowdim-prover` backward-CROWN refinement of pre-activation layers no
    /// wider than this (0 = off, the default). See `lowdim::refine_node`.
    #[serde(default)]
    pub input_split_lowdim_prover_refine_width: usize,

    /// #lsnc-lane stage 2: the `#lowdim-prover` lane's JOINT QUADRATIC BOUND
    /// at `ReduceSum(Mul(a, b))` sites (`lowdim::LowDimConfig::joint_quadratic`).
    /// Default false. NOT delivered by `preset::apply`: the CLI resolves
    /// `bab.branching.input_split.lowdim_joint_quadratic`
    /// (`NY_LOWDIM_JOINT_QUADRATIC`) from the scored run's frozen LeverSet, or
    /// the typed preset over the environment interactively, and writes it
    /// here (`commands::beta_crown::resolve_lowdim_joint_quadratic`).
    #[serde(default)]
    pub input_split_lowdim_joint_quadratic: bool,

    /// #lsnc-lane stage 3: the `#lowdim-prover` lane's split rule
    /// (`lowdim::LowDimConfig::split`; default `Widest`, the historical rule).
    /// NOT delivered by `preset::apply`: the CLI resolves
    /// `bab.branching.input_split.lowdim_split` (`NY_LOWDIM_SPLIT`) from the
    /// scored run's frozen LeverSet, or the typed preset over the environment
    /// interactively (`commands::beta_crown::resolve_lowdim_split`).
    #[serde(default)]
    pub input_split_lowdim_split: crate::lowdim::LowDimSplit,

    /// #lsnc-lane stage 3: the lane's RELAXED CLIP-AND-VERIFY
    /// (`lowdim::LowDimConfig::clip`). Default false. Resolved by the CLI
    /// from `bab.branching.input_split.lowdim_clip` (`NY_LOWDIM_CLIP`).
    #[serde(default)]
    pub input_split_lowdim_clip: bool,

    /// #lsnc-lane stage 3: decide each domain inside its rayon chunk task
    /// (`lowdim::LowDimParallel::Full`) instead of serially after the bound
    /// (`Bound`, the historical lane). Outcome-identical by construction.
    /// Default false. Resolved by the CLI from
    /// `bab.branching.input_split.lowdim_parallel` (`NY_LOWDIM_PARALLEL`).
    #[serde(default)]
    pub input_split_lowdim_parallel: bool,

    /// #lowdim-bundle (1): admit the lane by VARYING input dimensions (a box
    /// wider than `lowdim::MAX_INPUT_DIM` whose other coordinates are pinned
    /// is re-expressed over its free ones) and lower `Sigmoid` to the lane's
    /// rigorous sigmoid planes (`lowdim::LowDimConfig::varying_dims`).
    /// Default false. NOT delivered by `preset::apply`: the CLI resolves
    /// `bab.branching.input_split.lowdim_varying_dims` from the scored run's
    /// frozen LeverSet, or the typed preset over the environment
    /// interactively (`commands::beta_crown::resolve_lowdim_varying_dims`).
    #[serde(default)]
    pub input_split_lowdim_varying_dims: bool,

    /// #lowdim-bundle (2): boundary-alpha iterations of the lane's objective
    /// pass (`lowdim::LowDimConfig::alpha_iters`; 0 = off, the default).
    /// Resolved by the CLI from `bab.branching.input_split.lowdim_alpha_iters`.
    #[serde(default)]
    pub input_split_lowdim_alpha_iters: usize,

    /// #lowdim-bundle (3): in-lane worst-leaf PGD every this many waves
    /// (`lowdim::LowDimConfig::leaf_attack_every`; 0 = off, the default).
    /// Attack-only. Resolved by the CLI from
    /// `bab.branching.input_split.lowdim_leaf_attack_every`.
    #[serde(default)]
    pub input_split_lowdim_leaf_attack_every: usize,

    /// #lowdim-bundle (4): prove the lowdim fold's families on one shared
    /// worklist (`lowdim::LowDimConfig::shared_worklist`). Default false.
    /// Resolved by the CLI from
    /// `bab.branching.input_split.lowdim_shared_worklist`.
    #[serde(default)]
    pub input_split_lowdim_shared_worklist: bool,

    /// M8a (`#m8-input-split-batch`): the lowdim lane's wave BOUNDER
    /// (`lowdim::LowDimConfig::batch_engine`; default `Legacy`, the lane's own
    /// f64 bounder on every domain). `Bbe` hands each wave to the injected
    /// UNTRUSTED forward-linear search, whose claims the lane re-bounds with
    /// its own f64 forms. NOT delivered by `preset::apply`: the CLI resolves
    /// `bab.branching.input_split.lowdim_batch_engine`
    /// (`NY_LOWDIM_BATCH_ENGINE`) from the scored run's frozen LeverSet, or
    /// the typed preset over the environment interactively
    /// (`commands::beta_crown::resolve_lowdim_batch_engine`).
    #[serde(default)]
    pub input_split_lowdim_batch_engine: crate::lowdim::LowDimBatchEngine,

    /// M8a (`#m8-input-split-batch`): the lowdim lane's MULTI-CLAUSE scheduler
    /// (`lowdim::LowDimConfig::engine`; default `Legacy`). `Bbe` proves
    /// same-box clauses on one disjunction tree and runs the fold on the
    /// shared worklist. Resolved by the CLI from
    /// `bab.branching.input_split.engine` (`NY_INPUT_SPLIT_ENGINE`,
    /// `commands::beta_crown::resolve_input_split_engine`).
    #[serde(default)]
    pub input_split_engine: crate::lowdim::InputSplitEngine,

    /// `#lowdim-fold`: before the per-clause box screen of a graph model with
    /// per-clause input boxes (nn4sys mscn), run the lowdim lane on each
    /// clause over its FREE coordinates, folding the fixed ones (see
    /// `lowdim::prove_clause_folded`). Proofs only; default off.
    #[serde(default)]
    pub input_split_lowdim_fold: bool,

    // --- Domain-stacked batched rebound for Input Split (#cgan-batched-stack) ---
    /// Enable the domain-stacked dense-spec batched rebound in the input-split
    /// BaB loop.
    ///
    /// When enabled, the batched dense-spec CROWN backward kernel:
    /// 1. stacks the active domains' spec rows into ONE backward call per
    ///    Conv2d / ConvTranspose2d / BatchNorm node (instead of one call per
    ///    domain), amortizing the per-call im2col / weight-reshape / sound f64
    ///    recompute across the batch. Sound: those backward maps are
    ///    row-independent linear operators; any per-node box consumed by the
    ///    dispatch (BatchNorm certified-error discharge / precompute widening)
    ///    uses the elementwise HULL of the active domains' boxes, which can
    ///    only widen (equal-or-looser than the per-domain path, never tighter).
    ///    Any shape/op the stacked path cannot handle falls back to the
    ///    per-domain loop unchanged.
    /// 2. when `input_split_ibp_enhancement` is also set, seeds each domain's
    ///    intermediate-bounds cache with fresh per-subdomain IBP intersected
    ///    with the shared warmup reference bounds (the intersection of two
    ///    sound enclosures — sound, and matching the rayon per-domain
    ///    `ibp_enhancement` path that the batched kernel previously omitted).
    ///    Without this, all empty-history domains share bit-identical
    ///    backward relaxations and only the final concretization varies.
    ///
    /// Default: false (per-domain backward loop, verbatim shared cache — the
    /// historical behavior).
    #[serde(default)]
    pub input_split_stacked_rebound: bool,

    /// Nested-rebound deadline authority for the input-split dense-spec
    /// rebound (#dist-shift-5291; the `6f49a660` deadline-presence cliff —
    /// see `input_split/shared_specs.rs` and the cgan row-7 bisect note
    /// documented there).
    ///
    /// `None` / `Some(true)`: the nested rebound verifier keeps the caller's
    /// finite deadline — the shipped historical behavior since `6f49a660`.
    ///
    /// `Some(false)`: the nested rebound's bound computations run WITHOUT the
    /// inner finite deadline (the pre-`6f49a660` shape). Wall-clock safety is
    /// then provided by the OUTER input-split BaB loop, which polls its own
    /// deadline at the head of every batch iteration
    /// (`input_split/single_objective.rs` `check_termination`), plus the
    /// harness's external per-instance budget. What is ceded is INTRA-rebound
    /// interruptibility: a run can overrun the internal tier by at most one
    /// rebound batch, so only categories whose per-batch rebound cost is
    /// known-small (e.g. dist_shift's 25-Gemm FC net) should set `false`.
    /// Soundness is unaffected either way — the nested deadline only selects
    /// routes and fallbacks (the BUG #18 deadline-PRESENCE class), never
    /// bound math.
    ///
    /// Preset surface: `bab.branching.input_split.nested_deadline`. The env
    /// lever `NY_INPUT_SPLIT_NESTED_DEADLINE` overrides in BOTH directions
    /// (`0` forces drop, `1` forces keep, absent defers to this field, then
    /// to the declaration default of keep).
    #[serde(default)]
    pub input_split_nested_deadline: Option<bool>,

    /// Certified f64 tail re-bounding of knife-edge survivors on the
    /// grouped-disjunctive input-split lane (#cgan-f64-tail, design B).
    ///
    /// `None` (default) = dark. `Some(true)` arms BOTH escalation seams (the
    /// batch seam at the end of every deferred rebound AND the survivor-polish
    /// hook on the eager lane); `Some(false)` explicitly disarms. The env
    /// lever `NY_F64_TAIL` overrides in both directions (`1` force-on, any
    /// other present value force-off, absent defers here). ADDITIVE parity
    /// class: only certified `Verified` outcomes mutate anything, and only by
    /// monotonically raising per-row certified lower bounds through the
    /// existing grouped funnel.
    ///
    /// Preset surface: `bab.branching.input_split.f64_tail.enabled`.
    #[serde(default)]
    pub input_split_f64_tail_enabled: Option<bool>,

    /// Escalation guard band for the f64 tail (both seams): a still-unverified
    /// domain is eligible iff its grouped f32 gap is `>= -band`. `None` keeps
    /// the legacy 5e-3 default; `NY_F64_TAIL_BAND` overrides when set+valid.
    /// The cgan ship shape is 5e-4 (inside `CGAN_POLISH_BAND` 5e-3 — the
    /// scalar f64 conv walk is ~1-10 s/domain, so the band is the cost gate).
    ///
    /// Preset surface: `bab.branching.input_split.f64_tail.band`.
    #[serde(default)]
    pub input_split_f64_tail_band: Option<f32>,

    /// Hard per-run escalation cap SHARED by both f64-tail seams (0 or `None`
    /// = unlimited — the legacy env-armed lsnc behavior). Config-only (no env
    /// override; a new raw env read would raise the ny-levers ratchet). The
    /// cgan ship shape starts at 8: worst case 8 × 10 s slice ≈ 9% of the
    /// 900 s budget, owned by the pre-registered wall-parity kill criterion.
    ///
    /// Preset surface: `bab.branching.input_split.f64_tail.max_escalations`.
    #[serde(default)]
    pub input_split_f64_tail_max_escalations: Option<usize>,

    /// Per-CHILD wall cap, in SECONDS, for the grouped-disjunctive input-split
    /// per-child bound calls (#cgan-dark-tail FIX-1).
    ///
    /// THE SHIPPED DEFAULT IS `None` = NO CAP. With `None` every per-child
    /// bound keeps exactly the `crown_deadline` it is handed today and the
    /// route is byte-for-byte unchanged; this key ships DARK and changes
    /// nothing until a preset names it.
    ///
    /// `Some(secs)` (finite, `> 0`) makes each per-child bound call run against
    /// `min(crown_deadline, now + secs)` — `crown_deadline` remains the HARD
    /// absolute ceiling and is never extended, only possibly brought forward
    /// for the duration of one call.
    ///
    /// WHY: `disjunctive_multi_clause.rs` sets `crown_deadline` to the ABSOLUTE
    /// BaB deadline and threads it into every per-child bound, so a single
    /// child bound is legally entitled to the entire remaining budget — and
    /// because the BaB trajectory only prints BETWEEN batches, it does so
    /// invisibly. Measured on cgan row 2 in 13/13 logs: ~290 s (32% of the
    /// official 900 s budget) expanding 0-10 domains against a running
    /// 1.3-3.9 domains/s, with tail calls of ~70 s to >=285 s against a 0.13 s
    /// median over the preceding 1640 calls.
    ///
    /// SOUNDNESS: a per-call deadline can only make a bound LOOSER, never
    /// wrong. It selects routes and fallbacks, never bound math (the BUG #18
    /// deadline-PRESENCE class). The recovery path already exists and is
    /// already exercised in production: `screen_child.rs`'s
    /// `compute_child_bounds` catches the `Err` arm, retries the frozen-root-α
    /// bound and records `frozen_fallback_completions`. No new soundness
    /// surface; the moat is unaffected.
    ///
    /// Config-only (no env override: a new raw env read would raise the
    /// ny-levers ratchet).
    ///
    /// Preset surface: `bab.branching.input_split.child_bound_max_secs`.
    #[serde(default)]
    pub input_split_child_bound_max_secs: Option<f64>,

    /// #cost-observer: SELF-CALIBRATING replacement for
    /// [`Self::input_split_child_bound_max_secs`].
    ///
    /// THE SHIPPED DEFAULT IS `None` = the constant above governs, unchanged
    /// and byte-for-byte. Naming this key makes each per-child bound call run
    /// against an allowance DERIVED at runtime from (a) the median cost of the
    /// calls that have already completed on this row and (b) the wall actually
    /// remaining — instead of a number chosen in advance.
    ///
    /// WHY THE CONSTANT CANNOT BE RIGHT. A/B'd 2026-08-26 at the official
    /// 1200 s on the two 1200 s cgan rows at `child_bound_max_secs: 20.0`:
    /// `nonlinear_activations` went `explored 2 -> 957` with ZERO collection
    /// truncations (20 s is CORRECT there — it removes pure waste), while
    /// `upsample` took 37 truncations that cut a 27-layer intermediate
    /// collection to 5-6 layers with the gap unmoved (20 s is an order of
    /// magnitude TOO SHORT there). The distinguishing fact is the seam's own
    /// per-call cost on that row on this host — `0.13 s` median versus
    /// `>= 300 s` — which no constant can carry and nothing in the instance
    /// declares.
    ///
    /// SOUNDNESS: unchanged from the constant, and for the same structural
    /// reason. The derived allowance is composed by the SAME
    /// `min(crown_deadline, now + allowance)` law, so the absolute ceiling is
    /// never extended — only ever brought forward — and an earlier deadline can
    /// only make a bound LOOSER or fail the call outright (the frozen-root-α
    /// retry in `screen_child.rs` already catches that arm and already records
    /// `frozen_fallback_completions`). `ny_core::cost_observer::Slice` has no
    /// variant that can extend a ceiling, so no allowance the observer can
    /// emit — including a wrong one — can change what is CONCLUDED from a
    /// bound, only how much work is spent computing it.
    ///
    /// Config-only (no env override: a new raw env read would raise the
    /// ny-levers ratchet).
    ///
    /// Preset surface: `bab.branching.input_split.child_bound_calibration`.
    #[serde(default)]
    pub input_split_child_bound_calibration: Option<ChildBoundCalibrationConfig>,

    /// #cap-extend (DEFER-dont-DROP): when the input-split disjunctive BaB
    /// loop trips `max_domains` with the frontier NEARLY CLOSED, extend the
    /// search (retain the open queue and keep exploring, bounded by
    /// `CAP_EXTEND_FACTOR` and the wall) instead of discarding the frontier.
    ///
    /// THE SHIPPED DEFAULT IS `false` = the domain-limit stop is byte-for-byte
    /// identical to today. Naming the key changes SCHEDULE only, never bound
    /// math: a granted extension permits MORE search before Unknown/Timeout,
    /// and a Verified verdict still requires every clause refuted on every
    /// open domain by the unchanged bounding code.
    ///
    /// THE DEFECT IT ADDRESSES (cgan2026 row 7, cGAN_imgSz32_nCh_3 prop_3,
    /// published truth unsat, official 900 s): the multi-seed restart plan
    /// caps probe restarts at 2,500 domains. The probe reached
    /// `best_gap=-0.0000` with ~40% of explored domains verified, tripped the
    /// cap at t=464 s, and the committing restart RE-TRACED the identical
    /// trajectory from `explored=1` — burning the remaining ~430 s re-deriving
    /// bounds the probe already held, publishing timeout at 895 s. The
    /// near-closure gate (see `cap_extend_decision` in
    /// `disjunctive_multi_clause.rs`) grants extension only when the frontier
    /// has demonstrably approached closure; a cap that trips far from closure
    /// stops exactly as today (the FC cap-lift hazard,
    /// docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md, is why this is
    /// cap-CONDITIONAL, not a cap lift).
    ///
    /// Config-only (no env override: a new raw env read would raise the
    /// ny-levers ratchet).
    ///
    /// Preset surface: `bab.branching.input_split.cap_extend_near_closure`.
    #[serde(default)]
    pub input_split_cap_extend_near_closure: bool,

    /// #sb-plane: keep the popped domain's rebound plane available to the SB
    /// split-dimension scorer while the fresh current-domain clip is armed.
    ///
    /// The fresh clip route (`input_split_fresh_domain_clip`) exists because a
    /// batch-stack-unsafe graph's domain-stacked rebound planes are not
    /// trusted as clip/verdict AUTHORITY, so `process_batch` drops
    /// `domain.linear_bounds` on every pop before any helper can consume it —
    /// and the SB scorer, which is a HEURISTIC that cannot touch soundness,
    /// loses its coefficients with it and falls back to width-only scoring
    /// (`input_split_scores`' `width_only` arm). With this key on, the plane
    /// is moved OUT of the domain into a split-decision-only local: the fresh
    /// clip still computes its own plane on the exact box, no clip or verified
    /// check can reach the moved plane, and children are still born with
    /// `linear_bounds: None`.
    ///
    /// Inert unless the fresh clip is actually armed on the lane (announced by
    /// `[sb-plane] armed=`). Config-only (no env override: a new raw env read
    /// would raise the ny-levers ratchet). Ships dark; `cgan_2023.yaml` is the
    /// measurement target (canary copy, not the production yaml).
    ///
    /// Preset surface: `bab.branching.input_split.sb_keep_plane_under_fresh_clip`.
    #[serde(default)]
    pub input_split_sb_keep_plane_under_fresh_clip: bool,

    /// Run eligible per-domain warm-alpha refinements concurrently during the
    /// deferred reordered input-split rebound.
    ///
    /// This is a preset-scoped activation gate: the global
    /// `NY_INPUT_SPLIT_WARM_PARALLEL` environment setting cannot arm a category
    /// whose preset leaves this false. Default false preserves the historical
    /// serial refinement loop for every existing preset.
    #[serde(default)]
    pub input_split_warm_parallel: bool,

    /// Evaluate complete-clip child-local override rebounds two at a time in
    /// the scalar and dense-spec deferred reordered input-split loops.
    ///
    /// The executor is deliberately fixed at two workers: each worker owns a
    /// full CROWN carrier, so unconstrained Rayon fan-out can multiply peak
    /// memory by the host's core count. Results are collected and applied in
    /// domain order, preserving deterministic heap mutation and error choice.
    /// A conservative workload estimate must also fit within one eighth of a
    /// live, kernel-enforced process envelope; otherwise the route falls back
    /// to the serial reference. This limits incremental OOM risk but is not a
    /// general whole-CROWN peak-memory guarantee, so presets should opt in only
    /// for measured small models. This typed gate takes precedence over the
    /// separate collection-verify shortcut whenever the environment selector
    /// remains armed, including when memory admission selects serial execution.
    /// The environment kill switch restores independent shortcut selection. A
    /// preset must opt in before
    /// `NY_INPUT_SPLIT_OVERRIDE_PARALLEL` may select the parallel arm. Default
    /// false preserves the historical serial path.
    #[serde(default)]
    pub input_split_override_parallel: bool,

    /// Arm Saturation-Escape Branching (SEB, #nn4sys-seb-dark) by default for
    /// this config: the input-split dim scorer that de-saturates a binding
    /// sigmoid/tanh logit (`engine/graph/input_split/sat_escape.rs`), plus the
    /// disjunctive precheck-fraction cap that reserves per-clause BaB budget
    /// for the brancher (ny-cli `verify/disjunctive.rs`). Advisory + budget
    /// only — the split partition stays an exact cover and every verdict path
    /// is unchanged, so soundness is untouched either way.
    ///
    /// Probe (scratchpad/invprop_ab/mscn_dual_SEB_probe_RESULTS_2026-07-20.txt):
    /// 1-D mscn dual disjuncts close in 7–20 leaves, 2-D in 57–369 with SEB vs
    /// 697/timeout blind, while CROWN-per-box flips 0/1200. Preset opt-in
    /// (nn4sys); every preset that does not name the key keeps today's dark
    /// default. Env `NY_SAT_ESCAPE_BRANCH` overrides either way: `1` force-on,
    /// `0` kill switch, unset falls back to this field
    /// (see [`Self::sat_escape_branch_armed`]).
    #[serde(default)]
    pub sat_escape_branch: bool,

    /// INTERNAL companion to `input_split_stacked_rebound`: enables the fresh
    /// per-domain IBP intersect in the batched kernel's forward stage. Set by
    /// the input-split batched adapter (`stacked_rebound && ibp_enhancement`);
    /// not a preset surface. Default: false (verbatim shared-cache clone).
    #[serde(default)]
    pub input_split_batched_ibp_refresh: bool,

    /// #relational-bab lever 1 (default OFF): in the multi-clause DISJUNCTIVE
    /// input-split rebound, derive the spec-row bounds by interval arithmetic
    /// over the per-domain intermediate collection's OUTPUT entry and SKIP the
    /// spec backward for domains that already verify. For pure `±e_i` band
    /// rows with per-node CROWN-IBP intermediates the two are BIT-IDENTICAL
    /// (measured on the relational ACAS difference nets), so the backward is
    /// pure redundancy for the verified majority at the frontier; survivors
    /// still run the standard spec backward over the SAME collected
    /// intermediates (no recollection) so split scoring / clipping keep their
    /// linear bounds. Sound: the interval combination of a sound output
    /// enclosure is a sound spec bound; the monotonic parent-tighten guard is
    /// applied identically on both paths.
    #[serde(default)]
    pub input_split_collection_verify_shortcut: bool,

    /// #scored-wall-clamp (default `None`): the ABSOLUTE instant by which the
    /// process owes its caller a verdict, independent of `timeout`.
    ///
    /// `timeout` (and any deadline derived from it) is an EFFORT HORIZON: the
    /// disjunctive input-split lane sizes every per-node CROWN-IBP grant as a
    /// fraction of the budget it believes remains, so a NEARER deadline buys
    /// LOOSER bounds and MORE splits. That makes the search non-monotonic in
    /// its own timeout, and a frontend may legitimately inflate the timeout
    /// past the wall it actually has in order to keep a converging trajectory.
    ///
    /// What it may NOT do is run that long. Supplying this field lets the two
    /// separate: effort keeps being shaped by `timeout`, while the loop stops
    /// here. It can only ever SHORTEN a run — a wall beyond the horizon is
    /// ignored — so it is never a way to extend a configured timeout.
    ///
    /// `None`, the default and the only value any serialized configuration can
    /// produce, means "no wall of my own": stop and horizon coincide and the
    /// lane behaves byte-identically to the single-deadline code.
    ///
    /// NOT SERIALIZABLE, deliberately, for the same reason
    /// [`Self::verification_artifact_authority`] is not: this is a LIVE
    /// instant on the running process's monotonic clock. A preset that
    /// restated it as a duration would silently forget the seconds already
    /// spent before the solver was constructed, which is exactly the accounting
    /// error the field exists to prevent. Frontends compute it once, at their
    /// own anchor, from the scored budget.
    #[serde(skip)]
    pub input_split_scored_wall: Option<Instant>,

    /// #relational-bab lever 2 (default OFF): honor `input_split_depth` in
    /// the multi-clause DISJUNCTIVE input-split lane — split the top-k SB
    /// dims per pop (up to `2^k` children exactly covering the parent),
    /// mirroring the conjunctive multi-objective lane. OFF preserves the
    /// historical single-dim split byte-identically.
    #[serde(default)]
    pub input_split_disjunctive_multi_dim: bool,

    /// Offer every still-undecided grouped-disjunctive input subdomain to the
    /// attached input-box leaf oracle before the Graph-MIP edge escalation
    /// collects intermediate node bounds.
    ///
    /// The oracle receives the exact input box and complete grouped objective
    /// layout. Only a deadline-valid `VerifiedAllRows` may discharge the
    /// domain; `Undecided`, advisory violations, malformed requests, and late
    /// results fall through to the historical edge-MIP/split path unchanged.
    /// This foundation requires certification of every objective row; grouped
    /// one-row-per-clause authority is deferred until a typed certified-row
    /// mask can be validated by the caller.
    /// Default false keeps existing configurations and Graph-MIP-only oracles
    /// inert at this new seam.
    #[serde(default)]
    pub input_split_input_leaf_oracle: bool,

    /// #relational-bab EDGE-DOMAIN ESCALATION (default OFF): in the
    /// multi-clause DISJUNCTIVE input-split lane, a near-verified deep domain
    /// (every unverified single-row clause within `input_split_edge_milp_gap`
    /// of its threshold, `depth >= input_split_edge_milp_depth`) is offered to
    /// the attached [`GraphMipLeafOracle`](crate::beta_crown::graph_mip_leaf)
    /// with NO split premises (the input box IS the subdomain) before being
    /// split further. `VerifiedAllRows` (certified-UNSAT only, the leaf lane's
    /// 0-wrong contract) counts the domain verified; anything else requeues
    /// unchanged. This DECIDES relaxation-floor edge domains — boundary
    /// -unstable neuron pairs whose plain-CROWN slack no amount of input
    /// splitting eliminates — instead of splitting them forever. Inert unless
    /// an oracle is attached.
    #[serde(default)]
    pub input_split_edge_milp: bool,
    /// Edge gate: max distance-to-threshold (over unverified rows) for a
    /// domain to qualify as an escalation candidate.
    #[serde(default = "default_edge_milp_gap")]
    pub input_split_edge_milp_gap: f32,
    /// Edge gate: minimum BaB depth before escalation is considered.
    #[serde(default = "default_edge_milp_depth")]
    pub input_split_edge_milp_depth: usize,

    /// #relational-bab OPTION B (default OFF): α-OPTIMIZED slopes on EDGE
    /// domains. A popped near-verified domain (same gap/depth gates as the
    /// MILP escalation) gets a per-domain α-CROWN pass — optimized lower
    /// slopes over its exact sub-box — on its unverified rows BEFORE
    /// splitting; verified ⇒ done, still-short ⇒ split with the improved
    /// (monotonicity-guarded) bounds. Composes with the MILP consult (α
    /// first; the certified MILP finishes when the free-binary count
    /// collapses at depth). Measured α-over-plain-CROWN gains (1e-3..1e-1)
    /// cover most of the −0.0002..−0.03 relaxation floor.
    #[serde(default)]
    pub input_split_edge_alpha: bool,
    /// Per-WAVE cap on α edge passes (the pop order is worst-gap-first, so
    /// the cap keeps the most negative gaps; ~50-200ms per pass).
    #[serde(default = "default_edge_alpha_top")]
    pub input_split_edge_alpha_top: usize,
    /// α iterations per edge pass (modest; deadline-guarded).
    #[serde(default = "default_edge_alpha_iters")]
    pub input_split_edge_alpha_iters: usize,
    /// #iso-edge-cap (default `None` = DARK, byte-identical): bound the SERIAL
    /// edge work above — the α edge pass and the edge-MILP escalation — by the
    /// scored stop (`input_split_scored_wall`, when the frontend supplies one)
    /// and by a per-wave wall allowance: `max_secs`, or `rebound_share` times
    /// the batched rebound that precedes the wave, whichever is smaller.
    ///
    /// MEASURED defect (Mac, `624477367`, isomorphic_acasxu_2026 08-30
    /// resweep): single batches of 15-63 s against a ~1.6 s norm on 6 of 18
    /// gap rows, each starting exactly when the tree reaches the edge gate's
    /// depth, one running the process to 100.96 s of a 100 s budget. Both edge
    /// helpers took the INFLATED effort horizon as their deadline and nothing
    /// bounded their share of a wave.
    ///
    /// SOUNDNESS: it can only REMOVE edge work, which turns a would-be verify
    /// into a split; bound-shaping deadlines are unchanged, an α optimisation
    /// cut short still yields a sound (looser) bound, and the edge-MILP still
    /// counts a domain only on the certified `VerifiedAllRows`.
    ///
    /// Config-only (no env override: a new raw env read would raise the
    /// ny-levers ratchet). The relational lane builds its config inline, so its
    /// frontend projects this key from the category preset itself
    /// (`RelationalPresetKeys` in ny-cli `vnncomp.rs`).
    ///
    /// Preset surface: `bab.branching.input_split.edge_wave_cap`.
    #[serde(default)]
    pub input_split_edge_wave_cap: Option<EdgeWaveCapConfig>,

    // --- Reorder BaB for Input Split ---
    /// Use the reordered BaB loop for input splitting.
    ///
    /// In regular mode, the loop order is: pop → split → clip → bound children → enqueue.
    /// In reorder mode, the loop order is: pop → bound → filter verified → split → clip → enqueue.
    ///
    /// Reorder mode bounds the domain before deciding whether to split. Verified
    /// domains are filtered before splitting, saving clip + child construction
    /// overhead. Split decisions use fresh CROWN linear coefficients from the
    /// bounding step.
    ///
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.reorder_bab`
    /// (`alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/input_split/batch_branch_and_bound.py:58-63`)
    ///
    /// Default: false (regular BaB loop order).
    #[serde(default)]
    pub reorder_bab: bool,

    /// Domain-count threshold for adversarial checking during input-split BaB.
    ///
    /// After `adv_check` domains have been explored, run a lightweight PGD probe
    /// on the worst queued domains each iteration to find counterexamples early.
    /// - `-1`: disabled (never run adv_check during BaB)
    /// - `0`: run from the first iteration
    /// - `N > 0`: run after N domains have been explored
    ///
    /// Reference: alpha-beta-CROWN `bab.branching.input_split.adv_check`
    /// (`alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/input_split/batch_branch_and_bound.py:512-522`)
    ///
    /// Default: -1 (disabled).
    #[serde(default = "default_adv_check")]
    pub adv_check: i32,

    // --- Adaptive Batch Sizing ---
    /// Enable opt-in adaptive BaB microbatch sizing.
    ///
    /// By itself this retains the legacy full-batch enlargement policy. When
    /// the independent `NY_ADAPTIVE_MICROBATCH_CONTROLLER=1` gate is also set
    /// exactly, graph ReLU-split and DomainList input-split routes grow only
    /// when their tensor-byte estimate leaves the configured reserve available.
    /// They back off on retryable allocation/dispatch refusals and may shrink
    /// after underfill, memory pressure, or repeated long passes. Other BaB
    /// routes retain the legacy helper, capped at
    /// `AUTO_ENLARGE_BATCH_CAP` (8192).
    ///
    /// Reference: alpha-beta-CROWN `auto_LiRPA/utils.py:348-381` (`AutoBatchSize`).
    /// WGPU/Metal currently uses NY's configured/system-memory backend-budget
    /// fallback plus measured tensor bytes; it does not query live free VRAM.
    ///
    /// Default: false. The adaptive controller's independent environment gate
    /// is also default-dark; unset, `0`, or malformed values preserve the
    /// historical execution and fallback path even for existing presets that
    /// set this field.
    #[serde(default)]
    pub auto_enlarge_batch_size: bool,

    // --- #dd-zonotope per-category admission overrides (#metaroom-ddzono) ---
    /// Preset-driven override of `DdZonoConfig::min_input_numel`.
    ///
    /// These four knobs make the certified double-double zonotope's
    /// ADMISSION caps reachable from a category preset, because the scored
    /// `ny vnncomp v1` entry point sets no environment variables. They only
    /// resize the fail-closed detector's blast-radius/resource caps; every
    /// soundness gate — the self-policing precision gate, the rounding-channel
    /// safety factor, `dd_selfcheck`, the FP-environment probe, and outward
    /// f64→f32 narrowing at the verdict — is untouched and NOT preset-
    /// configurable. `None` (the default for every existing preset and
    /// serialized config) is byte-identical to today. An explicitly set
    /// `NY_DD_ZONOTOPE_*` environment variable keeps precedence over these
    /// (see `DdZonoConfig::with_admission_overrides`).
    #[serde(default)]
    pub dd_zonotope_min_input_numel: Option<usize>,
    /// Preset-driven override of `DdZonoConfig::max_k` (perturbed-input cap).
    #[serde(default)]
    pub dd_zonotope_max_k: Option<usize>,
    /// Preset-driven override of `DdZonoConfig::max_generators` (live
    /// generator-column cap; exceeding it still FAILS CLOSED).
    #[serde(default)]
    pub dd_zonotope_max_generators: Option<usize>,
    /// Preset-driven override of `DdZonoConfig::collect_interm`
    /// (`#dd-zono-interm`): populate the certified per-node enclosure map so
    /// the root coordinator may INTERSECT it into the stored intermediate
    /// bounds. Intersection of two certified enclosures of the same quantity
    /// is itself a certified enclosure, so this is bound-tightening only.
    #[serde(default)]
    pub dd_zonotope_collect_interm: Option<bool>,

    // --- Phase Budget Policy ---
    /// Phase-level time budget fractions for β-CROWN verification.
    ///
    /// Centralizes the timeout policy that was previously scattered across
    /// `attack_budget.rs`, `sequential.rs`, `graph.rs`, `disjunctive.rs`,
    /// and `mod.rs` as hardcoded magic constants.
    ///
    /// Default values preserve today's behavior exactly.
    #[serde(default)]
    pub phase_budget: PhaseBudgetConfig,
}

/// Maximum batch size for auto-enlargement.
/// Matches alpha-beta-CROWN's practical cap for GPU memory safety.
pub const AUTO_ENLARGE_BATCH_CAP: usize = 8192;

fn default_adv_check() -> i32 {
    -1
}

impl Default for BetaCrownConfig {
    fn default() -> Self {
        Self {
            verification_artifact_authority: VerificationArtifactAuthority::CertificateExport,
            frozen_levers: None,
            certified_root_cut_owner: None,
            certified_root_cut_raw_source: false,
            max_domains: 100_000,
            max_queue_size: default_max_queue_size(),
            // #ml4acopf-bab-queue-mem: 0 = unlimited. Opt-in per preset so
            // every category that does not set it is byte-identical.
            max_queue_bytes: 0,
            timeout: Duration::from_mins(5),
            max_depth: 100,
            initial_bounds_warmup_fallback: false,
            disjunctive_clause_retry_passes: 0,
            disjunctive_shared_bootstrap: false,
            genbab_on_precomputed_lane: false,
            use_alpha_crown: true,
            use_forward_bounds: false,
            optimize_disjuncts_separately: false,
            root_alpha_cap_secs: None,
            root_alpha_phase_checkpoint: false,
            atomic_root_c_margin_iterations: 0,
            use_crown_ibp: false, // Disabled by default (more expensive)
            chunk_wave_parallel_enabled: false,
            max_crown_ibp_nodes: None,
            crown_ibp_per_node_floor_secs: None,
            crown_ibp_per_node_floor_dim_scaled: false,
            pwl_lookup_exact_range: false,
            forward_root_screen: false,
            forward_root_screen_max_mb: default_forward_root_screen_max_mb(),
            vit_zono_root: false,
            vit_zono_root_max_mb: default_vit_zono_root_max_mb(),
            vit_zono_root_max_secs: default_vit_zono_root_max_secs(),
            bilinear_rowpar_kernel: false,
            linear_position_stack: false,
            reciprocal_unbounded_upper: false,
            bilinear_gemm_kernel: false,
            single_hidden_hull: SingleHiddenHullMode::Off,
            graph_search: GraphLaneSearch::Off,
            la_warm_frontier: false,
            kfsb_single_shot: false,
            attack_concurrent: false,
            relational_point_vjp: false,
            mip_refine_intermediates: None,
            gpu_patches_crown: None,
            crown_ibp_per_node_cap_secs: None,
            crown_ibp_prefix_cost_weighted: false,
            crown_ibp_max_target_share_frac: None,
            crown_backward_layers: None,
            alpha_config: AlphaCrownConfig::default(),
            branching_heuristic: BranchingHeuristic::LargestBoundWidth,
            input_split_depth: default_input_split_depth(),
            input_split_coeff_thresh: default_input_split_coeff_thresh(),
            input_split_alpha_iteration: default_input_split_alpha_iteration(),
            input_split_lr_alpha: default_input_split_lr_alpha(),
            input_split_touch_zero_score: default_input_split_touch_zero_score(),
            input_split_sb_margin_weight: default_input_split_sb_margin_weight(),
            input_split_sb_sum: false,
            input_split_sb_primary_spec: None,
            fsb_candidates: default_fsb_candidates(),
            // #general-defaults 2026-08-31: was KfsbReduceOp::default() (Min). ALL 13
            // scored configs that set reduceop set `max` -- unanimous, so Min governed
            // nothing except untuned benchmarks.
            kfsb_reduce_op: KfsbReduceOp::Max,
            kfsb_multi_candidates: None,
            kfsb_multi_reduce_op: None,
            depth_two_branch_lookahead: DepthTwoBranchLookaheadConfig::default(),
            // #general-defaults 2026-08-31: was 0.05 (abc's arguments.py value).
            // 11 of the 14 scored configs that set lr_beta set 0.15; none ship 0.05.
            // A default that every serious config overrides is a wrong default, and an
            // UNTUNED benchmark (the 2026 set) is the only thing it actually governs.
            beta_lr: 0.15,
            // #general-defaults 2026-08-31: was 0, i.e. per-domain beta ascent shipped
            // OFF. All 14 scored configs that set this set it > 0 (modal 15); NOTHING
            // ships 0. An untuned benchmark got a verifier with beta ascent disabled.
            beta_iterations: 15,
            beta_tolerance: 1e-5,
            early_stop_patience: default_early_stop_patience(),
            root_beta_iterations: default_root_beta_iterations(), // Root-level optimization
            beta_max_depth: default_beta_max_depth(), // Limit per-domain optimization depth
            use_analytical_beta_gradients: true, // Use analytical gradients for ~3x faster optimization
            // #general-defaults 2026-08-31: was 0.01. 13 of the 14 scored configs that
            // set beta_crown.lr_alpha set 0.1 (cora is the lone 0.03 and keeps its key).
            alpha_lr: 0.1,
            alpha_momentum: true, // Use momentum for α updates
            batch_size: 64,       // Process 64 domains in parallel (GPU-optimized)
            build_batch_size: None,
            parallel_children: true, // Enable parallel child creation by default
            use_adaptive: false,     // Disabled by default for backward compatibility
            adaptive_config: AdaptiveOptConfig::default(),
            enable_cuts: false, // Disabled by default for backward compatibility
            conv_mode: ConvMode::default(), // Auto: patches unless cuts enabled
            max_cuts: default_max_cuts(),
            min_cut_depth: default_min_cut_depth(),
            enable_near_miss_cuts: false, // Disabled by default
            near_miss_margin: default_near_miss_margin(),
            enable_proactive_cuts: false, // Disabled by default
            max_proactive_cuts: default_max_proactive_cuts(),
            enable_biccos_constraint_strengthening: false, // Disabled by default
            biccos_drop_ratio: default_biccos_drop_ratio(),
            lambda_opt_interval: default_lambda_opt_interval(),
            lambda_lr: default_lambda_lr(),
            cut_eviction_policy: CutEvictionPolicy::default(),
            cut_stale_iters: default_cut_stale_iters(),
            cut_hard_stale_iters: default_cut_hard_stale_iters(),
            cut_lambda_min: default_cut_lambda_min(),
            cut_proactive_fraction: default_cut_proactive_fraction(),
            cut_score_weights: CutScoreWeights::default(),
            enable_biccos_cold_start: false, // BICCOS cold-start gating
            biccos_min_verified: default_biccos_min_verified(),
            biccos_min_verified_rate: default_biccos_min_verified_rate(),
            biccos_verified_rate_window: default_biccos_verified_rate_window(),
            biccos_min_cuts: default_biccos_min_cuts(),
            biccos_min_bound_gain: default_biccos_min_bound_gain(),
            biccos_bound_gain_window: default_biccos_bound_gain_window(),
            biccos_cold_max_iters: default_biccos_cold_max_iters(),
            biccos_cut_window: default_biccos_cut_window(),
            biccos_min_cut_yield: default_biccos_min_cut_yield(),
            biccos_cut_yield_window: default_biccos_cut_yield_window(),
            biccos_cut_yield_patience: default_biccos_cut_yield_patience(),
            verify_upper_bound: false,
            enable_pgd_attack: false,
            pgd_restarts: default_pgd_restarts(),
            pgd_steps: default_pgd_steps(),
            pgd_restart_when_stuck: false,
            pgd_initialization: PgdInitialization::Uniform,
            pgd_osi_steps: default_pgd_osi_steps(),
            pgd_gama: false,
            pgd_optimizer: PgdOptimizer::AdamClipping,
            pgd_alpha_mode: PgdAlphaMode::Auto,
            pgd_lr_decay: default_pgd_lr_decay(),
            pgd_surrogate_sign_gradient: false,
            pgd_dense_low_dim_sweep: false,
            pgd_dense_sweep_max_dims: default_pgd_dense_sweep_max_dims(),
            pgd_dense_sweep_points: default_pgd_dense_sweep_points(),
            pgd_corner_seed_restarts: 0,
            open_disjunct_attack: OpenDisjunctAttackConfig::default(),
            pgd_exact_vjp: ExactVjpPgdConfig::default(),
            enable_relaxed_clip: false,
            relaxed_clip_iterations: default_relaxed_clip_iterations(),
            input_clip_type: InputClipType::default(),
            input_split_fresh_domain_clip: false,
            clip_neuron_selection_ratio: default_clip_neuron_selection_ratio(),
            enable_clip_interm_domain: false,
            clip_interm_topk: default_clip_interm_topk(),
            clip_in_alpha_crown: false,
            clip_interm_prune: false,
            clip_interm_use_final_layer: false,
            enable_interm_transfer: true,
            root_interm_cuda_factory: false,
            mo_cuda_factory_engine_handoff: false,
            mo_cuda_bounded_shared_executor: false,
            mo_containment_fence: false, // #vit-bank-gate: default-off fence
            mo_root_ort_adjudication: false, // #vit-margin-alpha-complex (P8)
            mo_cover_certify: MoCoverCertify::Off, // #vit-v3-bankable: dark
            root_crown_interm_dense_head: false,
            root_crown_interm_max_secs: default_root_crown_interm_max_secs(),
            root_crown_interm_max_dim: default_root_crown_interm_max_dim(),
            root_sparse_interm_crown: false,
            root_sparse_interm_crown_max_secs: default_root_sparse_interm_crown_max_secs(),
            root_sparse_interm_crown_max_dim: default_root_sparse_interm_crown_max_dim(),
            root_sparse_interm_crown_max_rows: default_root_sparse_interm_crown_max_rows(),
            root_sparse_interm_crown_max_targets: default_root_sparse_interm_crown_max_targets(),
            root_comprehensive_gpu_interm: false,
            // #bab-floor: None, not 0.0 — absent means the arbitration does
            // not exist, which is the byte-identical shipped ladder. 0.0 is a
            // different statement: an explicit kill switch that still leaves
            // the two shares parseable.
            root_bab_reserve_frac: None,
            root_spec_frac: None,
            root_alpha_frac: None,
            root_comprehensive_gpu_interm_chunks: default_root_comprehensive_gpu_interm_chunks(),
            root_post_c_survivor: false,
            mo_beta_graft: false, // #mo-beta-graft (env NY_MO_BETA_GRAFT overrides)
            bab_chain_wide: false, // #chain-wide (preset bab.chain_wide; graft-composed FC routing)
            use_kfsb_multi_branching: false, // #kfsb-multi: CIFAR-100 preset opt-in (env NY_MO_KFSB overrides)
            kfsb_cert_reuse: false, // proof-carrying kFSB simulation reuse; preset-scoped opt-in
            kfsb_cert_reuse_runtime: None, // scored frontend freezes the governed effective value
            use_multi_objective_critical_kfsb: false, // auto high-dimensional conjunction only
            max_relu_split_depth: default_max_relu_split_depth(),
            min_batch_fill_ratio: default_min_batch_fill_ratio(),
            enable_la_warm_start: default_enable_la_warm_start(),
            input_split_ibp_enhancement: false,
            input_split_conic_objective: false,
            input_split_conic_queue_refresh_batch_size:
                default_input_split_conic_queue_refresh_batch_size(),
            input_split_independent_singleton_disjunction: false,
            input_split_lowdim_prover: false,
            input_split_lowdim_prover_share: default_lowdim_prover_share(),
            input_split_lowdim_prover_refine_width: 0,
            input_split_lowdim_joint_quadratic: false,
            input_split_lowdim_split: crate::lowdim::LowDimSplit::Widest,
            input_split_lowdim_clip: false,
            input_split_lowdim_parallel: false,
            input_split_lowdim_varying_dims: false,
            input_split_lowdim_alpha_iters: 0,
            input_split_lowdim_leaf_attack_every: 0,
            input_split_lowdim_shared_worklist: false,
            input_split_lowdim_batch_engine: crate::lowdim::LowDimBatchEngine::Legacy,
            input_split_engine: crate::lowdim::InputSplitEngine::Legacy,
            input_split_lowdim_fold: false,
            input_split_stacked_rebound: false, // #cgan-batched-stack
            input_split_nested_deadline: None,  // #dist-shift-5291 nested-deadline authority (keep)
            input_split_f64_tail_enabled: None, // #cgan-f64-tail: dark until the measured gate
            input_split_f64_tail_band: None,    // #cgan-f64-tail: 5e-3 legacy default beneath
            input_split_f64_tail_max_escalations: None, // #cgan-f64-tail: unlimited (legacy lsnc)
            input_split_child_bound_max_secs: None, // #cgan-dark-tail FIX-1: NO CAP (ships dark)
            input_split_child_bound_calibration: None, // #cost-observer: constant governs (dark)
            input_split_cap_extend_near_closure: false, // #cap-extend: DEFER-dont-DROP (ships dark)
            input_split_sb_keep_plane_under_fresh_clip: false, // #sb-plane: SB keeps the rebound plane (ships dark)
            input_split_warm_parallel: false, // #cgan-warm-par (preset-scoped opt-in)
            input_split_override_parallel: false, // bounded complete-clip rebound fan-out
            sat_escape_branch: false, // #nn4sys-seb-dark: nn4sys preset opt-in (env NY_SAT_ESCAPE_BRANCH overrides)
            input_split_batched_ibp_refresh: false, // #cgan-batched-stack
            input_split_collection_verify_shortcut: false, // #relational-bab lever 1
            input_split_scored_wall: None, // #scored-wall-clamp: no wall of my own
            input_split_disjunctive_multi_dim: false, // #relational-bab lever 2
            input_split_input_leaf_oracle: false, // input-box-only certified leaf seam
            input_split_edge_milp: false, // #relational-bab edge escalation
            input_split_edge_milp_gap: default_edge_milp_gap(),
            input_split_edge_milp_depth: default_edge_milp_depth(),
            input_split_edge_alpha: false, // #relational-bab option B
            input_split_edge_alpha_top: default_edge_alpha_top(),
            input_split_edge_alpha_iters: default_edge_alpha_iters(),
            input_split_edge_wave_cap: None, // #iso-edge-cap: edge work ungoverned (ships dark)
            reorder_bab: false,
            adv_check: -1,                              // #3870
            auto_enlarge_batch_size: false,             // #4303
            dd_zonotope_min_input_numel: None,          // #metaroom-ddzono (preset-scoped)
            dd_zonotope_max_k: None,                    // #metaroom-ddzono (preset-scoped)
            dd_zonotope_max_generators: None,           // #metaroom-ddzono (preset-scoped)
            dd_zonotope_collect_interm: None,           // #metaroom-ddzono (preset-scoped)
            phase_budget: PhaseBudgetConfig::default(), // #2206
        }
    }
}

impl BetaCrownConfig {
    /// Smallest admissible `forward_root_screen_max_mb` (#fwd-root-screen).
    pub const FORWARD_ROOT_SCREEN_MIN_MAX_MB: u32 = 64;
    /// Largest admissible `forward_root_screen_max_mb` (1 TiB).
    pub const FORWARD_ROOT_SCREEN_MAX_MAX_MB: u32 = 1 << 20;

    /// Whether `mb` is an admissible #fwd-root-screen ceiling (0 never is).
    #[must_use]
    pub const fn forward_root_screen_max_mb_is_valid(mb: u32) -> bool {
        mb >= Self::FORWARD_ROOT_SCREEN_MIN_MAX_MB && mb <= Self::FORWARD_ROOT_SCREEN_MAX_MAX_MB
    }

    /// Smallest admissible `vit_zono_root_max_mb` (#vit-zono-root).
    pub const VIT_ZONO_ROOT_MIN_MAX_MB: u32 = 64;
    /// Largest admissible `vit_zono_root_max_mb` (1 TiB).
    pub const VIT_ZONO_ROOT_MAX_MAX_MB: u32 = 1 << 20;
    /// Largest admissible `vit_zono_root_max_secs` (one hour).
    pub const VIT_ZONO_ROOT_MAX_MAX_SECS: f32 = 3600.0;

    /// Whether `mb` is an admissible #vit-zono-root ceiling (0 never is).
    #[must_use]
    pub const fn vit_zono_root_max_mb_is_valid(mb: u32) -> bool {
        mb >= Self::VIT_ZONO_ROOT_MIN_MAX_MB && mb <= Self::VIT_ZONO_ROOT_MAX_MAX_MB
    }

    /// Whether `secs` is an admissible #vit-zono-root wall cap (finite, > 0,
    /// at most an hour).
    #[must_use]
    pub fn vit_zono_root_max_secs_is_valid(secs: f32) -> bool {
        secs.is_finite() && secs > 0.0 && secs <= Self::VIT_ZONO_ROOT_MAX_MAX_SECS
    }

    /// Whether derived conjunctive input-split objectives have runtime verdict
    /// authority under this resolved configuration.
    ///
    /// This intentionally excludes property-shape checks, which remain at the
    /// graph frontend seam. Certificate-export runs decline until the external
    /// format can represent and replay the typed conic provenance.
    #[inline]
    pub fn input_split_conic_objective_eligible(&self) -> bool {
        self.input_split_conic_objective
            && self.verification_artifact_authority == VerificationArtifactAuthority::VerdictOnly
    }

    /// Resolve the proof-carrying kFSB simulation-reuse policy from its typed
    /// setting and exact environment override.
    ///
    /// This is the single production resolution point used by both the engine
    /// and effective-treatment reporting. A scored frontend's frozen override
    /// wins without reopening the environment. Interactive use layers exact
    /// `0`/`1` environment input over the typed setting and fails dark on every
    /// other present byte string.
    pub fn kfsb_cert_reuse_armed(&self) -> bool {
        self.kfsb_cert_reuse_runtime
            .unwrap_or_else(|| ny_levers::decls::kfsb::resolve_cert_reuse(self.kfsb_cert_reuse))
    }

    /// Whether cutting planes may influence certificate-bearing bounds.
    ///
    /// Cut proof authority is quarantined, and every consumer this gate used to
    /// guard has since been DELETED outright: the legacy post-hoc scalar
    /// contribution (applied after CROWN concretization instead of being folded
    /// through the backward relaxation), the `NY_CUT_FOLD` C2 registry fold, and
    /// the `arelu_cut` backward integration (the fold-through-the-relaxation
    /// variant). No bound-producing path reads the cut pool any more; the
    /// near-miss/proactive generators still create constraints without a proof.
    /// Keep this fail-closed gate separate from `enable_cuts` so even an
    /// internal caller that skips `validate()` cannot grant proof authority to
    /// the remaining cut machinery.
    ///
    /// Do NOT re-open this without a proof-producing, outward-rounded fold.
    #[inline]
    #[allow(dead_code)] // Deliberate quarantine seam, pinned directly by tests.
    pub(crate) fn cut_proof_authority_enabled(&self) -> bool {
        false
    }

    /// Whether CROWN backward should use Patches mode based on conv_mode and cuts.
    /// Reference: alpha-beta-CROWN `abcrown.py:228-231` — matrix when cuts enabled.
    #[inline]
    pub fn use_patches(&self) -> bool {
        self.conv_mode.use_patches(self.enable_cuts)
    }

    /// Resolve the Saturation-Escape Branching gate (#nn4sys-seb-dark): env
    /// `NY_SAT_ESCAPE_BRANCH=1` force-arms, `=0` force-disarms (the A/B kill
    /// switch), anything else falls back to [`Self::sat_escape_branch`] — the
    /// typed preset opt-in. Mirrors the `NY_MO_KFSB` /
    /// `use_kfsb_multi_branching` convention, and is the ONE resolution point
    /// for both consumers (the input-split SEB scorer gate and the disjunctive
    /// precheck budget cap), so the brancher and its budget can never disagree.
    pub fn sat_escape_branch_armed(&self) -> bool {
        match std::env::var("NY_SAT_ESCAPE_BRANCH").ok().as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => self.sat_escape_branch,
        }
    }

    /// Per-node CROWN-IBP time-budget overrides as the collector-facing type
    /// (#cgan-bn11-budget). All-`None` keeps the 2-second floor and
    /// selects the adaptive remaining-budget cap.
    #[inline]
    pub fn crown_ibp_per_node_time_budget(&self) -> crate::types::CrownIbpPerNodeTimeBudget {
        crate::types::CrownIbpPerNodeTimeBudget {
            floor_secs: self.crown_ibp_per_node_floor_secs,
            cap_secs: self.crown_ibp_per_node_cap_secs,
            prefix_cost_weighted: self.crown_ibp_prefix_cost_weighted,
            max_target_share_frac: self.crown_ibp_max_target_share_frac,
            floor_dim_scaled: self.crown_ibp_per_node_floor_dim_scaled,
        }
    }

    /// ACAS-Xu optimized configuration preset.
    ///
    /// Configures input splitting with relaxed clipping for the ACAS-Xu benchmark.
    /// These settings match the α,β-CROWN VNN-COMP configuration for ACAS-Xu.
    ///
    /// Key settings:
    /// - Input splitting (not neuron branching)
    /// - Relaxed clipping enabled for tighter bounds
    /// - PGD attack for counterexample finding
    /// - Large batch size for GPU throughput
    ///
    /// Reference: VNN-COMP 2024 winning configuration.
    ///
    /// Disables alpha-CROWN: the reference uses plain CROWN (`bound_prop_method:
    /// crown`) for ACAS-Xu. With frozen root alpha reuse, alpha-CROWN
    /// intermediate bounds are computed once on the full root domain and reused
    /// for all sub-domains. For ACAS-Xu's tight 5-dim input space, these root
    /// bounds are too loose to help sub-domains, producing 0% verification.
    /// Plain CROWN uses per-domain IBP intermediates that tighten naturally as
    /// domains split, matching the reference's behavior.
    ///
    /// Reference: alpha-beta-CROWN exp_configs/vnncomp21/acasxu.yaml:7
    ///   `bound_prop_method: crown` (no alpha optimization)
    pub fn acas_xu() -> Self {
        Self {
            branching_heuristic: BranchingHeuristic::InputSplit,
            // Plain CROWN, not alpha-CROWN. Reference uses `bound_prop_method: crown`.
            // Frozen root alpha regresses ACAS-Xu from 99.8% to 0% verified
            // because root intermediate bounds don't tighten for sub-domains.
            use_alpha_crown: false,
            use_forward_bounds: false,
            enable_relaxed_clip: true,
            enable_pgd_attack: true,
            pgd_restarts: 10_000,
            pgd_restart_when_stuck: true, // Reference: acasxu.yaml:20
            batch_size: 16_384,
            reorder_bab: true, // Reference: acasxu.yaml:13
            // Multi-dimensional input splitting: split the top 2 SB dims per parent,
            // producing up to 2^2=4 children that exactly cover the parent (BaB
            // completeness preserved). Mirrors alpha-beta-CROWN's storage_depth
            // batch-filling, but tuned empirically: on acasxu prop_1 (1_1/1_2/1_3),
            // depth=2 verifies in 29/105/85 domains (vs 32/234/184 at depth=1),
            // while depth>=3 regresses and depth=5 times out — the per-child screen
            // loop over-splits at high depth instead of cheaply batch-filling.
            input_split_depth: 2,
            ..Default::default()
        }
    }

    /// Build a runtime PGD config from the shared verifier settings.
    pub fn pgd_attack_config(
        &self,
        num_restarts: usize,
        num_steps: usize,
        deadline: Option<Instant>,
    ) -> PgdConfig {
        PgdConfig {
            num_restarts,
            num_steps,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true,
            deadline,
            restart_when_stuck: self.pgd_restart_when_stuck,
            initialization: self.pgd_initialization,
            osi_steps: self.pgd_osi_steps,
            gama_lambda: self.pgd_gama.then_some(GAMA_LAMBDA_DEFAULT),
            optimizer: self.pgd_optimizer,
            alpha_mode: self.pgd_alpha_mode,
            adam: AdamClippingParams {
                lr_decay: self.pgd_lr_decay,
                ..AdamClippingParams::default()
            },
            surrogate_sign_gradient: self.pgd_surrogate_sign_gradient,
            dense_low_dim_sweep: self.pgd_dense_low_dim_sweep,
            dense_sweep_max_dims: self.pgd_dense_sweep_max_dims,
            dense_sweep_points: self.pgd_dense_sweep_points,
            corner_seed_restarts: self.pgd_corner_seed_restarts,
            exact_vjp: self.pgd_exact_vjp,
            cancel: None,
            setup_regrant: None,
        }
    }

    /// Returns true when the domain bounds are sufficient to prove the property.
    #[inline]
    /// The `#lowdim-prover` lane's budget share, clamped to [0.05, 0.95];
    /// a non-finite configured value reads as the 0.5 default.
    /// The `#lowdim-prover` lane config this BaB config asks for.
    pub fn lowdim_config(&self) -> crate::lowdim::LowDimConfig {
        crate::lowdim::LowDimConfig {
            refine_max_width: self.input_split_lowdim_prover_refine_width,
            joint_quadratic: self.input_split_lowdim_joint_quadratic,
            split: self.input_split_lowdim_split,
            clip: self.input_split_lowdim_clip,
            parallel: if self.input_split_lowdim_parallel {
                crate::lowdim::LowDimParallel::Full
            } else {
                crate::lowdim::LowDimParallel::Bound
            },
            varying_dims: self.input_split_lowdim_varying_dims,
            alpha_iters: self.input_split_lowdim_alpha_iters,
            leaf_attack_every: self.input_split_lowdim_leaf_attack_every,
            shared_worklist: self.input_split_lowdim_shared_worklist,
            batch_engine: self.input_split_lowdim_batch_engine,
            engine: self.input_split_engine,
            ..crate::lowdim::LowDimConfig::default()
        }
    }

    pub fn lowdim_prover_share(&self) -> f32 {
        let share = self.input_split_lowdim_prover_share;
        if share.is_finite() {
            share.clamp(0.05, 0.95)
        } else {
            0.5
        }
    }

    pub(crate) fn domain_is_verified(&self, lower: f32, upper: f32, threshold: f32) -> bool {
        Self::domain_is_verified_for_mode(self.verify_upper_bound, lower, upper, threshold)
    }

    /// Returns true when the domain bounds are sufficient to prove violation.
    #[inline]
    pub(crate) fn domain_is_violation(&self, lower: f32, upper: f32, threshold: f32) -> bool {
        Self::domain_is_violation_for_mode(self.verify_upper_bound, lower, upper, threshold)
    }

    /// Mode-parametrized verification check used by shared dispatch/check helpers.
    ///
    /// Returns `false` when any input is non-finite (NaN or Inf). NaN bounds
    /// must never produce a Verified result. Inf bounds indicate propagation
    /// failure (e.g., reciprocal zero-crossing), not genuine verification.
    /// Inf threshold would trivially verify/reject all domains.
    /// A single `is_finite()` check covers both NaN and Inf per IEEE 754.
    #[inline]
    pub(crate) fn domain_is_verified_for_mode(
        verify_upper_bound: bool,
        lower: f32,
        upper: f32,
        threshold: f32,
    ) -> bool {
        // Reject all non-finite inputs: NaN (propagation corruption) and Inf
        // (propagation failure, e.g., reciprocal zero-crossing). Inf threshold
        // would make every domain trivially verified or never verified.
        if !lower.is_finite() || !upper.is_finite() || !threshold.is_finite() || lower > upper {
            return false;
        }
        if verify_upper_bound {
            upper < threshold
        } else {
            lower > threshold
        }
    }

    /// Mode-parametrized violation check used by shared dispatch/check helpers.
    ///
    /// Returns `false` when any input is non-finite (NaN or Inf). NaN bounds
    /// must never produce a Violation result. Inf bounds indicate propagation
    /// failure, not a genuine violation. Inf threshold would make every domain
    /// trivially violated or never violated.
    ///
    /// Also returns `false` for an INVERTED interval (`lower > upper`, #violdrop).
    /// This predicate is a PROOF — the decision-relevant bound is being asserted
    /// to hold over the whole region — and an inverted interval is numerically
    /// contradictory: a valid upper bound can never sit below a valid lower
    /// bound, so at most one of the two is sound and neither can be trusted as
    /// the proof side. Inversions DO occur here in practice (see
    /// `multi_objective/shared.rs::tighten_child_bounds_with_parent`, which detects and
    /// tolerates them, and the α-CROWN elementwise best-bound merge, which
    /// repairs them via `repair_inverted_bounds_nd`). Rejecting them makes this
    /// predicate strictly harder to satisfy: fewer drops, more search, never a
    /// verdict the previous predicate would not also have allowed.
    #[inline]
    pub(crate) fn domain_is_violation_for_mode(
        verify_upper_bound: bool,
        lower: f32,
        upper: f32,
        threshold: f32,
    ) -> bool {
        // Reject all non-finite inputs.
        if !lower.is_finite() || !upper.is_finite() || !threshold.is_finite() {
            return false;
        }
        // Reject inverted intervals (#violdrop).
        if lower > upper {
            return false;
        }
        if verify_upper_bound {
            lower >= threshold
        } else {
            upper < threshold
        }
    }

    /// Returns BaB queue priority for a domain with given bounds.
    ///
    /// Higher priority = process first. When `verify_upper_bound` is true, we want
    /// to process domains with the highest upper bound first (worst case). When false,
    /// we negate the lower bound so the max-heap orders by lowest lower bound first.
    #[inline]
    pub(crate) fn domain_priority(&self, lower: f32, upper: f32) -> Result<f32> {
        Self::domain_priority_for_mode(self.verify_upper_bound, lower, upper)
    }

    /// Mode-parametrized BaB queue priority.
    ///
    /// Returns `Err(NumericalInstability)` if either bound is non-finite, preventing
    /// zombie domains from entering the BaB queue.
    #[inline]
    pub(crate) fn domain_priority_for_mode(
        verify_upper_bound: bool,
        lower: f32,
        upper: f32,
    ) -> Result<f32> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "BaB domain bounds are non-finite: lower={lower}, upper={upper}"
            )));
        }
        if lower > upper {
            return Err(NyError::NumericalInstability(format!(
                "BaB domain bounds are inverted: lower={lower}, upper={upper}"
            )));
        }
        Ok(if verify_upper_bound { upper } else { -lower })
    }

    /// Returns the relevant bound for threshold comparison.
    ///
    /// When `verify_upper_bound` is true, the upper bound is the decision-relevant
    /// bound (checking upper < threshold). When false, the lower bound is relevant
    /// (checking lower > threshold).
    #[inline]
    pub(crate) fn relevant_bound(&self, lower: f32, upper: f32) -> f32 {
        if self.verify_upper_bound {
            upper
        } else {
            lower
        }
    }

    /// Compute tightening gain from parent to child domain.
    ///
    /// Measures how much the relevant bound improved (moved toward threshold).
    #[inline]
    pub(crate) fn bound_gain(
        &self,
        parent_lower: f32,
        parent_upper: f32,
        child_lower: f32,
        child_upper: f32,
    ) -> f32 {
        // NaN-safe: propagate NaN instead of silently reporting 0.0 gain
        if self.verify_upper_bound {
            nan_propagating_max(parent_upper - child_upper, 0.0)
        } else {
            nan_propagating_max(child_lower - parent_lower, 0.0)
        }
    }

    /// Human-readable verification direction for logging.
    ///
    /// Returns `"upper < threshold"` when verifying upper bound,
    /// `"lower > threshold"` when verifying lower bound.
    #[inline]
    pub(crate) fn verification_direction_str(&self) -> &'static str {
        if self.verify_upper_bound {
            "upper < threshold"
        } else {
            "lower > threshold"
        }
    }

    /// Counterexample direction symbol for logging.
    ///
    /// Returns `">="` when `verify_upper_bound` (counterexample is output >= threshold),
    /// `"<="` otherwise (counterexample is output <= threshold).
    #[inline]
    pub(crate) fn violation_direction_str(&self) -> &'static str {
        if self.verify_upper_bound {
            ">="
        } else {
            "<="
        }
    }

    /// Compatibility alias for BaB queue priority used by legacy call sites.
    ///
    /// Queue ordering must respect the active verification mode:
    /// - `verify_upper_bound=true`: prioritize larger upper bounds
    /// - `verify_upper_bound=false`: prioritize smaller lower bounds
    ///
    /// This now matches [`domain_priority`] exactly so sequential and graph BaB
    /// paths cannot diverge due to helper selection.
    #[inline]
    pub(crate) fn violation_priority(&self, lower: f32, upper: f32) -> Result<f32> {
        Self::domain_priority_for_mode(self.verify_upper_bound, lower, upper)
    }

    /// Extract the worst-case FSB score from two child branches.
    ///
    /// For each branch, extracts the relevant bound (upper when `verify_upper_bound`,
    /// lower otherwise), then returns the negated worst (max upper / min lower).
    /// Returns `None` if both branches are `None`.
    #[inline]
    pub(crate) fn fsb_worst_case_score(
        &self,
        active: Option<(f32, f32)>,
        inactive: Option<(f32, f32)>,
    ) -> Option<f32> {
        if self.verify_upper_bound {
            let mut worst_upper = f32::NEG_INFINITY;
            if let Some((_, u)) = active {
                worst_upper = worst_upper.max(u);
            }
            if let Some((_, u)) = inactive {
                worst_upper = worst_upper.max(u);
            }
            if worst_upper == f32::NEG_INFINITY {
                None
            } else {
                Some(-worst_upper)
            }
        } else {
            let mut worst_lower = f32::INFINITY;
            if let Some((l, _)) = active {
                worst_lower = worst_lower.min(l);
            }
            if let Some((l, _)) = inactive {
                worst_lower = worst_lower.min(l);
            }
            if worst_lower == f32::INFINITY {
                None
            } else {
                Some(worst_lower)
            }
        }
    }

    /// Extract the relevant bound value from child domain bounds for kFSB scoring.
    ///
    /// Returns `-upper` when `verify_upper_bound`, `lower` otherwise.
    /// Returns `f32::NEG_INFINITY` when bounds are `None`.
    #[inline]
    pub(crate) fn child_bound_value(&self, bounds: Option<(f32, f32)>) -> f32 {
        match bounds {
            Some((lower, upper)) => {
                if self.verify_upper_bound {
                    -upper
                } else {
                    lower
                }
            }
            None => f32::NEG_INFINITY,
        }
    }

    /// Whether typed depth-two kFSB has execution authority on this BaB wave.
    ///
    /// This pure resolver is shared by the engine and effective-treatment
    /// projection. Legacy adaptive-depth and f64 observer variables are
    /// intentionally absent: they are advice-only and never veto typed policy.
    /// When typed work receives a budget, its authority has priority and the
    /// legacy observations remain eligible for later scheduling.
    #[inline]
    pub fn depth_two_lookahead_supported_at_round(
        &self,
        bab_round: usize,
        effective_candidates: usize,
        effective_reduce_op: KfsbReduceOp,
    ) -> bool {
        self.depth_two_branch_lookahead.enabled_at_round(bab_round)
            && self.depth_two_branch_lookahead.validate().is_ok()
            && matches!(self.branching_heuristic, BranchingHeuristic::Kfsb)
            // Shared historical prep ranks raw lower bounds. Reinterpreting it
            // in upper mode would violate Off/Shadow identity until a separate
            // direction-normalized `-upper` prep exists.
            && !self.verify_upper_bound
            && effective_reduce_op == KfsbReduceOp::Min
            && effective_candidates > 0
    }

    /// Validate all learning rate and optimizer configuration fields.
    ///
    /// Checks top-level `beta_lr`, `alpha_lr`, and delegates to
    /// `adaptive_config.validate()` for Adam hyperparameters.
    /// Call after CLI flag application and preset merging.
    pub fn validate(&self) -> Result<()> {
        self.pgd_exact_vjp
            .validate()
            .map_err(NyError::InvalidConfig)?;
        if let Some(refine) = self.mip_refine_intermediates {
            refine.validate().map_err(NyError::InvalidConfig)?;
        }
        if let Some(pass) = self.gpu_patches_crown {
            pass.validate().map_err(NyError::InvalidConfig)?;
        }
        if !Self::forward_root_screen_max_mb_is_valid(self.forward_root_screen_max_mb) {
            return Err(NyError::InvalidConfig(format!(
                "forward_root_screen_max_mb must be in {}..={} (0 is never \"unlimited\"), got {}",
                Self::FORWARD_ROOT_SCREEN_MIN_MAX_MB,
                Self::FORWARD_ROOT_SCREEN_MAX_MAX_MB,
                self.forward_root_screen_max_mb
            )));
        }
        if !Self::vit_zono_root_max_mb_is_valid(self.vit_zono_root_max_mb) {
            return Err(NyError::InvalidConfig(format!(
                "vit_zono_root_max_mb must be in {}..={} (0 is never \"unlimited\"), got {}",
                Self::VIT_ZONO_ROOT_MIN_MAX_MB,
                Self::VIT_ZONO_ROOT_MAX_MAX_MB,
                self.vit_zono_root_max_mb
            )));
        }
        if !Self::vit_zono_root_max_secs_is_valid(self.vit_zono_root_max_secs) {
            return Err(NyError::InvalidConfig(format!(
                "vit_zono_root_max_secs must be finite and in (0, {}], got {}",
                Self::VIT_ZONO_ROOT_MAX_MAX_SECS,
                self.vit_zono_root_max_secs
            )));
        }
        if Instant::now().checked_add(self.timeout).is_none() {
            return Err(NyError::InvalidConfig(format!(
                "timeout {:?} is too large for the platform monotonic clock",
                self.timeout
            )));
        }
        if self.enable_cuts || self.enable_near_miss_cuts || self.enable_proactive_cuts {
            return Err(NyError::InvalidConfig(
                "cut proof authority is quarantined: the legacy post-hoc scalar \
                 contribution is not a certified GCP-CROWN fold, and near-miss/proactive \
                 cuts are not proof-derived"
                    .to_string(),
            ));
        }
        if self.use_alpha_crown && self.use_forward_bounds {
            return Err(NyError::InvalidConfig(
                "use_forward_bounds cannot be enabled together with use_alpha_crown".to_string(),
            ));
        }
        if !self.beta_lr.is_finite() || self.beta_lr < 0.0 {
            return Err(NyError::InvalidConfig(format!(
                "beta_lr must be finite and >= 0, got {}",
                self.beta_lr
            )));
        }
        if !self.alpha_lr.is_finite() || self.alpha_lr < 0.0 {
            return Err(NyError::InvalidConfig(format!(
                "alpha_lr must be finite and >= 0, got {}",
                self.alpha_lr
            )));
        }
        if self.build_batch_size == Some(0) {
            return Err(NyError::InvalidConfig(
                "build_batch_size must be >= 1 when set".to_string(),
            ));
        }
        if self.input_split_conic_queue_refresh_batch_size == 0 {
            return Err(NyError::InvalidConfig(
                "input_split_conic_queue_refresh_batch_size must be >= 1".to_string(),
            ));
        }
        if self.atomic_root_c_margin_iterations > ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS {
            return Err(NyError::InvalidConfig(format!(
                "atomic_root_c_margin_iterations must be <= {}, got {}",
                ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS, self.atomic_root_c_margin_iterations
            )));
        }
        if self.input_split_depth == 0
            && matches!(self.branching_heuristic, BranchingHeuristic::InputSplit)
        {
            // Depth 0 selects no split dimensions, so every domain is
            // unsplittable and input-split BaB can make no progress.
            return Err(NyError::InvalidConfig(
                "input_split_depth must be >= 1 with the InputSplit branching heuristic"
                    .to_string(),
            ));
        }
        if self.enable_clip_interm_domain && self.clip_interm_prune {
            return Err(NyError::InvalidConfig(
                "clip_interm_prune is quarantined for certificate-backed Complete Clip; \
                 pruning authority is not implemented"
                    .to_string(),
            ));
        }
        if self.input_split_fresh_domain_clip
            && (!matches!(self.branching_heuristic, BranchingHeuristic::InputSplit)
                || !self.reorder_bab
                || !self.input_split_ibp_enhancement
                || !self.enable_relaxed_clip
                || !matches!(self.input_clip_type, InputClipType::Relaxed)
                || self.relaxed_clip_iterations == 0)
        {
            return Err(NyError::InvalidConfig(
                "input_split_fresh_domain_clip requires InputSplit branching, reorder_bab, \
                 input_split_ibp_enhancement, enabled relaxed clipping, relaxed clip type, \
                 and at least one relaxed clip iteration"
                    .to_string(),
            ));
        }
        self.phase_budget.validate()?;
        self.depth_two_branch_lookahead.validate()?;
        self.adaptive_config.validate()
    }
}

/// #cost-observer: preset surface for the self-calibrating per-child bound
/// allowance (`bab.branching.input_split.child_bound_calibration`).
///
/// Every field is OPTIONAL and every field is DIMENSIONLESS. That is the whole
/// contract: an omitted field falls back to
/// [`ny_core::cost_observer::SlicePolicy`]'s default, and no field carries a
/// unit, so nothing here has to know how long a bound takes, how big the budget
/// is, or how fast the host is. Those three facts arrive by measurement.
///
/// The block's PRESENCE is the arming switch. Absent — the shipped state —
/// leaves `input_split_child_bound_max_secs` in charge, unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ChildBoundCalibrationConfig {
    /// How many times the MEASURED median cost of a completed call another call
    /// may run before it is cut off. Omitted = 160 (the safe end of the
    /// `[154, 538]` interval that reproduces the one row where a hand-set cap
    /// was measured good: 0.13 s median, healthy at 20 s, tail from 70 s).
    #[serde(default)]
    pub headroom: Option<f64>,
    /// Largest share of the REMAINING wall one call may take once calibrated.
    /// Omitted = 0.25. The `alpha_opt` / `#phase-window` starvation guard,
    /// against the live remaining budget rather than a fixed number of seconds.
    #[serde(default)]
    pub max_share_of_remaining: Option<f64>,
    /// Share of the remaining wall spent on the probe that finds out which
    /// regime this row is in, before any call has completed. Omitted = 0.05 —
    /// small, because on a row where every call is expensive the probe is pure
    /// overhead before the observer falls back to the caller's own ceiling.
    #[serde(default)]
    pub probe_share: Option<f64>,
    /// Completed observations required before the measured term may govern.
    /// Omitted = 8.
    #[serde(default)]
    pub min_samples: Option<usize>,
    /// Completed observations retained in the sliding window. Omitted = 128.
    #[serde(default)]
    pub window: Option<usize>,
    /// Multiplicative growth per CONSECUTIVE truncated call — how fast the
    /// observer corrects an allowance that is too tight. Omitted = 2.0.
    #[serde(default)]
    pub growth: Option<f64>,
}

impl ChildBoundCalibrationConfig {
    /// Resolve to the runtime policy, field by field, defaulting anything the
    /// preset did not name. Sanitization of hostile values happens once more
    /// inside `CostObserver::new`, so this cannot hand a panicking value on.
    #[must_use]
    pub fn to_policy(self) -> ny_core::cost_observer::SlicePolicy {
        let d = ny_core::cost_observer::SlicePolicy::default();
        ny_core::cost_observer::SlicePolicy {
            headroom: self.headroom.unwrap_or(d.headroom),
            max_share_of_remaining: self
                .max_share_of_remaining
                .unwrap_or(d.max_share_of_remaining),
            probe_share: self.probe_share.unwrap_or(d.probe_share),
            min_samples: self.min_samples.unwrap_or(d.min_samples),
            window: self.window.unwrap_or(d.window),
            growth: self.growth.unwrap_or(d.growth),
            growth_steps_max: d.growth_steps_max,
        }
    }
}

/// #iso-edge-cap: the per-wave edge-work allowance,
/// `bab.branching.input_split.edge_wave_cap`
/// ([`BetaCrownConfig::input_split_edge_wave_cap`]).
///
/// The block's PRESENCE is the arming switch; absent (the shipped state) the
/// edge work is ungoverned, exactly as before. Every field is optional: an
/// omitted field takes the WAVE6_PLAN §7.C6 default, resolved (and sanitized —
/// a non-finite or negative value fails open to the dark state) by the
/// grouped-disjunctive lane itself, so this type carries only what the preset
/// named.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct EdgeWaveCapConfig {
    /// Hard ceiling on one wave's edge wall, in seconds. Omitted = 2.0. Zero
    /// is honoured and means "no edge work at all".
    #[serde(default)]
    pub max_secs: Option<f64>,
    /// Share of the batched rebound that precedes the wave, dimensionless.
    /// Omitted = 0.25.
    #[serde(default)]
    pub rebound_share: Option<f64>,
}

/// Hard resource cap for exact full-`C` root margin iterations.
pub const ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS: usize = 8;

/// Hard resource bounds for the July-2026 depth-2 lookahead experiment.
pub const DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES: usize = 15;
pub const DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS: usize = 5;

/// Authority granted to branch-specific depth-2 lookahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DepthTwoBranchLookaheadMode {
    /// Preserve the historical one-step kFSB path byte-for-byte.
    #[default]
    Off,
    /// Compute and report complete advice, but retain the one-step winner.
    Shadow,
    /// Allow complete, revalidated advice to choose a first-level root split.
    Select,
}

/// Typed policy for the bounded July-2026 Lookahead Branching experiment.
///
/// Depth is intentionally fixed at two. The defaults reproduce the published
/// alpha-beta-CROWN setting (15 BaBSR candidates, first five BaB rounds,
/// discount λ=0.5) while leaving the experiment disabled. Unknown serialized
/// keys are rejected deliberately: silently accepting a misspelled authority
/// or resource bound would make a sealed experiment's treatment ambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DepthTwoBranchLookaheadConfig {
    pub mode: DepthTwoBranchLookaheadMode,
    pub candidates: usize,
    pub top_rounds: usize,
    pub discount: f64,
}

impl Default for DepthTwoBranchLookaheadConfig {
    fn default() -> Self {
        Self {
            mode: DepthTwoBranchLookaheadMode::Off,
            candidates: DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES,
            top_rounds: DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS,
            discount: 0.5,
        }
    }
}

impl DepthTwoBranchLookaheadConfig {
    /// Whether the typed experiment may run on this canonical outer BaB wave.
    #[inline]
    pub fn enabled_at_round(self, bab_round: usize) -> bool {
        self.mode != DepthTwoBranchLookaheadMode::Off && bab_round < self.top_rounds
    }

    fn validate(self) -> Result<()> {
        if !(1..=DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES).contains(&self.candidates) {
            return Err(NyError::InvalidConfig(format!(
                "depth_two_branch_lookahead.candidates must be in 1..={}, got {}",
                DEPTH_TWO_LOOKAHEAD_MAX_CANDIDATES, self.candidates
            )));
        }
        if !(1..=DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS).contains(&self.top_rounds) {
            return Err(NyError::InvalidConfig(format!(
                "depth_two_branch_lookahead.top_rounds must be in 1..={}, got {}",
                DEPTH_TWO_LOOKAHEAD_MAX_ROUNDS, self.top_rounds
            )));
        }
        if !self.discount.is_finite() || !(0.0..=1.0).contains(&self.discount) {
            return Err(NyError::InvalidConfig(format!(
                "depth_two_branch_lookahead.discount must be finite and in [0, 1], got {}",
                self.discount
            )));
        }
        Ok(())
    }
}

/// Reduce operation for combining branch scores in kFSB.
/// Per alpha-beta-CROWN: min=conservative, max=optimistic, mean=balanced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum KfsbReduceOp {
    /// Conservative: take worst-case of both branches (default).
    #[default]
    Min,
    /// Optimistic: take best-case of both branches.
    Max,
    /// Balanced: average of both branches.
    Mean,
}

/// Input-domain clipping algorithm type.
///
/// Controls how input bounds are tightened using CROWN linear constraints
/// during input-split branch-and-bound.
///
/// - `Relaxed`: Closed-form 1D updates per dimension (fast, axis-aligned).
/// - `Complete`: LP-optimal tightening via Lagrangian dual coordinate ascent.
///   Tighter than relaxed because it accounts for cross-constraint dependencies.
///
/// Reference: alpha-beta-CROWN `clip_input_domain.clip_type`
/// (`abcrown_all_params.yaml:190`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InputClipType {
    /// Closed-form relaxed clipping (matches alpha-beta-CROWN `clip_type: relaxed`).
    #[default]
    Relaxed,
    /// LP-optimal complete clipping via Lagrangian dual (matches `clip_type: complete`).
    Complete,
}

/// Convolution backward mode for CROWN propagation.
///
/// Controls whether Conv2d backward passes use Patches (per-position receptive
/// field) or Matrix (dense A-coefficient) representation.
///
/// Reference: alpha-beta-CROWN `general.conv_mode` (`abcrown.py:228-231`):
/// the reference forces Matrix mode when cuts are enabled because cutting
/// planes operate on flattened neurons and require dense A-matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ConvMode {
    /// Auto-select: Patches when cuts are disabled, Matrix when cuts are enabled.
    /// Matches reference `abcrown.py:228-231` policy.
    #[default]
    Auto,
    /// Always use Patches mode (per-position receptive field composition).
    Patches,
    /// Always use Matrix mode (dense A-coefficient backward).
    Matrix,
}

impl ConvMode {
    /// Returns `true` if the CROWN backward should use Patches mode.
    ///
    /// - `Patches` → always true
    /// - `Matrix` → always false
    /// - `Auto` → true unless cuts are enabled (reference: `abcrown.py:228-231`)
    pub fn use_patches(self, cuts_enabled: bool) -> bool {
        match self {
            Self::Patches => true,
            Self::Matrix => false,
            Self::Auto => !cuts_enabled,
        }
    }
}

/// #hull-root / #hull-bab (wave-6 C4): how far the certified
/// single-hidden-layer per-neuron hull runs in the sequential relational lane.
///
/// The enum is CLOSED: `off | root | bab`, snake_case; anything else fails to
/// parse, so a misspelled arming can never silently resolve to `off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SingleHiddenHullMode {
    /// Historical behavior, byte-identical: the lane never runs.
    #[default]
    Off,
    /// One certified hull bound at the root; `unsat` iff it is `> 0`.
    Root,
    /// Phase 2 (#hull-bab): the root bound, then a deadline-bounded,
    /// best-first ReLU-split branch-and-bound over the unstable concave
    /// (`c_j < 0`) neurons, every node certified by the closed form plus one
    /// weak-duality term per split halfspace. `unsat` iff EVERY leaf is
    /// certified `> 0`; it never publishes `sat`. Undecided rows fall through
    /// to the historical lane with the wall the tree did not spend.
    /// Observable: `[hull-bab] armed mode=bab ...` on every armed run.
    Bab,
}

impl SingleHiddenHullMode {
    /// Whether the hull lane is armed at all.
    #[must_use]
    pub const fn is_armed(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Stable lowercase label for observables.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Root => "root",
            Self::Bab => "bab",
        }
    }
}

/// #vit-v3-bankable: the typed `bab.mo_cover_certify` policy (see
/// `BetaCrownConfig::mo_cover_certify`).
///
/// The enum is CLOSED: `off | ledger | replay`, snake_case; anything else
/// fails to parse, so a misspelled arming can never silently resolve to `off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MoCoverCertify {
    /// Historical behavior, byte-identical: no frame, no replay.
    #[default]
    Off,
    /// The per-row completeness ledger (refuse on a cover gap; token when
    /// proved; legacy door when unbuildable).
    Ledger,
    /// The ledger plus the certify-once replay of every witness leaf; a
    /// `Verified` stands only when both pass.
    Replay,
}

impl MoCoverCertify {
    /// The preset vocabulary, in declaration order.
    pub const TOKENS: [&'static str; 3] = ["off", "ledger", "replay"];

    /// Whether the ledger is armed (both `ledger` and `replay`).
    #[must_use]
    pub const fn is_armed(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Whether the certify-once replay gates the verdict.
    #[must_use]
    pub const fn replays(self) -> bool {
        matches!(self, Self::Replay)
    }

    /// Stable lowercase label (the preset token).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Ledger => "ledger",
            Self::Replay => "replay",
        }
    }

    /// Parse an exact preset token.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        match token {
            "off" => Some(Self::Off),
            "ledger" => Some(Self::Ledger),
            "replay" => Some(Self::Replay),
            _ => None,
        }
    }
}

/// #graph-lane-search (ENGINE_PROGRAM §2 M7): the typed `bab.graph_search`
/// arming — which untrusted batched α/β search backend the graph β-CROWN lane
/// consults (see `BetaCrownConfig::graph_search`).
///
/// The enum is CLOSED: `off | cpu_f32`, snake_case; anything else fails to
/// parse, so a misspelled arming can never silently resolve to `off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GraphLaneSearch {
    /// Historical behavior, byte-identical: no backend is consulted.
    #[default]
    Off,
    /// The injected CPU f32 backend (`ny_bbe::CpuF32Backend::graph_bab_batch`,
    /// registered by ny-cli), certified per domain by the lane.
    CpuF32,
}

impl GraphLaneSearch {
    /// The preset vocabulary, in declaration order.
    pub const TOKENS: [&'static str; 2] = ["off", "cpu_f32"];

    /// Whether any search is armed.
    #[must_use]
    pub const fn is_armed(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Stable lowercase label for observables (the preset token).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::CpuF32 => "cpu_f32",
        }
    }
}

/// #relusplitter-lp-refine (wave-6 C7): the typed
/// `bab.mip_refine_intermediates` arming. Presence arms it; both fields have
/// Vibecheck-recipe defaults. Unknown fields are refused, so a misspelled
/// sub-key can never silently resolve to a default.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MipRefineIntermediates {
    /// Deepest hidden pre-activation layer (0-based) the LP refines; layers
    /// `1..=layers` are refined (layer 0 is exact already). Default 3.
    #[serde(default = "default_mip_refine_layers")]
    pub layers: usize,
    /// Share of the remaining root tightening window the refiner may spend,
    /// in `(0, MAX_BUDGET_FRAC]`. Default 0.3.
    #[serde(default = "default_mip_refine_budget_frac")]
    pub budget_frac: f64,
}

impl Default for MipRefineIntermediates {
    fn default() -> Self {
        Self {
            layers: default_mip_refine_layers(),
            budget_frac: default_mip_refine_budget_frac(),
        }
    }
}

impl MipRefineIntermediates {
    /// Largest admissible `layers` (a deeper request is refused, not clamped).
    pub const MAX_LAYERS: usize = 64;
    /// Largest admissible `budget_frac`: the BaB always keeps a share.
    pub const MAX_BUDGET_FRAC: f64 = 0.9;

    /// Refuse an out-of-range arming instead of clamping it.
    ///
    /// # Errors
    /// `layers` outside `1..=MAX_LAYERS`, or `budget_frac` not a finite value
    /// in `(0, MAX_BUDGET_FRAC]`.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.layers == 0 || self.layers > Self::MAX_LAYERS {
            return Err(format!(
                "mip_refine_intermediates.layers must be in 1..={}, got {}",
                Self::MAX_LAYERS,
                self.layers
            ));
        }
        if !(self.budget_frac.is_finite()
            && self.budget_frac > 0.0
            && self.budget_frac <= Self::MAX_BUDGET_FRAC)
        {
            return Err(format!(
                "mip_refine_intermediates.budget_frac must be finite in (0, {}], got {}",
                Self::MAX_BUDGET_FRAC,
                self.budget_frac
            ));
        }
        Ok(())
    }
}

fn default_mip_refine_layers() -> usize {
    3
}

fn default_mip_refine_budget_frac() -> f64 {
    0.3
}

fn default_lowdim_prover_share() -> f32 {
    0.5
}

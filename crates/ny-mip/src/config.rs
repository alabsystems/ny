// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// MIP backend selection.
///
/// SOLVER POLICY (docs/SOLVER_POLICY.md): all solving in ny happens on
/// ay (or ny's own native engines). `Ay` is the default and the only
/// production backend. The encoder targets the solver-neutral
/// [`crate::ir::MilpProblem`]; the backend only decides which solver the IR
/// is lowered to at solve time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MipBackend {
    /// ay in-process (the `ay-milp` library): typed exact lowering, no
    /// subprocess, certificates as data. R3 of the ay-as-library plan
    /// (ay repo, designs/2026-07-12-ay-as-library-for-ny.md).
    #[default]
    Ay,
    /// ay external binary via `$NY_AY`/`$PATH` — the frozen P0 subprocess
    /// lane (exact QF_LRA text lowering). Debug/bootstrap only: builds ny
    /// without compiling ay, and feeds the mip-diff lib-vs-proc gate during
    /// the R3 transition.
    AyProc,
}

/// Configuration for the MIP/LP solver.
#[derive(Debug, Clone, Copy)]
pub struct MipConfig {
    /// Which MIP solver backend to lower the IR to.
    pub backend: MipBackend,
    /// Parallel phase-split racing width (designs/scip.md Phase C).
    ///
    /// Feasibility checks split the 2^k assignments of the k widest unstable
    /// ReLU indicator binaries into independent subproblems solved
    /// concurrently (one solver per thread, built from the shared IR).
    /// `0` = auto (use `std::thread::available_parallelism()`), `1` = disabled
    /// (single serial solve — the disable path per the default-on principle),
    /// `n > 1` = race up to `n` subproblems.
    ///
    /// SOUND: the {0,1}^k assignments exactly partition the binary space, so
    /// any-Sat => Sat and all-Unsat => Unsat; anything else aggregates to
    /// Timeout (never Unsat).
    pub parallel_split: usize,
    /// Timeout per solve call in seconds.
    pub timeout_secs: f64,
    /// Maximum neurons to tighten per layer (0 = all unstable neurons).
    ///
    /// Limits the number of LP solves per layer for performance control.
    /// When >0, only the first N unstable neurons are tightened.
    pub max_tighten_per_layer: usize,
    /// Run LP bound tightening on IBP bounds before MIP encoding.
    ///
    /// LP tightening solves per-neuron LP relaxations (triangle relaxation,
    /// no binary indicators) to produce tighter Big-M values for the MIP
    /// encoding. Tighter Big-M → tighter LP relaxation at B&B nodes →
    /// faster branch-and-bound. Also fixes some unstable neurons as stable,
    /// eliminating binary variables entirely.
    ///
    /// Overhead is small (LP solves, not MIP) and amortized across all
    /// disjunctive clauses. Recommended for UNSAT-heavy benchmarks where
    /// MIP must exhaust the B&B tree (e.g., sat_relu).
    ///
    /// Reference: alpha-beta-CROWN complete_verifier/lp_mip_solver/bounds_core.py
    pub lp_tighten: bool,
    /// Optimization-based bound tightening (OBBT) rounds per layer (P3).
    ///
    /// `0` (default) keeps the single-pass per-neuron min/max tighten. When
    /// `> 0`, a layer's unstable pre-activation columns are tightened as a
    /// COUPLED set on one `ay_milp::LpSession`: each committed bound is a
    /// proven rigorous bound, so tightening one neuron's box can tighten a
    /// sibling that shares the previous layer's variables — a fixpoint the
    /// independent per-neuron pass cannot reach. Every committed bound has
    /// the SAME soundness contract (rigorous, outward-rounded); OBBT only
    /// ever produces bounds at least as tight, never looser.
    ///
    /// Reference: ay `crates/ay-milp/design/P3-SPEC.md` (`LpSession::obbt`).
    pub obbt_rounds: usize,
}

impl Default for MipConfig {
    fn default() -> Self {
        Self {
            backend: MipBackend::default(), // ay: the only production solver
            parallel_split: 0,              // auto: split across available cores
            timeout_secs: 300.0,            // VNN-COMP standard
            max_tighten_per_layer: 0,       // tighten all unstable neurons
            lp_tighten: false,
            obbt_rounds: 0, // single-pass per-neuron tighten unless opted in
        }
    }
}

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

/// Typed feasibility-session ingress policy.
///
/// The default preserves the historical environment-canary behavior.  The
/// required variants are reserved for a caller that has already admitted a
/// narrowly scoped direct-MIP experiment and must not silently fall through
/// to NY's cloned phase-split or serial feasibility routes if the requested
/// shared AY session cannot be formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MipFeasibilityIngress {
    /// Preserve the historical/default feasibility route, including the
    /// exact `NY_MIP_SAFENLP_SHARED_PREFIX=1` canary when it is admissible.
    #[default]
    Historical,
    /// Require one AY shared-binary-prefix session.  Any backend admission
    /// decline is a solver error/unknown under the caller's existing clock;
    /// no alternate MIP feasibility session may be launched.
    RequireSafeNlpSharedBinaryPrefix,
    /// Require one AY marked-margin shared-binary-prefix session.
    ///
    /// The caller must have explicitly marked its unique one-sided unsafe row.
    /// A missing marker, backend decline, or incomplete solve is an
    /// error/unknown under the existing clock; it never falls through to the
    /// plain shared-prefix API or another feasibility session.
    RequireSafeNlpMarkedMarginSharedBinaryPrefix,
}

/// Configuration for the MIP/LP solver.
#[derive(Debug, Clone, Copy)]
pub struct MipConfig {
    /// Which MIP solver backend to lower the IR to.
    pub backend: MipBackend,
    /// How the final feasibility session must enter the backend.
    ///
    /// This is typed caller-local state rather than an environment mutation.
    /// LP tightening and all historical callers leave it at
    /// [`MipFeasibilityIngress::Historical`].
    pub feasibility_ingress: MipFeasibilityIngress,
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
    /// Per-node warm LP ceiling for AY's branch-and-bound engine.
    ///
    /// `None` is the historical/default behavior. A finite duration asks AY
    /// to stop a stalled warm attempt at the earlier of this ceiling and the
    /// outer solve deadline, then retry that node cold exactly once. The
    /// value is typed and scoped to one solver session; it never extends the
    /// outer deadline.
    ///
    /// Only exact neural Graph-MIP callers should set this field. LP
    /// tightening, certified-linear proof replay, and other authority paths
    /// leave it `None`.
    pub ay_node_warm_time_limit: Option<std::time::Duration>,
    /// Absolute caller-owned wall deadline for AY solves.
    ///
    /// `None` is the historical/default policy: a serial window-class solve
    /// may raise [`Self::timeout_secs`] to `NY_MIP_WINDOW_TIMEOUT_SECS`.
    /// `Some(deadline)` is a lexically scoped hard cap: every AY feasibility
    /// solve, phase-split worker, cached re-solve, and optimization ends at
    /// the earlier of this instant and its relative `timeout_secs`, and the
    /// window floor is not reapplied.
    ///
    /// This is used when a caller carves preliminary work out of the same
    /// historical solve envelope.  It is typed process-local state rather
    /// than an environment mutation, and therefore cannot leak into an
    /// unrelated solver.
    pub ay_hard_deadline: Option<std::time::Instant>,
}

impl Default for MipConfig {
    fn default() -> Self {
        Self {
            backend: MipBackend::default(), // ay: the only production solver
            feasibility_ingress: MipFeasibilityIngress::default(),
            parallel_split: 0,        // auto: split across available cores
            timeout_secs: 300.0,      // VNN-COMP standard
            max_tighten_per_layer: 0, // tighten all unstable neurons
            lp_tighten: false,
            obbt_rounds: 0, // single-pass per-neuron tighten unless opted in
            ay_node_warm_time_limit: None,
            ay_hard_deadline: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ay_node_warm_limit_defaults_off() {
        assert_eq!(MipConfig::default().ay_node_warm_time_limit, None);
    }

    #[test]
    fn ay_hard_deadline_defaults_off() {
        assert_eq!(MipConfig::default().ay_hard_deadline, None);
    }

    #[test]
    fn feasibility_ingress_defaults_historical() {
        assert_eq!(
            MipConfig::default().feasibility_ingress,
            MipFeasibilityIngress::Historical
        );
    }
}

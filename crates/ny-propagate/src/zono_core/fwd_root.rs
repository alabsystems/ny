// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#fwd-root-screen` (V4 of the Vibecheck plan, route (b)): the typed-key
//! plan and the screen's entry point.
//!
//! The screen compiles the verifier's graph into a [`DagProgram`], runs ONE
//! certified correlation-preserving zonotope forward pass over the root input
//! box (every relaxation interval intersected with ny's IBP box and C1's
//! exact lookup-table range), and returns the certified output box. The CLI
//! turns that box into clause refutations and UNIONS them with the existing
//! precheck: the screen only ever ADDS refutations that are individually
//! certified; it never replaces IBP/C1 (F1 measured that a lone forward screen
//! loses banked IBP rows through rounding).
//!
//! Dark by default: `bab.forward_root_screen` absent or `false` means no plan,
//! no work and no output. The key has NO environment reader, so an armed plan
//! always reports [`Source::Config`] (`fwd_root_screen_preset_source_is_typed`).

use std::time::{Duration, Instant};

use ny_levers::Source;
use ny_tensor::BoundedTensor;

use super::dag::{DagForwardOptions, DagForwardStats, DagProgram};
use super::ZonoRefusal;
use crate::beta_crown::config::BetaCrownConfig;
use crate::GraphNetwork;

/// Byte ceiling when the process memory headroom is unobservable.
pub const FWD_ROOT_UNOBSERVED_HEADROOM_CAP: u64 = 4 << 30;

/// Resolved `bab.forward_root_screen` for one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FwdRootPlan {
    /// Configured live-state ceiling (`bab.forward_root_screen_max_mb`).
    pub max_bytes: u64,
    /// Where the arming came from; always [`Source::Config`] (no env reader).
    pub source: Source,
}

/// The plan for `config`, or `None` when the key is dark (the default).
#[must_use]
pub fn forward_root_screen_plan(config: &BetaCrownConfig) -> Option<FwdRootPlan> {
    if !config.forward_root_screen {
        return None;
    }
    Some(FwdRootPlan {
        max_bytes: u64::from(config.forward_root_screen_max_mb.max(1)) << 20,
        source: Source::Config,
    })
}

/// The live-state budget actually admitted: the configured ceiling, clamped
/// to half the process memory headroom (MemAvailable-derived when there is no
/// cgroup — competition runs uncapped, so this is the only guard), or to
/// [`FWD_ROOT_UNOBSERVED_HEADROOM_CAP`] when no headroom is observable.
#[must_use]
pub fn admitted_byte_budget(max_bytes: u64) -> u64 {
    match crate::network::crown_memory::process_memory_headroom_bytes() {
        Some(headroom) => max_bytes.min(headroom / 2),
        None => max_bytes.min(FWD_ROOT_UNOBSERVED_HEADROOM_CAP),
    }
}

/// A completed screen.
#[derive(Debug, Clone)]
pub struct FwdRootScreen {
    /// Certified output box, lower ends (binary64, outward).
    pub output_lo: Vec<f64>,
    /// Certified output box, upper ends.
    pub output_hi: Vec<f64>,
    /// Compiled op census (`kind:count,...`).
    pub census: String,
    /// Forward-pass statistics.
    pub stats: DagForwardStats,
    /// Admitted live-state budget.
    pub budget_bytes: u64,
    /// Process VmHWM after the pass, MiB (Linux).
    pub peak_rss_mb: Option<u64>,
    /// Wall time including compilation and the IBP pass.
    pub elapsed: Duration,
}

/// A declined screen (fail closed: no clause is refuted by it).
#[derive(Debug, Clone)]
pub struct FwdRootDecline {
    /// Why.
    pub reason: ZonoRefusal,
    /// Compiled op census, or `-` when compilation itself refused.
    pub census: String,
    /// Admitted live-state budget.
    pub budget_bytes: u64,
    /// Wall time until the refusal.
    pub elapsed: Duration,
}

/// Run the certified forward root screen over `input`.
///
/// `intersect_ibp` is `true` on every production call; `false` exists only
/// for the replica-parity pin (the F1 gate's pure ZONO pass).
pub fn run_forward_root_screen(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    max_bytes: u64,
    deadline: Option<Instant>,
    intersect_ibp: bool,
) -> Result<FwdRootScreen, FwdRootDecline> {
    let started = Instant::now();
    let budget_bytes = admitted_byte_budget(max_bytes);
    let (program, ibp) = match DagProgram::compile_with_ibp(graph, input, deadline) {
        Ok(compiled) => compiled,
        Err(reason) => {
            return Err(FwdRootDecline {
                reason,
                census: "-".to_string(),
                budget_bytes,
                elapsed: started.elapsed(),
            })
        }
    };
    let census = program.census();
    let opts = DagForwardOptions {
        intersect_ibp,
        byte_budget: budget_bytes,
    };
    match program.forward(input, &opts, &ibp, deadline, None) {
        Ok(outcome) => Ok(FwdRootScreen {
            output_lo: outcome.output_lo,
            output_hi: outcome.output_hi,
            census,
            stats: outcome.stats,
            budget_bytes,
            peak_rss_mb: super::cgan::self_rss_mb().1,
            elapsed: started.elapsed(),
        }),
        Err(reason) => Err(FwdRootDecline {
            reason,
            census,
            budget_bytes,
            elapsed: started.elapsed(),
        }),
    }
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#frozen-dual-state` — V1 STAGE A of the frozen alpha-zonotope dual-ascent
//! BaB (VIBECHECK_PLAN.md §6.V1; gates P1 cora FC and P1x cifar100 ResNet,
//! validated against ny's own root to 2e-5).
//!
//! Stage A is the certified QUERY STATE and its node arithmetic. It builds,
//! from the margin-row lane's frozen root artifacts (the trunk boxes after
//! backward-interm and interm-alpha, the committed trunk and head alpha, and
//! the root y-box):
//!
//! * [`FrozenDualFrame`] — the class-independent part: every ReLU's certified
//!   box and slopes, the generator column layout (one input symbol per
//!   non-degenerate input dimension, one relaxation symbol per unstable ReLU,
//!   trunk and head), and a certified magnitude bound for every tensor;
//! * [`FrozenDualState`] — per open class, the generator-space linearisation
//!   of the margin (`c_obj`, `d`, `r_obj`) and of every unstable
//!   pre-activation (`c_k`, `a_k`, `r_k`), with direction-adaptive slopes and
//!   the `mu` parallelogram, all binary64 with explicit gamma charges and
//!   outward rounding (see [`build`] for the invariant and every charge).
//!   Built per class, or (ENGINE_PROGRAM M3, `margin_row.frozen_dual_bab_state:
//!   shared`) assembled from ONE class-independent trunk of rows plus a
//!   one-column margin tail per class (`build`'s "shared trunk" section);
//! * [`FrozenDualProblem`] — the rank-1 split substitution under a split order
//!   and the CERTIFIED node bound `g(L)` at clamped multipliers (see
//!   [`problem`]);
//! * [`static_order`] — the box + single-halfspace split order (ordering
//!   only);
//! * [`check`] — a double-double pointwise containment checker (diagnostic).
//!
//! # What stage A does NOT do
//!
//! There is no search here and no verdict. Stage B
//! ([`super::zono_bab`], the certified BFS over [`FrozenDualProblem`]) is the
//! only consumer and is where a pruned node may ever turn into `unsat`: the
//! typed key `margin_row.frozen_dual_bab` (declaration default `false`, enabled
//! by the shipped CIFAR100 and TinyImageNet presets) arms that tier inside
//! the tree driver with bounded automatic admission. The
//! stage-A-only [`stage_a_probe`] (build, report, DROP) remains as a
//! diagnostic entry and is no longer called by the lane.
//!
//! # Soundness rules stage B inherits
//!
//! * Prune a node ONLY on `NodeBound::certified > 0` (strict). The raw value
//!   is a search signal.
//! * Any multiplier vector is a valid dual point once clamped to `>= 0`; the
//!   evaluator clamps and recomputes `rc` at the clamped point itself. NEVER
//!   port Vibecheck's far probe (`rc` advanced with UNclamped multipliers,
//!   `b`-term with clamped ones: not a dual bound even in exact arithmetic,
//!   DECISION.md U4b) or its slope-drop infeasibility test (U3).
//! * Never reuse `star_dual::outward_pad` (it scales by the RESULT magnitude,
//!   not `sum|terms|`, and under-pads under cancellation).
//! * This module has no satisfiable-side output at all: it can only ever
//!   support an `unsat` (every class's frontier certified empty), never emit a
//!   counterexample.

pub mod build;
pub mod check;
pub mod frame;
pub mod order;
pub mod problem;

use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Original trunk topology shared with phase-relation discovery and its
/// `fd_trunk_tail_layers_are_read_by_no_row` controls. Relation proposals use
/// this structural mask; the original evaluator still certifies every prune.
pub(crate) use build::tail_layers;
/// `#fd-compact-checkable` (B2.3 x #breakpoint-dual): a ranked compact
/// state's class relaxation, kept only under hint capture for its proof trace.
pub(crate) use build::CompactRelax;
pub use build::{
    admitted_bytes, certified_mu, estimate_ranked_state_bytes, estimate_state_bytes,
    FrozenDualBuildStats, FrozenDualOptions, FrozenDualState, RankedFrozenDualState,
    FROZEN_DUAL_DEFAULT_CHUNK_COLS, FROZEN_DUAL_DEFAULT_MAX_STATE_BYTES,
    FROZEN_DUAL_UNOBSERVED_HEADROOM_CAP,
};
/// `#fd-shared-state` (ENGINE_PROGRAM M3): the class-independent trunk (every
/// unstable row, built once under slopes shared across the open classes) and
/// the per-class margin tail over it; see [`build`]'s "shared trunk" section.
pub(crate) use build::{class_margin, ClassTail, FrozenDualTrunk, TrunkRelax};
pub use check::{check_point, dd_forward, Dd, PointCheck, DD_TOLERANCE};
pub use frame::{FrameInput, FrozenDualFrame, LayerInput, RelaxLayer, ReluSite, MAX_DIM, SAFE};
pub use order::{split_scores, static_order};
pub use problem::{FrozenDualProblem, NodeBound, NodeScratch, Side, SplitRow, Subgradient};

use super::engine::BackwardEngine;
use super::net::TwinNet;
use super::root::RootGates;

/// `margin_row.frozen_dual_bab`, latched once per process (the governed
/// env-over-preset gate; the CLI bridge writes the preset cell before any lane
/// starts). Declaration default `false`.
///
/// #router-r5: OR the acting router ROUTED the tier on this row ([`routed`]).
/// Public so the routed arming is pinned end to end from its own test
/// process (`tests/router_r5_routed_fd.rs`).
#[must_use]
pub fn armed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(crate::preset_gates::margin_row_frozen_dual_bab_enabled) || routed()
}

/// #router-r5 CR-RS: whether the acting router ROUTED the frozen-dual tier on
/// this row — only at SR.seal (S7 + S8 done on its envelope), where the row's
/// facts are inside that envelope and the row did not pin
/// `margin_row.frozen_dual_bab` (a pin, either way, keeps the key's own
/// answer). One atomic load answering `false` unless the route authority is
/// armed; no capability is sealed today, so this is `false` on every row.
/// Counted by the authority (its consults are the route's arrival evidence).
#[must_use]
pub fn routed() -> bool {
    ny_core::route_authority::global().routed(ny_core::router::decl::ids::FROZEN_DUAL_BAB)
}

/// Share of the time left to the lane deadline the stage-A probe may spend.
const STAGE_A_SHARE: f64 = 0.25;

/// What one stage-A probe did (telemetry; nothing reads it for a bound).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageAReport {
    /// Classes the root left open.
    pub classes: usize,
    /// States built.
    pub built: usize,
    /// Classes refused (budget, memory, non-finite, deadline, ...).
    pub refused: usize,
    /// Wall seconds.
    pub secs: f64,
}

/// Peak resident set (`VmHWM`) in MiB, `None` where `/proc` is unavailable.
pub(crate) fn peak_rss_mb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

/// The stage-A arrival probe: evaluate the frozen root exactly as the tree
/// will (`bab::root_eval`), build the certified state of every class it leaves
/// open, print one fixed-schema receipt per class plus a `done` line, and DROP
/// the states. Verdict-inert by construction: it returns only telemetry.
///
/// Since stage B (`super::zono_bab`) the armed lane no longer calls this: the
/// key now arms the certified search tier inside the tree driver, which builds
/// the same states and CONSUMES them. The probe stays as a diagnostic entry
/// (and is pinned verdict-inert by its test).
///
/// Receipts (every field on every line; `refused=-` when built):
///
/// ```text
/// [frozen-dual-bab] armed=1 stage=A class=<j> state_gens=<n> in_gens=<i> unstable=<S>
///   build_ms=<t> passes=<p> root_g=<g> lane_root=<dj> r_obj=<e> max_r_k=<e> state_mb=<m>
///   order_ms=<t> problem_root=<g|nan> refused=<token|-> verdict=state_only
/// [frozen-dual-bab] done stage=A classes=<c> built=<k> refused=<r> secs=<t>
///   peak_rss_mb=<m|-> verdict=state_only
/// ```
pub fn stage_a_probe(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
) -> StageAReport {
    let t0 = Instant::now();
    let mut report = StageAReport::default();
    let probe_deadline = deadline.map(|d| {
        let left = d.saturating_duration_since(t0).as_secs_f64();
        t0 + Duration::from_secs_f64((left * STAGE_A_SHARE).max(0.0))
    });
    let eng = BackwardEngine::new(net, root);
    let reval = super::bab::root_eval(&eng, net, t, adv);
    let (tree_classes, dj, ybox) = match reval {
        Ok(re) => {
            let yb = re.root_ybox();
            (re.tree_classes.clone(), re.dj, Some(yb))
        }
        Err(_) => (Vec::new(), Vec::new(), None),
    };
    report.classes = tree_classes.len();
    let frame = ybox
        .as_ref()
        .ok_or(crate::zono_core::ZonoRefusal::Internal(
            "root_eval failed".to_string(),
        ))
        .and_then(|yb| FrozenDualFrame::from_root(net, root, yb, t));
    for &j in &tree_classes {
        let lane_root = adv
            .iter()
            .position(|&c| c == j)
            .and_then(|i| dj.get(i).copied())
            .unwrap_or(f64::NAN);
        let built = frame.as_ref().map_err(Clone::clone).and_then(|fr| {
            let opts = FrozenDualOptions {
                deadline: probe_deadline,
                ..FrozenDualOptions::default()
            };
            FrozenDualState::build(net, fr, j, &opts).map(|st| (st, fr.n_in_gens))
        });
        match built {
            Ok((st, in_gens)) => {
                report.built += 1;
                // The static order and the split substitution are what stage B
                // starts from; pricing them here is telemetry. The substituted
                // problem copies the rows, so it is admitted separately.
                let t_order = Instant::now();
                let order = static_order(&st);
                let copy_bytes = (order.len() as u64)
                    .saturating_mul(st.n_gens as u64)
                    .saturating_mul(8);
                let problem_root = if copy_bytes.saturating_add(st.bytes())
                    <= admitted_bytes(FROZEN_DUAL_DEFAULT_MAX_STATE_BYTES)
                {
                    FrozenDualProblem::new(&st, &order).map_or(f64::NAN, |p| p.root_bound())
                } else {
                    f64::NAN
                };
                let order_ms = t_order.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "[frozen-dual-bab] armed=1 stage=A class={j} state_gens={} in_gens={in_gens} \
                     unstable={} build_ms={:.0} passes={} root_g={:+.6e} lane_root={:+.6e} \
                     r_obj={:.3e} max_r_k={:.3e} state_mb={} order_ms={order_ms:.0} \
                     problem_root={problem_root:+.6e} refused=- verdict=state_only",
                    st.n_gens,
                    st.num_unstable(),
                    st.stats.build_ms,
                    st.stats.passes,
                    st.root_bound(),
                    lane_root,
                    st.r_obj,
                    st.max_row_remainder(),
                    st.bytes() >> 20,
                );
            }
            Err(why) => {
                report.refused += 1;
                eprintln!(
                    "[frozen-dual-bab] armed=1 stage=A class={j} state_gens=0 in_gens=0 \
                     unstable=0 build_ms=0 passes=0 root_g=nan lane_root={lane_root:+.6e} \
                     r_obj=nan max_r_k=nan state_mb=0 order_ms=0 problem_root=nan refused={} \
                     verdict=state_only",
                    why.token()
                );
            }
        }
    }
    report.secs = t0.elapsed().as_secs_f64();
    let rss = peak_rss_mb().map_or_else(|| "-".to_string(), |m| m.to_string());
    eprintln!(
        "[frozen-dual-bab] done stage=A classes={} built={} refused={} secs={:.3} \
         peak_rss_mb={rss} verdict=state_only",
        report.classes, report.built, report.refused, report.secs
    );
    report
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #root-certify-once: Phase-0.5 root alpha — SEARCH UNTRUSTED, CERTIFY ONCE.
//!
//! # Why
//!
//! The lane's root alpha ascent (`alpha_opt`) spends one CERTIFIED Outward
//! backward pass per candidate alpha purely to decide which alpha to keep. On
//! tinyimagenet that ascent costs 27-29 s of a 100 s row for 9-20 iterations
//! (`scratchpad/vbs/vibecheck/study_3.md`, idea 2), while alpha-beta-CROWN and
//! Vibecheck run the same free parameter uncertified (Phase 0.5: 10 joint
//! intermediate iterations + 20 spec iterations) and never certify at all.
//! Certifying a RANKING buys nothing: every `alpha in [0, 1]` is a valid
//! DeepPoly lower line (`root.rs`, `relu_lower` in `Crownproof/Basic.lean`),
//! so a mis-ranked candidate costs tightness and never validity. Only the alpha
//! that will be PUBLISHED needs a certified pass.
//!
//! # What
//!
//! Run where the root is still owned mutably, after every existing root-alpha
//! owner (`#alpha-opt`, `#rebake`, `#joint-refresh`) and before `#spec-alpha`
//! is installed, so the INCUMBENT this phase must beat is exactly the root the
//! lane would publish without it ("today's" root):
//!
//! 1. **Scratch.** Copy the certified root into an UNTRUSTED scratch whose
//!    `mode` is [`RoundMode::Parity`]: plain f64 round-to-nearest, no `|W|`
//!    error lane (about half the arithmetic of a certified pass), no clip rows,
//!    an empty evaluation memo.
//! 2. **Joint phase** (`margin_row.root_certify_once_refresh` rounds, default
//!    0): abc's per-iteration intermediate refresh — every trunk box of the
//!    SCRATCH re-derived from its current gates by the UNTRUSTED refresh
//!    (`backward_interm::refresh_untrusted_intermediates`), then one untrusted
//!    ascent step on the refreshed frame. Requires `#backward-interm` (it
//!    re-runs that phase's frozen `topk`/`chunk`/`window` policy) and no Tier-0
//!    retained rows (they forbid the certified refresh the rounds exist to
//!    feed). BUDGET NOTE: round 1 needs `0.35 * budget >= 2.625 * eval + 0.5 s`
//!    (the certification reserve takes `2.25 * eval + 0.3 * budget` and the
//!    joint window is half of what is left), i.e. `budget >= ~32 s` at the
//!    default 4 s evaluation price — at the default 20 s the rounds refuse
//!    loudly (`joint stop round=1`) and only the spec phase runs.
//! 3. **Spec phase** (`margin_row.root_certify_once_iters`, default 20): the
//!    untrusted ascent (`alpha_opt::untrusted_ascent`, i.e. the lane's own
//!    `search_alpha` on the Parity scratch) with the intermediates frozen.
//!    A certified incumbent that already closes every class refuses the phase
//!    (`stop=closed`): the tree publishes it without the search wall.
//! 4. **Certify once.** Install the scratch's alpha into the CERTIFIED root
//!    through the ingest gate (finite, clamped to `[0, 1]`, `ms = max(alpha,
//!    s)` re-derived, only on neurons unstable in the CERTIFIED boxes) and run
//!    ONE certified Outward `root_eval`. When the joint phase tightened
//!    anything (and no Tier-0 rows are retained), also run ONE certified
//!    `backward_interm::refresh_root_intermediates` of that candidate — the
//!    boxes the search assumed, now certified — and ONE certified `root_eval`
//!    of it.
//! 5. **Keep the better.** The root is replaced only by a candidate whose
//!    CERTIFIED bound strictly beats the certified incumbent (and, for the
//!    refreshed candidate, the alpha-only one); otherwise the incumbent is
//!    restored bit-for-bit. The published root bound is therefore never worse
//!    than today's.
//!
//! # Soundness
//!
//! * The untrusted numbers (`untrusted_best`, the scratch's boxes, every
//!   `[alpha-opt:untrusted]` line) RANK alphas and are never published: the
//!   scratch is a separate `RootGates` that no verdict path reads, the untrusted
//!   ascent and refresh refuse an Outward root, and the only values that leave
//!   this phase are (a) alphas, validated at the ingest gate, and (b) boxes
//!   produced by the CERTIFIED refresh.
//! * The published bound is recomputed by the unchanged certified `root_eval`
//!   (`rebake::certified_root_bound`) for exactly the gates that are kept; the
//!   tree's own root evaluation reads the same gates through the same code.
//! * Any `alpha in [0, 1]` is a valid lower line on any box; the upper chords
//!   `(s, c)` and the boxes `(l, u)` are never touched by the alpha install;
//!   `ms = max(alpha, s)` is re-derived with every alpha write (a stale `ms`
//!   would under-scale the carried error — the one false-UNSAT generator of
//!   this family, `root.rs`).
//! * The certified refresh is the `#joint-refresh` refresh verbatim (certified
//!   enclosures intersected shrink-only, chords re-certified, clip rows filtered
//!   to their neurons, a crossed intersection refuses the whole refresh).
//!
//! # Budget and memory
//!
//! Wall request `margin_row.root_certify_once_secs` (default 20), capped to 40%
//! of the remaining deadline; one certified evaluation (and, with the joint
//! phase, the certified refresh slice) is reserved before the search starts.
//! The certified evaluation of the kept root is not extra work: it is STORED on
//! that root (`bab::root_eval_store`) and the tree's own root evaluation is a
//! memo hit on it; the incumbent's bound is handed over by the previous owner
//! when it already computed one for exactly these gates. RSS admission:
//! the scratch, one pass working set and the refresh candidate are estimated and
//! refused against half the observed memory headroom (4 GiB when unobservable)
//! — ny has no memory governor without a cgroup, so this IS the guard.
//!
//! # GPU
//!
//! `margin_row.gpu_conv` routes CERTIFIED passes only (`gpu_conv::attempt`
//! refuses Parity), so the certified evaluations here ride the GPU seam when it
//! is armed, and the untrusted Parity passes run on the CPU kernel (counted as
//! `Refusal::Parity` in the `[mr-gpu-conv] done` receipt). An f32/GPU searcher
//! is the next step of the design; this phase is its CPU certify-once half.
//!
//! # Observable
//!
//! Armed, the phase prints `[root-certify-once] start:` on entry and exactly
//! one fixed-schema terminal line on EVERY return path, refusals included:
//!
//! ```text
//! [root-certify-once] armed=1 iters=<n> untrusted_best=<f> certified=<f> kept=<none|incumbent|alpha|refresh>
//!   incumbent=<f> certified_alpha=<f> certified_refresh=<f> published=<f> refresh_rounds=<n>
//!   refresh_tightened=<n> untrusted_passes=<n> moved=<n> rss_need_mb=<f> rss_admitted_mb=<f>
//!   budget=<f> secs=<f> stop=<reason>
//! ```
//!
//! (one line; wrapped here). Dark (the declaration default) it prints nothing
//! and touches nothing.
//!
//! # The search selector (`margin_row.root_search`, #bbe-joint-root)
//!
//! `ascent` (the declaration default) is everything above, bit for bit: no
//! backend is consulted. `joint_f32` replaces steps 2-3 — the untrusted joint
//! rounds and the spec phase — with the injected UNTRUSTED joint-alpha root
//! search (`super::root_search`: abc's CROWN init, spec prune, per-start-node
//! alpha with an in-loop refresh, early exits, run by `ny-bbe`'s CPU f32
//! backend). Its final slopes enter step 4 through the SAME ingest gate
//! ([`install_candidate_alpha`]) and step 5 is unchanged: one certified
//! evaluation, kept only if it strictly beats the incumbent. The phase then
//! also prints the `[bbe] phase=joint_root ...` line on every return path.

use std::time::{Duration, Instant};

use ny_core::search::SearchBackend;

use super::alpha_opt;
use super::backward_interm;
use super::net::TwinNet;
use super::rebake::certified_root_bound;
use super::root::{LayerGates, RootGates};
use super::root_search::{self, BbeReport, RootSearch};
use super::rounding::RoundMode;

/// Fraction of the REMAINING deadline the phase may spend (the lane's shared
/// root-phase starvation guard: `#alpha-opt`, `#rebake`, `#joint-refresh`).
pub(crate) const REMAINING_FRAC: f64 = 0.4;

/// Smallest slice worth starting a search step in.
const MIN_SLICE_SECS: f64 = 0.5;

/// Certified `root_eval` cost assumed when the incumbent bound was handed over
/// by the previous owner (nothing was timed here) or its evaluation was too
/// fast to price with. Measured 4.5 s on cct idx448 (`rebake`'s
/// `JOINT_DEFAULT_EVAL_SECS`); the fail-safe direction is to reserve more.
const DEFAULT_EVAL_SECS: f64 = 4.0;

/// A timed incumbent evaluation faster than this prices nothing (the default
/// is reserved instead).
const MIN_TIMED_EVAL_SECS: f64 = 0.05;

/// Share of the reserved-for-certification budget given to the CERTIFIED
/// refresh when the joint phase runs (the refresh is cut at its slice; a
/// partial certified refresh is sound, only looser).
const CERT_REFRESH_SHARE: f64 = 0.3;

/// Share of the search window the joint phase may take before the spec phase.
const JOINT_SHARE: f64 = 0.5;

/// Untrusted ascent steps per joint round (abc: one Adam step per full
/// intermediate recomputation).
const JOINT_STEPS_PER_ROUND: usize = 1;

/// Admission ceiling when the memory headroom is unobservable (competition runs
/// uncapped; `zono_state::FROZEN_DUAL_UNOBSERVED_HEADROOM_CAP` is the sibling).
const UNOBSERVED_HEADROOM_CAP: u64 = 4 << 30;

/// Which root the phase left behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kept {
    /// Refused before the incumbent was scored.
    None,
    /// The certified incumbent ("today's" root), bit-for-bit.
    Incumbent,
    /// The certified root with the searched alpha installed (boxes unchanged).
    Alpha,
    /// The searched alpha with its boxes re-derived by the CERTIFIED refresh.
    Refresh,
}

impl Kept {
    /// Fixed token for the observable.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Incumbent => "incumbent",
            Self::Alpha => "alpha",
            Self::Refresh => "refresh",
        }
    }
}

/// What one armed run did (telemetry + test surface). Every `f64` that is not
/// computed stays NaN, so "did not run" is distinguishable from a value.
#[derive(Debug, Clone)]
pub struct RootCertifyOnceReport {
    /// Effective wall slice after the 40%-of-remaining cap.
    pub budget: f64,
    /// Untrusted candidate iterations attempted (joint steps + spec phase).
    pub iters: usize,
    /// Joint rounds whose untrusted refresh completed.
    pub refresh_rounds: usize,
    /// Scratch boxes the untrusted refresh tightened, across all rounds (with
    /// `margin_row.root_search: joint_f32`: the reference box sides the joint
    /// search tightened — it gates the certified refresh the same way).
    pub refresh_tightened: usize,
    /// Untrusted (Parity) ascent passes spent, baselines included.
    pub untrusted_passes: usize,
    /// First untrusted objective (Parity frozen frame; a ranking number).
    pub untrusted_start: f64,
    /// Best untrusted objective (Parity frozen frame; a ranking number, NEVER
    /// published).
    pub untrusted_best: f64,
    /// Certified-root slopes (trunk + head) the candidate changes.
    pub moved: usize,
    /// Certified root bound of the root this phase STARTED from.
    pub incumbent: f64,
    /// Certified root bound of the alpha-only candidate.
    pub certified_alpha: f64,
    /// Certified root bound of the certified-refresh candidate.
    pub certified_refresh: f64,
    /// Best certified candidate bound (NaN when none was certified).
    pub certified: f64,
    /// Certified root bound of the root this phase LEFT (`>= incumbent`).
    pub published: f64,
    /// Which root was kept.
    pub kept: Kept,
    /// RSS admission estimate and admitted ceiling (bytes).
    pub rss_need: u64,
    pub rss_admitted: u64,
    /// Wall seconds of the whole phase.
    pub secs: f64,
    /// Why the phase stopped: `refused` (entry refusal, nothing scored),
    /// `rss`, `eval_error`, `closed` (the certified incumbent already closes
    /// every class), `budget` (no search step affordable after the reserve),
    /// `refresh_error` (an untrusted joint refresh refused), `search_refused`
    /// (the `joint_f32` backend produced no hint), `deadline` (the
    /// instance deadline skipped a certification), `unchanged` (the search
    /// moved nothing), or `searched`. A failure reason is never overwritten by
    /// `unchanged`, so the terminal line says WHY a run kept its incumbent.
    pub stop: &'static str,
    /// #bbe-joint-root telemetry: `Some` exactly when the phase ran with
    /// `margin_row.root_search: joint_f32` (printed as the `[bbe]` line on
    /// every return path); `None` on the default `ascent` search.
    pub bbe: Option<BbeReport>,
}

impl RootCertifyOnceReport {
    fn new(budget: f64) -> Self {
        Self {
            budget,
            iters: 0,
            refresh_rounds: 0,
            refresh_tightened: 0,
            untrusted_passes: 0,
            untrusted_start: f64::NAN,
            untrusted_best: f64::NAN,
            moved: 0,
            incumbent: f64::NAN,
            certified_alpha: f64::NAN,
            certified_refresh: f64::NAN,
            certified: f64::NAN,
            published: f64::NAN,
            kept: Kept::None,
            rss_need: 0,
            rss_admitted: 0,
            secs: 0.0,
            stop: "refused",
            bbe: None,
        }
    }

    /// The fixed-schema terminal line (one line, every armed return path).
    #[must_use]
    pub fn line(&self) -> String {
        const MB: f64 = 1024.0 * 1024.0;
        format!(
            "[root-certify-once] armed=1 iters={} untrusted_best={:.6} certified={:.6} kept={} \
incumbent={:.6} certified_alpha={:.6} certified_refresh={:.6} published={:.6} \
refresh_rounds={} refresh_tightened={} untrusted_passes={} moved={} rss_need_mb={:.1} \
rss_admitted_mb={:.1} budget={:.1} secs={:.2} stop={}",
            self.iters,
            self.untrusted_best,
            self.certified,
            self.kept.token(),
            self.incumbent,
            self.certified_alpha,
            self.certified_refresh,
            self.published,
            self.refresh_rounds,
            self.refresh_tightened,
            self.untrusted_passes,
            self.moved,
            self.rss_need as f64 / MB,
            self.rss_admitted as f64 / MB,
            self.budget,
            self.secs,
            self.stop
        )
    }
}

fn done(t0: Instant, mut report: RootCertifyOnceReport) -> RootCertifyOnceReport {
    report.secs = t0.elapsed().as_secs_f64();
    // The lever's OWN terminal receipt, printed on EVERY armed return path so
    // "ran and kept the incumbent" is distinguishable from "never ran".
    eprintln!("{}", report.line());
    // #bbe-joint-root: the search selector's own receipt (joint_f32 only).
    if let Some(bbe) = report.bbe.as_ref() {
        eprintln!("{}", bbe.line(&report));
    }
    report
}

/// Frozen-policy entry: no-op (and silent) unless `margin_row.root_certify_once`
/// is armed.
///
/// Called from `lane_impl` after `#alpha-opt`, `#rebake` and `#joint-refresh`,
/// while the root is still owned mutably and before `#spec-alpha` or any
/// consumer reads it.
#[allow(clippy::too_many_arguments)]
pub fn maybe_certify_once_root(
    net: &TwinNet,
    root: &mut RootGates,
    retained_present: bool,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    known_incumbent: Option<f64>,
    policy: super::RootCertifyOnceRuntimePolicy,
    root_alpha: super::RootAlphaRuntimePolicy,
    backward_interm: super::BackwardIntermRuntimePolicy,
) -> Option<RootCertifyOnceReport> {
    if !policy.enabled {
        return None;
    }
    // #bbe-joint-root: the injected backend is resolved ONLY for `joint_f32`;
    // the default `ascent` search never touches the registry.
    let backend = match policy.search {
        RootSearch::Ascent => None,
        RootSearch::JointF32 => root_search::registered_backend(),
    };
    Some(certify_once_root_with(
        net,
        root,
        retained_present,
        t,
        adv,
        deadline,
        known_incumbent,
        policy,
        root_alpha,
        backward_interm,
        backend.as_deref(),
    ))
}

/// Panic-free `t0 + secs`, capped by the instance deadline. An unrepresentable
/// end means "no slice cut" (the deadline, if any, still stops the work; a cut
/// is only ever a looser, still-sound frame).
fn slice_end(t0: Instant, secs: f64, deadline: Option<Instant>) -> Option<Instant> {
    let end = Duration::try_from_secs_f64(secs.max(0.0))
        .ok()
        .and_then(|slice| t0.checked_add(slice));
    match (end, deadline) {
        (Some(end), Some(dl)) => Some(end.min(dl)),
        (Some(end), None) => Some(end),
        (None, dl) => dl,
    }
}

/// Is the instance deadline already behind us?
fn expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|dl| Instant::now() >= dl)
}

/// The UNTRUSTED scratch: a copy of the certified gates in
/// [`RoundMode::Parity`], without clip rows, `#spec-alpha` or an inherited
/// evaluation memo. Nothing in it is ever read by a verdict path.
pub(crate) fn untrusted_scratch(root: &RootGates) -> RootGates {
    RootGates {
        mode: RoundMode::Parity,
        mid: root.mid.clone(),
        rad: root.rad.clone(),
        xabs: root.xabs.clone(),
        layers: root
            .layers
            .iter()
            .map(|layer| LayerGates {
                op: layer.op,
                n: layer.n,
                l: layer.l.clone(),
                u: layer.u.clone(),
                alpha: layer.alpha.clone(),
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms: layer.ms.clone(),
                unst: layer.unst.clone(),
                clip_rows: None,
            })
            .collect(),
        lo: root.lo.clone(),
        hi: root.hi.clone(),
        backward_interm_execution: root.backward_interm_execution,
        root_precision_policy: root.root_precision_policy,
        root_precision_execution: root.root_precision_execution,
        spec_alpha_policy: None,
        head_alpha: root.head_alpha.clone(),
        root_eval_memo: super::bab::RootEvalMemo::default(),
        head_group_source: None,
    }
}

/// Trunk `(alpha, ms)` per layer, plus the #head-alpha slopes.
type AlphaState = (Vec<(Vec<f64>, Vec<f64>)>, Option<Vec<f64>>);

fn snapshot_alpha_state(root: &RootGates) -> AlphaState {
    (
        root.layers
            .iter()
            .map(|layer| (layer.alpha.clone(), layer.ms.clone()))
            .collect(),
        root.head_alpha.clone(),
    )
}

fn restore_alpha_state(root: &mut RootGates, state: AlphaState) {
    let (layers, head) = state;
    for (layer, (alpha, ms)) in root.layers.iter_mut().zip(layers) {
        layer.alpha = alpha;
        layer.ms = ms;
    }
    root.head_alpha = head;
}

/// THE INGEST GATE: install the untrusted search's alpha into the CERTIFIED
/// root. Returns how many slopes changed.
///
/// * Only neurons UNSTABLE IN THE CERTIFIED BOXES are written; stable and
///   split-baked neurons keep their exact lines.
/// * A proposal must be finite and is clamped to `[0, 1]` — an out-of-range
///   alpha is a VALIDITY break, not a tightness one, so the clamp is a
///   soundness gate; a non-finite proposal leaves the certified value.
/// * `ms = max(alpha, s)` is re-derived with every write, from the CERTIFIED
///   `s`; `l`, `u`, `s`, `c`, `unst` and the clip rows are never touched.
/// * Head slopes are installed only as a whole vector of the head's width with
///   every entry finite (clamped); they are consumed only where a head neuron
///   is unstable in the evaluated y-box, where any slope in `[0, 1]` is valid.
pub(crate) fn install_candidate_alpha(
    root: &mut RootGates,
    scratch: &RootGates,
    n_y: usize,
) -> usize {
    let mut moved = 0usize;
    for (layer, cand) in root.layers.iter_mut().zip(&scratch.layers) {
        if cand.n != layer.n || cand.alpha.len() != layer.n || layer.alpha.len() != layer.n {
            continue;
        }
        for pos in 0..layer.unst.len() {
            let j = layer.unst[pos];
            let proposed = cand.alpha[j];
            if !proposed.is_finite() {
                continue;
            }
            let a = proposed.clamp(0.0, 1.0);
            if a != layer.alpha[j] {
                layer.alpha[j] = a;
                layer.ms[j] = a.max(layer.s[j]);
                moved += 1;
            }
        }
    }
    if let Some(head) = scratch.head_alpha.as_ref() {
        if head.len() == n_y && head.iter().all(|a| a.is_finite()) {
            let clamped: Vec<f64> = head.iter().map(|a| a.clamp(0.0, 1.0)).collect();
            let changed = match root.head_alpha.as_ref() {
                Some(current) if current.len() == clamped.len() => current
                    .iter()
                    .zip(&clamped)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count(),
                _ => clamped.len(),
            };
            if changed > 0 {
                root.head_alpha = Some(clamped);
                moved += changed;
            }
        }
    }
    moved
}

/// RSS admission estimate (bytes): the scratch gates (plus the refresh's new
/// candidate gates when the joint phase runs), one untrusted pass working set
/// (`~4` live `(tensor x cols)` coefficient matrices plus the `(n_in x cols)`
/// input rows and the frozen frame's two y-row passes), and — with the joint
/// phase — the certified refresh candidate's filtered clip rows.
pub(crate) fn estimate_bytes(net: &TwinNet, root: &RootGates, refresh: bool, chunk: usize) -> u64 {
    const F: u64 = 8;
    let n_in = root.mid.len() as u64;
    let gates = root
        .layers
        .iter()
        .map(|layer| {
            (layer.n as u64)
                .saturating_mul(6 * F)
                .saturating_add((layer.unst.len() as u64).saturating_mul(F))
        })
        .fold(n_in.saturating_mul(5 * F), u64::saturating_add);
    let widest = net.tsize.iter().copied().max().unwrap_or(0) as u64;
    let mut cols = (2 * net.n_y).max(net.n_out).max(1) as u64;
    if refresh {
        cols = cols.max(chunk.max(1) as u64);
    }
    let pass = widest
        .saturating_mul(cols)
        .saturating_mul(4 * F)
        .saturating_add(n_in.saturating_mul(cols).saturating_mul(3 * F));
    let clip = if refresh {
        root.layers
            .iter()
            .filter_map(|layer| layer.clip_rows.as_ref())
            .map(|rows| ((rows.m.len() + rows.d.len()) as u64).saturating_mul(F))
            .fold(0u64, u64::saturating_add)
    } else {
        0
    };
    gates
        .saturating_mul(if refresh { 3 } else { 1 })
        .saturating_add(pass)
        .saturating_add(clip)
}

/// The admitted ceiling: half the observed memory headroom (cgroup- or
/// `MemAvailable`-derived), or `UNOBSERVED_HEADROOM_CAP`.
fn admitted_bytes() -> u64 {
    match crate::network::crown_memory::process_memory_headroom_bytes() {
        Some(headroom) => headroom / 2,
        None => UNOBSERVED_HEADROOM_CAP,
    }
}

/// Fold one untrusted ascent into the report.
fn note_ascent(report: &mut RootCertifyOnceReport, step: &alpha_opt::UntrustedAscent) {
    report.iters += step.iters;
    report.untrusted_passes += step.passes;
    if report.untrusted_start.is_nan() {
        report.untrusted_start = step.baseline;
    }
    report.untrusted_best = step.best;
}

/// The phase proper with no search backend (the test surface; the policy
/// gate lives in [`maybe_certify_once_root`], which calls
/// [`certify_once_root_with`] directly — so outside tests this wrapper has no
/// caller and would be `dead_code`, which the workspace gate's `-D warnings`
/// refuses). See the module docs.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn certify_once_root(
    net: &TwinNet,
    root: &mut RootGates,
    retained_present: bool,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    known_incumbent: Option<f64>,
    policy: super::RootCertifyOnceRuntimePolicy,
    root_alpha: super::RootAlphaRuntimePolicy,
    backward_interm: super::BackwardIntermRuntimePolicy,
) -> RootCertifyOnceReport {
    certify_once_root_with(
        net,
        root,
        retained_present,
        t,
        adv,
        deadline,
        known_incumbent,
        policy,
        root_alpha,
        backward_interm,
        None,
    )
}

/// The phase proper. `search_backend` is consulted only when `policy.search`
/// is [`RootSearch::JointF32`] (a missing backend is then a refusal of the
/// search, never a verdict). See the module docs.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn certify_once_root_with(
    net: &TwinNet,
    root: &mut RootGates,
    retained_present: bool,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    known_incumbent: Option<f64>,
    policy: super::RootCertifyOnceRuntimePolicy,
    root_alpha: super::RootAlphaRuntimePolicy,
    backward_interm: super::BackwardIntermRuntimePolicy,
    search_backend: Option<&dyn SearchBackend>,
) -> RootCertifyOnceReport {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::Certify);
    let t0 = Instant::now();
    let requested = if policy.secs.is_finite() && policy.secs >= 0.0 {
        policy.secs
    } else {
        super::RootCertifyOnceRuntimePolicy::default().secs
    };
    let budget = match deadline {
        Some(dl) => requested.min(REMAINING_FRAC * dl.saturating_duration_since(t0).as_secs_f64()),
        None => requested,
    };
    let iters = policy
        .iters
        .clamp(1, super::RootCertifyOnceRuntimePolicy::MAX_ITERS);
    let rounds = policy
        .refresh
        .min(super::RootCertifyOnceRuntimePolicy::MAX_REFRESH);
    // The joint phase exists to hand its boxes to ONE certified refresh, which
    // Tier-0 retained rows forbid (they index the unrefreshed frame, exactly
    // `#joint-refresh`'s refusal). With them present the untrusted rounds could
    // only steer alpha toward boxes that are never certified, while their
    // reserve (`CERT_REFRESH_SHARE` of the budget) is withheld from the spec
    // phase — so they are refused up front, loudly.
    let refresh_on = rounds > 0 && backward_interm.enabled && !retained_present;
    // Engagement telemetry (R9): an armed run is visible in one line even when
    // everything below is refused.
    eprintln!(
        "[root-certify-once] start: iters={iters} refresh={rounds} refresh_on={refresh_on} \
budget={budget:.1}s lr={:.4} cols={} topk={} chunk={} window={} retained={retained_present}",
        root_alpha.alpha_lr,
        root_alpha.alpha_cols,
        backward_interm.topk,
        backward_interm.chunk,
        backward_interm.window
    );
    let mut report = RootCertifyOnceReport::new(budget);
    if policy.search == RootSearch::JointF32 {
        let name = search_backend.map_or("none", |backend| backend.name());
        eprintln!("[bbe] start: phase=joint_root backend={name} iters={iters}");
        report.bbe = Some(BbeReport::new(name));
    }
    let refusal = if !root.mode.outward() {
        Some("parity mode (the incumbent must be a certified root)")
    } else if root.spec_alpha_policy.is_some() {
        Some("#spec-alpha is already installed")
    } else if adv.is_empty() || root.layers.iter().all(|layer| layer.unst.is_empty()) {
        Some("nothing to optimize (no adv classes or no unstable neurons)")
    } else if budget.is_nan() || budget < MIN_SLICE_SECS {
        Some("no budget (deadline too close or a zero wall request)")
    } else {
        None
    };
    if let Some(why) = refusal {
        eprintln!("[root-certify-once] refuse: {why}");
        return done(t0, report);
    }
    if rounds > 0 && !backward_interm.enabled {
        eprintln!(
            "[root-certify-once] refresh refused: #backward-interm is not armed (the refresh \
re-runs that phase's frozen policy); spec phase only"
        );
    } else if rounds > 0 && retained_present {
        eprintln!(
            "[root-certify-once] refresh refused: Tier-0 retained rows index the unrefreshed \
frame, so the certified refresh could never run; spec phase only"
        );
    }
    // RSS admission (the joint search also holds a plain-data copy of the net;
    // its own working set is capped by what admission leaves).
    report.rss_need = estimate_bytes(net, root, refresh_on, backward_interm.chunk);
    if policy.search == RootSearch::JointF32 {
        report.rss_need = report
            .rss_need
            .saturating_add(root_search::search_copy_bytes(net));
    }
    report.rss_admitted = admitted_bytes();
    if report.rss_need > report.rss_admitted {
        eprintln!(
            "[root-certify-once] refuse: RSS admission need={}B admitted={}B",
            report.rss_need, report.rss_admitted
        );
        report.stop = "rss";
        return done(t0, report);
    }
    // THE INCUMBENT: today's certified root bound. `known_incumbent` is the
    // certified `root_eval` bound the previous root-alpha owner already computed
    // for EXACTLY these gates (the lane driver's `#joint-refresh` hand-off); a
    // known value saves one evaluation and is otherwise recomputed here by the
    // unchanged certified path (which also stores it for the tree's lookup).
    let (incumbent, eval_secs) = match known_incumbent.filter(|value| value.is_finite()) {
        Some(value) => (value, DEFAULT_EVAL_SECS),
        None => {
            let te = Instant::now();
            let value = match certified_root_bound(net, root, t, adv) {
                Ok(value) if value.is_finite() => value,
                Ok(_) => {
                    eprintln!("[root-certify-once] refuse: non-finite incumbent root_eval");
                    report.stop = "eval_error";
                    return done(t0, report);
                }
                Err(error) => {
                    eprintln!("[root-certify-once] refuse: incumbent root_eval: {error}");
                    report.stop = "eval_error";
                    return done(t0, report);
                }
            };
            let measured = te.elapsed().as_secs_f64();
            (
                value,
                if measured > MIN_TIMED_EVAL_SECS {
                    measured
                } else {
                    DEFAULT_EVAL_SECS
                },
            )
        }
    };
    report.incumbent = incumbent;
    report.published = incumbent;
    report.kept = Kept::Incumbent;
    // A certified incumbent that already closes every class (the lane's own
    // strict outward predicate) has nothing to buy: the tree publishes it on
    // its first evaluation. Spending the search wall first would only delay a
    // verdict that is already in hand.
    if super::bab::closed(root.mode, incumbent) {
        eprintln!(
            "[root-certify-once] refuse: the certified incumbent {incumbent:.6} already closes \
every class — the tree publishes it as is"
        );
        report.stop = "closed";
        return done(t0, report);
    }
    // Reserve the certification before the search may spend anything.
    let reserve = 1.25 * eval_secs
        + if refresh_on {
            CERT_REFRESH_SHARE * budget + eval_secs
        } else {
            0.0
        };
    let search_end = budget - reserve;
    if search_end - t0.elapsed().as_secs_f64() < eval_secs.max(MIN_SLICE_SECS) {
        eprintln!(
            "[root-certify-once] refuse: budget {budget:.1}s cannot pay a search step \
({:.1}s) after reserving {reserve:.1}s for the certification — incumbent stands",
            eval_secs.max(MIN_SLICE_SECS)
        );
        report.stop = "budget";
        return done(t0, report);
    }

    // ---- The UNTRUSTED search, on a Parity scratch -------------------------
    let mut scratch = untrusted_scratch(root);
    let mut stop = "searched";
    if policy.search == RootSearch::JointF32 {
        // #bbe-joint-root: the injected joint-alpha root search replaces BOTH
        // untrusted phases below. Only its slopes reach the scratch, and the
        // scratch reaches the certified root only through the ingest gate.
        let (hint, mut bbe) = root_search::run_joint_search(
            net,
            root,
            t,
            adv,
            search_backend,
            iters,
            root_search::JOINT_ROOT_LR,
            slice_end(t0, search_end, deadline),
            report.rss_admitted.saturating_sub(report.rss_need),
        );
        report.iters = bbe.iters;
        report.untrusted_passes = bbe.passes;
        report.refresh_tightened = bbe.tightened;
        match hint.as_ref() {
            Some(hint) => {
                let (layers, head) = root_search::hint_into_scratch(&mut scratch, hint, net.n_y);
                bbe.hint_layers = layers;
                bbe.hint_head = head;
            }
            None => stop = "search_refused",
        }
        report.bbe = Some(bbe);
    } else if refresh_on {
        let joint_end = {
            let now = t0.elapsed().as_secs_f64();
            now + JOINT_SHARE * (search_end - now)
        };
        // One step = the frozen frame + a baseline + one candidate pass.
        let mut step_secs = 1.5 * eval_secs;
        for round in 1..=rounds {
            let now = t0.elapsed().as_secs_f64();
            if now + step_secs + MIN_SLICE_SECS > joint_end || expired(deadline) {
                eprintln!(
                    "[root-certify-once] joint stop round={round}: {:.1}s left of the joint \
window cannot pay a refresh and a {step_secs:.1}s step",
                    (joint_end - now).max(0.0)
                );
                break;
            }
            let stop_at = slice_end(t0, joint_end - step_secs, deadline);
            let (next, stats) = match backward_interm::refresh_untrusted_intermediates(
                net,
                &scratch,
                backward_interm.topk,
                backward_interm.chunk,
                backward_interm.window,
                stop_at,
            ) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!(
                        "[root-certify-once] joint stop round={round}: untrusted refresh: {error} \
— the scratch keeps its previous frame"
                    );
                    stop = "refresh_error";
                    break;
                }
            };
            scratch = next;
            report.refresh_rounds += 1;
            report.refresh_tightened += stats.tightened;
            let ta = Instant::now();
            let step_budget = (joint_end - t0.elapsed().as_secs_f64()).max(0.0);
            let step = alpha_opt::untrusted_ascent(
                net,
                &mut scratch,
                t,
                adv,
                step_budget,
                JOINT_STEPS_PER_ROUND,
                root_alpha.alpha_lr,
                root_alpha.alpha_cols,
            );
            step_secs = ta.elapsed().as_secs_f64().max(0.05);
            if let Some(step) = step.as_ref() {
                note_ascent(&mut report, step);
            }
            let step_moved = step.as_ref().map_or(0, |s| s.moved);
            eprintln!(
                "[root-certify-once] round={round} layers={} columns={} tightened={} \
stabilized={} windowed_pairs={} dense_pairs={} refresh_secs={:.2} cut={} step_iters={} \
step_best={:.6} step_moved={step_moved} total_secs={:.2}",
                stats.layers,
                stats.columns,
                stats.tightened,
                stats.stabilized,
                stats.windowed_pairs,
                stats.dense_pairs,
                stats.secs,
                stats.deadline_cut,
                step.as_ref().map_or(0, |s| s.iters),
                step.as_ref().map_or(f64::NAN, |s| s.best),
                t0.elapsed().as_secs_f64()
            );
            if stats.tightened == 0 && step_moved == 0 {
                eprintln!(
                    "[root-certify-once] joint stop round={round}: neither the refresh nor the \
step moved the scratch — fixed point"
                );
                break;
            }
        }
    }
    let spec_budget = search_end - t0.elapsed().as_secs_f64();
    if policy.search == RootSearch::JointF32 {
        // The joint search above replaced the spec phase.
    } else if spec_budget >= eval_secs.max(MIN_SLICE_SECS) && !expired(deadline) {
        if let Some(step) = alpha_opt::untrusted_ascent(
            net,
            &mut scratch,
            t,
            adv,
            spec_budget,
            iters,
            root_alpha.alpha_lr,
            root_alpha.alpha_cols,
        ) {
            note_ascent(&mut report, &step);
            eprintln!(
                "[root-certify-once] spec phase iters={} passes={} untrusted_start={:.6} \
untrusted_best={:.6} moved={} budget_exhausted={} secs={:.2}",
                step.iters,
                step.passes,
                step.baseline,
                step.best,
                step.moved,
                step.budget_exhausted,
                t0.elapsed().as_secs_f64()
            );
        } else {
            // The search's own refuse/abort line (tagged `[alpha-opt:untrusted]`)
            // says why; this one ties it to the phase.
            eprintln!("[root-certify-once] spec phase refused by the untrusted ascent");
        }
    } else {
        eprintln!(
            "[root-certify-once] spec phase skipped: {:.1}s left of the search window",
            spec_budget.max(0.0)
        );
    }

    // ---- CERTIFY ONCE ------------------------------------------------------
    // Nothing untrusted crosses this line except alphas, through the gate.
    let certify_t0 = Instant::now();
    let snapshot = snapshot_alpha_state(root);
    report.moved = install_candidate_alpha(root, &scratch, net.n_y);
    let mut certified_alpha = f64::NAN;
    if report.moved > 0 {
        if expired(deadline) {
            eprintln!(
                "[root-certify-once] certify: the instance deadline passed before the alpha \
candidate could be certified — incumbent stands"
            );
            stop = "deadline";
        } else {
            match certified_root_bound(net, root, t, adv) {
                Ok(value) if value.is_finite() => certified_alpha = value,
                Ok(_) => eprintln!("[root-certify-once] certify: non-finite alpha candidate"),
                Err(error) => eprintln!("[root-certify-once] certify: alpha candidate: {error}"),
            }
        }
    }
    report.certified_alpha = certified_alpha;
    // The boxes the search assumed, CERTIFIED once — only when the joint phase
    // actually tightened the scratch, the root carries no Tier-0 rows (they
    // index the unrefreshed frame) and the budget still pays a slice.
    let mut refreshed: Option<(RootGates, f64)> = None;
    if refresh_on && report.refresh_tightened > 0 {
        let left = budget - t0.elapsed().as_secs_f64();
        if retained_present {
            eprintln!(
                "[root-certify-once] certified refresh refused: Tier-0 retained rows index the \
unrefreshed frame"
            );
        } else if expired(deadline) {
            eprintln!(
                "[root-certify-once] certified refresh refused: the instance deadline has passed"
            );
            stop = "deadline";
        } else if left < eval_secs + MIN_SLICE_SECS {
            eprintln!(
                "[root-certify-once] certified refresh refused: {left:.1}s left cannot pay a \
refresh slice and a {eval_secs:.1}s evaluation"
            );
        } else {
            let stop_at = slice_end(t0, budget - eval_secs, deadline);
            match backward_interm::refresh_root_intermediates(
                net,
                root,
                backward_interm.topk,
                backward_interm.chunk,
                backward_interm.window,
                stop_at,
                // #bi-gpu is not threaded into the certify-once refresh: the CPU
                // prefix pairs are bit-identical to the dark route.
                None,
            ) {
                Ok((cand, stats)) if stats.tightened > 0 => {
                    match certified_root_bound(net, &cand, t, adv) {
                        Ok(value) if value.is_finite() => {
                            report.certified_refresh = value;
                            refreshed = Some((cand, value));
                        }
                        Ok(_) => {
                            eprintln!("[root-certify-once] certify: non-finite refreshed candidate")
                        }
                        Err(error) => {
                            eprintln!("[root-certify-once] certify: refreshed candidate: {error}");
                        }
                    }
                    eprintln!(
                        "[root-certify-once] certified refresh columns={} tightened={} \
stabilized={} secs={:.2} cut={} certified={:.6}",
                        stats.columns,
                        stats.tightened,
                        stats.stabilized,
                        stats.secs,
                        stats.deadline_cut,
                        report.certified_refresh
                    );
                }
                Ok((_, stats)) => eprintln!(
                    "[root-certify-once] certified refresh tightened nothing (columns={} cut={})",
                    stats.columns, stats.deadline_cut
                ),
                Err(error) => eprintln!("[root-certify-once] certified refresh: {error}"),
            }
        }
    }

    if let Some(bbe) = report.bbe.as_mut() {
        bbe.certify_s = certify_t0.elapsed().as_secs_f64();
    }

    // ---- KEEP THE BETTER (certified numbers only) ---------------------------
    let alpha_wins = certified_alpha.is_finite() && certified_alpha > incumbent;
    let floor = if alpha_wins {
        certified_alpha
    } else {
        incumbent
    };
    report.certified = match (
        certified_alpha.is_finite(),
        report.certified_refresh.is_finite(),
    ) {
        (true, true) => certified_alpha.max(report.certified_refresh),
        (true, false) => certified_alpha,
        (false, true) => report.certified_refresh,
        (false, false) => f64::NAN,
    };
    match refreshed {
        // The refreshed candidate replaces the WHOLE root (its boxes, gates,
        // filtered clip rows and its own stored evaluation), and only when its
        // certified bound strictly beats both the incumbent and the alpha-only
        // candidate.
        Some((cand, value)) if value.is_finite() && value > floor => {
            *root = cand;
            report.published = value;
            report.kept = Kept::Refresh;
        }
        _ if alpha_wins => {
            report.published = certified_alpha;
            report.kept = Kept::Alpha;
        }
        _ => {
            restore_alpha_state(root, snapshot);
            report.published = incumbent;
            report.kept = Kept::Incumbent;
        }
    }
    // A failure reason (`refresh_error`, `deadline`) is never masked by
    // `unchanged`: the terminal line must say WHY the incumbent stood.
    report.stop = if stop == "searched" && report.moved == 0 && report.refresh_tightened == 0 {
        "unchanged"
    } else {
        stop
    };
    done(t0, report)
}

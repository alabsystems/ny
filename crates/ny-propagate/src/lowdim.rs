//! `#lowdim-prover` — a sound input-split branch-and-bound for graphs whose
//! INPUT dimension is small (linearizenn_2024: 4 inputs, a 4x256 ReLU trunk and
//! an affine head that re-reads the input through a `Slice`).
//!
//! # Why this exists
//!
//! Measured 2026-09-22: every other VNN-COMP 2026 entrant solves linearizenn
//! 60/60 (alpha-beta-CROWN in 6-8 s per row) while ny banks 16/60 and times
//! out on 44 at 855 s. ny routes the rows through the DomainList input-split
//! driver, whose per-domain intermediate bounds are the ROOT map intersected
//! with a per-domain IBP pass. IBP through a 4x256 trunk is loose, so the tree
//! explodes: one timed-out row explored 730,111 domains (~840/s), hit the
//! 500k queue cap and evicted, after which it can never be proved.
//!
//! With only `d` inputs, FORWARD-mode linear bounds cost `d + 1` coefficients
//! per neuron, i.e. one `n_out x n_in` GEMM against an `n_in x (d + 1)` block per
//! layer and domain. A numpy prototype of exactly this lane (forward-linear
//! intermediates, backward-CROWN objective, widest-dimension split) proves
//! those rows in 132-11,632 domains; ny's timed-out `prop_30_30` needs 132. A
//! forward-ONLY objective is >100x worse, so the objective is bounded by a
//! backward pass over the DAG that reuses the forward pass's relaxations.
//!
//! # Soundness
//!
//! Every value this module publishes is a rigorous enclosure under IEEE f64
//! round-to-nearest, in any summation order:
//!
//! * a node carries lower/upper linear forms `a·x + c` in the input `x`; the
//!   invariant is `lower(x) <= v(x) <= upper(x)` for every `x` in the domain;
//! * each floating-point step that forms a new linear form is charged a slack
//!   of `gamma_n * (row-abs-sum * magnitude)` (Higham, Thm 3.1), where
//!   `magnitude` bounds `sum_k |a_k| * max|x_k| + |c|` over the input forms,
//!   and the slack is subtracted from a lower constant (added to an upper one)
//!   with an outward `next_down` / `next_up`;
//! * ReLU relaxations use an upper slope rounded UP and an intercept rounded UP
//!   (any `s >= u/(u-l)`, `t >= -s*l` is valid on `[l, u]`), and a lower slope
//!   in `{0, 1}` (exact);
//! * the backward objective pass charges every multiplier product and every
//!   multiplier merge against the node's rigorous concrete magnitude;
//! * #lsnc-lane: a product of two lane values uses McCormick planes built from
//!   the operands' rigorous concrete boxes, with the plane constant rounded
//!   outward ([`mc_lower`] / [`mc_upper`]) and the composed form charged like
//!   every other; an interval-valued constant (`IntervalConst`, from a model's
//!   [`LaneTwin`]) is lowered WITH its radius, which is charged against the
//!   operand's rigorous magnitude — so every published bound holds for every
//!   member of the enclosed constant set, hence for the true constant;
//! * #lsnc-lane stage 2: a program with a product intersects every affine
//!   op's concrete bounds with the outward interval image of its sources
//!   (`interval_meet`), and — armed by `LowDimConfig::joint_quadratic` — bounds
//!   each `ReduceSum(Mul)` site by the joint quadratic bound (`joint`), whose
//!   every rounding and whose eigen-decomposition residual are charged; the
//!   site's interval is intersected with it, and its planes, from the same
//!   pass and box, are substituted by a second backward pass. Every one of
//!   these is an intersection of rigorous enclosures or a maximum of rigorous
//!   lower bounds.
//! * #lowdim-bundle (each part dark behind its own typed key): an
//!   `Op::Sigmoid` is relaxed by planes whose slopes are heuristic and whose
//!   intercepts are rigorous extremes of `sigmoid(z) - a z` over the source's
//!   box (`sigmoid`), composed and charged like ReLU's upper form; a box
//!   admitted by its VARYING coordinates is the same program over the free
//!   coordinates, the pinned ones folded into outward-rounded interval
//!   biases (`pinned`); boundary alpha takes any lower ReLU slope in `[0, 1]`
//!   and publishes a maximum of rigorous bounds (`alpha`); the leaf attack is
//!   attack-only (`attack`); the shared worklist only schedules which leaves
//!   are bounded when (`shared`).
//!
//! The lane only ever claims `lower(objective · y) > threshold` over the WHOLE
//! box (every leaf closed). It never claims a violation: a leaf it cannot close
//! stops the lane and the caller falls back to the existing driver with the
//! remaining time. So the worst outcome of a defect that makes a bound too
//! LOOSE is "no proof"; only a bound that is too TIGHT could mint a false
//! unsat, which is why the tests check the published bound against sampled
//! points and against ny's own graph forward.

use std::sync::Arc;
use std::time::Instant;

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::layers::Layer;
use crate::margin_row::rounding::{gamma_n, next_down, next_up};
use crate::network::{GraphNetwork, NETWORK_INPUT};

/// Largest input dimension the lane admits. The forward forms cost `d + 1`
/// columns per neuron, so the advantage over backward intermediates shrinks as
/// `d` grows; beyond this the existing lanes are the better tool.
pub const MAX_INPUT_DIM: usize = 16;

/// Largest total neuron count (sum of every node's width) the lane admits.
pub const MAX_TOTAL_NEURONS: usize = 200_000;

/// Multiplicative safety factor applied to every magnitude before it is
/// charged. Magnitudes are sums of a few hundred rounded terms, so their own
/// relative error is below `1e-13`; `1e-6` covers it with enormous margin
/// while costing nothing measurable (the charged slacks are ~`1e-13` absolute).
const SAFETY: f64 = 1.0 + 1e-6;

/// Rigorous slack `gamma_n(n) * s` for a non-negative magnitude `s`, rounded up.
#[inline]
fn gslack(n: usize, s: f64) -> f64 {
    // Relative Higham term plus an ABSOLUTE term for the subnormal range, where
    // each of the (at most 2n + 8) products and sums can carry up to one
    // subnormal ulp that no relative bound captures.
    next_up(gamma_n(n) * (s * SAFETY) + (2 * n + 8) as f64 * f64::from_bits(1))
}

/// A value above this magnitude poisons its domain (bound published as -inf).
/// Far above any real bound, far below overflow, so no finite-looking value
/// derived from an overflowed one can slip through.
const HUGE: f64 = 1e300;

/// Largest per-chunk working set (both forms of every node) in bytes; the
/// chunk size is lowered to fit. Competition runs have no memory governor.
const CHUNK_BYTES: usize = 64 << 20;

/// How the lane chooses which input coordinate to bisect.
///
/// The serde tokens are the typed preset key's values
/// (`bab.branching.input_split.lowdim_split`, #lsnc-lane stage 3). The rule
/// only chooses WHERE to bisect: both children are always the two halves of
/// the (clipped) parent, so no rule can drop a point (`search::choose_split`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LowDimSplit {
    /// Bisect the widest coordinate (alpha-beta-CROWN's `naive` input split).
    #[default]
    #[serde(rename = "widest")]
    Widest,
    /// Bisect the coordinate maximising `|grad_k| * width_k` (smart branching).
    #[serde(rename = "gradient")]
    Gradient,
    /// #lsnc-lane stage 3: alpha-beta-CROWN's `sb_sum` score
    /// (`branching_heuristics.py`): `sum_i max(|lA_ik|, 0.01) * w_k / 2 +
    /// 0.1 * w_k * [the box touches 0 on k]` over every objective `i` of the
    /// clause, `lA_i` the input multiplier of objective `i`'s best pass —
    /// restricted by the ASPECT GUARD: only a coordinate whose width relative
    /// to the root box is at least 1/8 of the largest relative width is
    /// eligible (plain SB builds 1000-deep chains on lsnc, measured on the
    /// numpy prototype 2026-09-22).
    #[serde(rename = "sb")]
    SbSum,
    /// #lsnc-lane stage 3: the numpy gate's `sbwr` rule — the split of the
    /// measured `allmax` variant (2.0e5-1.4e6 domains per lsnc unsat row on
    /// rows 23/79/2/10, vs 3.1e5-3.7e6 for plain `sb` (no aspect guard) with
    /// the same stack, variant `allr`): `|lA_k| w_k / max_j |lA_j|
    /// w_j + rw_k / max_j rw_j` for the objective closest to closing, `rw`
    /// the width relative to the root box, under the same aspect guard.
    #[serde(rename = "sbwr")]
    SbWr,
}

impl LowDimSplit {
    /// The preset / lever token.
    pub const fn token(self) -> &'static str {
        match self {
            Self::Widest => "widest",
            Self::Gradient => "gradient",
            Self::SbSum => "sb",
            Self::SbWr => "sbwr",
        }
    }

    /// Parse a preset / lever token (`None` for anything else).
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "widest" => Some(Self::Widest),
            "gradient" => Some(Self::Gradient),
            "sb" => Some(Self::SbSum),
            "sbwr" => Some(Self::SbWr),
            _ => None,
        }
    }
}

/// #lsnc-lane stage 3: where a wave's work runs. Every mode partitions a wave
/// into the SAME chunks and decides each domain with the SAME function
/// (`search::fate`), and the merge walks the fates in domain order, so the
/// outcome and every counter are identical across modes (pinned by
/// `search_tests`); only the wall time differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LowDimParallel {
    /// Bound each wave's chunks on rayon, then decide every domain (close
    /// test, clip, violation-candidate test, split) serially on the calling
    /// thread. The historical lane.
    #[default]
    Bound,
    /// Bound AND decide each domain inside its chunk's rayon task (the typed
    /// key `bab.branching.input_split.lowdim_parallel`): the serial part of a
    /// wave shrinks to the merge.
    Full,
    /// Everything on the calling thread: the determinism tests' and the
    /// throughput bench's one-thread reference. No key selects it.
    Serial,
}

impl LowDimParallel {
    /// Observable token.
    pub const fn token(self) -> &'static str {
        match self {
            Self::Bound => "bound",
            Self::Full => "full",
            Self::Serial => "serial",
        }
    }
}

/// M8a (`#m8-input-split-batch`): which BOUNDER a lane wave uses.
///
/// The serde tokens are the typed preset key's values
/// (`bab.branching.input_split.lowdim_batch_engine`). `Bbe` hands each wave
/// to the injected UNTRUSTED forward-linear search backend (`bbe`), which
/// only CLAIMS leaves: a claimed leaf is re-bounded by the lane's own f64
/// forms and closes only on the unchanged strict `lb > thr`; every other leaf
/// is split into its exact halves. Without a registered backend, or on any
/// refusal, the unchanged legacy bounder runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LowDimBatchEngine {
    /// The lane's own f64 bounder on every domain (bit-identical to before).
    #[default]
    #[serde(rename = "legacy")]
    Legacy,
    /// Untrusted wave search claims leaves; the lane certifies the claims.
    #[serde(rename = "bbe")]
    Bbe,
}

impl LowDimBatchEngine {
    /// The preset / lever token.
    pub const fn token(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Bbe => "bbe",
        }
    }

    /// Parse a preset / lever token (`None` for anything else).
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "legacy" => Some(Self::Legacy),
            "bbe" => Some(Self::Bbe),
            _ => None,
        }
    }
}

/// M8a (`#m8-input-split-batch`): the lowdim lane's MULTI-CLAUSE scheduler.
///
/// The serde tokens are the typed preset key's values
/// (`bab.branching.input_split.engine`). `Bbe` proves same-box clauses on
/// ONE disjunction tree (`disjunction`: one forward per node for every open
/// clause, a per-leaf open-clause mask) and runs the fold's families on the
/// shared worklist (`shared`). The graph input-split drivers are untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum InputSplitEngine {
    /// One tree per clause; the fold's sequential family loop (unless
    /// `shared_worklist`).
    #[default]
    #[serde(rename = "legacy")]
    Legacy,
    /// One disjunction tree for same-box clauses; the fold's shared worklist.
    #[serde(rename = "bbe")]
    Bbe,
}

impl InputSplitEngine {
    /// The preset / lever token.
    pub const fn token(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Bbe => "bbe",
        }
    }

    /// Parse a preset / lever token (`None` for anything else).
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "legacy" => Some(Self::Legacy),
            "bbe" => Some(Self::Bbe),
            _ => None,
        }
    }
}

/// Tuning for [`prove_objective_lower_bound`].
#[derive(Debug, Clone)]
pub struct LowDimConfig {
    /// Domains bounded per wave.
    pub batch: usize,
    /// Domains per rayon task inside a wave.
    pub chunk: usize,
    /// Refuse (fall back) once this many open domains are queued.
    pub max_queue: usize,
    /// A leaf whose widest coordinate is below this is not split further; the
    /// lane stops unproved (the leaf may hold a genuine violation).
    pub min_width: f64,
    /// Split rule.
    pub split: LowDimSplit,
    /// Refine pre-activation layers no wider than this with backward CROWN
    /// (0 = off). Costs two backward passes per unstable neuron of each such
    /// layer, so it pays on NARROW, DEEP trunks where forward-linear
    /// intermediates loosen with depth (acasxu 6x50).
    pub refine_max_width: usize,
    /// #lsnc-lane stage 2: bound every `ReduceSum(Mul(a, b))` site of a
    /// program by the JOINT QUADRATIC bound (`lowdim::joint`): two CROWN
    /// certificates per site operand element, one quadratic over the input
    /// box, exact vertex maximisation of its convex part, the eigen residual
    /// charged in full; the site interval is intersected with it and its
    /// tangent planes are substituted in a second backward pass. Default off.
    /// A program without such a site is unaffected.
    pub joint_quadratic: bool,
    /// #lsnc-lane stage 3: RELAXED CLIP-AND-VERIFY. An open domain's box is
    /// shrunk to the smallest axis box containing every point that satisfies
    /// each conjunct's plane `a . x + c <= threshold` — the planes of THIS
    /// domain's own backward passes, rounded outward (`search::clip_box`);
    /// an emptied box is closed. Planes are never carried to another box.
    /// Default off.
    pub clip: bool,
    /// #lsnc-lane stage 3: where a wave's work runs (see [`LowDimParallel`]).
    pub parallel: LowDimParallel,
    /// #lowdim-bundle (1): admit a root box by its VARYING coordinates — a box
    /// wider than [`MAX_INPUT_DIM`] whose other coordinates are pinned
    /// (`lo == hi`) is re-expressed over its free ones (`pinned`) — and lower
    /// `Sigmoid` nodes to `Op::Sigmoid` (rigorous planes, `sigmoid`).
    /// Default off: such boxes and graphs are declined exactly as before.
    pub varying_dims: bool,
    /// #lowdim-bundle (2): per-leaf BOUNDARY ALPHA — projected steps on the
    /// lower ReLU slopes `alpha in [0, 1]` of the objective's backward pass,
    /// for open domains whose bound is within `alpha::ALPHA_BAND` of the
    /// threshold; the published bound is the maximum of valid bounds. This is
    /// the iteration count (0 = off, the default; clamped to
    /// `alpha::ALPHA_MAX_ITERS`).
    pub alpha_iters: usize,
    /// #lowdim-bundle (3): every this many waves, the
    /// `attack::ATTACK_LEAVES` open leaves furthest from closing get a
    /// PGD attack on the lowered program confined to each leaf box
    /// (attack-only: a hit is a CANDIDATE, confirmed by the caller's trusted
    /// forward). 0 = off, the default.
    pub leaf_attack_every: usize,
    /// #lowdim-bundle (4): prove the fold's families on ONE shared worklist
    /// (per-clause round-robin waves, no per-clause domain cap, the relaxed
    /// clip on single-objective families when `clip` is on) instead of one
    /// family after another. Default off.
    pub shared_worklist: bool,
    /// M8a: the wave BOUNDER (see [`LowDimBatchEngine`]). Default legacy.
    pub batch_engine: LowDimBatchEngine,
    /// M8a: the multi-clause SCHEDULER (see [`InputSplitEngine`]). Default
    /// legacy. The fold takes the shared worklist iff `fold_shared(cfg)`
    /// (see [`fold_shared`]).
    pub engine: InputSplitEngine,
}

impl Default for LowDimConfig {
    fn default() -> Self {
        Self {
            batch: 2048,
            chunk: 32,
            max_queue: 16_000_000,
            min_width: 1e-9,
            split: LowDimSplit::Widest,
            refine_max_width: 0,
            joint_quadratic: false,
            clip: false,
            parallel: LowDimParallel::Bound,
            varying_dims: false,
            alpha_iters: 0,
            leaf_attack_every: 0,
            shared_worklist: false,
            batch_engine: LowDimBatchEngine::Legacy,
            engine: InputSplitEngine::Legacy,
        }
    }
}

/// M8a: the fold proves its families on the shared worklist iff the typed
/// `lowdim_shared_worklist` key is on OR the multi-clause scheduler is
/// `bbe` — one pure function, printed by the CLI resolver
/// (`fold_shared=implied|explicit`). Folding the older key in is M10 work.
pub fn fold_shared(cfg: &LowDimConfig) -> bool {
    cfg.shared_worklist || cfg.engine == InputSplitEngine::Bbe
}

/// The worker threads a lane run with `cfg` uses: the current rayon pool's
/// size, or 1 for [`LowDimParallel::Serial`] (the `threads=` observable).
pub fn lane_threads(cfg: &LowDimConfig) -> u32 {
    match cfg.parallel {
        LowDimParallel::Serial => 1,
        LowDimParallel::Bound | LowDimParallel::Full => {
            u32::try_from(rayon::current_num_threads()).unwrap_or(u32::MAX)
        }
    }
}

/// #lsnc-lane stage 3: the lane configuration's engagement line, printed
/// unconditionally by the grouped entry before any clause runs:
/// `[lowdim] lane-config split=.. clip=0|1 joint=0|1 parallel=bound|full|serial
/// threads=.. refine_width=.. batch=.. chunk=.. batch_engine=legacy|bbe
/// engine=legacy|bbe` (the last two appended by M8a). A measurement that armed a
/// key whose value does not appear here is VOID, not negative.
pub fn emit_config_line(cfg: &LowDimConfig) {
    eprintln!(
        "[lowdim] lane-config split={} clip={} joint={} parallel={} threads={} \
         refine_width={} batch={} chunk={} batch_engine={} engine={}",
        cfg.split.token(),
        u8::from(cfg.clip),
        u8::from(cfg.joint_quadratic),
        cfg.parallel.token(),
        lane_threads(cfg),
        cfg.refine_max_width,
        cfg.batch,
        cfg.chunk,
        cfg.batch_engine.token(),
        cfg.engine.token()
    );
}

/// Counters for one lane run.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LowDimStats {
    /// Domains bounded (root included).
    pub domains: u64,
    /// Domains closed (`lower > threshold`, or emptied by the clip).
    pub closed: u64,
    /// Deepest split depth reached.
    pub max_depth: u32,
    /// Waves run.
    pub waves: u64,
    /// Wall seconds.
    pub secs: f64,
    /// #lsnc-lane stage 3: domains the relaxed clip EMPTIED (also counted in
    /// `closed`).
    pub clip_closed: u64,
    /// #lsnc-lane stage 3: open domains whose box the clip shrank before the
    /// split.
    pub clip_shrunk: u64,
    /// #lsnc-lane stage 3: (domain, site) pairs the joint pass bounded (its
    /// planes existed) — the joint bound's per-domain engagement count.
    pub joint_hits: u64,
    /// Joint sites the armed joint bound serves (0 when unarmed).
    pub joint_sites: u32,
    /// Worker threads ([`lane_threads`]); 0 when not recorded.
    pub threads: u32,
    /// #lowdim-bundle (2): domains the boundary alpha ran on (in band for at
    /// least one objective).
    pub alpha_domains: u64,
    /// #lowdim-bundle (2): (domain, objective) pairs whose bound alpha raised.
    pub alpha_improved: u64,
    /// #lowdim-bundle (2): domains closed that the unrefined bounds left open.
    pub alpha_rescued: u64,
    /// #lowdim-bundle (3): leaf-attack rounds run.
    pub attack_rounds: u64,
    /// #lowdim-bundle (3): due rounds the time share skipped.
    pub attack_skipped: u64,
    /// #lowdim-bundle (3): leaves attacked.
    pub attack_leaves: u64,
    /// #lowdim-bundle (3): point evaluations (forward + gradient) spent.
    pub attack_evals: u64,
    /// #lowdim-bundle (3): candidates the attack handed back (0 or 1).
    pub attack_hits: u64,
    /// #lowdim-bundle (3): wall seconds spent attacking.
    pub attack_secs: f64,
    /// M8a: domains the untrusted wave search CLAIMED (then re-bounded by
    /// the lane's own f64 forms).
    pub bbe_claimed: u64,
    /// M8a: claimed domains the lane's f64 bound closed (or clip-closed).
    pub bbe_certified: u64,
    /// M8a: claimed domains the lane's f64 bound did NOT close (split).
    pub bbe_refuted: u64,
    /// M8a router progress (kind `Volume`): the fraction of the root box
    /// proved free of counterexamples, `sum_closed prod_i (w_i / w0_i)` over
    /// the closed leaves plus the volume every clip removed (coordinates with
    /// `w0_i = 0` count 1). A counter only: nothing reads it back into a
    /// bound or a verdict.
    pub closed_volume: f64,
}

/// Why the lane stopped without a proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowDimStop {
    /// The deadline passed.
    Deadline,
    /// The open queue exceeded [`LowDimConfig::max_queue`].
    Queue,
    /// A leaf reached [`LowDimConfig::min_width`] still open.
    Resolution,
    /// A leaf's centre evaluates at or below the threshold (likely a genuine
    /// violation; the caller's attack/confirm pipeline owns that verdict).
    Candidate,
    /// A bound came back non-finite.
    NonFinite,
    /// #router-r4 CR-F: at a wave boundary (the lane's declared safe point)
    /// the acting router read two trailing FAR windows and handed the lane's
    /// unspent contract to a strictly sealed recipient. Nothing is claimed.
    HandOn,
}

impl LowDimStop {
    fn tag(self) -> &'static str {
        match self {
            Self::Deadline => "deadline",
            Self::Queue => "queue",
            Self::Resolution => "resolution",
            Self::Candidate => "candidate",
            Self::NonFinite => "nonfinite",
            Self::HandOn => "hand_on",
        }
    }
}

/// #router-r6 `rule.trial_width`: the largest wave width a lowdim width trial
/// may reach (the lane's own `CHUNK_BYTES` clamp bounds each chunk's working
/// set whatever the width; this bounds the wave's own fate vectors).
pub const LOWDIM_TRIAL_MAX_BATCH: u64 = 1 << 16;

/// #router-r6 `rule.trial_split_restart`: the OTHER split heuristic a restart
/// switches to — the aspect-guarded smart rule (`sbwr`) from the plain
/// widest-coordinate rule, and back to `widest` from any smart rule.
#[must_use]
pub const fn restart_split(split: LowDimSplit) -> LowDimSplit {
    match split {
        LowDimSplit::Widest => LowDimSplit::SbWr,
        LowDimSplit::Gradient | LowDimSplit::SbSum | LowDimSplit::SbWr => LowDimSplit::Widest,
    }
}

/// #router-r6 `rule.trial_split_restart` (review fix: one named, tested
/// function, because it is the step a proof rests on). DISCARD the tree:
/// the open queue becomes EXACTLY the root box `[lo0, hi0]` at depth 0, with
/// no parent margin (a root has none). Nothing of the discarded queue may
/// survive — a stale box left in `q_lo`/`q_hi` would be bounded in the
/// root's place, and an EMPTY queue would read as a proof at the next wave
/// top — so after a restart a proof needs the new tree to close every leaf.
pub(crate) fn restart_queue_from_root(
    q_lo: &mut Vec<f64>,
    q_hi: &mut Vec<f64>,
    q_depth: &mut Vec<u32>,
    q_parent: &mut Vec<f64>,
    lo0: &[f64],
    hi0: &[f64],
) {
    q_lo.clear();
    q_lo.extend_from_slice(lo0);
    q_hi.clear();
    q_hi.extend_from_slice(hi0);
    q_depth.clear();
    q_depth.push(0);
    q_parent.clear();
    q_parent.push(f64::NAN);
}

/// #router-r4: the router's progress cadence on the write-only bus, in waves
/// (a tap called every wave records one Volume sample per this many).
const ROUTER_BUS_STRIDE_WAVES: u64 = 4;

static ROUTER_BUS_STRIDE: ny_core::router::bus::Stride = ny_core::router::bus::Stride::new();

/// #router-r4 CR-F: lowdim's wave boundary is its declared safe point
/// (`SafePoints::BetweenUnits`, registry `lowdim`). Its progress — the
/// certified volume fraction of the root box and the open queue after
/// `stats.waves` waves, counters only — goes to the write-only bus and to the
/// FAR authority, which may answer that the lane's unspent contract was handed
/// to a strictly sealed recipient: the lane then stops HERE and claims
/// nothing. Unarmed (every run without `router.mode: on`) this is two atomic
/// loads and `false`; nothing it reads feeds back into a bound.
fn router_safe_point(stats: &LowDimStats, open: usize) -> bool {
    let sample = ny_core::far_authority::volume_sample(
        ny_core::router::decl::ids::LOWDIM,
        stats.waves,
        stats.closed_volume,
        u64::try_from(open).unwrap_or(u64::MAX),
    );
    if ROUTER_BUS_STRIDE.due(stats.waves, ROUTER_BUS_STRIDE_WAVES) {
        ny_core::router::bus::emit(sample);
    }
    ny_core::far_authority::at_safe_point(sample, stats.waves).stops()
}

/// Result of one lane run.
#[derive(Debug, Clone, PartialEq)]
pub enum LowDimOutcome {
    /// `lower(objective · y) > threshold` holds on the whole input box.
    Verified(LowDimStats),
    /// The lane stopped without a proof; nothing is claimed.
    Unproved(LowDimStats, LowDimStop),
    /// The graph or spec is outside the lane's fragment; nothing ran.
    Declined(String),
}

// ---------------------------------------------------------------- program

#[derive(Debug, Clone)]
enum Op {
    Input,
    Linear {
        src: usize,
        n_in: usize,
        /// `max(W, 0)` and `min(W, 0)`, row-major `n_out x n_in`, exact f64.
        /// Shared (`Arc`) so a folded program reuses one copy per clause.
        w_pos: Arc<[f64]>,
        w_neg: Arc<[f64]>,
        /// `W` row-major `n_out x n_in`, exact f64 (backward pass).
        w: Arc<[f64]>,
        /// The bias lies in `[bias_lo, bias_hi]` (equal unless constants were
        /// folded into it, see `fold`).
        bias_lo: Vec<f64>,
        bias_hi: Vec<f64>,
        /// `sum_j |W_ij|` per output row.
        row_abs: Arc<[f64]>,
    },
    /// Per output element `j`: `sum_t coef_t * src_t[idx_t] + k` with
    /// `k` in `[k_lo[j], k_hi[j]]`. A term's TRUE coefficient lies in
    /// `[coef - err, coef + err]`; `err` is charged against the source
    /// element's rigorous magnitude. Built only by `fold`.
    Sparse {
        terms: Vec<Vec<SparseTerm>>,
        k_lo: Vec<f64>,
        k_hi: Vec<f64>,
        /// Largest number of terms reading one source element (rounding
        /// charge of the backward accumulation).
        fanin: usize,
    },
    Relu {
        src: usize,
    },
    Slice {
        src: usize,
        start: usize,
    },
    Concat {
        srcs: Vec<usize>,
    },
    Identity {
        src: usize,
    },
    Add {
        a: usize,
        b: usize,
    },
    Sub {
        a: usize,
        b: usize,
    },
    /// `v + k`
    AddConst {
        src: usize,
        k: Vec<f64>,
    },
    /// `k - v`
    RevSubConst {
        src: usize,
        k: Vec<f64>,
    },
    /// `v * k`
    MulConst {
        src: usize,
        k: Vec<f64>,
    },
    /// `max_j v_j` over the WHOLE source vector (a global max pool). Only the
    /// same-LHS conjunction reduction builds this shape
    /// (`Reshape[1,1,k] -> MaxPool2d((1,k)) -> Reshape[1]`).
    MaxGlobal {
        src: usize,
    },
    /// #lsnc-lane: the elementwise product `a_j * b_j` of two lane values of
    /// equal width (ONNX `Mul` with two free operands, `Layer::MulBinary`).
    ///
    /// Relaxed by the McCormick envelope of each neuron on its operands'
    /// rigorous concrete box `[la, ua] x [lb, ub]`: two lower planes and two
    /// upper planes (see [`mc_lower`] / [`mc_upper`]). The forward pass keeps,
    /// per neuron and domain, the lower plane whose composed form concretizes
    /// higher and the upper plane that concretizes lower (`NodeVal::pick`);
    /// the backward pass uses the chosen LOWER plane for a non-negative
    /// multiplier and the chosen UPPER plane for a negative one. Any of the
    /// four planes is valid on the whole operand box, so the choice only
    /// affects tightness, never soundness.
    Mul {
        a: usize,
        b: usize,
    },
    /// #lowdim-bundle (1): the elementwise logistic sigmoid. Lowered only when
    /// [`LowDimConfig::varying_dims`] is armed. Relaxed per neuron and domain
    /// by the planes of `sigmoid::planes` on the source's rigorous concrete
    /// box (slopes in `[0, 1/4]`, intercepts rigorous extremes of
    /// `sigmoid(z) - a z`); the forward pass composes them with the source's
    /// forms and intersects the result with the monotone interval image
    /// `[sigmoid_down(l), sigmoid_up(u)]`; the backward pass uses the lower
    /// plane for a non-negative multiplier and the upper plane for a
    /// negative one.
    Sigmoid {
        src: usize,
    },
}

/// One term of an [`Op::Sparse`] output.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SparseTerm {
    src: u32,
    idx: u32,
    coef: f64,
    err: f64,
}

/// The graph lowered to a topologically ordered op list over flat vectors.
/// `pub(crate)` because crate-visible items (`FamClause::prog`,
/// `same_structure`, `prove_conjunction_core`) name it; its fields stay private.
#[derive(Debug, Clone)]
pub(crate) struct Program {
    ops: Vec<Op>,
    sizes: Vec<usize>,
    out: usize,
    d: usize,
    /// Upper bound on the number of rounded terms folded into the backward
    /// constant (for its accumulated-error charge).
    const_terms: usize,
    /// Sum of every node's width.
    total: usize,
    /// The node the objectives are over (the backward seed and the candidate
    /// check's evaluation point). Normally `out`; the trailing-global-max
    /// decomposition moves it to the pre-max vector.
    seed: usize,
    /// Backward-CROWN refinement of pre-activation layers no wider than this
    /// (0 = off). See `refine_node`.
    refine_max: usize,
    /// `feeds_relu[i]`: node `i` is the input of some ReLU.
    feeds_relu: Vec<bool>,
    /// #lsnc-lane: `mul_operand[i]`: node `i` is an operand of some
    /// [`Op::Mul`] (its concrete box shapes the McCormick envelope).
    mul_operand: Vec<bool>,
    /// #lsnc-lane: the program has at least one [`Op::Mul`]. Such programs —
    /// and only they, so every other program bounds bit-identically to before
    /// — run the numpy gate's `allmax` forward: every affine op's concrete
    /// bounds are INTERSECTED with the interval image of its sources'
    /// ([`interval_meet`]; on lsnc the controller's clamp is invisible to the
    /// forms and the root dV bound halves), refinement covers every
    /// non-affine neuron of a relaxed op's input (not only unstable ReLU
    /// inputs), and an objective is also bounded by the seed's forms and box.
    bilinear: bool,
    /// #lsnc-lane stage 2: the `ReduceSum(Mul)` sites ([`joint::detect_sites`]).
    sites: Vec<joint::JointSite>,
    /// `site_of[i]`: the index into `sites` of the site whose reduction is
    /// node `i`.
    site_of: Vec<Option<usize>>,
    /// `site_operand[i]`: node `i` is an operand of a site's product (it gets
    /// CROWN certificates when the joint bound is armed).
    site_operand: Vec<bool>,
    /// Arm the joint quadratic bound ([`LowDimConfig::joint_quadratic`]).
    joint: bool,
}

impl Program {
    /// (Re)derive every structural index from `ops`: the ReLU and product
    /// inputs, and the joint sites (their names are left empty).
    fn derive_structure(&mut self) {
        let n = self.ops.len();
        self.feeds_relu = vec![false; n];
        self.mul_operand = vec![false; n];
        self.bilinear = false;
        for op in &self.ops {
            match op {
                // #lowdim-bundle (1): a sigmoid input's box shapes its planes
                // exactly as a ReLU input's box shapes its relaxation, so it
                // is refined under the same rule (only programs with a
                // sigmoid, which were declined before, are affected).
                Op::Relu { src } | Op::Sigmoid { src } => self.feeds_relu[*src] = true,
                Op::Mul { a, b } => {
                    self.mul_operand[*a] = true;
                    self.mul_operand[*b] = true;
                    self.bilinear = true;
                }
                _ => {}
            }
        }
        self.sites = joint::detect_sites(&self.ops, &self.sizes);
        self.site_of = vec![None; n];
        self.site_operand = vec![false; n];
        for (s, site) in self.sites.iter().enumerate() {
            self.site_of[site.node] = Some(s);
            self.site_operand[site.a] = true;
            self.site_operand[site.b] = true;
        }
    }
}

/// What [`compile`] lowered, for the lane's engagement observable
/// (#lsnc-lane). Counts only; nothing reads them back into a bound.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CompileStats {
    /// `Layer::IntervalConst` nodes lowered (each a `Sparse` op whose
    /// coefficient or constant carries the enclosure's radius).
    interval_consts: usize,
    /// Two-operand products lowered to [`Op::Mul`].
    muls: usize,
    /// #lowdim-bundle (1): `Sigmoid` nodes lowered to [`Op::Sigmoid`].
    sigmoids: usize,
}

/// #lowdim-bundle (1): what [`compile`] may lower beyond the historical
/// fragment. The default lowers exactly the historical fragment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LowerOpts {
    /// Lower `Layer::Sigmoid` to [`Op::Sigmoid`] (otherwise it declines).
    sigmoid: bool,
    /// #m8-cgan-leaf-batch: admit a NON-vector output where the flat
    /// row-major order is still exact: `Reshape` / `Flatten` / `Squeeze` /
    /// `Unsqueeze` with equal element counts (a pure relabel), and a `Concat`
    /// whose dims BEFORE its axis multiply to 1 (then the output's flat data
    /// is its inputs' flat data concatenated). Otherwise these decline as
    /// before, and every other op still needs a vector output.
    flat_reshape: bool,
}

impl LowerOpts {
    fn of(cfg: &LowDimConfig) -> Self {
        Self {
            sigmoid: cfg.varying_dims,
            flat_reshape: cfg.varying_dims,
        }
    }
}

/// #m8-cgan-leaf-batch: the concat axis of `ins` into `out` read from the
/// REAL shapes (the loader may have shifted the stored axis): the unique axis
/// along which the inputs' extents add up to the output's while every other
/// extent matches. `None` when the shapes do not describe one concat.
fn concat_axis_of(ins: &[&Vec<usize>], out: &[usize]) -> Option<usize> {
    if ins.is_empty() || ins.iter().any(|s| s.len() != out.len()) {
        return None;
    }
    let mut found = None;
    for axis in 0..out.len() {
        let others_match = ins.iter().all(|s| {
            s.iter()
                .zip(out)
                .enumerate()
                .all(|(k, (a, b))| k == axis || a == b)
        });
        let sums = ins.iter().map(|s| s[axis]).sum::<usize>() == out[axis];
        if others_match && sums {
            if found.is_some() {
                return None;
            }
            found = Some(axis);
        }
    }
    found
}

/// McCormick LOWER plane `k` (0 or 1) of `z = a * b` on the operand box
/// `[la, ua] x [lb, ub]`: `(alpha, beta, gamma)` with
/// `z >= alpha * a + beta * b + gamma` for every `(a, b)` in the box.
///
/// * plane 0 is `(a - la)(b - lb) >= 0`: `z >= lb a + la b - la lb`;
/// * plane 1 is `(ua - a)(ub - b) >= 0`: `z >= ub a + ua b - ua ub`.
///
/// `alpha` and `beta` are operand bounds themselves (exact); only the product
/// in `gamma` rounds, and `next_down` of the negated round-to-nearest product
/// is at or below the exact `-la lb` (the gap below a rounded value is at
/// least the half-gap its rounding error can reach).
#[inline]
fn mc_lower(la: f64, ua: f64, lb: f64, ub: f64, k: u8) -> (f64, f64, f64) {
    if k == 0 {
        (lb, la, next_down(-(la * lb)))
    } else {
        (ub, ua, next_down(-(ua * ub)))
    }
}

/// McCormick UPPER plane `k` (0 or 1) of `z = a * b` on `[la, ua] x [lb, ub]`:
/// `z <= alpha * a + beta * b + gamma` for every `(a, b)` in the box.
///
/// * plane 0 is `(a - la)(ub - b) >= 0`: `z <= ub a + la b - la ub`;
/// * plane 1 is `(ua - a)(b - lb) >= 0`: `z <= lb a + ua b - ua lb`.
///
/// `gamma` is rounded UP (mirror of [`mc_lower`]).
#[inline]
fn mc_upper(la: f64, ua: f64, lb: f64, ub: f64, k: u8) -> (f64, f64, f64) {
    if k == 0 {
        (ub, la, next_up(-(la * ub)))
    } else {
        (lb, ua, next_up(-(ua * lb)))
    }
}

/// Write `alpha * A + beta * B` into `out` (`d + 1` columns, constant last),
/// plus `gamma` in the constant column, where `A` / `B` are the operand forms
/// that bound `alpha * a` / `beta * b` in the requested direction: for a LOWER
/// form, a non-negative coefficient takes the operand's lower form and a
/// negative one its upper form (and the reverse for an UPPER form). Returns
/// the magnitude `|alpha| s_A + |beta| s_B + |gamma|` of the ingredients, which
/// bounds the rounding of this form at any point of the domain (the caller
/// charges `gslack` of it).
#[allow(clippy::too_many_arguments)]
fn mul_form(
    out: &mut [f64],
    sa: &NodeVal,
    sb: &NodeVal,
    i: usize,
    d: usize,
    (alpha, beta, gamma): (f64, f64, f64),
    upper: bool,
) -> f64 {
    let d1 = d + 1;
    let r = i * d1;
    let a_lo = (alpha >= 0.0) != upper;
    let b_lo = (beta >= 0.0) != upper;
    let (fa, s_a) = if a_lo {
        (&sa.lo_form[r..r + d1], sa.s_lo[i])
    } else {
        (&sa.hi_form[r..r + d1], sa.s_hi[i])
    };
    let (fb, s_b) = if b_lo {
        (&sb.lo_form[r..r + d1], sb.s_lo[i])
    } else {
        (&sb.hi_form[r..r + d1], sb.s_hi[i])
    };
    for k in 0..d1 {
        out[k] = alpha * fa[k] + beta * fb[k];
    }
    out[d] += gamma;
    alpha.abs() * s_a + beta.abs() * s_b + gamma.abs()
}

/// Plain (unrounded) extreme of a form over a domain box: its minimum for a
/// LOWER form, its maximum for an UPPER one. Used only to CHOOSE between
/// McCormick planes; the published value is always the rigorously rounded
/// [`concretize`].
fn form_extreme(f: &[f64], lo: &[f64], hi: &[f64], d: usize, upper: bool) -> f64 {
    let mut v = f[d];
    for k in 0..d {
        let a = f[k];
        v += if (a >= 0.0) != upper {
            a * lo[k]
        } else {
            a * hi[k]
        };
    }
    v
}

/// `(max(W,0), min(W,0), row |W| sums)` of a row-major `n_out x n_in` `W`.
fn linear_parts(w: &[f64], n_out: usize, n_in: usize) -> (Arc<[f64]>, Arc<[f64]>, Arc<[f64]>) {
    let w_pos: Arc<[f64]> = w.iter().map(|&v| v.max(0.0)).collect();
    let w_neg: Arc<[f64]> = w.iter().map(|&v| v.min(0.0)).collect();
    let row_abs: Arc<[f64]> = (0..n_out)
        .map(|i| w[i * n_in..(i + 1) * n_in].iter().map(|v| v.abs()).sum())
        .collect();
    (w_pos, w_neg, row_abs)
}

fn broadcast(constant: &ndarray::ArrayD<f32>, n: usize, what: &str) -> Result<Vec<f64>, String> {
    let len = constant.len();
    if len == 1 {
        let v = f64::from(*constant.iter().next().expect("len 1"));
        Ok(vec![v; n])
    } else if len == n {
        Ok(constant.iter().map(|&v| f64::from(v)).collect())
    } else {
        Err(format!(
            "{what} constant has {len} elements for a width-{n} operand"
        ))
    }
}

/// At most one dimension larger than 1: the tensor's row-major flat order is
/// then its only ordering, which is what every op below relies on.
fn is_vector(shape: &[usize]) -> bool {
    shape.iter().filter(|&&n| n > 1).count() <= 1
}

/// Lower the graph to flat-vector ops.
///
/// `shapes` holds every node's REAL output shape (from one IBP pass on the
/// root box), including `NETWORK_INPUT`. Admission is decided from those
/// shapes, not from stored axis numbers: the ONNX loader may squeeze a leading
/// batch dimension and rewrite an axis (`ConcatLayer::restored_batch_axis_shift`),
/// so an axis alone does not say which flat elements an op touches. Every node
/// must produce a vector, which makes Slice a contiguous flat range, Concat a
/// flat concatenation, and Reshape/Flatten/Squeeze/Unsqueeze flat identities.
///
/// #lsnc-lane adds the Lyapunov-certificate fragment: `MulBinary` of two lane
/// values ([`Op::Mul`], McCormick), `ReduceSum` and constant-index `Gather`
/// (exact 0/1 [`Op::Sparse`] rows), `DivConstant` (an exact reciprocal for a
/// power of two, else an outward-rounded coefficient interval), and the two
/// `IntervalConst` shapes (`Linear`: a `Sparse` row whose coefficients carry
/// the enclosure's radius as their error, charged against the operand's
/// rigorous magnitude; `Offset`: a unit `Sparse` row whose constant is the
/// outward interval `[c - r, c + r]`). `Neg` arrives as `MulConstant(-1)` and a
/// `Cast` to FLOAT is dropped by the loader, so neither needs an arm.
fn compile(
    graph: &GraphNetwork,
    d: usize,
    shapes: &std::collections::HashMap<String, Vec<usize>>,
    opts: LowerOpts,
) -> Result<(Program, CompileStats), String> {
    let mut stats = CompileStats::default();
    let order = graph
        .topological_sort()
        .map_err(|e| format!("topological sort failed: {e}"))?;
    let shape_of = |name: &str| -> Result<&Vec<usize>, String> {
        shapes
            .get(name)
            .ok_or_else(|| format!("no shape recorded for `{name}`"))
    };
    let mut index = std::collections::HashMap::<String, usize>::new();
    let mut ops = vec![Op::Input];
    let mut sizes = vec![d];
    index.insert(NETWORK_INPUT.to_string(), 0);
    let lookup = |index: &std::collections::HashMap<String, usize>, name: &str| {
        index
            .get(name)
            .copied()
            .ok_or_else(|| format!("input `{name}` is not produced before use"))
    };
    for name in &order {
        if name == NETWORK_INPUT {
            continue;
        }
        let node = graph
            .node(name)
            .ok_or_else(|| format!("node `{name}` missing"))?;
        let ins = node.inputs();
        let one = |i: usize| -> Result<usize, String> {
            ins.get(i)
                .ok_or_else(|| format!("node `{name}` lacks input {i}"))
                .and_then(|s| lookup(&index, s))
        };
        // Set by an arm whose flat row-major meaning is exact for ANY output
        // shape (#m8-cgan-leaf-batch, armed only).
        let mut flat_relabel = false;
        let (op, size) = match node.layer() {
            Layer::Linear(l) => {
                let src = one(0)?;
                let w32 = l.weight();
                let (n_out, n_in) = (w32.nrows(), w32.ncols());
                let src_shape = shape_of(&ins[0])?;
                if n_in != sizes[src] || src_shape.last() != Some(&n_in) {
                    return Err(format!(
                        "Linear `{name}` expects {n_in} inputs along the last axis, operand is {src_shape:?}"
                    ));
                }
                let mut w = Vec::with_capacity(n_out * n_in);
                for i in 0..n_out {
                    for j in 0..n_in {
                        w.push(f64::from(w32[(i, j)]));
                    }
                }
                if w.iter().any(|v| !v.is_finite()) {
                    return Err(format!("Linear `{name}` has a non-finite weight"));
                }
                let bias: Vec<f64> = match l.bias() {
                    Some(b) => b.iter().map(|&v| f64::from(v)).collect(),
                    None => vec![0.0; n_out],
                };
                if bias.len() != n_out || bias.iter().any(|v| !v.is_finite()) {
                    return Err(format!("Linear `{name}` has a bad bias"));
                }
                let (w_pos, w_neg, row_abs) = linear_parts(&w, n_out, n_in);
                (
                    Op::Linear {
                        src,
                        n_in,
                        w_pos,
                        w_neg,
                        w: w.into(),
                        bias_lo: bias.clone(),
                        bias_hi: bias,
                        row_abs,
                    },
                    n_out,
                )
            }
            Layer::ReLU(_) => {
                let src = one(0)?;
                (Op::Relu { src }, sizes[src])
            }
            Layer::Slice(s) => {
                let src = one(0)?;
                let src_shape = shape_of(&ins[0])?;
                let (axis, start, end) = s
                    .resolved_range(src_shape)
                    .map_err(|e| format!("Slice `{name}`: {e}"))?;
                let off_axis_trivial = src_shape
                    .iter()
                    .enumerate()
                    .all(|(i, &n)| i == axis || n == 1);
                if !off_axis_trivial || end <= start {
                    return Err(format!(
                        "Slice `{name}` on {src_shape:?} axis {axis} is not a contiguous flat range"
                    ));
                }
                (Op::Slice { src, start }, end - start)
            }
            Layer::Concat(c) => {
                if c.constant_inputs
                    .as_ref()
                    .is_some_and(|v| v.iter().any(Option::is_some))
                {
                    return Err(format!("Concat `{name}` has a constant operand"));
                }
                let srcs = (0..ins.len()).map(one).collect::<Result<Vec<_>, _>>()?;
                let size = srcs.iter().map(|&s| sizes[s]).sum();
                if opts.flat_reshape && !is_vector(shape_of(name)?) {
                    // #m8-cgan-leaf-batch: an outer-axis concat is a flat
                    // concatenation; an inner-axis one interleaves (decline).
                    let out_shape = shape_of(name)?;
                    let in_shapes = ins
                        .iter()
                        .map(|i| shape_of(i))
                        .collect::<Result<Vec<_>, _>>()?;
                    let axis = concat_axis_of(&in_shapes, out_shape).ok_or_else(|| {
                        format!("Concat `{name}`: shapes {in_shapes:?} -> {out_shape:?} are not one concat")
                    })?;
                    if out_shape[..axis].iter().product::<usize>() != 1 {
                        return Err(format!(
                            "Concat `{name}` along axis {axis} of {out_shape:?} interleaves (not a flat concat)"
                        ));
                    }
                    flat_relabel = true;
                }
                (Op::Concat { srcs }, size)
            }
            Layer::MaxPool2d(p) => {
                let src = one(0)?;
                let src_shape = shape_of(&ins[0])?;
                let k = sizes[src];
                // Only a pool whose single window covers the WHOLE vector.
                if p.padding != (0, 0) || p.kernel_size.0 * p.kernel_size.1 != k || k == 0 {
                    return Err(format!(
                        "MaxPool2d `{name}` on {src_shape:?} is not a global max over the vector"
                    ));
                }
                (Op::MaxGlobal { src }, 1)
            }
            Layer::Reshape(_) | Layer::Flatten(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_) => {
                let src = one(0)?;
                // #m8-cgan-leaf-batch: armed, any equal-count relabel (the
                // element-count check below still applies).
                flat_relabel = opts.flat_reshape;
                (Op::Identity { src }, sizes[src])
            }
            Layer::Add(_) | Layer::Sub(_) => {
                if ins.len() != 2 {
                    return Err(format!("binary `{name}` has {} inputs", ins.len()));
                }
                let (a, b) = (one(0)?, one(1)?);
                if sizes[a] != sizes[b] || shape_of(&ins[0])? != shape_of(&ins[1])? {
                    return Err(format!("binary `{name}` broadcasts; not admitted"));
                }
                let op = if matches!(node.layer(), Layer::Add(_)) {
                    Op::Add { a, b }
                } else {
                    Op::Sub { a, b }
                };
                (op, sizes[a])
            }
            Layer::AddConstant(c) => {
                let src = one(0)?;
                let k = broadcast(&c.constant, sizes[src], "AddConstant")?;
                (Op::AddConst { src, k }, sizes[src])
            }
            Layer::SubConstant(c) => {
                let src = one(0)?;
                let k = broadcast(&c.constant, sizes[src], "SubConstant")?;
                if c.reverse {
                    (Op::RevSubConst { src, k }, sizes[src])
                } else {
                    let k = k.into_iter().map(|v| -v).collect();
                    (Op::AddConst { src, k }, sizes[src])
                }
            }
            Layer::MulConstant(c) => {
                let src = one(0)?;
                let k = broadcast(&c.constant, sizes[src], "MulConstant")?;
                (Op::MulConst { src, k }, sizes[src])
            }
            // #lsnc-lane lowerings (see this function's doc).
            Layer::DivConstant(c) => {
                let src = one(0)?;
                let k = broadcast(&c.constant, sizes[src], "DivConstant")?;
                (lower_div_constant(src, &k, name)?, sizes[src])
            }
            Layer::MulBinary(_) => {
                if ins.len() != 2 {
                    return Err(format!("MulBinary `{name}` has {} inputs", ins.len()));
                }
                let (a, b) = (one(0)?, one(1)?);
                if sizes[a] != sizes[b] || shape_of(&ins[0])? != shape_of(&ins[1])? {
                    return Err(format!("MulBinary `{name}` broadcasts; not admitted"));
                }
                stats.muls += 1;
                (Op::Mul { a, b }, sizes[a])
            }
            Layer::ReduceSum(r) => {
                let src = one(0)?;
                let in_shape = shape_of(&ins[0])?.clone();
                let out_shape = shape_of(name)?.clone();
                lower_reduce_sum(src, sizes[src], &in_shape, &out_shape, r, name)?
            }
            Layer::Gather(g) => {
                let src = one(0)?;
                let in_shape = shape_of(&ins[0])?.clone();
                lower_gather(src, sizes[src], &in_shape, g, name)?
            }
            Layer::IntervalConst(ic) => {
                let src = one(0)?;
                let src_shape = shape_of(&ins[0])?.clone();
                stats.interval_consts += 1;
                lower_interval_const(src, sizes[src], &src_shape, ic, name)?
            }
            // #lowdim-bundle (1): only when armed; unarmed, a Sigmoid falls
            // to the arm below and declines exactly as before.
            Layer::Sigmoid(_) if opts.sigmoid => {
                let src = one(0)?;
                stats.sigmoids += 1;
                (Op::Sigmoid { src }, sizes[src])
            }
            other => {
                return Err(format!(
                    "node `{name}` is {} (outside the lane's fragment)",
                    other.layer_type()
                ))
            }
        };
        let real = shape_of(name)?;
        if (!flat_relabel && !is_vector(real)) || real.iter().product::<usize>() != size {
            return Err(format!(
                "node `{name}` produces {real:?}, not a width-{size} vector"
            ));
        }
        if let Op::AddConst { k, .. } | Op::RevSubConst { k, .. } | Op::MulConst { k, .. } = &op {
            if k.iter().any(|v| !v.is_finite()) {
                return Err(format!("node `{name}` has a non-finite constant"));
            }
        }
        if let Op::Sparse {
            terms, k_lo, k_hi, ..
        } = &op
        {
            let bad_term = terms.iter().flatten().any(|t| {
                !t.coef.is_finite()
                    || !t.err.is_finite()
                    || t.err < 0.0
                    || t.src as usize >= ops.len()
                    || t.idx as usize >= sizes[t.src as usize]
            });
            let bad_k = k_lo.len() != size
                || k_hi.len() != size
                || terms.len() != size
                || k_lo
                    .iter()
                    .zip(k_hi)
                    .any(|(l, h)| !(l.is_finite() && h.is_finite() && l <= h));
            if bad_term || bad_k {
                return Err(format!("node `{name}` lowered to a malformed sparse row"));
            }
        }
        index.insert(name.clone(), ops.len());
        ops.push(op);
        sizes.push(size);
    }
    let out = lookup(&index, graph.output_name())?;
    let total: usize = sizes.iter().sum();
    if total > MAX_TOTAL_NEURONS {
        return Err(format!(
            "{total} neurons exceed the lane cap {MAX_TOTAL_NEURONS}"
        ));
    }
    let mut prog = Program {
        ops,
        sizes,
        out,
        d,
        const_terms: 4 * total + 64,
        total,
        seed: out,
        refine_max: 0,
        feeds_relu: Vec::new(),
        mul_operand: Vec::new(),
        bilinear: false,
        sites: Vec::new(),
        site_of: Vec::new(),
        site_operand: Vec::new(),
        joint: false,
    };
    prog.derive_structure();
    // Name the joint sites after their graph nodes (observability only).
    let mut names = vec![String::new(); prog.ops.len()];
    for (name, &i) in &index {
        if let Some(slot) = names.get_mut(i) {
            slot.clone_from(name);
        }
    }
    for site in &mut prog.sites {
        site.name = std::mem::take(&mut names[site.node]);
    }
    Ok((prog, stats))
}

/// An [`Op::Sparse`] over `terms` (one row per output element) with constant
/// intervals `[k_lo, k_hi]`; `fanin` is the most terms reading one source
/// element (the backward accumulation's rounding charge).
fn sparse_op(terms: Vec<Vec<SparseTerm>>, k_lo: Vec<f64>, k_hi: Vec<f64>) -> Op {
    let mut reads: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
    for t in terms.iter().flatten() {
        *reads.entry((t.src, t.idx)).or_default() += 1;
    }
    let fanin = reads.values().copied().max().unwrap_or(1);
    Op::Sparse {
        terms,
        k_lo,
        k_hi,
        fanin,
    }
}

/// A sparse term reading `src[idx]` with coefficient `coef` whose TRUE value
/// lies in `[coef - err, coef + err]`.
fn term(src: usize, idx: usize, coef: f64, err: f64) -> Result<SparseTerm, String> {
    let (Ok(src), Ok(idx)) = (u32::try_from(src), u32::try_from(idx)) else {
        return Err("sparse term index exceeds u32".into());
    };
    Ok(SparseTerm {
        src,
        idx,
        coef,
        err,
    })
}

/// `1 / k` when it is EXACT in f64: `k` a normal power of two whose reciprocal
/// is normal (then `1 / k` is a power of two too, and `r * k == 1` confirms
/// no rounding happened).
fn exact_reciprocal(k: f64) -> Option<f64> {
    const MANTISSA: u64 = (1u64 << 52) - 1;
    if !k.is_normal() || k.to_bits() & MANTISSA != 0 {
        return None;
    }
    let r = 1.0 / k;
    (r.is_normal() && r * k == 1.0 && r.to_bits() & MANTISSA == 0).then_some(r)
}

/// `DivConstant`: `y_j = x_j / k_j`. Every divisor a power of two lowers to an
/// EXACT [`Op::MulConst`] by the reciprocal. Otherwise each element becomes a
/// one-term sparse row whose coefficient encloses the true `1 / k_j`: the
/// round-to-nearest quotient `q` is within half a gap of it, so
/// `[next_down(q), next_up(q)]` contains it, carried as midpoint and radius
/// (the radius is charged against `|x_j|` like every coefficient error).
fn lower_div_constant(src: usize, k: &[f64], name: &str) -> Result<Op, String> {
    if k.iter().any(|&v| !v.is_finite() || v == 0.0) {
        return Err(format!(
            "DivConstant `{name}` divides by zero or a non-finite value"
        ));
    }
    let exact: Option<Vec<f64>> = k.iter().map(|&v| exact_reciprocal(v)).collect();
    if let Some(recip) = exact {
        return Ok(Op::MulConst { src, k: recip });
    }
    let mut terms = Vec::with_capacity(k.len());
    for (j, &kj) in k.iter().enumerate() {
        let q = 1.0 / kj;
        let (mid, rad) = fold::mid_rad(next_down(q), next_up(q));
        terms.push(vec![term(src, j, mid, rad)?]);
    }
    let zeros = vec![0.0; k.len()];
    Ok(sparse_op(terms, zeros.clone(), zeros))
}

/// `ReduceSum` over a vector: each output element is the EXACT 0/1 sum of the
/// input elements its reduced axes collapse onto (axes decided by the real
/// shapes, cross-checked against the layer's own axes as `#lowdim-fold` does).
fn lower_reduce_sum(
    src: usize,
    n_in: usize,
    in_shape: &[usize],
    out_shape: &[usize],
    r: &crate::layers::ReduceSumLayer,
    name: &str,
) -> Result<(Op, usize), String> {
    if in_shape.iter().product::<usize>() != n_in {
        return Err(format!(
            "ReduceSum `{name}`: {in_shape:?} is not a width-{n_in} vector"
        ));
    }
    let axes = fold::reduce_axes(in_shape, out_shape, &r.axes, r.keepdims).ok_or_else(|| {
        format!("ReduceSum `{name}`: axes not determined by {in_shape:?} -> {out_shape:?}")
    })?;
    let mut kept = in_shape.to_vec();
    for &a in &axes {
        kept[a] = 1;
    }
    let n_out: usize = kept.iter().product();
    let kst = fold::strides(&kept);
    let mut terms: Vec<Vec<SparseTerm>> = vec![Vec::new(); n_out];
    for f in 0..n_in {
        let mut o = fold::unravel(f, in_shape);
        for &a in &axes {
            o[a] = 0;
        }
        let of: usize = o.iter().zip(&kst).map(|(a, b)| a * b).sum();
        terms[of].push(term(src, f, 1.0, 0.0)?);
    }
    let zeros = vec![0.0; n_out];
    Ok((sparse_op(terms, zeros.clone(), zeros), n_out))
}

/// Constant-index `Gather` over a vector: output `j` is exactly input
/// `indices[j]` (negative indices wrap). The gather axis must be the vector's
/// only non-unit axis — resolved against the REAL input rank, allowing the
/// loader's squeezed batch dimension (`axis - 1`), and refused when no
/// candidate axis spans the whole vector.
fn lower_gather(
    src: usize,
    n_in: usize,
    in_shape: &[usize],
    g: &crate::layers::GatherLayer,
    name: &str,
) -> Result<(Op, usize), String> {
    if g.is_runtime_last_axis_len() {
        return Err(format!("Gather `{name}` is a runtime shape query"));
    }
    let indices = g
        .constant_indices()
        .ok_or_else(|| format!("Gather `{name}` has dynamic indices"))?;
    if in_shape.iter().product::<usize>() != n_in || n_in == 0 {
        return Err(format!(
            "Gather `{name}`: {in_shape:?} is not a width-{n_in} vector"
        ));
    }
    let rank = i64::try_from(in_shape.len()).map_err(|_| format!("Gather `{name}`: rank"))?;
    let raw = g.axis_raw();
    let spans = [raw, raw - 1].into_iter().any(|a| {
        let a = if a < 0 { a + rank } else { a };
        (0..rank).contains(&a) && usize::try_from(a).is_ok_and(|a| in_shape[a] == n_in)
    });
    if !spans {
        return Err(format!(
            "Gather `{name}` axis {raw} on {in_shape:?} does not select along the vector"
        ));
    }
    let n = i64::try_from(n_in).map_err(|_| format!("Gather `{name}`: width"))?;
    let mut terms = Vec::with_capacity(indices.len());
    for &ix in indices {
        let j = if ix < 0 { ix + n } else { ix };
        if !(0..n).contains(&j) {
            return Err(format!(
                "Gather `{name}` index {ix} out of range for {n_in}"
            ));
        }
        let j = usize::try_from(j).map_err(|_| format!("Gather `{name}`: index"))?;
        terms.push(vec![term(src, j, 1.0, 0.0)?]);
    }
    let n_out = terms.len();
    let zeros = vec![0.0; n_out];
    Ok((sparse_op(terms, zeros.clone(), zeros), n_out))
}

/// `IntervalConst`: the enclosure lowers WITH its radius.
///
/// * `Linear` (`y = x W^T`, `W` in `Wc +- R`): row `i` is
///   `sum_j Wc[i, j] x_j + bias_i` with each term's coefficient error
///   `R[i, j]`, which the sparse op charges as `R[i, j] * max|x_j|` in the
///   forward form and `|lambda_i| * R[i, j] * max|x_j|` in the backward pass —
///   rigorous for EVERY member of the weight set, hence for the true weight.
/// * `Offset` (`y = x + c`, `c` in `center +- radius`, ONNX-broadcast into the
///   activation): a unit row whose constant is `[next_down(c - r),
///   next_up(c + r)]`.
fn lower_interval_const(
    src: usize,
    n_src: usize,
    src_shape: &[usize],
    ic: &crate::layers::IntervalConstLayer,
    name: &str,
) -> Result<(Op, usize), String> {
    use crate::layers::interval_const::IntervalConstView;
    match ic.lane_view() {
        IntervalConstView::Linear { center, radius } => {
            let w = center.weight();
            let (n_out, n_in) = (w.nrows(), w.ncols());
            if n_in != n_src || src_shape.last() != Some(&n_in) || radius.dim() != (n_out, n_in) {
                return Err(format!(
                    "IntervalLinear `{name}` expects {n_in} inputs along the last axis, operand is {src_shape:?}"
                ));
            }
            let bias: Vec<f64> = match center.bias() {
                Some(b) => b.iter().map(|&v| f64::from(v)).collect(),
                None => vec![0.0; n_out],
            };
            if bias.len() != n_out {
                return Err(format!("IntervalLinear `{name}` has a bad bias"));
            }
            let mut terms = Vec::with_capacity(n_out);
            for i in 0..n_out {
                let mut row = Vec::new();
                for j in 0..n_in {
                    let (c, r) = (f64::from(w[(i, j)]), f64::from(radius[(i, j)]));
                    if c != 0.0 || r != 0.0 {
                        row.push(term(src, j, c, r)?);
                    }
                }
                terms.push(row);
            }
            Ok((sparse_op(terms, bias.clone(), bias), n_out))
        }
        IntervalConstView::Offset {
            center,
            radius,
            input_len,
        } => {
            if input_len != n_src || src_shape.iter().product::<usize>() != n_src {
                return Err(format!(
                    "IntervalOffset `{name}` built for {input_len} elements, operand is {src_shape:?}"
                ));
            }
            let map = fold::broadcast_map(center.shape(), src_shape)
                .map_err(|e| format!("IntervalOffset `{name}`: {e}"))?;
            if map.len() != n_src || radius.shape() != center.shape() {
                return Err(format!("IntervalOffset `{name}` does not tile its operand"));
            }
            let c: Vec<f64> = center.iter().map(|&v| f64::from(v)).collect();
            let r: Vec<f64> = radius.iter().map(|&v| f64::from(v)).collect();
            let mut terms = Vec::with_capacity(n_src);
            let (mut k_lo, mut k_hi) = (Vec::with_capacity(n_src), Vec::with_capacity(n_src));
            for (j, &from) in map.iter().enumerate() {
                terms.push(vec![term(src, j, 1.0, 0.0)?]);
                k_lo.push(next_down(c[from] - r[from]));
                k_hi.push(next_up(c[from] + r[from]));
            }
            Ok((sparse_op(terms, k_lo, k_hi), n_src))
        }
    }
}

// ---------------------------------------------------------------- forward

/// One node's value over a chunk of `m` domains.
///
/// Forms are row-major `[n][m][d + 1]`; column `d` is the constant.
#[derive(Debug, Clone)]
struct NodeVal {
    lo_form: Vec<f64>,
    hi_form: Vec<f64>,
    /// Rigorous concrete bounds `[n][m]`.
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// Per-row magnitude `sum_k |a_k| X_k + |c|` of the lower / upper form.
    s_lo: Vec<f64>,
    s_hi: Vec<f64>,
    /// Per-domain `max_j max(s_lo, s_hi)` (forms' magnitude).
    mag: Vec<f64>,
    /// Per-domain `max_j max(|lo_j|, |hi_j|)` (value magnitude).
    zmax: Vec<f64>,
    /// Per-domain poison: any non-finite or > HUGE bound or magnitude. A
    /// poisoned domain's published bound is -inf (fail closed): `f64::max`
    /// and `min` silently DROP a NaN operand, so a NaN must never be allowed
    /// to reach a magnitude or an intersection unflagged.
    bad: Vec<bool>,
    /// [`Op::Mul`] only, `[n][m]`: bit 0 is the McCormick LOWER plane the
    /// forward pass kept, bit 1 the UPPER plane (see [`Op::Mul`]). Empty for
    /// every other op.
    pick: Vec<u8>,
    /// #lsnc-lane stage 2, joint-site OPERANDS only (`[n][m][d + 1]`, the
    /// forms' layout): the CROWN lower / upper planes of each element — a
    /// backward pass seeded at this node with `+e_i` / `-e_i` over this
    /// chunk's own forward pass, valid on the domain's box. An element the
    /// refinement skipped (its forms are exactly affine) carries its forward
    /// forms. Empty for every other node.
    cert_lo: Vec<f64>,
    cert_hi: Vec<f64>,
    /// #lsnc-lane stage 2, joint SITES only (`[m][d + 1]`): the joint pass's
    /// lower / upper plane of the site on each domain (`joint_ok[b]`), from
    /// THIS forward pass and box — the planes the backward pass substitutes.
    joint_lo: Vec<f64>,
    joint_hi: Vec<f64>,
    joint_ok: Vec<bool>,
    /// #lowdim-bundle (1), [`Op::Sigmoid`] only, `[n][m]`: the planes
    /// `[a_lo, b_lo, a_hi, b_hi]` the forward pass built from the source's
    /// box (NaN where none exists; that domain is poisoned). The backward pass
    /// reuses them. Empty for every other op.
    sig: Vec<[f64; 4]>,
}

/// A chunk of domains: `lo`/`hi` are `[m][d]`, `xmax` is `[m][d]`.
struct Chunk<'a> {
    m: usize,
    d: usize,
    lo: &'a [f64],
    hi: &'a [f64],
    xmax: Vec<f64>,
}

impl<'a> Chunk<'a> {
    fn new(m: usize, d: usize, lo: &'a [f64], hi: &'a [f64]) -> Self {
        let xmax = lo
            .iter()
            .zip(hi)
            .map(|(l, h)| l.abs().max(h.abs()))
            .collect();
        Self { m, d, lo, hi, xmax }
    }
}

/// Fill `lo/hi/s_lo/s_hi/mag/zmax` of a node from its forms.
fn concretize(v: &mut NodeVal, n: usize, ch: &Chunk<'_>) {
    let (m, d) = (ch.m, ch.d);
    let d1 = d + 1;
    v.lo.resize(n * m, 0.0);
    v.hi.resize(n * m, 0.0);
    v.s_lo.resize(n * m, 0.0);
    v.s_hi.resize(n * m, 0.0);
    v.mag = vec![0.0; m];
    v.zmax = vec![0.0; m];
    v.bad = vec![false; m];
    for j in 0..n {
        for b in 0..m {
            let row = (j * m + b) * d1;
            let (bl, bh, bx) = (
                &ch.lo[b * d..(b + 1) * d],
                &ch.hi[b * d..(b + 1) * d],
                &ch.xmax[b * d..(b + 1) * d],
            );
            let fl = &v.lo_form[row..row + d1];
            let fh = &v.hi_form[row..row + d1];
            let (mut vl, mut sl) = (fl[d], fl[d].abs());
            let (mut vh, mut sh) = (fh[d], fh[d].abs());
            for k in 0..d {
                let a = fl[k];
                vl += if a >= 0.0 { a * bl[k] } else { a * bh[k] };
                sl += a.abs() * bx[k];
                let a = fh[k];
                vh += if a >= 0.0 { a * bh[k] } else { a * bl[k] };
                sh += a.abs() * bx[k];
            }
            let lo = next_down(vl - gslack(d + 2, sl));
            let hi = next_up(vh + gslack(d + 2, sh));
            let i = j * m + b;
            v.lo[i] = lo;
            v.hi[i] = hi;
            v.s_lo[i] = sl;
            v.s_hi[i] = sh;
            v.mag[b] = v.mag[b].max(sl).max(sh);
            v.zmax[b] = v.zmax[b].max(lo.abs()).max(hi.abs());
            let finite = lo.is_finite() && hi.is_finite() && sl.is_finite() && sh.is_finite();
            if !finite || lo.abs() > HUGE || hi.abs() > HUGE || sl > HUGE || sh > HUGE {
                v.bad[b] = true;
            }
        }
    }
}

/// For a global max over `k` source rows at domain `b`: the branch with the
/// largest concrete lower bound, the largest concrete upper bound, and whether
/// that branch DOMINATES (its lower bound is at least every other branch's upper
/// bound, so the max IS that branch). Deterministic, so forward and backward
/// make the same choice.
fn max_global_choice(s: &NodeVal, k: usize, m: usize, b: usize) -> (usize, f64, bool) {
    let mut jstar = 0;
    for j in 1..k {
        if s.lo[j * m + b] > s.lo[jstar * m + b] {
            jstar = j;
        }
    }
    let lo_star = s.lo[jstar * m + b];
    let mut maxhi = f64::NEG_INFINITY;
    let mut dominated = true;
    for j in 0..k {
        let hi = s.hi[j * m + b];
        maxhi = maxhi.max(hi);
        if j != jstar && hi > lo_star {
            dominated = false;
        }
    }
    (jstar, maxhi, dominated)
}

/// ReLU relaxation on a rigorous `[l, u]` with `l < 0 < u`: `(s, t)` with
/// `relu(z) <= s*z + t` on `[l, u]`, both rounded up.
#[inline]
fn relu_upper(l: f64, u: f64) -> (f64, f64) {
    let width = u - l;
    if !width.is_finite() || width.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        // Overflowed or degenerate: an infinite relaxation poisons the domain
        // downstream instead of asserting relu(z) <= ~0 on a huge interval.
        return (f64::INFINITY, f64::INFINITY);
    }
    let s = next_up(u / next_down(width));
    let t = next_up(s * (-l));
    (s, t)
}

/// Lower slope for an unstable neuron: 1 when `u >= -l`, else 0 (CROWN-adaptive).
#[inline]
fn relu_lower_is_identity(l: f64, u: f64) -> bool {
    u >= -l
}

fn gemm_rows(
    dst: &mut [f64],
    w: &[f64],
    rows: usize,
    inner: usize,
    rhs: &[f64],
    cols: usize,
    accum: Accum,
) {
    let a = MatRef::from_row_major_slice(w, rows, inner);
    let b = MatRef::from_row_major_slice(rhs, inner, cols);
    let c = MatMut::from_row_major_slice_mut(dst, rows, cols);
    matmul(c, accum, a, b, 1.0, Par::Seq);
}

/// How far past its threshold a leaf centre must be to count as a violation
/// CANDIDATE. A centre that violates only marginally (measured: -9.3e-8 on
/// acas 1_5 prop_2, whose accepted counterexample is 2e-4..9e-4 deep) is a
/// boundary point a different summation order (ONNX Runtime) can flip; such a
/// leaf keeps splitting instead of stopping the lane on a point the trusted
/// gate will reject.
pub(crate) fn candidate_margin(thr: f64) -> f64 {
    1e-5 * (1.0 + thr.abs())
}

/// M8a router progress: the fraction of the root box `w0` (its widths) the
/// box `[lo, hi]` covers, `prod_i (w_i / w0_i)` over the coordinates with
/// `w0_i > 0` (a pinned coordinate counts 1). A counter only.
pub(crate) fn box_volume(lo: &[f64], hi: &[f64], w0: &[f64]) -> f64 {
    lo.iter()
        .zip(hi)
        .zip(w0)
        .filter(|(_, &w)| w > 0.0)
        .map(|((l, h), w)| ((h - l) / w).max(0.0))
        .product()
}

/// The programs a chunk's domains belong to. `progs[0]` supplies the STRUCTURE
/// and the shared weights (every program of a family has the same ops, sources,
/// sizes and `Arc`-identical `Linear` weights — see [`same_structure`]); domain
/// `b`'s own program (`which[b]`, or `progs[0]` for a family of one) supplies
/// that domain's constants: `Linear` biases, `Sparse` terms and constants,
/// `*Const` constants and its constant-rounding term count.
#[derive(Clone, Copy)]
struct Fam<'a> {
    progs: &'a [&'a Program],
    which: Option<&'a [usize]>,
}

impl<'a> Fam<'a> {
    fn single(prog: &'a [&'a Program]) -> Self {
        Self {
            progs: prog,
            which: None,
        }
    }
    fn base(&self) -> &'a Program {
        self.progs[0]
    }
    fn of(&self, b: usize) -> &'a Program {
        match self.which {
            None => self.progs[0],
            Some(w) => self.progs[w[b]],
        }
    }
    fn linear_bias(&self, idx: usize, b: usize) -> (&'a [f64], &'a [f64]) {
        match &self.of(b).ops[idx] {
            Op::Linear {
                bias_lo, bias_hi, ..
            } => (bias_lo, bias_hi),
            _ => unreachable!("family structure: Linear"),
        }
    }
    fn const_k(&self, idx: usize, b: usize) -> &'a [f64] {
        match &self.of(b).ops[idx] {
            Op::AddConst { k, .. } | Op::RevSubConst { k, .. } | Op::MulConst { k, .. } => k,
            _ => unreachable!("family structure: *Const"),
        }
    }
    #[allow(clippy::type_complexity)]
    fn sparse(&self, idx: usize, b: usize) -> (&'a [Vec<SparseTerm>], &'a [f64], &'a [f64], usize) {
        match &self.of(b).ops[idx] {
            Op::Sparse {
                terms,
                k_lo,
                k_hi,
                fanin,
            } => (terms, k_lo, k_hi, *fanin),
            _ => unreachable!("family structure: Sparse"),
        }
    }
}

/// Two programs of the same structure may share a chunk: same ops in order,
/// same sources and sizes, `Arc`-identical `Linear` weights, same `Sparse`
/// term sources. Constants (biases, coefficients, `k`) may differ.
pub(crate) fn same_structure(a: &Program, b: &Program) -> bool {
    if a.d != b.d
        || a.seed != b.seed
        || a.out != b.out
        || a.sizes != b.sizes
        || a.ops.len() != b.ops.len()
    {
        return false;
    }
    a.ops.iter().zip(&b.ops).all(|(x, y)| match (x, y) {
        (Op::Input, Op::Input) => true,
        (
            Op::Linear {
                src: s1,
                n_in: n1,
                w: w1,
                ..
            },
            Op::Linear {
                src: s2,
                n_in: n2,
                w: w2,
                ..
            },
        ) => s1 == s2 && n1 == n2 && Arc::ptr_eq(w1, w2),
        (Op::Relu { src: s1 }, Op::Relu { src: s2 })
        | (Op::Sigmoid { src: s1 }, Op::Sigmoid { src: s2 }) => s1 == s2,
        (Op::Mul { a: a1, b: b1 }, Op::Mul { a: a2, b: b2 }) => a1 == a2 && b1 == b2,
        (Op::Sparse { terms: t1, .. }, Op::Sparse { terms: t2, .. }) => {
            t1.len() == t2.len()
                && t1.iter().zip(t2).all(|(u, v)| {
                    u.len() == v.len()
                        && u.iter()
                            .zip(v)
                            .all(|(p, q)| p.src == q.src && p.idx == q.idx)
                })
        }
        _ => false,
    })
}

/// Returns every node's value plus the per-domain poison flag (OR of every
/// node's `bad`).
fn forward(fam: Fam<'_>, ch: &Chunk<'_>) -> (Vec<Option<NodeVal>>, Vec<bool>) {
    let prog = fam.base();
    let (m, d) = (ch.m, ch.d);
    let d1 = d + 1;
    let mut vals: Vec<Option<NodeVal>> = Vec::with_capacity(prog.ops.len());
    let mut poison = vec![false; m];
    for (idx, op) in prog.ops.iter().enumerate() {
        let n = prog.sizes[idx];
        let mut v = NodeVal {
            lo_form: vec![0.0; n * m * d1],
            hi_form: vec![0.0; n * m * d1],
            lo: Vec::new(),
            hi: Vec::new(),
            s_lo: Vec::new(),
            s_hi: Vec::new(),
            mag: Vec::new(),
            zmax: Vec::new(),
            bad: Vec::new(),
            pick: Vec::new(),
            cert_lo: Vec::new(),
            cert_hi: Vec::new(),
            joint_lo: Vec::new(),
            joint_hi: Vec::new(),
            joint_ok: Vec::new(),
            sig: Vec::new(),
        };
        let get = |i: usize| vals[i].as_ref().expect("topological order");
        match op {
            Op::Input => {
                for j in 0..n {
                    for b in 0..m {
                        let r = (j * m + b) * d1;
                        v.lo_form[r + j] = 1.0;
                        v.hi_form[r + j] = 1.0;
                    }
                }
                concretize(&mut v, n, ch);
            }
            Op::Linear {
                src,
                n_in,
                w_pos,
                w_neg,
                row_abs,
                ..
            } => {
                let s = get(*src);
                let cols = m * d1;
                gemm_rows(
                    &mut v.lo_form,
                    w_pos,
                    n,
                    *n_in,
                    &s.lo_form,
                    cols,
                    Accum::Replace,
                );
                gemm_rows(
                    &mut v.lo_form,
                    w_neg,
                    n,
                    *n_in,
                    &s.hi_form,
                    cols,
                    Accum::Add,
                );
                gemm_rows(
                    &mut v.hi_form,
                    w_pos,
                    n,
                    *n_in,
                    &s.hi_form,
                    cols,
                    Accum::Replace,
                );
                gemm_rows(
                    &mut v.hi_form,
                    w_neg,
                    n,
                    *n_in,
                    &s.lo_form,
                    cols,
                    Accum::Add,
                );
                let biases: Vec<(&[f64], &[f64])> =
                    (0..m).map(|b| fam.linear_bias(idx, b)).collect();
                for i in 0..n {
                    for (b, &(bias_lo, bias_hi)) in biases.iter().enumerate() {
                        let babs = bias_lo[i].abs().max(bias_hi[i].abs());
                        let c = (i * m + b) * d1 + d;
                        let slack = gslack(2 * n_in + 4, row_abs[i] * s.mag[b] + babs);
                        v.lo_form[c] = next_down(v.lo_form[c] + bias_lo[i] - slack);
                        v.hi_form[c] = next_up(v.hi_form[c] + bias_hi[i] + slack);
                    }
                }
                concretize(&mut v, n, ch);
            }
            Op::Relu { src } => {
                let s = get(*src);
                for j in 0..n {
                    for b in 0..m {
                        let i = j * m + b;
                        let (l, u) = (s.lo[i], s.hi[i]);
                        let r = i * d1;
                        if u <= 0.0 {
                            continue; // both forms stay zero
                        }
                        if l >= 0.0 {
                            v.lo_form[r..r + d1].copy_from_slice(&s.lo_form[r..r + d1]);
                            v.hi_form[r..r + d1].copy_from_slice(&s.hi_form[r..r + d1]);
                            continue;
                        }
                        let (sl, tl) = relu_upper(l, u);
                        for k in 0..d {
                            v.hi_form[r + k] = sl * s.hi_form[r + k];
                        }
                        let slack = gslack(4, sl * s.s_hi[i] + tl);
                        v.hi_form[r + d] = next_up(sl * s.hi_form[r + d] + tl + slack);
                        if relu_lower_is_identity(l, u) {
                            v.lo_form[r..r + d1].copy_from_slice(&s.lo_form[r..r + d1]);
                        }
                    }
                }
                concretize(&mut v, n, ch);
                // The ReLU output's own rigorous range is [max(l,0), max(u,0)];
                // intersect (it is never looser than the concretized forms).
                for j in 0..n {
                    for b in 0..m {
                        let i = j * m + b;
                        v.lo[i] = v.lo[i].max(s.lo[i].max(0.0)).max(0.0);
                        v.hi[i] = v.hi[i].min(s.hi[i].max(0.0));
                    }
                }
                v.zmax = vec![0.0; m];
                for j in 0..n {
                    for b in 0..m {
                        let i = j * m + b;
                        v.zmax[b] = v.zmax[b].max(v.lo[i].abs()).max(v.hi[i].abs());
                    }
                }
            }
            Op::Slice { src, start } => {
                let s = get(*src);
                let a = start * m * d1;
                v.lo_form.copy_from_slice(&s.lo_form[a..a + n * m * d1]);
                v.hi_form.copy_from_slice(&s.hi_form[a..a + n * m * d1]);
                concretize(&mut v, n, ch);
            }
            Op::Concat { srcs } => {
                let mut off = 0;
                for &si in srcs {
                    let s = get(si);
                    let len = prog.sizes[si] * m * d1;
                    v.lo_form[off..off + len].copy_from_slice(&s.lo_form);
                    v.hi_form[off..off + len].copy_from_slice(&s.hi_form);
                    off += len;
                }
                concretize(&mut v, n, ch);
            }
            Op::Identity { src } => {
                let s = get(*src);
                v.lo_form.copy_from_slice(&s.lo_form);
                v.hi_form.copy_from_slice(&s.hi_form);
                concretize(&mut v, n, ch);
            }
            Op::Add { a, b } | Op::Sub { a, b } => {
                let (sa, sb) = (get(*a), get(*b));
                let sub = matches!(op, Op::Sub { .. });
                for j in 0..n {
                    for bb in 0..m {
                        let i = j * m + bb;
                        let r = i * d1;
                        for k in 0..d1 {
                            if sub {
                                v.lo_form[r + k] = sa.lo_form[r + k] - sb.hi_form[r + k];
                                v.hi_form[r + k] = sa.hi_form[r + k] - sb.lo_form[r + k];
                            } else {
                                v.lo_form[r + k] = sa.lo_form[r + k] + sb.lo_form[r + k];
                                v.hi_form[r + k] = sa.hi_form[r + k] + sb.hi_form[r + k];
                            }
                        }
                        let slack =
                            gslack(3, sa.s_lo[i].max(sa.s_hi[i]) + sb.s_lo[i].max(sb.s_hi[i]));
                        v.lo_form[r + d] = next_down(v.lo_form[r + d] - slack);
                        v.hi_form[r + d] = next_up(v.hi_form[r + d] + slack);
                    }
                }
                concretize(&mut v, n, ch);
            }
            Op::AddConst { src, .. } | Op::RevSubConst { src, .. } => {
                let s = get(*src);
                let rev = matches!(op, Op::RevSubConst { .. });
                let ks: Vec<&[f64]> = (0..m).map(|b| fam.const_k(idx, b)).collect();
                for j in 0..n {
                    for (b, k) in ks.iter().enumerate() {
                        let i = j * m + b;
                        let r = i * d1;
                        if rev {
                            for kk in 0..d1 {
                                v.lo_form[r + kk] = -s.hi_form[r + kk];
                                v.hi_form[r + kk] = -s.lo_form[r + kk];
                            }
                        } else {
                            v.lo_form[r..r + d1].copy_from_slice(&s.lo_form[r..r + d1]);
                            v.hi_form[r..r + d1].copy_from_slice(&s.hi_form[r..r + d1]);
                        }
                        let slack = gslack(3, s.s_lo[i].max(s.s_hi[i]) + k[j].abs());
                        v.lo_form[r + d] = next_down(v.lo_form[r + d] + k[j] - slack);
                        v.hi_form[r + d] = next_up(v.hi_form[r + d] + k[j] + slack);
                    }
                }
                concretize(&mut v, n, ch);
            }
            Op::MaxGlobal { src } => {
                // lower form: the branch with the largest concrete lower bound
                // (max >= z_j for EVERY j). upper: max_j hi_j (a constant), or the
                // dominating branch's own upper form when one branch dominates.
                let s = get(*src);
                let k = prog.sizes[*src];
                for b in 0..m {
                    let (jstar, maxhi, dominated) = max_global_choice(s, k, m, b);
                    let rs = (jstar * m + b) * d1;
                    let r = b * d1;
                    v.lo_form[r..r + d1].copy_from_slice(&s.lo_form[rs..rs + d1]);
                    if dominated {
                        v.hi_form[r..r + d1].copy_from_slice(&s.hi_form[rs..rs + d1]);
                    } else {
                        v.hi_form[r..r + d].fill(0.0);
                        v.hi_form[r + d] = maxhi;
                    }
                }
                concretize(&mut v, n, ch);
                for b in 0..m {
                    let (lo_max, hi_max) =
                        (0..k).fold((f64::NEG_INFINITY, f64::NEG_INFINITY), |acc, j| {
                            let i = j * m + b;
                            (acc.0.max(s.lo[i]), acc.1.max(s.hi[i]))
                        });
                    v.lo[b] = v.lo[b].max(lo_max);
                    v.hi[b] = v.hi[b].min(hi_max);
                    v.zmax[b] = v.lo[b].abs().max(v.hi[b].abs());
                    if s.bad[b] {
                        v.bad[b] = true;
                    }
                }
            }
            Op::Sparse { .. } => {
                let per: Vec<_> = (0..m).map(|b| fam.sparse(idx, b)).collect();
                for j in 0..n {
                    for (b, &(terms, k_lo, k_hi, _)) in per.iter().enumerate() {
                        let tj = &terms[j];
                        let kabs = k_lo[j].abs().max(k_hi[j].abs());
                        let r = (j * m + b) * d1;
                        let (mut mag, mut err) = (kabs, 0.0f64);
                        for t in tj {
                            let sv = get(t.src as usize);
                            let si = t.idx as usize * m + b;
                            let rs = si * d1;
                            let (from_lo, from_hi) = if t.coef >= 0.0 {
                                (&sv.lo_form, &sv.hi_form)
                            } else {
                                (&sv.hi_form, &sv.lo_form)
                            };
                            for kk in 0..d1 {
                                v.lo_form[r + kk] += t.coef * from_lo[rs + kk];
                                v.hi_form[r + kk] += t.coef * from_hi[rs + kk];
                            }
                            mag += t.coef.abs() * sv.s_lo[si].max(sv.s_hi[si]);
                            if t.err > 0.0 {
                                err = next_up(
                                    err + next_up(t.err * sv.lo[si].abs().max(sv.hi[si].abs())),
                                );
                            }
                        }
                        let slack = next_up(gslack(tj.len() + 3, mag) + err);
                        v.lo_form[r + d] = next_down(v.lo_form[r + d] + k_lo[j] - slack);
                        v.hi_form[r + d] = next_up(v.hi_form[r + d] + k_hi[j] + slack);
                    }
                }
                concretize(&mut v, n, ch);
            }
            Op::MulConst { src, .. } => {
                let s = get(*src);
                let ks: Vec<&[f64]> = (0..m).map(|b| fam.const_k(idx, b)).collect();
                for j in 0..n {
                    for (b, k) in ks.iter().enumerate() {
                        let kj = k[j];
                        let i = j * m + b;
                        let r = i * d1;
                        let (from_lo, from_hi) = if kj >= 0.0 {
                            (&s.lo_form, &s.hi_form)
                        } else {
                            (&s.hi_form, &s.lo_form)
                        };
                        for kk in 0..d1 {
                            v.lo_form[r + kk] = kj * from_lo[r + kk];
                            v.hi_form[r + kk] = kj * from_hi[r + kk];
                        }
                        let slack = gslack(3, kj.abs() * s.s_lo[i].max(s.s_hi[i]));
                        v.lo_form[r + d] = next_down(v.lo_form[r + d] - slack);
                        v.hi_form[r + d] = next_up(v.hi_form[r + d] + slack);
                    }
                }
                concretize(&mut v, n, ch);
            }
            Op::Mul { a, b } => {
                let (sa, sb) = (get(*a), get(*b));
                v.pick = vec![0u8; n * m];
                let mut cand = [vec![0.0f64; d1], vec![0.0f64; d1]];
                let mut mags = [0.0f64; 2];
                for j in 0..n {
                    for bb in 0..m {
                        let i = j * m + bb;
                        let r = i * d1;
                        let (la, ua, lb, ub) = (sa.lo[i], sa.hi[i], sb.lo[i], sb.hi[i]);
                        let (bl, bh) = (&ch.lo[bb * d..(bb + 1) * d], &ch.hi[bb * d..(bb + 1) * d]);
                        let mut pick = 0u8;
                        for (upper, bit) in [(false, 0u8), (true, 1u8)] {
                            let mut score = [0.0f64; 2];
                            for k in 0..2u8 {
                                let plane = if upper {
                                    mc_upper(la, ua, lb, ub, k)
                                } else {
                                    mc_lower(la, ua, lb, ub, k)
                                };
                                let kk = usize::from(k);
                                mags[kk] = mul_form(&mut cand[kk], sa, sb, i, d, plane, upper);
                                score[kk] = form_extreme(&cand[kk], bl, bh, d, upper);
                            }
                            // Lower: keep the plane whose minimum is HIGHER;
                            // upper: the one whose maximum is LOWER. A NaN
                            // score compares false and keeps plane 0 (the
                            // domain is poisoned by `concretize` anyway).
                            let second = if upper {
                                score[1] < score[0]
                            } else {
                                score[1] > score[0]
                            };
                            let k = usize::from(second);
                            let slack = gslack(4, mags[k]);
                            if upper {
                                v.hi_form[r..r + d1].copy_from_slice(&cand[k]);
                                v.hi_form[r + d] = next_up(v.hi_form[r + d] + slack);
                            } else {
                                v.lo_form[r..r + d1].copy_from_slice(&cand[k]);
                                v.lo_form[r + d] = next_down(v.lo_form[r + d] - slack);
                            }
                            pick |= u8::from(second) << bit;
                        }
                        v.pick[i] = pick;
                    }
                }
                concretize(&mut v, n, ch);
                // The product's own rigorous range is the interval product of
                // the operand boxes; intersect (both are enclosures). Only
                // when every corner is finite: `f64::min`/`max` drop a NaN
                // (`0 * inf`), which could otherwise NARROW the result.
                for j in 0..n {
                    for bb in 0..m {
                        let i = j * m + bb;
                        let (la, ua, lb, ub) = (sa.lo[i], sa.hi[i], sb.lo[i], sb.hi[i]);
                        let corners = [la * lb, la * ub, ua * lb, ua * ub];
                        if corners.iter().all(|c| c.is_finite()) {
                            let lo =
                                next_down(corners.iter().copied().fold(f64::INFINITY, f64::min));
                            let hi =
                                next_up(corners.iter().copied().fold(f64::NEG_INFINITY, f64::max));
                            v.lo[i] = v.lo[i].max(lo);
                            v.hi[i] = v.hi[i].min(hi);
                        }
                        if !matches!(
                            v.lo[i].partial_cmp(&v.hi[i]),
                            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                        ) || sa.bad[bb]
                            || sb.bad[bb]
                        {
                            v.bad[bb] = true;
                        }
                    }
                }
                v.zmax = vec![0.0; m];
                for j in 0..n {
                    for bb in 0..m {
                        let i = j * m + bb;
                        v.zmax[bb] = v.zmax[bb].max(v.lo[i].abs()).max(v.hi[i].abs());
                    }
                }
            }
            Op::Sigmoid { src } => {
                // #lowdim-bundle (1). A plane `a z + b` bounds sigmoid(z) on
                // the source box; composed with the source's forms (a
                // non-negative slope takes the same-side form, a negative one
                // the other side's) and charged like ReLU's upper form:
                // `gslack(4, |a| s + |b|)`, `s` the form's magnitude.
                let s = get(*src);
                v.sig = vec![[f64::NAN; 4]; n * m];
                for j in 0..n {
                    for b in 0..m {
                        let i = j * m + b;
                        let r = i * d1;
                        let Some(p) = sigmoid::planes(s.lo[i], s.hi[i]) else {
                            // Non-finite source box: poison through the forms.
                            v.lo_form[r + d] = f64::NEG_INFINITY;
                            v.hi_form[r + d] = f64::INFINITY;
                            continue;
                        };
                        v.sig[i] = p;
                        let [a_lo, b_lo, a_hi, b_hi] = p;
                        let (fl, sl) = if a_lo >= 0.0 {
                            (&s.lo_form, s.s_lo[i])
                        } else {
                            (&s.hi_form, s.s_hi[i])
                        };
                        for k in 0..d {
                            v.lo_form[r + k] = a_lo * fl[r + k];
                        }
                        let slack = gslack(4, a_lo.abs() * sl + b_lo.abs());
                        v.lo_form[r + d] = next_down(a_lo * fl[r + d] + b_lo - slack);
                        let (fh, sh) = if a_hi >= 0.0 {
                            (&s.hi_form, s.s_hi[i])
                        } else {
                            (&s.lo_form, s.s_lo[i])
                        };
                        for k in 0..d {
                            v.hi_form[r + k] = a_hi * fh[r + k];
                        }
                        let slack = gslack(4, a_hi.abs() * sh + b_hi.abs());
                        v.hi_form[r + d] = next_up(a_hi * fh[r + d] + b_hi + slack);
                    }
                }
                concretize(&mut v, n, ch);
                // The monotone interval image [sigmoid_down(l), sigmoid_up(u)]
                // is also an enclosure; intersect (a NaN side is dropped by
                // max/min, keeping the valid bound; an empty result poisons).
                for j in 0..n {
                    for b in 0..m {
                        let i = j * m + b;
                        let (l, u) = (s.lo[i], s.hi[i]);
                        if l.is_finite() && u.is_finite() {
                            v.lo[i] = v.lo[i].max(sigmoid::sig_down(l));
                            v.hi[i] = v.hi[i].min(sigmoid::sig_up(u));
                        }
                        if !matches!(
                            v.lo[i].partial_cmp(&v.hi[i]),
                            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                        ) || s.bad[b]
                        {
                            v.bad[b] = true;
                        }
                    }
                }
                v.zmax = vec![0.0; m];
                for j in 0..n {
                    for b in 0..m {
                        let i = j * m + b;
                        v.zmax[b] = v.zmax[b].max(v.lo[i].abs()).max(v.hi[i].abs());
                    }
                }
            }
        }
        // #lsnc-lane: a bilinear program intersects every affine op with the
        // interval image of its sources (see `Program::bilinear`), and a joint
        // site takes the joint quadratic bound — both before refinement.
        if prog.bilinear {
            interval_meet(fam, &vals, ch, idx, &mut v);
        }
        if prog.joint {
            if let Some(s) = prog.site_of.get(idx).copied().flatten() {
                joint::apply_site(fam, &vals, ch, &prog.sites[s], &mut v);
            }
        }
        let relaxed_input = prog.feeds_relu[idx]
            || (prog.bilinear && prog.mul_operand.get(idx).copied().unwrap_or(false));
        let refine = prog.refine_max > 0 && relaxed_input && n <= prog.refine_max;
        let certs = prog.joint && prog.site_operand.get(idx).copied().unwrap_or(false);
        if refine || certs {
            refine_node(
                fam,
                &vals,
                ch,
                idx,
                &mut v,
                &poison,
                RefineMode {
                    all: prog.bilinear,
                    certs,
                },
            );
        }
        for (p, &b) in poison.iter_mut().zip(&v.bad) {
            *p |= b;
        }
        vals.push(Some(v));
    }
    (vals, poison)
}

/// Intersect element `j` of `v` at domain `b` with `[lo, hi]`; an empty
/// result poisons the domain. A NaN side is ignored (`f64::max` / `min` drop
/// it), which keeps the existing — valid — bound.
#[inline]
fn meet_at(v: &mut NodeVal, j: usize, b: usize, m: usize, lo: f64, hi: f64) {
    let k = j * m + b;
    v.lo[k] = v.lo[k].max(lo);
    v.hi[k] = v.hi[k].min(hi);
    if !matches!(
        v.lo[k].partial_cmp(&v.hi[k]),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    ) {
        v.bad[b] = true;
    }
}

/// #lsnc-lane (bilinear programs only, see `Program::bilinear`): intersect an
/// AFFINE op's concrete bounds with the interval image of its sources'
/// concrete bounds, rounded outward. Both are rigorous enclosures of the same
/// value, so the intersection is; an empty one can only be numerical and
/// poisons the domain. Forms are untouched.
///
/// Why: forward forms lose the RANGE a ReLU output was clipped to — a
/// clamp `c - relu(c - relu(z - l) - l)` (lsnc's controller saturation) has a
/// width-8 range the forms see as width 19-29 — and the loss compounds
/// through every later layer. The numpy gate's engine has this meet on every
/// affine op; without it the row-2 root dV bound more than doubles (13.1 ->
/// 28.0, measured on the prototype 2026-09-25).
///
/// ROUNDING: a Linear / Sparse row is a sum of `2 n + 2` rounded products and
/// sums whose magnitudes are at most `(|W| or |coef| + err) * max|src| +
/// |bias|`, charged by `gslack`; a two-operand add, subtract or scale is one
/// rounding, closed by `next_down` / `next_up`; copies are exact.
fn interval_meet(
    fam: Fam<'_>,
    vals: &[Option<NodeVal>],
    ch: &Chunk<'_>,
    idx: usize,
    v: &mut NodeVal,
) {
    let prog = fam.base();
    let (m, n) = (ch.m, prog.sizes[idx]);
    let get = |i: usize| vals[i].as_ref().expect("topological order");
    match &prog.ops[idx] {
        Op::Linear {
            src,
            n_in,
            w_pos,
            w_neg,
            row_abs,
            ..
        } => {
            let s = get(*src);
            for b in 0..m {
                let (bias_lo, bias_hi) = fam.linear_bias(idx, b);
                for i in 0..n {
                    let (mut lo, mut hi) = (bias_lo[i], bias_hi[i]);
                    let row = i * n_in;
                    for j in 0..*n_in {
                        let k = j * m + b;
                        lo += w_pos[row + j] * s.lo[k] + w_neg[row + j] * s.hi[k];
                        hi += w_pos[row + j] * s.hi[k] + w_neg[row + j] * s.lo[k];
                    }
                    let babs = bias_lo[i].abs().max(bias_hi[i].abs());
                    let e = gslack(2 * n_in + 4, row_abs[i] * s.zmax[b] + babs);
                    meet_at(v, i, b, m, next_down(lo - e), next_up(hi + e));
                }
            }
        }
        Op::Sparse { .. } => {
            for b in 0..m {
                let (terms, k_lo, k_hi, _) = fam.sparse(idx, b);
                for j in 0..n {
                    let (mut lo, mut hi) = (k_lo[j], k_hi[j]);
                    let mut mag = k_lo[j].abs().max(k_hi[j].abs());
                    for t in &terms[j] {
                        let sv = get(t.src as usize);
                        let k = t.idx as usize * m + b;
                        let (sl, sh) = (sv.lo[k], sv.hi[k]);
                        let xm = sl.abs().max(sh.abs());
                        if t.coef >= 0.0 {
                            lo += t.coef * sl;
                            hi += t.coef * sh;
                        } else {
                            lo += t.coef * sh;
                            hi += t.coef * sl;
                        }
                        if t.err > 0.0 {
                            lo -= t.err * xm;
                            hi += t.err * xm;
                        }
                        mag += (t.coef.abs() + t.err) * xm;
                    }
                    let e = gslack(3 * terms[j].len() + 4, mag);
                    meet_at(v, j, b, m, next_down(lo - e), next_up(hi + e));
                }
            }
        }
        Op::Add { a, b: bsrc } | Op::Sub { a, b: bsrc } => {
            let (sa, sb) = (get(*a), get(*bsrc));
            let sub = matches!(prog.ops[idx], Op::Sub { .. });
            for j in 0..n {
                for b in 0..m {
                    let k = j * m + b;
                    let (lo, hi) = if sub {
                        (sa.lo[k] - sb.hi[k], sa.hi[k] - sb.lo[k])
                    } else {
                        (sa.lo[k] + sb.lo[k], sa.hi[k] + sb.hi[k])
                    };
                    meet_at(v, j, b, m, next_down(lo), next_up(hi));
                }
            }
        }
        Op::AddConst { src, .. } | Op::RevSubConst { src, .. } => {
            let s = get(*src);
            let rev = matches!(prog.ops[idx], Op::RevSubConst { .. });
            for b in 0..m {
                let kc = fam.const_k(idx, b);
                for j in 0..n {
                    let k = j * m + b;
                    let (lo, hi) = if rev {
                        (kc[j] - s.hi[k], kc[j] - s.lo[k])
                    } else {
                        (s.lo[k] + kc[j], s.hi[k] + kc[j])
                    };
                    meet_at(v, j, b, m, next_down(lo), next_up(hi));
                }
            }
        }
        Op::MulConst { src, .. } => {
            let s = get(*src);
            for b in 0..m {
                let kc = fam.const_k(idx, b);
                for j in 0..n {
                    let k = j * m + b;
                    let c = kc[j];
                    let (lo, hi) = if c >= 0.0 {
                        (c * s.lo[k], c * s.hi[k])
                    } else {
                        (c * s.hi[k], c * s.lo[k])
                    };
                    meet_at(v, j, b, m, next_down(lo), next_up(hi));
                }
            }
        }
        Op::Slice { src, start } => {
            let s = get(*src);
            for j in 0..n {
                for b in 0..m {
                    let ks = (start + j) * m + b;
                    meet_at(v, j, b, m, s.lo[ks], s.hi[ks]);
                }
            }
        }
        Op::Identity { src } => {
            let s = get(*src);
            for j in 0..n {
                for b in 0..m {
                    let k = j * m + b;
                    meet_at(v, j, b, m, s.lo[k], s.hi[k]);
                }
            }
        }
        Op::Concat { srcs } => {
            let mut off = 0;
            for &si in srcs {
                let s = get(si);
                let w = prog.sizes[si];
                for j in 0..w {
                    for b in 0..m {
                        let k = j * m + b;
                        meet_at(v, off + j, b, m, s.lo[k], s.hi[k]);
                    }
                }
                off += w;
            }
        }
        // Already intersected with their own interval images (or exact).
        Op::Input
        | Op::Relu { .. }
        | Op::Mul { .. }
        | Op::MaxGlobal { .. }
        | Op::Sigmoid { .. } => return,
    }
    for b in 0..m {
        let mut z = 0.0f64;
        for j in 0..n {
            let k = j * m + b;
            z = z.max(v.lo[k].abs()).max(v.hi[k].abs());
        }
        v.zmax[b] = z;
    }
}

/// Element `i`'s lower and upper forms carry bit-identical input
/// coefficients on every domain of the chunk: it is an affine function of the
/// input up to its constant slack, so a backward pass cannot tighten it.
fn affine_exact(v: &NodeVal, i: usize, m: usize, d: usize) -> bool {
    let d1 = d + 1;
    (0..m).all(|b| {
        let r = (i * m + b) * d1;
        v.lo_form[r..r + d] == v.hi_form[r..r + d]
    })
}

/// Which neurons [`refine_node`] refines, and whether it keeps certificates.
#[derive(Debug, Clone, Copy)]
struct RefineMode {
    /// Every neuron whose forms are not exactly affine (bilinear programs);
    /// otherwise only neurons unstable in some domain (the historical rule).
    all: bool,
    /// Keep each pass's plane as the element's CROWN certificate.
    certs: bool,
}

// ---------------------------------------------------------------- backward

/// Backward-CROWN refinement of a pre-activation node's concrete bounds. For
/// each neuron unstable in some domain of the chunk, a backward pass seeded at
/// this node with `+e_i` (a lower bound) and `-e_i` (an upper bound, by exact
/// negation), through the relaxations the forward pass already built for the
/// earlier nodes — themselves refined, since refinement runs in topological
/// order. The result is INTERSECTED with the forward bounds: both are rigorous
/// enclosures, so the intersection is. An empty intersection can only be
/// numerical and poisons the domain. Forms are untouched; only the concrete
/// bounds (and so the next ReLU's relaxation) tighten.
///
/// #lsnc-lane: `mode.all` (bilinear programs) refines every neuron whose forms
/// are not exactly affine instead of only the unstable ones — a stable
/// neuron's tighter box still tightens the interval meet downstream and the
/// McCormick envelope of a product it feeds. `mode.certs` (joint-site operands)
/// additionally KEEPS each pass's plane as the element's CROWN certificate
/// (`NodeVal::cert_lo` / `cert_hi`): `grad . x + pconst <= value(x)` from the
/// `+e_i` pass and `value(x) <= -(grad' . x + pconst')` from the `-e_i` pass,
/// both from this chunk's forward pass and valid on the domain's box.
fn refine_node(
    fam: Fam<'_>,
    vals: &[Option<NodeVal>],
    ch: &Chunk<'_>,
    idx: usize,
    v: &mut NodeVal,
    poison: &[bool],
    mode: RefineMode,
) {
    let prog = fam.base();
    let (m, n, d) = (ch.m, prog.sizes[idx], ch.d);
    let d1 = d + 1;
    if mode.certs {
        v.cert_lo.clone_from(&v.lo_form);
        v.cert_hi.clone_from(&v.hi_form);
    }
    let mut e = vec![0.0f64; n];
    for i in 0..n {
        let wanted = if mode.all || mode.certs {
            !affine_exact(v, i, m, d)
        } else {
            (0..m).any(|b| {
                let k = i * m + b;
                v.lo[k] < 0.0 && v.hi[k] > 0.0
            })
        };
        if !wanted {
            continue;
        }
        e[i] = 1.0;
        let lower = backward(fam, vals, ch, idx, &vec![&e[..]; m], poison, false);
        e[i] = -1.0;
        let neg_upper = backward(fam, vals, ch, idx, &vec![&e[..]; m], poison, false);
        e[i] = 0.0;
        for b in 0..m {
            let k = i * m + b;
            if lower.lb[b].is_finite() {
                v.lo[k] = v.lo[k].max(lower.lb[b]);
            }
            if neg_upper.lb[b].is_finite() {
                v.hi[k] = v.hi[k].min(-neg_upper.lb[b]);
            }
            if !matches!(
                v.lo[k].partial_cmp(&v.hi[k]),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ) {
                v.bad[b] = true;
            }
            if mode.certs {
                let r = k * d1;
                if lower.plane_ok(b, d) {
                    v.cert_lo[r..r + d].copy_from_slice(&lower.grad[b * d..(b + 1) * d]);
                    v.cert_lo[r + d] = lower.pconst[b];
                }
                if neg_upper.plane_ok(b, d) {
                    for kk in 0..d {
                        v.cert_hi[r + kk] = -neg_upper.grad[b * d + kk];
                    }
                    v.cert_hi[r + d] = -neg_upper.pconst[b];
                }
            }
        }
    }
    for b in 0..m {
        let mut z = 0.0f64;
        for i in 0..n {
            let k = i * m + b;
            z = z.max(v.lo[k].abs()).max(v.hi[k].abs());
        }
        v.zmax[b] = z;
    }
}

/// One backward pass over a chunk.
struct Pass {
    /// Rigorous lower bound of `objective · y` per domain (`-inf` when
    /// poisoned or non-finite).
    lb: Vec<f64>,
    /// The input-space multiplier `[m][d]` (the split gradient).
    grad: Vec<f64>,
    /// #lsnc-lane stage 2: the plane constant of THIS pass, `[m]`:
    /// `grad[b] · x + pconst[b] <= objective · y(x)` for every `x` in domain
    /// `b`'s box, in exact arithmetic over the stored floats (`-inf` exactly
    /// when `lb` is). It is the pass's own constant minus its own rounding
    /// charges — never deconstructed from a (max'd) bound.
    pconst: Vec<f64>,
    /// The multiplier reached a joint site's reduction node.
    hit_site: bool,
    /// #lowdim-bundle (2), recorded only by an ALPHA pass
    /// ([`backward_alpha`] with a table): per node, `Some([n][m])` for a
    /// ReLU node, holding the positive multiplier `l_j` of each unstable
    /// neuron whose lower slope came from the table (0 elsewhere). The alpha
    /// step's gradient; never read back into a bound. Empty otherwise.
    relu_mult: Vec<Option<Vec<f64>>>,
}

impl Pass {
    /// Domain `b`'s plane is usable (finite, not poisoned).
    fn plane_ok(&self, b: usize, d: usize) -> bool {
        self.lb[b].is_finite()
            && self.pconst[b].is_finite()
            && self.grad[b * d..(b + 1) * d].iter().all(|v| v.is_finite())
    }
}

/// Rigorous lower bound of `objective · y` per domain of the chunk, plus the
/// input-space multiplier (the split gradient), `[m][d]`, and the pass's own
/// plane constant (see [`Pass`]).
///
/// #lsnc-lane stage 2: with `intercept`, a multiplier `l` reaching a joint
/// site's reduction node is NOT propagated into the product: it is replaced
/// by `l` times the site's joint LOWER plane (`l >= 0`) or UPPER plane (`l <
/// 0`) from the same forward pass — or the node's own forms on a domain the
/// joint pass did not bound — which are valid linear bounds of the site on
/// the box. The substitution is charged like every other multiplier product.
#[allow(clippy::too_many_arguments)]
fn backward(
    fam: Fam<'_>,
    vals: &[Option<NodeVal>],
    ch: &Chunk<'_>,
    seed: usize,
    objective: &[&[f64]],
    poison: &[bool],
    intercept: bool,
) -> Pass {
    backward_alpha(fam, vals, ch, seed, objective, poison, intercept, None)
}

/// [`backward`], and with `alpha` (#lowdim-bundle (2)) the lower slope of
/// each UNSTABLE ReLU neuron with a positive multiplier is taken from the
/// table instead of the CROWN-adaptive `{0, 1}`: any `alpha in [0, 1]` is a
/// valid lower relaxation (`relu(z) >= alpha z` for every real `z`), and its
/// multiplier product is charged like the upper slope's. A table entry that
/// is missing, NaN or outside `[0, 1]` falls back to the historical rule.
/// With `alpha == None` this is the historical pass exactly.
#[allow(clippy::too_many_arguments)]
fn backward_alpha(
    fam: Fam<'_>,
    vals: &[Option<NodeVal>],
    ch: &Chunk<'_>,
    seed: usize,
    objective: &[&[f64]],
    poison: &[bool],
    intercept: bool,
    alpha: Option<&alpha::AlphaTable>,
) -> Pass {
    let prog = fam.base();
    let (m, d) = (ch.m, ch.d);
    let nodes = prog.ops.len();
    let mut hit_site = false;
    let mut relu_mult: Vec<Option<Vec<f64>>> = if alpha.is_some() {
        vec![None; nodes]
    } else {
        Vec::new()
    };
    // lam[node] is [m][n] (domain-major).
    let mut lam: Vec<Option<Vec<f64>>> = vec![None; nodes];
    let mut constv = vec![0.0f64; m];
    let mut const_abs = vec![0.0f64; m];
    let mut slack = vec![0.0f64; m];
    let val = |i: usize| vals[i].as_ref().expect("forward ran");

    let n_out = prog.sizes[seed];
    let mut root = Vec::with_capacity(m * n_out);
    for o in objective.iter().take(m) {
        root.extend_from_slice(o);
    }
    lam[seed] = Some(root);

    // Merge `mu` ([m][n_src], scaled by `sign`) into lam[src] over rows
    // [offset, offset + width), charging the merge rounding.
    let merge = |lam: &mut Vec<Option<Vec<f64>>>,
                 slack: &mut [f64],
                 src: usize,
                 offset: usize,
                 mu: &[f64],
                 width: usize,
                 sign: f64| {
        let n_src = prog.sizes[src];
        let z = &val(src).zmax;
        match &mut lam[src] {
            slot @ None => {
                let mut fresh = vec![0.0; m * n_src];
                for b in 0..m {
                    for j in 0..width {
                        fresh[b * n_src + offset + j] = sign * mu[b * width + j];
                    }
                }
                *slot = Some(fresh);
            }
            Some(acc) => {
                for b in 0..m {
                    let mut abs_sum = 0.0;
                    for j in 0..width {
                        let e = &mut acc[b * n_src + offset + j];
                        *e += sign * mu[b * width + j];
                        abs_sum += e.abs();
                    }
                    slack[b] = next_up(slack[b] + gslack(2, abs_sum * z[b]));
                }
            }
        }
    };

    for idx in (1..nodes).rev() {
        let Some(l) = lam[idx].take() else { continue };
        let n = prog.sizes[idx];
        if prog.site_of.get(idx).copied().flatten().is_some() {
            hit_site = true;
            let joint = vals
                .get(idx)
                .and_then(Option::as_ref)
                .filter(|sv| sv.joint_ok.len() == m && n == 1);
            if let (true, Some(sv)) = (intercept, joint) {
                // The site is width 1: domain b's multiplier is l[b], its
                // plane row starts at b * (d + 1) in both the joint planes
                // and the node's own forms.
                let d1 = d + 1;
                let zin = &val(0).zmax;
                let mut mu = vec![0.0f64; m * d];
                for b in 0..m {
                    let lj = l[b];
                    if lj == 0.0 {
                        continue;
                    }
                    let r = b * d1;
                    let plane: &[f64] = match (sv.joint_ok[b], lj >= 0.0) {
                        (true, true) => &sv.joint_lo[r..r + d1],
                        (true, false) => &sv.joint_hi[r..r + d1],
                        (false, true) => &sv.lo_form[r..r + d1],
                        (false, false) => &sv.hi_form[r..r + d1],
                    };
                    let mut abs_q = 0.0f64;
                    for k in 0..d {
                        let q = lj * plane[k];
                        mu[b * d + k] = q;
                        abs_q += q.abs();
                    }
                    let p = lj * plane[d];
                    constv[b] += p;
                    const_abs[b] += p.abs();
                    slack[b] = next_up(slack[b] + gslack(3, abs_q * zin[b]));
                }
                merge(&mut lam, &mut slack, 0, 0, &mu, d, 1.0);
                continue;
            }
        }
        match &prog.ops[idx] {
            Op::Input => unreachable!("input is index 0"),
            Op::Linear {
                src,
                n_in,
                w,
                row_abs,
                ..
            } => {
                let mut mu = vec![0.0; m * n_in];
                gemm_rows(&mut mu, &l, m, n, w, *n_in, Accum::Replace);
                let z = &val(*src).zmax;
                for b in 0..m {
                    let (bias_lo, bias_hi) = fam.linear_bias(idx, b);
                    let row = &l[b * n..(b + 1) * n];
                    let (mut dot, mut dabs, mut rl) = (0.0, 0.0, 0.0);
                    for i in 0..n {
                        // Lower bound: the bias end that minimises row * bias.
                        let bi = if row[i] >= 0.0 {
                            bias_lo[i]
                        } else {
                            bias_hi[i]
                        };
                        dot += row[i] * bi;
                        dabs += (row[i] * bi).abs();
                        rl += row[i].abs() * row_abs[i];
                    }
                    constv[b] += dot;
                    const_abs[b] += dabs;
                    slack[b] = next_up(slack[b] + gslack(n + 4, rl * z[b]));
                }
                merge(&mut lam, &mut slack, *src, 0, &mu, *n_in, 1.0);
            }
            Op::Relu { src } => {
                let s = val(*src);
                // #lowdim-bundle (2): this node's alpha slopes, `[n][m]`.
                let slopes: Option<&[f64]> = alpha
                    .and_then(|t| t.slopes.get(idx))
                    .and_then(Option::as_deref)
                    .filter(|v| v.len() == n * m);
                let mut rec: Option<Vec<f64>> = slopes.map(|_| vec![0.0; n * m]);
                let mut mu = vec![0.0; m * n];
                for b in 0..m {
                    let mut abs_mu = 0.0;
                    for j in 0..n {
                        let lj = l[b * n + j];
                        let i = j * m + b;
                        let (lo, hi) = (s.lo[i], s.hi[i]);
                        let out = if hi <= 0.0 || lj == 0.0 {
                            0.0
                        } else if lo >= 0.0 {
                            lj
                        } else if lj > 0.0 {
                            match slopes.map(|sl| sl[i]).filter(|a| (0.0..=1.0).contains(a)) {
                                Some(a) => {
                                    if let Some(r) = rec.as_mut() {
                                        r[i] = lj;
                                    }
                                    let q = lj * a;
                                    abs_mu += q.abs();
                                    q
                                }
                                None => {
                                    if relu_lower_is_identity(lo, hi) {
                                        lj
                                    } else {
                                        0.0
                                    }
                                }
                            }
                        } else {
                            let (sl, tl) = relu_upper(lo, hi);
                            let p = lj * tl;
                            constv[b] += p;
                            const_abs[b] += p.abs();
                            let q = lj * sl;
                            abs_mu += q.abs();
                            q
                        };
                        mu[b * n + j] = out;
                    }
                    slack[b] = next_up(slack[b] + gslack(3, abs_mu * s.zmax[b]));
                }
                if let (Some(r), Some(slot)) = (rec, relu_mult.get_mut(idx)) {
                    *slot = Some(r);
                }
                merge(&mut lam, &mut slack, *src, 0, &mu, n, 1.0);
            }
            Op::Sigmoid { src } => {
                // #lowdim-bundle (1): lambda * sigmoid(z) >= lambda * (a z + b)
                // with the LOWER plane when lambda >= 0 and the UPPER plane
                // when lambda < 0, both valid on the source box; the planes
                // are the forward pass's (rebuilt from the source box when
                // this node is the seed of a refinement pass and has no
                // stored value yet). The multiplier product is charged like
                // ReLU's; a domain without planes (non-finite box) gets a -inf
                // constant, so its bound is -inf.
                let s = val(*src);
                let stored: Option<&[[f64; 4]]> = vals
                    .get(idx)
                    .and_then(Option::as_ref)
                    .map(|v| v.sig.as_slice())
                    .filter(|p| p.len() == n * m);
                let mut mu = vec![0.0; m * n];
                for b in 0..m {
                    let mut abs_mu = 0.0;
                    for j in 0..n {
                        let lj = l[b * n + j];
                        if lj == 0.0 {
                            continue;
                        }
                        let i = j * m + b;
                        let plane = match stored {
                            Some(p) => Some(p[i]).filter(|q| q.iter().all(|v| v.is_finite())),
                            None => sigmoid::planes(s.lo[i], s.hi[i]),
                        };
                        let Some([a_lo, b_lo, a_hi, b_hi]) = plane else {
                            constv[b] = f64::NEG_INFINITY;
                            continue;
                        };
                        let (a, c) = if lj > 0.0 { (a_lo, b_lo) } else { (a_hi, b_hi) };
                        let p = lj * c;
                        constv[b] += p;
                        const_abs[b] += p.abs();
                        let q = lj * a;
                        abs_mu += q.abs();
                        mu[b * n + j] = q;
                    }
                    slack[b] = next_up(slack[b] + gslack(3, abs_mu * s.zmax[b]));
                }
                merge(&mut lam, &mut slack, *src, 0, &mu, n, 1.0);
            }
            Op::Slice { src, start } => merge(&mut lam, &mut slack, *src, *start, &l, n, 1.0),
            Op::Concat { srcs } => {
                let mut off = 0;
                for &si in srcs {
                    let w = prog.sizes[si];
                    let mut part = vec![0.0; m * w];
                    for b in 0..m {
                        part[b * w..(b + 1) * w].copy_from_slice(&l[b * n + off..b * n + off + w]);
                    }
                    merge(&mut lam, &mut slack, si, 0, &part, w, 1.0);
                    off += w;
                }
            }
            Op::Identity { src } => merge(&mut lam, &mut slack, *src, 0, &l, n, 1.0),
            Op::Add { a, b } => {
                merge(&mut lam, &mut slack, *a, 0, &l, n, 1.0);
                merge(&mut lam, &mut slack, *b, 0, &l, n, 1.0);
            }
            Op::Sub { a, b } => {
                merge(&mut lam, &mut slack, *a, 0, &l, n, 1.0);
                merge(&mut lam, &mut slack, *b, 0, &l, n, -1.0);
            }
            Op::AddConst { src, .. } | Op::RevSubConst { src, .. } => {
                for b in 0..m {
                    let k = fam.const_k(idx, b);
                    for j in 0..n {
                        let p = l[b * n + j] * k[j];
                        constv[b] += p;
                        const_abs[b] += p.abs();
                    }
                }
                let sign = if matches!(prog.ops[idx], Op::RevSubConst { .. }) {
                    -1.0
                } else {
                    1.0
                };
                merge(&mut lam, &mut slack, *src, 0, &l, n, sign);
            }
            Op::MaxGlobal { src } => {
                // lambda * max >= lambda * z_{j*} when lambda >= 0 (max >= z_j*);
                // when lambda < 0, lambda * max >= lambda * max_j hi_j, or exactly
                // lambda * z_{j*} when that branch dominates.
                let s = val(*src);
                let k = prog.sizes[*src];
                let mut mu = vec![0.0; m * k];
                for b in 0..m {
                    let lj = l[b];
                    let (jstar, maxhi, dominated) = max_global_choice(s, k, m, b);
                    if lj >= 0.0 || dominated {
                        mu[b * k + jstar] = lj;
                    } else {
                        let p = lj * maxhi;
                        constv[b] += p;
                        const_abs[b] += p.abs();
                    }
                }
                merge(&mut lam, &mut slack, *src, 0, &mu, k, 1.0);
            }
            Op::Sparse { .. } => {
                let per: Vec<_> = (0..m).map(|b| fam.sparse(idx, b)).collect();
                // Dense multiplier per source node touched, [m][n_src].
                let mut mus: std::collections::BTreeMap<usize, Vec<f64>> =
                    std::collections::BTreeMap::new();
                let mut abs_mu: std::collections::BTreeMap<usize, Vec<f64>> =
                    std::collections::BTreeMap::new();
                for j in 0..n {
                    for (b, &(terms, k_lo, k_hi, _)) in per.iter().enumerate() {
                        let lj = l[b * n + j];
                        if lj == 0.0 {
                            continue;
                        }
                        let kj = if lj >= 0.0 { k_lo[j] } else { k_hi[j] };
                        let p = lj * kj;
                        constv[b] += p;
                        const_abs[b] += p.abs();
                        for t in &terms[j] {
                            let src = t.src as usize;
                            let ns = prog.sizes[src];
                            let q = lj * t.coef;
                            mus.entry(src).or_insert_with(|| vec![0.0; m * ns])
                                [b * ns + t.idx as usize] += q;
                            abs_mu.entry(src).or_insert_with(|| vec![0.0; m])[b] += q.abs();
                            if t.err > 0.0 {
                                let sv = val(src);
                                let si = t.idx as usize * m + b;
                                let xm = sv.lo[si].abs().max(sv.hi[si].abs());
                                slack[b] =
                                    next_up(slack[b] + next_up(lj.abs() * next_up(t.err * xm)));
                            }
                        }
                    }
                }
                for (src, mu) in mus {
                    let ns = prog.sizes[src];
                    let z = &val(src).zmax;
                    let am = &abs_mu[&src];
                    for b in 0..m {
                        let fanin = per[b].3;
                        slack[b] = next_up(slack[b] + gslack(fanin + 3, am[b] * z[b]));
                    }
                    merge(&mut lam, &mut slack, src, 0, &mu, ns, 1.0);
                }
            }
            Op::MulConst { src, .. } => {
                let z = &val(*src).zmax;
                let mut mu = vec![0.0; m * n];
                for b in 0..m {
                    let k = fam.const_k(idx, b);
                    let mut abs_mu = 0.0;
                    for j in 0..n {
                        let q = l[b * n + j] * k[j];
                        abs_mu += q.abs();
                        mu[b * n + j] = q;
                    }
                    slack[b] = next_up(slack[b] + gslack(3, abs_mu * z[b]));
                }
                merge(&mut lam, &mut slack, *src, 0, &mu, n, 1.0);
            }
            Op::Mul { a, b } => {
                // lambda * z >= lambda * (alpha a + beta b + gamma) with the
                // chosen LOWER plane when lambda >= 0, the chosen UPPER plane
                // when lambda < 0. The planes are rebuilt from the operands'
                // concrete bounds exactly as the forward pass built them. When
                // this node is the SEED of a refinement pass its own value is
                // not stored yet: plane 0 is used (any plane is valid).
                let (sa, sb) = (val(*a), val(*b));
                let picks: Option<&[u8]> = vals
                    .get(idx)
                    .and_then(Option::as_ref)
                    .map(|v| v.pick.as_slice())
                    .filter(|p| p.len() == n * m);
                let mut mu_a = vec![0.0; m * n];
                let mut mu_b = vec![0.0; m * n];
                for bb in 0..m {
                    let (mut abs_a, mut abs_b) = (0.0f64, 0.0f64);
                    for j in 0..n {
                        let lj = l[bb * n + j];
                        if lj == 0.0 {
                            continue;
                        }
                        let i = j * m + bb;
                        let (la, ua, lb, ub) = (sa.lo[i], sa.hi[i], sb.lo[i], sb.hi[i]);
                        let pick = picks.map_or(0, |p| p[i]);
                        let (alpha, beta, gamma) = if lj > 0.0 {
                            mc_lower(la, ua, lb, ub, pick & 1)
                        } else {
                            mc_upper(la, ua, lb, ub, (pick >> 1) & 1)
                        };
                        let qa = lj * alpha;
                        let qb = lj * beta;
                        mu_a[bb * n + j] = qa;
                        mu_b[bb * n + j] = qb;
                        abs_a += qa.abs();
                        abs_b += qb.abs();
                        let p = lj * gamma;
                        constv[bb] += p;
                        const_abs[bb] += p.abs();
                    }
                    slack[bb] = next_up(
                        slack[bb] + gslack(3, abs_a * sa.zmax[bb]) + gslack(3, abs_b * sb.zmax[bb]),
                    );
                }
                merge(&mut lam, &mut slack, *a, 0, &mu_a, n, 1.0);
                merge(&mut lam, &mut slack, *b, 0, &mu_b, n, 1.0);
            }
        }
    }

    let grad = lam[0].take().unwrap_or_else(|| vec![0.0; m * d]);
    let mut lb = vec![0.0; m];
    let mut pconst = vec![f64::NEG_INFINITY; m];
    for b in 0..m {
        let n_terms = fam.of(b).const_terms;
        let mu = &grad[b * d..(b + 1) * d];
        let (bl, bh, bx) = (
            &ch.lo[b * d..(b + 1) * d],
            &ch.hi[b * d..(b + 1) * d],
            &ch.xmax[b * d..(b + 1) * d],
        );
        let mut v = constv[b];
        let mut s = constv[b].abs();
        for k in 0..d {
            v += if mu[k] >= 0.0 {
                mu[k] * bl[k]
            } else {
                mu[k] * bh[k]
            };
            s += mu[k].abs() * bx[k];
        }
        let err = next_up(slack[b] + gslack(d + 4, s) + gslack(n_terms, const_abs[b]));
        let bound = next_down(v - err);
        lb[b] = if bound.is_finite() && !poison[b] {
            bound
        } else {
            f64::NEG_INFINITY
        };
        // The plane of this pass: every multiplier rounding is in `slack`
        // (charged against magnitudes over the box) and the constant's own
        // accumulation in `gslack(n_terms, const_abs)`; the concretization
        // charge `gslack(d + 4, s)` belongs to evaluating the plane, which a
        // consumer charges itself.
        if lb[b].is_finite() {
            pconst[b] = next_down(constv[b] - next_up(slack[b] + gslack(n_terms, const_abs[b])));
        }
    }
    Pass {
        lb,
        grad,
        pconst,
        hit_site,
        relu_mult,
    }
}

/// #lsnc-lane: the rigorous lower bound of `objective · seed` on each domain
/// and the split gradient, from every bound the program admits:
///
/// * the plain backward pass (every program — the only one for a program
///   without a product, which therefore bounds exactly as before);
/// * with the joint bound armed and a site on the path, the INTERCEPT pass
///   (the site's joint planes substituted), whose gradient is taken when its
///   bound is higher;
/// * on a bilinear program, the seed's forward forms and its concrete box
///   (the numpy gate's `fwd` / `itv` terms; the box carries the interval
///   meet's and the joint vertex bound's tightness forward).
///
/// Each is a rigorous lower bound of the same quantity, so their maximum is.
fn objective_bounds(
    fam: Fam<'_>,
    vals: &[Option<NodeVal>],
    ch: &Chunk<'_>,
    seed: usize,
    objective: &[&[f64]],
    poison: &[bool],
) -> (Vec<f64>, Vec<f64>) {
    let (lb, grad, _) = objective_bounds_passes(fam, vals, ch, seed, objective, poison, false);
    (lb, grad)
}

/// [`objective_bounds`], and with `keep` also every backward PASS it ran (the
/// plain pass, then the intercept pass when it ran), each carrying its OWN
/// plane `grad . x + pconst <= objective . y(x)` on each domain's box
/// ([`Pass`]) — #lsnc-lane stage 3's clip planes. The planes are the passes'
/// own; the returned lower bound (a maximum over passes and the seed's forms)
/// is never turned back into a plane.
fn objective_bounds_passes(
    fam: Fam<'_>,
    vals: &[Option<NodeVal>],
    ch: &Chunk<'_>,
    seed: usize,
    objective: &[&[f64]],
    poison: &[bool],
    keep: bool,
) -> (Vec<f64>, Vec<f64>, Vec<Pass>) {
    let prog = fam.base();
    let (m, d) = (ch.m, ch.d);
    let plain = backward(fam, vals, ch, seed, objective, poison, false);
    let hit = plain.hit_site;
    let mut passes = Vec::new();
    let (mut lb, mut grad) = if keep {
        let lg = (plain.lb.clone(), plain.grad.clone());
        passes.push(plain);
        lg
    } else {
        (plain.lb, plain.grad)
    };
    if prog.joint && hit {
        let cut = backward(fam, vals, ch, seed, objective, poison, true);
        for b in 0..m {
            if cut.lb[b] > lb[b] {
                lb[b] = cut.lb[b];
                grad[b * d..(b + 1) * d].copy_from_slice(&cut.grad[b * d..(b + 1) * d]);
            }
        }
        if keep {
            passes.push(cut);
        }
    }
    if prog.bilinear {
        if let Some(sv) = vals.get(seed).and_then(Option::as_ref) {
            for b in 0..m {
                if poison[b] {
                    continue;
                }
                let (fwd, itv) = seed_bounds(sv, objective[b], b, ch);
                lb[b] = lb[b].max(fwd).max(itv);
            }
        }
    }
    (lb, grad, passes)
}

/// Lower bounds of `objective · seed` on domain `b` from the seed node's own
/// forms (`fwd`) and concrete box (`itv`), rounded down. A non-negative
/// weight takes the lower form / bound, a negative one the upper.
///
/// ROUNDING: the combined form's coefficients carry at most `gamma_n *
/// sum_j |o_j| |form_jk|` error each, which over the box is at most
/// `gamma_n * sum_j |o_j| s_j` (`s_j` the form's magnitude); its
/// concretization adds `gamma_{d+2}` of the same; the box sum is `n + 1`
/// roundings of at most `sum_j |o_j end_j|`.
fn seed_bounds(sv: &NodeVal, objective: &[f64], b: usize, ch: &Chunk<'_>) -> (f64, f64) {
    let (m, d) = (ch.m, ch.d);
    let d1 = d + 1;
    let n = objective.len();
    let (bl, bh) = (&ch.lo[b * d..(b + 1) * d], &ch.hi[b * d..(b + 1) * d]);
    let (mut vi, mut si) = (0.0f64, 0.0f64);
    let mut coef = vec![0.0f64; d];
    let (mut c, mut sf) = (0.0f64, 0.0f64);
    for (j, &o) in objective.iter().enumerate() {
        if o == 0.0 {
            continue;
        }
        let k = j * m + b;
        let r = k * d1;
        let (end, form, s) = if o >= 0.0 {
            (sv.lo[k], &sv.lo_form[r..r + d1], sv.s_lo[k])
        } else {
            (sv.hi[k], &sv.hi_form[r..r + d1], sv.s_hi[k])
        };
        vi += o * end;
        si += (o * end).abs();
        for kk in 0..d {
            coef[kk] += o * form[kk];
        }
        c += o * form[d];
        sf += o.abs() * s;
    }
    let itv = next_down(vi - gslack(n + 2, si));
    let mut vf = c;
    for kk in 0..d {
        vf += if coef[kk] >= 0.0 {
            coef[kk] * bl[kk]
        } else {
            coef[kk] * bh[kk]
        };
    }
    let fwd = next_down(vf - gslack(n + d + 4, sf));
    (fwd, itv)
}

/// Plain f64 point evaluation of `objective · y(x)` (heuristic only: used to
/// stop early on a likely violation, never to publish a verdict).
fn point_objective(prog: &Program, x: &[f64], objective: &[f64]) -> f64 {
    point_values(prog, x)
        .iter()
        .zip(objective)
        .map(|(y, o)| y * o)
        .sum()
}

/// Plain f64 point evaluation of the whole seed vector `y(x)` (heuristic and
/// admission cross-check only, never a verdict).
fn point_values(prog: &Program, x: &[f64]) -> Vec<f64> {
    let mut vals: Vec<Vec<f64>> = Vec::with_capacity(prog.ops.len());
    for (idx, op) in prog.ops.iter().enumerate() {
        let n = prog.sizes[idx];
        let v = match op {
            Op::Input => x.to_vec(),
            Op::Linear {
                src,
                n_in,
                w,
                bias_lo,
                bias_hi,
                ..
            } => {
                let s = &vals[*src];
                (0..n)
                    .map(|i| {
                        f64::midpoint(bias_lo[i], bias_hi[i])
                            + (0..*n_in).map(|j| w[i * n_in + j] * s[j]).sum::<f64>()
                    })
                    .collect()
            }
            Op::Sparse {
                terms, k_lo, k_hi, ..
            } => (0..n)
                .map(|j| {
                    f64::midpoint(k_lo[j], k_hi[j])
                        + terms[j]
                            .iter()
                            .map(|t| t.coef * vals[t.src as usize][t.idx as usize])
                            .sum::<f64>()
                })
                .collect(),
            Op::Relu { src } => vals[*src].iter().map(|v| v.max(0.0)).collect(),
            Op::Slice { src, start } => vals[*src][*start..*start + n].to_vec(),
            Op::Concat { srcs } => srcs.iter().flat_map(|&s| vals[s].clone()).collect(),
            Op::Identity { src } => vals[*src].clone(),
            Op::Add { a, b } => vals[*a].iter().zip(&vals[*b]).map(|(p, q)| p + q).collect(),
            Op::Sub { a, b } => vals[*a].iter().zip(&vals[*b]).map(|(p, q)| p - q).collect(),
            Op::AddConst { src, k } => vals[*src].iter().zip(k).map(|(p, q)| p + q).collect(),
            Op::RevSubConst { src, k } => vals[*src].iter().zip(k).map(|(p, q)| q - p).collect(),
            Op::MulConst { src, k } => vals[*src].iter().zip(k).map(|(p, q)| p * q).collect(),
            Op::MaxGlobal { src } => {
                vec![vals[*src].iter().copied().fold(f64::NEG_INFINITY, f64::max)]
            }
            Op::Mul { a, b } => vals[*a].iter().zip(&vals[*b]).map(|(p, q)| p * q).collect(),
            Op::Sigmoid { src } => vals[*src].iter().map(|&v| sigmoid::sig(v)).collect(),
        };
        vals.push(v);
    }
    vals.swap_remove(prog.seed)
}

// ---------------------------------------------------------------- driver

/// Bound a batch of `count` domains (`lo`/`hi` are `[count][d]`), in parallel
/// chunks. Returns per-domain `(lower bound, gradient [d])`.
#[cfg(test)]
fn bound_batch(
    prog: &Program,
    lo: &[f64],
    hi: &[f64],
    count: usize,
    objective: &[f64],
    chunk: usize,
) -> Vec<(f64, Vec<f64>)> {
    bound_batch_multi(
        prog,
        lo,
        hi,
        count,
        std::slice::from_ref(&objective.to_vec()),
        chunk,
    )
    .into_iter()
    .map(|(lbs, grads)| (lbs[0], grads))
    .collect()
}

/// Bound every objective over a batch of domains. The forward pass (every
/// node's forms and concrete bounds) is shared; each objective gets its own
/// backward pass. Returns per domain `(lower bound per objective, gradient per
/// objective)`, gradients concatenated `[k][d]`.
fn bound_batch_multi(
    prog: &Program,
    lo: &[f64],
    hi: &[f64],
    count: usize,
    objectives: &[Vec<f64>],
    chunk: usize,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let d = prog.d;
    // chunk 0 would bound NOTHING while the caller has already dequeued the
    // domains; clamp here, and the driver also refuses a short result.
    let chunk = chunk.max(1);
    let starts: Vec<usize> = (0..count).step_by(chunk).collect();
    starts
        .par_iter()
        .flat_map_iter(|&s| {
            let m = chunk.min(count - s);
            let ch = Chunk::new(m, d, &lo[s * d..(s + m) * d], &hi[s * d..(s + m) * d]);
            search::bound_chunk(prog, &ch, objectives, false, None)
                .into_iter()
                .map(|db| (db.lbs, db.grads))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Probe every node's real output shape with one IBP pass on the root box,
/// then lower the graph (see [`compile`] for why shapes, not axes, decide).
///
/// #lsnc-lane: a graph carrying a BOUND [`LaneTwin`] is lowered through the
/// twin instead (see [`lower_graph_counted`]).
#[cfg(test)]
fn lower_graph(graph: &GraphNetwork, input: &BoundedTensor) -> Result<Program, String> {
    lower_graph_counted(graph, input, LowerOpts::default()).map(|lowered| lowered.prog)
}

/// [`lower_graph`] with the lowering options a lane config asks for
/// (#lowdim-bundle (1)); `LowerOpts::default()` is [`lower_graph`] exactly.
fn lower_graph_opts(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    opts: LowerOpts,
) -> Result<Program, String> {
    lower_graph_counted(graph, input, opts).map(|lowered| lowered.prog)
}

/// A lowering and what produced it.
struct Lowered {
    prog: Program,
    stats: CompileStats,
    /// Lowered from the graph's [`LaneTwin`], not the graph itself.
    twin: bool,
}

/// [`lower_graph`] with its counts. When `graph` carries a lane twin that is
/// still bound to it (`GraphNetwork::lowdim_twin`: same identity token as at
/// attach, so no structural mutation since), the TWIN is lowered — the same
/// model with its constant-only cones as rigorous interval constants — and the
/// lowering is cross-checked against the twin's own graph forward at several
/// points of the box before it is admitted.
fn lower_graph_counted(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    opts: LowerOpts,
) -> Result<Lowered, String> {
    let (source, twin) = match graph.lowdim_twin() {
        Some(t) => (&t.graph, true),
        None => (graph, false),
    };
    if !is_vector(input.shape()) {
        return Err(format!("input shape {:?} is not a vector", input.shape()));
    }
    let detailed = source
        .propagate_ibp_detailed(input, 0.0)
        .map_err(|e| format!("shape probe (IBP) failed: {e}"))?;
    let mut shapes: std::collections::HashMap<String, Vec<usize>> = detailed
        .nodes
        .into_iter()
        .map(|n| (n.name, n.output_shape))
        .collect();
    shapes.insert(NETWORK_INPUT.to_string(), input.shape().to_vec());
    let (prog, stats) = compile(source, input.len(), &shapes, opts)?;
    if twin {
        cross_check_lowering(source, &prog, input)?;
    }
    Ok(Lowered { prog, stats, twin })
}

/// Admission guard for a twin lowering: at the box centre and eight fixed
/// interior points (f32-representable, inside the box), the lowered program's
/// plain f64 value must lie inside `source`'s own IBP enclosure of that point
/// (relative `1e-4`). Not a proof — a guard against a lowering that compiled a
/// DIFFERENT function (a wrong gather index, reduce axis or broadcast).
fn cross_check_lowering(
    source: &GraphNetwork,
    prog: &Program,
    input: &BoundedTensor,
) -> Result<(), String> {
    const FRACS: [f64; 5] = [0.5, 0.125, 0.875, 0.3125, 0.6875];
    let lo: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let hi: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    for p in 0..9usize {
        let x32: Vec<f32> = (0..lo.len())
            .map(|k| {
                let f = if p == 0 {
                    0.5
                } else {
                    FRACS[(p + 2 * k) % FRACS.len()]
                };
                // Rounding an interior point of an f32-bounded box to f32 stays
                // inside the box; clamp anyway so a degenerate box cannot escape.
                let v = (lo[k] + (hi[k] - lo[k]) * f) as f32;
                v.max(lo[k] as f32).min(hi[k] as f32)
            })
            .collect();
        let x64: Vec<f64> = x32.iter().map(|&v| f64::from(v)).collect();
        let point = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(input.shape()), x32)
            .map_err(|e| format!("twin cross-check point: {e}"))?;
        let boxed = BoundedTensor::new(point.clone(), point)
            .map_err(|e| format!("twin cross-check point: {e}"))?;
        let y = source
            .propagate_ibp(&boxed)
            .map_err(|e| format!("twin cross-check forward failed: {e}"))?;
        let mine = point_values(prog, &x64);
        if mine.len() != y.len() {
            return Err(format!(
                "twin cross-check: lowered {} outputs, graph {}",
                mine.len(),
                y.len()
            ));
        }
        for (j, (m, (gl, gh))) in mine
            .iter()
            .zip(y.lower().iter().zip(y.upper().iter()))
            .enumerate()
        {
            let (gl, gh) = (f64::from(*gl), f64::from(*gh));
            let tol = 1e-4 * (1.0 + gl.abs().max(gh.abs()));
            if !(m.is_finite() && gl.is_finite() && gh.is_finite())
                || *m < gl - tol
                || *m > gh + tol
            {
                return Err(format!(
                    "twin cross-check failed at output {j}, point {p}: lowered {m:e} outside the \
                     graph's point enclosure [{gl:e}, {gh:e}]"
                ));
            }
        }
    }
    Ok(())
}

/// #lsnc-lane — the lane-private INTERVAL TWIN of a model.
///
/// # Why
///
/// `lsnc_relu`'s Lyapunov certificate carries four constant-only `MatMul`
/// cones (the matrix `P = eps I + W^T W`, `0 @ P`, and the two
/// equilibrium-point network evaluations). A rounded binary32 product is not
/// the authored real product, so the exact loader declines them, the real
/// graph carries `OpaqueSkip` nodes, and every bound on it is `[-inf, +inf]`.
/// The loader's interval carrier (`NY_CONST_MATMUL_INTERVAL`) folds exactly
/// those cones into rigorous enclosures, but it is `Deliver::EnvOnly` and dark
/// by policy: globally it would put the enclosure in front of EVERY bound path,
/// on a family that has published false proofs twice (807bf511, and the
/// `graph_alpha` sibling-merge defect the carrier exposed).
///
/// # The decision this type carries (owner, 2026-09-24)
///
/// The enclosure is LANE-PRIVATE. `ny_onnx::lowdim_twin` loads a second copy
/// of the model with the carrier's audited interval fold enabled for that ONE
/// load only (a thread-scoped guard; the global lever stays dark), and moves
/// the graph straight into this type. The graph is a PRIVATE field: only this
/// module reads it, and this module's lowering discharges every radius itself
/// (see [`compile`]). The type does not deref to, borrow as, or convert into a
/// [`GraphNetwork`], so no other bound path — alpha-CROWN, graph BaB, MIP, GPU
/// — can run on the carrier-on graph.
#[derive(Debug)]
pub struct LaneTwin {
    graph: GraphNetwork,
    interval_weights: usize,
}

impl LaneTwin {
    /// Wrap a graph loaded with the lane-private interval fold.
    /// `interval_weights` is the loader's count of interval-valued constants
    /// (observable only).
    ///
    /// Refuses a graph that still carries an `OpaqueSkip` (the enclosure did
    /// not reach every constant cone) or no `IntervalConst` node at all
    /// (nothing needed enclosing: the twin would just be the real graph).
    pub fn new(graph: GraphNetwork, interval_weights: usize) -> Result<Self, String> {
        let (mut opaque, mut interval) = (0usize, 0usize);
        for name in graph.node_names() {
            match graph.node(name).map(crate::network::GraphNode::layer) {
                Some(Layer::OpaqueSkip(_)) => opaque += 1,
                Some(Layer::IntervalConst(_)) => interval += 1,
                _ => {}
            }
        }
        if opaque > 0 {
            return Err(format!(
                "the lane twin still carries {opaque} OpaqueSkip node(s)"
            ));
        }
        if interval == 0 {
            return Err("the lane twin carries no IntervalConst node".into());
        }
        Ok(Self {
            graph,
            interval_weights,
        })
    }

    /// Interval-valued constants the lane-private load created.
    pub fn interval_weights(&self) -> usize {
        self.interval_weights
    }

    /// Nodes of the twin graph.
    pub fn node_count(&self) -> usize {
        self.graph.num_nodes()
    }
}

/// A graph lowered ONCE for the lane (#lsnc-lane), reusable across the clauses
/// of one property: the program, the root box, and what the lowering admitted.
#[derive(Debug, Clone)]
pub struct LaneAdmission {
    prog: Program,
    lo0: Vec<f64>,
    hi0: Vec<f64>,
    stats: CompileStats,
    twin: bool,
    /// #lowdim-bundle (1): the varying-coordinate fold the program and the
    /// root box were re-expressed through (`admit_with`, armed); `None` for
    /// every historical admission. `prog`, `lo0` and `hi0` are then over the
    /// FREE coordinates, and a candidate is expanded to the full input.
    pinned: Option<pinned::PinnedFold>,
}

impl LaneAdmission {
    /// Ops lowered (graph nodes, the input excluded).
    pub fn lowered_ops(&self) -> usize {
        self.prog.ops.len().saturating_sub(1)
    }

    /// `IntervalConst` nodes lowered with their radius.
    pub fn interval_consts(&self) -> usize {
        self.stats.interval_consts
    }

    /// Two-operand products lowered to McCormick envelopes.
    pub fn bilinear_ops(&self) -> usize {
        self.stats.muls
    }

    /// #lowdim-bundle (1): `Sigmoid` nodes lowered to rigorous planes (0
    /// unless `LowDimConfig::varying_dims` armed the lowering).
    pub fn sigmoid_ops(&self) -> usize {
        self.stats.sigmoids
    }

    /// Lowered from the graph's lane twin.
    pub fn from_twin(&self) -> bool {
        self.twin
    }

    /// Input dimension of the lowered program.
    pub fn input_dim(&self) -> usize {
        self.prog.d
    }

    /// Width of the output the objectives are over.
    pub fn output_width(&self) -> usize {
        self.prog.sizes[self.prog.seed]
    }

    /// #lsnc-lane stage 2: `ReduceSum(Mul)` sites the joint bound can serve.
    pub fn joint_sites(&self) -> usize {
        self.prog.sites.len()
    }

    /// The joint sites' graph node names, in program order.
    pub fn joint_site_names(&self) -> Vec<String> {
        self.prog.sites.iter().map(|s| s.name.clone()).collect()
    }
}

/// Lower `graph` over the root box `input` for the lane (through its bound
/// lane twin when it has one). `Err` is a decline reason.
pub fn admit(graph: &GraphNetwork, input: &BoundedTensor) -> Result<LaneAdmission, String> {
    let lo0: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let hi0: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    let d = lo0.len();
    if d == 0 || d > MAX_INPUT_DIM {
        return Err(format!("input dimension {d} outside 1..={MAX_INPUT_DIM}"));
    }
    if lo0.iter().chain(&hi0).any(|v| !v.is_finite()) || lo0.iter().zip(&hi0).any(|(l, h)| l > h) {
        return Err("input box is not a finite, ordered box".into());
    }
    let Lowered { prog, stats, twin } = lower_graph_counted(graph, input, LowerOpts::default())?;
    if prog.d != d {
        return Err(format!("box of {d} dims for a {}-input program", prog.d));
    }
    Ok(LaneAdmission {
        prog,
        lo0,
        hi0,
        stats,
        twin,
        pinned: None,
    })
}

/// #m8-cgan-leaf-batch: the pinned-coordinate classification (exact, narrow
/// or free; see `pinned`), public so the CLI can pin it against the VNN-LIB
/// loader's real widening.
pub use pinned::{pin_counts, pin_kind, pin_tolerance, PinCounts, PinKind};

/// #lowdim-bundle (1): the `[lowdim-varying]` engagement line, printed
/// unconditionally by every lane entry whose config arms
/// [`LowDimConfig::varying_dims`]: `[lowdim-varying] armed=1
/// status=folded|within_cap|declined d_full=.. d_free=.. pinned=..
/// input_linears_folded=.. sigmoid_ops=.. pinned_exact=.. pinned_narrow=..`
/// (the last two appended by #m8-cgan-leaf-batch: the box's exact and
/// narrow pins). An armed run on a pinned box that never prints
/// `status=folded` is VOID, not negative.
fn emit_varying_line(
    status: pinned::FoldStatus,
    d_full: usize,
    d_free: usize,
    input_linears_folded: usize,
    sigmoid_ops: usize,
    pins: PinCounts,
) {
    eprintln!(
        "[lowdim-varying] armed=1 status={} d_full={d_full} d_free={d_free} pinned={} \
         input_linears_folded={input_linears_folded} sigmoid_ops={sigmoid_ops} \
         pinned_exact={} pinned_narrow={}",
        status.token(),
        d_full.saturating_sub(d_free),
        pins.exact,
        pins.narrow,
    );
}

/// The f64 image of an `f32` box (the lane's entry conversion).
fn box_f64(input: &BoundedTensor) -> (Vec<f64>, Vec<f64>) {
    (
        input.lower().iter().map(|&v| f64::from(v)).collect(),
        input.upper().iter().map(|&v| f64::from(v)).collect(),
    )
}

/// Review fix (#lowdim-bundle (1)): the `[lowdim-varying] status=declined`
/// line for an ARMED entry whose lowering declined before the fold was
/// reached (the Sequential entry lowers the full graph first), so an armed
/// run is never silent. `d_free` counts the box's FREE coordinates under the
/// fold's own classification (#m8-cgan-leaf-batch: a one-ulp-widened
/// declared point is a narrow pin, not a free coordinate).
fn emit_varying_lowering_decline(cfg: &LowDimConfig, input: &BoundedTensor) {
    if cfg.varying_dims {
        let (lo, hi) = box_f64(input);
        let pins = pin_counts(&lo, &hi);
        emit_varying_line(
            pinned::FoldStatus::Declined,
            input.len(),
            pins.free,
            0,
            0,
            pins,
        );
    }
}

/// `Op::Sigmoid` nodes of a program (observable only).
fn sigmoid_ops(prog: &Program) -> usize {
    prog.ops
        .iter()
        .filter(|op| matches!(op, Op::Sigmoid { .. }))
        .count()
}

/// Re-express a program lowered over the FULL input through `fold`, with the
/// admission cross-check; prints the engagement line either way. `Err` is a
/// decline reason.
fn apply_fold(fold: &pinned::PinnedFold, prog: Program) -> Result<Program, String> {
    let sig = sigmoid_ops(&prog);
    let applied = fold.apply(prog.clone()).and_then(|(folded, n)| {
        fold.cross_check(&prog, &folded)?;
        Ok((folded, n))
    });
    match applied {
        Ok((folded, n)) => {
            emit_varying_line(
                pinned::FoldStatus::Folded,
                fold.full_dim(),
                fold.free_dim(),
                n,
                sig,
                fold.counts(),
            );
            Ok(folded)
        }
        Err(reason) => {
            emit_varying_line(
                pinned::FoldStatus::Declined,
                fold.full_dim(),
                fold.free_dim(),
                0,
                sig,
                fold.counts(),
            );
            Err(reason)
        }
    }
}

/// [`admit`] under a lane config: with [`LowDimConfig::varying_dims`] armed,
/// a root box wider than [`MAX_INPUT_DIM`] whose other coordinates are
/// pinned is admitted over its VARYING coordinates (#lowdim-bundle (1),
/// `pinned`), and `Sigmoid` nodes are lowered. Unarmed, this is [`admit`]
/// exactly.
pub fn admit_with(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    cfg: &LowDimConfig,
) -> Result<LaneAdmission, String> {
    if !cfg.varying_dims {
        return admit(graph, input);
    }
    let lo0: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let hi0: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    let d = lo0.len();
    let pins = pin_counts(&lo0, &hi0);
    let fold = match pinned::PinnedFold::of(&lo0, &hi0) {
        Ok(fold) => fold,
        Err(reason) => {
            emit_varying_line(pinned::FoldStatus::Declined, d, 0, 0, 0, pins);
            return Err(reason);
        }
    };
    // Review fix: the engagement line on EVERY armed decline (the lowering's
    // own included), not only on the fold's success or failure — an armed
    // entry that declined in the lowering was silent.
    let d_free = fold.as_ref().map_or(d, pinned::PinnedFold::free_dim);
    let declined = |reason: String| -> Result<LaneAdmission, String> {
        emit_varying_line(pinned::FoldStatus::Declined, d, d_free, 0, 0, pins);
        Err(reason)
    };
    if fold.is_none() {
        if d == 0 || d > MAX_INPUT_DIM {
            return declined(format!("input dimension {d} outside 1..={MAX_INPUT_DIM}"));
        }
        if lo0.iter().chain(&hi0).any(|v| !v.is_finite())
            || lo0.iter().zip(&hi0).any(|(l, h)| l > h)
        {
            return declined("input box is not a finite, ordered box".into());
        }
    }
    let lowered = match lower_graph_counted(graph, input, LowerOpts::of(cfg)) {
        Ok(lowered) => lowered,
        Err(reason) => return declined(reason),
    };
    let Lowered { prog, stats, twin } = lowered;
    if prog.d != d {
        return declined(format!("box of {d} dims for a {}-input program", prog.d));
    }
    match fold {
        None => {
            emit_varying_line(
                pinned::FoldStatus::WithinCap,
                d,
                d,
                0,
                sigmoid_ops(&prog),
                pins,
            );
            Ok(LaneAdmission {
                prog,
                lo0,
                hi0,
                stats,
                twin,
                pinned: None,
            })
        }
        Some(fold) => {
            let folded = apply_fold(&fold, prog)?;
            Ok(LaneAdmission {
                prog: folded,
                lo0: fold.restrict(&lo0),
                hi0: fold.restrict(&hi0),
                stats,
                twin,
                pinned: Some(fold),
            })
        }
    }
}

/// [`prove_conjunction_infeasible`] on an already-admitted lowering, over the
/// admission's root box. On a [`LowDimStop::Candidate`] stop the candidate
/// centre is written to `witness` (a CANDIDATE only: the caller must
/// re-evaluate it on a trusted forward before treating it as a
/// counterexample). Never claims a violation.
pub fn prove_conjunction_admitted(
    admission: &LaneAdmission,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    deadline: Instant,
    cfg: &LowDimConfig,
    witness: &mut Option<Vec<f64>>,
) -> LowDimOutcome {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::LowDim);
    if objectives.is_empty() || objectives.len() != thresholds.len() {
        return LowDimOutcome::Declined(format!(
            "{} objectives vs {} thresholds",
            objectives.len(),
            thresholds.len()
        ));
    }
    let objs: Vec<Vec<f64>> = objectives
        .iter()
        .map(|o| o.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let thrs: Vec<f64> = thresholds.iter().map(|&t| f64::from(t)).collect();
    let Some(fold) = admission.pinned.as_ref() else {
        return prove_conjunction_core_taped(
            admission.prog.clone(),
            admission.lo0.clone(),
            admission.hi0.clone(),
            objs,
            thrs,
            deadline,
            cfg,
            witness,
            None,
        );
    };
    // #lowdim-bundle (1): the lane ran over the free coordinates; hand the
    // caller a FULL-dimension candidate.
    let mut free_witness = None;
    let outcome = prove_conjunction_core_taped(
        admission.prog.clone(),
        admission.lo0.clone(),
        admission.hi0.clone(),
        objs,
        thrs,
        deadline,
        cfg,
        &mut free_witness,
        Some(fold),
    );
    if let Some(x) = free_witness {
        *witness = Some(fold.expand(&x));
    }
    outcome
}

/// SOUNDNESS-TEST FACE (#lsnc-lane): the rigorous lower bound of every
/// objective on every box — exactly the values the branch-and-bound compares
/// against its thresholds. `lo` / `hi` are `[boxes][d]`; each box must lie in
/// the admission's root box. `cfg.refine_max_width` is honoured. Never a
/// verdict.
pub fn bound_boxes(
    admission: &LaneAdmission,
    lo: &[f64],
    hi: &[f64],
    objectives: &[Vec<f64>],
    cfg: &LowDimConfig,
) -> Result<Vec<Vec<f64>>, String> {
    let d = admission.prog.d;
    if d == 0 || lo.len() != hi.len() || !lo.len().is_multiple_of(d) {
        return Err("boxes are not [count][d]".into());
    }
    let width = admission.output_width();
    if objectives.is_empty() || objectives.iter().any(|o| o.len() != width) {
        return Err(format!("objectives must have {width} entries"));
    }
    let inside = lo.chunks(d).zip(hi.chunks(d)).all(|(l, h)| {
        (0..d).all(|k| admission.lo0[k] <= l[k] && l[k] <= h[k] && h[k] <= admission.hi0[k])
    });
    if !inside {
        return Err("every box must be an ordered sub-box of the root box".into());
    }
    let mut prog = admission.prog.clone();
    prog.refine_max = cfg.refine_max_width;
    prog.joint = cfg.joint_quadratic;
    let count = lo.len() / d;
    Ok(
        bound_batch_multi(&prog, lo, hi, count, objectives, cfg.chunk)
            .into_iter()
            .map(|(lbs, _)| lbs)
            .collect(),
    )
}

/// The lowered program's plain f64 value of its output at `x` (interval
/// constants at their midpoints). Heuristic and test face only; never a bound.
pub fn lowered_point_values(admission: &LaneAdmission, x: &[f64]) -> Result<Vec<f64>, String> {
    if x.len() != admission.prog.d || x.iter().any(|v| !v.is_finite()) {
        return Err("point must be finite with the program's input dimension".into());
    }
    Ok(point_values(&admission.prog, x))
}

/// #lsnc-lane stage 3: one box's clip evidence (test face; never a verdict).
#[derive(Debug, Clone, PartialEq)]
pub struct ClipReport {
    /// Every plane of the box's OWN backward passes, as `(objective index,
    /// a, c)` with `a . x + c <= objectives[index] . y(x)` for every `x` in
    /// the box — exactly the planes the branch and bound clips with.
    pub planes: Vec<(usize, Vec<f64>, f64)>,
    /// The relaxed clip of the box against `objectives[i] . y <= thresholds[i]`
    /// for every `i`: `Some((lo, hi))` contains every point of the box that
    /// satisfies all of them; `None` when the clip proved there is none.
    pub clipped: Option<(Vec<f64>, Vec<f64>)>,
}

/// SOUNDNESS-TEST FACE (#lsnc-lane stage 3): the clip planes and the relaxed
/// clip of every box, exactly as the branch and bound computes them
/// (`search::bound_chunk` with planes kept, then `search::clip_box`). `lo` /
/// `hi` are `[boxes][d]`, each an ordered sub-box of the admission's root
/// box; `thresholds[i]` is objective `i`'s conjunct `objective . y <=
/// threshold`. `cfg.refine_max_width`, `cfg.joint_quadratic` and `cfg.chunk`
/// are honoured; `cfg.clip` is not consulted (the clip always runs here).
pub fn clip_report(
    admission: &LaneAdmission,
    lo: &[f64],
    hi: &[f64],
    objectives: &[Vec<f64>],
    thresholds: &[f64],
    cfg: &LowDimConfig,
) -> Result<Vec<ClipReport>, String> {
    let d = admission.prog.d;
    if d == 0 || lo.len() != hi.len() || !lo.len().is_multiple_of(d) {
        return Err("boxes are not [count][d]".into());
    }
    let width = admission.output_width();
    if objectives.is_empty()
        || objectives.len() != thresholds.len()
        || objectives.iter().any(|o| o.len() != width)
    {
        return Err(format!(
            "need one threshold per objective and {width} entries per objective"
        ));
    }
    let inside = lo.chunks(d).zip(hi.chunks(d)).all(|(l, h)| {
        (0..d).all(|k| admission.lo0[k] <= l[k] && l[k] <= h[k] && h[k] <= admission.hi0[k])
    });
    if !inside {
        return Err("every box must be an ordered sub-box of the root box".into());
    }
    let mut prog = admission.prog.clone();
    prog.refine_max = cfg.refine_max_width;
    prog.joint = cfg.joint_quadratic;
    let count = lo.len() / d;
    let chunk = cfg.chunk.max(1);
    let mut out = Vec::with_capacity(count);
    for s in (0..count).step_by(chunk) {
        let m = chunk.min(count - s);
        let ch = Chunk::new(m, d, &lo[s * d..(s + m) * d], &hi[s * d..(s + m) * d]);
        for (j, db) in search::bound_chunk(&prog, &ch, objectives, true, None)
            .into_iter()
            .enumerate()
        {
            let b = s + j;
            let (mut cl, mut chi) = (
                lo[b * d..(b + 1) * d].to_vec(),
                hi[b * d..(b + 1) * d].to_vec(),
            );
            let planes: Vec<search::ClipPlane<'_>> = db
                .planes
                .iter()
                .map(|(i, a, c)| search::ClipPlane {
                    a: a.as_slice(),
                    c: *c,
                    t: thresholds[*i],
                })
                .collect();
            let kept = search::clip_box(&mut cl, &mut chi, &planes);
            let clipped = kept.map(|_| (cl, chi));
            out.push(ClipReport {
                planes: db.planes.clone(),
                clipped,
            });
        }
    }
    Ok(out)
}

/// #lsnc-lane stage 2: one joint site on one box (test and observability
/// face; never a verdict).
#[derive(Debug, Clone, PartialEq)]
pub struct JointSiteBox {
    /// Graph node name of the site's reduction.
    pub name: String,
    /// The site node's rigorous concrete bounds on the box — after the joint
    /// meet when it ran; `[-inf, inf]` on a poisoned box.
    pub lo: f64,
    pub hi: f64,
    /// The joint pass's planes on the box: `lower.0 · x + lower.1 <= site(x)
    /// <= upper.0 · x + upper.1` for every `x` in it. `None` when the joint
    /// pass did not bound the box (unarmed, poisoned or non-finite).
    pub lower: Option<(Vec<f64>, f64)>,
    pub upper: Option<(Vec<f64>, f64)>,
}

/// SOUNDNESS-TEST FACE (#lsnc-lane stage 2): every joint site's concrete
/// bounds and joint planes on every box, exactly as the branch-and-bound's
/// forward pass computes them. `lo` / `hi` are `[boxes][d]`, each an ordered
/// sub-box of the admission's root box; `cfg.refine_max_width` and
/// `cfg.joint_quadratic` are honoured. Never a verdict.
pub fn joint_site_boxes(
    admission: &LaneAdmission,
    lo: &[f64],
    hi: &[f64],
    cfg: &LowDimConfig,
) -> Result<Vec<Vec<JointSiteBox>>, String> {
    let d = admission.prog.d;
    if d == 0 || lo.len() != hi.len() || !lo.len().is_multiple_of(d) {
        return Err("boxes are not [count][d]".into());
    }
    let inside = lo.chunks(d).zip(hi.chunks(d)).all(|(l, h)| {
        (0..d).all(|k| admission.lo0[k] <= l[k] && l[k] <= h[k] && h[k] <= admission.hi0[k])
    });
    if !inside {
        return Err("every box must be an ordered sub-box of the root box".into());
    }
    let mut prog = admission.prog.clone();
    prog.refine_max = cfg.refine_max_width;
    prog.joint = cfg.joint_quadratic;
    let one = [&prog];
    let fam = Fam::single(&one);
    let mut out = Vec::with_capacity(lo.len() / d);
    for (l, h) in lo.chunks(d).zip(hi.chunks(d)) {
        let ch = Chunk::new(1, d, l, h);
        let (vals, poison) = forward(fam, &ch);
        let row = prog
            .sites
            .iter()
            .map(|site| {
                let v = vals[site.node].as_ref().expect("forward ran");
                let planes = !poison[0] && v.joint_ok.first().copied().unwrap_or(false);
                let (slo, shi) = if poison[0] {
                    (f64::NEG_INFINITY, f64::INFINITY)
                } else {
                    (v.lo[0], v.hi[0])
                };
                JointSiteBox {
                    name: site.name.clone(),
                    lo: slo,
                    hi: shi,
                    lower: planes.then(|| (v.joint_lo[..d].to_vec(), v.joint_lo[d])),
                    upper: planes.then(|| (v.joint_hi[..d].to_vec(), v.joint_hi[d])),
                }
            })
            .collect();
        out.push(row);
    }
    Ok(out)
}

/// #lsnc-lane stage 2: the joint bound at the admission's ROOT box — the
/// lane's engagement observable ([`emit_joint_line`]).
#[derive(Debug, Clone, PartialEq)]
pub struct JointRootReport {
    /// `cfg.joint_quadratic`.
    pub armed: bool,
    /// Every site's root result (the plain forward's bounds when unarmed).
    pub sites: Vec<JointSiteBox>,
    /// The PUBLISHED root upper bound of output 0 and lower bound of output 1
    /// (the values the branch and bound compares): lsnc_relu's `Y_0 = dV`
    /// and `Y_1 = V`. NaN when the output is narrower than 2.
    pub dv_ub: f64,
    pub v_lb: f64,
    /// Wall seconds of the report itself.
    pub secs: f64,
}

/// Bound the admission's root box for [`JointRootReport`].
pub fn joint_root_report(admission: &LaneAdmission, cfg: &LowDimConfig) -> JointRootReport {
    let t0 = Instant::now();
    let (lo, hi) = (&admission.lo0, &admission.hi0);
    let sites = joint_site_boxes(admission, lo, hi, cfg)
        .ok()
        .and_then(|mut rows| rows.pop())
        .unwrap_or_default();
    let width = admission.output_width();
    let mut objectives = Vec::new();
    if width >= 2 {
        let mut e0 = vec![0.0; width];
        e0[0] = -1.0;
        let mut e1 = vec![0.0; width];
        e1[1] = 1.0;
        objectives.push(e0);
        objectives.push(e1);
    }
    let (dv_ub, v_lb) = match bound_boxes(admission, lo, hi, &objectives, cfg) {
        Ok(rows) if rows.len() == 1 && rows[0].len() == 2 => (-rows[0][0], rows[0][1]),
        _ => (f64::NAN, f64::NAN),
    };
    JointRootReport {
        armed: cfg.joint_quadratic,
        sites,
        dv_ub,
        v_lb,
        secs: t0.elapsed().as_secs_f64(),
    }
}

/// The joint engagement line, printed unconditionally by the grouped entry:
/// `[lowdim] joint sites=N armed=0|1 root dv_ub=.. v_lb=.. site[<name>]=[lo,hi]
/// planes=0|1 .. secs=..`. `planes=1` on a site is the proof the joint pass
/// ran there; `armed=1` with `planes=0` everywhere is VOID, not negative.
pub fn emit_joint_line(report: &JointRootReport) {
    use std::fmt::Write as _;
    let mut sites = String::new();
    for s in &report.sites {
        let _ = write!(
            sites,
            " site[{}]=[{:.6e},{:.6e}] planes={}",
            s.name,
            s.lo,
            s.hi,
            u8::from(s.lower.is_some())
        );
    }
    eprintln!(
        "[lowdim] joint sites={} armed={} root dv_ub={:.6e} v_lb={:.6e}{sites} secs={:.3}",
        report.sites.len(),
        u8::from(report.armed),
        report.dv_ub,
        report.v_lb,
        report.secs
    );
}

/// Try to prove `lower(objective · y(x)) > threshold` for every `x` in the
/// input box by input-space branch and bound.
///
/// Never claims a violation. `Unproved` and `Declined` claim nothing.
pub fn prove_objective_lower_bound(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objective: &[f32],
    threshold: f32,
    deadline: Instant,
    cfg: &LowDimConfig,
) -> LowDimOutcome {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::LowDim);
    prove_conjunction_infeasible(
        graph,
        input,
        std::slice::from_ref(&objective.to_vec()),
        &[threshold],
        deadline,
        cfg,
    )
}

/// The Sequential lane's contract: prove `network(x)[0] > threshold` over the
/// box. Builds the one-hot objective at the lowered graph's real output width.
pub fn prove_first_output_exceeds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    threshold: f32,
    deadline: Instant,
    cfg: &LowDimConfig,
) -> LowDimOutcome {
    prove_first_output(graph, input, threshold, false, deadline, cfg)
}

/// The Sequential lane's contract in either mode: `upper == false` proves
/// `network(x)[0] > threshold`; `upper == true` proves
/// `network(x)[0] < threshold`, which is `lower(-network(x)[0]) > -threshold`
/// (f32 negation is exact, so this is the same predicate, not an approximation).
pub fn prove_first_output(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    threshold: f32,
    upper: bool,
    deadline: Instant,
    cfg: &LowDimConfig,
) -> LowDimOutcome {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::LowDim);
    prove_first_output_witnessed(graph, input, threshold, upper, deadline, cfg, &mut None)
}

/// [`prove_first_output`], and when the lane stops on
/// [`LowDimStop::Candidate`] it also writes that leaf's centre (input
/// coordinates, f64) to `witness`. The point is only a CANDIDATE: the lane
/// evaluated it on its f64 lowering, so a caller must re-evaluate it on the
/// real network before treating it as a counterexample. `witness` is left
/// untouched on every other outcome.
pub fn prove_first_output_witnessed(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    threshold: f32,
    upper: bool,
    deadline: Instant,
    cfg: &LowDimConfig,
    witness: &mut Option<Vec<f64>>,
) -> LowDimOutcome {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::LowDim);
    let mut prog = match lower_graph_opts(graph, input, LowerOpts::of(cfg)) {
        Ok(prog) => prog,
        Err(reason) => {
            emit_varying_lowering_decline(cfg, input);
            return LowDimOutcome::Declined(reason);
        }
    };
    let width = prog.sizes[prog.out];
    if width == 0 {
        return LowDimOutcome::Declined("empty output".into());
    }
    // A trailing GLOBAL max (the same-LHS conjunction reduction, through any
    // Reshape/Flatten identities): `max_j z_j > t` holds on a leaf as soon as
    // ANY `lower(z_j) > t` — exactly the conjunction rule — so seed the objectives
    // at the pre-max vector, one per branch, with full backward bounds for each,
    // instead of committing to one branch inside the max relaxation.
    if !upper {
        let mut node = prog.out;
        while let Op::Identity { src } = prog.ops[node] {
            node = src;
        }
        if let Op::MaxGlobal { src } = prog.ops[node] {
            let k = prog.sizes[src];
            prog.seed = src;
            let objectives: Vec<Vec<f32>> = (0..k)
                .map(|j| {
                    let mut e = vec![0.0f32; k];
                    e[j] = 1.0;
                    e
                })
                .collect();
            let thresholds = vec![threshold; k];
            return prove_conjunction_impl(
                graph,
                input,
                &objectives,
                &thresholds,
                deadline,
                cfg,
                Some(prog),
                witness,
            );
        }
    }
    let sign = if upper { -1.0f32 } else { 1.0f32 };
    let mut first = vec![0.0f32; width];
    first[0] = sign;
    prove_conjunction_impl(
        graph,
        input,
        &[first],
        &[sign * threshold],
        deadline,
        cfg,
        Some(prog),
        witness,
    )
}

/// Try to prove a CONJUNCTION infeasible over the whole input box: every leaf
/// must satisfy `lower(objectives[i] · y) > thresholds[i]` for AT LEAST ONE
/// `i` (if any conjunct is impossible on a region, the conjunction is). With a
/// single objective this is exactly [`prove_objective_lower_bound`].
///
/// A leaf's centre counts as a violation candidate only when it satisfies EVERY
/// conjunct (`objective_i · y <= threshold_i` for all `i`). Never claims a
/// violation; `Unproved` and `Declined` claim nothing.
pub fn prove_conjunction_infeasible(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    deadline: Instant,
    cfg: &LowDimConfig,
) -> LowDimOutcome {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::LowDim);
    prove_conjunction_impl(
        graph, input, objectives, thresholds, deadline, cfg, None, &mut None,
    )
}

/// Shared driver. `prepared`, when given, is a program already lowered from
/// `graph` (possibly with its objective `seed` moved); objectives are over
/// `prog.sizes[prog.seed]`. On a [`LowDimStop::Candidate`] stop the candidate
/// centre is written to `witness`.
#[allow(clippy::too_many_arguments)]
fn prove_conjunction_impl(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    deadline: Instant,
    cfg: &LowDimConfig,
    prepared: Option<Program>,
    witness: &mut Option<Vec<f64>>,
) -> LowDimOutcome {
    let lo0: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let hi0: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    let d = lo0.len();
    // #lowdim-bundle (1): armed, a box wider than the cap is admitted by its
    // VARYING coordinates (`pinned`); unarmed (and at or below the cap) the
    // historical checks run unchanged.
    let pins = if cfg.varying_dims {
        pin_counts(&lo0, &hi0)
    } else {
        PinCounts::default()
    };
    let fold = if cfg.varying_dims {
        match pinned::PinnedFold::of(&lo0, &hi0) {
            Ok(fold) => fold,
            Err(reason) => {
                emit_varying_line(pinned::FoldStatus::Declined, d, 0, 0, 0, pins);
                return LowDimOutcome::Declined(reason);
            }
        }
    } else {
        None
    };
    if fold.is_none() {
        if d == 0 || d > MAX_INPUT_DIM {
            return LowDimOutcome::Declined(format!(
                "input dimension {d} outside 1..={MAX_INPUT_DIM}"
            ));
        }
        if lo0.iter().chain(&hi0).any(|v| !v.is_finite())
            || lo0.iter().zip(&hi0).any(|(l, h)| l > h)
        {
            return LowDimOutcome::Declined("input box is not a finite, ordered box".into());
        }
    }
    if objectives.is_empty() || objectives.len() != thresholds.len() {
        return LowDimOutcome::Declined(format!(
            "{} objectives vs {} thresholds",
            objectives.len(),
            thresholds.len()
        ));
    }
    if thresholds.iter().any(|t| !t.is_finite())
        || objectives.iter().flatten().any(|v| !v.is_finite())
    {
        return LowDimOutcome::Declined("non-finite objective or threshold".into());
    }
    let prog = match prepared {
        Some(p) => p,
        None => match lower_graph_opts(graph, input, LowerOpts::of(cfg)) {
            Ok(p) => p,
            Err(reason) => {
                // Review fix: an armed entry that declines in the lowering
                // still prints its `[lowdim-varying]` line.
                if cfg.varying_dims {
                    let d_free = fold.as_ref().map_or(d, pinned::PinnedFold::free_dim);
                    emit_varying_line(pinned::FoldStatus::Declined, d, d_free, 0, 0, pins);
                }
                return LowDimOutcome::Declined(reason);
            }
        },
    };
    let objs: Vec<Vec<f64>> = objectives
        .iter()
        .map(|o| o.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let thrs: Vec<f64> = thresholds.iter().map(|&t| f64::from(t)).collect();
    let Some(fold) = fold else {
        if cfg.varying_dims {
            emit_varying_line(
                pinned::FoldStatus::WithinCap,
                d,
                d,
                0,
                sigmoid_ops(&prog),
                pins,
            );
        }
        return prove_conjunction_core_taped(
            prog, lo0, hi0, objs, thrs, deadline, cfg, witness, None,
        );
    };
    let folded = match apply_fold(&fold, prog) {
        Ok(folded) => folded,
        Err(reason) => return LowDimOutcome::Declined(reason),
    };
    let (flo, fhi) = (fold.restrict(&lo0), fold.restrict(&hi0));
    let mut free_witness = None;
    let outcome = prove_conjunction_core_taped(
        folded,
        flo,
        fhi,
        objs,
        thrs,
        deadline,
        cfg,
        &mut free_witness,
        Some(&fold),
    );
    if let Some(x) = free_witness {
        *witness = Some(fold.expand(&x));
    }
    outcome
}

/// The branch-and-bound over a prepared program and an f64 input box `[lo0,
/// hi0]` of the program's input dimension. Objectives are over
/// `prog.sizes[prog.seed]`; a leaf closes as soon as ANY objective's lower
/// bound exceeds its threshold.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_conjunction_core(
    prog: Program,
    lo0: Vec<f64>,
    hi0: Vec<f64>,
    objs: Vec<Vec<f64>>,
    thrs: Vec<f64>,
    deadline: Instant,
    cfg: &LowDimConfig,
    witness: &mut Option<Vec<f64>>,
) -> LowDimOutcome {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::LowDim);
    prove_conjunction_core_taped(prog, lo0, hi0, objs, thrs, deadline, cfg, witness, None)
}

/// [`prove_conjunction_core`] with the fold that maps the lane's box to the
/// FULL input box (`None`: the lane's box is the full box). B2.2: under the
/// Proof profile (the lane proof capture armed) the tree keeps a write-only
/// proof tape — every closed leaf and every clip slab as a full-input box,
/// discharging the conjunction (SOME conjunct certified: one group of all
/// objectives) — deposited COMPLETE when the queue drains (every fate of
/// every dequeued box is recorded, so the leaves and slabs tile the root).
/// Under Competition no tape exists: each fate costs one `Option` test.
#[allow(clippy::too_many_arguments)]
fn prove_conjunction_core_taped(
    mut prog: Program,
    lo0: Vec<f64>,
    hi0: Vec<f64>,
    objs: Vec<Vec<f64>>,
    thrs: Vec<f64>,
    deadline: Instant,
    cfg: &LowDimConfig,
    witness: &mut Option<Vec<f64>>,
    embed: Option<&pinned::PinnedFold>,
) -> LowDimOutcome {
    let t0 = Instant::now();
    let d = lo0.len();
    if d == 0 || d != prog.d || hi0.len() != d {
        return LowDimOutcome::Declined(format!("box of {d} dims for a {}-input program", prog.d));
    }
    if lo0.iter().chain(&hi0).any(|v| !v.is_finite()) || lo0.iter().zip(&hi0).any(|(l, h)| l > h) {
        return LowDimOutcome::Declined("input box is not a finite, ordered box".into());
    }
    if objs.is_empty()
        || objs.len() != thrs.len()
        || thrs.iter().any(|t| !t.is_finite())
        || objs.iter().flatten().any(|v| !v.is_finite())
    {
        return LowDimOutcome::Declined("empty or non-finite objectives/thresholds".into());
    }
    prog.refine_max = cfg.refine_max_width;
    prog.joint = cfg.joint_quadratic;
    let width = prog.sizes[prog.seed];
    if let Some(bad) = objs.iter().find(|o| o.len() != width) {
        return LowDimOutcome::Declined(format!(
            "objective has {} entries for a width-{width} output",
            bad.len()
        ));
    }
    // Config is public: clamp it rather than trust it. The chunk also shrinks
    // so one chunk's forms (2 per node, (d + 1) columns, f64) fit CHUNK_BYTES.
    let batch = cfg.batch.max(1);
    // Review fix (#lowdim-bundle (1)): an `Op::Sigmoid` node also stores its
    // four planes per neuron and domain (`NodeVal::sig`, 32 bytes) beside its
    // forms; count them so the clamp bounds the whole forward working set. 0
    // for every program without a sigmoid (declined before the bundle), so
    // the historical chunk is unchanged.
    let sig_bytes = prog
        .ops
        .iter()
        .zip(&prog.sizes)
        .filter(|(op, _)| matches!(op, Op::Sigmoid { .. }))
        .map(|(_, &n)| n)
        .sum::<usize>()
        .saturating_mul(32);
    let per_domain_bytes = prog
        .total
        .saturating_mul(d + 1)
        .saturating_mul(16)
        .saturating_add(sig_bytes)
        .max(1);
    let chunk = cfg
        .chunk
        .max(1)
        .min((CHUNK_BYTES / per_domain_bytes).max(1));

    // #lsnc-lane stage 3: the root widths (the split rules' aspect guard).
    let w0: Vec<f64> = lo0.iter().zip(&hi0).map(|(l, h)| h - l).collect();
    let mut stats = LowDimStats {
        joint_sites: if prog.joint {
            u32::try_from(prog.sites.len()).unwrap_or(u32::MAX)
        } else {
            0
        },
        threads: lane_threads(cfg),
        ..LowDimStats::default()
    };
    let mut q_lo: Vec<f64> = lo0.clone();
    let mut q_hi: Vec<f64> = hi0.clone();
    let mut q_depth: Vec<u32> = vec![0];
    // #lowdim-bundle (2)/(3): each armed lever prints its own engagement line
    // when the run ends, however it ends.
    let (alpha_iters, attack_every) = (cfg.alpha_iters, cfg.leaf_attack_every);
    // M8a: the untrusted wave search, armed only under
    // `lowdim_batch_engine: bbe` (never consulted otherwise; see `bbe`).
    let mut bbe_run = (cfg.batch_engine == LowDimBatchEngine::Bbe)
        .then(|| bbe::arm_single(&prog, &lo0, &hi0, &objs, cfg));
    let finish = |mut stats: LowDimStats, run: Option<&bbe::BbeRun>, stop: Option<LowDimStop>| {
        stats.secs = t0.elapsed().as_secs_f64();
        if let Some(run) = run {
            run.emit();
        }
        if alpha_iters > 0 {
            alpha::emit_line(&stats, alpha_iters);
        }
        if attack_every > 0 {
            attack::emit_line(&stats, attack_every);
        }
        match stop {
            None => LowDimOutcome::Verified(stats),
            Some(s) => LowDimOutcome::Unproved(stats, s),
        }
    };
    // B2.2: the write-only proof tape (Proof profile only; see the doc). Only
    // objectives over the network OUTPUT are instance objectives: a seed moved
    // to an inner node (the trailing-global-max decomposition) streams none.
    let tape = if prog.seed == prog.out {
        let (tlo, thi) = embed.map_or((lo0.as_slice(), hi0.as_slice()), |f| f.full_box());
        crate::certify::lane_proof::TapeRecorder::for_boxes("lowdim", tlo, thi, &objs, &thrs)
    } else {
        None
    };
    let all_objs: Vec<u32> = (0..objs.len())
        .filter_map(|i| u32::try_from(i).ok())
        .collect();
    let full = |lo: &[f64], hi: &[f64]| -> (Vec<f64>, Vec<f64>) {
        embed.map_or_else(|| (lo.to_vec(), hi.to_vec()), |f| f.embed_box(lo, hi))
    };
    // #router-r6 `rule.trial_width`: the wave width's AIMD machine (`None`
    // unless the acting optimizer armed the site: every wave then takes
    // exactly `batch`, today's width), inside `[batch / 4, batch x 4]` and
    // [`LOWDIM_TRIAL_MAX_BATCH`]. A wave is the unit, its top the safe point;
    // the metric is closed domains per second; lease use is the open queue
    // against the lane's own `max_queue` bound.
    let trials = crate::trial_sites::global();
    let mut width_trial = crate::trial_sites::width_trial(
        trials,
        ny_core::trial_authority::TrialSite::LowdimWidth,
        batch as u64,
        1,
        LOWDIM_TRIAL_MAX_BATCH,
    );
    // #router-r6 `rule.trial_split_restart`: at most ONE restart with the
    // other split heuristic, on FAR AND a zero-lift fraction > 0.9 (`None`
    // unless armed). Zero lift is tracked only while a machine exists: each
    // queued child carries its parent's margin, compared with its own when
    // it is bounded.
    let mut split_trial = crate::trial_sites::split_restart(
        trials,
        cfg.split.token(),
        restart_split(cfg.split).token(),
    );
    let mut q_parent: Vec<f64> = if split_trial.is_some() {
        vec![f64::NAN]
    } else {
        Vec::new()
    };
    let (mut lift_children, mut lift_zero) = (0u64, 0u64);
    // The configuration the waves run with: today's `cfg`, or (after the
    // one restart) a copy with the other split heuristic. Nothing else of
    // the configuration changes.
    let mut restarted: Option<LowDimConfig> = None;
    loop {
        let live_cfg: &LowDimConfig = restarted.as_ref().unwrap_or(cfg);
        let wave = search::Wave {
            prog: &prog,
            objs: &objs,
            thrs: &thrs,
            cfg: live_cfg,
            w0: &w0,
            chunk,
        };
        let open = q_depth.len();
        if open == 0 {
            if let Some(tape) = &tape {
                tape.deposit(true);
            }
            return finish(stats, bbe_run.as_ref(), None);
        }
        // The deadline is polled once per wave: a wave is at most `batch`
        // domains, bounded (and, `LowDimParallel::Full`, decided) in parallel.
        if Instant::now() >= deadline {
            return finish(stats, bbe_run.as_ref(), Some(LowDimStop::Deadline));
        }
        if open > cfg.max_queue {
            return finish(stats, bbe_run.as_ref(), Some(LowDimStop::Queue));
        }
        if router_safe_point(&stats, open) {
            return finish(stats, bbe_run.as_ref(), Some(LowDimStop::HandOn));
        }
        // #router-r6: the wave top is also the trial rules' safe point. A
        // restart DISCARDS the tree — the queue goes back to the root box,
        // the current tree's certified volume back to zero — and carries no
        // fact of it into the new one: a proof now needs the new tree to
        // close every leaf. Counters of work done (`domains`, `waves`,
        // `closed`) stay cumulative.
        if split_trial.is_some() {
            let (t_s, remaining_s) = crate::trial_sites::lane_clock(t0, Some(deadline));
            let restart = crate::trial_sites::observe_split(
                trials,
                &mut split_trial,
                ny_core::router::trial::SplitObs {
                    units: stats.waves,
                    closed_volume: stats.closed_volume,
                    open: u64::try_from(open).unwrap_or(u64::MAX),
                    children: lift_children,
                    zero_lift: lift_zero,
                    t_s,
                    remaining_s,
                },
            );
            if restart {
                restart_queue_from_root(
                    &mut q_lo,
                    &mut q_hi,
                    &mut q_depth,
                    &mut q_parent,
                    &lo0,
                    &hi0,
                );
                stats.closed_volume = 0.0;
                restarted = Some(LowDimConfig {
                    split: restart_split(cfg.split),
                    ..cfg.clone()
                });
                continue;
            }
        }
        let take = crate::trial_sites::width_value(width_trial.as_ref(), batch).min(open);
        let at = open - take;
        let b_lo = q_lo.split_off(at * d);
        let b_hi = q_hi.split_off(at * d);
        let b_depth = q_depth.split_off(at);
        let b_parent = if split_trial.is_some() {
            q_parent.split_off(at.min(q_parent.len()))
        } else {
            Vec::new()
        };
        // Every domain's FATE — closed, emptied by its own clip, a violation
        // candidate, unsplittable, or two children of its (clipped) box — is
        // decided by `search::fate` from that domain alone; the merge below
        // walks the fates in domain order, so the outcome and the counters do
        // not depend on where the fates were computed.
        let fates = match bbe_run.as_mut() {
            Some(run) => match bbe::single_wave(run, &wave, &b_lo, &b_hi, take, deadline) {
                bbe::WaveResult::Done(fates) => fates,
                bbe::WaveResult::Deadline => {
                    return finish(stats, bbe_run.as_ref(), Some(LowDimStop::Deadline))
                }
                bbe::WaveResult::Legacy => wave.run(&b_lo, &b_hi, take),
            },
            None => wave.run(&b_lo, &b_hi, take),
        };
        stats.waves += 1;
        stats.domains += take as u64;
        if fates.len() != take {
            // Every dequeued domain must come back decided; a short result
            // would silently drop open leaves and could read as a proof.
            return finish(stats, bbe_run.as_ref(), Some(LowDimStop::NonFinite));
        }
        // #lowdim-bundle (3): on a wave the leaf attack is due, this wave's
        // open leaves (margin, box) are its candidates.
        let attack_due = attack_every > 0
            && stats
                .waves
                .is_multiple_of(u64::try_from(attack_every).unwrap_or(u64::MAX));
        let mut open_leaves: Vec<(f64, Vec<f64>, Vec<f64>)> = Vec::new();
        for (b, (fate, note)) in fates.into_iter().enumerate() {
            // #router-r6: zero lift (counter only, while a restart machine
            // exists): did this child's bound rise above its parent's?
            if let Some(parent) = b_parent.get(b).copied() {
                if let Some(zero) = ny_core::router::trial::zero_lift(parent, note.margin) {
                    lift_children = lift_children.saturating_add(1);
                    lift_zero = lift_zero.saturating_add(u64::from(zero));
                }
            }
            stats.joint_hits += u64::from(note.joint_hits);
            stats.alpha_domains += u64::from(note.alpha.ran);
            stats.alpha_improved += u64::from(note.alpha.improved);
            stats.alpha_rescued += u64::from(note.alpha.rescued);
            match note.claim {
                bbe::ClaimMark::Certified => {
                    stats.bbe_claimed += 1;
                    stats.bbe_certified += 1;
                }
                bbe::ClaimMark::Refuted => {
                    stats.bbe_claimed += 1;
                    stats.bbe_refuted += 1;
                }
                bbe::ClaimMark::Unclaimed | bbe::ClaimMark::None => {}
            }
            // M8a router progress (counter only): a closed leaf's volume, or
            // the volume its clip removed.
            let leaf_vol = box_volume(&b_lo[b * d..(b + 1) * d], &b_hi[b * d..(b + 1) * d], &w0);
            if let Some(tape) = &tape {
                let (plo, phi) = full(&b_lo[b * d..(b + 1) * d], &b_hi[b * d..(b + 1) * d]);
                let groups = vec![all_objs.clone()];
                match &fate {
                    search::Fate::Closed => tape.note_box(
                        &plo,
                        &phi,
                        crate::certify::lane_proof::TapeClose::Bound,
                        groups,
                        note.margin,
                    ),
                    search::Fate::ClipClosed => tape.note_box(
                        &plo,
                        &phi,
                        crate::certify::lane_proof::TapeClose::Clip,
                        groups,
                        f64::NAN,
                    ),
                    search::Fate::Split {
                        lo,
                        hi,
                        shrunk: true,
                        ..
                    }
                    | search::Fate::Requeue { lo, hi } => {
                        let (qlo, qhi) = full(lo, hi);
                        tape.note_clip(&plo, &phi, &qlo, &qhi, &groups);
                    }
                    _ => {}
                }
            }
            match &fate {
                search::Fate::Closed | search::Fate::ClipClosed => stats.closed_volume += leaf_vol,
                search::Fate::Split {
                    lo,
                    hi,
                    shrunk: true,
                    ..
                }
                | search::Fate::Requeue { lo, hi } => {
                    stats.closed_volume += (leaf_vol - box_volume(lo, hi, &w0)).max(0.0);
                }
                _ => {}
            }
            match fate {
                search::Fate::NonFinite => {
                    return finish(stats, bbe_run.as_ref(), Some(LowDimStop::NonFinite))
                }
                // Closed iff SOME conjunct is proved impossible on this leaf
                // (the strict `lb > thr` itself, never a subtracted margin).
                search::Fate::Closed => stats.closed += 1,
                // The leaf's own planes admit no point of its box.
                search::Fate::ClipClosed => {
                    stats.closed += 1;
                    stats.clip_closed += 1;
                }
                search::Fate::Candidate(centre) => {
                    *witness = Some(centre);
                    return finish(stats, bbe_run.as_ref(), Some(LowDimStop::Candidate));
                }
                // Too narrow, or adjacent floats (a child would equal its
                // parent and the lane would spin until the deadline).
                search::Fate::Resolution => {
                    return finish(stats, bbe_run.as_ref(), Some(LowDimStop::Resolution))
                }
                // The clip left an unsplittable (strictly smaller) box:
                // bound it again, at the same depth.
                search::Fate::Requeue { lo, hi } => {
                    stats.clip_shrunk += 1;
                    q_lo.extend_from_slice(&lo);
                    q_hi.extend_from_slice(&hi);
                    q_depth.push(b_depth[b]);
                    if split_trial.is_some() {
                        // The same domain again, not a child: no lift is
                        // measured on it.
                        q_parent.push(f64::NAN);
                    }
                    if attack_due {
                        open_leaves.push((note.margin, lo, hi));
                    }
                }
                search::Fate::Split {
                    lo,
                    hi,
                    dim,
                    mid,
                    shrunk,
                } => {
                    stats.clip_shrunk += u64::from(shrunk);
                    let depth = b_depth[b] + 1;
                    stats.max_depth = stats.max_depth.max(depth);
                    // left child [lo, mid], right child [mid, hi] of the
                    // (clipped) box
                    q_lo.extend_from_slice(&lo);
                    q_hi.extend_from_slice(&hi);
                    let li = q_hi.len() - d + dim;
                    q_hi[li] = mid;
                    q_lo.extend_from_slice(&lo);
                    q_hi.extend_from_slice(&hi);
                    let ri = q_lo.len() - d + dim;
                    q_lo[ri] = mid;
                    q_depth.push(depth);
                    q_depth.push(depth);
                    if split_trial.is_some() {
                        q_parent.push(note.margin);
                        q_parent.push(note.margin);
                    }
                    if attack_due {
                        open_leaves.push((note.margin, lo, hi));
                    }
                }
            }
        }
        // #router-r6 `rule.trial_width`: the wave is done; the machine
        // decides the NEXT wave's width from closed domains per second.
        if width_trial.is_some() {
            let (t_s, remaining_s) = crate::trial_sites::lane_clock(t0, Some(deadline));
            let queued = q_depth.len() as f64;
            let bound = cfg.max_queue.max(1) as f64;
            crate::trial_sites::observe_width(
                trials,
                &mut width_trial,
                ny_core::router::trial::WidthObs {
                    units: stats.waves,
                    closes: stats.closed,
                    t_s,
                    remaining_s,
                    lease_use: Some(queued / bound),
                },
            );
        }
        // #lowdim-bundle (3): attack-only. A hit is a CANDIDATE on the
        // lowered program (the centre test's own predicate); the caller's
        // trusted forward decides it.
        if attack_due && !open_leaves.is_empty() {
            if let Some(x) = attack::round(
                &prog,
                &objs,
                &thrs,
                cfg,
                open_leaves,
                &mut stats,
                t0,
                deadline,
            ) {
                *witness = Some(x);
                return finish(stats, bbe_run.as_ref(), Some(LowDimStop::Candidate));
            }
        }
    }
}

/// Bound a batch of domains drawn from a FAMILY of programs (same structure,
/// shared weights — see [`same_structure`]): `which[i]` is domain `i`'s
/// program and `objs[i]` its objectives (over the common seed). One forward
/// per chunk serves every domain, so a shared weight matrix is read once per
/// chunk instead of once per clause. Returns per domain `(lower bound per
/// objective, gradient per objective)` for that domain's own objectives.
fn bound_batch_family(
    progs: &[&Program],
    which: &[usize],
    lo: &[f64],
    hi: &[f64],
    objs: &[&[Vec<f64>]],
    chunk: usize,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let base = progs[0];
    let d = base.d;
    let count = which.len();
    let zero = vec![0.0f64; base.sizes[base.seed]];
    let kmax = objs.iter().map(|o| o.len()).max().unwrap_or(0);
    let chunk = chunk.max(1);
    let starts: Vec<usize> = (0..count).step_by(chunk).collect();
    starts
        .par_iter()
        .flat_map_iter(|&s| {
            let m = chunk.min(count - s);
            let ch = Chunk::new(m, d, &lo[s * d..(s + m) * d], &hi[s * d..(s + m) * d]);
            let fam = Fam {
                progs,
                which: Some(&which[s..s + m]),
            };
            let (vals, poison) = forward(fam, &ch);
            let per_k: Vec<(Vec<f64>, Vec<f64>)> = (0..kmax)
                .map(|k| {
                    let ob: Vec<&[f64]> = (0..m)
                        .map(|b| objs[s + b].get(k).map_or(&zero[..], |v| &v[..]))
                        .collect();
                    objective_bounds(fam, &vals, &ch, base.seed, &ob, &poison)
                })
                .collect();
            (0..m)
                .map(|b| {
                    let nk = objs[s + b].len();
                    let lbs = per_k[..nk].iter().map(|(lb, _)| lb[b]).collect();
                    let grads = per_k[..nk]
                        .iter()
                        .flat_map(|(_, g)| g[b * d..(b + 1) * d].iter().copied())
                        .collect();
                    (lbs, grads)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// One clause of a family run: its program, input box (over the program's
/// inputs), objectives and thresholds. The clause closes on a leaf as soon as
/// ANY conjunct is impossible there; `combine` says how a conjunct reads the
/// objectives' lower bounds (one objective against its threshold for every
/// clause kind except `#lowdim-fold-sigdiff`).
pub(crate) struct FamClause {
    pub(crate) prog: Program,
    pub(crate) lo: Vec<f64>,
    pub(crate) hi: Vec<f64>,
    pub(crate) objs: Vec<Vec<f64>>,
    /// One threshold per CONJUNCT.
    pub(crate) thrs: Vec<f64>,
    pub(crate) combine: Combine,
}

/// How a [`FamClause`]'s objective lower bounds decide one of its conjuncts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Combine {
    /// Conjunct `j` is objective `j` alone: impossible on a leaf iff
    /// `lb_j > thrs[j]` (strict). `objs.len() == thrs.len()`.
    Each,
    /// `#lowdim-fold-sigdiff` (the mscn dual with BOTH copies free): conjunct
    /// `j` reads objectives `2j` and `2j + 1`, whose rigorous lower bounds `la`,
    /// `lb` bound the sign-folded sigmoid pre-activations `wA`, `wB` from below.
    /// It is impossible on a leaf iff both are FINITE and
    /// `sigmoid_sum_lower(la, lb) > thrs[j]` (strict; [`sigmoid_sum_lower`]): sigmoid is
    /// increasing, so that is a rigorous lower bound of
    /// `sigmoid(wA) + sigmoid(wB)` over the leaf. `objs.len() == 2 * thrs.len()`.
    ///
    /// `centre[j] = (t, shift)` serves only the violation-CANDIDATE heuristic:
    /// at a point, the conjunct's side `s * Y` of the output is
    /// `sigmoid(wA) + sigmoid(wB) + shift` (the interval constant at its
    /// midpoint), a candidate when it is at most `t - candidate_margin(t)`.
    SigmoidSum { centre: Vec<(f64, f64)> },
}

/// A rigorous lower bound of `sigmoid(x) = 1 / (1 + exp(-x))` for a FINITE
/// `x`, never above the real value. The f64 evaluation carries a few ulps of
/// relative error (libm `exp`, one add, one divide); `1e-12` relative plus one
/// `next_down` dominates it by four orders of magnitude, and the absolute
/// `1e-300` covers the subnormal range where a relative bound does not hold.
/// A non-finite `x` must never reach here (`f64::max` would drop a NaN).
pub(crate) fn sigmoid_down(x: f64) -> f64 {
    let s = 1.0 / (1.0 + (-x).exp());
    (next_down(s * (1.0 - 1e-12)) - 1e-300).max(0.0)
}

/// A rigorous lower bound of `sigmoid(wA) + sigmoid(wB)` given rigorous lower
/// bounds `la <= wA`, `lb <= wB`; `-inf` (never closes) unless both are finite.
pub(crate) fn sigmoid_sum_lower(la: f64, lb: f64) -> f64 {
    if !(la.is_finite() && lb.is_finite()) {
        return f64::NEG_INFINITY;
    }
    // Both terms lie in [0, 1]: the rounded sum is within half an ulp of the
    // exact one, so one `next_down` puts it at or below.
    next_down(sigmoid_down(la) + sigmoid_down(lb))
}

impl FamClause {
    /// Objectives per conjunct.
    fn per_conjunct(&self) -> usize {
        match self.combine {
            Combine::Each => 1,
            Combine::SigmoidSum { .. } => 2,
        }
    }

    /// The rigorous lower bound of conjunct `j`'s sigmoid sum on a leaf
    /// (`-inf` when a bound is missing or non-finite).
    fn sig_sum_bound(lbs: &[f64], j: usize) -> f64 {
        match (lbs.get(2 * j), lbs.get(2 * j + 1)) {
            (Some(&la), Some(&lb)) => sigmoid_sum_lower(la, lb),
            _ => f64::NEG_INFINITY,
        }
    }

    /// Some conjunct is impossible on this leaf. The comparison is the strict
    /// `bound > thr` itself, never a subtracted margin.
    fn closes(&self, lbs: &[f64]) -> bool {
        match self.combine {
            Combine::Each => lbs.iter().zip(&self.thrs).any(|(lb, thr)| lb > thr),
            Combine::SigmoidSum { .. } => {
                lbs.len() == 2 * self.thrs.len()
                    && self
                        .thrs
                        .iter()
                        .enumerate()
                        .any(|(j, thr)| Self::sig_sum_bound(lbs, j) > *thr)
            }
        }
    }

    /// Every conjunct is violated (by `candidate_margin`) at the leaf centre:
    /// a likely genuine violation (heuristic; gives the clause up, never a
    /// verdict).
    fn is_candidate(&self, centre: &[f64]) -> bool {
        match &self.combine {
            Combine::Each => self.objs.iter().zip(&self.thrs).all(|(o, thr)| {
                point_objective(&self.prog, centre, o) <= thr - candidate_margin(*thr)
            }),
            Combine::SigmoidSum { centre: cs } => {
                let y = point_values(&self.prog, centre);
                let dot = |o: &[f64]| y.iter().zip(o).map(|(v, w)| v * w).sum::<f64>();
                let sig = |w: f64| 1.0 / (1.0 + (-w).exp());
                cs.iter().enumerate().all(|(j, &(t, shift))| {
                    let v = sig(dot(&self.objs[2 * j][..]))
                        + sig(dot(&self.objs[2 * j + 1][..]))
                        + shift;
                    v <= t - candidate_margin(t)
                })
            }
        }
    }

    /// The objective whose input-space multiplier steers the split: the
    /// conjunct closest to closing, and (for a sigmoid sum) the worse of its
    /// two terms. Any choice terminates; the choice never affects soundness.
    fn split_objective(&self, lbs: &[f64]) -> usize {
        match self.combine {
            Combine::Each => first_argmax(lbs.iter().zip(&self.thrs).map(|(lb, thr)| lb - thr)),
            Combine::SigmoidSum { .. } => {
                let j = first_argmax(
                    self.thrs
                        .iter()
                        .enumerate()
                        .map(|(j, thr)| Self::sig_sum_bound(lbs, j) - thr),
                );
                match (lbs.get(2 * j), lbs.get(2 * j + 1)) {
                    (Some(la), Some(lb)) if lb < la => 2 * j + 1,
                    _ => 2 * j,
                }
            }
        }
    }
}

/// Index of the first maximum (`0` when nothing exceeds `-inf`).
fn first_argmax(it: impl Iterator<Item = f64>) -> usize {
    it.enumerate()
        .fold((0usize, f64::NEG_INFINITY), |acc, (j, v)| {
            if v > acc.1 {
                (j, v)
            } else {
                acc
            }
        })
        .0
}

/// Clauses that may share one family run: the same program structure (see
/// [`same_structure`]) AND the same conjunct kind.
pub(crate) fn same_family(a: &FamClause, b: &FamClause) -> bool {
    std::mem::discriminant(&a.combine) == std::mem::discriminant(&b.combine)
        && same_structure(&a.prog, &b.prog)
}

/// Most domains one clause of a family may consume before it is given up: a
/// clause that cannot close must not crowd the shared waves.
const FAMILY_CLAUSE_DOMAINS: u64 = 200_000;

/// Branch and bound over many clauses of one family at once: every wave mixes
/// the open leaves of all undecided clauses, bounded together by
/// [`bound_batch_family`]. Each clause gets its own outcome, with the same
/// semantics as [`prove_conjunction_core`] (proofs only; a candidate, the
/// resolution floor or the per-clause domain cap gives that clause up).
pub(crate) fn prove_family_core(
    mut clauses: Vec<FamClause>,
    deadline: Instant,
    cfg: &LowDimConfig,
) -> Vec<LowDimOutcome> {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::LowDim);
    let t0 = Instant::now();
    let n = clauses.len();
    if n == 0 {
        return Vec::new();
    }
    for c in &mut clauses {
        c.prog.refine_max = cfg.refine_max_width;
        c.prog.joint = cfg.joint_quadratic;
    }
    let mut out: Vec<Option<LowDimOutcome>> = vec![None; n];
    let (d, width) = {
        let b = &clauses[0].prog;
        (b.d, b.sizes[b.seed])
    };
    for (c, cl) in clauses.iter().enumerate() {
        let kind_ok = match &cl.combine {
            Combine::Each => true,
            Combine::SigmoidSum { centre } => {
                centre.len() == cl.thrs.len()
                    && centre.iter().all(|(t, s)| t.is_finite() && s.is_finite())
            }
        };
        let ok = cl.lo.len() == d
            && cl.hi.len() == d
            && same_family(&clauses[0], cl)
            && kind_ok
            && cl.lo.iter().chain(&cl.hi).all(|v| v.is_finite())
            && cl.lo.iter().zip(&cl.hi).all(|(l, h)| l <= h)
            && !cl.objs.is_empty()
            && cl.objs.len() == cl.per_conjunct() * cl.thrs.len()
            && cl
                .objs
                .iter()
                .all(|o| o.len() == width && o.iter().all(|v| v.is_finite()))
            && cl.thrs.iter().all(|t| t.is_finite());
        if !ok {
            out[c] = Some(LowDimOutcome::Declined("malformed family clause".into()));
        }
    }
    let progs: Vec<&Program> = clauses.iter().map(|c| &c.prog).collect();
    let batch = cfg.batch.max(1);
    let per_domain_bytes = progs[0]
        .total
        .saturating_mul(d + 1)
        .saturating_mul(16)
        .max(1);
    let chunk = cfg
        .chunk
        .max(1)
        .min((CHUNK_BYTES / per_domain_bytes).max(1));
    let mut stats = vec![
        LowDimStats {
            threads: lane_threads(cfg),
            ..LowDimStats::default()
        };
        n
    ];
    // #lsnc-lane stage 3: each clause's root widths (the split rules' aspect
    // guard; a malformed clause was declined above and is never split).
    let w0s: Vec<Vec<f64>> = clauses
        .iter()
        .map(|cl| cl.lo.iter().zip(&cl.hi).map(|(l, h)| h - l).collect())
        .collect();
    let mut open = vec![0u64; n];
    // #lowdim-fold-batch observable: kernel time vs everything else.
    let (mut bound_secs, mut waves_run, mut domains_bounded) = (0.0f64, 0u64, 0u64);
    let (mut q_lo, mut q_hi) = (Vec::<f64>::new(), Vec::<f64>::new());
    let (mut q_depth, mut q_clause) = (Vec::<u32>::new(), Vec::<u32>::new());
    for c in 0..n {
        if out[c].is_none() {
            q_lo.extend_from_slice(&clauses[c].lo);
            q_hi.extend_from_slice(&clauses[c].hi);
            q_depth.push(0);
            q_clause.push(c as u32);
            open[c] = 1;
        }
    }
    let give_up = |out: &mut Vec<Option<LowDimOutcome>>,
                   stats: &[LowDimStats],
                   c: usize,
                   stop: LowDimStop| {
        if out[c].is_none() {
            let mut st = stats[c];
            st.secs = t0.elapsed().as_secs_f64();
            out[c] = Some(LowDimOutcome::Unproved(st, stop));
        }
    };
    loop {
        let queued = q_depth.len();
        if queued == 0 {
            break;
        }
        if Instant::now() >= deadline || queued > cfg.max_queue {
            let stop = if queued > cfg.max_queue {
                LowDimStop::Queue
            } else {
                LowDimStop::Deadline
            };
            for c in 0..n {
                give_up(&mut out, &stats, c, stop);
            }
            break;
        }
        let take = batch.min(queued);
        let at = queued - take;
        let b_lo = q_lo.split_off(at * d);
        let b_hi = q_hi.split_off(at * d);
        let b_depth = q_depth.split_off(at);
        let b_clause = q_clause.split_off(at);
        let live: Vec<usize> = (0..take)
            .filter(|&i| out[b_clause[i] as usize].is_none())
            .collect();
        if live.is_empty() {
            continue;
        }
        let mut l_lo = Vec::with_capacity(live.len() * d);
        let mut l_hi = Vec::with_capacity(live.len() * d);
        let mut l_which = Vec::with_capacity(live.len());
        let mut l_objs: Vec<&[Vec<f64>]> = Vec::with_capacity(live.len());
        for &i in &live {
            let c = b_clause[i] as usize;
            l_lo.extend_from_slice(&b_lo[i * d..(i + 1) * d]);
            l_hi.extend_from_slice(&b_hi[i * d..(i + 1) * d]);
            l_which.push(c);
            l_objs.push(&clauses[c].objs);
        }
        let t_bound = Instant::now();
        let bounds = bound_batch_family(&progs, &l_which, &l_lo, &l_hi, &l_objs, chunk);
        bound_secs += t_bound.elapsed().as_secs_f64();
        waves_run += 1;
        domains_bounded += live.len() as u64;
        if bounds.len() != live.len() {
            // Every dequeued domain must come back bounded.
            for c in 0..n {
                give_up(&mut out, &stats, c, LowDimStop::NonFinite);
            }
            break;
        }
        // #lowdim-fold-parcand: the violation-candidate test is a pure scalar
        // walk of the clause's program at the domain centre, and it dominated
        // the family loop (measured 10.8 s of 12.4 s for 3962 domains, kernel
        // 1.7 s). Evaluate it for every open, not-yet-closed domain of the
        // wave in parallel; the sequential loop below consumes the flag at
        // exactly the position the inline test used to run, so outcomes are
        // unchanged.
        let cand_flags: Vec<bool> = live
            .par_iter()
            .enumerate()
            .map(|(k, &i)| {
                let cl = &clauses[b_clause[i] as usize];
                let Some((lbs, _)) = bounds.get(k) else {
                    return false;
                };
                if lbs.len() != cl.objs.len() || lbs.iter().any(|lb| lb.is_nan()) || cl.closes(lbs)
                {
                    return false;
                }
                let (l, h) = (&b_lo[i * d..(i + 1) * d], &b_hi[i * d..(i + 1) * d]);
                let centre: Vec<f64> = l.iter().zip(h).map(|(a, b)| 0.5 * (a + b)).collect();
                cl.is_candidate(&centre)
            })
            .collect();
        for (k, &i) in live.iter().enumerate() {
            let c = b_clause[i] as usize;
            if out[c].is_some() {
                continue;
            }
            stats[c].domains += 1;
            let (lbs, grads) = &bounds[k];
            let cl = &clauses[c];
            if lbs.len() != cl.objs.len() || lbs.iter().any(|lb| lb.is_nan()) {
                give_up(&mut out, &stats, c, LowDimStop::NonFinite);
                continue;
            }
            if cl.closes(lbs) {
                stats[c].closed += 1;
                open[c] -= 1;
                if open[c] == 0 {
                    let mut st = stats[c];
                    st.secs = t0.elapsed().as_secs_f64();
                    out[c] = Some(LowDimOutcome::Verified(st));
                }
                continue;
            }
            if stats[c].domains > FAMILY_CLAUSE_DOMAINS {
                give_up(&mut out, &stats, c, LowDimStop::Queue);
                continue;
            }
            let (l, h) = (&b_lo[i * d..(i + 1) * d], &b_hi[i * d..(i + 1) * d]);
            // The split below bisects at the centre; only the candidate TEST
            // moved to the parallel pre-pass.
            let centre: Vec<f64> = l.iter().zip(h).map(|(a, b)| 0.5 * (a + b)).collect();
            if cand_flags[k] {
                give_up(&mut out, &stats, c, LowDimStop::Candidate);
                continue;
            }
            // Split along the objective closest to closing (for a single-objective
            // conjunct: largest lb - thr, exactly as before `#lowdim-fold-sigdiff`).
            // #lsnc-lane stage 3: the rule is shared with the single-clause
            // lane (`search::choose_split`; Widest / Gradient are the
            // historical loops, bit for bit).
            let nearest = cl.split_objective(lbs);
            let best = search::choose_split(cfg.split, l, h, &w0s[c], grads, nearest);
            let mid = centre[best];
            if h[best] - l[best] < cfg.min_width || !(mid > l[best] && mid < h[best]) {
                give_up(&mut out, &stats, c, LowDimStop::Resolution);
                continue;
            }
            let depth = b_depth[i] + 1;
            stats[c].max_depth = stats[c].max_depth.max(depth);
            q_lo.extend_from_slice(l);
            q_hi.extend_from_slice(h);
            let li = q_hi.len() - d + best;
            q_hi[li] = mid;
            q_lo.extend_from_slice(l);
            q_hi.extend_from_slice(h);
            let ri = q_lo.len() - d + best;
            q_lo[ri] = mid;
            q_depth.push(depth);
            q_depth.push(depth);
            q_clause.push(c as u32);
            q_clause.push(c as u32);
            open[c] += 1;
        }
    }
    // Families of one clause are the single-clause path; only report batching,
    // and only the first few families per process (a row can hold hundreds).
    static FAMILY_CORE_LINES: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    if n > 1 && FAMILY_CORE_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
        eprintln!(
            "[lowdim-family-core] clauses={n} waves={waves_run} domains={domains_bounded} \
bound_secs={bound_secs:.3} total_secs={:.3} batch={batch} chunk={chunk}",
            t0.elapsed().as_secs_f64()
        );
    }
    out.into_iter()
        .zip(stats)
        .map(|(o, mut st)| {
            o.unwrap_or_else(|| {
                st.secs = t0.elapsed().as_secs_f64();
                LowDimOutcome::Unproved(st, LowDimStop::Deadline)
            })
        })
        .collect()
}

/// One always-on telemetry line per lane run (no verdict reads it back).
///
/// #lsnc-lane stage 3 appends the search counters: `clip_closed=` (domains
/// the relaxed clip emptied), `clip_shrunk=`, `joint_sites=` (sites the armed
/// joint bound serves), `joint_hits=` ((domain, site) pairs it bounded) and
/// `threads=`. An armed clip with `clip_closed=0 clip_shrunk=0`, or an armed
/// joint with `joint_hits=0`, is VOID, not negative.
pub fn emit_line(clause: usize, outcome: &LowDimOutcome) {
    match outcome {
        LowDimOutcome::Verified(s) => eprintln!(
            "[lowdim] clause={clause} verdict=verified domains={} closed={} depth={} waves={} secs={:.3} \
             clip_closed={} clip_shrunk={} joint_sites={} joint_hits={} threads={} vol={:.6}",
            s.domains,
            s.closed,
            s.max_depth,
            s.waves,
            s.secs,
            s.clip_closed,
            s.clip_shrunk,
            s.joint_sites,
            s.joint_hits,
            s.threads,
            s.closed_volume
        ),
        LowDimOutcome::Unproved(s, stop) => eprintln!(
            "[lowdim] clause={clause} verdict=unproved stop={} domains={} closed={} depth={} waves={} secs={:.3} \
             clip_closed={} clip_shrunk={} joint_sites={} joint_hits={} threads={} vol={:.6}",
            stop.tag(),
            s.domains,
            s.closed,
            s.max_depth,
            s.waves,
            s.secs,
            s.clip_closed,
            s.clip_shrunk,
            s.joint_sites,
            s.joint_hits,
            s.threads,
            s.closed_volume
        ),
        LowDimOutcome::Declined(reason) => {
            eprintln!("[lowdim] clause={clause} verdict=declined reason={reason}");
        }
    }
}

mod alpha;
mod attack;
mod bbe;
mod disjunction;
mod fold;
mod joint;
mod pinned;
mod search;
mod shared;
mod sigmoid;
pub use bbe::{
    forward_linear_search_backend_registered, set_forward_linear_search_backend_factory,
    take_last_bbe_summary, BbeStats, RouterRow, ROUTER_ROWS,
};
pub use disjunction::{
    disjunction_admits, prove_disjunction_admitted, prove_disjunction_admitted_with,
    take_last_disjunction_summary, DisjunctionClause, DisjunctionOutcome, DisjunctionStats,
};
pub use fold::{prove_clause_folded, prove_clauses_folded, FoldCache, FoldConjunct};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod lsnc_tests;

#[cfg(test)]
mod search_tests;

#[cfg(test)]
mod bundle_tests;

#[cfg(test)]
mod bbe_tests;

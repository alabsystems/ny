//! `#auto-alloc` — choose the margin-row ROOT phase parameters per instance,
//! from the loaded model, instead of from a category-wide constant.
//!
//! # The defect this closes
//!
//! `configs/vnncomp25/cifar100_2024.yaml` serves TWO models whose root work
//! differs ~4x (`resnet_medium` vs `resnet_large`), and they fail differently:
//! large STARVES, medium EXPLODES. Every root-phase constant in that file was
//! calibrated on ONE of them and then shipped to both. Four are on record:
//!
//! | constant | calibrated on | measured cost on the other half |
//! |---|---|---|
//! | `bi_secs: 40` | medium | 36-42s vs 8s |
//! | `closure_warmup: 24` | medium | large expands 20-38, so the ladder never observes |
//! | `k_head`/`k_trunk: 16` | medium | cited rows are all medium |
//! | `interm_alpha: 2` | medium (`idx_5242`) | 19.31s of a 38.80s phase on large (68%) |
//!
//! Each was found by a human noticing, months apart. The mechanism below is the
//! general fix: the numbers are DERIVED from the instance, so a model that costs
//! 4x does not silently inherit the cheap model's budget.
//!
//! # SCOPE: this prices the MARGIN-ROW root, and deliberately nothing else
//!
//! The proxy sums trunk-ReLU work in the margin-row root build. It does NOT
//! price the GRAPH lane, and a future reader should not "fix" that.
//!
//! The temptation is concrete. tinyimagenet spends ~35s of its 95s internal
//! budget on `#conv-patches-collect`, a CROWN-IBP DAG collection that runs only
//! there (its gate is `n_in > 5000`; tinyimagenet is 9408, cifar100 3072), and
//! which finishes 4-5 crown nodes of ~40. That looks exactly like the pre-tree
//! overrun this module exists to shed, so the allocator "obviously" ought to
//! see it. It ought not:
//!
//! * it is in the GRAPH lane, concurrent with the margin-row worker, not in the
//!   margin-row root build this module's parameters govern; and
//! * freeing it was MEASURED and converts nothing — 3/3 A/B pairs null, because
//!   the graph BaB explores 0-1 domains in 98s in BOTH arms. Reproduced
//!   independently 2026-09-18 on an unsolved row: `[bab-stop] status=timeout
//!   explored=1 verified=0 deepest=0 elapsed_s=51.172`.
//!
//! tinyimagenet's unsolved rows are BOUND-QUALITY bound, not wall bound, which
//! is why `decide()` returning `already-affordable` there is the right answer
//! and not a miss. Its 63-row gap is entirely unsat proofs (its sat side is
//! tied with the winner at 38 AND set-identical), so there is no wall for this
//! module to buy back.
//!
//! # Why the TREE window is the quantity to allocate
//!
//! Measured 2026-09-17/18 on cifar100 resnet_large, official 100s: pre-tree work
//! is ~57s of a 97s lane, so the branch-and-bound tree gets only ~27-31s — and
//! every bound-tightening mechanism available costs MORE than that window (the
//! closure ladder quotes 137.5s uncapped; `#rebake` needs a build <= 14.3s
//! against 36s; ladder rung 2 quotes 20.0s against 27.1s left). Three failures
//! previously diagnosed as separate "pricing bugs" are that one fact.
//!
//! Two conversions were then won by REMOVING medium-calibrated root work:
//!
//! * `bi_topk` 1024 -> 128 frees ~13s   -> `idx_9372_sidx_3515` timeout -> certified unsat
//!   (10/10 across two binaries, control 0/3, moat 80/80, ratchets green)
//! * `interm_alpha` 2 -> 0 frees ~7.44s -> `idx_1280_sidx_5310` timeout -> certified unsat
//!   (arm 4/4 vs control 0/4 interleaved, four independent tools agree)
//!
//! NEITHER improves bound quality. They convert by making an already-computable
//! proof LAND INSIDE the budget: walls move from 97-98s (never finishing) to
//! 88-96s. So the allocator's objective is not "a better bound" — it is "leave
//! the tree enough wall to finish the proof the root already enables".
//!
//! # The cost proxy, and why it is not a curve fit
//!
//! The backward-intermediate phase runs two `run_prefix` passes per column
//! chunk, at most `topk` columns per trunk ReLU, and one pass costs about
//! `cols x (conv MACs in the frozen prefix)`. So the work is
//!
//! ```text
//! proxy = SUM over trunk ReLUs of  ceil(min(topk, width) / chunk) x prefix_MACs
//! ```
//!
//! This weights a layer's columns by the cost of the prefix that must be
//! re-run for them, which is what makes it predictive rather than descriptive:
//!
//! * cifar100 resnet_large — column-heavy layers sit LATE, behind an expensive
//!   prefix, so capping `topk` at 128 removes real work: 35.8s -> 21.2s (-41%).
//! * tinyimagenet resnet_medium — its column-heavy layers are EARLY (layer 1 has
//!   603 columns on a short prefix) while its expensive late layers hold 4 and 88
//!   columns, already under any sane cap. Predicted saving: small. MEASURED
//!   2026-09-18 on three paired rows: 17.23->16.14, 16.90->15.90, 17.76->16.29,
//!   i.e. **-1.2s median (6-8%)** against cifar's 41%.
//!
//! That asymmetry was PREDICTED by this proxy and then measured, on a category
//! the proxy was not calibrated on. A column census alone would have got it
//! wrong: it tells you whether a cap BINDS (tiny's layer 1 is 603 > 128, so it
//! binds) and not whether it SAVES.
//!
//! # What this can and cannot do to a verdict
//!
//! Both parameters are bounded-loss, not soundness-bearing:
//!
//! * `interm_alpha` intersects SHRINK-ONLY, so any dose yields a subset of the
//!   admissible region — a wrong dose costs wall clock, never validity.
//! * `bi_topk` narrows which columns get a tightening pass. Untightened columns
//!   keep their prior (valid) bound, so the root bound can only be LOOSER, never
//!   unsound (measured: 0.006 looser at 128).
//!
//! and the lane they feed is fail-closed — `run_margin_row_lane` returns `Unsat`
//! or `Unknown` and never `Sat` — with every published proof carrying its
//! certificate ledger. So a mis-allocation can lose a proof. It cannot invent
//! one. That is why this ships behind a moat and not behind an argument.
//!
//! # Degradation
//!
//! Absent a usable spec, a non-finite budget, or a proxy of zero, the decision
//! is `applied: false` and every field is returned UNCHANGED — byte-identical to
//! the preset. The allocator only ever REMOVES optional root work; it never
//! raises a value above what the preset asked for, so it cannot spend budget the
//! category did not already authorise.

use super::spec::{TwinOpSpec, TwinSpec};

/// Seconds of backward-intermediate phase per unit of work proxy, at
/// `interm_alpha = 0`, where one unit is `1 chunk x 1 GMAC of frozen prefix`.
///
/// ⚠ CALIBRATED ON DOSE-0 DATA. The first version of this constant mixed doses:
/// cifar100's 6.8s and 35.8s are `interm_alpha: 2` measurements while
/// tinyimagenet's 17.0s is dose 0 (that category never arms the key). Treating
/// all three as comparable produced 3.7 and a 2.3x overestimate of the phase —
/// `est_bi_s=80.8` against 35.8s measured — which then silently compensated for
/// the missing root overhead below. Two errors cancelling is not a model.
///
/// Dose-0 anchors, from tonight's paired runs:
///
/// | model | proxy | measured phase | s/unit |
/// |---|---|---|---|
/// | cifar100 resnet_large | 7.48 | 21.8s (35.8 / 1.64) | 2.92 |
/// | tinyimagenet resnet_medium | 5.65 | 17.0s (dose 0 as shipped) | 3.01 |
/// | cifar100 resnet_medium | 1.84 | 4.1s (6.8 / 1.64) | 2.24 |
pub const SECS_PER_PROXY_UNIT: f64 = 2.9;

/// Fraction of the lane budget the tree must be left, or the allocator sheds
/// optional root work to get there.
///
/// JUSTIFIED BY THE TWO CONVERSIONS, not chosen: at the shipped constants
/// resnet_large leaves the tree ~27-31s of a ~97s lane (0.28-0.32) and converts
/// nothing; after shedding, the same rows run a ~50s tree (0.52) and two rows
/// cross. 0.45 sits between those and above every measured starving
/// configuration.
pub const TREE_WINDOW_TARGET_FRAC: f64 = 0.45;

/// Narrowed `topk` used when the root has to give wall back to the tree.
///
/// MEASURED: 1024 -> 128 costs 0.006 of root bound and returns 14.7s (35.8 ->
/// 21.2s) on resnet_large, deterministic over 16 runs.
pub const NARROW_TOPK: usize = 128;

/// Multiplier on the backward-intermediate phase per unit of `interm_alpha`
/// dose.
///
/// MEASURED tonight, paired on resnet_large at `topk = 128`: dose 2 costs
/// 21.7s where dose 0 costs 13.2s, so dose 2 is 1.64x — i.e. 0.32 per unit.
/// The earlier 0.96 came from reading "interm_alpha is 68% of the phase" as a
/// multiplier rather than a share.
pub const DOSE_FACTOR_PER_UNIT: f64 = 0.32;

/// Root work that is NOT the backward-intermediate phase, added to the estimate
/// so the tree window is computed against the whole pre-tree cost.
///
/// MEASURED on resnet_large: pre-tree is ~57s of a ~97s lane while the BI phase
/// is ~35.8s, leaving ~21s. That remainder is dominated by the root alpha
/// ascent, which `alpha_secs` CAPS — so it is DERIVED from the resolved preset
/// (`alpha_secs + this`) rather than being a fifth hand-calibrated constant.
/// Omitting it entirely is what the first version did, and it is why the phase
/// overestimate looked harmless.
pub const ROOT_FIXED_OVERHEAD_SECS: f64 = 1.0;

/// `alpha_secs` a category gets when its yaml authors none.
pub const DEFAULT_ALPHA_SECS: f64 = 20.0;

/// The `bi_topk` a category gets when its yaml authors none, mirrored from
/// `BackwardIntermRuntimePolicy::default()`. Kept here so the allocator's view
/// of "what the preset asked for" cannot drift from the policy's own default
/// without this constant becoming visibly wrong.
pub const DEFAULT_TOPK: usize = 1024;

/// The `bi_chunk` a category gets when its yaml authors none, mirrored from
/// `DEFAULT_BACKWARD_INTERM_CHUNK`. The chunk size is what turns a column count
/// into a pass count, so the proxy is meaningless without the right one.
pub const DEFAULT_CHUNK: usize = super::DEFAULT_BACKWARD_INTERM_CHUNK;

/// What the allocator decided, and everything needed to audit the decision from
/// one log line.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoAllocDecision {
    /// Did any field actually change?
    pub applied: bool,
    /// Why not, when `applied` is false.
    pub declined: Option<&'static str>,
    /// Work proxy at the INCOMING parameters, in chunk-GMAC units.
    pub proxy_in: f64,
    /// Work proxy at the OUTGOING parameters.
    pub proxy_out: f64,
    /// Estimated backward-intermediate seconds before / after.
    pub est_secs_in: f64,
    pub est_secs_out: f64,
    /// Lane budget the decision was made against.
    pub lane_budget_secs: f64,
    /// Tree window implied before / after, in seconds.
    pub tree_in: f64,
    pub tree_out: f64,
    /// Seconds the tree needed to clear the target.
    pub tree_target: f64,
    /// `(interm_alpha, topk)` before and after.
    pub before: (usize, usize),
    pub after: (usize, usize),
    /// Ordered list of what was shed, cheapest-value-first.
    pub shed: Vec<&'static str>,
    /// How the phase was priced: `"dense"` (the calibrated dense-prefix
    /// proxy) or `"patch"` (`#bi-patch` armed: each layer at its patch MACs).
    pub pricing: &'static str,
}

impl AutoAllocDecision {
    /// The declined case, returned unchanged.
    fn declined(
        why: &'static str,
        budget: f64,
        before: (usize, usize),
        pricing: &'static str,
    ) -> Self {
        Self {
            applied: false,
            declined: Some(why),
            proxy_in: 0.0,
            proxy_out: 0.0,
            est_secs_in: 0.0,
            est_secs_out: 0.0,
            lane_budget_secs: budget,
            tree_in: 0.0,
            tree_out: 0.0,
            tree_target: 0.0,
            before,
            after: before,
            shed: Vec::new(),
            pricing,
        }
    }

    /// One line, at default verbosity, carrying the decision AND its inputs.
    ///
    /// A silent allocator would be indistinguishable from an absent one, which
    /// is the failure mode this campaign calls VOID-not-negative: several
    /// mechanisms have read as armed while doing nothing. Every field a reader
    /// needs to re-derive the decision by hand is here.
    pub fn line(&self) -> String {
        // `#bi-patch` pricing announces itself; the dense line is unchanged.
        let pricing = if self.pricing == "patch" {
            " pricing=patch"
        } else {
            ""
        };
        if !self.applied {
            return format!(
                "[auto-alloc] applied=0 reason={} budget_s={:.1} interm_alpha={} topk={} \
                 (unchanged; byte-identical to the preset){pricing}",
                self.declined.unwrap_or("none"),
                self.lane_budget_secs,
                self.before.0,
                self.before.1,
            );
        }
        format!(
            "[auto-alloc] applied=1 proxy={:.2}->{:.2} est_bi_s={:.1}->{:.1} \
             tree_s={:.1}->{:.1} target_s={:.1} budget_s={:.1} \
             interm_alpha={}->{} topk={}->{} shed={}{pricing}",
            self.proxy_in,
            self.proxy_out,
            self.est_secs_in,
            self.est_secs_out,
            self.tree_in,
            self.tree_out,
            self.tree_target,
            self.lane_budget_secs,
            self.before.0,
            self.after.0,
            self.before.1,
            self.after.1,
            if self.shed.is_empty() {
                "none".to_string()
            } else {
                self.shed.join("+")
            },
        )
    }
}

/// MACs of a single op, counted as multiply-accumulates against the frozen
/// prefix. Only ops that dominate a `run_prefix` pass are counted; elementwise
/// ops are rounding error against a conv and are deliberately ignored.
fn op_macs(op: &TwinOpSpec) -> f64 {
    match op {
        TwinOpSpec::Conv { kernel, oshape, .. }
        | TwinOpSpec::ConvTranspose { kernel, oshape, .. } => {
            let (_cout, cin, kh, kw) = *kernel;
            let (oc, oh, ow) = *oshape;
            (oc as f64) * (oh as f64) * (ow as f64) * (cin as f64) * (kh as f64) * (kw as f64)
        }
        TwinOpSpec::Gemm { weight, .. } => weight.len() as f64,
        _ => 0.0,
    }
}

/// Flat output size of every tensor, indexed by TENSOR ID.
///
/// `spec.rs:14` fixes the id scheme: tensor 0 is the network input and the op at
/// index `k` produces tensor `k + 1`. An earlier version read `ops[input]`
/// directly — off by one, and blind through `Add`/`ChannelAffine`. On a real
/// ResNet it found no trunk ReLU at all and the allocator declined with
/// `reason=no-proxy`; the synthetic fixture hid it by making tensor ids
/// coincide with op indices. This mirrors the `tsize` walk the real consumer
/// does in `net.rs:255-330`, so the two cannot drift without that consumer
/// breaking too.
fn tensor_sizes(spec: &TwinSpec) -> Vec<usize> {
    let mut t = Vec::with_capacity(spec.ops.len() + 1);
    t.push(spec.n_in);
    for op in &spec.ops {
        let sz = match op {
            TwinOpSpec::Conv { oshape, .. } | TwinOpSpec::ConvTranspose { oshape, .. } => {
                let (c, h, w) = *oshape;
                c * h * w
            }
            TwinOpSpec::ChannelAffine { shape, .. } => {
                let (c, h, w) = *shape;
                c * h * w
            }
            TwinOpSpec::Relu { input } => t.get(*input).copied().unwrap_or(0),
            TwinOpSpec::Add { lhs, .. } => t.get(*lhs).copied().unwrap_or(0),
            TwinOpSpec::Flatten { input } => t.get(*input).copied().unwrap_or(0),
            TwinOpSpec::Gemm { shape, .. } => shape.0,
        };
        t.push(sz);
    }
    t
}

/// `SUM over trunk ReLUs of ceil(min(topk, width)/chunk) x prefix GMACs`.
///
/// Returns `None` when the spec yields no usable trunk ReLU, which is the
/// degradation path: the caller then leaves the preset untouched.
pub fn root_work_proxy(spec: &TwinSpec, topk: usize, chunk: usize) -> Option<f64> {
    if chunk == 0 || topk == 0 {
        return None;
    }
    let sizes = tensor_sizes(spec);
    let mut prefix_macs = 0.0f64;
    let mut proxy = 0.0f64;
    let mut seen_relu = false;
    let mut seen_gemm = false;
    for op in &spec.ops {
        match op {
            // TRUNK is what the real consumer calls trunk: a ReLU seen BEFORE
            // any Gemm (`net.rs:318`, `if gemm_idx.is_empty()`). The earlier
            // "every ReLU but the last" rule is not the same set on a net whose
            // head holds more than one ReLU, and the head frame is priced by the
            // head path, not here.
            TwinOpSpec::Relu { input } if !seen_gemm => {
                let width = sizes.get(*input).copied().unwrap_or(0);
                if width > 0 {
                    let cols = width.min(topk);
                    proxy += (cols.div_ceil(chunk) as f64) * (prefix_macs / 1.0e9);
                    seen_relu = true;
                }
            }
            TwinOpSpec::Gemm { .. } => seen_gemm = true,
            _ => {}
        }
        prefix_macs += op_macs(op);
    }
    (seen_relu && proxy > 0.0).then_some(proxy)
}

/// `#bi-patch`: the same proxy with every trunk ReLU's prefix priced at its
/// PATCH-mode signed MACs per column — the planner's own cost model
/// (`patch::Geometry`, built from the spec's geometry exactly as the phase
/// builds it from the compiled net), so the allocator and the phase can never
/// price different frames. A layer the planner refuses is priced at its dense
/// prefix (the phase then runs the crop or the dense pair there).
///
/// WHY. With patch mode armed, pricing the phase as dense is the measured
/// failure the design names: on cct idx1031 the dense proxy estimated the
/// phase at `est_bi_s=50.2` against `work_secs=7.23` measured and shed
/// `interm_alpha` 2 -> 0, the inferred cause of that row's root regression.
/// The per-second calibration ([`SECS_PER_PROXY_UNIT`]) is the dense one:
/// an unmeasured assumption that a patch MAC costs what a dense MAC costs.
pub fn root_work_proxy_patch(
    spec: &TwinSpec,
    topk: usize,
    chunk: usize,
    ratio: f64,
) -> Option<f64> {
    if chunk == 0 || topk == 0 {
        return None;
    }
    let sizes = tensor_sizes(spec);
    let geo = super::patch::geo_of_spec(spec);
    let mut prefix_macs = 0.0f64;
    let mut proxy = 0.0f64;
    let mut seen_relu = false;
    let mut seen_gemm = false;
    for (k, op) in spec.ops.iter().enumerate() {
        match op {
            TwinOpSpec::Relu { input } if !seen_gemm => {
                let width = sizes.get(*input).copied().unwrap_or(0);
                if width > 0 {
                    let cols = width.min(topk);
                    let macs = super::patch::Geometry::build(&geo, &sizes, k, ratio)
                        .map_or(prefix_macs, |plan| plan.macs_col_patch as f64);
                    proxy += (cols.div_ceil(chunk) as f64) * (macs / 1.0e9);
                    seen_relu = true;
                }
            }
            TwinOpSpec::Gemm { .. } => seen_gemm = true,
            _ => {}
        }
        prefix_macs += op_macs(op);
    }
    (seen_relu && proxy > 0.0).then_some(proxy)
}

/// The crossover ratio `#auto-alloc` prices with: `Some` only when the preset
/// ARMS `margin_row.bi_patch`, clamped exactly as the phase's reader clamps
/// `margin_row.bi_patch_ratio` (absent or non-finite: the declaration
/// default). `None` keeps the dense pricing, byte-identical.
pub fn patch_pricing_ratio(armed: Option<bool>, ratio: Option<f64>) -> Option<f64> {
    (armed == Some(true)).then(|| {
        ratio
            .filter(|value| value.is_finite())
            .map_or(super::patch::DEFAULT_RATIO, |value| {
                value.clamp(super::patch::RATIO_MIN, super::patch::RATIO_MAX)
            })
    })
}

/// Estimated backward-intermediate seconds at a given proxy.
pub fn est_bi_secs(proxy: f64) -> f64 {
    proxy * SECS_PER_PROXY_UNIT
}

/// Decide the root parameters for THIS instance.
///
/// `interm_alpha` and `topk` are passed and returned rather than mutated in
/// place so the whole decision is a pure function of its inputs — it is unit
/// testable with no model, no clock and no box slot, which is what lets the
/// tests below pin the two measured models' behaviour directly.
///
/// SHEDDING ORDER is measured, not guessed. `interm_alpha` goes first: it is
/// 68% of the phase on resnet_large and the conversion it was armed for is a
/// resnet_MEDIUM row that HOLDS without it. `bi_topk` goes second: it costs
/// 0.006 of root bound, which is small but not nothing.
pub fn decide(
    spec: Option<&TwinSpec>,
    lane_budget_secs: f64,
    alpha_secs: f64,
    interm_alpha: usize,
    topk: usize,
    chunk: usize,
) -> AutoAllocDecision {
    decide_with_patch(
        spec,
        lane_budget_secs,
        alpha_secs,
        interm_alpha,
        topk,
        chunk,
        None,
    )
}

/// [`decide`] with the phase priced for `#bi-patch` when `patch_ratio` is
/// `Some` (see [`root_work_proxy_patch`] and [`patch_pricing_ratio`]);
/// `None` is [`decide`] exactly. The shedding rule, its order and its
/// "only ever REMOVE" contract are unchanged: only the work proxy differs.
pub fn decide_with_patch(
    spec: Option<&TwinSpec>,
    lane_budget_secs: f64,
    alpha_secs: f64,
    interm_alpha: usize,
    topk: usize,
    chunk: usize,
    patch_ratio: Option<f64>,
) -> AutoAllocDecision {
    let before = (interm_alpha, topk);
    let pricing = if patch_ratio.is_some() {
        "patch"
    } else {
        "dense"
    };
    let Some(spec) = spec else {
        return AutoAllocDecision::declined("no-spec", lane_budget_secs, before, pricing);
    };
    if !lane_budget_secs.is_finite() || lane_budget_secs <= 0.0 {
        return AutoAllocDecision::declined("no-budget", lane_budget_secs, before, pricing);
    }
    let work_proxy = |topk: usize| match patch_ratio {
        Some(ratio) => root_work_proxy_patch(spec, topk, chunk, ratio),
        None => root_work_proxy(spec, topk, chunk),
    };
    let Some(proxy_in) = work_proxy(topk) else {
        return AutoAllocDecision::declined("no-proxy", lane_budget_secs, before, pricing);
    };

    // Dose 0 is the pre-#interm-alpha phase; each unit of dose multiplies the
    // per-column work by a MEASURED 0.32 (dose 2 = 1.64x on resnet_large).
    let dose_factor = |dose: usize| 1.0 + DOSE_FACTOR_PER_UNIT * dose as f64;
    // The tree window is the budget minus ALL pre-tree work, not just the BI
    // phase. The remainder is dominated by the root alpha ascent, which
    // `alpha_secs` caps, so it is derived rather than assumed.
    let overhead = alpha_secs.max(0.0) + ROOT_FIXED_OVERHEAD_SECS;

    let est_in = est_bi_secs(proxy_in) * dose_factor(interm_alpha);
    let tree_target = lane_budget_secs * TREE_WINDOW_TARGET_FRAC;
    let tree_in = lane_budget_secs - est_in - overhead;

    let mut shed: Vec<&'static str> = Vec::new();
    let mut alpha_out = interm_alpha;
    let mut topk_out = topk;
    let mut proxy_out = proxy_in;
    let mut est_out = est_in;

    if tree_in < tree_target && alpha_out > 0 {
        alpha_out = 0;
        shed.push("interm_alpha");
        est_out = est_bi_secs(proxy_out) * dose_factor(alpha_out);
    }
    if lane_budget_secs - est_out - overhead < tree_target && topk_out > NARROW_TOPK {
        if let Some(p) = work_proxy(NARROW_TOPK) {
            topk_out = NARROW_TOPK;
            proxy_out = p;
            shed.push("bi_topk");
            est_out = est_bi_secs(proxy_out) * dose_factor(alpha_out);
        }
    }

    let after = (alpha_out, topk_out);
    AutoAllocDecision {
        applied: after != before,
        declined: (after == before).then_some("already-affordable"),
        proxy_in,
        proxy_out,
        est_secs_in: est_in,
        est_secs_out: est_out,
        lane_budget_secs,
        tree_in,
        tree_out: lane_budget_secs - est_out - overhead,
        tree_target,
        before,
        after,
        shed,
        pricing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spec whose trunk ReLUs sit behind a given prefix cost. `layers` is
    /// `(relu_width, conv_macs_before_it)`, so a test can place expensive
    /// prefixes early or late and reproduce either measured model's shape.
    ///
    /// TENSOR IDS FOLLOW THE REAL SCHEME (`spec.rs:14`): tensor 0 is the input
    /// and op `k` produces tensor `k + 1`. The first version of this helper
    /// numbered ReLU inputs by OP INDEX, which made every unit test pass over a
    /// width lookup that found nothing on a real ResNet. It also ends with a
    /// `Gemm`, because the consumer defines the trunk as "before any Gemm"
    /// (`net.rs:318`) — without a head the last ReLU would be charged as trunk.
    fn spec_with(layers: &[(usize, f64)]) -> TwinSpec {
        let mut ops: Vec<TwinOpSpec> = Vec::new();
        let mut cur = 0usize; // tensor id currently carrying the activation
        let mut last_width = 0usize;
        for (width, macs) in layers {
            // op_macs computes cout*oh*ow*cin*kh*kw and oshape makes
            // cout*oh*ow == width, so cin = macs/width costs exactly `macs`.
            let cin = ((*macs / (*width).max(1) as f64) as usize).max(1);
            ops.push(TwinOpSpec::Conv {
                input: cur,
                weight: Vec::new(),
                bias: Vec::new(),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                kernel: (1, cin, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (*width, 1, 1),
            });
            cur = ops.len(); // the conv just pushed produced tensor `ops.len()`
            ops.push(TwinOpSpec::Relu { input: cur });
            cur = ops.len();
            last_width = *width;
        }
        // A head, so the trunk is bounded the way the consumer bounds it.
        ops.push(TwinOpSpec::Flatten { input: cur });
        cur = ops.len();
        ops.push(TwinOpSpec::Gemm {
            input: cur,
            weight: Vec::new(),
            bias: Vec::new(),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (10, last_width),
        });
        TwinSpec { n_in: 3072, ops }
    }

    /// The helper must agree with the REAL consumer about which ReLUs are trunk.
    /// Guards the exact confusion that produced `reason=no-proxy` on a live row:
    /// unit tests green over a proxy that saw nothing.
    #[test]
    fn the_fixture_uses_real_tensor_ids_and_the_proxy_finds_its_trunk_relus() {
        let s = spec_with(&[(4096, 1.0e9), (2048, 2.0e9)]);
        let sizes = tensor_sizes(&s);
        // Every Relu's input id must resolve to the width we asked for.
        let widths: Vec<usize> = s
            .ops
            .iter()
            .filter_map(|o| match o {
                TwinOpSpec::Relu { input } => Some(sizes[*input]),
                _ => None,
            })
            .collect();
        assert_eq!(
            widths,
            vec![4096, 2048],
            "tensor ids do not resolve: {widths:?}"
        );
        assert!(
            root_work_proxy(&s, 1024, 256).is_some(),
            "the proxy must find trunk ReLUs in a spec built the real way"
        );
    }

    /// The estimate must land near the MEASURED phase times, not merely produce
    /// the right threshold answer.
    ///
    /// The first calibration got all three engagement probes right while
    /// estimating 80.8s for a phase measured at 35.8s — a 2.3x error that was
    /// cancelling an omitted overhead term. A threshold test cannot see that;
    /// this one can. Anchors are tonight's dose-0 measurements.
    #[test]
    fn the_phase_estimate_tracks_the_measured_phase_times() {
        // (proxy, measured dose-0 seconds) for the three models on record.
        for (proxy, measured) in [(7.48_f64, 21.8_f64), (5.65, 17.0), (1.84, 4.1)] {
            let est = est_bi_secs(proxy);
            let rel = (est - measured).abs() / measured;
            assert!(
                rel < 0.40,
                "estimate {est:.1}s is {:.0}% off the measured {measured:.1}s at proxy {proxy}",
                rel * 100.0
            );
        }
        // And the dose multiplier must reproduce the paired measurement:
        // resnet_large at topk 128 costs 21.7s at dose 2 vs 13.2s at dose 0.
        let dose2 = 1.0 + DOSE_FACTOR_PER_UNIT * 2.0;
        let measured_ratio = 21.7 / 13.2;
        assert!(
            (dose2 - measured_ratio).abs() < 0.15,
            "dose factor {dose2:.2} vs measured {measured_ratio:.2}"
        );
    }

    /// The cost model is calibrated on the OP-WALK clock, and this test is what
    /// stops the two clocks being swapped.
    ///
    /// `backward_interm.rs`'s module header warns that the `done:` line's
    /// `secs=` is TABLEAU-OP-WALK wall while `work_secs=` is the phase's own
    /// summed compute, and that reading one as the other already produced a
    /// banked, campaign-steering falsehood that had to be retracted. The
    /// ALLOCATOR wants the op-walk figure and not the other one: the tree
    /// cannot start until the whole interleaved walk finishes, so the op-walk
    /// wall is what the tree window is actually measured against. Substituting
    /// `work_secs` would roughly halve every estimate and stop the allocator
    /// shedding anything at all — a silent, total regression that no other
    /// test here would catch, because the shed decision would simply never
    /// fire.
    ///
    /// MEASURED over the allocator's own 80-row moat by pairing each run's
    /// `[auto-alloc] est_bi_s=` against its `[backward-interm] done:` line:
    ///
    /// | arm                              | n  | predicted | op-walk mean | work_secs mean |
    /// |----------------------------------|----|-----------|--------------|----------------|
    /// | resnet_large, dose 0, topk 1024  | 35 | 21.7      | 20.84 (+4.1%)| 10.51 (+107%)  |
    /// | resnet_medium, dose 2, topk 1024 | 27 | 8.75      | 8.49  (+3.1%)| 3.22  (+171%)  |
    ///
    /// The bias is +3-4% on both models: a slight OVERestimate of pre-tree
    /// cost, which sheds marginally early. That is the safe direction — an
    /// underestimate would leave the tree starved with no rung left to spend.
    #[test]
    fn the_cost_model_is_calibrated_on_the_op_walk_clock_not_phase_compute() {
        let dose = |d: usize| 1.0 + DOSE_FACTOR_PER_UNIT * d as f64;
        // (label, proxy, dose, measured op-walk mean, measured work_secs mean, n)
        let arms = [
            (
                "resnet_large dose0",
                7.48_f64,
                0usize,
                20.84_f64,
                10.51_f64,
                35,
            ),
            ("resnet_medium dose2", 1.84, 2, 8.49, 3.22, 27),
        ];
        for (label, proxy, d, op_walk, work_secs, n) in arms {
            let est = est_bi_secs(proxy) * dose(d);
            let rel_walk = (est - op_walk).abs() / op_walk;
            assert!(
                rel_walk < 0.12,
                "{label} (n={n}): estimate {est:.2}s is {:.1}% off the measured \
                 op-walk {op_walk:.2}s — the model has drifted from its calibration",
                rel_walk * 100.0
            );
            // The clock guard. If someone re-fits the constant against
            // `work_secs`, this is the assertion that fails.
            let rel_work = (est - work_secs).abs() / work_secs;
            assert!(
                rel_work > 0.50,
                "{label} (n={n}): estimate {est:.2}s now tracks the PHASE-COMPUTE \
                 clock ({work_secs:.2}s) rather than the op-walk clock \
                 ({op_walk:.2}s). Re-read backward_interm.rs's TWO CLOCKS header: \
                 the allocator must price the whole walk the tree waits on."
            );
        }
    }

    #[test]
    fn the_mirrored_defaults_match_the_policy_defaults() {
        let p = super::super::BackwardIntermRuntimePolicy::default();
        assert_eq!(DEFAULT_TOPK, p.topk, "DEFAULT_TOPK drifted from the policy");
        assert_eq!(
            DEFAULT_CHUNK, p.chunk,
            "DEFAULT_CHUNK drifted from the policy"
        );
    }

    #[test]
    fn a_missing_spec_declines_and_changes_nothing() {
        let d = decide(None, 97.0, DEFAULT_ALPHA_SECS, 2, 1024, 256);
        assert!(!d.applied);
        assert_eq!(d.declined, Some("no-spec"));
        assert_eq!(d.before, d.after, "the preset must survive untouched");
        assert!(d.line().contains("applied=0"));
    }

    #[test]
    fn a_non_finite_budget_declines() {
        let s = spec_with(&[(4096, 1.0e9)]);
        for budget in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let d = decide(Some(&s), budget, DEFAULT_ALPHA_SECS, 2, 1024, 256);
            assert!(!d.applied, "budget {budget} must decline");
            assert_eq!(d.before, d.after);
        }
    }

    /// The cheap model keeps everything: its root already leaves the tree more
    /// than the target, so the allocator must not touch a conversion-bearing
    /// dose it did not need to shed.
    #[test]
    fn a_cheap_model_is_left_exactly_as_the_preset_asked() {
        // ~1.9 proxy units -> ~7s at dose 0, ~21s at dose 2; tree 76s of 97s.
        let s = spec_with(&[(512, 0.2e9), (512, 0.3e9)]);
        let d = decide(Some(&s), 97.0, DEFAULT_ALPHA_SECS, 2, 1024, 256);
        assert!(!d.applied, "cheap model must not be altered: {}", d.line());
        assert_eq!(d.after, (2, 1024));
        assert!(d.tree_in > d.tree_target);
    }

    /// The expensive model sheds `interm_alpha` FIRST — the measured order.
    #[test]
    fn an_expensive_model_sheds_interm_alpha_before_topk() {
        // Late, expensive prefix: exactly resnet_large's shape.
        let s = spec_with(&[(4096, 2.0e9), (4096, 6.0e9), (4096, 10.0e9)]);
        let d = decide(Some(&s), 97.0, DEFAULT_ALPHA_SECS, 2, 1024, 256);
        assert!(d.applied, "{}", d.line());
        assert_eq!(d.after.0, 0, "interm_alpha is shed first");
        assert_eq!(d.shed.first(), Some(&"interm_alpha"));
        assert!(d.est_secs_out < d.est_secs_in);
        assert!(d.tree_out > d.tree_in);
    }

    /// Shedding stops at the first rung that suffices.
    ///
    /// `bi_topk` costs real bound (0.006), so it must only be spent when
    /// dropping `interm_alpha` alone cannot buy the window. Stated as TWO
    /// deterministic cases rather than one conditional, because the first
    /// version of this test asked whether the target was met AFTER shedding —
    /// which is what shedding is FOR, so it fired on the success path and could
    /// also pass vacuously when the guard was false.
    ///
    /// Budget 97s, target 43.65s, so the root must fit in 53.35s.
    /// `dose_factor(2) = 2.92`, and `est = proxy x 3.7 x factor`:
    /// Budget 97s, overhead 21s (alpha_secs 20 + 1), target 43.65s, so the BI
    /// phase must fit in 32.35s. `dose_factor(2) = 1.64`, `s/unit = 2.9`:
    ///   proxy 9.0 -> 42.8s at dose 2 (sheds), 26.1s at dose 0 (fits) => alpha alone
    ///   proxy 16  -> 76.1s at dose 2, 46.4s at dose 0 (still over)   => needs topk too
    #[test]
    fn interm_alpha_alone_is_spent_when_it_suffices() {
        let s = spec_with(&[(4096, 0.5e9), (4096, 1.25e9)]); // proxy 9.0
        let d = decide(Some(&s), 97.0, DEFAULT_ALPHA_SECS, 2, 1024, 256);
        assert!(d.applied, "this model must need shedding: {}", d.line());
        assert_eq!(d.shed, vec!["interm_alpha"], "{}", d.line());
        assert_eq!(d.after.1, 1024, "topk must be untouched: {}", d.line());
        assert!(d.tree_out >= d.tree_target, "{}", d.line());
    }

    #[test]
    fn topk_is_spent_only_when_interm_alpha_alone_is_not_enough() {
        let s = spec_with(&[(4096, 1.0e9), (4096, 2.0e9)]);
        let d = decide(Some(&s), 97.0, DEFAULT_ALPHA_SECS, 2, 1024, 256);
        assert!(d.applied, "{}", d.line());
        assert_eq!(d.shed, vec!["interm_alpha", "bi_topk"], "{}", d.line());
        assert_eq!(d.after, (0, NARROW_TOPK), "{}", d.line());
    }

    /// The allocator may only ever REMOVE optional root work.
    #[test]
    fn it_never_raises_a_value_above_the_preset() {
        for (alpha, topk) in [(0usize, 128usize), (1, 256), (2, 1024), (0, 1024)] {
            let s = spec_with(&[(4096, 4.0e9), (4096, 12.0e9)]);
            let d = decide(Some(&s), 97.0, DEFAULT_ALPHA_SECS, alpha, topk, 256);
            assert!(d.after.0 <= alpha, "interm_alpha rose: {}", d.line());
            assert!(d.after.1 <= topk, "topk rose: {}", d.line());
        }
    }

    /// A model whose column-heavy layers are EARLY must price near-identically
    /// at 1024 and at 128 — the tinyimagenet shape, where the measured saving
    /// was -1.2s (6-8%) versus cifar100 resnet_large's 41%.
    #[test]
    fn early_wide_layers_are_cheap_to_cap_and_late_wide_layers_are_not() {
        let early_wide = spec_with(&[(4096, 0.05e9), (64, 8.0e9)]);
        let late_wide = spec_with(&[(64, 0.05e9), (4096, 8.0e9)]);
        let e_full = root_work_proxy(&early_wide, 1024, 256).unwrap();
        let e_cap = root_work_proxy(&early_wide, NARROW_TOPK, 256).unwrap();
        let l_full = root_work_proxy(&late_wide, 1024, 256).unwrap();
        let l_cap = root_work_proxy(&late_wide, NARROW_TOPK, 256).unwrap();
        let e_saving = (e_full - e_cap) / e_full;
        let l_saving = (l_full - l_cap) / l_full;
        assert!(
            l_saving > e_saving * 2.0,
            "capping topk must save far more when the wide layers sit behind an \
             expensive prefix: early={e_saving:.3} late={l_saving:.3}"
        );
    }

    /// The receipt must carry every input needed to re-derive the decision.
    #[test]
    fn the_receipt_line_is_auditable() {
        let s = spec_with(&[(4096, 4.0e9), (4096, 12.0e9)]);
        let line = decide(Some(&s), 97.0, DEFAULT_ALPHA_SECS, 2, 1024, 256).line();
        for field in [
            "applied=1",
            "proxy=",
            "est_bi_s=",
            "tree_s=",
            "target_s=",
            "budget_s=",
            "interm_alpha=",
            "topk=",
            "shed=",
        ] {
            assert!(line.contains(field), "receipt missing {field}: {line}");
        }
    }

    #[test]
    fn a_spec_with_no_trunk_relu_declines_rather_than_guessing() {
        let s = TwinSpec {
            n_in: 3072,
            ops: vec![TwinOpSpec::Relu { input: 0 }],
        };
        let d = decide(Some(&s), 97.0, DEFAULT_ALPHA_SECS, 2, 1024, 256);
        assert!(!d.applied);
        assert_eq!(d.declined, Some("no-proxy"));
        assert_eq!(d.before, d.after);
    }

    // ---------------------------------------------------------------------
    //  #bi-patch pricing
    // ---------------------------------------------------------------------

    /// The cct narrow CNN7 trunk GEOMETRY (3x32x32 -> conv 64 -> conv 64 ->
    /// conv 128 stride 2 -> conv 128 -> conv 128, all 3x3 pad 1) with the
    /// two-Gemm head. The allocator never reads weights, so none are carried.
    fn cnn7_spec() -> TwinSpec {
        let conv = |input: usize, (co, ci): (usize, usize), stride: usize, hw: usize| {
            let o = (hw - 1) / stride + 1;
            TwinOpSpec::Conv {
                input,
                weight: Vec::new(),
                bias: Vec::new(),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                kernel: (co, ci, 3, 3),
                stride: (stride, stride),
                pads: (1, 1, 1, 1),
                ishape: (ci, hw, hw),
                oshape: (co, o, o),
            }
        };
        let head = |input: usize, shape: (usize, usize)| TwinOpSpec::Gemm {
            input,
            weight: Vec::new(),
            bias: Vec::new(),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape,
        };
        TwinSpec {
            n_in: 3 * 32 * 32,
            ops: vec![
                conv(0, (64, 3), 1, 32),           // t1
                TwinOpSpec::Relu { input: 1 },     // t2 (relu_a)
                conv(2, (64, 64), 1, 32),          // t3
                TwinOpSpec::Relu { input: 3 },     // t4 (relu_b)
                conv(4, (128, 64), 2, 32),         // t5
                TwinOpSpec::Relu { input: 5 },     // t6 (relu_c)
                conv(6, (128, 128), 1, 16),        // t7
                TwinOpSpec::Relu { input: 7 },     // t8 (relu_d)
                conv(8, (128, 128), 1, 16),        // t9
                TwinOpSpec::Relu { input: 9 },     // t10 (relu_e)
                TwinOpSpec::Flatten { input: 10 }, // t11
                head(11, (2, 128 * 16 * 16)),      // t12
                TwinOpSpec::Relu { input: 12 },    // t13
                head(13, (2, 2)),                  // t14
            ],
        }
    }

    /// THE COST MODEL THE PHASE AND THE ALLOCATOR SHARE, on the real cct
    /// CNN7 geometry: the per-column patch MACs are the design's
    /// (0.052 / 0.449 / 2.76 / 8.07 M at relu_b..relu_e), the dense MACs are
    /// the nominal prefix counts the dense proxy uses, relu_b is >= 700x
    /// cheaper, relu_e ~16.6x, and CNN7 at 32x32 never converts at the
    /// default ratio (the relu_e input patch is 15x15).
    #[test]
    fn cnn7_patch_plan_reproduces_the_design_cost_model() {
        let spec = cnn7_spec();
        let geo = super::super::patch::geo_of_spec(&spec);
        let sizes = tensor_sizes(&spec);
        let want = [
            (3usize, 52_416u64, 39_518_208u64),
            (5, 448_704, 58_392_576),
            (7, 2_757_312, 96_141_312),
            (9, 8_070_336, 133_890_048),
        ];
        for (relu_op, patch, dense) in want {
            let plan = super::super::patch::Geometry::build(
                &geo,
                &sizes,
                relu_op,
                super::super::patch::DEFAULT_RATIO,
            )
            .unwrap_or_else(|why| panic!("relu op {relu_op} must plan, refused={why}"));
            assert_eq!(plan.macs_col_patch, patch, "relu op {relu_op} patch MACs");
            assert_eq!(plan.macs_col_dense, dense, "relu op {relu_op} dense MACs");
            assert_eq!(plan.convert_at, None, "CNN7 at 32x32 never converts");
        }
        let ratio = |relu_op: usize| {
            let plan = super::super::patch::Geometry::build(&geo, &sizes, relu_op, 0.5).unwrap();
            plan.macs_col_dense as f64 / plan.macs_col_patch as f64
        };
        assert!(ratio(3) >= 700.0, "relu_b ratio {}", ratio(3));
        assert!((ratio(9) - 16.6).abs() < 0.1, "relu_e ratio {}", ratio(9));
        // The relu_e input patch is 15x15 over 3 channels.
        let plan = super::super::patch::Geometry::build(&geo, &sizes, 9, 0.5).unwrap();
        let f0 = plan.frames[0].expect("the walk reaches the input");
        assert_eq!((f0.c, f0.ph, f0.pw, f0.full), (3, 15, 15, false));
    }

    /// `#bi-patch` ARMED: the allocator prices the phase at the patch MACs, so
    /// the cct CNN7 shapes at `interm_alpha: 2, topk: 8192` are
    /// `already-affordable` — where the DENSE pricing reproduces the measured
    /// idx1031 shed (`est_bi_s=50.2`, interm_alpha 2 -> 0). Dark pricing is
    /// the historical decision, and its receipt line is unchanged.
    #[test]
    fn auto_alloc_prices_patch_mode_and_keeps_interm_alpha() {
        let spec = cnn7_spec();
        let budget = 118.0;
        let dense = decide(Some(&spec), budget, DEFAULT_ALPHA_SECS, 2, 8192, 256);
        assert!(dense.applied, "{}", dense.line());
        assert_eq!(dense.shed, vec!["interm_alpha"], "{}", dense.line());
        assert!(
            (dense.est_secs_in - 50.2).abs() < 0.5,
            "the dense pricing must reproduce the measured idx1031 estimate: {}",
            dense.line()
        );
        assert_eq!(dense.pricing, "dense");
        assert!(!dense.line().contains("pricing="), "{}", dense.line());
        let same = decide_with_patch(Some(&spec), budget, DEFAULT_ALPHA_SECS, 2, 8192, 256, None);
        assert_eq!(same, dense, "no patch ratio is the dense decision exactly");

        let ratio = patch_pricing_ratio(Some(true), None);
        assert_eq!(ratio, Some(super::super::patch::DEFAULT_RATIO));
        let patch = decide_with_patch(Some(&spec), budget, DEFAULT_ALPHA_SECS, 2, 8192, 256, ratio);
        assert!(!patch.applied, "{}", patch.line());
        assert_eq!(
            patch.declined,
            Some("already-affordable"),
            "{}",
            patch.line()
        );
        assert_eq!(patch.after, (2, 8192), "{}", patch.line());
        assert_eq!(patch.pricing, "patch");
        assert!(patch.line().contains("pricing=patch"), "{}", patch.line());
        assert!(patch.est_secs_in < 5.0, "{}", patch.line());
        let pd = root_work_proxy(&spec, 8192, 256).unwrap();
        let pp = root_work_proxy_patch(&spec, 8192, 256, 0.5).unwrap();
        assert!(
            pd / pp > 20.0 && pd / pp < 40.0,
            "full-coverage CNN7 prices ~29x under dense, got {:.1}",
            pd / pp
        );
    }

    /// A layer the planner refuses is priced at its DENSE prefix, so a spec
    /// the planner can express nowhere prices exactly as the dense proxy.
    #[test]
    fn a_refused_patch_layer_is_priced_dense() {
        for layers in [
            vec![(4096usize, 1.0e9f64), (2048, 2.0e9)],
            vec![(512, 0.2e9), (512, 0.3e9)],
        ] {
            let s = spec_with(&layers);
            assert_eq!(
                root_work_proxy_patch(&s, 1024, 256, 0.5),
                root_work_proxy(&s, 1024, 256),
                "1x1 fixtures pin no spatial geometry: every layer refuses"
            );
        }
    }

    /// The pricing ratio is `Some` only when the preset ARMS the key, and is
    /// clamped exactly as the phase's reader clamps it.
    #[test]
    fn patch_pricing_follows_the_armed_key_and_the_reader_clamp() {
        use super::super::patch::{DEFAULT_RATIO, RATIO_MAX, RATIO_MIN};
        assert_eq!(patch_pricing_ratio(None, Some(0.25)), None);
        assert_eq!(patch_pricing_ratio(Some(false), Some(0.25)), None);
        assert_eq!(patch_pricing_ratio(Some(true), None), Some(DEFAULT_RATIO));
        assert_eq!(patch_pricing_ratio(Some(true), Some(0.25)), Some(0.25));
        assert_eq!(patch_pricing_ratio(Some(true), Some(7.0)), Some(RATIO_MAX));
        assert_eq!(patch_pricing_ratio(Some(true), Some(0.0)), Some(RATIO_MIN));
        assert_eq!(
            patch_pricing_ratio(Some(true), Some(f64::NAN)),
            Some(DEFAULT_RATIO)
        );
    }
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Orchestration: integrity → inputs → structure → root bounds → per-node
//! discharge ladder → both cover passes → refutation → self-containment →
//! report (brief §4.2).
//!
//! # What confirms a discharge
//!
//! A discharge `(node n, disjunct k)` is confirmed iff for some atom of `k`
//! the checker's own back-substitution certifies `L > 0` for that atom's
//! objective `C y − d` over `R_n` (any single atom with a positive lower
//! bound is violated everywhere on `R_n`, so the disjunct cannot hold). The
//! hint only chooses the dual point (α, β); every certified quantity — site
//! bounds, magnitudes, charges, the concretization — is the checker's.
//!
//! # The ladder (per node)
//!
//! * **fd** (B2.3, a node carrying `hint.fd`) the checker's OWN frozen-dual
//!   state of the hint's state id over the region ([`crate::fd`]: its own
//!   generator-space back-substitution over its own root bounds, the hint's
//!   slopes clamped into `[0, 1]`), and its own directed-rounding Lagrangian
//!   at the hinted dual point. That certified bound ALONE confirms a
//!   discharge (`g > 0`): by weak duality it is valid at every dual point and
//!   slope a bundle can carry, so a wrong or hostile hint can only fail to
//!   confirm. Every built state also passes a sampled containment check
//!   (self-containment (2)); a violation is a checker fault. A fd-confirmed
//!   NODE's certified bound is in addition sampled at points of its node box
//!   (self-containment (3)) when the node is selected by [`fd_node_sample`]:
//!   every node of at most [`FD_NODE_SAMPLE_ALL_DEPTH`] distinct Relu sites,
//!   and a deterministic hashed one in [`FD_NODE_SAMPLE_ONE_IN`] of the
//!   deeper ones (keyed by the node id and the bundle's sha256, so a replay
//!   of the same bundle samples the same nodes). A box point lies in a
//!   depth-`d` node region with probability ~`2^-d`, so on a deep prune that
//!   check is nearly vacuous while each point costs a full interval forward
//!   pass; the per-STATE check above audits the forms every prune's bound is
//!   computed from, and it runs on every state. Both self-containment checks
//!   audit the CHECKER's arithmetic (defence in depth against a checker bug):
//!   neither is a guarantee against a hostile producer, and the per-node
//!   selection in particular is producer-steerable (see [`fd_node_sample`]).
//!   The counts are reported under `fd.node_sampling`.
//! * **L1** root bounds ∩ literal clamps, hint α and β.
//! * **L2** the same bounds, β = 0 and adaptive α.
//! * **L3** (`--refine on`) node bounds re-derived: an interval forward over
//!   the node box with the literal clamps applied at every site, then CROWN
//!   rows for every still-unstable neuron of every site tensor downstream of
//!   the node's shallowest literal; then L1 and L2 again.
//! * **lift** (hint `lift: true`) L1/L2 on up to three ancestor regions (the
//!   node's literals with trailing branch-order literals dropped): a bound on
//!   a superset region is a bound on the node.
//! * **clip** (B2.2, `--refine on`, inputs of dimension <= 64) the checker's
//!   own input clip, per still pending disjunct: every counterexample of the disjunct in `R_n` satisfies
//!   `ℓ_a(x) <= 0` for EVERY atom `a` (each `ℓ_a` a certified linear lower
//!   bound of the atom's objective on the current box ∩ `R_n`), so the box
//!   shrinks to the outward-rounded box hull of those half-spaces; the node's
//!   site bounds are then re-derived on the smaller box (L3 over it) and the
//!   rows re-run. An empty box, an empty region, or one atom with `L > 0` on
//!   the shrunk box confirms the discharge. This is what confirms a lane's
//!   own clip closes (an input-split leaf whose CONJUNCTION of atoms is
//!   infeasible though no single atom's bound is positive).
//!
//! Anything unconfirmed after the ladder is probed for a rigorous
//! counterexample (argmin vertices of the final linear forms, the node box
//! centre, seeded points), which is the only way this module ever reports
//! `refuted`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_cert::json::{Json, Map};
use rayon::prelude::*;

use crate::arith::{add_down, add_up, down, up};
use crate::backsub::{backsub, eval_lin_down, AlphaMap, BetaTerm, Ctx, RowOut, Seeds, SiteView};
use crate::bundle::{parse_bundle, sha256_hex, Bundle, Close, Dis, FdNodeHint, Lit, Malformed};
use crate::cover::{
    check_box_cover, check_cover, BoxCoverResult, BoxMember, CoverLit, CoverResult, DEFAULT_BUDGET,
};
use crate::fd::{self, Form, Frame, LambdaMap, RowRef, Seed};
use crate::ibp::{eval_point, forward_node, Bounds, Magnitudes};
use crate::lower::{lower, Net};
use crate::onnx_wire::parse_model;
use crate::report::{decade_hist, done_line, fnum, object, onum, render, Verdict, REPORT_SCHEMA};
use crate::vnnlib::{self, Disjunct, Region, Spec};

/// A test-only fault the self-containment check must catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultInjection {
    /// Raise every root lower bound of the first site tensor by 1000 after
    /// the root bounds are computed (a wrong box the containment sample sees).
    InflateRootLower,
    /// B2.3: raise the constant of every frozen-dual objective form by 1000
    /// after the FD states are built (a wrong state the FD containment
    /// sample sees).
    InflateFdMargin,
    /// B2.3: raise every prune bound the fd rung certifies by 1000 (a wrong
    /// NODE bound over a correct state: only the per-node sample of a
    /// selected node sees it, see [`fd_node_sample`]).
    InflateFdNodeBound,
}

/// B2.3: every fd-confirmed node of at most this many DISTINCT Relu sites has
/// its certified bound sampled at points of its node box.
pub const FD_NODE_SAMPLE_ALL_DEPTH: usize = 4;

/// B2.3: one in this many deeper fd-confirmed nodes (hash-selected, see
/// [`fd_node_sample`]) is sampled as well.
pub const FD_NODE_SAMPLE_ONE_IN: u64 = 64;

/// B2.3: whether the per-node sample of a fd-confirmed node runs, and which
/// rule decided it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdNodeSample {
    /// Relu depth `<= FD_NODE_SAMPLE_ALL_DEPTH`: always sampled.
    Shallow,
    /// Deeper, selected by the hash: sampled.
    Hashed,
    /// Deeper, not selected: not sampled.
    Skipped,
}

impl FdNodeSample {
    /// Whether the node's certified bound is sampled.
    #[must_use]
    pub const fn sampled(self) -> bool {
        !matches!(self, Self::Skipped)
    }
}

/// B2.3: the per-node sample decision for a fd-confirmed node with `depth`
/// distinct Relu sites and bundle id `node_id`, in the bundle whose sha256
/// (lower hex, as the report prints it) is `bundle_sha256`.
///
/// Deterministic and thread-independent: a pure function of its arguments
/// (sha256 of a domain tag, the bundle hash and the id, little-endian), so a
/// replay of the same bundle samples the same nodes. The depth counts
/// DISTINCT Relu sites only: the sample points are drawn from the node box,
/// which already honours the input-threshold literals, and a repeated
/// literal does not thin the region (the cover dedups it the same way), so
/// only distinct Relu sites lower the chance that a point lies in the node
/// region.
///
/// # What the sample is NOT (review(fd-sampling))
///
/// The key is bundle CONTENT, and the bundle is the producer's input: any
/// byte it changes (a node id, a hint, the order of two lines) re-keys every
/// selection, so a producer can steer which deep nodes are sampled for the
/// price of a few hashes (the tests grind it exactly that way). The sample
/// is therefore a self-containment AUDIT of the checker's OWN fd rung —
/// defence in depth against a CHECKER bug, exercised on honest bundles —
/// and carries no guarantee against a hostile producer; it must never be
/// read as one. (Even an unsteerable selection would audit a deep node only
/// when a sampled point lands inside its region, ~`2^-d` per point.)
///
/// What a hostile producer faces is the rung itself: a discharge is
/// confirmed only by the checker's own certified bound `g > 0`
/// ([`crate::fd::node_bound`]), valid by weak duality for EVERY dual point
/// and slope a bundle can carry (multipliers clamped to `>= 0`, slopes into
/// `[0, 1]`), over a state the checker builds from its own lowering and root
/// bounds. Those forms are audited at sampled region points on every built
/// state (self-containment (2), which this sampling leaves unchanged), and
/// the Lagrangian against an exact-rational re-derivation in the tests,
/// including nodes deeper than [`FD_NODE_SAMPLE_ALL_DEPTH`]. Whether a node
/// is sampled never decides whether it is confirmed.
#[must_use]
pub fn fd_node_sample(depth: usize, node_id: u64, bundle_sha256: &str) -> FdNodeSample {
    if depth <= FD_NODE_SAMPLE_ALL_DEPTH {
        return FdNodeSample::Shallow;
    }
    let mut msg: Vec<u8> = Vec::with_capacity(32 + bundle_sha256.len() + 8);
    msg.extend_from_slice(b"ny-proofcheck/fd-node-sample/v1\0");
    msg.extend_from_slice(bundle_sha256.as_bytes());
    msg.extend_from_slice(&node_id.to_le_bytes());
    let h = sha256_hex(&msg);
    // sha256_hex is 64 lower-hex digits; a parse failure (impossible) samples.
    let v = u64::from_str_radix(&h[..16], 16).unwrap_or(0);
    if v.is_multiple_of(FD_NODE_SAMPLE_ONE_IN) {
        FdNodeSample::Hashed
    } else {
        FdNodeSample::Skipped
    }
}

/// `ny proof-check` arguments.
#[derive(Debug, Clone)]
pub struct CheckArgs {
    /// The bundle.
    pub bundle: PathBuf,
    /// ONNX override (default: the header path).
    pub onnx: Option<PathBuf>,
    /// VNN-LIB override (default: the header path).
    pub vnnlib: Option<PathBuf>,
    /// Report path (default `<BUNDLE>.check.json`).
    pub report: Option<PathBuf>,
    /// Worker threads (default: rayon's).
    pub threads: Option<usize>,
    /// Memory budget for row chunks, GiB.
    pub mem_gib: f64,
    /// Wall budget, seconds.
    pub budget_s: f64,
    /// L3 refinement.
    pub refine: bool,
    /// Probe points per node: the box centre plus `refute_points - 2`
    /// seeded box points, for refutation on a node with an unconfirmed
    /// discharge and for the per-node fd sample of a selected fd-confirmed
    /// node ([`fd_node_sample`]).
    pub refute_points: usize,
    /// The running checker's build line (recorded in the report).
    pub checker_build: String,
    /// Test-only fault injection. The CLI never sets it.
    #[doc(hidden)]
    pub fault: Option<FaultInjection>,
}

impl CheckArgs {
    /// Defaults for `bundle` (the CLI's defaults).
    #[must_use]
    pub fn new(bundle: PathBuf) -> Self {
        Self {
            bundle,
            onnx: None,
            vnnlib: None,
            report: None,
            threads: None,
            mem_gib: 8.0,
            budget_s: 3600.0,
            refine: true,
            refute_points: 16,
            checker_build: String::new(),
            fault: None,
        }
    }
}

/// The outcome.
#[derive(Debug, Clone)]
pub struct CheckOutcome {
    /// Verdict.
    pub verdict: Verdict,
    /// Exit code (`verdict.code()`).
    pub exit_code: i32,
    /// Where the report was written.
    pub report_path: PathBuf,
    /// The report text.
    pub report_json: String,
    /// sha256 of the report file.
    pub report_sha256: String,
    /// `[proof-check] verdict=...` line.
    pub summary_line: String,
    /// `PROOF_CHECK_DONE rc=...` line.
    pub done_line: String,
}

/// The default report path: `<BUNDLE>.check.json`.
#[must_use]
pub fn default_report_path(bundle: &Path) -> PathBuf {
    let mut s = bundle.as_os_str().to_owned();
    s.push(".check.json");
    PathBuf::from(s)
}

// ---------------------------------------------------------------------------
// Reasons and state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Reason {
    code: String,
    detail: Json,
}

fn reason(code: &str, detail: Json) -> Reason {
    Reason {
        code: code.to_string(),
        detail,
    }
}

fn text_detail(s: impl Into<String>) -> Json {
    object(vec![("text", Json::Str(s.into()))])
}

/// A resolved literal.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RLit {
    Relu { t: usize, i: usize, pos: bool },
    X { i: usize, t: f64, pos: bool },
}

impl RLit {
    fn key(self) -> CoverLit {
        match self {
            Self::Relu { t, i, pos } => ((0, t as u64, i as u64), pos),
            Self::X { i, t, pos } => ((1, i as u64, t.to_bits()), pos),
        }
    }
}

/// A node resolved against the checker's net and spec.
struct RNode<'b> {
    rec: &'b crate::bundle::NodeRec,
    region: usize,
    lits: Vec<RLit>,
    /// Checker disjunct indices claimed; `None` = every disjunct (`"*"`).
    claims: Option<Vec<usize>>,
    beta: Option<Vec<f64>>,
    alpha: Option<u64>,
    empty_site: Option<(usize, usize)>,
    /// B2.3: the frozen-dual prune hint (aligned with `lits`).
    fd: Option<&'b FdNodeHint>,
}

/// B2.3: the checker's own frozen-dual state for one (region, hint state):
/// its frame, one objective form per (disjunct, atom) the state's nodes
/// claim, and one row per Relu literal site they split.
struct FdBuilt {
    frame: Frame,
    margins: BTreeMap<(usize, usize), Form>,
    rows: BTreeMap<(usize, usize), Form>,
}

struct RegionRoot {
    bounds: Bounds,
    mags: Magnitudes,
    ok: bool,
}

struct Overlay<'a> {
    root: &'a Bounds,
    over: HashMap<usize, (Vec<f64>, Vec<f64>)>,
}

impl SiteView for Overlay<'_> {
    fn site(&self, t: usize) -> (&[f64], &[f64]) {
        match self.over.get(&t) {
            Some((l, h)) => (l, h),
            None => (&self.root.lo[t], &self.root.hi[t]),
        }
    }
}

impl<'a> Overlay<'a> {
    fn new(root: &'a Bounds) -> Self {
        Self {
            root,
            over: HashMap::new(),
        }
    }

    fn entry(&mut self, t: usize) -> &mut (Vec<f64>, Vec<f64>) {
        let root = self.root;
        self.over
            .entry(t)
            .or_insert_with(|| (root.lo[t].clone(), root.hi[t].clone()))
    }

    /// Apply the relu literals' clamps; `false` when a clamp empties a site.
    fn clamp(&mut self, lits: &[RLit]) -> bool {
        let mut ok = true;
        for l in lits {
            if let RLit::Relu { t, i, pos } = *l {
                let e = self.entry(t);
                if pos {
                    e.0[i] = e.0[i].max(0.0);
                } else {
                    e.1[i] = e.1[i].min(0.0);
                }
                if e.0[i] > e.1[i] {
                    ok = false;
                }
            }
        }
        ok
    }
}

/// Node box from the region box and the input literals; `None` when empty.
fn node_box(region: &Region, lits: &[RLit]) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut lo = region.lo_out.clone();
    let mut hi = region.hi_out.clone();
    for l in lits {
        if let RLit::X { i, t, pos } = *l {
            if pos {
                lo[i] = lo[i].max(t);
            } else {
                hi[i] = hi[i].min(t);
            }
        }
    }
    if lo.iter().zip(&hi).any(|(a, b)| a > b) {
        return None;
    }
    Some((lo, hi))
}

/// One objective row: `(disjunct k, atom a)`.
#[derive(Debug, Clone, Copy)]
struct RowKey {
    k: usize,
}

fn objective_seeds(net: &Net, region: &Region, rows: &[(usize, usize)]) -> (Seeds, Vec<RowKey>) {
    let mut seeds = Seeds {
        tensor: net.output,
        rows: Vec::with_capacity(rows.len()),
        kappa0: Vec::with_capacity(rows.len()),
        nterms: 1,
    };
    let mut keys = Vec::with_capacity(rows.len());
    for &(k, a) in rows {
        let atom = &region.disjuncts[k].atoms[a];
        seeds.rows.push(atom.coeffs.clone());
        // objective = C y - d >= C y - d_up
        seeds.kappa0.push(-atom.d_up);
        keys.push(RowKey { k });
    }
    (seeds, keys)
}

fn all_rows(region: &Region, ks: &[usize]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for &k in ks {
        for a in 0..region.disjuncts[k].atoms.len() {
            out.push((k, a));
        }
    }
    out
}

fn beta_terms(lits: &[RLit], beta: Option<&[f64]>) -> Vec<BetaTerm> {
    let Some(beta) = beta else {
        return Vec::new();
    };
    if beta.len() != lits.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (l, &b) in lits.iter().zip(beta) {
        if let RLit::Relu { t, i, pos } = *l {
            if b > 0.0 && b.is_finite() {
                out.push(BetaTerm {
                    tensor: t,
                    index: i,
                    coef: if pos { -b } else { b },
                });
            }
        }
    }
    out
}

struct Env<'a> {
    net: &'a Net,
    spec: &'a Spec,
    roots: &'a [RegionRoot],
    alphas: &'a BTreeMap<u64, AlphaMap>,
    chunk_rows: usize,
    refine: bool,
    deadline: Instant,
    refute_points: usize,
    /// B2.3: the built frozen-dual states, by (region, hint state id).
    fd: &'a BTreeMap<(usize, u64), FdBuilt>,
    /// B2.3: the bundle's sha256 (keys the per-node fd sample).
    bundle_sha256: &'a str,
    /// Test-only fault injection (`None` from the CLI).
    fault: Option<FaultInjection>,
}

#[derive(Debug, Clone, Default)]
struct NodeOut {
    empty: bool,
    /// k -> (certified L, ladder step).
    confirmed: BTreeMap<usize, (f64, &'static str)>,
    /// k -> (best L seen, reason).
    unconfirmed: BTreeMap<usize, (f64, &'static str)>,
    refuted: Option<(usize, usize, Vec<f64>, Vec<(f64, f64)>)>,
    fault: Option<String>,
    hint_gain: Vec<f64>,
    l3_rows: usize,
    /// B2.3: `(disjunct, atom, certified g)` the fd rung confirmed.
    fd_confirmed: Vec<(usize, usize, f64)>,
    /// B2.3: the fd rung evaluated this node.
    fd_tried: bool,
    /// B2.3: the per-node sample decision (a fd-confirmed node only).
    fd_sample: Option<FdNodeSample>,
    /// B2.3: points the per-node fd sample evaluated (forward passes).
    fd_points: u64,
    /// B2.3: of those, points inside the node region (where it applied).
    fd_in_region: u64,
}

struct Attempt<'a> {
    view: &'a Overlay<'a>,
    box_lo: &'a [f64],
    box_hi: &'a [f64],
    alpha: Option<&'a AlphaMap>,
    betas: Vec<BetaTerm>,
}

fn run_rows(
    env: &Env<'_>,
    root: &RegionRoot,
    region: &Region,
    at: &Attempt<'_>,
    rows: &[(usize, usize)],
) -> Vec<(RowKey, RowOut)> {
    let (seeds, keys) = objective_seeds(env.net, region, rows);
    let ctx = Ctx {
        net: env.net,
        mags: &root.mags,
        view: at.view,
        alpha: at.alpha,
        betas: &at.betas,
        box_lo: at.box_lo,
        box_hi: at.box_hi,
        chunk_rows: env.chunk_rows,
        want_lin: true,
    };
    keys.into_iter().zip(backsub(&ctx, &seeds)).collect()
}

/// Certified bounds of neurons `idx` of tensor `t` over the view/box:
/// `(lower, upper)` per neuron (NaN when uncertified).
fn site_rows(
    env: &Env<'_>,
    root: &RegionRoot,
    view: &Overlay<'_>,
    box_lo: &[f64],
    box_hi: &[f64],
    t: usize,
    idx: &[usize],
) -> (Vec<f64>, Vec<f64>, bool) {
    let mut seeds = Seeds {
        tensor: t,
        rows: Vec::with_capacity(idx.len() * 2),
        kappa0: vec![0.0; idx.len() * 2],
        nterms: 1,
    };
    for &i in idx {
        seeds.rows.push(vec![(i, 1.0)]);
        seeds.rows.push(vec![(i, -1.0)]);
    }
    let ctx = Ctx {
        net: env.net,
        mags: &root.mags,
        view,
        alpha: None,
        betas: &[],
        box_lo,
        box_hi,
        chunk_rows: env.chunk_rows,
        want_lin: false,
    };
    let out = backsub(&ctx, &seeds);
    let mut lo = Vec::with_capacity(idx.len());
    let mut hi = Vec::with_capacity(idx.len());
    let mut empty = false;
    for p in out.chunks(2) {
        empty |= p[0].empty || p[1].empty;
        lo.push(p[0].lb);
        hi.push(-p[1].lb);
    }
    (lo, hi, empty)
}

/// L3: re-derive the node's site bounds. Returns `false` when the node is
/// certified empty.
fn refine_node(
    env: &Env<'_>,
    root: &RegionRoot,
    view: &mut Overlay<'_>,
    lits: &[RLit],
    box_lo: &[f64],
    box_hi: &[f64],
    rows_used: &mut usize,
    last_only: bool,
) -> bool {
    let net = env.net;
    // Interval forward over the node box with every site clamped.
    let mut fb = Bounds::empty(net);
    fb.lo[net.input] = box_lo.to_vec();
    fb.hi[net.input] = box_hi.to_vec();
    for k in 0..net.nodes.len() {
        forward_node(net, k, &mut fb);
        let t = net.nodes[k].out;
        if net.is_site(t) {
            let (vl, vh) = view.site(t);
            let (vl, vh) = (vl.to_vec(), vh.to_vec());
            let mut changed = false;
            for i in 0..vl.len() {
                let (l, h) = (fb.lo[t][i].max(vl[i]), fb.hi[t][i].min(vh[i]));
                fb.lo[t][i] = l;
                fb.hi[t][i] = h;
                if l > vl[i] || h < vh[i] {
                    changed = true;
                }
            }
            if changed {
                view.over.insert(t, (fb.lo[t].clone(), fb.hi[t].clone()));
            }
            if fb.lo[t].iter().zip(&fb.hi[t]).any(|(a, b)| a > b) {
                return false;
            }
        }
    }
    // CROWN rows downstream of the shallowest literal.
    let first = lits
        .iter()
        .map(|l| match *l {
            RLit::Relu { t, .. } => net.topo(t),
            RLit::X { .. } => None,
        })
        .min_by(|a, b| match (a, b) {
            (None, _) => std::cmp::Ordering::Less,
            (_, None) => std::cmp::Ordering::Greater,
            (Some(x), Some(y)) => x.cmp(y),
        })
        .flatten();
    const MAX_L3_ROWS: usize = 200_000;
    // The deepest site (the head Relu's input in the margin-row family):
    // refined alone first, because its bounds are what a per-domain head
    // relaxation needs and it costs a handful of rows.
    let last_site = (0..net.nodes.len())
        .rev()
        .find(|&k| net.is_site(net.nodes[k].out));
    for k in 0..net.nodes.len() {
        let t = net.nodes[k].out;
        if !net.is_site(t) {
            continue;
        }
        if first.is_some_and(|f| k <= f) {
            continue;
        }
        if last_only && Some(k) != last_site {
            continue;
        }
        if Instant::now() > env.deadline {
            break;
        }
        let (vl, vh) = view.site(t);
        let idx: Vec<usize> = (0..vl.len())
            .filter(|&i| vl[i] < 0.0 && vh[i] > 0.0)
            .collect();
        if idx.is_empty() {
            continue;
        }
        if *rows_used + 2 * idx.len() > MAX_L3_ROWS {
            break;
        }
        *rows_used += 2 * idx.len();
        let (lo, hi, empty) = site_rows(env, root, view, box_lo, box_hi, t, &idx);
        if empty {
            return false;
        }
        let e = view.entry(t);
        for (n, &i) in idx.iter().enumerate() {
            if lo[n].is_finite() {
                e.0[i] = e.0[i].max(lo[n]);
            }
            if hi[n].is_finite() {
                e.1[i] = e.1[i].min(hi[n]);
            }
        }
        // Literal clamps stay applied (a refined bound may have relaxed
        // nothing, but re-apply for clarity).
        for l in lits {
            if let RLit::Relu { t: lt, i, pos } = *l {
                if lt == t {
                    if pos {
                        e.0[i] = e.0[i].max(0.0);
                    } else {
                        e.1[i] = e.1[i].min(0.0);
                    }
                }
            }
        }
        if e.0.iter().zip(&e.1).any(|(a, b)| a > b) {
            return false;
        }
    }
    true
}

/// Emptiness of candidate sites over the view (brief §4.2 step 5).
fn emptiness(
    env: &Env<'_>,
    root: &RegionRoot,
    view: &Overlay<'_>,
    lits: &[RLit],
    box_lo: &[f64],
    box_hi: &[f64],
    candidates: &[(usize, usize)],
) -> bool {
    let mut by_t: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &(t, i) in candidates {
        by_t.entry(t).or_default().insert(i);
    }
    for (t, set) in by_t {
        let idx: Vec<usize> = set.into_iter().collect();
        let (lo, hi, empty) = site_rows(env, root, view, box_lo, box_hi, t, &idx);
        if empty {
            return true;
        }
        for (n, &i) in idx.iter().enumerate() {
            let (l, u) = (lo[n], hi[n]);
            if l.is_finite() && u.is_finite() && l > u {
                return true;
            }
            for lit in lits {
                if let RLit::Relu { t: lt, i: li, pos } = *lit {
                    if lt == t && li == i {
                        if pos && u.is_finite() && u < 0.0 {
                            return true;
                        }
                        if !pos && l.is_finite() && l > 0.0 {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// xorshift64* stream.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Project `p` into the region's INWARD box (inside the exact VNN-LIB box).
fn inward(region: &Region, p: &[f64]) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(p.len());
    for (i, &v) in p.iter().enumerate() {
        let (l, h) = (region.lo_in[i], region.hi_in[i]);
        if !matches!(
            l.partial_cmp(&h),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ) {
            return None;
        }
        out.push(v.clamp(l, h));
    }
    Some(out)
}

/// Is `x` a rigorous counterexample for some disjunct of the region? Returns
/// the disjunct and the output enclosures.
fn refutes(net: &Net, region: &Region, x: &[f64]) -> Option<(usize, Vec<(f64, f64)>)> {
    let b = eval_point(net, x);
    // A counterexample is only as rigorous as EVERY enclosure on its path:
    // a non-finite or inverted interval anywhere (an overflow upstream) is
    // not evidence, so it can never manufacture `refuted` (review(b2)).
    if !b.all_finite() {
        return None;
    }
    let (yl, yh) = (&b.lo[net.output], &b.hi[net.output]);
    for (k, d) in region.disjuncts.iter().enumerate() {
        if d.atoms.iter().all(|a| atom_surely_holds(a, yl, yh)) {
            let ys = yl.iter().zip(yh.iter()).map(|(a, b)| (*a, *b)).collect();
            return Some((k, ys));
        }
    }
    None
}

fn atom_surely_holds(a: &vnnlib::Atom, yl: &[f64], yh: &[f64]) -> bool {
    // upper bound of Σ c_i y_i
    let mut s = 0.0f64;
    for &(j, c) in &a.coeffs {
        let v = if c >= 0.0 {
            up(c * yh[j])
        } else {
            up(c * yl[j])
        };
        s = add_up(s, v);
    }
    if !s.is_finite() {
        return false;
    }
    if a.strict {
        s < a.d_down
    } else {
        s <= a.d_down
    }
}

/// Upper bound of the objective `C y − d` of `(k, atom)` at an evaluated point.
fn objective_hi(a: &vnnlib::Atom, yl: &[f64], yh: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &(j, c) in &a.coeffs {
        let v = if c >= 0.0 {
            up(c * yh[j])
        } else {
            up(c * yl[j])
        };
        s = add_up(s, v);
    }
    add_up(s, -a.d_down)
}

/// Whether `x` lies in `R_n` (rigorously: every literal surely holds).
fn in_region(lits: &[RLit], x: &[f64], b: &Bounds) -> bool {
    lits.iter().all(|l| match *l {
        RLit::Relu { t, i, pos } => {
            if pos {
                b.lo[t][i] >= 0.0
            } else {
                b.hi[t][i] <= 0.0
            }
        }
        RLit::X { i, t, pos } => {
            if pos {
                x[i] >= t
            } else {
                x[i] <= t
            }
        }
    })
}

/// Fold one attempt's rows into the node's state. Returns `true` when a row
/// certified the node region empty.
#[allow(clippy::too_many_arguments)]
fn settle(
    rows: &[(usize, usize)],
    res: Vec<(RowKey, RowOut)>,
    step: &'static str,
    pending: &mut BTreeSet<usize>,
    out: &mut NodeOut,
    best: &mut BTreeMap<usize, f64>,
    confirmed_lin: &mut Vec<(usize, usize, Vec<f64>, f64)>,
    probe_lins: &mut Vec<(f64, Vec<f64>)>,
) -> bool {
    let mut empty = false;
    for (&(_, a), (key, r)) in rows.iter().zip(res) {
        if r.empty {
            empty = true;
            continue;
        }
        let e = best.entry(key.k).or_insert(f64::NEG_INFINITY);
        if r.lb > *e {
            *e = r.lb;
        }
        if r.lb > 0.0 && r.lb.is_finite() {
            if pending.remove(&key.k) {
                out.confirmed.insert(key.k, (r.lb, step));
            }
            if let Some((lin, c)) = r.lin {
                confirmed_lin.push((key.k, a, lin, c));
            }
        } else if let Some((lin, _)) = r.lin {
            probe_lins.push((r.lb, lin));
        }
    }
    empty
}

/// Most clip rounds per pending disjunct (each costs one L3 re-derivation).
const CLIP_ROUNDS: usize = 4;

/// The clip runs only on networks with at most this many inputs: box
/// clipping empties boxes in LOW dimension (the input-split and lowdim
/// lanes); on an image input it would only spend the budget.
const CLIP_MAX_INPUTS: usize = 64;

/// Shrink `[lo, hi]` to the box hull of `{x : lin·x + c <= 0}` within it,
/// every step rounded OUTWARD (the result contains every point of the box
/// satisfying the half-space). Returns `false` when the box becomes empty.
/// A non-finite form leaves the box unchanged.
fn clip_halfspace(lin: &[f64], c: f64, lo: &mut [f64], hi: &mut [f64]) -> bool {
    if !c.is_finite() || lin.len() != lo.len() || lin.iter().any(|v| !v.is_finite()) {
        return true;
    }
    // m_j <= min over the box of lin_j x_j (rounded down), and their sum.
    let m: Vec<f64> = lin
        .iter()
        .zip(lo.iter().zip(hi.iter()))
        .map(
            |(&a, (&l, &h))| {
                if a >= 0.0 {
                    down(a * l)
                } else {
                    down(a * h)
                }
            },
        )
        .collect();
    let mut total = 0.0f64;
    for &v in &m {
        total = add_down(total, v);
    }
    if !total.is_finite() {
        return true;
    }
    for i in 0..lin.len() {
        let a = lin[i];
        if a == 0.0 {
            continue;
        }
        // S_i <= Σ_{j≠i} lin_j x_j on the box; the half-space needs
        // a x_i <= R = -c - S_i (rounded up: a weaker, sound bound).
        let rest = down(total - m[i]);
        let r = add_up(-c, -rest);
        if !r.is_finite() {
            continue;
        }
        if a > 0.0 {
            let t = up(r / a);
            if t < hi[i] {
                hi[i] = t;
            }
        } else {
            let t = down(r / a);
            if t > lo[i] {
                lo[i] = t;
            }
        }
        if lo[i] > hi[i] {
            return false;
        }
    }
    true
}

/// The clip step (see the module docs) for disjunct `k` of node `n`: `true`
/// when the disjunct is certified to have no counterexample in `R_n`.
///
/// Invariant: every counterexample of `k` in `R_n` lies in `[lo, hi]`. It
/// holds at the node box; site bounds re-derived over `[lo, hi]` (starting
/// from bounds valid on a superset) are valid on `[lo, hi] ∩ R_n`, so every
/// atom's linear form lower-bounds its objective there, and a counterexample
/// satisfies every atom, hence every `ℓ_a <= 0`: the clipped box keeps it.
#[allow(clippy::too_many_arguments)]
fn clip_disjunct(
    env: &Env<'_>,
    root: &RegionRoot,
    region: &Region,
    n: &RNode<'_>,
    k: usize,
    base: &Overlay<'_>,
    box_lo: &[f64],
    box_hi: &[f64],
    rows_used: &mut usize,
) -> bool {
    let mut lo = box_lo.to_vec();
    let mut hi = box_hi.to_vec();
    let mut view = Overlay::new(base.root);
    view.over = base.over.clone();
    let rows = all_rows(region, &[k]);
    for _ in 0..CLIP_ROUNDS {
        if Instant::now() > env.deadline {
            return false;
        }
        let mut used = 0usize;
        if !refine_node(env, root, &mut view, &n.lits, &lo, &hi, &mut used, false) {
            return true;
        }
        *rows_used += used;
        let at = Attempt {
            view: &view,
            box_lo: &lo,
            box_hi: &hi,
            alpha: None,
            betas: beta_terms(&n.lits, n.beta.as_deref()),
        };
        let res = run_rows(env, root, region, &at, &rows);
        let mut forms: Vec<(Vec<f64>, f64)> = Vec::with_capacity(res.len());
        for (_, r) in res {
            if r.empty {
                return true;
            }
            if r.lb > 0.0 && r.lb.is_finite() {
                return true;
            }
            if let Some(form) = r.lin {
                forms.push(form);
            }
        }
        let before: Vec<(u64, u64)> = lo
            .iter()
            .zip(&hi)
            .map(|(a, b)| (a.to_bits(), b.to_bits()))
            .collect();
        for (lin, c) in &forms {
            if !clip_halfspace(lin, *c, &mut lo, &mut hi) {
                return true;
            }
        }
        let after: Vec<(u64, u64)> = lo
            .iter()
            .zip(&hi)
            .map(|(a, b)| (a.to_bits(), b.to_bits()))
            .collect();
        if before == after {
            return false;
        }
    }
    false
}

#[allow(clippy::too_many_lines)]
/// B2.3: the fd rung. For a node carrying `hint.fd` whose state was built,
/// every pending disjunct is confirmed by ANY of its atoms whose certified
/// prune bound (the checker's own state, the hinted dual point) is `> 0`.
/// Input-threshold literals are skipped (a bound over a larger region is a
/// bound over the node); a Relu literal without a built row refuses the rung.
fn fd_rung(
    env: &Env<'_>,
    n: &RNode<'_>,
    region: &Region,
    pending: &mut BTreeSet<usize>,
    best: &mut BTreeMap<usize, f64>,
    out: &mut NodeOut,
) {
    let Some(h) = n.fd else {
        return;
    };
    let Some(st) = env.fd.get(&(n.region, h.state)) else {
        return;
    };
    let mut rows: Vec<RowRef<'_>> = Vec::with_capacity(n.lits.len());
    let mut l0 = Vec::with_capacity(n.lits.len());
    let mut l1 = Vec::with_capacity(n.lits.len());
    for (j, lit) in n.lits.iter().enumerate() {
        if let RLit::Relu { t, i, pos } = *lit {
            let Some(form) = st.rows.get(&(t, i)) else {
                return;
            };
            rows.push(RowRef {
                form,
                relax: st.frame.relu(t, i),
                on: pos,
            });
            l0.push(h.l0[j]);
            l1.push(h.l1[j]);
        }
    }
    out.fd_tried = true;
    let ks: Vec<usize> = pending.iter().copied().collect();
    for k in ks {
        for a in 0..region.disjuncts[k].atoms.len() {
            let Some(margin) = st.margins.get(&(k, a)) else {
                continue;
            };
            let mut g = fd::node_bound(margin, &rows, &l0, &l1);
            if env.fault == Some(FaultInjection::InflateFdNodeBound) {
                g += 1000.0;
            }
            let e = best.entry(k).or_insert(f64::NEG_INFINITY);
            if g > *e {
                *e = g;
            }
            if g.is_finite() && g > 0.0 {
                pending.remove(&k);
                out.confirmed.insert(k, (g, "fd"));
                out.fd_confirmed.push((k, a, g));
                break;
            }
        }
    }
}

/// B2.3: what was asked of one (region, hint state): the Relu literal sites
/// its nodes split and the (disjunct, atom) objectives they claim.
#[derive(Default)]
struct FdWant {
    sites: BTreeSet<(usize, usize)>,
    margins: BTreeSet<(usize, usize)>,
    nodes: usize,
}

/// B2.3: build the checker's own frozen-dual state of every (region, hint
/// state) some node names. Returns the built states and one report row per
/// requested state (built or refused, with why).
#[allow(clippy::too_many_arguments)]
fn build_fd_states(
    net: &Net,
    spec: &Spec,
    roots: &[RegionRoot],
    rnodes: &[RNode<'_>],
    lambda: &BTreeMap<u64, LambdaMap>,
    hints: &[crate::bundle::FdStateHint],
    mem_bytes: usize,
    deadline: Instant,
) -> (BTreeMap<(usize, u64), FdBuilt>, Vec<Json>) {
    let mut want: BTreeMap<(usize, u64), FdWant> = BTreeMap::new();
    for n in rnodes {
        let Some(h) = n.fd else {
            continue;
        };
        let w = want.entry((n.region, h.state)).or_default();
        w.nodes += 1;
        for l in &n.lits {
            if let RLit::Relu { t, i, .. } = *l {
                w.sites.insert((t, i));
            }
        }
        let region = &spec.regions[n.region];
        let ks: Vec<usize> = match &n.claims {
            Some(v) => v.clone(),
            None => (0..region.disjuncts.len()).collect(),
        };
        for k in ks {
            for a in 0..region.disjuncts[k].atoms.len() {
                w.margins.insert((k, a));
            }
        }
    }
    let mut built = BTreeMap::new();
    let mut rows_json = Vec::new();
    let mut spent: usize = 0;
    for ((r, id), w) in want {
        let hint = hints.iter().find(|h| h.id == id);
        let mut row = vec![
            ("region", Json::U64(r as u64)),
            ("id", Json::U64(id)),
            ("nodes", Json::U64(w.nodes as u64)),
            (
                "class",
                hint.and_then(|h| h.class).map_or(Json::Null, Json::U64),
            ),
            (
                "lane_sha256",
                hint.and_then(|h| h.state_sha256.clone())
                    .map_or(Json::Null, Json::Str),
            ),
            (
                "lane_n_gens",
                hint.and_then(|h| h.n_gens).map_or(Json::Null, Json::U64),
            ),
            (
                "lane_unstable",
                hint.and_then(|h| h.unstable).map_or(Json::Null, Json::U64),
            ),
            (
                "lane_relaxed",
                hint.and_then(|h| h.relaxed).map_or(Json::Null, Json::U64),
            ),
            ("lane_root", onum(hint.and_then(|h| h.root))),
        ];
        let refuse = |mut row: Vec<(&'static str, Json)>, why: &str| {
            row.push(("built", Json::Bool(false)));
            row.push(("refused", Json::Str(why.into())));
            object(row)
        };
        // #fd-compact-checkable: a producer that DECLARES how many
        // relaxation symbols its slope table covers (`relaxed`; a ranked
        // compact state's class relaxation included) must deliver exactly
        // that table. A short or padded one (a dropped, duplicated or
        // unresolvable entry) is refused rather than silently filled at the
        // chord default: downgrade-only, its prunes stay unconfirmed.
        let entries: usize = lambda
            .get(&id)
            .map_or(0, |m| m.values().map(HashMap::len).sum());
        row.push(("table_entries", Json::U64(entries as u64)));
        if hint
            .and_then(|h| h.relaxed)
            .is_some_and(|declared| declared != entries as u64)
        {
            rows_json.push(refuse(row, "fd_table_count"));
            continue;
        }
        let root = &roots[r];
        if !root.ok {
            rows_json.push(refuse(row, "nonfinite_root"));
            continue;
        }
        if Instant::now() > deadline {
            rows_json.push(refuse(row, "budget"));
            continue;
        }
        let region = &spec.regions[r];
        let frame = match Frame::new(
            net,
            &root.bounds,
            &region.lo_out,
            &region.hi_out,
            lambda.get(&id),
        ) {
            Ok(f) => f,
            Err(e) => {
                rows_json.push(refuse(row, &format!("frame:{e}")));
                continue;
            }
        };
        let need = (w.sites.len() + w.margins.len())
            .saturating_mul(frame.n_gens.max(1))
            .saturating_mul(8);
        if spent.saturating_add(need) > mem_bytes / 2 {
            rows_json.push(refuse(row, "memory"));
            continue;
        }
        spent = spent.saturating_add(need);
        let mut seeds: Vec<Seed> = Vec::with_capacity(w.sites.len() + w.margins.len());
        let margin_keys: Vec<(usize, usize)> = w.margins.iter().copied().collect();
        for &(k, a) in &margin_keys {
            let atom = &region.disjuncts[k].atoms[a];
            seeds.push(Seed {
                tensor: net.output,
                coeffs: atom.coeffs.clone(),
                k0: crate::arith::Iv::new(-atom.d_up, -atom.d_down),
            });
        }
        let site_keys: Vec<(usize, usize)> = w.sites.iter().copied().collect();
        for &(t, i) in &site_keys {
            seeds.push(fd::site_seed(t, i));
        }
        let mut forms = fd::forms(net, &root.mags, &frame, &seeds).into_iter();
        let margins: BTreeMap<(usize, usize), Form> = margin_keys
            .iter()
            .map(|&key| (key, forms.next().expect("one form per seed")))
            .collect();
        let rows: BTreeMap<(usize, usize), Form> = site_keys
            .iter()
            .map(|&key| (key, forms.next().expect("one form per seed")))
            .collect();
        // The checker's own state root per objective (depth 0, no rows).
        let root_chk = margins
            .values()
            .map(|m| fd::node_bound(m, &[], &[], &[]))
            .fold(f64::INFINITY, f64::min);
        let failed = margins
            .values()
            .chain(rows.values())
            .filter(|f| !f.ok)
            .count();
        let digest = frame.digest();
        row.push(("built", Json::Bool(true)));
        row.push(("checker_digest", Json::Str(digest[..16].to_string())));
        row.push(("n_gens", Json::U64(frame.n_gens as u64)));
        row.push(("in_gens", Json::U64(frame.n_in_gens as u64)));
        row.push((
            "unstable",
            Json::U64((frame.n_gens - frame.n_in_gens) as u64),
        ));
        row.push(("hinted_lambda", Json::U64(frame.hinted as u64)));
        // #fd-compact-checkable: how much of the checker's own relaxation the
        // producer's table chose (diagnostic; the bound is the checker's).
        row.push(("chord_lambda", Json::U64(frame.chord as u64)));
        row.push(("hint_unused", Json::U64(frame.hint_unused as u64)));
        row.push(("rows", Json::U64(rows.len() as u64)));
        row.push(("margins", Json::U64(margins.len() as u64)));
        row.push(("forms_failed", Json::U64(failed as u64)));
        row.push(("root_chk", onum(Some(root_chk))));
        rows_json.push(object(row));
        built.insert(
            (r, id),
            FdBuilt {
                frame,
                margins,
                rows,
            },
        );
    }
    (built, rows_json)
}

/// B2.3 self-containment (2): at sampled points of the region, every built
/// FD objective form lies below the evaluated objective and every row form
/// meets the evaluated pre-activation. A violation is a checker fault.
fn fd_containment(net: &Net, region: &Region, st: &FdBuilt, seed: u64) -> Option<String> {
    let mut rng = Rng::new(seed);
    for p in 0..16 {
        let pt: Vec<f64> = region
            .lo_out
            .iter()
            .zip(&region.hi_out)
            .map(|(a, b)| {
                if p == 0 {
                    0.5 * a + 0.5 * b
                } else {
                    a + rng.next_f64() * (b - a)
                }
            })
            .collect();
        let Some(x) = inward(region, &pt) else {
            break;
        };
        let b = eval_point(net, &x);
        if !b.all_finite() {
            continue;
        }
        let e = fd::symbols_at(net, &st.frame, &x, &b);
        let (yl, yh) = (&b.lo[net.output], &b.hi[net.output]);
        for (&(k, a), f) in &st.margins {
            if !f.ok {
                continue;
            }
            let v = fd::form_at(f, &e);
            let hi = objective_hi(&region.disjuncts[k].atoms[a], yl, yh);
            if v.lo > hi {
                return Some(format!(
                    "fd objective form of disjunct {k} atom {a} ({:e}) exceeds the evaluated \
                     objective ({hi:e}) at a sampled point",
                    v.lo
                ));
            }
        }
        for (&(t, i), f) in &st.rows {
            if !f.ok {
                continue;
            }
            let v = fd::form_at(f, &e);
            if v.lo > b.hi[t][i] || v.hi < b.lo[t][i] {
                return Some(format!(
                    "fd row form of `{}`[{i}] excludes the evaluated pre-activation at a \
                     sampled point",
                    net.tensors[t].name
                ));
            }
        }
    }
    None
}

fn process_node(env: &Env<'_>, n: &RNode<'_>) -> NodeOut {
    let mut out = NodeOut::default();
    let region = &env.spec.regions[n.region];
    let root = &env.roots[n.region];
    let claims: Vec<usize> = match &n.claims {
        Some(v) => v.clone(),
        None => (0..region.disjuncts.len()).collect(),
    };
    if matches!(n.rec.close, Close::Subtree) {
        return out;
    }
    if matches!(n.rec.close, Close::Opaque) {
        for &k in &claims {
            out.unconfirmed.insert(k, (f64::NAN, "opaque"));
        }
        return out;
    }
    if Instant::now() > env.deadline {
        for &k in &claims {
            out.unconfirmed.insert(k, (f64::NAN, "budget"));
        }
        return out;
    }
    if !root.ok {
        for &k in &claims {
            out.unconfirmed.insert(k, (f64::NAN, "nonfinite_root"));
        }
        return out;
    }
    let Some((box_lo, box_hi)) = node_box(region, &n.lits) else {
        out.empty = true;
        return out;
    };
    let mut view = Overlay::new(&root.bounds);
    if !view.clamp(&n.lits) {
        out.empty = true;
        return out;
    }
    let alpha = n.alpha.and_then(|id| env.alphas.get(&id));
    let fail_reason = if matches!(n.rec.close, Close::Fd) {
        "needs_fd_check"
    } else if n.fd.is_some() {
        "fd_weaker_state"
    } else {
        "weaker_relaxation"
    };
    let relu_sites: Vec<(usize, usize)> = n
        .lits
        .iter()
        .filter_map(|l| match *l {
            RLit::Relu { t, i, .. } => Some((t, i)),
            RLit::X { .. } => None,
        })
        .collect();
    // Emptiness claims first.
    if matches!(n.rec.close, Close::Empty) {
        let mut cands: Vec<(usize, usize)> = n.empty_site.into_iter().collect();
        cands.extend(relu_sites.iter().copied());
        if emptiness(env, root, &view, &n.lits, &box_lo, &box_hi, &cands) {
            out.empty = true;
            return out;
        }
    }
    let mut pending: BTreeSet<usize> = claims.iter().copied().collect();
    let mut best: BTreeMap<usize, f64> = BTreeMap::new();
    let mut confirmed_lin: Vec<(usize, usize, Vec<f64>, f64)> = Vec::new();
    let mut probe_lins: Vec<(f64, Vec<f64>)> = Vec::new();
    // B2.3: the frozen-dual rung — the checker's OWN FD state of this node's
    // hint state and its own directed-rounding Lagrangian at the hinted dual
    // point (any `L >= 0` is a valid dual point; see `crate::fd`).
    fd_rung(env, n, region, &mut pending, &mut best, &mut out);
    // L1: hint α and β.
    let betas = beta_terms(&n.lits, n.beta.as_deref());
    let l1 = Attempt {
        view: &view,
        box_lo: &box_lo,
        box_hi: &box_hi,
        alpha,
        betas,
    };
    let rows = all_rows(region, &pending.iter().copied().collect::<Vec<_>>());
    let res = run_rows(env, root, region, &l1, &rows);
    let l1_best: BTreeMap<usize, f64> = {
        let mut m = BTreeMap::new();
        for (key, r) in &res {
            let e = m.entry(key.k).or_insert(f64::NEG_INFINITY);
            if r.lb > *e {
                *e = r.lb;
            }
        }
        m
    };
    if settle(
        &rows,
        res,
        "L1",
        &mut pending,
        &mut out,
        &mut best,
        &mut confirmed_lin,
        &mut probe_lins,
    ) {
        out.empty = true;
        out.confirmed.clear();
        return out;
    }
    // L2: β = 0, adaptive α.
    if !pending.is_empty() && Instant::now() <= env.deadline {
        let l2 = Attempt {
            view: &view,
            box_lo: &box_lo,
            box_hi: &box_hi,
            alpha: None,
            betas: Vec::new(),
        };
        let rows = all_rows(region, &pending.iter().copied().collect::<Vec<_>>());
        let res = run_rows(env, root, region, &l2, &rows);
        for (key, r) in &res {
            if let Some(b1) = l1_best.get(&key.k) {
                if b1.is_finite() && r.lb.is_finite() {
                    out.hint_gain.push(b1 - r.lb);
                }
            }
        }
        if settle(
            &rows,
            res,
            "L2",
            &mut pending,
            &mut out,
            &mut best,
            &mut confirmed_lin,
            &mut probe_lins,
        ) {
            out.empty = true;
            out.confirmed.clear();
            return out;
        }
    }
    // L3: refined node bounds — the deepest site alone first (cheap), then
    // every site downstream of the node's shallowest literal.
    let mut refined = Overlay::new(&root.bounds);
    refined.over = view.over.clone();
    for last_only in [true, false] {
        if pending.is_empty() || !env.refine || Instant::now() > env.deadline {
            break;
        }
        let mut used = 0usize;
        if !refine_node(
            env,
            root,
            &mut refined,
            &n.lits,
            &box_lo,
            &box_hi,
            &mut used,
            last_only,
        ) {
            out.empty = true;
            out.confirmed.clear();
            return out;
        }
        out.l3_rows += used;
        if matches!(n.rec.close, Close::Empty) {
            let mut cands: Vec<(usize, usize)> = n.empty_site.into_iter().collect();
            cands.extend(relu_sites.iter().copied());
            if emptiness(env, root, &refined, &n.lits, &box_lo, &box_hi, &cands) {
                out.empty = true;
                out.confirmed.clear();
                return out;
            }
        }
        for with_hints in [true, false] {
            if pending.is_empty() || Instant::now() > env.deadline {
                break;
            }
            let at = Attempt {
                view: &refined,
                box_lo: &box_lo,
                box_hi: &box_hi,
                alpha: if with_hints { alpha } else { None },
                betas: if with_hints {
                    beta_terms(&n.lits, n.beta.as_deref())
                } else {
                    Vec::new()
                },
            };
            let rows = all_rows(region, &pending.iter().copied().collect::<Vec<_>>());
            let res = run_rows(env, root, region, &at, &rows);
            if settle(
                &rows,
                res,
                "L3",
                &mut pending,
                &mut out,
                &mut best,
                &mut confirmed_lin,
                &mut probe_lins,
            ) {
                out.empty = true;
                out.confirmed.clear();
                return out;
            }
        }
    }
    // Lift: ancestor regions.
    if !pending.is_empty() && n.rec.hint.lift {
        for d in 1..=3usize {
            if pending.is_empty() || d > n.lits.len() || Instant::now() > env.deadline {
                break;
            }
            let anc = &n.lits[..n.lits.len() - d];
            let Some((alo, ahi)) = node_box(region, anc) else {
                continue;
            };
            let mut av = Overlay::new(&root.bounds);
            if !av.clamp(anc) {
                continue;
            }
            let abeta = n
                .beta
                .as_ref()
                .map(|b| b[..b.len().min(anc.len())].to_vec());
            for with_hints in [true, false] {
                if pending.is_empty() {
                    break;
                }
                let at = Attempt {
                    view: &av,
                    box_lo: &alo,
                    box_hi: &ahi,
                    alpha: if with_hints { alpha } else { None },
                    betas: if with_hints {
                        beta_terms(anc, abeta.as_deref())
                    } else {
                        Vec::new()
                    },
                };
                let rows = all_rows(region, &pending.iter().copied().collect::<Vec<_>>());
                let res = run_rows(env, root, region, &at, &rows);
                // An ancestor's emptiness says nothing new; skip it.
                let res: Vec<(RowKey, RowOut)> = res
                    .into_iter()
                    .map(|(k, mut r)| {
                        r.empty = false;
                        (k, r)
                    })
                    .collect();
                let mut dummy_lins = Vec::new();
                settle(
                    &rows,
                    res,
                    "lift",
                    &mut pending,
                    &mut out,
                    &mut best,
                    &mut confirmed_lin,
                    &mut dummy_lins,
                );
            }
        }
    }
    // Clip: the checker's own input clip per pending disjunct (low-dimensional
    // inputs only; a frozen-dual claim needs fd-check, not a clip).
    if !pending.is_empty()
        && env.refine
        && env.net.n_in() <= CLIP_MAX_INPUTS
        && !matches!(n.rec.close, Close::Fd)
    {
        let ks: Vec<usize> = pending.iter().copied().collect();
        for k in ks {
            if Instant::now() > env.deadline {
                break;
            }
            let mut used = 0usize;
            if clip_disjunct(
                env, root, region, n, k, &refined, &box_lo, &box_hi, &mut used,
            ) {
                pending.remove(&k);
                out.confirmed.insert(k, (f64::NAN, "clip"));
            }
            out.l3_rows += used;
        }
    }
    for &k in &pending {
        let b = best.get(&k).copied().unwrap_or(f64::NAN);
        let why = if Instant::now() > env.deadline {
            "budget"
        } else if b.is_nan() {
            "nonfinite"
        } else {
            fail_reason
        };
        out.unconfirmed.insert(k, (b, why));
    }
    // Probe points: refutation for the unconfirmed discharges and
    // self-containment for the confirmed ones. B2.3: a fd-confirmed node's
    // bound is sampled only when `fd_node_sample` selects the node (every
    // shallow node, a hashed fraction of the deep ones; see the module docs).
    if !out.fd_confirmed.is_empty() {
        // review(fd-sampling): the depth is the count of DISTINCT Relu sites
        // (a repeated literal does not thin the region; the cover dedups it
        // the same way), so repeating a literal cannot push a shallow node
        // out of the always-sampled set.
        let depth = relu_sites.iter().collect::<BTreeSet<_>>().len();
        out.fd_sample = Some(fd_node_sample(depth, n.rec.id, env.bundle_sha256));
    }
    let fd_check = out.fd_sample.is_some_and(FdNodeSample::sampled);
    let mut points: Vec<Vec<f64>> = Vec::new();
    probe_lins.sort_by(|a, b| a.0.total_cmp(&b.0));
    if !out.unconfirmed.is_empty() {
        for (_, lin) in probe_lins.iter().take(8) {
            let v: Vec<f64> = lin
                .iter()
                .enumerate()
                .map(|(i, &c)| if c >= 0.0 { box_lo[i] } else { box_hi[i] })
                .collect();
            points.push(v);
        }
    }
    let centre: Vec<f64> = box_lo
        .iter()
        .zip(&box_hi)
        .map(|(a, b)| 0.5 * a + 0.5 * b)
        .collect();
    points.push(centre);
    if !out.unconfirmed.is_empty() || fd_check {
        let mut rng = Rng::new(n.rec.id ^ 0x5eed_0000);
        for _ in 0..env.refute_points.saturating_sub(2) {
            let p: Vec<f64> = box_lo
                .iter()
                .zip(&box_hi)
                .map(|(a, b)| {
                    let u = rng.next_f64();
                    (a + u * (b - a)).clamp(*a, *b)
                })
                .collect();
            points.push(p);
        }
    }
    for p in points {
        let Some(x) = inward(region, &p) else {
            continue;
        };
        if !out.unconfirmed.is_empty() && out.refuted.is_none() {
            if let Some((k, ys)) = refutes(env.net, region, &x) {
                out.refuted = Some((n.region, k, x.clone(), ys));
            }
        }
        if fd_check && out.fault.is_none() {
            // B2.3: a confirmed fd bound is a lower bound of the objective at
            // every point of the node region.
            let b = eval_point(env.net, &x);
            out.fd_points += 1;
            if b.all_finite() && in_region(&n.lits, &x, &b) {
                out.fd_in_region += 1;
                let (yl, yh) = (&b.lo[env.net.output], &b.hi[env.net.output]);
                for &(k, a, g) in &out.fd_confirmed {
                    let atom = &region.disjuncts[k].atoms[a];
                    let g_hi = objective_hi(atom, yl, yh);
                    if g > g_hi {
                        out.fault = Some(format!(
                            "node {} disjunct {k}: certified fd bound {g:e} exceeds the \
                             evaluated objective {g_hi:e} at a point inside the node region",
                            n.rec.id
                        ));
                        break;
                    }
                }
            }
        }
        if !confirmed_lin.is_empty() && out.fault.is_none() {
            let b = eval_point(env.net, &x);
            if in_region(&n.lits, &x, &b) {
                let (yl, yh) = (&b.lo[env.net.output], &b.hi[env.net.output]);
                for (k, a, lin, c) in &confirmed_lin {
                    let atom = &region.disjuncts[*k].atoms[*a];
                    let g_hi = objective_hi(atom, yl, yh);
                    let l_lo = eval_lin_down(lin, *c, &x);
                    if l_lo > g_hi {
                        out.fault = Some(format!(
                            "node {} disjunct {k}: certified linear form {l_lo:e} exceeds the \
                             evaluated objective {g_hi:e} at a point inside the node region",
                            n.rec.id
                        ));
                        break;
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Root bounds
// ---------------------------------------------------------------------------

fn root_bounds(
    net: &Net,
    region: &Region,
    chunk_rows: usize,
    refine: bool,
    deadline: Instant,
) -> RegionRoot {
    let mut b = Bounds::empty(net);
    b.lo[net.input] = region.lo_out.clone();
    b.hi[net.input] = region.hi_out.clone();
    let mut mags = Magnitudes::empty(net);
    mags.set_tensor(net.input, &b.lo[net.input], &b.hi[net.input]);
    for k in 0..net.nodes.len() {
        forward_node(net, k, &mut b);
        mags.set_node(net, k);
        let t = net.nodes[k].out;
        if refine && net.is_site(t) && Instant::now() <= deadline {
            let idx: Vec<usize> = (0..b.lo[t].len())
                .filter(|&i| b.lo[t][i] < 0.0 && b.hi[t][i] > 0.0)
                .collect();
            if !idx.is_empty() {
                // Magnitudes of `t` are needed by nothing upstream; the rows
                // are seeded AT `t`, so its own magnitude enters the seed
                // charge: record the IBP magnitude first.
                mags.set_tensor(t, &b.lo[t], &b.hi[t]);
                let root_view = RootView { b: &b };
                let mut seeds = Seeds {
                    tensor: t,
                    rows: Vec::with_capacity(idx.len() * 2),
                    kappa0: vec![0.0; idx.len() * 2],
                    nterms: 1,
                };
                for &i in &idx {
                    seeds.rows.push(vec![(i, 1.0)]);
                    seeds.rows.push(vec![(i, -1.0)]);
                }
                let ctx = Ctx {
                    net,
                    mags: &mags,
                    view: &root_view,
                    alpha: None,
                    betas: &[],
                    box_lo: &region.lo_out,
                    box_hi: &region.hi_out,
                    chunk_rows,
                    want_lin: false,
                };
                let res = backsub(&ctx, &seeds);
                for (n, &i) in idx.iter().enumerate() {
                    let l = res[2 * n].lb;
                    let h = -res[2 * n + 1].lb;
                    if l.is_finite() {
                        b.lo[t][i] = b.lo[t][i].max(l);
                    }
                    if h.is_finite() {
                        b.hi[t][i] = b.hi[t][i].min(h);
                    }
                }
            }
        }
        mags.set_tensor(t, &b.lo[t], &b.hi[t]);
    }
    let ok = b.all_finite();
    RegionRoot {
        bounds: b,
        mags,
        ok,
    }
}

struct RootView<'a> {
    b: &'a Bounds,
}

impl SiteView for RootView<'_> {
    fn site(&self, t: usize) -> (&[f64], &[f64]) {
        (&self.b.lo[t], &self.b.hi[t])
    }
}

// ---------------------------------------------------------------------------
// The entry point
// ---------------------------------------------------------------------------

struct Early {
    verdict: Verdict,
    reasons: Vec<Reason>,
}

fn malformed(m: Malformed) -> Early {
    Early {
        verdict: Verdict::Malformed,
        reasons: vec![reason(&m.code, text_detail(m.detail))],
    }
}

fn unsupported(code: &str, detail: impl Into<String>) -> Early {
    Early {
        verdict: Verdict::Unsupported,
        reasons: vec![reason(code, text_detail(detail))],
    }
}

/// Peak resident set size of this process in MiB (Linux `VmHWM`), when
/// readable. Timing-only: excluded from `result_digest`.
fn peak_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

/// Run the checker. `Err` is an OPERATIONAL failure (an unreadable input or
/// an unwritable report: exit 4); every checking outcome, including a
/// malformed bundle, is an `Ok` outcome with its report written.
pub fn check_bundle(args: &CheckArgs) -> Result<CheckOutcome, String> {
    let pool = {
        let mut b = rayon::ThreadPoolBuilder::new();
        if let Some(n) = args.threads {
            b = b.num_threads(n.max(1));
        }
        b.build().map_err(|e| format!("thread pool: {e}"))?
    };
    pool.install(|| check_inner(args))
}

#[allow(clippy::too_many_lines)]
fn check_inner(args: &CheckArgs) -> Result<CheckOutcome, String> {
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_secs_f64(args.budget_s.max(0.0));
    let report_path = args
        .report
        .clone()
        .unwrap_or_else(|| default_report_path(&args.bundle));
    let bytes = std::fs::read(&args.bundle)
        .map_err(|e| format!("cannot read bundle {}: {e}", args.bundle.display()))?;
    let bundle_sha = sha256_hex(&bytes);
    let mut rep = Map::new();
    rep.insert("schema".into(), Json::Str(REPORT_SCHEMA.into()));
    rep.insert(
        "checker_build".into(),
        Json::Str(args.checker_build.clone()),
    );
    let mut timing = Map::new();
    let bundle_is_partial = args
        .bundle
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(".partial"));
    let parsed: Result<Bundle, Early> = if bundle_is_partial {
        Err(malformed(Malformed {
            code: "partial_bundle".into(),
            detail: "a `.partial` bundle was never finalised by a published unsat".into(),
        }))
    } else {
        parse_bundle(&bytes).map_err(malformed)
    };
    let mut bundle_obj = vec![
        ("path", Json::Str(args.bundle.display().to_string())),
        ("sha256", Json::Str(bundle_sha.clone())),
    ];
    let finish = |mut rep: Map,
                  verdict: Verdict,
                  reasons: Vec<Reason>,
                  mut timing: Map,
                  counts: (usize, usize, usize),
                  cover_tok: &str,
                  refuted: bool|
     -> Result<CheckOutcome, String> {
        rep.insert("verdict".into(), Json::Str(verdict.token().into()));
        rep.insert("exit_code".into(), Json::I64(i64::from(verdict.code())));
        rep.insert(
            "reasons".into(),
            Json::Arr(
                reasons
                    .into_iter()
                    .map(|r| object(vec![("code", Json::Str(r.code)), ("detail", r.detail)]))
                    .collect(),
            ),
        );
        let wall = t0.elapsed().as_secs_f64();
        timing.insert("total_s".into(), fnum(wall));
        timing.insert("peak_rss_mb".into(), onum(peak_rss_mb()));
        let rendered = render(rep, Json::Obj(timing));
        let mut file = rendered.text.clone();
        file.push('\n');
        std::fs::write(&report_path, file.as_bytes())
            .map_err(|e| format!("cannot write report {}: {e}", report_path.display()))?;
        let summary = format!(
            "[proof-check] verdict={} exit={} nodes={} confirmed={} unconfirmed={} cover={} \
             result_digest={}",
            verdict.token(),
            verdict.code(),
            counts.0,
            counts.1,
            counts.2,
            cover_tok,
            rendered.result_digest
        );
        let done = done_line(
            verdict,
            counts.0,
            counts.1,
            counts.2,
            cover_tok,
            refuted,
            wall,
            &report_path.display().to_string(),
            &rendered.file_sha256,
        );
        Ok(CheckOutcome {
            verdict,
            exit_code: verdict.code(),
            report_path: report_path.clone(),
            report_json: rendered.text,
            report_sha256: rendered.file_sha256,
            summary_line: summary,
            done_line: done,
        })
    };
    let bundle = match parsed {
        Ok(b) => b,
        Err(e) => {
            rep.insert("bundle".into(), object(bundle_obj));
            return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
        }
    };
    bundle_obj.push(("schema", Json::Str(bundle.header.schema.clone())));
    bundle_obj.push(("lines", Json::U64(bundle.lines as u64)));
    bundle_obj.push(("header_build", Json::Str(bundle.header.build.clone())));
    rep.insert("bundle".into(), object(bundle_obj));
    // Inputs. v2.1: a header input the producer could not hash as one
    // regular file is `unavailable`: it is not read (an `--onnx`/`--vnnlib`
    // override cannot bind it either), and nothing that needs it is checked.
    let onnx_path = args
        .onnx
        .clone()
        .unwrap_or_else(|| PathBuf::from(&bundle.header.onnx.path));
    let vnn_path = args
        .vnnlib
        .clone()
        .unwrap_or_else(|| PathBuf::from(&bundle.header.vnnlib.path));
    let read_input =
        |path: &Path, f: &crate::bundle::FileRef, what: &str| -> Result<Option<Vec<u8>>, String> {
            if f.unavailable.is_some() {
                return Ok(None);
            }
            std::fs::read(path)
                .map(Some)
                .map_err(|e| format!("cannot read {what} {}: {e}", path.display()))
        };
    let onnx_bytes = read_input(&onnx_path, &bundle.header.onnx, "onnx")?;
    let vnn_bytes = read_input(&vnn_path, &bundle.header.vnnlib, "vnnlib")?;
    let onnx_sha = onnx_bytes.as_deref().map(sha256_hex);
    let vnn_sha = vnn_bytes.as_deref().map(sha256_hex);
    let input_json = |path: &Path, sha: Option<&str>, f: &crate::bundle::FileRef| match sha {
        Some(sha) => object(vec![
            ("path", Json::Str(path.display().to_string())),
            ("sha256", Json::Str(sha.to_string())),
        ]),
        None => object(vec![
            ("path", Json::Str(f.path.clone())),
            ("sha256", Json::Null),
            (
                "unavailable",
                Json::Str(f.unavailable.clone().unwrap_or_default()),
            ),
        ]),
    };
    rep.insert(
        "onnx".into(),
        input_json(&onnx_path, onnx_sha.as_deref(), &bundle.header.onnx),
    );
    rep.insert(
        "vnnlib".into(),
        input_json(&vnn_path, vnn_sha.as_deref(), &bundle.header.vnnlib),
    );
    let differs =
        |sha: Option<&str>, f: &crate::bundle::FileRef| sha.is_some_and(|s| s != f.sha256);
    if differs(onnx_sha.as_deref(), &bundle.header.onnx)
        || differs(vnn_sha.as_deref(), &bundle.header.vnnlib)
    {
        let which = if differs(onnx_sha.as_deref(), &bundle.header.onnx) {
            "onnx"
        } else {
            "vnnlib"
        };
        let e = malformed(Malformed {
            code: "hash_mismatch".into(),
            detail: format!("the {which} file's sha256 differs from the bundle header"),
        });
        return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
    }
    rep.insert(
        "verdict_source".into(),
        Json::Str(bundle.verdict.source.clone()),
    );
    // B2.2: the checker re-derives every discharge and the cover itself, so
    // the PRODUCER's authority (`authorized`, or a counted legacy door) does
    // not enter the verdict; what matters is whether the lane streamed a
    // proof. Legacy well-formedness: a published `unsat` bundle claims every
    // clause it requires with at least one node (the producer's writer adds a
    // whole-region `opaque` claim where a lane streamed nothing), so a legacy
    // source with NO node at all claims nothing: the claimed cover is a gap
    // (`invalid`) — a dropped legacy claim, never a scope limit.
    if bundle.verdict.source != "authorized" && bundle.nodes.is_empty() {
        let e = Early {
            verdict: Verdict::Invalid,
            reasons: vec![reason(
                "uncovered_source",
                text_detail(format!(
                    "verdict source `{}` publishes the instance but its bundle claims no \
                     clause (no node): the claimed cover is a gap",
                    bundle.verdict.source
                )),
            )],
        };
        return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "gap", false);
    }
    // v2.1: nothing in an `unavailable` bundle is checkable. It is
    // well-formed, and the most it can be is UNCONFIRMED: when every clause
    // is required and claimed by a WHOLE-REGION node (no literal — the only
    // claim that covers without the property's geometry). A missing claim or
    // requirement is `invalid`, exactly as in a checkable bundle.
    let unavailable = bundle.unavailable();
    if !unavailable.is_empty() {
        let mut reasons: Vec<Reason> = unavailable
            .iter()
            .map(|(code, detail)| reason(code, text_detail(detail.clone())))
            .collect();
        let required: BTreeSet<u64> = bundle.verdict.required.iter().copied().collect();
        let mut gap = false;
        let mut cover_rows = Vec::new();
        for (&region, list) in &bundle.clauses {
            for c in list {
                let claimed = bundle.nodes.iter().any(|n| {
                    n.region == region
                        && n.lits.is_empty()
                        && !matches!(n.close, Close::Subtree)
                        && match &n.dis {
                            Dis::All => true,
                            Dis::Ids(v) => v.contains(&c.id),
                        }
                });
                let at = || {
                    object(vec![
                        ("region", Json::U64(region)),
                        ("clause", Json::U64(c.id)),
                    ])
                };
                if !required.contains(&c.id) {
                    gap = true;
                    reasons.push(reason("required_mismatch", at()));
                }
                if !claimed {
                    gap = true;
                    reasons.push(reason("cover_gap", at()));
                }
                cover_rows.push(object(vec![
                    ("region", Json::U64(region)),
                    ("clause", Json::U64(c.id)),
                    ("claimed_ok", Json::Bool(claimed)),
                    ("confirmed_ok", Json::Bool(false)),
                    ("nodes_confirmed", Json::U64(0)),
                    ("kind", Json::Str("unavailable".into())),
                ]));
            }
        }
        rep.insert("cover".into(), Json::Arr(cover_rows));
        let n_total = bundle.nodes.len();
        let n_opaque = bundle
            .nodes
            .iter()
            .filter(|n| matches!(n.close, Close::Opaque))
            .count();
        rep.insert(
            "nodes".into(),
            object(vec![
                ("total", Json::U64(n_total as u64)),
                ("confirmed", Json::U64(0)),
                ("unconfirmed", Json::U64(n_total as u64)),
                ("opaque", Json::U64(n_opaque as u64)),
                ("hints_ignored", Json::U64(bundle.hints_ignored as u64)),
                ("hints_invalid", Json::U64(bundle.hints_invalid as u64)),
            ]),
        );
        let verdict = if gap {
            Verdict::Invalid
        } else {
            reasons.push(reason(
                "unconfirmed",
                text_detail("the claims cover, but nothing in an unavailable bundle is checkable"),
            ));
            Verdict::Unconfirmed
        };
        reasons.sort_by(|a, b| a.code.cmp(&b.code));
        let cover_tok = if gap { "gap" } else { "unconfirmed" };
        return finish(
            rep,
            verdict,
            reasons,
            timing,
            (n_total, 0, n_total),
            cover_tok,
            false,
        );
    }
    let (Some(onnx_bytes), Some(vnn_bytes)) = (onnx_bytes, vnn_bytes) else {
        return Err("an unavailable input reached the checkable path".to_string());
    };
    let net = match parse_model(&onnx_bytes) {
        Ok(m) => match lower(&m) {
            Ok(n) => n,
            Err(e) => {
                let e = unsupported("onnx_lowering", e.text().to_string());
                return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
            }
        },
        Err(e) => {
            let e = unsupported("onnx_decode", e.0);
            return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
        }
    };
    if vnn_path.extension().is_some_and(|e| e == "gz") {
        let e = unsupported("vnnlib_form", "gzip-compressed VNN-LIB");
        return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
    }
    let spec = match std::str::from_utf8(&vnn_bytes)
        .map_err(|_| vnnlib::SpecError("not UTF-8".into()))
        .and_then(vnnlib::parse)
    {
        Ok(s) => s,
        Err(e) => {
            let e = unsupported("vnnlib_form", e.0);
            return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
        }
    };
    if spec.n_x != net.n_in() || spec.n_y != net.n_out() {
        let e = unsupported(
            "shape_mismatch",
            format!(
                "vnnlib declares {} inputs / {} outputs, the network has {} / {}",
                spec.n_x,
                spec.n_y,
                net.n_in(),
                net.n_out()
            ),
        );
        return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
    }
    timing.insert("load_s".into(), fnum(t0.elapsed().as_secs_f64()));
    // Memory governor.
    let threads = rayon::current_num_threads().max(1);
    let total: usize = net.tensors.iter().map(|t| t.size).sum();
    let widest = net.tensors.iter().map(|t| t.size).max().unwrap_or(1);
    let mem_bytes = (args.mem_gib.max(0.0) * (1u64 << 30) as f64) as usize;
    let per_row = 16usize.saturating_mul(total.max(1));
    let per_thread = mem_bytes / threads;
    if 16usize.saturating_mul(widest) > per_thread {
        let e = unsupported("memory", "a single row does not fit the --mem-gib budget");
        return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
    }
    let chunk_rows = (per_thread / per_row).clamp(1, 4096);
    // Network / spec descriptors.
    let mut ops: BTreeMap<&str, u64> = BTreeMap::new();
    for nd in &net.nodes {
        *ops.entry(nd.op.kind()).or_insert(0) += 1;
    }
    let mut ops_m = Map::new();
    for (k, v) in ops {
        ops_m.insert(k.to_string(), Json::U64(v));
    }
    // Structural validation.
    let mut reasons: Vec<Reason> = Vec::new();
    let mut invalid = false;
    let mut site_of: BTreeMap<u64, usize> = BTreeMap::new();
    let mut bundle_id_of: BTreeMap<usize, u64> = BTreeMap::new();
    for s in &bundle.sites {
        let Some(&t) = net.by_name.get(&s.name) else {
            let e = malformed(Malformed {
                code: "unknown_site_tensor".into(),
                detail: format!("site `{}` names no tensor of the network", s.name),
            });
            return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
        };
        if !net.is_site(t) {
            let e = malformed(Malformed {
                code: "not_a_relu_input".into(),
                detail: format!("site `{}` is not the input of a Relu node", s.name),
            });
            return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
        }
        if net.tensors[t].size as u64 != s.size {
            let e = malformed(Malformed {
                code: "site_size".into(),
                detail: format!(
                    "site `{}` size {} != {}",
                    s.name, s.size, net.tensors[t].size
                ),
            });
            return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
        }
        site_of.insert(s.id, t);
        bundle_id_of.insert(t, s.id);
    }
    // Clause matching: producer (region, id) -> checker k.
    let mut clause_k: BTreeMap<(u64, u64), usize> = BTreeMap::new();
    let mut matched: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (&region, list) in &bundle.clauses {
        let r = usize::try_from(region)
            .ok()
            .filter(|&r| r < spec.regions.len());
        for c in list {
            let mut atoms = c.atoms.clone();
            atoms.sort();
            atoms.dedup();
            let canon = atoms.join("&");
            let hit = r.and_then(|r| {
                spec.regions[r]
                    .disjuncts
                    .iter()
                    .position(|d: &Disjunct| d.canonical == canon)
                    .map(|k| (r, k))
            });
            match hit {
                Some((r, k)) => {
                    clause_k.insert((region, c.id), k);
                    matched.insert((r, k));
                }
                None => {
                    invalid = true;
                    reasons.push(reason(
                        "clause_mismatch",
                        object(vec![
                            ("region", Json::U64(region)),
                            ("clause", Json::U64(c.id)),
                            ("canonical", Json::Str(canon)),
                        ]),
                    ));
                }
            }
        }
    }
    let required: BTreeSet<u64> = bundle.verdict.required.iter().copied().collect();
    for (r, region) in spec.regions.iter().enumerate() {
        for (k, d) in region.disjuncts.iter().enumerate() {
            let producer_ids: Vec<u64> = clause_k
                .iter()
                .filter(|((pr, _), kk)| *pr == r as u64 && **kk == k)
                .map(|((_, id), _)| *id)
                .collect();
            if !matched.contains(&(r, k)) || !producer_ids.iter().any(|id| required.contains(id)) {
                invalid = true;
                reasons.push(reason(
                    "required_mismatch",
                    object(vec![
                        ("region", Json::U64(r as u64)),
                        ("disjunct", Json::U64(k as u64)),
                        ("canonical", Json::Str(d.canonical.clone())),
                    ]),
                ));
            }
        }
    }
    // α tables.
    let mut alphas: BTreeMap<u64, AlphaMap> = BTreeMap::new();
    let mut hints_invalid = bundle.hints_invalid;
    for a in &bundle.alphas {
        let Some(&t) = site_of.get(&a.tensor) else {
            hints_invalid += 1;
            continue;
        };
        let size = net.tensors[t].size;
        let table = alphas.entry(a.id).or_default().entry(t).or_default();
        for (&i, &v) in a.idx.iter().zip(&a.val) {
            if (i as usize) < size {
                table.insert(i as usize, v);
            } else {
                hints_invalid += 1;
            }
        }
    }
    // B2.3: frozen-dual slope tables, per hint state (site ids resolved).
    let mut fd_lambda: BTreeMap<u64, LambdaMap> = BTreeMap::new();
    for l in &bundle.fd_lambdas {
        let Some(&t) = site_of.get(&l.tensor) else {
            hints_invalid += 1;
            continue;
        };
        let size = net.tensors[t].size;
        let table = fd_lambda.entry(l.id).or_default().entry(t).or_default();
        for (&i, &v) in l.idx.iter().zip(&l.val) {
            if (i as usize) < size {
                table.insert(i as usize, v);
            } else {
                hints_invalid += 1;
            }
        }
    }
    // Resolve nodes.
    let mut rnodes: Vec<RNode<'_>> = Vec::with_capacity(bundle.nodes.len());
    for nd in &bundle.nodes {
        let Some(list) = bundle.clauses.get(&nd.region) else {
            let e = malformed(Malformed {
                code: "unknown_region".into(),
                detail: format!(
                    "node {} names region {} with no clauses record",
                    nd.id, nd.region
                ),
            });
            return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
        };
        let region_ok = usize::try_from(nd.region)
            .ok()
            .filter(|&r| r < spec.regions.len());
        let mut lits = Vec::with_capacity(nd.lits.len());
        let mut seen: BTreeMap<(u8, u64, u64), bool> = BTreeMap::new();
        for l in &nd.lits {
            let r = match *l {
                Lit::Relu { tensor, index, pos } => {
                    let Some(&t) = site_of.get(&tensor) else {
                        let e = malformed(Malformed {
                            code: "unknown_site_id".into(),
                            detail: format!("node {} names site id {tensor}", nd.id),
                        });
                        return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
                    };
                    if index as usize >= net.tensors[t].size {
                        let e = malformed(Malformed {
                            code: "literal_index".into(),
                            detail: format!("node {} index {index} out of range", nd.id),
                        });
                        return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
                    }
                    RLit::Relu {
                        t,
                        i: index as usize,
                        pos,
                    }
                }
                Lit::X { index, t, pos } => {
                    if index as usize >= net.n_in() {
                        let e = malformed(Malformed {
                            code: "literal_index".into(),
                            detail: format!("node {} input index {index} out of range", nd.id),
                        });
                        return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
                    }
                    RLit::X {
                        i: index as usize,
                        t,
                        pos,
                    }
                }
            };
            let (key, side) = r.key();
            if let Some(prev) = seen.insert(key, side) {
                if prev != side {
                    let e = malformed(Malformed {
                        code: "both_sides".into(),
                        detail: format!("node {} carries both sides of one site", nd.id),
                    });
                    return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
                }
            }
            lits.push(r);
        }
        let claims = match &nd.dis {
            Dis::All => None,
            Dis::Ids(ids) => {
                let mut ks = Vec::new();
                for id in ids {
                    if !list.iter().any(|c| c.id == *id) {
                        let e = malformed(Malformed {
                            code: "unknown_clause_id".into(),
                            detail: format!("node {} claims clause {id}", nd.id),
                        });
                        return finish(rep, e.verdict, e.reasons, timing, (0, 0, 0), "none", false);
                    }
                    if let Some(&k) = clause_k.get(&(nd.region, *id)) {
                        ks.push(k);
                    }
                }
                ks.sort_unstable();
                ks.dedup();
                Some(ks)
            }
        };
        let Some(region) = region_ok else {
            // Its clauses already failed to match (invalid); nothing to check.
            continue;
        };
        let empty_site = nd
            .hint
            .empty_site
            .and_then(|(t, i)| site_of.get(&t).map(|&tt| (tt, i as usize)))
            .filter(|&(t, i)| i < net.tensors[t].size);
        let beta = nd.hint.beta.clone().filter(|b| {
            let ok = b.len() == nd.lits.len();
            if !ok {
                hints_invalid += 1;
            }
            ok
        });
        let alpha = nd.hint.alpha.filter(|id| alphas.contains_key(id));
        if nd.hint.alpha.is_some() && alpha.is_none() {
            hints_invalid += 1;
        }
        let fd = nd.hint.fd.as_ref().filter(|h| {
            let ok = h.l0.len() == nd.lits.len();
            if !ok {
                hints_invalid += 1;
            }
            ok
        });
        rnodes.push(RNode {
            rec: nd,
            region,
            lits,
            claims,
            beta,
            alpha,
            empty_site,
            fd,
        });
    }
    // Root bounds per region.
    let t_root = Instant::now();
    let mut roots: Vec<RegionRoot> = spec
        .regions
        .iter()
        .map(|r| root_bounds(&net, r, chunk_rows, args.refine, deadline))
        .collect();
    if args.fault == Some(FaultInjection::InflateRootLower) {
        if let Some(t) = (0..net.tensors.len()).find(|&t| net.is_site(t)) {
            for root in &mut roots {
                for v in &mut root.bounds.lo[t] {
                    *v += 1000.0;
                }
            }
        }
    }
    let unstable_root: usize = roots
        .first()
        .map(|r| {
            (0..net.tensors.len())
                .filter(|&t| net.is_site(t))
                .map(|t| {
                    r.bounds.lo[t]
                        .iter()
                        .zip(&r.bounds.hi[t])
                        .filter(|(l, h)| **l < 0.0 && **h > 0.0)
                        .count()
                })
                .sum()
        })
        .unwrap_or(0);
    timing.insert("root_s".into(), fnum(t_root.elapsed().as_secs_f64()));
    rep.insert(
        "network".into(),
        object(vec![
            ("ops", Json::Obj(ops_m)),
            ("relu_neurons", Json::U64(net.relu_neurons() as u64)),
            ("unstable_root", Json::U64(unstable_root as u64)),
        ]),
    );
    let mut dis_j = Vec::new();
    for (r, region) in spec.regions.iter().enumerate() {
        for (k, d) in region.disjuncts.iter().enumerate() {
            dis_j.push(object(vec![
                ("region", Json::U64(r as u64)),
                ("disjunct", Json::U64(k as u64)),
                (
                    "fp",
                    Json::Str(sha256_hex(d.canonical.as_bytes())[..16].to_string()),
                ),
                ("atoms", Json::U64(d.atoms.len() as u64)),
            ]));
        }
    }
    rep.insert(
        "spec".into(),
        object(vec![
            ("regions", Json::U64(spec.regions.len() as u64)),
            ("disjuncts", Json::Arr(dis_j)),
            ("producer_match", Json::Bool(!invalid)),
        ]),
    );
    // B2.3: the checker's own frozen-dual states.
    let t_fd = Instant::now();
    let (mut fd_built, fd_rows) = build_fd_states(
        &net,
        &spec,
        &roots,
        &rnodes,
        &fd_lambda,
        &bundle.fd_states,
        mem_bytes,
        deadline,
    );
    if args.fault == Some(FaultInjection::InflateFdMargin) {
        for st in fd_built.values_mut() {
            for m in st.margins.values_mut() {
                m.c_lo += 1000.0;
                m.c_hi += 1000.0;
            }
        }
    }
    timing.insert("fd_s".into(), fnum(t_fd.elapsed().as_secs_f64()));
    // Self-containment (1): root bounds contain evaluated pre-activations.
    let mut fault: Option<String> = None;
    for (r, region) in spec.regions.iter().enumerate() {
        let root = &roots[r];
        let mut rng = Rng::new(0xC0FF_EE00 + r as u64);
        for _ in 0..16 {
            let p: Vec<f64> = region
                .lo_out
                .iter()
                .zip(&region.hi_out)
                .map(|(a, b)| a + rng.next_f64() * (b - a))
                .collect();
            let Some(x) = inward(region, &p) else {
                break;
            };
            let b = eval_point(&net, &x);
            for t in 0..net.tensors.len() {
                if !net.is_site(t) {
                    continue;
                }
                for i in 0..net.tensors[t].size {
                    if b.hi[t][i] < root.bounds.lo[t][i] || b.lo[t][i] > root.bounds.hi[t][i] {
                        fault.get_or_insert_with(|| {
                            format!(
                                "root bound of `{}`[{i}] excludes the evaluated value at a sampled point",
                                net.tensors[t].name
                            )
                        });
                    }
                }
            }
        }
    }
    // Self-containment (2), B2.3: every built FD state contains the network.
    for (&(r, id), st) in &fd_built {
        if fault.is_none() {
            fault = fd_containment(&net, &spec.regions[r], st, 0xFD00_0000 ^ id ^ (r as u64));
        }
    }
    // Per-node discharge ladder (parallel across nodes).
    let env = Env {
        net: &net,
        spec: &spec,
        roots: &roots,
        alphas: &alphas,
        chunk_rows,
        refine: args.refine,
        deadline,
        refute_points: args.refute_points,
        fd: &fd_built,
        bundle_sha256: &bundle_sha,
        fault: args.fault,
    };
    let t_nodes = Instant::now();
    let done = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let total_nodes = rnodes.len();
    let hb = {
        let done = Arc::clone(&done);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let start = Instant::now();
            let mut last = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                if last.elapsed() >= Duration::from_secs(30) {
                    last = Instant::now();
                    eprintln!(
                        "[proof-check] progress nodes_done={}/{total_nodes} wall_s={:.1}",
                        done.load(Ordering::Relaxed),
                        start.elapsed().as_secs_f64()
                    );
                }
            }
        })
    };
    let outs: Vec<NodeOut> = rnodes
        .par_iter()
        .map(|n| {
            let o = process_node(&env, n);
            done.fetch_add(1, Ordering::Relaxed);
            o
        })
        .collect();
    stop.store(true, Ordering::Relaxed);
    let _ = hb.join();
    timing.insert("nodes_s".into(), fnum(t_nodes.elapsed().as_secs_f64()));
    // Global refutation points.
    let mut refuted: Option<(usize, usize, Vec<f64>, Vec<(f64, f64)>)> =
        outs.iter().find_map(|o| o.refuted.clone());
    if refuted.is_none() {
        'regions: for (r, region) in spec.regions.iter().enumerate() {
            let mut rng = Rng::new(0x9_1A57 + r as u64);
            let mut pts: Vec<Vec<f64>> = vec![region
                .lo_out
                .iter()
                .zip(&region.hi_out)
                .map(|(a, b)| 0.5 * a + 0.5 * b)
                .collect()];
            for _ in 0..16 {
                pts.push(
                    region
                        .lo_out
                        .iter()
                        .zip(&region.hi_out)
                        .map(|(a, b)| a + rng.next_f64() * (b - a))
                        .collect(),
                );
            }
            for p in pts {
                let Some(x) = inward(region, &p) else {
                    continue;
                };
                if let Some((k, ys)) = refutes(&net, region, &x) {
                    refuted = Some((r, k, x, ys));
                    break 'regions;
                }
            }
        }
    }
    if fault.is_none() {
        fault = outs.iter().find_map(|o| o.fault.clone());
    }
    // Covers.
    let mut cover_rows = Vec::new();
    let mut claimed_gap = false;
    let mut confirmed_gap = false;
    let mut budget_hit = false;
    for (r, region) in spec.regions.iter().enumerate() {
        // B2.2: a region whose nodes carry input-threshold literals is
        // checked GEOMETRICALLY (box cover with Relu literals inside boxes),
        // so a tiling that is not a split tree still covers; a region of
        // Relu literals only keeps the literal case-split check.
        let boxed = rnodes
            .iter()
            .any(|n| n.region == r && n.lits.iter().any(|l| matches!(l, RLit::X { .. })));
        for k in 0..region.disjuncts.len() {
            let mut claimed: Vec<Vec<CoverLit>> = Vec::new();
            let mut confirmed: Vec<Vec<CoverLit>> = Vec::new();
            let mut claimed_b: Vec<BoxMember> = Vec::new();
            let mut confirmed_b: Vec<BoxMember> = Vec::new();
            for (n, o) in rnodes.iter().zip(&outs) {
                if n.region != r || matches!(n.rec.close, Close::Subtree) {
                    continue;
                }
                let claims_k = match &n.claims {
                    None => true,
                    Some(v) => v.contains(&k),
                };
                let is_confirmed = o.empty || o.confirmed.contains_key(&k);
                if boxed {
                    // An empty node box is an empty region: it covers no
                    // point, and no point needs it.
                    let Some((lo, hi)) = node_box(region, &n.lits) else {
                        continue;
                    };
                    let mut relu: Vec<CoverLit> = n
                        .lits
                        .iter()
                        .filter(|l| matches!(l, RLit::Relu { .. }))
                        .map(|l| l.key())
                        .collect();
                    relu.sort_unstable();
                    relu.dedup();
                    let m = BoxMember { lo, hi, relu };
                    if claims_k {
                        claimed_b.push(m.clone());
                    }
                    if is_confirmed {
                        confirmed_b.push(m);
                    }
                    continue;
                }
                let mut key: Vec<CoverLit> = n.lits.iter().map(|l| l.key()).collect();
                key.sort_unstable();
                key.dedup();
                if claims_k {
                    claimed.push(key.clone());
                }
                if is_confirmed {
                    confirmed.push(key);
                }
            }
            let relu_json = |w: &[CoverLit]| -> Vec<Json> {
                w.iter()
                    .map(|&((kind, a, b), side)| {
                        if kind == 0 {
                            Json::Arr(vec![
                                Json::U64(bundle_id_of.get(&(a as usize)).copied().unwrap_or(a)),
                                Json::U64(b),
                                Json::U64(u64::from(side)),
                            ])
                        } else {
                            Json::Arr(vec![
                                Json::Str("x".into()),
                                Json::U64(a),
                                Json::Str(format!("{:?}", f64::from_bits(b))),
                                Json::U64(u64::from(side)),
                            ])
                        }
                    })
                    .collect()
            };
            let (c1, c2) = if boxed {
                let map = |res: BoxCoverResult| match res {
                    BoxCoverResult::Covered => CoverResult::Covered,
                    BoxCoverResult::Budget => CoverResult::Budget,
                    BoxCoverResult::Gap { lo, hi, path } => {
                        // The witness as literals: the gap cell's Relu path
                        // and every face of its box inside the region box.
                        let mut w = path;
                        for d in 0..lo.len() {
                            if lo[d] > region.lo_out[d] {
                                w.push(((1, d as u64, lo[d].to_bits()), true));
                            }
                            if hi[d] < region.hi_out[d] {
                                w.push(((1, d as u64, hi[d].to_bits()), false));
                            }
                        }
                        CoverResult::Gap(w)
                    }
                };
                let c1 = map(check_box_cover(
                    &region.lo_out,
                    &region.hi_out,
                    &claimed_b,
                    DEFAULT_BUDGET,
                ));
                let c2 = match &c1 {
                    CoverResult::Covered => map(check_box_cover(
                        &region.lo_out,
                        &region.hi_out,
                        &confirmed_b,
                        DEFAULT_BUDGET,
                    )),
                    _ => CoverResult::Budget,
                };
                (c1, c2)
            } else {
                let c1 = check_cover(&claimed, DEFAULT_BUDGET);
                let c2 = match &c1 {
                    CoverResult::Covered => check_cover(&confirmed, DEFAULT_BUDGET),
                    _ => CoverResult::Budget,
                };
                (c1, c2)
            };
            let claimed_ok = matches!(c1, CoverResult::Covered);
            let confirmed_ok = claimed_ok && matches!(c2, CoverResult::Covered);
            match &c1 {
                CoverResult::Gap(w) => {
                    claimed_gap = true;
                    invalid = true;
                    let witness: Vec<Json> = relu_json(w);
                    reasons.push(reason(
                        "cover_gap",
                        object(vec![
                            ("region", Json::U64(r as u64)),
                            ("disjunct", Json::U64(k as u64)),
                            ("witness", Json::Arr(witness)),
                        ]),
                    ));
                }
                CoverResult::Budget => {
                    budget_hit = true;
                    reasons.push(reason(
                        "cover_budget",
                        object(vec![
                            ("region", Json::U64(r as u64)),
                            ("disjunct", Json::U64(k as u64)),
                        ]),
                    ));
                }
                CoverResult::Covered => {}
            }
            if claimed_ok && !confirmed_ok {
                confirmed_gap = true;
                if matches!(c2, CoverResult::Budget) {
                    budget_hit = true;
                }
            }
            cover_rows.push(object(vec![
                ("region", Json::U64(r as u64)),
                ("disjunct", Json::U64(k as u64)),
                ("claimed_ok", Json::Bool(claimed_ok)),
                ("confirmed_ok", Json::Bool(confirmed_ok)),
                (
                    "nodes_confirmed",
                    Json::U64((confirmed.len() + confirmed_b.len()) as u64),
                ),
                (
                    "kind",
                    Json::Str(if boxed { "box" } else { "split" }.into()),
                ),
            ]));
        }
    }
    rep.insert("cover".into(), Json::Arr(cover_rows));
    // Node statistics.
    let n_total = rnodes.len();
    let n_empty = outs.iter().filter(|o| o.empty).count();
    let n_opaque = rnodes
        .iter()
        .filter(|n| matches!(n.rec.close, Close::Opaque))
        .count();
    let n_confirmed = outs
        .iter()
        .filter(|o| o.empty || (!o.confirmed.is_empty() && o.unconfirmed.is_empty()))
        .count();
    let n_unconfirmed = outs.iter().filter(|o| !o.unconfirmed.is_empty()).count();
    let fd_unconfirmed = rnodes
        .iter()
        .zip(&outs)
        .filter(|(n, o)| matches!(n.rec.close, Close::Fd) && !o.unconfirmed.is_empty())
        .count();
    rep.insert(
        "nodes".into(),
        object(vec![
            ("total", Json::U64(n_total as u64)),
            ("confirmed", Json::U64(n_confirmed as u64)),
            ("unconfirmed", Json::U64(n_unconfirmed as u64)),
            ("empty", Json::U64(n_empty as u64)),
            ("opaque", Json::U64(n_opaque as u64)),
            ("fd_unconfirmed", Json::U64(fd_unconfirmed as u64)),
            ("hints_ignored", Json::U64(bundle.hints_ignored as u64)),
            ("hints_invalid", Json::U64(hints_invalid as u64)),
        ]),
    );
    // B2.3: the frozen-dual rung.
    let fd_nodes = rnodes.iter().filter(|n| n.fd.is_some()).count();
    let fd_tried = outs.iter().filter(|o| o.fd_tried).count();
    let fd_confirmed: usize = outs.iter().map(|o| o.fd_confirmed.len()).sum();
    // B2.3: the per-node fd sample (self-containment (3)): which fd-confirmed
    // nodes were sampled, by which rule, and how many forward passes it cost.
    let mut fd_sampling = [0u64; 3];
    let (mut fd_points, mut fd_in_region) = (0u64, 0u64);
    for o in &outs {
        match o.fd_sample {
            Some(FdNodeSample::Shallow) => fd_sampling[0] += 1,
            Some(FdNodeSample::Hashed) => fd_sampling[1] += 1,
            Some(FdNodeSample::Skipped) => fd_sampling[2] += 1,
            None => {}
        }
        fd_points += o.fd_points;
        fd_in_region += o.fd_in_region;
    }
    // B2.3: which producer routes contributed nodes (`src`, diagnostic).
    let mut routes: BTreeMap<String, u64> = BTreeMap::new();
    for n in &bundle.nodes {
        *routes.entry(n.src.clone()).or_insert(0) += 1;
    }
    let mut routes_m = Map::new();
    for (k, v) in routes {
        routes_m.insert(k, Json::U64(v));
    }
    rep.insert("routes".into(), Json::Obj(routes_m));
    rep.insert(
        "fd".into(),
        object(vec![
            ("states_hinted", Json::U64(bundle.fd_states.len() as u64)),
            ("states_built", Json::U64(fd_built.len() as u64)),
            ("nodes_hinted", Json::U64(fd_nodes as u64)),
            ("nodes_evaluated", Json::U64(fd_tried as u64)),
            ("discharges_confirmed", Json::U64(fd_confirmed as u64)),
            (
                "node_sampling",
                object(vec![
                    ("all_depth", Json::U64(FD_NODE_SAMPLE_ALL_DEPTH as u64)),
                    ("one_in", Json::U64(FD_NODE_SAMPLE_ONE_IN)),
                    ("nodes", Json::U64(fd_sampling.iter().sum())),
                    ("shallow", Json::U64(fd_sampling[0])),
                    ("hashed", Json::U64(fd_sampling[1])),
                    ("skipped", Json::U64(fd_sampling[2])),
                    ("points", Json::U64(fd_points)),
                    ("in_region", Json::U64(fd_in_region)),
                ]),
            ),
            ("states", Json::Arr(fd_rows)),
        ]),
    );
    let mut ladder: BTreeMap<&str, u64> = BTreeMap::new();
    for step in ["L1", "L2", "L3", "lift", "clip", "fd"] {
        ladder.insert(step, 0);
    }
    let mut unconf = Vec::new();
    let mut chk_margins = Vec::new();
    let mut prod_margins = Vec::new();
    let mut shortfall = Vec::new();
    let mut gains = Vec::new();
    for (n, o) in rnodes.iter().zip(&outs) {
        for (l, step) in o.confirmed.values() {
            *ladder.entry(step).or_insert(0) += 1;
            chk_margins.push(*l);
        }
        if let Some(m) = n.rec.hint.margin {
            prod_margins.push(m);
        }
        gains.extend(o.hint_gain.iter().copied());
        for (&k, &(l, why)) in &o.unconfirmed {
            let pm = n.rec.hint.margin;
            if let (Some(p), true) = (pm, l.is_finite()) {
                shortfall.push(p - l);
            }
            unconf.push(object(vec![
                ("node", Json::U64(n.rec.id)),
                ("clause", Json::U64(k as u64)),
                ("margin_chk", fnum(l)),
                ("margin_producer", onum(pm)),
                ("reason", Json::Str(why.into())),
            ]));
        }
    }
    let mut lm = Map::new();
    for (k, v) in ladder {
        lm.insert(k.to_string(), Json::U64(v));
    }
    rep.insert("ladder".into(), Json::Obj(lm));
    rep.insert("unconfirmed".into(), Json::Arr(unconf));
    let min_of = |v: &[f64]| {
        v.iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(None, |a: Option<f64>, x| Some(a.map_or(x, |a| a.min(x))))
    };
    gains.sort_by(f64::total_cmp);
    let gain_median = (!gains.is_empty()).then(|| gains[gains.len() / 2]);
    rep.insert(
        "margins".into(),
        object(vec![
            ("min_chk", onum(min_of(&chk_margins))),
            ("min_producer", onum(min_of(&prod_margins))),
            ("hist_chk", decade_hist(&chk_margins)),
            ("shortfall_hist", decade_hist(&shortfall)),
            ("hint_gain_median", onum(gain_median)),
        ]),
    );
    let cex_json = match &refuted {
        None => Json::Null,
        Some((r, k, x, ys)) => {
            let mut text = String::new();
            for (i, v) in x.iter().enumerate() {
                text.push_str(&format!("(X_{i} {v:?})\n"));
            }
            for (j, (l, h)) in ys.iter().enumerate() {
                text.push_str(&format!("(Y_{j} {:?})\n", 0.5 * l + 0.5 * h));
            }
            object(vec![
                ("region", Json::U64(*r as u64)),
                ("disjunct", Json::U64(*k as u64)),
                ("x", Json::Arr(x.iter().map(|v| fnum(*v)).collect())),
                (
                    "y",
                    Json::Arr(
                        ys.iter()
                            .map(|(l, h)| Json::Arr(vec![fnum(*l), fnum(*h)]))
                            .collect(),
                    ),
                ),
                ("vnncomp", Json::Str(text)),
            ])
        }
    };
    rep.insert(
        "refutation".into(),
        object(vec![
            ("points", Json::U64(args.refute_points as u64)),
            ("counterexample", cex_json),
        ]),
    );
    // Verdict by precedence.
    let mut verdict = Verdict::Verified;
    if confirmed_gap || budget_hit {
        verdict = Verdict::Unconfirmed;
        if !claimed_gap {
            reasons.push(reason(
                "unconfirmed",
                text_detail("the claims cover, but not every needed discharge was confirmed"),
            ));
        }
    }
    if invalid {
        verdict = Verdict::Invalid;
    }
    if let Some((r, k, _, _)) = &refuted {
        verdict = Verdict::Refuted;
        reasons.push(reason(
            "counterexample",
            object(vec![
                ("region", Json::U64(*r as u64)),
                ("disjunct", Json::U64(*k as u64)),
            ]),
        ));
    }
    if let Some(f) = &fault {
        verdict = Verdict::CheckerFault;
        reasons.push(reason("self_containment", text_detail(f.clone())));
    }
    let cover_tok = if claimed_gap {
        "gap"
    } else if confirmed_gap || budget_hit {
        "unconfirmed"
    } else {
        "ok"
    };
    // Deterministic order of reasons.
    reasons.sort_by(|a, b| a.code.cmp(&b.code));
    finish(
        rep,
        verdict,
        reasons,
        timing,
        (n_total, n_confirmed, n_unconfirmed),
        cover_tok,
        refuted.is_some(),
    )
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M7 — BATCHED α/β search over graph β-CROWN domains
//! (ENGINE_PROGRAM §2 M7, `#graph-lane-search`), UNTRUSTED, on the CPU.
//!
//! # What it searches
//!
//! A batch shares one [`GraphSearchNet`] (the graph lane's self-checked V2
//! copy), one objective and the lane's certified boxes as FROZEN reference
//! boxes. Domain `k` is the input box intersected with its ReLU split literals
//! `s z_j ≥ 0` (`s = +1` active, `−1` inactive), optionally with its own
//! (tighter) reference boxes and input box. Per domain, abc's β-CROWN
//! bounding recipe over the frozen boxes:
//!
//! * every split neuron is PINNED to its exact piece (active: identity,
//!   inactive: zero) — its domain box is one-signed there;
//! * each literal carries a multiplier `β ≥ 0` and the Lagrangian term
//!   `−β s z_j` is added to `z_j`'s backward coefficient just before `z_j`'s
//!   producer propagates it (weak duality: `min f ≥ min (f − Σ β s z)` over
//!   the relaxation, since `s z ≥ 0` on the domain) — the M6 wave's
//!   convention, and the lane's (`GraphBetaEntry`: sign `+1` active);
//! * every other neuron unstable in the domain's boxes carries its own lower
//!   slope `α ∈ [0, 1]` (upper chord of the domain's box).
//!
//! The bound of a row is ONE backward pass (V2's
//! [`crate::graph_root::backward`] under the domain's [`View`]), and its
//! gradient the envelope rule at the
//! concretisation corner `x*` of the linearised forward walk: V2's slope and
//! plane gradients, plus `∂ lb / ∂ β = −s ẑ_j(x*)`. The loss is abc's (the
//! sum of the bounds of the rows not above their threshold), projected Adam
//! on `(α, β)` (α clamped into `[0, 1]`, β into `[0, ∞)`), lr decay 0.98 per
//! iteration, keep-best under the lane's ranking (hinge, then worst slack),
//! V2's early exits. With no split and no box of its own a domain IS the V2
//! root search, bit for bit
//! (`graph_bab_empty_domain_reproduces_the_root_search_bit_for_bit`).
//!
//! # Cost shape and R6
//!
//! One domain per worker at a time, its rows in ROW ORDER on that worker;
//! domains are claimed from an atomic counter and scattered back by index, so
//! every domain's hint depends only on its own inputs — not on the thread
//! count, the claim order or which domains share the batch
//! (`graph_bab_is_thread_count_independent`,
//! `graph_bab_domain_results_do_not_depend_on_the_batch`). The deadline only
//! decides whether a domain STARTS (a late one gets no hint) and stops a
//! started one at its best iterate. Every worker's working set is admitted
//! against the request's memory cap before anything runs.
//!
//! # Trust
//!
//! UNTRUSTED end to end: f32 in production, round to nearest, no rounding
//! charged, frozen boxes rounded to the search scalar. Only HINTS leave (the
//! slopes and the multipliers); the lane clamps them again at its ingest gate
//! (any `α ∈ [0, 1]` and any `β ≥ 0` is a valid relaxation of the domain),
//! certifies each domain ONCE with its own evaluator and keeps a row's bound
//! only when strictly better. A bug here costs a missed improvement, never a
//! verdict.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ny_core::search::{
    GraphAlphaHint, GraphBabOutcome, GraphBabRequest, GraphBabStats, GraphDomainHint,
    GraphDomainRequest, GraphDomainStats, GraphSearchNet, Refusal, TensorBox, Untrusted,
};

use crate::graph_root::{
    eval_row, finite_ordered_box, improves, planes, reference_box_ok, score, Adam, Alpha, GProgram,
    ReluSite, Score, View, ACTIVE, DEFAULT_LR, INACTIVE, LR_DECAY, PATIENCE,
};
use crate::real::Real;

/// The multiplier step when the request carries no usable one (abc's
/// `beta-crown.lr_beta` default).
pub(crate) const DEFAULT_LR_BETA: f64 = 0.05;

const ADAM_B1: f64 = 0.9;
const ADAM_B2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;

/// One domain, ready to search: its re-derived ReLU sites, input box, split
/// literals and searched rows.
pub(crate) struct DomainSetup<R> {
    /// The program's sites with statuses / chords / starting slopes from the
    /// domain's boxes, split neurons pinned.
    pub(crate) relu: Vec<ReluSite<R>>,
    pub(crate) lo: Vec<R>,
    pub(crate) hi: Vec<R>,
    /// `(pre-activation tensor, neuron, s)` per literal, request order.
    pub(crate) lits: Vec<(usize, u32, R)>,
    /// Starting multipliers, aligned with `lits`.
    pub(crate) beta0: Vec<R>,
    /// Searched rows (indices into the program's rows), ascending order kept
    /// as given.
    pub(crate) rows: Vec<usize>,
}

/// Why a domain got no hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DomainRefusal {
    /// A literal names an op that is not a ReLU site.
    SplitNotRelu,
    /// A literal's neuron is past its site's width.
    SplitNeuronRange,
    /// The same neuron is split on both sides.
    SplitContradiction,
    /// A domain box is malformed or duplicated.
    BadBox,
    /// The domain's input box is not a finite ordered box of the input width.
    BadInputBox,
    /// A searched row index is out of range or repeated.
    BadRows,
    /// An unstable neuron of the domain has a non-finite box side.
    NonFiniteUnstable,
}

/// Build domain `dom`'s setup over the lowered batch `prog` (`batch_box[t]` =
/// the index of tensor `t`'s box in `req.boxes`).
pub(crate) fn setup_domain<R: Real>(
    prog: &GProgram<R>,
    req: &GraphBabRequest,
    batch_box: &[Option<usize>],
    dom: &GraphDomainRequest,
) -> Result<DomainSetup<R>, DomainRefusal> {
    let n_in = prog.tsize[0];
    let (lo, hi): (&[f64], &[f64]) = if dom.lo.is_empty() && dom.hi.is_empty() {
        (&req.lo, &req.hi)
    } else if finite_ordered_box(&dom.lo, &dom.hi, n_in) {
        (&dom.lo, &dom.hi)
    } else {
        return Err(DomainRefusal::BadInputBox);
    };
    let mut own: Vec<Option<&TensorBox>> = vec![None; prog.tsize.len()];
    for b in &dom.boxes {
        if !reference_box_ok(b, &prog.tsize) || own[b.tensor].is_some() {
            return Err(DomainRefusal::BadBox);
        }
        own[b.tensor] = Some(b);
    }
    let mut relu = Vec::with_capacity(prog.relu.len());
    for s in &prog.relu {
        let (l, u): (&[f64], &[f64]) = match own[s.input] {
            Some(b) => (&b.l, &b.u),
            None => {
                let Some(i) = batch_box[s.input] else {
                    // `lower_parts` required a batch box for every ReLU input.
                    return Err(DomainRefusal::BadBox);
                };
                (&req.boxes[i].l, &req.boxes[i].u)
            }
        };
        let site = ReluSite::<R>::from_box(s.op, s.input, l, u, &dom.init_relu)
            .map_err(|_| DomainRefusal::NonFiniteUnstable)?;
        relu.push(site);
    }
    let mut lits = Vec::with_capacity(dom.splits.len());
    let mut seen: Vec<(usize, u32, bool)> = Vec::with_capacity(dom.splits.len());
    for split in &dom.splits {
        let Some(k) = relu.iter().position(|s| s.op == split.op) else {
            return Err(DomainRefusal::SplitNotRelu);
        };
        let j = split.neuron as usize;
        if j >= relu[k].status.len() {
            return Err(DomainRefusal::SplitNeuronRange);
        }
        if seen
            .iter()
            .any(|&(kk, jj, a)| kk == k && jj == split.neuron && a != split.active)
        {
            return Err(DomainRefusal::SplitContradiction);
        }
        seen.push((k, split.neuron, split.active));
        relu[k].status[j] = if split.active { ACTIVE } else { INACTIVE };
        let s = if split.active { R::ONE } else { -R::ONE };
        lits.push((relu[k].input, split.neuron, s));
    }
    let beta0 = (0..lits.len())
        .map(|i| {
            let b = dom.init_beta.get(i).map_or(0.0, |&b| f64::from(b));
            R::from_f64(if b.is_finite() && b > 0.0 { b } else { 0.0 })
        })
        .collect();
    let n_rows = prog.rows.len();
    let rows: Vec<usize> = if dom.rows.is_empty() {
        (0..n_rows).collect()
    } else {
        let mut sorted = dom.rows.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != dom.rows.len() || sorted.last().is_some_and(|&r| r >= n_rows) {
            return Err(DomainRefusal::BadRows);
        }
        dom.rows.clone()
    };
    let cvt = |v: &[f64]| -> Vec<R> { v.iter().map(|&x| R::from_f64(x)).collect() };
    Ok(DomainSetup {
        relu,
        lo: cvt(lo),
        hi: cvt(hi),
        lits,
        beta0,
        rows,
    })
}

/// One full pass of a domain: its rows' bounds (in `dom.rows` order) and the
/// ROW-ORDER sums of the slope and multiplier gradients over the rows not
/// above their threshold.
pub(crate) struct DomainPass<R> {
    pub(crate) bounds: Vec<f64>,
    pub(crate) grad: Alpha<R>,
    pub(crate) gbeta: Vec<f64>,
}

/// The Lagrangian injections of `dom` at multipliers `beta`: coefficient
/// `−β s` on each literal's pre-activation neuron.
pub(crate) fn injections<R: Real>(dom: &DomainSetup<R>, beta: &[R]) -> Vec<(usize, u32, R)> {
    dom.lits
        .iter()
        .zip(beta)
        .map(|(&(t, j, s), &b)| (t, j, -(s * b)))
        .collect()
}

/// Evaluate every searched row of `dom` at `(alpha, beta)`.
pub(crate) fn domain_pass<R: Real>(
    prog: &GProgram<R>,
    dom: &DomainSetup<R>,
    alpha: &Alpha<R>,
    beta: &[R],
) -> DomainPass<R> {
    let inject = injections(dom, beta);
    let view = View {
        relu: &dom.relu,
        lo: &dom.lo,
        hi: &dom.hi,
        inject: &inject,
    };
    let pl = planes(prog, alpha);
    let mut bounds = Vec::with_capacity(dom.rows.len());
    let mut grad = alpha.zeros_like();
    let mut gbeta = vec![0.0f64; dom.lits.len()];
    for &r in &dom.rows {
        let out = eval_row(prog, &view, alpha, &pl, r);
        bounds.push(out.bound);
        let slack = out.bound - prog.thr[r];
        if slack.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            if let Some(g) = out.grad.as_ref() {
                grad.add_finite(g);
            }
            for ((gb, &(_, _, s)), &z) in gbeta.iter_mut().zip(&dom.lits).zip(&out.inject_grad) {
                // ∂ lb / ∂ β = (∂ lb / ∂ c) · (∂ c / ∂ β) = ẑ · (−s).
                let d = -(s.to_f64()) * z;
                if d.is_finite() {
                    *gb += d;
                }
            }
        }
    }
    DomainPass {
        bounds,
        grad,
        gbeta,
    }
}

/// Projected Adam ASCENT on the multipliers (`β ≥ 0`).
struct BetaAdam {
    m: Vec<f64>,
    v: Vec<f64>,
}

impl BetaAdam {
    fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    fn step<R: Real>(&mut self, beta: &mut [R], grad: &[f64], lr: f64, t: usize) {
        let t = i32::try_from(t.max(1)).unwrap_or(i32::MAX);
        let bc1 = 1.0 - ADAM_B1.powi(t);
        let bc2 = 1.0 - ADAM_B2.powi(t);
        for (k, (b, &g)) in beta.iter_mut().zip(grad).enumerate() {
            let g = if g.is_finite() { g } else { 0.0 };
            self.m[k] = ADAM_B1 * self.m[k] + (1.0 - ADAM_B1) * g;
            self.v[k] = ADAM_B2 * self.v[k] + (1.0 - ADAM_B2) * g * g;
            let step = lr * (self.m[k] / bc1) / ((self.v[k] / bc2).sqrt() + ADAM_EPS);
            let next = (b.to_f64() + step).max(0.0);
            if next.is_finite() {
                *b = R::from_f64(next);
            }
        }
    }
}

/// Search controls of one batch.
pub(crate) struct BabConfig {
    pub(crate) iters: usize,
    pub(crate) lr_alpha: f64,
    pub(crate) lr_beta: f64,
    pub(crate) deadline: Option<Instant>,
}

/// What [`search_domain`] found.
pub(crate) struct DomainResult<R> {
    /// The returned parameters in the search scalar (tests compare them).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) best_alpha: Alpha<R>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) best_beta: Vec<R>,
    pub(crate) hint: GraphDomainHint,
    pub(crate) iters: usize,
    pub(crate) evals: usize,
    pub(crate) init: Score,
    pub(crate) best_score: Score,
    pub(crate) stop: &'static str,
}

/// Search one domain from its starting parameters (V2's loop, with the
/// multipliers stepped alongside the slopes).
pub(crate) fn search_domain<R: Real>(
    prog: &GProgram<R>,
    dom: &DomainSetup<R>,
    cfg: &BabConfig,
) -> DomainResult<R> {
    let base = prog.init_alpha();
    let mut alpha = Alpha {
        relu: dom.relu.iter().map(|s| s.init.clone()).collect(),
        bil: base.bil,
        mul: base.mul,
    };
    let mut beta = dom.beta0.clone();
    let thr: Vec<f64> = dom.rows.iter().map(|&r| prog.thr[r]).collect();
    let n_rows = dom.rows.len();
    let mut pass = domain_pass(prog, dom, &alpha, &beta);
    let init = score(&pass.bounds, &thr);
    let mut best_alpha = alpha.clone();
    let mut best_beta = beta.clone();
    let mut best_score = init;
    let mut adam = Adam::new(alpha.len());
    let mut adam_beta = BetaAdam::new(beta.len());
    let mut lr_a = cfg.lr_alpha;
    let mut lr_b = cfg.lr_beta;
    let mut iters = 0usize;
    let mut evals = 1usize;
    let mut no_improve = 0usize;
    let stop = loop {
        let current = score(&pass.bounds, &thr);
        if current.verified == n_rows {
            break "verified";
        }
        if iters >= cfg.iters {
            break "iters";
        }
        if no_improve >= PATIENCE {
            break "patience";
        }
        if cfg.deadline.is_some_and(|d| Instant::now() >= d) {
            break "deadline";
        }
        adam.step(prog, &dom.relu, &mut alpha, &pass.grad, lr_a, iters + 1);
        adam_beta.step(&mut beta, &pass.gbeta, lr_b, iters + 1);
        lr_a *= LR_DECAY;
        lr_b *= LR_DECAY;
        iters += 1;
        pass = domain_pass(prog, dom, &alpha, &beta);
        evals += 1;
        let s = score(&pass.bounds, &thr);
        if improves(s, best_score) {
            best_alpha = alpha.clone();
            best_beta.clone_from(&beta);
            best_score = s;
            no_improve = 0;
        } else {
            no_improve += 1;
        }
    };
    let hint = GraphDomainHint {
        alpha: prog.hint_with(&dom.relu, &best_alpha),
        beta: best_beta.iter().map(|b| b.to_f32()).collect(),
    };
    DomainResult {
        best_alpha,
        best_beta,
        hint,
        iters,
        evals,
        init,
        best_score,
        stop,
    }
}

/// Bytes one worker holds for one domain: the re-derived sites, the
/// parameters (current, best, gradient accumulator: `3 × R`) and their Adam
/// moments (`2 × f64`), one iteration's planes, one in-flight row and the
/// literals with their multipliers.
pub(crate) fn domain_bytes<R: Real>(prog: &GProgram<R>, max_lits: usize) -> u64 {
    let relu_n: usize = prog.relu.iter().map(|s| s.status.len()).sum();
    let alpha_len = prog.init_alpha().len();
    let r = R::BYTES as u64;
    let sites = (relu_n as u64).saturating_mul(1 + 3 * r);
    let params = (alpha_len as u64).saturating_mul(3 * r + 16);
    let lits = (max_lits as u64).saturating_mul(16 + 3 * r + 16 + 8);
    sites
        .saturating_add(params)
        .saturating_add(prog.planes_bytes())
        .saturating_add(prog.row_bytes(alpha_len))
        .saturating_add(lits)
        .saturating_add(prog.tsize[0] as u64 * 2 * r)
}

/// One domain's fate.
enum Fate<R> {
    Done(Box<DomainResult<R>>),
    Refused,
    Late,
}

/// Lower the batch, search every domain in `R`, and wrap the results as
/// untrusted outputs.
pub(crate) fn run_graph_bab<R: Real>(
    net: &GraphSearchNet,
    request: &GraphBabRequest,
    threads: usize,
) -> Result<GraphBabOutcome, Refusal> {
    let t0 = Instant::now();
    if request.domains.is_empty() {
        return Err(Refusal::Nothing("no domain"));
    }
    if request.deadline.is_some_and(|d| Instant::now() >= d) {
        return Err(Refusal::Deadline);
    }
    let (prog, batch_box) = lower_batch::<R>(net, request)?;
    let cfg = BabConfig {
        iters: request.iters,
        lr_alpha: if request.lr_alpha.is_finite() && request.lr_alpha > 0.0 {
            request.lr_alpha
        } else {
            DEFAULT_LR
        },
        lr_beta: if request.lr_beta.is_finite() && request.lr_beta > 0.0 {
            request.lr_beta
        } else {
            DEFAULT_LR_BETA
        },
        deadline: request.deadline,
    };
    let n = request.domains.len();
    let max_lits = request
        .domains
        .iter()
        .map(|d| d.splits.len())
        .max()
        .unwrap_or(0);
    let fixed = prog.bytes();
    let per_domain = domain_bytes(&prog, max_lits);
    let mut threads = threads.clamp(1, n);
    while threads > 1
        && fixed.saturating_add(per_domain.saturating_mul(threads as u64)) > request.mem_cap_bytes
    {
        threads -= 1;
    }
    let need = fixed.saturating_add(per_domain.saturating_mul(threads as u64));
    if need > request.mem_cap_bytes {
        return Err(Refusal::Memory {
            need,
            cap: request.mem_cap_bytes,
        });
    }

    let next = AtomicUsize::new(0);
    let work = |_: usize| -> Vec<(usize, Fate<R>)> {
        let mut done = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            if cfg.deadline.is_some_and(|d| Instant::now() >= d) {
                done.push((i, Fate::Late));
                continue;
            }
            let fate = match setup_domain(&prog, request, &batch_box, &request.domains[i]) {
                Ok(dom) => Fate::Done(Box::new(search_domain(&prog, &dom, &cfg))),
                Err(_) => Fate::Refused,
            };
            done.push((i, fate));
        }
        done
    };
    let mut fates: Vec<Option<Fate<R>>> = (0..n).map(|_| None).collect();
    let collected: Vec<Vec<(usize, Fate<R>)>> = if threads == 1 {
        vec![work(0)]
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|w| {
                    let work = &work;
                    scope.spawn(move || work(w))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_default())
                .collect()
        })
    };
    for (i, fate) in collected.into_iter().flatten() {
        fates[i] = Some(fate);
    }

    let mut hints = Vec::with_capacity(n);
    let mut per_domain_stats = Vec::with_capacity(n);
    let (mut searched, mut refused, mut late, mut splits) = (0usize, 0usize, 0usize, 0usize);
    for (i, fate) in fates.into_iter().enumerate() {
        match fate {
            Some(Fate::Done(r)) => {
                searched += 1;
                splits += request.domains[i].splits.len();
                per_domain_stats.push(Some(GraphDomainStats {
                    iters: r.iters,
                    evals: r.evals,
                    init_slack: Untrusted::new(r.init.min_slack),
                    best_slack: Untrusted::new(r.best_score.min_slack),
                    verified_init: r.init.verified,
                    verified_best: r.best_score.verified,
                    stop: r.stop,
                }));
                hints.push(Some(r.hint));
            }
            Some(Fate::Refused) => {
                refused += 1;
                per_domain_stats.push(None);
                hints.push(None);
            }
            // A worker that panicked leaves its claims unset: no hint.
            Some(Fate::Late) | None => {
                late += 1;
                per_domain_stats.push(None);
                hints.push(None);
            }
        }
    }
    Ok(GraphBabOutcome {
        hints: Untrusted::new(hints),
        stats: GraphBabStats {
            domains: n,
            searched,
            refused,
            late,
            rows: prog.rows.len(),
            relu_sites: prog.relu.len(),
            splits,
            threads,
            search_secs: t0.elapsed().as_secs_f64(),
            per_domain: per_domain_stats,
        },
    })
}

/// Lower a batch's shared net, boxes and objective (no starting parameters:
/// every domain brings its own), and index its boxes by tensor.
pub(crate) fn lower_batch<R: Real>(
    net: &GraphSearchNet,
    request: &GraphBabRequest,
) -> Result<(GProgram<R>, Vec<Option<usize>>), Refusal> {
    let prog = GProgram::<R>::lower_parts(
        net,
        &request.lo,
        &request.hi,
        &request.boxes,
        &request.rows,
        &request.thresholds,
        &GraphAlphaHint::default(),
    )?;
    let mut batch_box = vec![None; prog.tsize.len()];
    for (i, b) in request.boxes.iter().enumerate() {
        batch_box[b.tensor] = Some(i);
    }
    Ok((prog, batch_box))
}

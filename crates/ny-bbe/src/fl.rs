// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M8 — the lowdim lane's FORWARD-LINEAR wave search, UNTRUSTED, on the CPU
//! (ENGINE_PROGRAM §2 M8, `#m8-input-split-batch`;
//! `bab.branching.input_split.lowdim_batch_engine: bbe`).
//!
//! # What it computes
//!
//! Per domain (an input box of one clause of one family), the formulas of the
//! `#lowdim-prover` lane's certified bounder (`ny-propagate`
//! `lowdim::forward` + `objective_bounds_passes`) in the search scalar `R`,
//! WITHOUT any rounding charge:
//!
//! * FORWARD: every node carries a lower and an upper linear form
//!   `a . x + c` in the input (`d + 1` coefficients). `Linear` is the lane's
//!   four-product rule `L' = W+ L + W- U + b_lo`, `U' = W+ U + W- L + b_hi`;
//!   `Sparse` is its terms plus the `[k_lo, k_hi]` constant; Slice, Concat,
//!   Identity, Add, Sub and the `*Const` ops are form arithmetic. Each node is
//!   concretised over the box. ReLU: a stable node is the identity or zero; an
//!   unstable one takes the upper chord `s = u / (u - l)`, `t = -s l` and the
//!   CROWN-adaptive lower line (identity iff `u >= -l`, else zero), and its
//!   box is intersected with `[max(l, 0), max(u, 0)]`. Sigmoid: the lane's
//!   slope candidates (chord, midpoint tangent, end tangents, the constant
//!   planes) with each intercept the extreme of `sigmoid(z) - a z` over the
//!   endpoints and the stationary points in the box (exact in real
//!   arithmetic, untrusted in `R`), intersected with `[sigmoid(l),
//!   sigmoid(u)]`.
//! * BACKWARD, per requested objective row: CROWN over the forward's
//!   relaxations (a non-negative multiplier takes the lower relaxation, a
//!   negative one the upper), `lb = c + sum_j min(g_j lo_j, g_j hi_j)`,
//!   `grad = g` (the input multiplier: the split hint) and `scale` the sum of
//!   the absolute terms of that bound.
//!
//! # Trust
//!
//! UNTRUSTED end to end (f32 in production, f64 as the in-crate reference):
//! round to nearest, no rounding charged, the lane's constants rounded to `R`.
//! Only HINTS leave: the lane CLAIMS a leaf when a hint (shifted by its probe's
//! `tau`) clears the clause's own predicate, then re-bounds it with its own
//! certified f64 forms and closes it only on the unchanged strict `lb > thr`;
//! every other leaf is split into its exact halves (any split dimension is a
//! cover). A bug here costs time, never a verdict.
//!
//! # Cost shape and determinism (R6)
//!
//! Domains run in blocks of [`FL_LANES`]: a block's forms are laid out
//! `[n][FL_LANES * (d + 1)]`, so a `Linear` is a rank-1-update sweep of the
//! weight matrix against contiguous column rows (one SIMD-friendly inner loop
//! per weight). Every column's accumulation runs in a FIXED order (the weight
//! matrix's `k` order forward, its `i` order backward) and every branch is
//! decided per column, so a domain's outputs depend on neither its
//! block-mates nor the thread count. Workers claim blocks from an atomic
//! counter and write their rows' outputs by index; the deadline is polled
//! between blocks.
//!
//! `Mul` and `MaxGlobal` are refused (the lane's McCormick and max rules are
//! not mirrored): the lane then runs its legacy bounder.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ny_core::search::{
    DeviceLease, FlBatch, FlHandle, FlOp, FlOpConsts, FlOut, FlProblem, FlStats, Refusal,
};

use crate::real::Real;

/// Domains per block (the column groups of one forms row).
pub(crate) const FL_LANES: usize = 8;

/// One lowered weight matrix (row-major `n_out x n_in`).
struct WUp<R> {
    n_out: usize,
    n_in: usize,
    w: Vec<R>,
}

/// One lowered op. Sources are node indices.
enum OpUp {
    Input,
    Linear {
        src: usize,
        w: usize,
    },
    Sparse {
        terms: Vec<Vec<(usize, usize)>>,
        offs: Vec<usize>,
    },
    Relu {
        src: usize,
    },
    Sigmoid {
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
    AddConst {
        src: usize,
    },
    RevSubConst {
        src: usize,
    },
    MulConst {
        src: usize,
    },
}

/// One op's per-clause constants in `R` (each distinct source `Arc` lowered
/// once and shared).
enum ConstUp<R> {
    None,
    Bias {
        lo: Arc<[R]>,
        hi: Arc<[R]>,
    },
    K(Arc<[R]>),
    Sparse {
        coef: Arc<[R]>,
        k_lo: Arc<[R]>,
        k_hi: Arc<[R]>,
    },
}

/// One lowered family: nodes `0..=seed` (later nodes cannot reach the
/// objective).
struct FamUp<R> {
    d: usize,
    sizes: Vec<usize>,
    seed: usize,
    ops: Vec<OpUp>,
    clauses: Vec<Vec<ConstUp<R>>>,
}

impl<R> FamUp<R> {
    fn nodes(&self) -> usize {
        self.seed + 1
    }
}

/// The per-worker working set of one family: offsets into flat buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    /// `d + 1`.
    d1: usize,
    /// Per node, the first element of its forms (`[n][FL_LANES * d1]`).
    form_off: Vec<usize>,
    /// Per node, the first element of its values (`[n][FL_LANES]`).
    val_off: Vec<usize>,
    /// Per node, the first element of its sigmoid planes (`[n][FL_LANES][4]`).
    sig_off: Vec<usize>,
    forms: usize,
    vals: usize,
    sigs: usize,
}

impl Layout {
    fn of(d: usize, sizes: &[usize], sigmoid: &[bool]) -> Self {
        let d1 = d + 1;
        let (mut forms, mut vals, mut sigs) = (0usize, 0usize, 0usize);
        let (mut form_off, mut val_off, mut sig_off) = (Vec::new(), Vec::new(), Vec::new());
        for (k, &n) in sizes.iter().enumerate() {
            form_off.push(forms);
            val_off.push(vals);
            sig_off.push(sigs);
            forms += n * FL_LANES * d1;
            vals += n * FL_LANES;
            if sigmoid.get(k).copied().unwrap_or(false) {
                sigs += n * FL_LANES * 4;
            }
        }
        Self {
            d1,
            form_off,
            val_off,
            sig_off,
            forms,
            vals,
            sigs,
        }
    }

    /// Scalars one worker allocates for this layout (see [`Ws::new`]).
    fn scalars(&self, d: usize) -> usize {
        // lower + upper forms, lower + upper values, multipliers, sigmoid
        // planes, the block's boxes (2 x lanes x d) and its gradients.
        2 * self.forms + 3 * self.vals + self.sigs + 3 * FL_LANES * d
    }
}

/// The per-worker buffers.
struct Ws<R> {
    lf: Vec<R>,
    uf: Vec<R>,
    lo: Vec<R>,
    hi: Vec<R>,
    lam: Vec<R>,
    sig: Vec<R>,
    xlo: Vec<R>,
    xhi: Vec<R>,
    grad: Vec<R>,
}

impl<R: Real> Ws<R> {
    fn new(layout: &Layout, d: usize) -> Self {
        Self {
            lf: vec![R::ZERO; layout.forms],
            uf: vec![R::ZERO; layout.forms],
            lo: vec![R::ZERO; layout.vals],
            hi: vec![R::ZERO; layout.vals],
            lam: vec![R::ZERO; layout.vals],
            sig: vec![R::ZERO; layout.sigs],
            xlo: vec![R::ZERO; FL_LANES * d],
            xhi: vec![R::ZERO; FL_LANES * d],
            grad: vec![R::ZERO; FL_LANES * d],
        }
    }

    /// Bytes actually allocated (capacities).
    #[cfg(test)]
    fn bytes(&self) -> u64 {
        let n = self.lf.capacity()
            + self.uf.capacity()
            + self.lo.capacity()
            + self.hi.capacity()
            + self.lam.capacity()
            + self.sig.capacity()
            + self.xlo.capacity()
            + self.xhi.capacity()
            + self.grad.capacity();
        (n as u64).saturating_mul(R::BYTES as u64)
    }
}

/// A lowered forward-linear problem (the payload of an [`FlHandle`]).
pub(crate) struct FlUpload<R: Real> {
    fams: Vec<FamUp<R>>,
    layouts: Vec<Layout>,
    weights: Vec<WUp<R>>,
    objectives: Vec<Vec<R>>,
    bytes: u64,
    pub(crate) threads: usize,
}

type ArcKey = (usize, usize);

fn key(a: &Arc<[f64]>) -> ArcKey {
    (a.as_ptr() as usize, a.len())
}

struct Lowerer<R> {
    consts: HashMap<ArcKey, Arc<[R]>>,
    weights: HashMap<ArcKey, usize>,
    bytes: u64,
}

impl<R: Real> Lowerer<R> {
    fn arc(&mut self, a: &Arc<[f64]>) -> Arc<[R]> {
        let k = key(a);
        if let Some(v) = self.consts.get(&k) {
            return Arc::clone(v);
        }
        let v: Arc<[R]> = a.iter().map(|&x| R::from_f64(x)).collect();
        self.bytes = self
            .bytes
            .saturating_add((v.len() as u64).saturating_mul(R::BYTES as u64));
        self.consts.insert(k, Arc::clone(&v));
        v
    }
}

impl<R: Real> FlUpload<R> {
    /// Lower `problem` into `R`: every distinct weight and constant `Arc` once.
    pub(crate) fn lower(problem: &FlProblem, threads: usize) -> Result<Self, Refusal> {
        problem.validate().map_err(Refusal::Invalid)?;
        for fam in &problem.families {
            if let Some(op) = fam
                .program
                .ops
                .iter()
                .find(|op| matches!(op, FlOp::Mul { .. } | FlOp::MaxGlobal { .. }))
            {
                return Err(Refusal::Unsupported(match op {
                    FlOp::Mul { .. } => "forward_linear: Mul",
                    _ => "forward_linear: MaxGlobal",
                }));
            }
        }
        let mut low = Lowerer::<R> {
            consts: HashMap::new(),
            weights: HashMap::new(),
            bytes: 0,
        };
        let mut weights: Vec<WUp<R>> = Vec::new();
        let mut fams = Vec::with_capacity(problem.families.len());
        let mut layouts = Vec::with_capacity(problem.families.len());
        for fam in &problem.families {
            let p = &fam.program;
            let nodes = p.seed + 1;
            let mut ops = Vec::with_capacity(nodes);
            let mut sigmoid = vec![false; nodes];
            let mut term_bytes = 0u64;
            for (k, op) in p.ops.iter().take(nodes).enumerate() {
                let u = |s: u32| s as usize;
                ops.push(match op {
                    FlOp::Input => OpUp::Input,
                    FlOp::Linear(l) => {
                        let wk = key(&l.w);
                        let w = match low.weights.get(&wk) {
                            Some(&i) => i,
                            None => {
                                let lowered: Vec<R> = l.w.iter().map(|&x| R::from_f64(x)).collect();
                                low.bytes = low.bytes.saturating_add(
                                    (lowered.len() as u64).saturating_mul(R::BYTES as u64),
                                );
                                weights.push(WUp {
                                    n_out: l.n_out as usize,
                                    n_in: l.n_in as usize,
                                    w: lowered,
                                });
                                low.weights.insert(wk, weights.len() - 1);
                                weights.len() - 1
                            }
                        };
                        OpUp::Linear { src: u(l.src), w }
                    }
                    FlOp::Sparse { terms } => {
                        let mut offs = Vec::with_capacity(terms.len());
                        let mut at = 0usize;
                        let lowered: Vec<Vec<(usize, usize)>> = terms
                            .iter()
                            .map(|t| {
                                offs.push(at);
                                at += t.len();
                                t.iter().map(|&(s, i)| (s as usize, i as usize)).collect()
                            })
                            .collect();
                        term_bytes = term_bytes.saturating_add((at as u64).saturating_mul(16));
                        OpUp::Sparse {
                            terms: lowered,
                            offs,
                        }
                    }
                    FlOp::Relu { src } => OpUp::Relu { src: u(*src) },
                    FlOp::Sigmoid { src } => {
                        sigmoid[k] = true;
                        OpUp::Sigmoid { src: u(*src) }
                    }
                    FlOp::Slice { src, start } => OpUp::Slice {
                        src: u(*src),
                        start: *start as usize,
                    },
                    FlOp::Concat { srcs } => OpUp::Concat {
                        srcs: srcs.iter().map(|&s| s as usize).collect(),
                    },
                    FlOp::Identity { src } => OpUp::Identity { src: u(*src) },
                    FlOp::Add { a, b } => OpUp::Add { a: u(*a), b: u(*b) },
                    FlOp::Sub { a, b } => OpUp::Sub { a: u(*a), b: u(*b) },
                    FlOp::AddConst { src } => OpUp::AddConst { src: u(*src) },
                    FlOp::RevSubConst { src } => OpUp::RevSubConst { src: u(*src) },
                    FlOp::MulConst { src } => OpUp::MulConst { src: u(*src) },
                    FlOp::Mul { .. } | FlOp::MaxGlobal { .. } => {
                        return Err(Refusal::Unsupported("forward_linear: Mul/MaxGlobal"));
                    }
                });
            }
            low.bytes = low.bytes.saturating_add(term_bytes);
            let clauses = fam
                .clauses
                .iter()
                .map(|consts| {
                    consts
                        .iter()
                        .take(nodes)
                        .map(|c| match c {
                            FlOpConsts::None => ConstUp::None,
                            FlOpConsts::Bias { lo, hi } => ConstUp::Bias {
                                lo: low.arc(lo),
                                hi: low.arc(hi),
                            },
                            FlOpConsts::K(k) => ConstUp::K(low.arc(k)),
                            FlOpConsts::Sparse { coef, k_lo, k_hi } => ConstUp::Sparse {
                                coef: low.arc(coef),
                                k_lo: low.arc(k_lo),
                                k_hi: low.arc(k_hi),
                            },
                        })
                        .collect()
                })
                .collect();
            let sizes: Vec<usize> = p.sizes[..nodes].to_vec();
            layouts.push(Layout::of(p.d, &sizes, &sigmoid));
            fams.push(FamUp {
                d: p.d,
                sizes,
                seed: p.seed,
                ops,
                clauses,
            });
        }
        let objectives: Vec<Vec<R>> = problem
            .objectives
            .iter()
            .map(|o| o.iter().map(|&x| R::from_f64(x)).collect())
            .collect();
        let obj_bytes: u64 = objectives
            .iter()
            .map(|o| (o.len() as u64).saturating_mul(R::BYTES as u64))
            .fold(0, u64::saturating_add);
        Ok(Self {
            fams,
            layouts,
            weights,
            objectives,
            bytes: low.bytes.saturating_add(obj_bytes),
            threads: threads.max(1),
        })
    }

    /// Resident bytes of the lowered problem (weights and constants once).
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Distinct weight matrices uploaded.
    #[cfg(test)]
    pub(crate) fn weight_count(&self) -> usize {
        self.weights.len()
    }

    /// The largest per-worker working set over the families.
    pub(crate) fn max_work_bytes(&self) -> u64 {
        self.fams
            .iter()
            .zip(&self.layouts)
            .map(|(f, l)| work_bytes_of::<R>(l, f.d))
            .max()
            .unwrap_or(0)
    }
}

fn work_bytes_of<R: Real>(layout: &Layout, d: usize) -> u64 {
    (layout.scalars(d) as u64).saturating_mul(R::BYTES as u64)
}

/// Per-worker working-set bytes for a family of input width `d`, node widths
/// `sizes` (nodes `0..=seed`) and sigmoid flags, at [`FL_LANES`] domains per
/// block — at least what [`Ws::new`] allocates (pinned by
/// `fl_work_bytes_cover_the_workspace`).
#[cfg(test)]
pub(crate) fn work_bytes<R: Real>(d: usize, sizes: &[usize], sigmoid: &[bool]) -> u64 {
    work_bytes_of::<R>(&Layout::of(d, sizes, sigmoid), d)
}

/// The bytes a fresh workspace for that shape really allocates (test face).
#[cfg(test)]
pub(crate) fn workspace_alloc_bytes<R: Real>(d: usize, sizes: &[usize], sigmoid: &[bool]) -> u64 {
    Ws::<R>::new(&Layout::of(d, sizes, sigmoid), d).bytes()
}

/// M8 upload: lower `problem`, refusing when the copy plus every worker's
/// working set exceeds the lease.
pub(crate) fn upload<R: Real>(
    backend: &'static str,
    problem: &FlProblem,
    lease: &DeviceLease,
    threads: usize,
) -> Result<FlHandle, Refusal> {
    let up = FlUpload::<R>::lower(problem, threads)?;
    let need = up
        .bytes()
        .saturating_add((up.threads as u64).saturating_mul(up.max_work_bytes()));
    if need > lease.bytes {
        return Err(Refusal::Memory {
            need,
            cap: lease.bytes,
        });
    }
    Ok(FlHandle::new(backend, problem, up))
}

fn past(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

/// One block's slice of the output (its CSR rows are contiguous).
struct BlockOut<'a> {
    lb: &'a mut [f32],
    scale: &'a mut [f32],
    grad: &'a mut [f32],
}

/// M8 `forward_linear_batch` on the upload in `handle` with `threads`
/// workers (0 = the upload's).
pub(crate) fn forward_linear_batch<R: Real>(
    handle: &FlHandle,
    batch: &FlBatch<'_>,
    out: &mut FlOut,
    threads: usize,
) -> Result<FlStats, Refusal> {
    let t0 = Instant::now();
    let up = handle.payload::<FlUpload<R>>().ok_or_else(|| {
        Refusal::Invalid(format!(
            "foreign forward-linear handle ({})",
            handle.backend()
        ))
    })?;
    batch.validate(handle).map_err(Refusal::Invalid)?;
    if past(batch.deadline) {
        return Err(Refusal::Deadline);
    }
    let f = batch.family as usize;
    let (fam, layout) = (&up.fams[f], &up.layouts[f]);
    let d = fam.d;
    let rows = batch.obj_rows.len();
    out.prepare(rows, d);
    if batch.count == 0 {
        return Ok(FlStats::default());
    }
    let blocks = batch.count.div_ceil(FL_LANES);
    let threads = if threads > 0 { threads } else { up.threads };
    let workers = threads.max(1).min(blocks);
    // Pre-split the output into per-block row ranges.
    let mut parts: Vec<Option<BlockOut<'_>>> = Vec::with_capacity(blocks);
    {
        let w = out.writer();
        let (mut lb, mut scale, mut grad) = (w.lb, w.scale, w.grad);
        for blk in 0..blocks {
            let b0 = blk * FL_LANES;
            let b1 = (b0 + FL_LANES).min(batch.count);
            let n = (batch.obj_offsets[b1] - batch.obj_offsets[b0]) as usize;
            let (a, rest) = std::mem::take(&mut lb).split_at_mut(n);
            lb = rest;
            let (s, rest) = std::mem::take(&mut scale).split_at_mut(n);
            scale = rest;
            let (g, rest) = std::mem::take(&mut grad).split_at_mut(n * d);
            grad = rest;
            parts.push(Some(BlockOut {
                lb: a,
                scale: s,
                grad: g,
            }));
        }
    }
    let parts = Mutex::new(parts);
    let next = AtomicUsize::new(0);
    let cut = AtomicBool::new(false);
    let run = || {
        let mut ws = Ws::<R>::new(layout, d);
        loop {
            if cut.load(Ordering::Relaxed) {
                break;
            }
            if past(batch.deadline) {
                cut.store(true, Ordering::Relaxed);
                break;
            }
            let blk = next.fetch_add(1, Ordering::Relaxed);
            if blk >= blocks {
                break;
            }
            let part = parts
                .lock()
                .ok()
                .and_then(|mut p| p.get_mut(blk).and_then(Option::take));
            let Some(mut part) = part else {
                cut.store(true, Ordering::Relaxed);
                break;
            };
            run_block(up, fam, layout, &mut ws, batch, blk * FL_LANES, &mut part);
        }
    };
    if workers == 1 {
        run();
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers).map(|_| scope.spawn(&run)).collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().map_err(|_| {
                        Refusal::Invalid("forward-linear search worker panicked".into())
                    })
                })
                .collect::<Result<Vec<()>, Refusal>>()
        })?;
    }
    if cut.load(Ordering::Relaxed) {
        return Err(Refusal::Deadline);
    }
    Ok(FlStats {
        domains: batch.count as u64,
        rows: rows as u64,
        secs: t0.elapsed().as_secs_f64(),
        threads: workers,
        max_block: FL_LANES,
    })
}

/// Clause of domain `b` (clause 0 when the batch names none).
fn clause_of(batch: &FlBatch<'_>, b: usize) -> usize {
    batch.clause.get(b).map_or(0, |&c| c as usize)
}

/// Bound one block of up to [`FL_LANES`] domains starting at `b0`.
#[allow(clippy::too_many_arguments)]
fn run_block<R: Real>(
    up: &FlUpload<R>,
    fam: &FamUp<R>,
    layout: &Layout,
    ws: &mut Ws<R>,
    batch: &FlBatch<'_>,
    b0: usize,
    out: &mut BlockOut<'_>,
) {
    let d = fam.d;
    let nb = (batch.count - b0).min(FL_LANES);
    let mut clause = [0usize; FL_LANES];
    for lane in 0..FL_LANES {
        let (xl, xh) = (
            &mut ws.xlo[lane * d..(lane + 1) * d],
            &mut ws.xhi[lane * d..(lane + 1) * d],
        );
        if lane < nb {
            let b = b0 + lane;
            for k in 0..d {
                xl[k] = R::from_f64(batch.lo[b * d + k]);
                xh[k] = R::from_f64(batch.hi[b * d + k]);
            }
            clause[lane] = clause_of(batch, b);
        } else {
            // An idle lane: a point box of clause 0 (its outputs are dropped).
            xl.fill(R::ZERO);
            xh.fill(R::ZERO);
            clause[lane] = 0;
        }
    }
    forward(up, fam, layout, ws, &clause);
    let row0 = batch.obj_offsets[b0] as usize;
    let rows_of = |lane: usize| -> usize {
        if lane < nb {
            let b = b0 + lane;
            (batch.obj_offsets[b + 1] - batch.obj_offsets[b]) as usize
        } else {
            0
        }
    };
    let slots = (0..nb).map(rows_of).max().unwrap_or(0);
    for s in 0..slots {
        let mut obj: [Option<&[R]>; FL_LANES] = [None; FL_LANES];
        for (lane, o) in obj.iter_mut().enumerate() {
            if rows_of(lane) > s {
                let b = b0 + lane;
                let r = batch.obj_rows[batch.obj_offsets[b] as usize + s] as usize;
                *o = Some(&up.objectives[r]);
            }
        }
        let (cst, abs) = backward(up, fam, layout, ws, &clause, &obj);
        for lane in 0..nb {
            if obj[lane].is_none() {
                continue;
            }
            let b = b0 + lane;
            let r = batch.obj_offsets[b] as usize + s - row0;
            let (xl, xh) = (
                &ws.xlo[lane * d..(lane + 1) * d],
                &ws.xhi[lane * d..(lane + 1) * d],
            );
            let g = &ws.grad[lane * d..(lane + 1) * d];
            let (mut lb, mut sc) = (cst[lane], abs[lane]);
            for k in 0..d {
                let gk = g[k];
                lb += if gk >= R::ZERO {
                    gk * xl[k]
                } else {
                    gk * xh[k]
                };
                sc += gk.abs() * xl[k].abs().max(xh[k].abs());
            }
            out.lb[r] = lb.to_f32();
            out.scale[r] = sc.to_f32();
            for k in 0..d {
                out.grad[r * d + k] = g[k].to_f32();
            }
        }
    }
}

/// `dst[j] += a * src[j]` over a column row.
#[inline(always)]
fn axpy<R: Real>(dst: &mut [R], a: R, src: &[R]) {
    for (o, &s) in dst.iter_mut().zip(src) {
        *o += a * s;
    }
}

/// Concretise node `k`'s forms over each lane's box into its values.
fn concretize<R: Real>(fam: &FamUp<R>, layout: &Layout, ws: &mut Ws<R>, k: usize) {
    let (d, d1) = (fam.d, layout.d1);
    let n = fam.sizes[k];
    let (fo, vo) = (layout.form_off[k], layout.val_off[k]);
    let row = FL_LANES * d1;
    for i in 0..n {
        for lane in 0..FL_LANES {
            let c = fo + i * row + lane * d1;
            let (fl, fh) = (&ws.lf[c..c + d1], &ws.uf[c..c + d1]);
            let (xl, xh) = (
                &ws.xlo[lane * d..(lane + 1) * d],
                &ws.xhi[lane * d..(lane + 1) * d],
            );
            let (mut vl, mut vh) = (fl[d], fh[d]);
            for q in 0..d {
                let a = fl[q];
                vl += if a >= R::ZERO { a * xl[q] } else { a * xh[q] };
                let a = fh[q];
                vh += if a >= R::ZERO { a * xh[q] } else { a * xl[q] };
            }
            ws.lo[vo + i * FL_LANES + lane] = vl;
            ws.hi[vo + i * FL_LANES + lane] = vh;
        }
    }
}

/// The per-lane constant of an op (its lane's clause).
fn consts<R>(fam: &FamUp<R>, clause: usize, k: usize) -> &ConstUp<R> {
    &fam.clauses[clause][k]
}

/// The forward pass over nodes `0..=seed` of one block.
fn forward<R: Real>(
    up: &FlUpload<R>,
    fam: &FamUp<R>,
    layout: &Layout,
    ws: &mut Ws<R>,
    clause: &[usize; FL_LANES],
) {
    let (d, d1) = (fam.d, layout.d1);
    let row = FL_LANES * d1;
    for k in 0..fam.nodes() {
        let n = fam.sizes[k];
        let fo = layout.form_off[k];
        let len = n * row;
        ws.lf[fo..fo + len].fill(R::ZERO);
        ws.uf[fo..fo + len].fill(R::ZERO);
        // Forms of nodes < k are read-only; node k's are written.
        let (lb, la) = ws.lf.split_at_mut(fo);
        let (ub, ua) = ws.uf.split_at_mut(fo);
        let (dl, du) = (&mut la[..len], &mut ua[..len]);
        let src_rows = |s: usize, i: usize| {
            let o = layout.form_off[s] + i * row;
            (o, o + row)
        };
        match &fam.ops[k] {
            OpUp::Input => {
                for i in 0..n {
                    for lane in 0..FL_LANES {
                        dl[i * row + lane * d1 + i] = R::ONE;
                        du[i * row + lane * d1 + i] = R::ONE;
                    }
                }
            }
            OpUp::Linear { src, w } => {
                let wu = &up.weights[*w];
                for i in 0..wu.n_out {
                    let (ol, ou) = (
                        &mut dl[i * row..(i + 1) * row],
                        &mut du[i * row..(i + 1) * row],
                    );
                    let wrow = &wu.w[i * wu.n_in..(i + 1) * wu.n_in];
                    for (q, &a) in wrow.iter().enumerate() {
                        if a > R::ZERO {
                            let (s0, s1) = src_rows(*src, q);
                            axpy(ol, a, &lb[s0..s1]);
                            axpy(ou, a, &ub[s0..s1]);
                        } else if a < R::ZERO {
                            let (s0, s1) = src_rows(*src, q);
                            axpy(ol, a, &ub[s0..s1]);
                            axpy(ou, a, &lb[s0..s1]);
                        }
                    }
                }
                for lane in 0..FL_LANES {
                    if let ConstUp::Bias { lo, hi } = consts(fam, clause[lane], k) {
                        for i in 0..n {
                            let c = i * row + lane * d1 + d;
                            dl[c] += lo[i];
                            du[c] += hi[i];
                        }
                    }
                }
            }
            OpUp::Sparse { terms, offs } => {
                for lane in 0..FL_LANES {
                    let ConstUp::Sparse { coef, k_lo, k_hi } = consts(fam, clause[lane], k) else {
                        continue;
                    };
                    for (j, tj) in terms.iter().enumerate() {
                        let c0 = j * row + lane * d1;
                        for (t, &(s, idx)) in tj.iter().enumerate() {
                            let a = coef[offs[j] + t];
                            let (s0, _) = src_rows(s, idx);
                            let (sl, su) = (
                                &lb[s0 + lane * d1..s0 + (lane + 1) * d1],
                                &ub[s0 + lane * d1..s0 + (lane + 1) * d1],
                            );
                            if a >= R::ZERO {
                                axpy(&mut dl[c0..c0 + d1], a, sl);
                                axpy(&mut du[c0..c0 + d1], a, su);
                            } else {
                                axpy(&mut dl[c0..c0 + d1], a, su);
                                axpy(&mut du[c0..c0 + d1], a, sl);
                            }
                        }
                        dl[c0 + d] += k_lo[j];
                        du[c0 + d] += k_hi[j];
                    }
                }
            }
            OpUp::Relu { src } => {
                let vs = layout.val_off[*src];
                for i in 0..n {
                    let (s0, _) = src_rows(*src, i);
                    for lane in 0..FL_LANES {
                        let (l, u) = (
                            ws.lo[vs + i * FL_LANES + lane],
                            ws.hi[vs + i * FL_LANES + lane],
                        );
                        let c = lane * d1;
                        let (sl, su) = (&lb[s0 + c..s0 + c + d1], &ub[s0 + c..s0 + c + d1]);
                        let (ol, ou) = (
                            &mut dl[i * row + c..i * row + c + d1],
                            &mut du[i * row + c..i * row + c + d1],
                        );
                        if u <= R::ZERO {
                            continue;
                        }
                        if l >= R::ZERO {
                            ol.copy_from_slice(sl);
                            ou.copy_from_slice(su);
                            continue;
                        }
                        let s = u / (u - l);
                        let t = -(s * l);
                        for q in 0..d1 {
                            ou[q] = s * su[q];
                        }
                        ou[d] += t;
                        if u >= -l {
                            ol.copy_from_slice(sl);
                        }
                    }
                }
            }
            OpUp::Sigmoid { src } => {
                let vs = layout.val_off[*src];
                let so = layout.sig_off[k];
                for i in 0..n {
                    let (s0, _) = src_rows(*src, i);
                    for lane in 0..FL_LANES {
                        let (l, u) = (
                            ws.lo[vs + i * FL_LANES + lane],
                            ws.hi[vs + i * FL_LANES + lane],
                        );
                        let p = sigmoid_planes(l.to_f64(), u.to_f64());
                        let pr: [R; 4] = [
                            R::from_f64(p[0]),
                            R::from_f64(p[1]),
                            R::from_f64(p[2]),
                            R::from_f64(p[3]),
                        ];
                        let at = so + (i * FL_LANES + lane) * 4;
                        ws.sig[at..at + 4].copy_from_slice(&pr);
                        let c = lane * d1;
                        let (sl, su) = (&lb[s0 + c..s0 + c + d1], &ub[s0 + c..s0 + c + d1]);
                        let (ol, ou) = (
                            &mut dl[i * row + c..i * row + c + d1],
                            &mut du[i * row + c..i * row + c + d1],
                        );
                        let [a_lo, b_lo, a_hi, b_hi] = pr;
                        let from = if a_lo >= R::ZERO { sl } else { su };
                        for q in 0..d1 {
                            ol[q] = a_lo * from[q];
                        }
                        ol[d] += b_lo;
                        let from = if a_hi >= R::ZERO { su } else { sl };
                        for q in 0..d1 {
                            ou[q] = a_hi * from[q];
                        }
                        ou[d] += b_hi;
                    }
                }
            }
            OpUp::Slice { src, start } => {
                let (s0, _) = src_rows(*src, *start);
                dl.copy_from_slice(&lb[s0..s0 + len]);
                du.copy_from_slice(&ub[s0..s0 + len]);
            }
            OpUp::Concat { srcs } => {
                let mut at = 0usize;
                for &s in srcs {
                    let l = fam.sizes[s] * row;
                    let s0 = layout.form_off[s];
                    dl[at..at + l].copy_from_slice(&lb[s0..s0 + l]);
                    du[at..at + l].copy_from_slice(&ub[s0..s0 + l]);
                    at += l;
                }
            }
            OpUp::Identity { src } => {
                let s0 = layout.form_off[*src];
                dl.copy_from_slice(&lb[s0..s0 + len]);
                du.copy_from_slice(&ub[s0..s0 + len]);
            }
            OpUp::Add { a, b } | OpUp::Sub { a, b } => {
                let sub = matches!(fam.ops[k], OpUp::Sub { .. });
                let (a0, b0) = (layout.form_off[*a], layout.form_off[*b]);
                for q in 0..len {
                    if sub {
                        dl[q] = lb[a0 + q] - ub[b0 + q];
                        du[q] = ub[a0 + q] - lb[b0 + q];
                    } else {
                        dl[q] = lb[a0 + q] + lb[b0 + q];
                        du[q] = ub[a0 + q] + ub[b0 + q];
                    }
                }
            }
            OpUp::AddConst { src } | OpUp::RevSubConst { src } => {
                let rev = matches!(fam.ops[k], OpUp::RevSubConst { .. });
                let s0 = layout.form_off[*src];
                for q in 0..len {
                    if rev {
                        dl[q] = -ub[s0 + q];
                        du[q] = -lb[s0 + q];
                    } else {
                        dl[q] = lb[s0 + q];
                        du[q] = ub[s0 + q];
                    }
                }
                for lane in 0..FL_LANES {
                    if let ConstUp::K(kv) = consts(fam, clause[lane], k) {
                        for i in 0..n {
                            let c = i * row + lane * d1 + d;
                            dl[c] += kv[i];
                            du[c] += kv[i];
                        }
                    }
                }
            }
            OpUp::MulConst { src } => {
                let s0 = layout.form_off[*src];
                for lane in 0..FL_LANES {
                    let ConstUp::K(kv) = consts(fam, clause[lane], k) else {
                        continue;
                    };
                    for i in 0..n {
                        let c = i * row + lane * d1;
                        let kj = kv[i];
                        let (sl, su) = (&lb[s0 + c..s0 + c + d1], &ub[s0 + c..s0 + c + d1]);
                        let (fl, fh) = if kj >= R::ZERO { (sl, su) } else { (su, sl) };
                        for q in 0..d1 {
                            dl[c + q] = kj * fl[q];
                            du[c + q] = kj * fh[q];
                        }
                    }
                }
            }
        }
        if k == 0 {
            // The input's box is the lane's box exactly.
            let vo = layout.val_off[0];
            for i in 0..n {
                for lane in 0..FL_LANES {
                    ws.lo[vo + i * FL_LANES + lane] = ws.xlo[lane * d + i];
                    ws.hi[vo + i * FL_LANES + lane] = ws.xhi[lane * d + i];
                }
            }
            continue;
        }
        concretize(fam, layout, ws, k);
        match &fam.ops[k] {
            OpUp::Relu { src } => {
                let (vs, vo) = (layout.val_off[*src], layout.val_off[k]);
                for q in 0..n * FL_LANES {
                    let (l, u) = (ws.lo[vs + q], ws.hi[vs + q]);
                    ws.lo[vo + q] = ws.lo[vo + q].max(l.max(R::ZERO)).max(R::ZERO);
                    ws.hi[vo + q] = ws.hi[vo + q].min(u.max(R::ZERO));
                }
            }
            OpUp::Sigmoid { src } => {
                let (vs, vo) = (layout.val_off[*src], layout.val_off[k]);
                for q in 0..n * FL_LANES {
                    let (l, u) = (ws.lo[vs + q].to_f64(), ws.hi[vs + q].to_f64());
                    if l.is_finite() && u.is_finite() {
                        ws.lo[vo + q] = ws.lo[vo + q].max(R::from_f64(sig(l)));
                        ws.hi[vo + q] = ws.hi[vo + q].min(R::from_f64(sig(u)));
                    }
                }
            }
            _ => {}
        }
    }
}

/// One backward (CROWN) pass per active lane over the block's forward
/// relaxations: lane `l` seeds its objective row `obj[l]` (idle lanes seed
/// zero). Leaves the input multipliers in `ws.grad`; returns each lane's
/// constant and its absolute-term sum.
fn backward<R: Real>(
    up: &FlUpload<R>,
    fam: &FamUp<R>,
    layout: &Layout,
    ws: &mut Ws<R>,
    clause: &[usize; FL_LANES],
    obj: &[Option<&[R]>; FL_LANES],
) -> ([R; FL_LANES], [R; FL_LANES]) {
    let d = fam.d;
    let mut cst = [R::ZERO; FL_LANES];
    let mut abs = [R::ZERO; FL_LANES];
    let top = layout.val_off[fam.seed] + fam.sizes[fam.seed] * FL_LANES;
    ws.lam[..top].fill(R::ZERO);
    let vseed = layout.val_off[fam.seed];
    for (lane, o) in obj.iter().enumerate() {
        if let Some(o) = o {
            for (j, &v) in o.iter().enumerate() {
                ws.lam[vseed + j * FL_LANES + lane] = v;
            }
        }
    }
    for k in (1..fam.nodes()).rev() {
        let n = fam.sizes[k];
        let vo = layout.val_off[k];
        let (below, rest) = ws.lam.split_at_mut(vo);
        let lk = &rest[..n * FL_LANES];
        let at = |s: usize, i: usize| layout.val_off[s] + i * FL_LANES;
        match &fam.ops[k] {
            OpUp::Input => {}
            OpUp::Linear { src, w } => {
                let wu = &up.weights[*w];
                for i in 0..wu.n_out {
                    let li = &lk[i * FL_LANES..(i + 1) * FL_LANES];
                    let wrow = &wu.w[i * wu.n_in..(i + 1) * wu.n_in];
                    for (q, &a) in wrow.iter().enumerate() {
                        if a == R::ZERO {
                            continue;
                        }
                        let s0 = at(*src, q);
                        axpy(&mut below[s0..s0 + FL_LANES], a, li);
                    }
                }
                for lane in 0..FL_LANES {
                    if let ConstUp::Bias { lo, hi } = consts(fam, clause[lane], k) {
                        for i in 0..n {
                            let lj = lk[i * FL_LANES + lane];
                            let p = if lj >= R::ZERO {
                                lj * lo[i]
                            } else {
                                lj * hi[i]
                            };
                            cst[lane] += p;
                            abs[lane] += p.abs();
                        }
                    }
                }
            }
            OpUp::Relu { src } => {
                let vs = layout.val_off[*src];
                for j in 0..n {
                    for lane in 0..FL_LANES {
                        let lj = lk[j * FL_LANES + lane];
                        let (l, u) = (
                            ws.lo[vs + j * FL_LANES + lane],
                            ws.hi[vs + j * FL_LANES + lane],
                        );
                        let m = if u <= R::ZERO || lj == R::ZERO {
                            continue;
                        } else if l >= R::ZERO {
                            lj
                        } else if lj > R::ZERO {
                            if u >= -l {
                                lj
                            } else {
                                continue;
                            }
                        } else {
                            let s = u / (u - l);
                            let t = -(s * l);
                            let p = lj * t;
                            cst[lane] += p;
                            abs[lane] += p.abs();
                            lj * s
                        };
                        below[at(*src, j) + lane] += m;
                    }
                }
            }
            OpUp::Sigmoid { src } => {
                let so = layout.sig_off[k];
                for j in 0..n {
                    for lane in 0..FL_LANES {
                        let lj = lk[j * FL_LANES + lane];
                        if lj == R::ZERO {
                            continue;
                        }
                        let p = &ws.sig
                            [so + (j * FL_LANES + lane) * 4..so + (j * FL_LANES + lane) * 4 + 4];
                        let (a, c) = if lj > R::ZERO {
                            (p[0], p[1])
                        } else {
                            (p[2], p[3])
                        };
                        let q = lj * c;
                        cst[lane] += q;
                        abs[lane] += q.abs();
                        below[at(*src, j) + lane] += lj * a;
                    }
                }
            }
            OpUp::Slice { src, start } => {
                let s0 = at(*src, *start);
                for (o, &v) in below[s0..s0 + n * FL_LANES].iter_mut().zip(lk) {
                    *o += v;
                }
            }
            OpUp::Concat { srcs } => {
                let mut off = 0usize;
                for &s in srcs {
                    let w = fam.sizes[s] * FL_LANES;
                    let s0 = at(s, 0);
                    for (o, &v) in below[s0..s0 + w].iter_mut().zip(&lk[off..off + w]) {
                        *o += v;
                    }
                    off += w;
                }
            }
            OpUp::Identity { src } => {
                let s0 = at(*src, 0);
                for (o, &v) in below[s0..s0 + n * FL_LANES].iter_mut().zip(lk) {
                    *o += v;
                }
            }
            OpUp::Add { a, b } | OpUp::Sub { a, b } => {
                let sign = if matches!(fam.ops[k], OpUp::Sub { .. }) {
                    -R::ONE
                } else {
                    R::ONE
                };
                let a0 = at(*a, 0);
                for (o, &v) in below[a0..a0 + n * FL_LANES].iter_mut().zip(lk) {
                    *o += v;
                }
                let b0 = at(*b, 0);
                for (o, &v) in below[b0..b0 + n * FL_LANES].iter_mut().zip(lk) {
                    *o += sign * v;
                }
            }
            OpUp::AddConst { src } | OpUp::RevSubConst { src } => {
                let sign = if matches!(fam.ops[k], OpUp::RevSubConst { .. }) {
                    -R::ONE
                } else {
                    R::ONE
                };
                for lane in 0..FL_LANES {
                    if let ConstUp::K(kv) = consts(fam, clause[lane], k) {
                        for j in 0..n {
                            let p = lk[j * FL_LANES + lane] * kv[j];
                            cst[lane] += p;
                            abs[lane] += p.abs();
                        }
                    }
                }
                let s0 = at(*src, 0);
                for (o, &v) in below[s0..s0 + n * FL_LANES].iter_mut().zip(lk) {
                    *o += sign * v;
                }
            }
            OpUp::MulConst { src } => {
                for lane in 0..FL_LANES {
                    let ConstUp::K(kv) = consts(fam, clause[lane], k) else {
                        continue;
                    };
                    for j in 0..n {
                        below[at(*src, j) + lane] += lk[j * FL_LANES + lane] * kv[j];
                    }
                }
            }
            OpUp::Sparse { terms, offs } => {
                for lane in 0..FL_LANES {
                    let ConstUp::Sparse { coef, k_lo, k_hi } = consts(fam, clause[lane], k) else {
                        continue;
                    };
                    for (j, tj) in terms.iter().enumerate() {
                        let lj = lk[j * FL_LANES + lane];
                        if lj == R::ZERO {
                            continue;
                        }
                        let kj = if lj >= R::ZERO { k_lo[j] } else { k_hi[j] };
                        let p = lj * kj;
                        cst[lane] += p;
                        abs[lane] += p.abs();
                        for (t, &(s, idx)) in tj.iter().enumerate() {
                            below[at(s, idx) + lane] += lj * coef[offs[j] + t];
                        }
                    }
                }
            }
        }
    }
    let v0 = layout.val_off[0];
    for lane in 0..FL_LANES {
        for q in 0..d {
            ws.grad[lane * d + q] = ws.lam[v0 + q * FL_LANES + lane];
        }
    }
    (cst, abs)
}

// ---------------------------------------------------------------- sigmoid

/// The plain f64 sigmoid.
#[inline]
fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `sigmoid'(x) = s (1 - s)`.
#[inline]
fn dsig(x: f64) -> f64 {
    let s = sig(x);
    s * (1.0 - s)
}

/// The two stationary points `+-z*` of `sigmoid(z) - a z` (`sigmoid'(z*) =
/// a`), when `a` is in `(0, 1/4)`.
fn stationary(a: f64) -> Option<(f64, f64)> {
    if !(a > 0.0 && a < 0.25) {
        return None;
    }
    let r = (1.0 - 4.0 * a).sqrt();
    let small = 2.0 * a / (1.0 + r);
    let z = (small / (1.0 - small)).ln();
    z.is_finite().then_some((z, -z))
}

/// The extreme (`min` or max) of `h(z) = sigmoid(z) - a z` over `[l, u]`:
/// over the endpoints and the stationary points inside (exact in real
/// arithmetic; untrusted in floating point).
fn h_extreme(l: f64, u: f64, a: f64, min: bool) -> f64 {
    let h = |z: f64| sig(z) - a * z;
    let mut v = if min { h(l).min(h(u)) } else { h(l).max(h(u)) };
    if let Some((z0, z1)) = stationary(a) {
        for z in [z0, z1] {
            if z > l && z < u {
                v = if min { v.min(h(z)) } else { v.max(h(z)) };
            }
        }
    }
    v
}

/// `[a_lo, b_lo, a_hi, b_hi]` with `a_lo z + b_lo <= sigmoid(z) <= a_hi z +
/// b_hi` on `[l, u]` (untrusted): the lane's slope candidates (convex region:
/// midpoint tangent below, chord above; concave: the reverse; across 0: the
/// chord and the end tangents; always the constant planes), each intercept
/// the extreme of `sigmoid(z) - a z`, keeping the plane that is tightest at
/// the midpoint. NaN planes for a non-finite or crossed box.
pub(crate) fn sigmoid_planes(l: f64, u: f64) -> [f64; 4] {
    if !(l.is_finite() && u.is_finite() && l <= u) {
        return [f64::NAN; 4];
    }
    let (c_lo, c_hi) = (sig(l), sig(u));
    let mid = 0.5 * l + 0.5 * u;
    let w = u - l;
    let chord = if w > 0.0 && w.is_finite() {
        let k = (sig(u) - sig(l)) / w;
        if k.is_finite() {
            k.clamp(0.0, 0.25)
        } else {
            0.0
        }
    } else {
        dsig(mid)
    };
    let tan_mid = dsig(mid);
    let (lo_cands, hi_cands) = if u <= 0.0 {
        ([tan_mid, chord], [chord, chord])
    } else if l >= 0.0 {
        ([chord, chord], [tan_mid, chord])
    } else {
        ([chord, dsig(l)], [chord, dsig(u)])
    };
    let (mut lo, mut lo_at) = ((0.0, c_lo), c_lo);
    for a in lo_cands {
        if !(a.is_finite() && a >= 0.0) {
            continue;
        }
        let b = h_extreme(l, u, a, true);
        let v = a * mid + b;
        if b.is_finite() && v > lo_at {
            lo = (a, b);
            lo_at = v;
        }
    }
    let (mut hi, mut hi_at) = ((0.0, c_hi), c_hi);
    for a in hi_cands {
        if !(a.is_finite() && a >= 0.0) {
            continue;
        }
        let b = h_extreme(l, u, a, false);
        let v = a * mid + b;
        if b.is_finite() && v < hi_at {
            hi = (a, b);
            hi_at = v;
        }
    }
    [lo.0, lo.1, hi.0, hi.1]
}

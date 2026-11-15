// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M8a — `#m8-input-split-batch`: ONE input-split tree for several clauses
//! that share ONE box (`bab.branching.input_split.engine: bbe`).
//!
//! # Why
//!
//! The grouped-disjunctive prepass (lsnc, acasxu disjunctive: one lowering,
//! one box, `K` clauses) and the independent-singleton route run one lane
//! tree per clause, so a region every clause must split is bounded `K` times.
//! Here one tree serves every clause: each leaf carries the set (a bit mask)
//! of clauses still OPEN on it, one forward pass per node serves all of them,
//! and each clause's backward passes run only while it is open there.
//!
//! # Design
//!
//! * The queue is LIFO over `(lo, hi, depth, mask)`; `open[c]` counts the
//!   queued or in-flight leaves whose mask holds clause `c`.
//! * A wave takes up to `cfg.batch` leaves from the tail. The legacy bounder
//!   cuts the wave into the run's chunks and bounds each chunk against the
//!   UNION of the objectives of the clauses open in any of its leaves (a
//!   leaf's bound for one objective does not depend on which other
//!   objectives share its chunk). With `lowdim_batch_engine: bbe` the wave is
//!   searched first (each leaf asks exactly for its open clauses' rows); a
//!   leaf some open clause CLAIMS is bounded as above, every other leaf is
//!   split on its hints (see `bbe`) — except one that cannot be split, which
//!   is bounded as above too (a hint never gives a clause up at the
//!   resolution floor).
//! * `decide_multi` per leaf, clauses in ascending order: (1) clear `c` iff
//!   one of `c`'s conjuncts has f64 `lb > thr` (strict); (2) with
//!   `cfg.clip`, clip a copy of the leaf box with `c`'s OWN planes from this
//!   leaf's pass for each open `c` — an empty result clears `c` — and take the
//!   HULL of the surviving clipped boxes; (3) an empty mask closes the leaf;
//!   (4) the first open `c` whose centre violates all its conjuncts records
//!   `c`'s witness and gives `c` up (cleared lazily from queued leaves);
//!   (5) otherwise split along the nearest open objective, both children
//!   carrying the mask; (6) Resolution gives up every clause in the mask.
//! * `Verified(c)` iff `open[c]` reaches 0 and `c` was never given up. A
//!   deadline or queue stop leaves every clause without an outcome
//!   `Unproved`.
//!
//! # Soundness (the disjunction invariant)
//!
//! For every clause `c`: `V_c ∩ B0 ⊆ ∪ {box(L) : L queued ∧ c ∈ mask(L)}`,
//! `V_c` its counterexamples. Splits are exact halves and copy the mask; `c`
//! is cleared from a leaf only when `V_c ∩ box(L) = ∅` — a strict f64 bound
//! of one of `c`'s conjuncts over THAT leaf's box, or `c`'s own clip planes
//! (that leaf's passes, outward rounded) emptying it; the hull contains every
//! surviving clipped box, so every remaining clause's counterexamples in the
//! leaf lie in it; lazy drops apply only to clauses given up, which are never
//! verified. So `open[c] = 0` means `V_c = ∅`. The sat side is unchanged: a
//! witness is only a CANDIDATE (the lane's f64 point evaluation) for the
//! caller's trusted forward and the unchanged trusted-oracle gate.

use std::time::Instant;

use rayon::prelude::*;

use super::bbe::{BbeRun, BbeStats, ClaimMark, ProbeSpec, Scheduler, Searched};
use super::search::{self, choose_split, clip_box, AlphaSpec, ClipPlane, DomainBound};
use super::{
    box_volume, candidate_margin, first_argmax, lane_threads, point_values, Chunk, LaneAdmission,
    LowDimBatchEngine, LowDimConfig, LowDimOutcome, LowDimStats, LowDimStop, Op, Program,
    CHUNK_BYTES,
};

/// One clause of a disjunction: a CONJUNCTION of `objectives[i] . y >
/// thresholds[i]` targets, over the admission's output (as
/// `prove_conjunction_admitted` takes it).
#[derive(Debug, Clone, PartialEq)]
pub struct DisjunctionClause {
    /// Objective rows over the admission's output width.
    pub objectives: Vec<Vec<f32>>,
    /// One threshold per objective.
    pub thresholds: Vec<f32>,
}

/// Counters of one disjunction run (the `[lowdim-disjunction] done` line).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DisjunctionStats {
    /// Clauses given.
    pub clauses: usize,
    /// Leaves bounded (each once, whatever its mask).
    pub domains: u64,
    /// Waves run.
    pub waves: u64,
    /// Queued leaves dropped because every clause of their mask was given up.
    pub mask_drops: u64,
    /// Per clause, `LowDimStats::closed_volume`.
    pub closed_volume: Vec<f64>,
    /// Wall seconds.
    pub secs: f64,
    /// The wave search's counters (default when unarmed).
    pub bbe: BbeStats,
}

/// One disjunction run's result: per clause its outcome (`Verified` only
/// when every leaf of the clause closed) and its violation CANDIDATE, if it
/// stopped on one (full-dimension input; a candidate only).
#[derive(Debug, Clone)]
pub struct DisjunctionOutcome {
    /// One outcome per clause, in order.
    pub per_clause: Vec<LowDimOutcome>,
    /// Per clause, the candidate point it was given up on.
    pub witnesses: Vec<Option<Vec<f64>>>,
    /// Counters.
    pub stats: DisjunctionStats,
}

/// The named admission of the disjunction tree (router C2): at least two
/// clauses over the admission's ONE box.
pub fn disjunction_admits(n_clauses: usize, admission: &LaneAdmission) -> Result<(), &'static str> {
    if n_clauses < 2 {
        return Err("fewer than two clauses");
    }
    if admission.prog.d == 0
        || admission.lo0.len() != admission.prog.d
        || admission.hi0.len() != admission.prog.d
    {
        return Err("no single box");
    }
    Ok(())
}

/// Everything the per-leaf decisions share.
pub(super) struct Ctx<'a> {
    prog: &'a Program,
    /// The union objective table and its thresholds (clause after clause).
    objs: &'a [Vec<f64>],
    thrs: &'a [f64],
    /// Per clause, its rows of the table.
    rows_of: &'a [std::ops::Range<usize>],
    cfg: &'a LowDimConfig,
    w0: &'a [f64],
}

/// What a leaf becomes besides the clauses it clears.
#[derive(Debug, Clone, PartialEq)]
enum Rest {
    /// Nothing of the mask remains (every clause cleared or given up).
    Empty,
    /// A malformed bound: every clause of the mask is given up.
    NonFinite,
    /// Unsplittable: every remaining clause is given up.
    Resolution,
    /// The clip left an unsplittable, strictly smaller box: bound it again.
    Requeue { lo: Vec<f64>, hi: Vec<f64> },
    /// Bisect the (hull) box.
    Split {
        lo: Vec<f64>,
        hi: Vec<f64>,
        dim: usize,
        mid: f64,
        shrunk: bool,
    },
}

/// One leaf's decision.
#[derive(Debug, Clone, PartialEq)]
struct Decision {
    /// Clauses proved impossible on this leaf, with `true` when the clip
    /// (rather than a strict bound) proved it.
    cleared: Vec<(usize, bool)>,
    /// The clause given up on this leaf's centre, and the centre.
    candidate: Option<(usize, Vec<f64>)>,
    /// The clauses still open after the decision (ascending).
    remaining: Vec<usize>,
    rest: Rest,
    claim: ClaimMark,
}

/// The candidate test, then the split, for the clauses `open` (ascending) of
/// the box `[bl, bh]`: `lb(r)` / `grad(r)` give row `r`'s bound and input
/// multiplier (f64 bounds, or untrusted hints — they only choose where to
/// split). Never clears a clause.
#[allow(clippy::too_many_arguments)]
fn open_multi(
    ctx: &Ctx<'_>,
    bl: Vec<f64>,
    bh: Vec<f64>,
    shrunk: bool,
    mut open: Vec<usize>,
    lb: &dyn Fn(usize) -> f64,
    grad: &dyn Fn(usize) -> Vec<f64>,
    cleared: Vec<(usize, bool)>,
    claim: ClaimMark,
) -> Decision {
    let d = bl.len();
    let centre: Vec<f64> = bl.iter().zip(&bh).map(|(a, c)| 0.5 * (a + c)).collect();
    let mut candidate = None;
    if !open.is_empty() {
        // One point evaluation serves every clause (the same arithmetic as
        // `point_objective`, which evaluates the seed vector and dots it).
        let y = point_values(ctx.prog, &centre);
        let dot = |o: &[f64]| -> f64 { y.iter().zip(o).map(|(v, w)| v * w).sum() };
        if let Some(pos) = open.iter().position(|&c| {
            ctx.rows_of[c]
                .clone()
                .all(|r| dot(&ctx.objs[r]) <= ctx.thrs[r] - candidate_margin(ctx.thrs[r]))
        }) {
            let c = open.remove(pos);
            candidate = Some((c, centre.clone()));
        }
    }
    if open.is_empty() {
        return Decision {
            cleared,
            candidate,
            remaining: open,
            rest: Rest::Empty,
            claim,
        };
    }
    // Split along the open objective closest to closing.
    let rows: Vec<usize> = open.iter().flat_map(|&c| ctx.rows_of[c].clone()).collect();
    let nearest = first_argmax(rows.iter().map(|&r| lb(r) - ctx.thrs[r]));
    let mut grads = Vec::with_capacity(rows.len() * d);
    for &r in &rows {
        let g = grad(r);
        if g.len() == d {
            grads.extend(g);
        } else {
            grads.extend(std::iter::repeat_n(0.0, d));
        }
    }
    let dim = choose_split(ctx.cfg.split, &bl, &bh, ctx.w0, &grads, nearest);
    let mid = centre[dim];
    let rest = if bh[dim] - bl[dim] < ctx.cfg.min_width || !(mid > bl[dim] && mid < bh[dim]) {
        if shrunk {
            Rest::Requeue { lo: bl, hi: bh }
        } else {
            Rest::Resolution
        }
    } else {
        Rest::Split {
            lo: bl,
            hi: bh,
            dim,
            mid,
            shrunk,
        }
    };
    Decision {
        cleared,
        candidate,
        remaining: open,
        rest,
        claim,
    }
}

/// `decide_multi`: one leaf's decision from its OWN f64 bound `db` over the
/// rows `u` (sorted) of the objective table.
fn decide_multi(
    ctx: &Ctx<'_>,
    l: &[f64],
    h: &[f64],
    mask: &[usize],
    u: &[usize],
    db: &DomainBound,
    claim: ClaimMark,
) -> Decision {
    let d = l.len();
    let pos = |r: usize| u.binary_search(&r).ok();
    let malformed = db.lbs.len() != u.len()
        || db.lbs.iter().any(|v| v.is_nan())
        || mask
            .iter()
            .any(|&c| ctx.rows_of[c].clone().any(|r| pos(r).is_none()));
    if malformed {
        return Decision {
            cleared: Vec::new(),
            candidate: None,
            remaining: mask.to_vec(),
            rest: Rest::NonFinite,
            claim,
        };
    }
    // (1) strict closes, clauses ascending.
    let mut cleared: Vec<(usize, bool)> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for &c in mask {
        let closes = ctx.rows_of[c].clone().any(|r| {
            let p = pos(r).unwrap_or(usize::MAX);
            db.lbs.get(p).is_some_and(|&lb| lb > ctx.thrs[r])
        });
        if closes {
            cleared.push((c, false));
        } else {
            open.push(c);
        }
    }
    // (2) each open clause's own clip; the hull of the surviving boxes.
    let (mut bl, mut bh) = (l.to_vec(), h.to_vec());
    let mut shrunk = false;
    if ctx.cfg.clip && !open.is_empty() && !db.planes.is_empty() {
        let mut hull: Option<(Vec<f64>, Vec<f64>)> = None;
        let mut survivors = Vec::with_capacity(open.len());
        for &c in &open {
            let planes: Vec<ClipPlane<'_>> = db
                .planes
                .iter()
                .filter_map(|(i, a, cst)| {
                    let r = *u.get(*i)?;
                    ctx.rows_of[c].contains(&r).then(|| ClipPlane {
                        a: a.as_slice(),
                        c: *cst,
                        t: ctx.thrs[r],
                    })
                })
                .collect();
            let (mut cl, mut ch) = (l.to_vec(), h.to_vec());
            if clip_box(&mut cl, &mut ch, &planes).is_none() {
                cleared.push((c, true));
                continue;
            }
            survivors.push(c);
            hull = Some(match hull {
                None => (cl, ch),
                Some((hl, hh)) => (
                    hl.iter().zip(&cl).map(|(a, b)| a.min(*b)).collect(),
                    hh.iter().zip(&ch).map(|(a, b)| a.max(*b)).collect(),
                ),
            });
        }
        open = survivors;
        if let Some((hl, hh)) = hull {
            shrunk = (0..d).any(|k| hl[k] > l[k] || hh[k] < h[k]);
            bl = hl;
            bh = hh;
        }
    }
    cleared.sort_unstable();
    let lb = |r: usize| {
        pos(r)
            .and_then(|p| db.lbs.get(p).copied())
            .unwrap_or(f64::NEG_INFINITY)
    };
    let grad = |r: usize| -> Vec<f64> {
        pos(r)
            .and_then(|p| db.grads.get(p * d..(p + 1) * d))
            .map(<[f64]>::to_vec)
            .unwrap_or_default()
    };
    open_multi(ctx, bl, bh, shrunk, open, &lb, &grad, cleared, claim)
}

/// The legacy bounder over `n` leaves (`lo` / `hi` are `[n][d]`, `masks[b]`
/// the open clauses of leaf `b`): the run's chunks, each against the union
/// of its leaves' open clauses' rows. Per leaf, its rows `u` and bound.
#[allow(clippy::too_many_arguments)]
fn bound_leaves(
    ctx: &Ctx<'_>,
    lo: &[f64],
    hi: &[f64],
    masks: &[Vec<usize>],
    chunk: usize,
) -> Vec<(std::sync::Arc<[usize]>, DomainBound)> {
    let d = ctx.prog.d;
    let n = masks.len();
    let chunk = chunk.max(1);
    let starts: Vec<usize> = (0..n).step_by(chunk).collect();
    starts
        .par_iter()
        .flat_map_iter(|&s| {
            let m = chunk.min(n - s);
            let mut u: Vec<usize> = masks[s..s + m]
                .iter()
                .flatten()
                .flat_map(|&c| ctx.rows_of[c].clone())
                .collect();
            u.sort_unstable();
            u.dedup();
            let objs: Vec<Vec<f64>> = u.iter().map(|&r| ctx.objs[r].clone()).collect();
            let thrs: Vec<f64> = u.iter().map(|&r| ctx.thrs[r]).collect();
            let spec = AlphaSpec {
                iters: ctx.cfg.alpha_iters,
                thrs: &thrs,
            };
            let alpha = (ctx.cfg.alpha_iters > 0).then_some(&spec);
            let ch = Chunk::new(m, d, &lo[s * d..(s + m) * d], &hi[s * d..(s + m) * d]);
            let dbs = search::bound_chunk(ctx.prog, &ch, &objs, ctx.cfg.clip, alpha);
            let u: std::sync::Arc<[usize]> = std::sync::Arc::from(u);
            dbs.into_iter()
                .map(|db| (std::sync::Arc::clone(&u), db))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Decide `n` leaves with the legacy bounder.
fn legacy_decisions(
    ctx: &Ctx<'_>,
    lo: &[f64],
    hi: &[f64],
    masks: &[Vec<usize>],
    chunk: usize,
    claim: ClaimMark,
) -> Vec<Decision> {
    let d = ctx.prog.d;
    let bounds = bound_leaves(ctx, lo, hi, masks, chunk);
    if bounds.len() != masks.len() {
        return masks
            .iter()
            .map(|m| Decision {
                cleared: Vec::new(),
                candidate: None,
                remaining: m.clone(),
                rest: Rest::NonFinite,
                claim,
            })
            .collect();
    }
    bounds
        .par_iter()
        .enumerate()
        .map(|(b, (u, db))| {
            decide_multi(
                ctx,
                &lo[b * d..(b + 1) * d],
                &hi[b * d..(b + 1) * d],
                &masks[b],
                u,
                db,
                claim,
            )
        })
        .collect()
}

/// A searched wave: claimed leaves certified and decided, every other leaf
/// split on its hints. `None`: run the legacy path; `Some(Err)`: deadline.
fn bbe_decisions(
    run: &mut BbeRun,
    ctx: &Ctx<'_>,
    lo: &[f64],
    hi: &[f64],
    masks: &[Vec<usize>],
    chunk: usize,
    deadline: Instant,
) -> Option<Result<Vec<Decision>, LowDimStop>> {
    if !run.live() {
        return None;
    }
    let d = ctx.prog.d;
    let n = masks.len();
    let u32_of = |i: usize| u32::try_from(i).unwrap_or(u32::MAX);
    let mut off = vec![0u32];
    let mut rows: Vec<u32> = Vec::new();
    for m in masks {
        rows.extend(m.iter().flat_map(|&c| ctx.rows_of[c].clone()).map(u32_of));
        off.push(u32_of(rows.len()));
    }
    let ing = match run.search(0, lo, hi, n, &[], &off, &rows, deadline) {
        Searched::Hints(ing) => *ing,
        Searched::Deadline => return Some(Err(LowDimStop::Deadline)),
        Searched::Latched => return None,
    };
    let mut claimed: Vec<bool> = (0..n)
        .map(|b| {
            (off[b] as usize..off[b + 1] as usize)
                .any(|q| ing.shifted[q] > ctx.thrs[rows[q] as usize])
        })
        .collect();
    let t_cert = Instant::now();
    // Every unclaimed leaf: the candidate test, then a split on its hints.
    let unclaimed: Vec<usize> = (0..n).filter(|&b| !claimed[b]).collect();
    let opened: Vec<Decision> = unclaimed
        .par_iter()
        .map(|&b| {
            let q0 = off[b] as usize;
            let q1 = off[b + 1] as usize;
            let row_pos = |r: usize| (q0..q1).find(|&q| rows[q] as usize == r);
            let lb = |r: usize| row_pos(r).map_or(f64::NEG_INFINITY, |q| ing.hint_lb(q));
            let grad = |r: usize| -> Vec<f64> {
                row_pos(r).map_or_else(Vec::new, |q| ing.grads[q * d..(q + 1) * d].to_vec())
            };
            open_multi(
                ctx,
                lo[b * d..(b + 1) * d].to_vec(),
                hi[b * d..(b + 1) * d].to_vec(),
                false,
                masks[b].clone(),
                &lb,
                &grad,
                Vec::new(),
                ClaimMark::Unclaimed,
            )
        })
        .collect();
    // Review fix (an untrusted miss is never a terminal stop): an unclaimed
    // leaf whose open decision is `Resolution` would give up every clause
    // still open on it on a hint alone; it is certified like a claim instead
    // (the lane's f64 bound and the UNCHANGED `decide_multi`), so a clause is
    // given up at the resolution floor only when the legacy decision does.
    let mut open_of: Vec<Option<Decision>> = (0..n).map(|_| None).collect();
    for (&b, dec) in unclaimed.iter().zip(opened) {
        if dec.rest == Rest::Resolution {
            claimed[b] = true;
        } else {
            open_of[b] = Some(dec);
        }
    }
    let idx: Vec<usize> = (0..n).filter(|&b| claimed[b]).collect();
    let clo: Vec<f64> = idx
        .iter()
        .flat_map(|&b| lo[b * d..(b + 1) * d].iter().copied())
        .collect();
    let chi: Vec<f64> = idx
        .iter()
        .flat_map(|&b| hi[b * d..(b + 1) * d].iter().copied())
        .collect();
    let cmasks: Vec<Vec<usize>> = idx.iter().map(|&b| masks[b].clone()).collect();
    let mut certified =
        legacy_decisions(ctx, &clo, &chi, &cmasks, chunk, ClaimMark::Refuted).into_iter();
    run.stats.certify_s += t_cert.elapsed().as_secs_f64();
    let mut out = Vec::with_capacity(n);
    let (mut n_claimed, mut n_cert) = (0u64, 0u64);
    for (b, &c) in claimed.iter().enumerate() {
        let next = if c {
            certified.next()
        } else {
            open_of[b].take()
        };
        let mut dec = next?;
        if c {
            n_claimed += 1;
            if !dec.cleared.is_empty() {
                dec.claim = ClaimMark::Certified;
                n_cert += 1;
            }
        }
        out.push(dec);
    }
    run.note(
        n_claimed,
        n_cert,
        n_claimed - n_cert,
        (n as u64) - n_claimed,
    );
    Some(Ok(out))
}

/// [`prove_disjunction_admitted_with`] without a candidate callback (the run
/// continues after every candidate).
pub fn prove_disjunction_admitted(
    admission: &LaneAdmission,
    clauses: &[DisjunctionClause],
    deadline: Instant,
    cfg: &LowDimConfig,
) -> DisjunctionOutcome {
    prove_disjunction_admitted_with(
        admission,
        clauses,
        deadline,
        cfg,
        &mut |_: usize, _: &[f64]| false,
    )
}

/// Prove every clause of a same-box disjunction on ONE tree (see the module
/// docs). `on_candidate(c, x)` is called with each clause's violation
/// CANDIDATE (full-dimension input) as it is found; returning `true` (the
/// caller confirmed it on its trusted forward) stops the run, every clause
/// without an outcome `Unproved(Candidate)`. Proofs only: nothing here claims
/// a violation.
pub fn prove_disjunction_admitted_with(
    admission: &LaneAdmission,
    clauses: &[DisjunctionClause],
    deadline: Instant,
    cfg: &LowDimConfig,
    on_candidate: &mut dyn FnMut(usize, &[f64]) -> bool,
) -> DisjunctionOutcome {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::LowDim);
    let t0 = Instant::now();
    let k = clauses.len();
    let mut prog = admission.prog.clone();
    prog.refine_max = cfg.refine_max_width;
    prog.joint = cfg.joint_quadratic;
    let d = prog.d;
    let width = prog.sizes[prog.seed];
    let mut out: Vec<Option<LowDimOutcome>> = vec![None; k];
    let mut witnesses: Vec<Option<Vec<f64>>> = vec![None; k];
    let mut stats = vec![
        LowDimStats {
            threads: lane_threads(cfg),
            ..LowDimStats::default()
        };
        k
    ];
    // The union objective table (clause after clause) and each clause's rows.
    let mut objs: Vec<Vec<f64>> = Vec::new();
    let mut thrs: Vec<f64> = Vec::new();
    let mut rows_of: Vec<std::ops::Range<usize>> = Vec::with_capacity(k);
    let mut valid = vec![false; k];
    for (c, cl) in clauses.iter().enumerate() {
        let ok = !cl.objectives.is_empty()
            && cl.objectives.len() == cl.thresholds.len()
            && cl.thresholds.iter().all(|t| t.is_finite())
            && cl
                .objectives
                .iter()
                .all(|o| o.len() == width && o.iter().all(|v| v.is_finite()));
        let start = objs.len();
        if ok {
            objs.extend(
                cl.objectives
                    .iter()
                    .map(|o| o.iter().map(|&v| f64::from(v)).collect()),
            );
            thrs.extend(cl.thresholds.iter().map(|&t| f64::from(t)));
            valid[c] = true;
        } else {
            out[c] = Some(LowDimOutcome::Declined(
                "malformed disjunction clause".into(),
            ));
        }
        rows_of.push(start..objs.len());
    }
    let (lo0, hi0) = (admission.lo0.clone(), admission.hi0.clone());
    let w0: Vec<f64> = lo0.iter().zip(&hi0).map(|(l, h)| h - l).collect();
    let batch = cfg.batch.max(1);
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
    let words = k.div_ceil(64).max(1);
    let mut given_up = vec![false; k];
    let mut open = vec![0u64; k];
    let mut q_lo: Vec<f64> = Vec::new();
    let mut q_hi: Vec<f64> = Vec::new();
    let mut q_depth: Vec<u32> = Vec::new();
    let mut q_mask: Vec<u64> = Vec::new();
    let mut root_mask = vec![0u64; words];
    for c in (0..k).filter(|&c| valid[c]) {
        root_mask[c / 64] |= 1u64 << (c % 64);
        open[c] = 1;
    }
    let mut dst = DisjunctionStats {
        clauses: k,
        ..DisjunctionStats::default()
    };
    let ok_box = d > 0
        && lo0.len() == d
        && hi0.len() == d
        && lo0.iter().chain(&hi0).all(|v| v.is_finite())
        && lo0.iter().zip(&hi0).all(|(l, h)| l <= h);
    if ok_box && root_mask.iter().any(|&w| w != 0) {
        q_lo.extend_from_slice(&lo0);
        q_hi.extend_from_slice(&hi0);
        q_depth.push(0);
        q_mask.extend_from_slice(&root_mask);
    } else if !ok_box {
        for c in 0..k {
            if out[c].is_none() {
                out[c] = Some(LowDimOutcome::Declined(
                    "input box is not a finite, ordered box".into(),
                ));
            }
        }
    }
    let ctx = Ctx {
        prog: &prog,
        objs: &objs,
        thrs: &thrs,
        rows_of: &rows_of,
        cfg,
        w0: &w0,
    };
    // M8a: the untrusted wave search (probe: the root box, every row).
    let mut bbe_run = (cfg.batch_engine == LowDimBatchEngine::Bbe && !objs.is_empty()).then(|| {
        let rows: Vec<u32> = (0..objs.len())
            .map(|r| u32::try_from(r).unwrap_or(u32::MAX))
            .collect();
        BbeRun::arm(
            Scheduler::Disjunction,
            &[vec![&prog]],
            objs.clone(),
            &[ProbeSpec {
                family: 0,
                clause: 0,
                lo: &lo0,
                hi: &hi0,
                rows,
            }],
            cfg,
        )
    });
    let mask_of = |m: &[u64]| -> Vec<usize> {
        (0..k)
            .filter(|&c| m[c / 64] & (1u64 << (c % 64)) != 0)
            .collect()
    };
    let expand = |x: &[f64]| -> Vec<f64> {
        admission
            .pinned
            .as_ref()
            .map_or_else(|| x.to_vec(), |f| f.expand(x))
    };
    // B2.2: the write-only proof tape (Proof profile only; `None` under
    // Competition). Every leaf's cleared clauses (each an ANY group of the
    // clause's rows over the union table) and every hull shrink's slabs
    // (closed for every clause still open on the leaf) are recorded as
    // full-input boxes; deposited when some clause is verified. Only rows over
    // the network OUTPUT are instance objectives.
    let tape = if prog.seed == prog.out {
        let (tlo, thi) = admission
            .pinned
            .as_ref()
            .map_or((lo0.as_slice(), hi0.as_slice()), |f| f.full_box());
        crate::certify::lane_proof::TapeRecorder::for_boxes(
            "lowdim_disjunction",
            tlo,
            thi,
            &objs,
            &thrs,
        )
    } else {
        None
    };
    let group_of = |c: usize| -> Vec<u32> {
        rows_of[c]
            .clone()
            .filter_map(|r| u32::try_from(r).ok())
            .collect()
    };
    let full_box = |l: &[f64], h: &[f64]| -> (Vec<f64>, Vec<f64>) {
        admission
            .pinned
            .as_ref()
            .map_or_else(|| (l.to_vec(), h.to_vec()), |f| f.embed_box(l, h))
    };
    let mut stop: Option<LowDimStop> = None;
    'waves: loop {
        let queued = q_depth.len();
        if queued == 0 {
            break;
        }
        if Instant::now() >= deadline {
            stop = Some(LowDimStop::Deadline);
            break;
        }
        if queued > cfg.max_queue {
            stop = Some(LowDimStop::Queue);
            break;
        }
        let take = batch.min(queued);
        let at = queued - take;
        let b_lo = q_lo.split_off(at * d);
        let b_hi = q_hi.split_off(at * d);
        let b_depth = q_depth.split_off(at);
        let b_mask = q_mask.split_off(at * words);
        // Lazy drops: clauses given up leave every queued mask here.
        let mut lo = Vec::with_capacity(take * d);
        let mut hi = Vec::with_capacity(take * d);
        let mut depth = Vec::with_capacity(take);
        let mut masks: Vec<Vec<usize>> = Vec::with_capacity(take);
        for i in 0..take {
            let m: Vec<usize> = mask_of(&b_mask[i * words..(i + 1) * words])
                .into_iter()
                .filter(|&c| !given_up[c])
                .collect();
            if m.is_empty() {
                dst.mask_drops += 1;
                continue;
            }
            lo.extend_from_slice(&b_lo[i * d..(i + 1) * d]);
            hi.extend_from_slice(&b_hi[i * d..(i + 1) * d]);
            depth.push(b_depth[i]);
            masks.push(m);
        }
        if masks.is_empty() {
            continue;
        }
        let n = masks.len();
        dst.waves += 1;
        dst.domains += n as u64;
        let searched = match bbe_run.as_mut() {
            Some(run) => bbe_decisions(run, &ctx, &lo, &hi, &masks, chunk, deadline),
            None => None,
        };
        let decisions = match searched {
            Some(Ok(decs)) => decs,
            Some(Err(s)) => {
                stop = Some(s);
                break;
            }
            None => legacy_decisions(&ctx, &lo, &hi, &masks, chunk, ClaimMark::None),
        };
        if decisions.len() != n {
            stop = Some(LowDimStop::NonFinite);
            break;
        }
        // Merge in leaf order.
        for (b, dec) in decisions.into_iter().enumerate() {
            let (l, h) = (&lo[b * d..(b + 1) * d], &hi[b * d..(b + 1) * d]);
            let leaf_vol = box_volume(l, h, &w0);
            if let Some(tape) = &tape {
                let (fl, fh) = full_box(l, h);
                if !dec.cleared.is_empty() {
                    let by_clip = dec.cleared.iter().any(|&(_, clip)| clip);
                    let groups: Vec<Vec<u32>> =
                        dec.cleared.iter().map(|&(c, _)| group_of(c)).collect();
                    tape.note_box(
                        &fl,
                        &fh,
                        if by_clip {
                            crate::certify::lane_proof::TapeClose::Clip
                        } else {
                            crate::certify::lane_proof::TapeClose::Bound
                        },
                        groups,
                        f64::NAN,
                    );
                }
                let shrunk = match &dec.rest {
                    Rest::Requeue { lo: sl, hi: sh }
                    | Rest::Split {
                        lo: sl,
                        hi: sh,
                        shrunk: true,
                        ..
                    } => Some((sl, sh)),
                    _ => None,
                };
                if let Some((sl, sh)) = shrunk {
                    if !dec.remaining.is_empty() {
                        let (ql, qh) = full_box(sl, sh);
                        let groups: Vec<Vec<u32>> =
                            dec.remaining.iter().map(|&c| group_of(c)).collect();
                        tape.note_clip(&fl, &fh, &ql, &qh, &groups);
                    }
                }
            }
            for &c in &masks[b] {
                if out[c].is_some() {
                    continue;
                }
                stats[c].domains += 1;
                match dec.claim {
                    ClaimMark::Certified => {
                        stats[c].bbe_claimed += 1;
                        stats[c].bbe_certified += 1;
                    }
                    ClaimMark::Refuted => {
                        stats[c].bbe_claimed += 1;
                        stats[c].bbe_refuted += 1;
                    }
                    ClaimMark::Unclaimed | ClaimMark::None => {}
                }
            }
            for &(c, by_clip) in &dec.cleared {
                if out[c].is_some() || given_up[c] {
                    continue;
                }
                stats[c].closed += 1;
                stats[c].clip_closed += u64::from(by_clip);
                stats[c].closed_volume += leaf_vol;
                open[c] = open[c].saturating_sub(1);
                if open[c] == 0 {
                    let mut s = stats[c];
                    s.secs = t0.elapsed().as_secs_f64();
                    s.waves = dst.waves;
                    out[c] = Some(LowDimOutcome::Verified(s));
                }
            }
            if let Some((c, x)) = dec.candidate {
                if out[c].is_none() && !given_up[c] {
                    given_up[c] = true;
                    let mut s = stats[c];
                    s.secs = t0.elapsed().as_secs_f64();
                    s.waves = dst.waves;
                    out[c] = Some(LowDimOutcome::Unproved(s, LowDimStop::Candidate));
                    let full = expand(&x);
                    let confirmed = on_candidate(c, &full);
                    witnesses[c] = Some(full);
                    if confirmed {
                        stop = Some(LowDimStop::Candidate);
                        break 'waves;
                    }
                }
            }
            let remaining: Vec<usize> = dec
                .remaining
                .iter()
                .copied()
                .filter(|&c| out[c].is_none() && !given_up[c])
                .collect();
            let give_up_all = |stop: LowDimStop,
                               cs: &[usize],
                               out: &mut Vec<Option<LowDimOutcome>>,
                               given_up: &mut Vec<bool>,
                               stats: &[LowDimStats]| {
                for &c in cs {
                    if out[c].is_none() {
                        let mut s = stats[c];
                        s.secs = t0.elapsed().as_secs_f64();
                        out[c] = Some(LowDimOutcome::Unproved(s, stop));
                    }
                    given_up[c] = true;
                }
            };
            match dec.rest {
                Rest::Empty => {}
                Rest::NonFinite => {
                    give_up_all(
                        LowDimStop::NonFinite,
                        &remaining,
                        &mut out,
                        &mut given_up,
                        &stats,
                    );
                }
                Rest::Resolution => {
                    give_up_all(
                        LowDimStop::Resolution,
                        &remaining,
                        &mut out,
                        &mut given_up,
                        &stats,
                    );
                }
                Rest::Requeue { lo: rl, hi: rh } => {
                    if remaining.is_empty() {
                        continue;
                    }
                    let removed = (leaf_vol - box_volume(&rl, &rh, &w0)).max(0.0);
                    let mut m = vec![0u64; words];
                    for &c in &remaining {
                        m[c / 64] |= 1u64 << (c % 64);
                        stats[c].clip_shrunk += 1;
                        stats[c].closed_volume += removed;
                    }
                    q_lo.extend_from_slice(&rl);
                    q_hi.extend_from_slice(&rh);
                    q_depth.push(depth[b]);
                    q_mask.extend_from_slice(&m);
                }
                Rest::Split {
                    lo: sl,
                    hi: sh,
                    dim,
                    mid,
                    shrunk,
                } => {
                    if remaining.is_empty() {
                        continue;
                    }
                    let removed = if shrunk {
                        (leaf_vol - box_volume(&sl, &sh, &w0)).max(0.0)
                    } else {
                        0.0
                    };
                    let dp = depth[b] + 1;
                    let mut m = vec![0u64; words];
                    for &c in &remaining {
                        m[c / 64] |= 1u64 << (c % 64);
                        open[c] += 1;
                        stats[c].clip_shrunk += u64::from(shrunk);
                        stats[c].closed_volume += removed;
                        stats[c].max_depth = stats[c].max_depth.max(dp);
                    }
                    q_lo.extend_from_slice(&sl);
                    q_hi.extend_from_slice(&sh);
                    let li = q_hi.len() - d + dim;
                    q_hi[li] = mid;
                    q_lo.extend_from_slice(&sl);
                    q_hi.extend_from_slice(&sh);
                    let ri = q_lo.len() - d + dim;
                    q_lo[ri] = mid;
                    q_depth.push(dp);
                    q_depth.push(dp);
                    q_mask.extend_from_slice(&m);
                    q_mask.extend_from_slice(&m);
                }
            }
        }
    }
    dst.secs = t0.elapsed().as_secs_f64();
    // Every clause without an outcome is unproved: the run stopped (or, if
    // the queue drained with the clause still counted open — which the
    // counters exclude — conservatively so).
    let per_clause: Vec<LowDimOutcome> = out
        .into_iter()
        .zip(&stats)
        .map(|(o, s)| {
            o.unwrap_or_else(|| {
                let mut s = *s;
                s.secs = dst.secs;
                s.waves = dst.waves;
                LowDimOutcome::Unproved(s, stop.unwrap_or(LowDimStop::Deadline))
            })
        })
        .collect();
    if let Some(tape) = &tape {
        if per_clause
            .iter()
            .any(|o| matches!(o, LowDimOutcome::Verified(_)))
        {
            tape.deposit(false);
        }
    }
    dst.closed_volume = per_clause
        .iter()
        .map(|o| match o {
            LowDimOutcome::Verified(s) | LowDimOutcome::Unproved(s, _) => s.closed_volume,
            LowDimOutcome::Declined(_) => 0.0,
        })
        .collect();
    if let Some(run) = &bbe_run {
        dst.bbe = run.stats;
        run.emit();
    }
    emit_line(&dst, &per_clause);
    DisjunctionOutcome {
        per_clause,
        witnesses,
        stats: dst,
    }
}

thread_local! {
    /// The last `[lowdim-disjunction] done` line this THREAD printed.
    /// Telemetry only.
    static LAST_SUMMARY: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Take the last `[lowdim-disjunction] done ...` line printed on this thread
/// (`None` when no disjunction tree ran here since the last take): the
/// receipt a caller can check that ONE tree — not one per clause — ran.
pub fn take_last_disjunction_summary() -> Option<String> {
    LAST_SUMMARY.with(|c| c.borrow_mut().take())
}

/// The `[lowdim-disjunction] done` line (fixed schema), printed by every run.
fn emit_line(st: &DisjunctionStats, per_clause: &[LowDimOutcome]) {
    let proved = per_clause
        .iter()
        .filter(|o| matches!(o, LowDimOutcome::Verified(_)))
        .count();
    let vol: f64 = st.closed_volume.iter().sum();
    let line = format!(
        "[lowdim-disjunction] done clauses={} proved={proved} domains={} waves={} mask_drops={} \
         vol={vol:.6} secs={:.3}",
        st.clauses, st.domains, st.waves, st.mask_drops, st.secs
    );
    eprintln!("{line}");
    LAST_SUMMARY.with(|c| *c.borrow_mut() = Some(line));
}

/// TEST FACE: a decision context over a given table.
#[cfg(test)]
pub(super) fn test_ctx<'a>(
    prog: &'a Program,
    objs: &'a [Vec<f64>],
    thrs: &'a [f64],
    rows_of: &'a [std::ops::Range<usize>],
    cfg: &'a LowDimConfig,
    w0: &'a [f64],
) -> Ctx<'a> {
    Ctx {
        prog,
        objs,
        thrs,
        rows_of,
        cfg,
        w0,
    }
}

/// TEST FACE: the clauses `decide_multi` clears on a leaf, from a given bound.
#[cfg(test)]
pub(super) fn test_decide(
    ctx: &Ctx<'_>,
    l: &[f64],
    h: &[f64],
    mask: &[usize],
    u: &[usize],
    db: &DomainBound,
) -> Vec<usize> {
    decide_multi(ctx, l, h, mask, u, db, ClaimMark::None)
        .cleared
        .iter()
        .map(|&(c, _)| c)
        .collect()
}

/// TEST FACE: one leaf bounded by the legacy bounder and decided: the
/// cleared clauses, the clauses still open, and the box the open clauses
/// continue in (the hull; `None` when nothing continues).
#[cfg(test)]
#[allow(clippy::type_complexity)]
pub(super) fn test_decide_full(
    ctx: &Ctx<'_>,
    l: &[f64],
    h: &[f64],
    mask: &[usize],
) -> (Vec<usize>, Vec<usize>, Option<(Vec<f64>, Vec<f64>)>) {
    let dec = legacy_decisions(ctx, l, h, &[mask.to_vec()], 1, ClaimMark::None)
        .pop()
        .expect("one decision");
    let hull = match &dec.rest {
        Rest::Split { lo, hi, .. } | Rest::Requeue { lo, hi } => Some((lo.clone(), hi.clone())),
        _ => None,
    };
    (
        dec.cleared.iter().map(|&(c, _)| c).collect(),
        dec.remaining.clone(),
        hull,
    )
}

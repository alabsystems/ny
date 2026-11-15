// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lowdim-bundle (4) — ONE shared worklist across the fold's families
//! (study_5 idea 6 / WAVE6 E3; nn4sys `mscn_2048d_dual` 1_2260 .. 1_6670).
//!
//! # Why
//!
//! `prove_clauses_folded` runs its structure families one after another
//! until the deadline, and a clause is abandoned after 200,000 domains
//! (`FAMILY_CLAUSE_DOMAINS`). On the six nn4sys dual rows the field closes,
//! 2-52 residual clauses are left at the deadline (`diag6/longtail.md` §2.10).
//! Vibecheck keeps one worklist of (subbox, leaf) across every unclosed
//! subbox (`_multi_sub_input_split_bab`). This module is that design over the
//! lane's own certificates.
//!
//! # Design
//!
//! Every clause keeps its own LIFO queue. Each wave gives every live clause
//! an equal share of the batch (at least one leaf), so no clause — and no
//! family — can starve another, and the per-clause domain cap goes. The
//! wave's leaves are bounded grouped by family (a family shares one forward
//! pass per chunk, as in `bound_batch_family`), families in parallel. With
//! `LowDimConfig::clip` armed, single-objective families (`Combine::Each`)
//! also take stage 3's relaxed clip with each leaf's OWN planes.
//!
//! # Soundness
//!
//! * The per-leaf bound is exactly the family core's (`objective_bounds_passes`
//!   over the family forward; the planes are each pass's own, on that leaf's
//!   box). A clause is VERIFIED only when every one of its leaves closed —
//!   by `FamClause::closes` (the strict `bound > threshold`) or by the clip
//!   emptying the leaf. Splits keep both halves of the (clipped) box.
//! * The clip is taken only for `Combine::Each`: there objective `i` is
//!   conjunct `i`'s sufficient target, so a counterexample satisfies
//!   `objective_i . z <= t_i` for every `i`, and every face `search::clip_box`
//!   removes is certified by one plane over the unclipped box, outward
//!   rounded. A `SigmoidSum` conjunct is never clipped.
//! * Scheduling decides only WHICH leaves are bounded when; it never changes
//!   a leaf's bound.
//! * MEMORY: the queued boxes are capped by `cfg.max_queue` and by a byte
//!   budget (`SHARED_QUEUE_BYTES`, at most a quarter of the process's live
//!   memory headroom — ny has no governor without a cgroup); a cap stops
//!   every open clause as `Unproved(Queue)` (never a proof).

use std::time::Instant;

use rayon::prelude::*;

use super::bbe::ClaimMark;
use super::search::{choose_split, clip_box, AlphaNote, ClipPlane, DomainBound, Fate};
use super::{
    box_volume, forward, lane_threads, objective_bounds_passes, same_family, Chunk, Combine, Fam,
    FamClause, LowDimConfig, LowDimOutcome, LowDimStats, LowDimStop, Pass, Program, CHUNK_BYTES,
};

/// Largest queued-box memory the shared worklist may hold (both box corners
/// plus the depth of every queued leaf).
pub(super) const SHARED_QUEUE_BYTES: usize = 1 << 30;

/// Counters of one shared run (the `[lowdim-shared] done` line).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SharedStats {
    pub(crate) families: usize,
    pub(crate) clauses: usize,
    pub(crate) waves: u64,
    pub(crate) domains: u64,
    pub(crate) verified: usize,
    pub(crate) clip_closed: u64,
    pub(crate) queue_peak: usize,
    pub(crate) mem_stop: bool,
    pub(crate) secs: f64,
    /// M8a router progress: the sum over clauses of `closed_volume`.
    pub(crate) vol: f64,
}

/// One clause's search state.
struct Lane {
    out: Option<LowDimOutcome>,
    stats: LowDimStats,
    lo: Vec<f64>,
    hi: Vec<f64>,
    depth: Vec<u32>,
    /// Leaves not yet closed (queued or in flight).
    open: u64,
}

impl Lane {
    fn give_up(&mut self, stop: LowDimStop, t0: Instant) {
        if self.out.is_none() {
            let mut st = self.stats;
            st.secs = t0.elapsed().as_secs_f64();
            self.out = Some(LowDimOutcome::Unproved(st, stop));
        }
        self.lo = Vec::new();
        self.hi = Vec::new();
        self.depth = Vec::new();
    }
}

/// One leaf of a wave.
struct Item {
    f: usize,
    c: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    depth: u32,
}

/// The same admission `prove_family_core` applies to a clause of a family
/// whose first clause is `first`.
fn clause_ok(first: &FamClause, cl: &FamClause, d: usize, width: usize) -> bool {
    let kind_ok = match &cl.combine {
        Combine::Each => true,
        Combine::SigmoidSum { centre } => {
            centre.len() == cl.thrs.len()
                && centre.iter().all(|(t, s)| t.is_finite() && s.is_finite())
        }
    };
    cl.lo.len() == d
        && cl.hi.len() == d
        && same_family(first, cl)
        && kind_ok
        && cl.lo.iter().chain(&cl.hi).all(|v| v.is_finite())
        && cl.lo.iter().zip(&cl.hi).all(|(l, h)| l <= h)
        && !cl.objs.is_empty()
        && cl.objs.len() == cl.per_conjunct() * cl.thrs.len()
        && cl
            .objs
            .iter()
            .all(|o| o.len() == width && o.iter().all(|v| v.is_finite()))
        && cl.thrs.iter().all(|t| t.is_finite())
}

/// The queued-box byte budget: [`SHARED_QUEUE_BYTES`], or a quarter of the
/// live memory headroom when that is smaller.
fn queue_budget() -> usize {
    match crate::network::crown_memory::process_memory_headroom_bytes() {
        Some(h) => usize::try_from(h / 4)
            .unwrap_or(usize::MAX)
            .min(SHARED_QUEUE_BYTES),
        None => SHARED_QUEUE_BYTES,
    }
}

/// Bound a group of one family's leaves: one forward per chunk, then every
/// objective's backward passes; with `planes`, each pass's own plane per
/// leaf. With `planes == false` the bounds are exactly `bound_batch_family`'s.
pub(super) fn bound_group(
    progs: &[&Program],
    which: &[usize],
    lo: &[f64],
    hi: &[f64],
    objs: &[&[Vec<f64>]],
    chunk: usize,
    planes: bool,
) -> Vec<DomainBound> {
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
            let per_k: Vec<(Vec<f64>, Vec<f64>, Vec<Pass>)> = (0..kmax)
                .map(|k| {
                    let ob: Vec<&[f64]> = (0..m)
                        .map(|b| objs[s + b].get(k).map_or(&zero[..], |v| &v[..]))
                        .collect();
                    objective_bounds_passes(fam, &vals, &ch, base.seed, &ob, &poison, planes)
                })
                .collect();
            (0..m)
                .map(|b| {
                    let nk = objs[s + b].len();
                    let lbs = per_k[..nk].iter().map(|(lb, _, _)| lb[b]).collect();
                    let grads = per_k[..nk]
                        .iter()
                        .flat_map(|(_, g, _)| g[b * d..(b + 1) * d].iter().copied())
                        .collect();
                    let mut kept = Vec::new();
                    if planes {
                        for (k, (_, _, passes)) in per_k[..nk].iter().enumerate() {
                            for pass in passes {
                                if pass.plane_ok(b, d) {
                                    kept.push((
                                        k,
                                        pass.grad[b * d..(b + 1) * d].to_vec(),
                                        pass.pconst[b],
                                    ));
                                }
                            }
                        }
                    }
                    DomainBound {
                        lbs,
                        grads,
                        planes: kept,
                        joint_hits: 0,
                        alpha: AlphaNote::default(),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// One leaf's fate, from its own box and bounds alone: a malformed bound
/// gives the clause up; a closing conjunct closes the leaf; the armed clip
/// (single-objective families only) shrinks or empties it with the leaf's own
/// planes; a centre violating every conjunct gives the clause up as a
/// candidate; else the split rule bisects the (clipped) box, or requeues a
/// strictly shrunk unsplittable one. Without the clip this is
/// `prove_family_core`'s decision exactly.
pub(super) fn decide(
    cl: &FamClause,
    w0: &[f64],
    l: &[f64],
    h: &[f64],
    db: &DomainBound,
    cfg: &LowDimConfig,
    clip: bool,
) -> Fate {
    if db.lbs.len() != cl.objs.len() || db.lbs.iter().any(|v| v.is_nan()) {
        return Fate::NonFinite;
    }
    if cl.closes(&db.lbs) {
        return Fate::Closed;
    }
    let (mut bl, mut bh) = (l.to_vec(), h.to_vec());
    let mut shrunk = false;
    if clip && cl.combine == Combine::Each && !db.planes.is_empty() {
        let planes: Vec<ClipPlane<'_>> = db
            .planes
            .iter()
            .filter_map(|(i, a, c)| {
                cl.thrs.get(*i).map(|&t| ClipPlane {
                    a: a.as_slice(),
                    c: *c,
                    t,
                })
            })
            .collect();
        match clip_box(&mut bl, &mut bh, &planes) {
            None => return Fate::ClipClosed,
            Some(s) => shrunk = s,
        }
    }
    open_decide(cl, w0, bl, bh, shrunk, &db.grads, &db.lbs, cfg)
}

/// The OPEN part of [`decide`] (extracted verbatim, M8a): a candidate
/// centre gives the clause up; else the split rule bisects the (clipped) box
/// along the objective [`FamClause::split_objective`] picks from `lbs`
/// (steered by `grads`), or requeues / stops at Resolution. It NEVER closes:
/// `lbs` and `grads` only choose where to split, so the M8a search may pass
/// its untrusted hints here.
#[allow(clippy::too_many_arguments)]
pub(super) fn open_decide(
    cl: &FamClause,
    w0: &[f64],
    bl: Vec<f64>,
    bh: Vec<f64>,
    shrunk: bool,
    grads: &[f64],
    lbs: &[f64],
    cfg: &LowDimConfig,
) -> Fate {
    let centre: Vec<f64> = bl.iter().zip(&bh).map(|(a, b)| 0.5 * (a + b)).collect();
    if cl.is_candidate(&centre) {
        return Fate::Candidate(centre);
    }
    let nearest = cl.split_objective(lbs);
    let dim = choose_split(cfg.split, &bl, &bh, w0, grads, nearest);
    let mid = centre[dim];
    if bh[dim] - bl[dim] < cfg.min_width || !(mid > bl[dim] && mid < bh[dim]) {
        return if shrunk {
            Fate::Requeue { lo: bl, hi: bh }
        } else {
            Fate::Resolution
        };
    }
    Fate::Split {
        lo: bl,
        hi: bh,
        dim,
        mid,
        shrunk,
    }
}

/// Prove every clause of every family on ONE shared worklist. `families[f]`
/// is a group of clauses of one structure (as `prove_family_core` takes);
/// the result has one outcome per clause, in the same shape. Proofs only: a
/// candidate, the resolution floor, the queue or memory cap, or the deadline
/// gives a clause up (`Unproved`), and a malformed clause is `Declined`.
pub(crate) fn prove_families_shared(
    mut families: Vec<Vec<FamClause>>,
    deadline: Instant,
    cfg: &LowDimConfig,
) -> (Vec<Vec<LowDimOutcome>>, SharedStats) {
    let t0 = Instant::now();
    let mut st = SharedStats {
        families: families.len(),
        ..SharedStats::default()
    };
    for fam in &mut families {
        for c in fam.iter_mut() {
            c.prog.refine_max = cfg.refine_max_width;
            c.prog.joint = cfg.joint_quadratic;
        }
    }
    // Per family: input dimension, chunk (memory-clamped as in the family
    // core) and each clause's root widths; per clause: its lane.
    let mut dims = Vec::with_capacity(families.len());
    let mut chunks = Vec::with_capacity(families.len());
    let mut w0s: Vec<Vec<Vec<f64>>> = Vec::with_capacity(families.len());
    let mut lanes: Vec<Vec<Lane>> = Vec::with_capacity(families.len());
    for fam in &families {
        let (d, width, chunk) = match fam.first() {
            Some(first) => {
                let p = &first.prog;
                let d = p.d;
                let per_domain_bytes = p.total.saturating_mul(d + 1).saturating_mul(16).max(1);
                (
                    d,
                    p.sizes.get(p.seed).copied().unwrap_or(0),
                    cfg.chunk
                        .max(1)
                        .min((CHUNK_BYTES / per_domain_bytes).max(1)),
                )
            }
            None => (0, 0, 1),
        };
        let mut ls = Vec::with_capacity(fam.len());
        for cl in fam {
            let mut lane = Lane {
                out: None,
                stats: LowDimStats {
                    threads: lane_threads(cfg),
                    ..LowDimStats::default()
                },
                lo: Vec::new(),
                hi: Vec::new(),
                depth: Vec::new(),
                open: 0,
            };
            let ok = fam
                .first()
                .is_some_and(|first| clause_ok(first, cl, d, width));
            if ok {
                lane.lo.clone_from(&cl.lo);
                lane.hi.clone_from(&cl.hi);
                lane.depth.push(0);
                lane.open = 1;
            } else {
                lane.out = Some(LowDimOutcome::Declined("malformed family clause".into()));
            }
            ls.push(lane);
        }
        st.clauses += fam.len();
        dims.push(d);
        chunks.push(chunk);
        w0s.push(
            fam.iter()
                .map(|cl| cl.lo.iter().zip(&cl.hi).map(|(l, h)| h - l).collect())
                .collect(),
        );
        lanes.push(ls);
    }
    let batch = cfg.batch.max(1);
    // M8a: the untrusted wave search, armed only under
    // `lowdim_batch_engine: bbe` (never consulted otherwise).
    let mut bbe = (cfg.batch_engine == super::LowDimBatchEngine::Bbe).then(|| {
        let live: Vec<Vec<bool>> = lanes
            .iter()
            .map(|ls| ls.iter().map(|l| l.out.is_none()).collect())
            .collect();
        super::bbe::arm_shared(&families, &live, cfg)
    });
    loop {
        let live: Vec<(usize, usize)> = lanes
            .iter()
            .enumerate()
            .flat_map(|(f, ls)| {
                ls.iter()
                    .enumerate()
                    .filter(|(_, l)| l.out.is_none() && !l.depth.is_empty())
                    .map(move |(c, _)| (f, c))
            })
            .collect();
        if live.is_empty() {
            break;
        }
        let queued: usize = live.iter().map(|&(f, c)| lanes[f][c].depth.len()).sum();
        let queue_bytes: usize = live
            .iter()
            .map(|&(f, c)| lanes[f][c].depth.len().saturating_mul(16 * dims[f] + 4))
            .sum();
        st.queue_peak = st.queue_peak.max(queued);
        let stop = if Instant::now() >= deadline {
            Some(LowDimStop::Deadline)
        } else if queued > cfg.max_queue {
            Some(LowDimStop::Queue)
        } else if queue_bytes > queue_budget() {
            st.mem_stop = true;
            Some(LowDimStop::Queue)
        } else {
            None
        };
        if let Some(stop) = stop {
            for &(f, c) in &live {
                lanes[f][c].give_up(stop, t0);
            }
            break;
        }
        // Take an equal share from the END (LIFO) of every live clause.
        let share = (batch / live.len()).max(1);
        let mut items: Vec<Item> = Vec::new();
        for &(f, c) in &live {
            let d = dims[f];
            let lane = &mut lanes[f][c];
            let k = share.min(lane.depth.len());
            let at = lane.depth.len() - k;
            let lo = lane.lo.split_off(at * d);
            let hi = lane.hi.split_off(at * d);
            let depth = lane.depth.split_off(at);
            for (i, dp) in depth.into_iter().enumerate() {
                items.push(Item {
                    f,
                    c,
                    lo: lo[i * d..(i + 1) * d].to_vec(),
                    hi: hi[i * d..(i + 1) * d].to_vec(),
                    depth: dp,
                });
            }
        }
        // Group by family (items are in family order), then bound and decide
        // every group in parallel.
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        for (k, it) in items.iter().enumerate() {
            match groups.last_mut() {
                Some((f, idxs)) if *f == it.f => idxs.push(k),
                _ => groups.push((it.f, vec![k])),
            }
        }
        // M8a: with the wave search live, the groups are searched in order
        // and only their claims are re-bounded by `bound_group` (see
        // `bbe::group_wave`); a latch runs this wave (and the rest) legacy.
        let searched: Option<Vec<Vec<(usize, Fate, ClaimMark)>>> = match bbe.as_mut() {
            Some((run, obj_base)) if run.live() => {
                let gins: Vec<super::bbe::GroupIn> = groups
                    .iter()
                    .map(|(f, idxs)| super::bbe::GroupIn {
                        f: *f,
                        which: idxs.iter().map(|&k| items[k].c).collect(),
                        lo: idxs
                            .iter()
                            .flat_map(|&k| items[k].lo.iter().copied())
                            .collect(),
                        hi: idxs
                            .iter()
                            .flat_map(|&k| items[k].hi.iter().copied())
                            .collect(),
                    })
                    .collect();
                match super::bbe::group_wave(
                    run, &families, obj_base, &gins, &dims, &chunks, &w0s, cfg, deadline,
                ) {
                    super::bbe::WaveResult::Done(per) => Some(
                        groups
                            .iter()
                            .zip(per)
                            .map(|((_, idxs), fates)| {
                                idxs.iter()
                                    .zip(fates)
                                    .map(|(&k, (fate, mark))| (k, fate, mark))
                                    .collect()
                            })
                            .collect(),
                    ),
                    super::bbe::WaveResult::Deadline => {
                        for &(f, c) in &live {
                            lanes[f][c].give_up(LowDimStop::Deadline, t0);
                        }
                        break;
                    }
                    super::bbe::WaveResult::Legacy => None,
                }
            }
            _ => None,
        };
        let decided: Vec<Vec<(usize, Fate, ClaimMark)>> = if let Some(done) = searched {
            done
        } else {
            groups
                .par_iter()
                .map(|(f, idxs)| -> Vec<(usize, Fate, ClaimMark)> {
                    let fam = &families[*f];
                    let progs: Vec<&Program> = fam.iter().map(|cl| &cl.prog).collect();
                    let d = dims[*f];
                    let which: Vec<usize> = idxs.iter().map(|&k| items[k].c).collect();
                    let lo: Vec<f64> = idxs
                        .iter()
                        .flat_map(|&k| items[k].lo.iter().copied())
                        .collect();
                    let hi: Vec<f64> = idxs
                        .iter()
                        .flat_map(|&k| items[k].hi.iter().copied())
                        .collect();
                    let objs: Vec<&[Vec<f64>]> = idxs
                        .iter()
                        .map(|&k| fam[items[k].c].objs.as_slice())
                        .collect();
                    let clip = cfg.clip && fam.first().is_some_and(|c| c.combine == Combine::Each);
                    if progs.is_empty() || d == 0 {
                        return idxs
                            .iter()
                            .map(|&k| (k, Fate::NonFinite, ClaimMark::None))
                            .collect();
                    }
                    let bounds = bound_group(&progs, &which, &lo, &hi, &objs, chunks[*f], clip);
                    if bounds.len() != idxs.len() {
                        // Every dequeued leaf must come back bounded.
                        return idxs
                            .iter()
                            .map(|&k| (k, Fate::NonFinite, ClaimMark::None))
                            .collect();
                    }
                    idxs.iter()
                        .zip(&bounds)
                        .map(|(&k, db)| {
                            let it = &items[k];
                            let fate =
                                decide(&fam[it.c], &w0s[*f][it.c], &it.lo, &it.hi, db, cfg, clip);
                            (k, fate, ClaimMark::None)
                        })
                        .collect()
                })
                .collect()
        };
        st.waves += 1;
        // Merge in item order: the outcome does not depend on where a fate
        // was computed.
        for (k, fate, mark) in decided.into_iter().flatten() {
            let (f, c, depth0) = (items[k].f, items[k].c, items[k].depth);
            let d = dims[f];
            let lane = &mut lanes[f][c];
            if lane.out.is_some() {
                continue;
            }
            lane.stats.domains += 1;
            st.domains += 1;
            match mark {
                ClaimMark::Certified => {
                    lane.stats.bbe_claimed += 1;
                    lane.stats.bbe_certified += 1;
                }
                ClaimMark::Refuted => {
                    lane.stats.bbe_claimed += 1;
                    lane.stats.bbe_refuted += 1;
                }
                ClaimMark::Unclaimed | ClaimMark::None => {}
            }
            // M8a router progress: the leaf's own volume when it closes, the
            // volume its clip removed when it is split or requeued.
            let w0 = &w0s[f][c];
            let item_vol = box_volume(&items[k].lo, &items[k].hi, w0);
            match &fate {
                Fate::Closed | Fate::ClipClosed => lane.stats.closed_volume += item_vol,
                Fate::Split {
                    lo,
                    hi,
                    shrunk: true,
                    ..
                }
                | Fate::Requeue { lo, hi } => {
                    lane.stats.closed_volume += (item_vol - box_volume(lo, hi, w0)).max(0.0);
                }
                _ => {}
            }
            let close = |lane: &mut Lane| {
                lane.stats.closed += 1;
                lane.open = lane.open.saturating_sub(1);
                if lane.open == 0 {
                    let mut s = lane.stats;
                    s.secs = t0.elapsed().as_secs_f64();
                    lane.out = Some(LowDimOutcome::Verified(s));
                }
            };
            match fate {
                Fate::NonFinite => lane.give_up(LowDimStop::NonFinite, t0),
                Fate::Closed => close(lane),
                Fate::ClipClosed => {
                    lane.stats.clip_closed += 1;
                    st.clip_closed += 1;
                    close(lane);
                }
                Fate::Candidate(_) => lane.give_up(LowDimStop::Candidate, t0),
                Fate::Resolution => lane.give_up(LowDimStop::Resolution, t0),
                Fate::Requeue { lo, hi } => {
                    lane.stats.clip_shrunk += 1;
                    lane.lo.extend_from_slice(&lo);
                    lane.hi.extend_from_slice(&hi);
                    lane.depth.push(depth0);
                }
                Fate::Split {
                    lo,
                    hi,
                    dim,
                    mid,
                    shrunk,
                } => {
                    lane.stats.clip_shrunk += u64::from(shrunk);
                    let depth = depth0 + 1;
                    lane.stats.max_depth = lane.stats.max_depth.max(depth);
                    lane.lo.extend_from_slice(&lo);
                    lane.hi.extend_from_slice(&hi);
                    let li = lane.hi.len() - d + dim;
                    lane.hi[li] = mid;
                    lane.lo.extend_from_slice(&lo);
                    lane.hi.extend_from_slice(&hi);
                    let ri = lane.lo.len() - d + dim;
                    lane.lo[ri] = mid;
                    lane.depth.push(depth);
                    lane.depth.push(depth);
                    lane.open += 1;
                }
            }
        }
    }
    st.secs = t0.elapsed().as_secs_f64();
    let outcomes: Vec<Vec<LowDimOutcome>> = lanes
        .into_iter()
        .map(|ls| {
            ls.into_iter()
                .map(|lane| {
                    lane.out.unwrap_or_else(|| {
                        let mut s = lane.stats;
                        s.secs = st.secs;
                        LowDimOutcome::Unproved(s, LowDimStop::Deadline)
                    })
                })
                .collect()
        })
        .collect();
    st.verified = outcomes
        .iter()
        .flatten()
        .filter(|o| matches!(o, LowDimOutcome::Verified(_)))
        .count();
    st.vol = outcomes
        .iter()
        .flatten()
        .map(|o| match o {
            LowDimOutcome::Verified(s) | LowDimOutcome::Unproved(s, _) => s.closed_volume,
            LowDimOutcome::Declined(_) => 0.0,
        })
        .sum();
    if let Some((run, _)) = &bbe {
        run.emit();
    }
    (outcomes, st)
}

/// The `[lowdim-shared] done` line, printed unconditionally by every fold run
/// with the shared worklist armed (fixed schema). An armed fold run without
/// it is VOID, not negative.
pub(crate) fn emit_line(st: &SharedStats) {
    eprintln!(
        "[lowdim-shared] done families={} clauses={} waves={} domains={} verified={} \
         clip_closed={} queue_peak={} mem_stop={} secs={:.3} vol={:.6}",
        st.families,
        st.clauses,
        st.waves,
        st.domains,
        st.verified,
        st.clip_closed,
        st.queue_peak,
        u8::from(st.mem_stop),
        st.secs,
        st.vol
    );
}

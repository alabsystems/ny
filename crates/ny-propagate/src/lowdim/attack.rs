// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lowdim-bundle (3) — in-lane WORST-LEAF PGD (study_5 `#lowdim-leaf-attack`,
//! lsnc_relu row 58, field `sat`).
//!
//! # Why
//!
//! The lane's violation test evaluates only each leaf's CENTRE. Vibecheck's
//! lsnc config runs a light PGD on the worst open leaves every 40 waves (acas:
//! every 3; 64-128 leaves, 256 restarts x 50 steps), and the numpy prototype
//! found valid float32 counterexamples for rows 14 and 58 by input-split
//! search that a centre test misses.
//!
//! # Soundness
//!
//! ATTACK-ONLY. A hit is a point `x` inside a leaf box (so inside the root
//! box) at which the lane's own f64 lowering violates EVERY conjunct by the
//! lane's candidate margin — exactly the predicate of the centre test
//! (`point_objective(prog, x, o_i) <= t_i - candidate_margin(t_i)`,
//! re-evaluated on `super::point_objective`, not on this module's evaluator).
//! It leaves the lane as a `LowDimStop::Candidate` witness, which claims
//! nothing: the Sequential entry re-evaluates it on the network's own f32
//! forward (`lowdim_candidate_witness`) and the grouped entry on the trusted
//! ONNX-Runtime forward of the original model (`confirm_lowdim_candidate`),
//! and only a point that passes there reaches the unchanged
//! `gate_sat_with_trusted_oracle`. No bound, no closure and no proof depends
//! on anything here; the worst outcome of a defect is a wasted round or a
//! rejected candidate.
//!
//! COST. A round is capped by wall time: the attack's cumulative seconds may
//! not exceed [`ATTACK_SHARE`] of the lane's elapsed wall (a due round with no
//! allowance is skipped and counted), and every restart polls the round's
//! stop time. Memory per task is one value and one adjoint vector per node.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use super::{
    candidate_margin, point_objective, LowDimConfig, LowDimParallel, LowDimStats, Op, Program,
};

/// Leaves attacked per round (Vibecheck's acas 128 / lsnc 64).
pub(crate) const ATTACK_LEAVES: usize = 64;
/// Random restarts per leaf (the first is the leaf centre).
pub(crate) const ATTACK_RESTARTS: usize = 256;
/// Signed-gradient steps per restart.
pub(crate) const ATTACK_STEPS: usize = 50;
/// Largest share of the lane's elapsed wall the attack may have spent.
pub(crate) const ATTACK_SHARE: f64 = 0.2;
/// First step as a fraction of the leaf's width, decaying linearly.
const STEP0: f64 = 0.1;

/// Per-task buffers: every node's point value and adjoint.
struct Tape {
    vals: Vec<Vec<f64>>,
    adj: Vec<Vec<f64>>,
}

impl Tape {
    fn new(prog: &Program) -> Self {
        Self {
            vals: prog.sizes.iter().map(|&n| vec![0.0; n]).collect(),
            adj: prog.sizes.iter().map(|&n| vec![0.0; n]).collect(),
        }
    }

    /// The plain f64 value of every node at `x` (interval constants at their
    /// midpoints) — the same function as `super::point_values`.
    fn forward(&mut self, prog: &Program, x: &[f64]) {
        for (idx, op) in prog.ops.iter().enumerate() {
            let (before, rest) = self.vals.split_at_mut(idx);
            let out = &mut rest[0];
            match op {
                Op::Input => {
                    for (o, &v) in out.iter_mut().zip(x) {
                        *o = v;
                    }
                }
                Op::Linear {
                    src,
                    n_in,
                    w,
                    bias_lo,
                    bias_hi,
                    ..
                } => {
                    let s = &before[*src];
                    for (i, o) in out.iter_mut().enumerate() {
                        let row = &w[i * n_in..(i + 1) * n_in];
                        let mut acc = 0.0;
                        for (wj, sj) in row.iter().zip(s) {
                            acc += wj * sj;
                        }
                        *o = f64::midpoint(bias_lo[i], bias_hi[i]) + acc;
                    }
                }
                Op::Sparse {
                    terms, k_lo, k_hi, ..
                } => {
                    for (j, o) in out.iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for t in &terms[j] {
                            acc += t.coef * before[t.src as usize][t.idx as usize];
                        }
                        *o = f64::midpoint(k_lo[j], k_hi[j]) + acc;
                    }
                }
                Op::Relu { src } => {
                    for (o, &v) in out.iter_mut().zip(&before[*src]) {
                        *o = v.max(0.0);
                    }
                }
                Op::Sigmoid { src } => {
                    for (o, &v) in out.iter_mut().zip(&before[*src]) {
                        *o = super::sigmoid::sig(v);
                    }
                }
                Op::Slice { src, start } => {
                    let n = out.len();
                    out.copy_from_slice(&before[*src][*start..*start + n]);
                }
                Op::Concat { srcs } => {
                    let mut off = 0;
                    for &si in srcs {
                        let part = &before[si];
                        out[off..off + part.len()].copy_from_slice(part);
                        off += part.len();
                    }
                }
                Op::Identity { src } => out.copy_from_slice(&before[*src]),
                Op::Add { a, b } => {
                    for (j, o) in out.iter_mut().enumerate() {
                        *o = before[*a][j] + before[*b][j];
                    }
                }
                Op::Sub { a, b } => {
                    for (j, o) in out.iter_mut().enumerate() {
                        *o = before[*a][j] - before[*b][j];
                    }
                }
                Op::AddConst { src, k } => {
                    for (j, o) in out.iter_mut().enumerate() {
                        *o = before[*src][j] + k[j];
                    }
                }
                Op::RevSubConst { src, k } => {
                    for (j, o) in out.iter_mut().enumerate() {
                        *o = k[j] - before[*src][j];
                    }
                }
                Op::MulConst { src, k } => {
                    for (j, o) in out.iter_mut().enumerate() {
                        *o = before[*src][j] * k[j];
                    }
                }
                Op::MaxGlobal { src } => {
                    out[0] = before[*src]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                }
                Op::Mul { a, b } => {
                    for (j, o) in out.iter_mut().enumerate() {
                        *o = before[*a][j] * before[*b][j];
                    }
                }
            }
        }
    }

    /// Input gradient of `objective . seed(x)` at the point of the last
    /// [`Tape::forward`] (reverse mode; a ReLU at exactly 0 and a max tie take
    /// the subgradient 0 / the first maximum). Heuristic direction only.
    fn backward(&mut self, prog: &Program, objective: &[f64], grad: &mut [f64]) {
        for a in &mut self.adj {
            a.fill(0.0);
        }
        let seed = prog.seed;
        for (g, &o) in self.adj[seed].iter_mut().zip(objective) {
            *g = o;
        }
        let vals = &self.vals;
        for idx in (1..prog.ops.len()).rev() {
            let (before, rest) = self.adj.split_at_mut(idx);
            let g = &rest[0];
            if g.iter().all(|&v| v == 0.0) {
                continue;
            }
            match &prog.ops[idx] {
                Op::Input => {}
                Op::Linear { src, n_in, w, .. } => {
                    let t = &mut before[*src];
                    for (i, &gi) in g.iter().enumerate() {
                        if gi == 0.0 {
                            continue;
                        }
                        let row = &w[i * n_in..(i + 1) * n_in];
                        for (tj, wj) in t.iter_mut().zip(row) {
                            *tj += gi * wj;
                        }
                    }
                }
                Op::Sparse { terms, .. } => {
                    for (j, &gj) in g.iter().enumerate() {
                        for t in &terms[j] {
                            before[t.src as usize][t.idx as usize] += gj * t.coef;
                        }
                    }
                }
                Op::Relu { src } => {
                    for (j, &gj) in g.iter().enumerate() {
                        if vals[*src][j] > 0.0 {
                            before[*src][j] += gj;
                        }
                    }
                }
                Op::Sigmoid { src } => {
                    for (j, &gj) in g.iter().enumerate() {
                        let s = vals[idx][j];
                        before[*src][j] += gj * s * (1.0 - s);
                    }
                }
                Op::Slice { src, start } => {
                    for (j, &gj) in g.iter().enumerate() {
                        before[*src][start + j] += gj;
                    }
                }
                Op::Concat { srcs } => {
                    let mut off = 0;
                    for &si in srcs {
                        let w = prog.sizes[si];
                        for j in 0..w {
                            before[si][j] += g[off + j];
                        }
                        off += w;
                    }
                }
                Op::Identity { src } | Op::AddConst { src, .. } => {
                    for (j, &gj) in g.iter().enumerate() {
                        before[*src][j] += gj;
                    }
                }
                Op::RevSubConst { src, .. } => {
                    for (j, &gj) in g.iter().enumerate() {
                        before[*src][j] -= gj;
                    }
                }
                Op::MulConst { src, k } => {
                    for (j, &gj) in g.iter().enumerate() {
                        before[*src][j] += gj * k[j];
                    }
                }
                Op::Add { a, b } => {
                    for (j, &gj) in g.iter().enumerate() {
                        before[*a][j] += gj;
                        before[*b][j] += gj;
                    }
                }
                Op::Sub { a, b } => {
                    for (j, &gj) in g.iter().enumerate() {
                        before[*a][j] += gj;
                        before[*b][j] -= gj;
                    }
                }
                Op::MaxGlobal { src } => {
                    let v = &vals[*src];
                    let mut arg = 0;
                    for (j, &x) in v.iter().enumerate() {
                        if x > v[arg] {
                            arg = j;
                        }
                    }
                    if !v.is_empty() {
                        before[*src][arg] += g[0];
                    }
                }
                Op::Mul { a, b } => {
                    for (j, &gj) in g.iter().enumerate() {
                        let (va, vb) = (vals[*a][j], vals[*b][j]);
                        before[*a][j] += gj * vb;
                        before[*b][j] += gj * va;
                    }
                }
            }
        }
        for (o, &v) in grad.iter_mut().zip(&self.adj[0]) {
            *o = v;
        }
    }
}

/// Deterministic uniform `[0, 1)` stream (SplitMix64).
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The lane's own candidate predicate at `x` (see the module docs).
fn is_candidate(prog: &Program, objs: &[Vec<f64>], thrs: &[f64], x: &[f64]) -> bool {
    objs.iter()
        .zip(thrs)
        .all(|(o, &t)| point_objective(prog, x, o) <= t - candidate_margin(t))
}

/// PGD on one leaf box: minimise `max_i (o_i . y(x) - t_i + margin_i)` by
/// signed-gradient steps projected onto `[lo, hi]`. Returns a candidate (see
/// [`is_candidate`]) and the evaluations spent.
#[allow(clippy::too_many_arguments)]
fn attack_leaf(
    prog: &Program,
    objs: &[Vec<f64>],
    thrs: &[f64],
    lo: &[f64],
    hi: &[f64],
    seed: u64,
    stop: Instant,
    found: &AtomicBool,
) -> (Option<Vec<f64>>, u64) {
    let d = lo.len();
    if d == 0 || hi.len() != d || objs.is_empty() || objs.len() != thrs.len() {
        return (None, 0);
    }
    let margins: Vec<f64> = thrs.iter().map(|&t| candidate_margin(t)).collect();
    let mut tape = Tape::new(prog);
    let mut rng = Rng(seed);
    let mut x = vec![0.0f64; d];
    let mut grad = vec![0.0f64; d];
    let mut evals = 0u64;
    for r in 0..ATTACK_RESTARTS {
        if found.load(Ordering::Relaxed) || Instant::now() >= stop {
            break;
        }
        for k in 0..d {
            x[k] = if r == 0 {
                f64::midpoint(lo[k], hi[k])
            } else {
                lo[k] + (hi[k] - lo[k]) * rng.unit()
            };
            x[k] = x[k].max(lo[k]).min(hi[k]);
        }
        for step in 0..ATTACK_STEPS {
            tape.forward(prog, &x);
            evals += 1;
            let y = &tape.vals[prog.seed];
            let (mut worst, mut wi) = (f64::NEG_INFINITY, 0usize);
            for (i, (o, t)) in objs.iter().zip(thrs).enumerate() {
                let dot: f64 = y.iter().zip(o).map(|(a, b)| a * b).sum();
                let loss = dot - t + margins[i];
                if loss > worst {
                    worst = loss;
                    wi = i;
                }
            }
            if worst <= 0.0 && is_candidate(prog, objs, thrs, &x) {
                found.store(true, Ordering::Relaxed);
                return (Some(x), evals);
            }
            if step + 1 == ATTACK_STEPS {
                break;
            }
            tape.backward(prog, &objs[wi], &mut grad);
            let frac = STEP0 * (1.0 - step as f64 / ATTACK_STEPS as f64);
            for k in 0..d {
                let delta = frac * (hi[k] - lo[k]);
                if grad[k] > 0.0 {
                    x[k] -= delta;
                } else if grad[k] < 0.0 {
                    x[k] += delta;
                }
                x[k] = x[k].max(lo[k]).min(hi[k]);
            }
        }
    }
    (None, evals)
}

/// One leaf-attack round over this wave's open leaves `(margin, lo, hi)`: the
/// [`ATTACK_LEAVES`] with the most negative margin (furthest from closing),
/// within the wall-time share. Updates the attack counters; returns the
/// first candidate in leaf order.
#[allow(clippy::too_many_arguments)]
pub(super) fn round(
    prog: &Program,
    objs: &[Vec<f64>],
    thrs: &[f64],
    cfg: &LowDimConfig,
    mut leaves: Vec<(f64, Vec<f64>, Vec<f64>)>,
    stats: &mut LowDimStats,
    t0: Instant,
    deadline: Instant,
) -> Option<Vec<f64>> {
    let now = Instant::now();
    let allowance = ATTACK_SHARE * now.duration_since(t0).as_secs_f64() - stats.attack_secs;
    // RSS admission: every concurrent task holds one tape (a value and an
    // adjoint per node); refuse the round when that would exceed a quarter of
    // the live memory headroom (ny has no governor without a cgroup).
    let tape_bytes = prog
        .total
        .saturating_mul(16)
        .saturating_mul(rayon::current_num_threads().max(1));
    let admitted = crate::network::crown_memory::process_memory_headroom_bytes()
        .map_or(true, |h| {
            u64::try_from(tape_bytes).map_or(false, |t| t <= h / 4)
        });
    if allowance.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        || now >= deadline
        || leaves.is_empty()
        || !admitted
    {
        stats.attack_skipped += 1;
        return None;
    }
    let stop = (now + Duration::from_secs_f64(allowance.min(3600.0))).min(deadline);
    leaves.sort_by(|a, b| a.0.total_cmp(&b.0));
    leaves.truncate(ATTACK_LEAVES);
    let base = 0x6C73_6E63_u64 ^ stats.attack_rounds.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let found = AtomicBool::new(false);
    let run = |(i, (_, lo, hi)): (usize, &(f64, Vec<f64>, Vec<f64>))| {
        attack_leaf(
            prog,
            objs,
            thrs,
            lo,
            hi,
            base ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
            stop,
            &found,
        )
    };
    let results: Vec<(Option<Vec<f64>>, u64)> = if cfg.parallel == LowDimParallel::Serial {
        leaves.iter().enumerate().map(run).collect()
    } else {
        leaves.par_iter().enumerate().map(run).collect()
    };
    let evals: u64 = results.iter().map(|(_, e)| *e).sum();
    let hit = results.into_iter().find_map(|(x, _)| x);
    stats.attack_rounds += 1;
    stats.attack_leaves += leaves.len() as u64;
    stats.attack_evals += evals;
    stats.attack_hits += u64::from(hit.is_some());
    stats.attack_secs += now.elapsed().as_secs_f64();
    hit
}

/// The `[lowdim-leaf-attack] done` line, printed unconditionally at the end of
/// every lane run with the attack armed (fixed schema). An armed run with
/// `rounds=0 skipped=0` never reached a due wave; `evals=0` with `rounds>0`
/// is VOID, not negative.
pub(super) fn emit_line(stats: &LowDimStats, every: usize) {
    eprintln!(
        "[lowdim-leaf-attack] done every={every} rounds={} skipped={} leaves={} restarts={} \
         steps={} evals={} hits={} secs={:.3} lane_secs={:.3}",
        stats.attack_rounds,
        stats.attack_skipped,
        stats.attack_leaves,
        ATTACK_RESTARTS,
        ATTACK_STEPS,
        stats.attack_evals,
        stats.attack_hits,
        stats.attack_secs,
        stats.secs
    );
}

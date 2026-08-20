// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Straight-through-estimator PGD over the SAME admitted BNN fragment the
//! sign-space falsifier uses.
//!
//! # Why this exists next to the LP search rather than instead of it
//!
//! `docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md` §5 used to claim that
//! "PGD fails here because gradients are meaningless through `Sign`". That is
//! false for the STRAIGHT-THROUGH ESTIMATOR, which is the standard technique
//! for binarized networks: the forward pass runs the REAL `Sign` (so every
//! activation, accumulator and logit is the exact integer the network
//! computes), and only the BACKWARD pass substitutes a surrogate derivative.
//! The measured consequence is in §5 of that document.
//!
//! The LP search accepts 7-16 first-layer flips per lane and the reachable
//! witnesses sit 483-1483 free-bit flips from the box centre, which is why
//! the two lanes are complementary rather than redundant: this one moves in
//! PIXEL space with a step of several grey levels and reaches distances the
//! flip-at-a-time search cannot.
//!
//! # What is reused, and what is new
//!
//! Everything except the gradient and the step schedule is reused verbatim
//! from [`crate::bnn_sign_space`]:
//!
//! * [`super::admit`] — the STRUCTURAL admission of the fragment. There is no
//!   filename, category or shape heuristic anywhere in this module; a net
//!   outside the fragment is refused by the same typed
//!   [`SignSpaceRefusal`] the LP lane refuses it with, at the same cost.
//! * [`Admitted::z1_at`], [`Admitted::pooled_at`], [`Admitted::s1_from_z1`],
//!   [`Admitted::forward_from_pattern`] — the EXACT integer forward.
//! * [`super::finalize`] — the witness finalizer, which re-derives everything
//!   from scratch, enforces EXACT (zero-tolerance) box membership, enforces
//!   `f32`-replay stability of every sign decision, and refuses a
//!   non-positive margin.
//!
//! # It cannot conclude anything
//!
//! The entry point returns [`SignSpaceOutcome`], which has no
//! `Verified`/`Unsat` variant BY CONSTRUCTION (pinned by
//! [`crate::bnn_sign_space::bnn_sign_space_tests`] and again by this module's
//! own tests). An attack can only ever exhibit a witness, and the witness is a
//! CLAIM that the caller must route through the unchanged trusted-oracle gate.

use std::time::{Duration, Instant};

use super::{
    admit, finalize, Admitted, SignSpaceLimits, SignSpaceOutcome, SignSpaceRefusal,
    SignSpaceRequest, Stage, StageKind,
};

/// Schedule knobs for one STE-PGD consultation.
///
/// The defaults are the ones the numpy reference that produced the seven
/// verified traffic-sign witnesses ran with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StePgdLimits {
    /// Wall-clock budget for the whole call (both stages).
    pub max_wall_time: Duration,
    /// Fraction of the budget the ITERATED-LOCAL-SEARCH stage gets. Stage A
    /// (momentum STE-PGD with restarts) gets the rest.
    pub climb_fraction: f64,
    /// Gradient steps per restart.
    pub iters: usize,
    /// Hard cap on restarts, so a very long budget cannot spin unbounded.
    pub max_restarts: usize,
    /// Straight-through window, as a fraction of the layer's pre-activation
    /// RMS: the surrogate derivative is `1` where `|u| <= frac * rms(u)` and
    /// `0` elsewhere.
    pub ste_fraction: f64,
    /// Momentum coefficient on the normalized gradient.
    pub momentum: f64,
    /// How many improving points are kept as ILS restart anchors.
    pub anchor_cap: usize,
    /// How many of the best-ranked challengers the restart schedule cycles
    /// through, one target per restart.
    pub challenger_cycle: usize,
    /// Deterministic seed. The lane is reproducible: same inputs, same
    /// trajectory, same witness.
    pub seed: u64,
}

impl Default for StePgdLimits {
    fn default() -> Self {
        Self {
            max_wall_time: Duration::from_mins(4),
            climb_fraction: 0.25,
            iters: 120,
            max_restarts: 4096,
            ste_fraction: 0.25,
            momentum: 0.9,
            anchor_cap: 12,
            challenger_cycle: 6,
            seed: 0x5445_5F50_4744_0001,
        }
    }
}

/// Deterministic SplitMix64. Local so the lane's trajectory is reproducible
/// and depends on nothing outside this file.
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        // 53 significant bits, the exact f64 mantissa width.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[-1, 1)`.
    fn symmetric(&mut self) -> f64 {
        self.unit().mul_add(2.0, -1.0)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// conv1's patch geometry, hoisted out of the per-iteration loops.
///
/// `Admitted::conv1_patch` rebuilds the same `taps1` flat pixel indices once
/// per (position, channel) pair, i.e. `c1` times more often than the geometry
/// changes, and interleaves them with the weights. The search runs that same
/// convolution forward AND transposed on EVERY iterate, so the tables are
/// built once here instead: `indices[position]` are the pixel indices of that
/// patch and `weights[channel]` are the `+/-1` taps, in the SAME order — which
/// is what keeps the accumulation bit-identical to
/// [`Admitted::z1_at`], the forward the witness finalizer re-derives with.
struct Conv1Plan {
    /// `h1c * w1c` rows of `taps1` flat pixel indices.
    indices: Vec<u32>,
    /// `c1` rows of `taps1` weights.
    weights: Vec<f64>,
    taps: usize,
}

impl Conv1Plan {
    fn build(admitted: &Admitted<'_>) -> Self {
        let taps = admitted.taps1;
        let mut indices = Vec::with_capacity(admitted.h1c * admitted.w1c * taps);
        let mut patch = Vec::with_capacity(taps);
        for r in 0..admitted.h1c {
            for c in 0..admitted.w1c {
                admitted.conv1_patch(r, c, 0, &mut patch);
                indices.extend(patch.iter().map(|&(p, _)| p as u32));
            }
        }
        let mut weights = Vec::with_capacity(admitted.c1 * taps);
        for ch in 0..admitted.c1 {
            admitted.conv1_patch(0, 0, ch, &mut patch);
            weights.extend(patch.iter().map(|&(_, w)| w));
        }
        Self {
            indices,
            weights,
            taps,
        }
    }

    fn patch(&self, position: usize) -> &[u32] {
        &self.indices[position * self.taps..(position + 1) * self.taps]
    }

    fn channel(&self, channel: usize) -> &[f64] {
        &self.weights[channel * self.taps..(channel + 1) * self.taps]
    }
}

/// The frozen forward state one gradient is taken at.
struct Trace {
    /// Pooled conv1 value per first-layer unit.
    pooled1: Vec<f64>,
    /// RAW index attaining each pooled maximum (the pool's argmax route).
    member1: Vec<usize>,
    /// Pooled first-layer signs.
    s1: Vec<i8>,
    /// Post-pool accumulators of every binary stage.
    p: Vec<Vec<i32>>,
    /// Pre-pool accumulators of every binary stage.
    z: Vec<Vec<i32>>,
    /// Exact pre-softmax integer logits.
    logits: Vec<i64>,
}

impl Admitted<'_> {
    /// conv1 accumulators through the hoisted plan.
    ///
    /// Bit-identical to [`Admitted::z1_at`]: same taps, same order, same
    /// arithmetic — only the index/weight bookkeeping is precomputed.
    fn ste_z1(&self, plan: &Conv1Plan, x: &[f64]) -> Vec<f64> {
        let mut z1 = vec![0.0; self.n_raw1];
        let mut values = vec![0.0f64; plan.taps];
        for position in 0..self.h1c * self.w1c {
            for (value, &p) in values.iter_mut().zip(plan.patch(position)) {
                *value = x[p as usize];
            }
            for ch in 0..self.c1 {
                let weights = plan.channel(ch);
                let mut acc = 0.0;
                for (w, v) in weights.iter().zip(&values) {
                    acc += w * v;
                }
                z1[position * self.c1 + ch] = acc;
            }
        }
        z1
    }

    /// The exact forward, keeping the intermediates the STE backward needs.
    fn ste_trace(&self, plan: &Conv1Plan, x: &[f64]) -> Trace {
        let z1 = self.ste_z1(plan, x);
        let mut pooled1 = vec![0.0; self.n_units1];
        let mut member1 = vec![0usize; self.n_units1];
        let mut s1 = vec![0i8; self.n_units1];
        for k in 0..self.n_units1 {
            let (pooled, member) = self.pooled_at(k, &z1);
            pooled1[k] = pooled;
            member1[k] = member;
            s1[k] = if pooled >= self.t1[k % self.c1] {
                1
            } else {
                -1
            };
        }
        let engine = self.forward_from_pattern(&s1);
        Trace {
            pooled1,
            member1,
            s1,
            p: engine.p,
            z: engine.z,
            logits: engine.logits,
        }
    }

    /// The STRAIGHT-THROUGH gradient of `logit[challenger] - logit[target]`
    /// with respect to the input pixels, at the point `trace` was taken at.
    ///
    /// The forward values are the network's REAL `Sign` outputs; only the
    /// derivative is substituted. Each `Sign` contributes the rectangular
    /// surrogate slope `d/du sign(u) := [ |u| <= frac * rms(u) ]`, each
    /// `MaxPool` routes the incoming cotangent to the member attaining the
    /// maximum, and every linear map is transposed exactly.
    ///
    /// This is exactly the gradient of the LINEARIZED network obtained by
    /// freezing the pool routes and the surrogate masks at this point, which
    /// is what `ste_gradient_matches_finite_differences` finite-differences.
    fn ste_gradient(
        &self,
        plan: &Conv1Plan,
        trace: &Trace,
        challenger: usize,
        fraction: f64,
    ) -> Vec<f64> {
        // Seed: d(logit[c] - logit[t]) / d(final sign vector).
        let mut g: Vec<f64> = (0..self.n_flat)
            .map(|j| {
                (self.dense_weight(j, challenger) - self.dense_weight(j, self.target_class)) as f64
            })
            .collect();

        for si in (0..self.stages.len()).rev() {
            let stage = &self.stages[si];
            apply_ste_mask(&mut g, &trace.p[si], &stage.t, stage.channels, fraction);
            // MaxPool backward: route to the argmax member.
            let mut gz = vec![0.0f64; stage.z_len];
            for (oi, &value) in g.iter().enumerate() {
                if value != 0.0 {
                    gz[stage.pooled_argmax(oi, &trace.z[si])] += value;
                }
            }
            let in_len = if si == 0 {
                self.n_units1
            } else {
                self.stages[si - 1].out_len
            };
            g = stage.backward_to_input(&gz, in_len);
        }

        // The first layer: its own surrogate mask, its own pool route, then
        // the real-valued convolution transposed onto the pixels.
        let u1: Vec<f64> = (0..self.n_units1)
            .map(|k| trace.pooled1[k] - self.t1[k % self.c1])
            .collect();
        let window = fraction * rms(&u1);
        let mut gz1 = vec![0.0f64; self.n_raw1];
        for k in 0..self.n_units1 {
            if g[k] != 0.0 && u1[k].abs() <= window {
                gz1[trace.member1[k]] += g[k];
            }
        }
        let mut gx = vec![0.0f64; self.n_pixels];
        for position in 0..self.h1c * self.w1c {
            let cotangents = &gz1[position * self.c1..(position + 1) * self.c1];
            if cotangents.iter().all(|c| *c == 0.0) {
                continue;
            }
            let patch = plan.patch(position);
            for (ch, &cot) in cotangents.iter().enumerate() {
                if cot == 0.0 {
                    continue;
                }
                let weights = plan.channel(ch);
                for (&p, w) in patch.iter().zip(weights) {
                    gx[p as usize] += w * cot;
                }
            }
        }
        gx
    }
}

/// Zero every cotangent whose `Sign` sits outside the straight-through window.
fn apply_ste_mask(g: &mut [f64], p: &[i32], t: &[f64], channels: usize, fraction: f64) {
    let u: Vec<f64> = (0..p.len())
        .map(|oi| f64::from(p[oi]) - t[oi % channels])
        .collect();
    let window = fraction * rms(&u);
    for (value, u) in g.iter_mut().zip(&u) {
        if u.abs() > window {
            *value = 0.0;
        }
    }
}

/// Root mean square, `0.0` on an empty slice.
fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

impl Stage {
    /// The pre-pool index attaining this output's pooled maximum. Ties go to
    /// the lowest index, exactly as [`Admitted::pooled_at`] resolves them, so
    /// the route is deterministic.
    fn pooled_argmax(&self, oi: usize, z: &[i32]) -> usize {
        match &self.kind {
            StageKind::Conv {
                out_c,
                conv_w,
                pool,
                out_w,
                ..
            } => {
                let co = oi % out_c;
                let spatial = oi / out_c;
                let or = spatial / out_w;
                let oc = spatial % out_w;
                let mut best = i32::MIN;
                let mut best_index = 0usize;
                for a in 0..pool.kernel_h {
                    for b in 0..pool.kernel_w {
                        let i = or * pool.stride_h + a;
                        let j = oc * pool.stride_w + b;
                        let index = (i * conv_w + j) * out_c + co;
                        if z[index] > best {
                            best = z[index];
                            best_index = index;
                        }
                    }
                }
                best_index
            }
            StageKind::Dense { .. } => oi,
        }
    }

    /// Transpose of this stage's linear map: cotangents on the pre-pool
    /// accumulators pushed back onto the stage's `+/-1` inputs.
    fn backward_to_input(&self, gz: &[f64], in_len: usize) -> Vec<f64> {
        let mut g = vec![0.0f64; in_len];
        match &self.kind {
            StageKind::Conv {
                w,
                out_c,
                in_c,
                kh,
                kw,
                in_w,
                conv_h,
                conv_w,
                ..
            } => {
                for i in 0..*conv_h {
                    for j in 0..*conv_w {
                        for co in 0..*out_c {
                            let cot = gz[(i * conv_w + j) * out_c + co];
                            if cot == 0.0 {
                                continue;
                            }
                            for kr in 0..*kh {
                                for kc in 0..*kw {
                                    for ci in 0..*in_c {
                                        let weight = w[((co * in_c + ci) * kh + kr) * kw + kc];
                                        let idx = ((i + kr) * in_w + (j + kc)) * in_c + ci;
                                        g[idx] += f64::from(weight) * cot;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            StageKind::Dense { w, in_dim, out_dim } => {
                for i in 0..*in_dim {
                    let row = &w[i * out_dim..(i + 1) * out_dim];
                    let mut acc = 0.0f64;
                    for (o, &weight) in row.iter().enumerate() {
                        acc += f64::from(weight) * gz[o];
                    }
                    g[i] = acc;
                }
            }
        }
        g
    }
}

/// One integer-rounded, box-clamped PGD step.
///
/// EVERY coordinate of the result is inside `[lo, hi]` with ZERO tolerance, and
/// is an exact integer unless the clamp pinned it to a non-integral bound. The
/// rounding is why the witnesses this schedule finds are integer pixel vectors
/// and survive the `f32` replay with room to spare.
fn schedule_step(x: &mut [f64], direction: &[f64], alpha: f64, lo: &[f64], hi: &[f64]) {
    for (p, value) in x.iter_mut().enumerate() {
        let stepped = step_sign(direction[p]).mul_add(alpha, *value).round();
        *value = stepped.clamp(lo[p], hi[p]);
    }
}

/// `signum` with `0.0` at zero (`f64::signum` returns `1.0` for `+0.0`).
fn step_sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Clamp `x` into the box with zero tolerance, after integer rounding.
fn round_into_box(x: &mut [f64], lo: &[f64], hi: &[f64]) {
    for (p, value) in x.iter_mut().enumerate() {
        *value = value.round().clamp(lo[p], hi[p]);
    }
}

/// STE-PGD falsification of a binarized conv suffix.
///
/// Returns [`SignSpaceOutcome::Candidate`] with an in-box, `f32`-replay-stable
/// counterexample, [`SignSpaceOutcome::Exhausted`] when the budget runs out, or
/// [`SignSpaceOutcome::Refused`] when the request is outside the admitted
/// fragment.
///
/// **There is no verified/unsat outcome by construction.** A candidate is a
/// CLAIM: the caller MUST replay [`super::SignSpaceCandidate::input`] through
/// the ORIGINAL network and property before publishing anything.
pub fn falsify_bnn_ste_pgd_unwired(
    request: &SignSpaceRequest<'_>,
    limits: &SignSpaceLimits,
    schedule: &StePgdLimits,
) -> SignSpaceOutcome {
    let started = Instant::now();
    let admitted = match admit(request, limits) {
        Ok(admitted) => admitted,
        Err(refusal) => return SignSpaceOutcome::Refused(refusal),
    };
    match search(&admitted, schedule, started) {
        Ok(outcome) => outcome,
        Err(refusal) => SignSpaceOutcome::Refused(refusal),
    }
}

/// The two-stage search proper.
fn search(
    admitted: &Admitted<'_>,
    schedule: &StePgdLimits,
    started: Instant,
) -> Result<SignSpaceOutcome, SignSpaceRefusal> {
    let n = admitted.n_pixels;
    if admitted.lo.len() != n || admitted.hi.len() != n {
        return Err(SignSpaceRefusal::ShapeMismatch {
            detail: format!(
                "box has {}/{} entries, geometry needs {n}",
                admitted.lo.len(),
                admitted.hi.len()
            ),
        });
    }
    let free_units = admitted.classify().free.len();
    let deadline = started + schedule.max_wall_time;
    let climb_at = started
        + schedule
            .max_wall_time
            .mul_f64((1.0 - schedule.climb_fraction).clamp(0.0, 1.0));

    // Half-widths, and the pixel-space step the schedule opens at.
    let half: Vec<f64> = (0..n)
        .map(|p| 0.5 * (admitted.hi[p] - admitted.lo[p]))
        .collect();
    let half_max = half.iter().copied().fold(0.0f64, f64::max);
    let centre: Vec<f64> = (0..n)
        .map(|p| admitted.lo[p].midpoint(admitted.hi[p]))
        .collect();

    let plan = Conv1Plan::build(admitted);
    let base = admitted.ste_trace(&plan, &centre);
    let base_s1 = base.s1.clone();
    let (mut best_margin, _) = admitted.margin(&base.logits);
    // #lane-value-stall. The lane's own VALUE unit: how far the best margin
    // moved from the box centre over the whole two-stage search. Reported on
    // `Exhausted` next to the work count so the scheduler never has to guess.
    let initial_margin = best_margin;
    let mut best_x = centre.clone();

    // One target challenger per restart, best-ranked first — the reference's
    // `order[r % 6]`.
    let mut order: Vec<usize> = admitted.challengers.clone();
    order.sort_by_key(|&c| -(base.logits[c] - base.logits[admitted.target_class]));
    let cycle = schedule.challenger_cycle.clamp(1, order.len().max(1));

    let mut rng = Rng::new(schedule.seed);
    let mut anchors: Vec<Vec<f64>> = vec![centre.clone()];
    let mut restart = 0usize;

    // ---- stage A: momentum STE-PGD with ILS restarts ----------------------
    while restart < schedule.max_restarts && Instant::now() < climb_at {
        let challenger = order[restart % cycle];
        let mut x = if restart == 0 {
            centre.clone()
        } else if restart.is_multiple_of(3) {
            let mut x: Vec<f64> = (0..n)
                .map(|p| centre[p] + rng.symmetric() * half[p])
                .collect();
            round_into_box(&mut x, admitted.lo, admitted.hi);
            x
        } else {
            let anchor = &anchors[rng.below(anchors.len())];
            let mut x: Vec<f64> = (0..n)
                .map(|p| anchor[p] + rng.symmetric() * half[p] * 0.4)
                .collect();
            round_into_box(&mut x, admitted.lo, admitted.hi);
            x
        };

        let mut momentum = vec![0.0f64; n];
        for iteration in 0..schedule.iters {
            if Instant::now() >= climb_at {
                break;
            }
            let trace = admitted.ste_trace(&plan, &x);
            let (margin, _) = admitted.margin(&trace.logits);
            if margin > best_margin {
                best_margin = margin;
                best_x.copy_from_slice(&x);
                anchors.push(x.clone());
                if anchors.len() > schedule.anchor_cap.max(1) {
                    anchors.remove(0);
                }
            }
            if margin > 0 {
                if let Some(candidate) = finalize(
                    admitted,
                    &x,
                    0.0,
                    free_units,
                    hamming(&base_s1, &trace.s1),
                    0,
                    started.elapsed(),
                ) {
                    return Ok(SignSpaceOutcome::Candidate(Box::new(candidate)));
                }
            }

            let raw = admitted.ste_gradient(&plan, &trace, challenger, schedule.ste_fraction);
            let scale = raw.iter().map(|v| v.abs()).sum::<f64>() / raw.len().max(1) as f64;
            let mut moving = false;
            for (m, g) in momentum.iter_mut().zip(&raw) {
                *m = schedule.momentum * *m + g / (scale + 1e-12);
                moving |= *m != 0.0;
            }
            if !moving {
                // A dead gradient cannot be recovered by more steps of the
                // same restart; spend the budget on the next one instead.
                break;
            }
            let progress = iteration as f64 / schedule.iters.max(1) as f64;
            let alpha = (half_max * (1.0 - progress)).round().max(1.0);
            schedule_step(&mut x, &momentum, alpha, admitted.lo, admitted.hi);
        }
        restart += 1;
    }

    // ---- stage B: iterated local search around the incumbent ---------------
    let mut probes = 0usize;
    if schedule.climb_fraction > 0.0 {
        let mut current = best_x.clone();
        let (mut current_margin, _) = {
            let trace = admitted.ste_trace(&plan, &current);
            admitted.margin(&trace.logits)
        };
        while Instant::now() < deadline {
            let block = 1 + rng.below(39);
            let mut touched: Vec<(usize, f64)> = Vec::with_capacity(block);
            for _ in 0..block {
                let p = rng.below(n);
                let step = f64::from(rng.below(3) as i32 - 1) * half[p].round().max(1.0);
                touched.push((p, current[p]));
                current[p] = (current[p] + step)
                    .round()
                    .clamp(admitted.lo[p], admitted.hi[p]);
            }
            let trace = admitted.ste_trace(&plan, &current);
            let (margin, _) = admitted.margin(&trace.logits);
            probes += 1;
            if margin >= current_margin {
                current_margin = margin;
                if margin > best_margin {
                    best_margin = margin;
                    best_x.copy_from_slice(&current);
                }
                if margin > 0 {
                    if let Some(candidate) = finalize(
                        admitted,
                        &current,
                        0.0,
                        free_units,
                        hamming(&base_s1, &trace.s1),
                        probes,
                        started.elapsed(),
                    ) {
                        return Ok(SignSpaceOutcome::Candidate(Box::new(candidate)));
                    }
                }
            } else {
                // REVERSE order, because a block may draw the same pixel
                // twice: the second entry saved the value the first step had
                // already written, so replaying forwards would restore that
                // intermediate instead of the original and let a REJECTED
                // probe leak into the incumbent.
                for (p, old) in touched.into_iter().rev() {
                    current[p] = old;
                }
            }
        }
    }

    let flips = {
        let trace = admitted.ste_trace(&plan, &best_x);
        hamming(&base_s1, &trace.s1)
    };
    Ok(SignSpaceOutcome::Exhausted {
        best_logit_margin: best_margin,
        margin_gain: best_margin.saturating_sub(initial_margin).max(0),
        free_units,
        flips,
        lp_solves: probes,
        elapsed: started.elapsed(),
    })
}

/// First-layer sign flips between two patterns — the distance the §10 wall is
/// stated in.
fn hamming(a: &[i8], b: &[i8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[cfg(test)]
#[path = "bnn_ste_pgd_tests.rs"]
mod bnn_ste_pgd_tests;

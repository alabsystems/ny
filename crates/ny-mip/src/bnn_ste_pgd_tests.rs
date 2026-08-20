// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Oracle tests for [`crate::bnn_sign_space::bnn_ste_pgd`].
//!
//! What each one protects, in the order the task states them:
//!
//! (a) **The straight-through gradient is the real derivative of the
//!     linearized network.** A `Sign` has derivative `0` almost everywhere, so
//!     "matches finite differences" can only mean one thing that is not
//!     vacuous: the STE backward must equal the gradient of the function
//!     obtained by freezing the pool routes and the surrogate masks at the
//!     evaluation point and letting the pre-activations vary. That function is
//!     built INDEPENDENTLY here, forward-mode, and central-differenced.
//! (b) **Every returned point is inside the box with ZERO tolerance**, and
//!     integral wherever the schedule's rounding is not overridden by a clamp
//!     onto a non-integral bound.
//! (c) **The lane cannot emit a verdict**, pinned by an exhaustive match with
//!     no wildcard arm over the outcome type.
//! (d) is a front-end obligation (the lever is read in `ny-cli`) and is pinned
//!     in `crates/ny-cli/src/commands/beta_crown/sign_space_falsify.rs`.

use std::time::Duration;

use super::super::{
    admit, ConvSpec, InputGeometry, PoolSpec, SignSpaceActivation, SignSpaceLimits,
    SignSpaceOutcome, SignSpaceRequest,
};
use super::{falsify_bnn_ste_pgd_unwired, round_into_box, schedule_step, Conv1Plan, StePgdLimits};

// ---------------------------------------------------------------------------
// A tiny net inside the admitted fragment, owned by this file.
// ---------------------------------------------------------------------------

/// `Conv -> [MaxPool] -> Sign -> Conv -> Sign -> Dense`, all weights `+/-1`,
/// no affine — so every folded threshold is exactly `0` and the test's own
/// forward needs no threshold plumbing.
struct TinyNet {
    input: InputGeometry,
    conv1: Vec<f32>,
    conv1_out: usize,
    conv1_k: usize,
    pool1: Option<PoolSpec>,
    conv2: Vec<f32>,
    conv2_out: usize,
    conv2_k: usize,
    dense: Vec<f32>,
    num_classes: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    target: usize,
    challengers: Vec<usize>,
}

/// Deterministic `+/-1` filler; no RNG crate, no hidden state.
fn pm1(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if (*seed >> 33) & 1 == 0 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

impl TinyNet {
    /// `6x6x1 -> conv1 2x2x4 -> [pool] -> conv2 2x2x4 -> dense 3`.
    fn new(target: usize, pool1: Option<PoolSpec>) -> Self {
        let input = InputGeometry {
            height: 6,
            width: 6,
            channels: 1,
        };
        let mut seed = 0x1234_5678_9abc_def0_u64;
        let conv1 = pm1(&mut seed, 4 * input.channels * 2 * 2);
        let conv2 = pm1(&mut seed, 4 * 4 * 2 * 2);
        let net = Self {
            input,
            conv1,
            conv1_out: 4,
            conv1_k: 2,
            pool1,
            conv2,
            conv2_out: 4,
            conv2_k: 2,
            dense: Vec::new(),
            num_classes: 3,
            lo: vec![-4.0; 36],
            hi: vec![4.0; 36],
            target,
            challengers: (0..3).filter(|&c| c != target).collect(),
        };
        let dense = pm1(&mut seed, net.n_flat() * 3);
        Self { dense, ..net }
    }

    fn raw1_h(&self) -> usize {
        self.input.height - self.conv1_k + 1
    }
    fn raw1_w(&self) -> usize {
        self.input.width - self.conv1_k + 1
    }
    fn pool(&self) -> PoolSpec {
        self.pool1.unwrap_or(PoolSpec {
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
        })
    }
    fn h1(&self) -> usize {
        (self.raw1_h() - self.pool().kernel_h) / self.pool().stride_h + 1
    }
    fn w1(&self) -> usize {
        (self.raw1_w() - self.pool().kernel_w) / self.pool().stride_w + 1
    }
    fn h2(&self) -> usize {
        self.h1() - self.conv2_k + 1
    }
    fn w2(&self) -> usize {
        self.w1() - self.conv2_k + 1
    }
    fn n_units1(&self) -> usize {
        self.h1() * self.w1() * self.conv1_out
    }
    fn n_flat(&self) -> usize {
        self.h2() * self.w2() * self.conv2_out
    }

    fn request(&self) -> SignSpaceRequest<'_> {
        SignSpaceRequest {
            input: self.input,
            conv1: ConvSpec::valid_unit_stride(
                &self.conv1,
                self.conv1_out,
                self.input.channels,
                self.conv1_k,
                self.conv1_k,
            ),
            conv1_pool: self.pool1,
            conv1_affine: None,
            activation1: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            conv2: ConvSpec::valid_unit_stride(
                &self.conv2,
                self.conv2_out,
                self.conv1_out,
                self.conv2_k,
                self.conv2_k,
            ),
            conv2_pool: None,
            conv2_affine: None,
            activation2: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            stages: &[],
            dense: &self.dense,
            num_classes: self.num_classes,
            lo: &self.lo,
            hi: &self.hi,
            target_class: self.target,
            challengers: &self.challengers,
            reference_input: None,
            reference_forward: None,
        }
    }

    /// RAW conv1 accumulators, computed independently of the module.
    fn raw_z1(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.raw1_h() * self.raw1_w() * self.conv1_out];
        for r in 0..self.raw1_h() {
            for c in 0..self.raw1_w() {
                for ch in 0..self.conv1_out {
                    let mut acc = 0.0;
                    for kr in 0..self.conv1_k {
                        for kc in 0..self.conv1_k {
                            for ic in 0..self.input.channels {
                                let w = f64::from(
                                    self.conv1[((ch * self.input.channels + ic) * self.conv1_k
                                        + kr)
                                        * self.conv1_k
                                        + kc],
                                );
                                let p = ((r + kr) * self.input.width + (c + kc))
                                    * self.input.channels
                                    + ic;
                                acc += w * x[p];
                            }
                        }
                    }
                    out[(r * self.raw1_w() + c) * self.conv1_out + ch] = acc;
                }
            }
        }
        out
    }

    /// Pooled first-layer values and the RAW index attaining each, computed
    /// independently (lowest index wins a tie, the module's convention).
    fn pooled1(&self, x: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let raw = self.raw_z1(x);
        let pool = self.pool();
        let mut values = vec![0.0; self.n_units1()];
        let mut members = vec![0usize; self.n_units1()];
        for r in 0..self.h1() {
            for c in 0..self.w1() {
                for ch in 0..self.conv1_out {
                    let mut best = f64::NEG_INFINITY;
                    let mut best_index = 0usize;
                    for a in 0..pool.kernel_h {
                        for b in 0..pool.kernel_w {
                            let i = r * pool.stride_h + a;
                            let j = c * pool.stride_w + b;
                            let index = (i * self.raw1_w() + j) * self.conv1_out + ch;
                            if raw[index] > best {
                                best = raw[index];
                                best_index = index;
                            }
                        }
                    }
                    let k = (r * self.w1() + c) * self.conv1_out + ch;
                    values[k] = best;
                    members[k] = best_index;
                }
            }
        }
        (values, members)
    }

    /// conv2's `+/-1` weight coefficient from first-layer unit `k` to flat
    /// output `j`, or `0` when they are not connected.
    fn conv2_coefficient(&self, k: usize, j: usize) -> f64 {
        let ch = k % self.conv1_out;
        let spatial = k / self.conv1_out;
        let (ur, uc) = (spatial / self.w1(), spatial % self.w1());
        let co = j % self.conv2_out;
        let jspatial = j / self.conv2_out;
        let (or, oc) = (jspatial / self.w2(), jspatial % self.w2());
        let (Some(kr), Some(kc)) = (ur.checked_sub(or), uc.checked_sub(oc)) else {
            return 0.0;
        };
        if kr >= self.conv2_k || kc >= self.conv2_k {
            return 0.0;
        }
        f64::from(self.conv2[((co * self.conv1_out + ch) * self.conv2_k + kr) * self.conv2_k + kc])
    }
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// (a) the STE surrogate gradient against a finite-difference reference
// ---------------------------------------------------------------------------

/// The coefficient of each RAW conv1 accumulator in the LINEARIZED network at
/// `x`: masks and pool routes frozen, everything else exact. Built
/// forward-mode and independently of the module's backward pass.
fn linearized_raw_coefficients(net: &TinyNet, x: &[f64], challenger: usize, frac: f64) -> Vec<f64> {
    let raw = net.raw_z1(x);
    let (pooled, members) = net.pooled1(x);
    let mask1: Vec<bool> = {
        let window = frac * rms(&pooled);
        pooled.iter().map(|u| u.abs() <= window).collect()
    };
    let s1: Vec<f64> = pooled
        .iter()
        .map(|&v| if v >= 0.0 { 1.0 } else { -1.0 })
        .collect();

    let z2: Vec<f64> = (0..net.n_flat())
        .map(|j| {
            (0..net.n_units1())
                .map(|k| net.conv2_coefficient(k, j) * s1[k])
                .sum()
        })
        .collect();
    let mask2: Vec<bool> = {
        let window = frac * rms(&z2);
        z2.iter().map(|u| u.abs() <= window).collect()
    };
    let objective: Vec<f64> = (0..net.n_flat())
        .map(|j| {
            f64::from(net.dense[j * net.num_classes + challenger])
                - f64::from(net.dense[j * net.num_classes + net.target])
        })
        .collect();

    let mut coefficients = vec![0.0; raw.len()];
    for k in 0..net.n_units1() {
        if !mask1[k] {
            continue;
        }
        let mut acc = 0.0;
        for j in 0..net.n_flat() {
            if mask2[j] {
                acc += objective[j] * net.conv2_coefficient(k, j);
            }
        }
        coefficients[members[k]] += acc;
    }
    coefficients
}

/// The linearized objective's value at `y`, with the coefficients frozen at
/// the base point.
fn linearized_value(net: &TinyNet, coefficients: &[f64], y: &[f64]) -> f64 {
    net.raw_z1(y)
        .iter()
        .zip(coefficients)
        .map(|(z, c)| z * c)
        .sum()
}

#[test]
fn ste_gradient_matches_finite_differences() {
    let pools = [
        None,
        Some(PoolSpec {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        }),
    ];
    // The straight-through WINDOW is swept too, from "only exactly-on-threshold
    // units route gradient" to "no unit is masked at all": the surrogate's
    // defining parameter must be exactly what the documentation claims at every
    // setting, not just at the default.
    let fractions = [0.0, StePgdLimits::default().ste_fraction, 1.0, 4.0];
    let mut saw_a_live_reference = false;

    for pool in pools {
        let net = TinyNet::new(0, pool);
        let request = net.request();
        let limits = SignSpaceLimits::default();
        let admitted = admit(&request, &limits).expect("tiny net is inside the fragment");
        let challenger = net.challengers[0];
        // A point that is not the centre, so no mask sits on a symmetry.
        let x: Vec<f64> = (0..36).map(|p| ((p * 7) % 9) as f64 - 4.0).collect();
        let plan = Conv1Plan::build(&admitted);
        let trace = admitted.ste_trace(&plan, &x);

        for frac in fractions {
            let produced = admitted.ste_gradient(&plan, &trace, challenger, frac);
            let coefficients = linearized_raw_coefficients(&net, &x, challenger, frac);
            saw_a_live_reference |= coefficients.iter().any(|c| *c != 0.0);

            // The linearized objective is affine in x, so a central difference
            // is EXACT up to floating point: any step reproduces the derivative.
            let h = 0.5;
            for p in 0..x.len() {
                let mut up = x.clone();
                up[p] += h;
                let mut down = x.clone();
                down[p] -= h;
                let expected = (linearized_value(&net, &coefficients, &up)
                    - linearized_value(&net, &coefficients, &down))
                    / (2.0 * h);
                assert!(
                    (produced[p] - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                    "pixel {p}: STE gradient {} != finite difference {expected} \
                     (pool {pool:?}, ste_fraction {frac})",
                    produced[p]
                );
            }
        }
    }
    assert!(
        saw_a_live_reference,
        "the finite-difference reference was identically zero everywhere, so the \
         comparison proved nothing"
    );
}

#[test]
fn the_straight_through_window_is_what_selects_the_direction() {
    // A window of zero admits ONLY units sitting exactly on their threshold;
    // a very wide window admits every unit. If the two agree, the mask is not
    // doing anything and the comparison in the test above is vacuous.
    let net = TinyNet::new(0, None);
    let request = net.request();
    let admitted = admit(&request, &SignSpaceLimits::default()).expect("admitted");
    let x: Vec<f64> = (0..36).map(|p| ((p * 7) % 9) as f64 - 4.0).collect();
    let plan = Conv1Plan::build(&admitted);
    let trace = admitted.ste_trace(&plan, &x);
    let narrow = admitted.ste_gradient(&plan, &trace, net.challengers[0], 0.0);
    let wide = admitted.ste_gradient(&plan, &trace, net.challengers[0], 1e6);
    assert!(
        narrow != wide,
        "the straight-through window must change the search direction"
    );
    assert!(
        wide.iter().any(|g| *g != 0.0),
        "an unmasked straight-through pass must produce a live direction"
    );
}

// ---------------------------------------------------------------------------
// (b) every point the schedule produces is in the box, and integral
// ---------------------------------------------------------------------------

/// In-box with ZERO tolerance, and integral unless a clamp pinned the
/// coordinate to a non-integral bound.
fn assert_in_box_and_integral(x: &[f64], lo: &[f64], hi: &[f64]) {
    for (p, &v) in x.iter().enumerate() {
        assert!(
            v >= lo[p] && v <= hi[p],
            "pixel {p}: {v} outside [{}, {}] with zero tolerance",
            lo[p],
            hi[p]
        );
        assert!(
            v.fract() == 0.0 || v == lo[p] || v == hi[p],
            "pixel {p}: {v} is neither integral nor a box bound"
        );
    }
}

#[test]
fn schedule_steps_stay_in_the_box_and_round_to_integers() {
    // Deliberately awkward: non-integral widths, a non-integral centre, a
    // degenerate (pinned) coordinate, and a width narrower than one step.
    let lo = vec![-1.5, 0.25, -10.0, 3.0, 0.25];
    let hi = vec![2.5, 0.25, 10.0, 3.75, 0.75];
    let mut x: Vec<f64> = lo.iter().zip(&hi).map(|(l, h)| 0.5 * (l + h)).collect();
    round_into_box(&mut x, &lo, &hi);
    assert_in_box_and_integral(&x, &lo, &hi);

    let mut seed = 0xdead_beef_u64;
    for step in 0..200u32 {
        let direction: Vec<f64> = (0..lo.len())
            .map(|_| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                match (seed >> 33) % 3 {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                }
            })
            .collect();
        let alpha = f64::from(step % 7 + 1);
        schedule_step(&mut x, &direction, alpha, &lo, &hi);
        assert_in_box_and_integral(&x, &lo, &hi);
    }
}

#[test]
fn a_returned_candidate_is_exactly_in_box_and_integral() {
    // Target the WEAKEST class at the box centre, so a violation exists and
    // the lane returns a real candidate rather than exhausting.
    let probe = TinyNet::new(0, None);
    let centre: Vec<f64> = probe
        .lo
        .iter()
        .zip(&probe.hi)
        .map(|(l, h)| 0.5 * (l + h))
        .collect();
    let logits = crate::bnn_sign_space::logits_at_unwired(
        &probe.request(),
        &SignSpaceLimits::default(),
        &centre,
    )
    .expect("tiny net is inside the fragment");
    let weakest = (0..probe.num_classes)
        .min_by_key(|&c| logits[c])
        .expect("at least one class");

    let net = TinyNet::new(weakest, None);
    let outcome = falsify_bnn_ste_pgd_unwired(
        &net.request(),
        &SignSpaceLimits::default(),
        &StePgdLimits {
            max_wall_time: Duration::from_secs(5),
            ..StePgdLimits::default()
        },
    );
    let SignSpaceOutcome::Candidate(candidate) = outcome else {
        panic!("expected a candidate on a target the centre already loses: {outcome:?}");
    };
    assert!(
        candidate.logit_margin > 0,
        "a candidate must be a violation"
    );
    assert_in_box_and_integral(&candidate.input, &net.lo, &net.hi);
}

#[test]
fn a_search_that_finds_nothing_still_reports_only_data() {
    // Target the STRONGEST class with a tiny budget: whatever comes back, the
    // box/integrality contract holds and no verdict is produced.
    let probe = TinyNet::new(0, None);
    let centre: Vec<f64> = probe
        .lo
        .iter()
        .zip(&probe.hi)
        .map(|(l, h)| 0.5 * (l + h))
        .collect();
    let logits = crate::bnn_sign_space::logits_at_unwired(
        &probe.request(),
        &SignSpaceLimits::default(),
        &centre,
    )
    .expect("admitted");
    let strongest = (0..probe.num_classes)
        .max_by_key(|&c| logits[c])
        .expect("at least one class");
    let net = TinyNet::new(strongest, None);
    let outcome = falsify_bnn_ste_pgd_unwired(
        &net.request(),
        &SignSpaceLimits::default(),
        &StePgdLimits {
            max_wall_time: Duration::from_millis(400),
            ..StePgdLimits::default()
        },
    );
    match outcome {
        SignSpaceOutcome::Candidate(candidate) => {
            assert_in_box_and_integral(&candidate.input, &net.lo, &net.hi);
        }
        SignSpaceOutcome::Exhausted { .. } | SignSpaceOutcome::Refused(_) => {}
    }
}

// ---------------------------------------------------------------------------
// (c) the lane cannot emit a verified/unsat outcome
// ---------------------------------------------------------------------------

#[test]
fn the_lane_cannot_produce_a_verified_outcome() {
    let net = TinyNet::new(0, None);
    let outcome = falsify_bnn_ste_pgd_unwired(
        &net.request(),
        &SignSpaceLimits::default(),
        &StePgdLimits {
            max_wall_time: Duration::from_millis(200),
            ..StePgdLimits::default()
        },
    );
    // EXHAUSTIVE, no wildcard arm: adding a verdict-shaped variant to
    // `SignSpaceOutcome` breaks THIS match at compile time.
    let publishable = match &outcome {
        SignSpaceOutcome::Candidate(candidate) => Some(candidate.input.clone()),
        SignSpaceOutcome::Exhausted { .. } | SignSpaceOutcome::Refused(_) => None,
    };
    if let Some(input) = publishable {
        assert_in_box_and_integral(&input, &net.lo, &net.hi);
    }
}

#[test]
fn a_net_outside_the_fragment_is_refused_structurally() {
    // A conv1 weight that is not +/-1 puts the net outside the fragment. The
    // refusal is the SAME typed refusal the LP lane produces, from the SAME
    // `admit`, and it costs one structural scan — no search, no forward.
    let mut net = TinyNet::new(0, None);
    net.conv1[0] = 0.5;
    let started = std::time::Instant::now();
    let outcome = falsify_bnn_ste_pgd_unwired(
        &net.request(),
        &SignSpaceLimits::default(),
        &StePgdLimits::default(),
    );
    let elapsed = started.elapsed();
    assert!(
        matches!(outcome, SignSpaceOutcome::Refused(_)),
        "non-fragment net must be refused, got {outcome:?}"
    );
    assert!(
        elapsed < Duration::from_millis(200),
        "structural refusal must be cheap, took {elapsed:?}"
    );
}

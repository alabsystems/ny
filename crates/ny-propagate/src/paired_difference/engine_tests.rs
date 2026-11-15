// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn affine(
    rows: usize,
    cols: usize,
    wf: Vec<f64>,
    wg: Vec<f64>,
    bf: Vec<f64>,
    bg: Vec<f64>,
) -> Stage {
    Stage::Affine {
        rows,
        cols,
        wf,
        wg,
        bf,
        bg,
    }
}

fn scalar_shift(shift: f64) -> PairedChain {
    // Two ReLUs deliberately exercise transporting a certified signed strip
    // through another affine stage and activation, rather than a single gate.
    PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            affine(1, 1, vec![1.0], vec![1.0], vec![shift], vec![0.0]),
            Stage::Relu { width: 1 },
            affine(1, 1, vec![1.0], vec![1.0], vec![0.0], vec![0.0]),
            Stage::Relu { width: 1 },
            affine(1, 1, vec![1.0], vec![1.0], vec![0.0], vec![0.0]),
        ],
    }
}

fn unit_domain() -> BoxDomain {
    BoxDomain {
        lo: vec![-1.0],
        hi: vec![1.0],
    }
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

#[test]
fn signed_strip_closes_two_relu_root_that_independent_triangles_leave_open() {
    let shift = 1.0 / 1024.0;
    let epsilon = 2.0 * shift;
    let chain = scalar_shift(shift);
    let domain = unit_domain();
    let paired = chain.bound(&domain, true).unwrap();
    let plain = chain.bound(&domain, false).unwrap();
    // Exact extrema: min delta=0, max delta=shift.
    assert!(paired.lower[0] <= 0.0 && paired.lower[0] > -1e-11);
    assert!(paired.upper[0] >= shift && paired.upper[0] < epsilon);
    assert!(plain.lower[0].abs().max(plain.upper[0].abs()) > epsilon);
    let proof = chain.prove(&domain, epsilon, deadline(), 1, true);
    assert!(proof.proved && proof.closed == 1 && proof.open == 0);
    let control = chain.prove(&domain, epsilon, deadline(), 1, false);
    assert!(!control.proved && control.open > 0 && control.evaluated == 1);
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn signed(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }
}

#[test]
fn randomized_twins_with_weight_and_normalization_bias_changes_enclose_samples() {
    let mut rng = Rng(0xcafe_51a7_d00d_2026);
    let domain = BoxDomain {
        lo: vec![-0.75, -1.25, -0.125],
        hi: vec![0.75, 1.25, 0.375],
    };
    for case in 0..24 {
        let mut wf = Vec::new();
        let mut wg = Vec::new();
        for _ in 0..15 {
            let w = rng.signed();
            wf.push(w);
            wg.push(w + rng.signed() / 4096.0);
        }
        let mut bf: Vec<_> = (0..5).map(|_| rng.signed() * 0.25).collect();
        // Exercise permanently active and inactive gates alongside unstable
        // ones. The other biases remain close enough for relational coupling.
        bf[0] = 4.0;
        bf[1] = -4.0;
        let bg: Vec<_> = bf.iter().map(|&b| b + rng.signed() / 2048.0).collect();
        let first = affine(5, 3, wf, wg, bf, bg);
        let mut nf = vec![0.0; 25];
        let mut ng = vec![0.0; 25];
        for j in 0..5 {
            nf[j * 5 + j] = 0.75 + rng.next();
            ng[j * 5 + j] = nf[j * 5 + j] + rng.signed() / 4096.0;
        }
        let normalization_bias_f: Vec<_> = (0..5).map(|_| rng.signed() * 0.5).collect();
        let normalization_bias_g: Vec<_> = normalization_bias_f
            .iter()
            .map(|&b| b + rng.signed() / 4096.0)
            .collect();
        let normalization = affine(5, 5, nf, ng, normalization_bias_f, normalization_bias_g);
        let mut of = Vec::new();
        let mut og = Vec::new();
        for _ in 0..15 {
            let w = rng.signed();
            of.push(w);
            og.push(w + rng.signed() / 2048.0);
        }
        let output = affine(
            3,
            5,
            of,
            og,
            vec![0.125, -0.25, 0.5],
            vec![0.125, -0.250125, 0.499875],
        );
        let chain = PairedChain {
            input_dim: 3,
            output_dim: 3,
            stages: vec![
                first,
                Stage::Relu { width: 5 },
                normalization,
                Stage::Relu { width: 5 },
                output,
            ],
        };
        let paired = chain.bound(&domain, true).unwrap();
        let plain = chain.bound(&domain, false).unwrap();
        for sample in 0..136 {
            let x: Vec<_> = (0..3)
                .map(|j| {
                    let t = if sample < 8 {
                        ((sample >> j) & 1) as f64
                    } else {
                        rng.next()
                    };
                    domain.lo[j] + t * (domain.hi[j] - domain.lo[j])
                })
                .collect();
            let (f, g) = chain.forward(&x).unwrap();
            for j in 0..3 {
                let delta = f[j] - g[j];
                for bound in [&paired, &plain] {
                    assert!(
                        bound.lower[j] <= delta && delta <= bound.upper[j],
                        "case {case}, sample {sample}, output {j}: {} <= {delta} <= {}",
                        bound.lower[j],
                        bound.upper[j]
                    );
                }
            }
        }
    }
}

#[test]
fn identical_twins_prove_and_nonzero_output_bias_mutation_cannot_prove() {
    let mut chain = scalar_shift(0.0);
    let domain = unit_domain();
    let b = chain.bound(&domain, true).unwrap();
    assert!(b.lower[0] <= 0.0 && b.upper[0] >= 0.0);
    assert!(b.lower[0].abs().max(b.upper[0].abs()) < 1e-10);
    assert!(chain.prove(&domain, 1e-9, deadline(), 1, true).proved);
    if let Stage::Affine { bf, .. } = chain.stages.last_mut().unwrap() {
        bf[0] = 0.5;
    }
    let violation = chain.prove(&domain, 0.125, deadline(), 1000, true);
    assert!(!violation.proved && violation.open > 0);
    // A reachable violating point must remain in the enclosure even after
    // extensive splitting. No attack result is used as a proof authority.
    let point = BoxDomain {
        lo: vec![1.0],
        hi: vec![1.0],
    };
    let b = chain.bound(&point, true).unwrap();
    assert!(b.lower[0] <= 0.5 && b.upper[0] >= 0.5);
}

#[test]
fn zero_width_domains_are_proved_or_retained_as_open_points() {
    let chain = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![affine(1, 1, vec![1.0], vec![1.0], vec![0.25], vec![0.0])],
    };
    let point = BoxDomain {
        lo: vec![0.125],
        hi: vec![0.125],
    };
    let proof = chain.prove(&point, 0.5, deadline(), 4, true);
    assert!(proof.proved && proof.open == 0 && proof.closed == 1);
    let failure = chain.prove(&point, 0.125, deadline(), 4, true);
    assert!(!failure.proved && failure.open == 1 && failure.reason == "point_unclosed");
}

#[test]
fn many_large_cancelling_rows_enclose_exact_dyadic_difference() {
    const N: usize = 64;
    let mut wf = vec![1_000_000.0; N];
    let wg = wf.clone();
    let mut bf = vec![1_000_000.0; N];
    let bg = bf.clone();
    wf[0] += 1.0 / 1024.0;
    bf[0] += 1.0 / 1024.0;
    let signs: Vec<_> = (0..N)
        .map(|j| {
            if j % 2 == 0 {
                1_000_000.0
            } else {
                -1_000_000.0
            }
        })
        .collect();
    let chain = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            affine(N, 1, wf, wg, bf, bg),
            affine(1, N, signs.clone(), signs, vec![0.0], vec![0.0]),
        ],
    };
    let bound = chain.bound(&unit_domain(), true).unwrap();
    // Exact stored-real difference is (1e6 / 1024) * (x + 1).
    let exact_max = 2_000_000.0 / 1024.0;
    assert!(bound.lower[0] <= 0.0 && bound.upper[0] >= exact_max);
    for i in 0..=64 {
        let x = -1.0 + i as f64 / 32.0;
        let exact_delta = (1_000_000.0 / 1024.0) * (x + 1.0);
        assert!(bound.lower[0] <= exact_delta && exact_delta <= bound.upper[0]);
    }
    let report = chain.prove(&unit_domain(), 1900.0, deadline(), 1000, true);
    assert!(!report.proved && report.open > 0);
}

#[test]
fn subnormal_coefficient_rounding_has_an_absolute_error_charge() {
    let tiny = f64::from_bits(1);
    let chain = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![
            affine(
                2,
                1,
                vec![tiny, tiny],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
            ),
            affine(1, 2, vec![0.5, 0.5], vec![0.5, 0.5], vec![0.0], vec![0.0]),
        ],
    };
    // Each rounded product tiny*0.5 becomes zero; the EXACT sum of the two
    // products is tiny, so a purely relative dot-product error bound fails.
    let point = BoxDomain {
        lo: vec![1.0],
        hi: vec![1.0],
    };
    let b = chain.bound(&point, true).unwrap();
    assert!(b.lower[0] <= tiny && b.upper[0] >= tiny);
    assert!(dot_error(2, tiny) >= tiny);
}

#[test]
fn caps_and_expired_deadlines_never_discard_unclosed_domains() {
    let chain = scalar_shift(1.0 / 1024.0);
    let domain = unit_domain();
    let cap = chain.prove(&domain, 1.0 / 512.0, deadline(), 1, false);
    assert!(!cap.proved && cap.open == 2 && cap.reason == "domain_cap");
    let expired = chain.prove(&domain, 1.0, Instant::now(), 1000, true);
    assert!(
        !expired.proved
            && expired.evaluated == 0
            && expired.open == 1
            && expired.reason == "deadline"
    );
    let zero_cap = chain.prove(&domain, 1.0, deadline(), 0, true);
    assert!(!zero_cap.proved && zero_cap.open > 0 && zero_cap.reason == "invalid");
}

#[test]
fn malformed_shapes_nan_and_numeric_overflow_fail_closed() {
    let domain = unit_domain();
    let mut chain = scalar_shift(0.0);
    if let Stage::Affine { wf, .. } = &mut chain.stages[0] {
        wf[0] = f64::NAN;
    }
    assert!(chain.validate().is_err() && chain.bound(&domain, true).is_none());
    assert!(!chain.prove(&domain, 1.0, deadline(), 10, true).proved);
    chain = scalar_shift(0.0);
    chain.stages[1] = Stage::Relu { width: 2 };
    assert!(chain.validate().is_err() && chain.bound(&domain, true).is_none());
    chain = scalar_shift(0.0);
    assert!(chain
        .bound(
            &BoxDomain {
                lo: vec![f64::NAN],
                hi: vec![1.0]
            },
            true
        )
        .is_none());
    assert!(chain
        .bound(
            &BoxDomain {
                lo: vec![1.0],
                hi: vec![-1.0]
            },
            true
        )
        .is_none());
    assert!(chain
        .bound(
            &BoxDomain {
                lo: vec![],
                hi: vec![]
            },
            true
        )
        .is_none());
    assert!(!chain.prove(&domain, f64::NAN, deadline(), 10, true).proved);
    let oversized = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![affine(1, 1, vec![1e100], vec![1e100], vec![0.0], vec![0.0])],
    };
    let huge = BoxDomain {
        lo: vec![1e100],
        hi: vec![1e100],
    };
    assert!(oversized.validate().is_ok());
    assert!(oversized.bound(&huge, true).is_none());
    let report = oversized.prove(&huge, 1.0, deadline(), 10, true);
    assert!(!report.proved && report.open == 1 && report.reason == "bound_refused");
}

#[test]
fn search_threshold_retains_certified_forward_band_without_refining_closed_outputs() {
    let c = scalar_shift(1.0 / 1024.0);
    let root = unit_domain();
    let epsilon = 1.0 / 512.0;
    let states = forward::states(&c, &root, None, true).unwrap();
    let output = states.last().unwrap();
    assert!(output.d_lo[0] > -epsilon && output.d_hi[0] < epsilon);
    let budgeted = c.bound_until(&root, true, None, Some(epsilon)).unwrap();
    assert_eq!(budgeted.lower, output.d_lo);
    assert_eq!(budgeted.upper, output.d_hi);
    assert_eq!(budgeted.split_scores, vec![2.0]);
    let full = c.bound(&root, true).unwrap();
    assert!(full.lower[0] >= budgeted.lower[0] && full.upper[0] <= budgeted.upper[0]);
    let proof = c.prove(&root, epsilon, deadline(), 1, true);
    assert!(proof.proved && proof.open == 0 && proof.closed == 1);
}

#[test]
fn search_threshold_does_not_skip_a_reachable_violating_output() {
    let c = PairedChain {
        input_dim: 1,
        output_dim: 2,
        stages: vec![affine(
            2,
            1,
            vec![0.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0 / 1024.0, 0.0],
            vec![0.0, 0.0],
        )],
    };
    let root = unit_domain();
    let epsilon = 1.0 / 512.0;
    let states = forward::states(&c, &root, None, true).unwrap();
    let output = states.last().unwrap();
    let budgeted = c.bound_until(&root, true, None, Some(epsilon)).unwrap();
    let full = c.bound(&root, true).unwrap();
    assert_eq!(budgeted.lower[0], output.d_lo[0]);
    assert_eq!(budgeted.upper[0], output.d_hi[0]);
    assert_eq!(budgeted.lower[1], full.lower[1]);
    assert_eq!(budgeted.upper[1], full.upper[1]);
    assert!(budgeted.lower[1] <= -1.0 && budgeted.upper[1] >= 1.0);
    let failed = c.prove(&root, epsilon, deadline(), 3, true);
    assert!(!failed.proved && failed.open > 0);
}

#[test]
fn search_threshold_equality_invalid_values_and_deadline_remain_fail_closed() {
    let epsilon = 0.125;
    let c = PairedChain {
        input_dim: 1,
        output_dim: 1,
        stages: vec![affine(1, 1, vec![0.0], vec![0.0], vec![epsilon], vec![0.0])],
    };
    let point = BoxDomain {
        lo: vec![0.0],
        hi: vec![0.0],
    };
    let b = c.bound_until(&point, true, None, Some(epsilon)).unwrap();
    assert!(b.lower[0] <= epsilon && b.upper[0] >= epsilon);
    let failed = c.prove(&point, epsilon, deadline(), 1, true);
    assert!(!failed.proved && failed.open == 1);
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(c.bound_until(&point, true, None, Some(invalid)).is_none());
    }
    assert!(c
        .bound_until(&point, true, Some(Instant::now()), Some(2.0 * epsilon))
        .is_none());
    assert!(c.bound(&point, true).is_some());
}

// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(2)
}

fn expired() -> Instant {
    Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
}

fn spec(bias: f32) -> InputHeadSpec {
    InputHeadSpec::new(
        2,
        3,
        vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5],
        vec![0.0; 3],
        vec![0.5, 0.0, 0.5, 0.0, -1.0, 0.0],
        vec![bias, 0.0],
        [23; 32],
    )
    .unwrap()
}

fn source(bias: f32) -> Source {
    Source::build(&spec(bias), &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap()
}

fn cut_hint(model: &Model<'_>, negative: usize) -> HeadFunctionalLpHint {
    let mut row_duals = vec![0.0; model.request().rows.len()];
    let coordinate = model.source.spec.inputs + negative;
    let row = model
        .rows
        .iter()
        .enumerate()
        .skip(model.rows.len() - model.cuts())
        .find(|(_, row)| {
            row.coefficients
                .iter()
                .any(|&(j, a)| j == coordinate && a.hi < 0.0)
        })
        .expect("expected a source-owned Jensen cut")
        .0;
    row_duals[row] = 1.0;
    HeadFunctionalLpHint { row_duals }
}

#[test]
fn triangle_fractional_point_is_open_but_owned_jensen_dual_closes() {
    let s = source(1.0 / 16.0);
    assert!(s.baseline(deadline()).unwrap() < 0.0);
    assert!(s.replay(&s.slopes().unwrap(), deadline()).unwrap() < 0.0);
    let triangle = Model::build_with_cuts(&s, false, deadline()).unwrap();
    assert_eq!(triangle.cuts(), 0);
    // At the original input origin the outer triangle admits r_negative=1/4,
    // although the genuine network has all three heads equal to zero.
    // Check feasibility against every OWNED row enclosure, not solver status.
    let fractional = [0.0, 0.0, 0.0, 0.0, 0.25];
    for (&point, box_) in fractional.iter().zip(&triangle.boxes) {
        assert!(box_.lo <= point && point <= box_.hi);
    }
    for row in &triangle.rows {
        let lhs = row.coefficients.iter().fold(I::p(0.0), |sum, &(j, a)| {
            sum.add(a.mul(I::p(fractional[j])).unwrap()).unwrap()
        });
        assert!(
            lhs.lo >= row.lower.hi,
            "fractional point must satisfy every triangle row"
        );
    }
    assert_eq!(1.0 / 16.0 - fractional[4], -3.0 / 16.0);
    let triangle_hint = HeadFunctionalLpHint {
        row_duals: vec![0.25; triangle.rows.len()],
    };
    assert!(
        triangle
            .replay(&s, &triangle_hint, 1.0, deadline())
            .unwrap()
            <= -3.0 / 16.0
    );
    let coupled = Model::build(&s, deadline()).unwrap();
    assert!(coupled.cuts() > 0);
    let lower = coupled
        .replay(&s, &cut_hint(&coupled, 2), 1.0, deadline())
        .unwrap();
    assert!(lower > 1.0 / 32.0 && lower <= 1.0 / 16.0);
}

#[test]
fn arbitrary_full_row_duals_bound_the_authored_network_on_all_dyadic_controls() {
    let s = source(1.0 / 16.0);
    let model = Model::build(&s, deadline()).unwrap();
    for variant in 0..4 {
        let row_duals = (0..model.rows.len())
            .map(|j| {
                let magnitude = (j + 1) as f64 / 4.0;
                if (j + variant) % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect();
        let hint = HeadFunctionalLpHint { row_duals };
        for sign in [1.0, -1.0] {
            let lower = model.replay(&s, &hint, sign, deadline()).unwrap();
            for x in [-1.0_f64, 0.0, 1.0] {
                for z in [-1.0_f64, 0.0, 1.0] {
                    let actual = 1.0 / 16.0 + 0.5 * x.max(0.0) + 0.5 * z.max(0.0)
                        - f64::midpoint(x, z).max(0.0);
                    assert!(
                        lower <= actual,
                        "arbitrary duals cannot import a certified objective"
                    );
                }
            }
        }
    }
}

#[test]
fn both_dual_conventions_replay_the_same_points_and_bad_signs_are_refused() {
    let s = source(1.0 / 16.0);
    let model = Model::build(&s, deadline()).unwrap();
    let positive = cut_hint(&model, 2);
    let negative = HeadFunctionalLpHint {
        row_duals: positive.row_duals.iter().map(|v| -*v).collect(),
    };
    let a = model.replay(&s, &positive, 1.0, deadline()).unwrap();
    let b = model.replay(&s, &negative, -1.0, deadline()).unwrap();
    assert_eq!(a.to_bits(), b.to_bits());
    assert!(model.replay(&s, &positive, -1.0, deadline()).unwrap() <= 0.0);
    for sign in [0.0, 2.0, f64::NAN, f64::INFINITY] {
        assert!(model.replay(&s, &positive, sign, deadline()).is_none());
    }
}

#[test]
fn malformed_and_nonfinite_duals_cannot_publish_a_partial_replay() {
    let s = source(1.0 / 16.0);
    let model = Model::build(&s, deadline()).unwrap();
    let good = cut_hint(&model, 2);
    for fault in 0..5 {
        let mut bad = good.clone();
        match fault {
            0 => {
                bad.row_duals.pop();
            }
            1 => bad.row_duals.push(0.0),
            2 => bad.row_duals[0] = f64::NAN,
            3 => bad.row_duals[0] = f64::INFINITY,
            _ => bad.row_duals[0] = f64::MAX,
        }
        assert!(model.replay(&s, &bad, 1.0, deadline()).is_none());
    }
}

#[test]
fn identical_exported_geometry_from_a_fresh_generation_has_no_replay_authority() {
    let s = source(1.0 / 16.0);
    let foreign = source(1.0 / 16.0);
    let model = Model::build(&s, deadline()).unwrap();
    let other = Model::build(&foreign, deadline()).unwrap();
    assert_eq!(model.request().bounds, other.request().bounds);
    assert_eq!(model.request().objective, other.request().objective);
    assert!(!Arc::ptr_eq(&s.spec, &foreign.spec));
    assert!(model
        .replay(&foreign, &cut_hint(&model, 2), 1.0, deadline())
        .is_none());
}

#[test]
fn negative_and_zero_original_bias_remain_unsafe_after_the_same_jensen_cut() {
    for bias in [-1.0 / 16.0, 0.0] {
        let s = source(bias);
        let model = Model::build(&s, deadline()).unwrap();
        let lower = model
            .replay(&s, &cut_hint(&model, 2), 1.0, deadline())
            .unwrap();
        // The original origin is a genuine witness with margin exactly bias.
        assert!(lower <= f64::from(bias) && lower <= 0.0);
    }
}

#[test]
fn the_last_original_input_is_contracted_in_every_mass_cut() {
    let spec = InputHeadSpec::new(
        3,
        3,
        vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0],
        vec![0.0; 3],
        vec![0.5, 0.0, 0.5, 0.0, -1.0, 0.0],
        vec![1.0 / 16.0, 0.0],
        [24; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 3], &[1.0; 3], 0, 1, deadline()).unwrap();
    let cut = MassCut {
        positive: vec![(0, 0.5), (1, 0.5)],
        negative: 2,
        tau: 1.0,
    };
    // At (0,0,1), the last authored feature makes the actual margin -15/16.
    assert!(source_cut_lower(&s, &cut, deadline()).unwrap() <= -1.0);
    let model = Model::build(&s, deadline()).unwrap();
    assert!(
        model
            .replay(&s, &cut_hint(&model, 2), 1.0, deadline())
            .unwrap()
            <= -15.0 / 16.0
    );
}

#[test]
fn every_remaining_head_including_the_last_head_is_repriced() {
    let spec = InputHeadSpec::new(
        2,
        4,
        vec![1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0],
        vec![0.0, 0.0, 0.0, 2.0],
        vec![0.5, 0.0, 0.5, 0.0, -1.0, 0.0, -1.0, 0.0],
        vec![1.0 / 16.0, 0.0],
        [25; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 2], &[1.0; 2], 0, 1, deadline()).unwrap();
    let model = Model::build(&s, deadline()).unwrap();
    let lower = model
        .replay(&s, &cut_hint(&model, 2), 1.0, deadline())
        .unwrap();
    // The untouched last ReLU is constantly2: the original origin is unsafe.
    assert!(lower <= 1.0 / 16.0 - 2.0);
}

#[test]
fn full_q_bias_uncertainty_and_output_error_cannot_be_nominally_cancelled() {
    for fault in 0..3 {
        let mut s = source(1.0 / 16.0);
        match fault {
            0 => {
                s.q[0] = I {
                    lo: 3.0 / 8.0,
                    hi: 5.0 / 8.0,
                };
                s.q[1] = s.q[0];
                s.q[2] = I {
                    lo: -9.0 / 8.0,
                    hi: -7.0 / 8.0,
                };
            }
            1 => {
                s.bias = I {
                    lo: -1.0 / 16.0,
                    hi: 1.0 / 16.0,
                }
            }
            _ => s.output_error = 0.25,
        }
        let model = Model::build(&s, deadline()).unwrap();
        assert!(
            model
                .replay(&s, &cut_hint(&model, 2), 1.0, deadline())
                .unwrap()
                <= 0.0
        );
    }
}

#[test]
fn independent_first_layer_errors_survive_exact_affine_cancellation() {
    let mut s = source(1.0 / 16.0);
    s.first_error.fill(0.25);
    let cut = MassCut {
        positive: vec![(0, 0.5), (1, 0.5)],
        negative: 2,
        tau: 1.0,
    };
    assert!(source_cut_lower(&s, &cut, deadline()).unwrap() <= -0.5);
}

#[test]
fn direct_nonunit_and_overdrawn_point_masses_keep_their_source_debt() {
    let s = source(1.0 / 16.0);
    let cut = MassCut {
        positive: vec![(0, 2.0), (1, 0.125)],
        negative: 2,
        tau: 3.0,
    };
    let lower = source_cut_lower(&s, &cut, deadline()).unwrap();
    assert!(lower <= -15.0 / 8.0);
    for x in [-1.0_f64, 0.0, 1.0] {
        for z in [-1.0_f64, 0.0, 1.0] {
            let actual = 2.0 * x.max(0.0) + 0.125 * z.max(0.0) - 3.0 * f64::midpoint(x, z).max(0.0);
            assert!(lower <= actual);
        }
    }
    for bad in [
        MassCut {
            positive: vec![],
            negative: 2,
            tau: 1.0,
        },
        MassCut {
            positive: vec![(0, 0.5), (0, 0.5)],
            negative: 2,
            tau: 1.0,
        },
        MassCut {
            positive: vec![(2, 1.0)],
            negative: 2,
            tau: 1.0,
        },
        MassCut {
            positive: vec![(0, -0.5)],
            negative: 2,
            tau: 1.0,
        },
        MassCut {
            positive: vec![(0, 1.0)],
            negative: 3,
            tau: 1.0,
        },
        MassCut {
            positive: vec![(0, 1.0)],
            negative: 2,
            tau: f64::NAN,
        },
    ] {
        assert!(source_cut_lower(&s, &bad, deadline()).is_none());
    }
}

#[test]
fn subnormal_point_product_is_enclosed_before_large_input_support() {
    let weight = f32::from_bits(1);
    let spec = InputHeadSpec::new(
        1,
        2,
        vec![weight, 0.0],
        vec![0.0; 2],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![0.0; 2],
        [26; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1e30], &[1e30], 0, 1, deadline()).unwrap();
    let tiny = f64::from_bits(1);
    assert_eq!(tiny * f64::from(weight), 0.0);
    let cut = MassCut {
        positive: vec![(0, tiny)],
        negative: 1,
        tau: 0.0,
    };
    let lower = source_cut_lower(&s, &cut, deadline()).unwrap();
    assert!(
        lower.is_finite() && lower < -1e-300,
        "the directed lost-product width must be priced before input amplification"
    );
}

#[test]
fn original_thirty_input_128_head_layout_uses_only_genuine_relu_auxiliaries() {
    let mut first = vec![0.0; 30 * 128];
    first[29 * 128 + 127] = 1.0;
    let mut final_weight = vec![0.0; 128 * 2];
    final_weight[127 * 2] = 1.0;
    let spec = InputHeadSpec::new(
        30,
        128,
        first,
        vec![0.0; 128],
        final_weight,
        vec![1.0 / 16.0, 0.0],
        [27; 32],
    )
    .unwrap();
    let s = Source::build(&spec, &[-1.0; 30], &[1.0; 30], 0, 1, deadline()).unwrap();
    let model = Model::build(&s, deadline()).unwrap();
    let request = model.request();
    assert!(request.validate());
    assert_eq!(request.feature_count, 30);
    assert_eq!(request.bounds.len(), 158);
    assert_eq!(request.objective.len(), 158);
    assert_eq!(request.slope_rows.len(), 128);
    assert!(request.slope_rows.iter().all(Option::is_some));
    assert_eq!((request.phase_rows).len(), 0);
    let row = &model.rows[request.slope_rows[127].unwrap()];
    assert!(row
        .coefficients
        .iter()
        .any(|&(j, a)| j == 29 && a.lo == -1.0 && a.hi == -1.0));
    assert!(row
        .coefficients
        .iter()
        .any(|&(j, a)| j == 157 && a.lo == 1.0 && a.hi == 1.0));
}

#[test]
fn completed_improvement_is_rolled_back_when_original_authority_expires() {
    let s = source(1.0 / 16.0);
    let before = s.replay(&s.slopes().unwrap(), deadline()).unwrap();
    assert!(before < 0.0);
    let model = Model::build(&s, deadline()).unwrap();
    let hint = cut_hint(&model, 2);
    let improved = model.replay(&s, &hint, 1.0, deadline()).unwrap();
    assert!(improved > 1.0 / 32.0);
    assert_eq!(
        commit_triangle_best(before, improved, deadline(), deadline()).to_bits(),
        improved.to_bits()
    );
    assert_eq!(
        commit_triangle_best(before, improved, deadline(), expired()).to_bits(),
        before.to_bits()
    );
    assert_eq!(
        commit_triangle_best(before, improved, expired(), deadline()).to_bits(),
        before.to_bits()
    );
    assert!(model.replay(&s, &hint, 1.0, expired()).is_none());
    assert!(Model::build(&s, expired()).is_none());
}

/// review(eng14) — ENUMERATION-EXACT soundness of every OWNED input-head row
/// (genuine head epigraphs, triangles, legacy Jensen cuts, fitted cone cuts
/// and joint negative-group rows) and of weak-dual replay. Over seeded random
/// dense heads — cone-shaped four-feature heads (a negative head that is a
/// convex combination of four coordinate heads, as in the cone fixtures),
/// joint-negative two-feature heads (every head unstable with negative Q, as
/// in the group fixtures) and unstructured two-feature heads — on random
/// boxes, the exact minimum of the ORIGINAL margin over the box comes from the
/// arrangement oracle, and:
/// 1. every owned row holds at every arrangement vertex (its supremum over the
///    owned interval coefficients reaches its owned lower end; the row is
///    affine in the input and the head ReLUs, so the vertices decide it);
/// 2. neither the scalar baseline nor the slope replay exceeds the minimum;
/// 3. neither do arbitrary signed duals nor a greedy dual ASCENT that
///    maximises the replay — the adversarial search an unsound row or an
///    uncharged term would let climb past the minimum.
///
/// Fails VOID when no trial builds a legacy cut, a cone cut or a negative
/// group row, or the ascent never improves on its start.
#[test]
fn owned_rows_and_adversarial_dual_replay_never_exceed_the_exact_arrangement_minimum() {
    use crate::margin_row::exact_relu_oracle::{rat, relu, value, vertices, Line};
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        /// A dyadic `k / den` with `|k| <= span`.
        fn dyadic(&mut self, span: usize, den: f32) -> f32 {
            (self.below(2 * span + 1) as f32 - span as f32) / den
        }
    }
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (mut built, mut cuts, mut cones, mut groups) = (0usize, 0usize, 0usize, 0usize);
    let (mut replays, mut climbed) = (0usize, 0usize);
    for trial in 0..60 {
        let shape = trial % 3;
        let (d, h) = match shape {
            0 => (4, 5 + rng.below(2)),
            1 => (2, 4),
            _ => (2, 6),
        };
        let mut first = vec![0.0f32; d * h];
        let mut first_bias = vec![0.0f32; h];
        let mut final_weight = vec![0.0f32; h * 2];
        let final_bias: [f32; 2] = match shape {
            0 => {
                const MIXES: [[f32; 4]; 4] = [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.25, 0.125, 0.125],
                    [0.375, 0.375, 0.125, 0.125],
                    [0.125, 0.375, 0.25, 0.25],
                ];
                let mix = MIXES[rng.below(4)];
                let tau = if rng.below(2) == 0 { 0.5 } else { 1.0 };
                for k in 0..d {
                    first[k * h + k] = 0.5 + rng.below(3) as f32 / 4.0;
                    first_bias[k] = rng.dyadic(2, 16.0);
                    final_weight[k * 2] = tau * mix[k] + rng.below(2) as f32 / 16.0;
                }
                // Head d: the mix of the four coordinate heads, shifted.
                for k in 0..d {
                    first[k * h + d] = mix[k] * first[k * h + k];
                }
                first_bias[d] =
                    (0..d).map(|k| mix[k] * first_bias[k]).sum::<f32>() + rng.dyadic(1, 32.0);
                final_weight[d * 2] = -tau;
                for j in d + 1..h {
                    for k in 0..d {
                        first[k * h + j] = rng.dyadic(4, 8.0);
                    }
                    first_bias[j] = rng.dyadic(4, 8.0);
                    final_weight[j * 2] = rng.dyadic(2, 8.0);
                    final_weight[j * 2 + 1] = rng.dyadic(2, 8.0);
                }
                [rng.dyadic(2, 16.0), 0.0]
            }
            1 => {
                let scale = if rng.below(2) == 0 { 0.5 } else { 1.0 };
                for (j, (sx, sz)) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
                    .into_iter()
                    .enumerate()
                {
                    first[j] = sx * scale;
                    first[h + j] = sz * scale;
                    first_bias[j] = rng.dyadic(1, 16.0);
                    final_weight[j * 2] = -((1 + rng.below(4)) as f32) / 4.0;
                }
                [(rng.below(5) as f32 + 1.0) * scale, 0.0]
            }
            _ => {
                for v in first.iter_mut() {
                    *v = rng.dyadic(8, 8.0);
                }
                for v in first_bias.iter_mut() {
                    *v = rng.dyadic(4, 8.0);
                }
                for v in final_weight.iter_mut() {
                    *v = rng.dyadic(8, 8.0);
                }
                [rng.dyadic(8, 8.0), rng.dyadic(8, 8.0)]
            }
        };
        let lo: Vec<f64> = (0..d).map(|_| -((1 + rng.below(8)) as f64) / 8.0).collect();
        let hi: Vec<f64> = (0..d).map(|_| (1 + rng.below(8)) as f64 / 8.0).collect();
        let spec = InputHeadSpec::new(
            d,
            h,
            first.clone(),
            first_bias.clone(),
            final_weight.clone(),
            final_bias.to_vec(),
            [97; 32],
        )
        .unwrap();
        let Some(s) = Source::build(&spec, &lo, &hi, 0, 1, deadline()) else {
            continue;
        };
        // The exact original margin: first is [input, hidden], final is
        // [hidden, output], both row-major, all values exact dyadics.
        let lines: Vec<Line> = (0..h)
            .map(|j| {
                let a: Vec<f64> = (0..d).map(|k| f64::from(first[k * h + j])).collect();
                Line::of(&a, f64::from(first_bias[j]))
            })
            .collect();
        let q: Vec<BigRational> = (0..h)
            .map(|j| rat(f64::from(final_weight[j * 2])) - rat(f64::from(final_weight[j * 2 + 1])))
            .collect();
        let c = rat(f64::from(final_bias[0])) - rat(f64::from(final_bias[1]));
        let rlo: Vec<BigRational> = lo.iter().map(|&v| rat(v)).collect();
        let rhi: Vec<BigRational> = hi.iter().map(|&v| rat(v)).collect();
        let points = vertices(&rlo, &rhi, &lines);
        let exact = points
            .iter()
            .map(|x| value(&c, &q, &lines, x))
            .min()
            .unwrap();
        let below = |v: f64| rat(v) <= exact;
        assert!(
            below(s.baseline(deadline()).unwrap()),
            "trial {trial}: baseline"
        );
        if let Some(bound) = s.slopes().and_then(|v| s.replay(&v, deadline())) {
            assert!(below(bound), "trial {trial}: slope replay");
        }
        let Some(model) = Model::build(&s, deadline()) else {
            continue;
        };
        built += 1;
        cuts += model.cuts();
        cones += model.cone_cuts();
        groups += model.negative_groups();
        // 1. Every owned row at every arrangement vertex.
        for (r, row) in model.rows.iter().enumerate() {
            let lower = rat(row.lower.lo);
            for x in &points {
                let z: Vec<BigRational> = x
                    .iter()
                    .cloned()
                    .chain(lines.iter().map(|l| relu(l.at(x))))
                    .collect();
                let sup = row
                    .coefficients
                    .iter()
                    .fold(BigRational::zero(), |sum, &(j, a)| {
                        let pick = if z[j].is_negative() { a.lo } else { a.hi };
                        sum + rat(pick) * &z[j]
                    });
                assert!(
                    sup >= lower,
                    "trial {trial} row {r}: an owned row fails at the exact vertex {x:?}"
                );
            }
        }
        // 2-3. Replay under arbitrary and adversarially ascended duals.
        let n = model.rows.len();
        let replay = |duals: &[f64], sign: f64| {
            let hint = HeadFunctionalLpHint {
                row_duals: duals.to_vec(),
            };
            model.replay(&s, &hint, sign, deadline())
        };
        for _ in 0..6 {
            let duals: Vec<f64> = (0..n).map(|_| f64::from(rng.dyadic(16, 4.0))).collect();
            for sign in [1.0, -1.0] {
                if let Some(bound) = replay(&duals, sign) {
                    replays += 1;
                    assert!(below(bound), "trial {trial}: arbitrary duals");
                }
            }
        }
        let mut duals = vec![0.0; n];
        let start = replay(&duals, 1.0).expect("zero duals replay the box floor");
        let mut best = start;
        for _ in 0..400 {
            let row = rng.below(n);
            let step = [0.03125, 0.125, 0.5, 2.0][rng.below(4)];
            let mut next = duals.clone();
            next[row] = if rng.below(2) == 0 {
                next[row] + step
            } else {
                (next[row] - step).max(0.0)
            };
            if let Some(bound) = replay(&next, 1.0) {
                if bound > best {
                    best = bound;
                    duals = next;
                }
            }
        }
        assert!(
            below(best),
            "trial {trial}: the dual ascent passed the exact minimum"
        );
        climbed += usize::from(best > start);
    }
    eprintln!(
        "[input-head-exact] models={built} cuts={cuts} cones={cones} groups={groups} \
         replays={replays} climbed={climbed}"
    );
    assert!(built >= 30, "VOID: only {built} models");
    assert!(
        cuts > 0 && cones > 0 && groups > 0,
        "VOID: cuts={cuts} cones={cones} groups={groups}"
    );
    assert!(
        replays > 0 && climbed > 0,
        "VOID: replays={replays} climbed={climbed}"
    );
}

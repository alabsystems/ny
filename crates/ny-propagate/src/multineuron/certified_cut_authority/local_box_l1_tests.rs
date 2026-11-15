// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::layers::ReLULayer;
use crate::network::GraphNode;
use ndarray::{arr1, Array1, Array2};
use num_rational::BigRational as Q;
use num_traits::{Signed, ToPrimitive, Zero};
use std::time::Duration;

#[path = "local_box_l1_low_arity_tests.rs"]
mod low_arity_tests;

fn q(x: f32) -> Q {
    Q::from_float(f32_to_f64_exact(x)).unwrap()
}
fn q64(x: f64) -> Q {
    Q::from_float(x).unwrap()
}
fn end() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

fn fixture(
    w: Array2<f32>,
    b: Option<Array1<f32>>,
    bn: Option<BatchNormLayer>,
    lower: Vec<f32>,
    upper: Vec<f32>,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "linear",
        Layer::Linear(LinearLayer::new(w, b).unwrap()),
    ));
    let pre = if let Some(bn) = bn {
        g.add_node(GraphNode::new(
            "bn",
            Layer::BatchNorm(bn),
            vec!["linear".into()],
        ));
        "bn"
    } else {
        "linear"
    };
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec![pre.into()],
    ));
    g.set_output("relu");
    let x = BoundedTensor::new(
        Array1::from(lower).into_dyn(),
        Array1::from(upper).into_dyn(),
    )
    .unwrap();
    let map = g.collect_node_bounds(&x).unwrap();
    (g, x, map)
}
fn run<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    n: &[usize],
    w: &[f32],
) -> LocalBoxL1Arithmetic<'a> {
    let Layer::Linear(l) = g.node("linear").unwrap().layer() else {
        unreachable!()
    };
    evaluate(
        g,
        x,
        b,
        "relu",
        n,
        w,
        CertifiedWeightError::EXACT,
        quote(l.in_features(), l.out_features()).unwrap(),
        end(),
        &mut || true,
    )
    .unwrap()
}
fn sign(row: usize, i: usize) -> Q {
    q(if i == 0 || row & (1 << (i - 1)) == 0 {
        1.
    } else {
        -1.
    })
}
fn exact_center(s: &LocalBoxL1Arithmetic<'_>, row: usize) -> (Q, Q) {
    let l = s.source.linear;
    let e = l.in_features();
    let flat = l.weight.as_slice().unwrap();
    let bias = l.bias.as_ref().map(|v| v.as_slice().unwrap());
    let mut a = vec![Q::zero(); e];
    let mut c = Q::zero();
    for k in 0..s.active {
        let n = s.neurons[k];
        let factor = sign(row, k) * q(s.weights[k]);
        let (scale, offset) = s.source.bn.map_or((q(1.), q(0.)), |b| {
            (
                q(b.scale.as_slice().unwrap()[n]),
                q(b.bias.as_slice().unwrap()[n]),
            )
        });
        c += &factor * (&scale * q(bias.map_or(0., |b| b[n])) + offset);
        for j in 0..e {
            a[j] += &factor * &scale * q(flat[n * e + j]);
        }
    }
    let mut lo = c.clone();
    let mut hi = c;
    for (j, a) in a.iter().enumerate() {
        let l = q(s.source.input.lower().as_slice().unwrap()[j]);
        let u = q(s.source.input.upper().as_slice().unwrap()[j]);
        if a < &Q::zero() {
            lo += a * &u;
            hi += a * &l;
        } else {
            lo += a * &l;
            hi += a * &u;
        }
    }
    (lo, hi)
}

#[test]
fn local_l1_sign_recurrence_encloses_exact_affine_rows_three_through_eight() {
    for m in 3..=8 {
        let a = Array2::from_shape_fn((m, 2), |(i, j)| {
            ((i as i32 * 3 + j as i32 * 5) % 9 - 4) as f32 / 8.
        });
        let b = Array1::from_shape_fn(m, |i| (i as i32 - 3) as f32 / 16.);
        let (g, x, map) = fixture(a, Some(b), None, vec![-0.75, -1.], vec![1., 0.5]);
        let weights: Vec<_> = (0..m).map(|i| (i + 1) as f32 / 8.).collect();
        let indices: Vec<_> = (0..m).collect();
        let s = run(&g, &x, &map, &indices, &weights);
        assert_eq!(s.sign_count(), 1 << (m - 1));
        assert!(std::ptr::eq(s.source.graph, &raw const g));
        assert!(std::ptr::eq(s.source.root_input, &raw const x));
        assert!(std::ptr::eq(s.source.map, &raw const map));
        assert_eq!(s.source.target, "relu");
        assert_eq!(s.source.linear_parent, NETWORK_INPUT);
        assert_eq!(s.neurons(), indices);
        assert_eq!(s.weights(), weights);
        for r in 0..s.sign_count() {
            let (l, u) = exact_center(&s, r);
            let (sl, su) = s.sign_bounds(r).unwrap();
            assert!(q64(sl) <= l && u <= q64(su));
            assert!(-l <= q(s.bound()) && u <= q(s.bound()));
        }
        assert!(s.sign_bounds(s.sign_count()).is_none());
        assert_eq!(s.bn_radius(), 0.);
    }
    let (g, x, map) = fixture(
        Array2::zeros((3, 2)),
        None,
        None,
        vec![0., 0.],
        vec![0., 0.],
    );
    let s = run(&g, &x, &map, &[0, 1, 2], &[1.; 3]);
    assert_eq!(s.bound(), 0.); // A stable point is not an empty-set certificate.
}

#[test]
fn local_l1_raw_bn_common_radius_encloses_original_parameter_and_input_corners() {
    let mut bn = BatchNormLayer::from_scale_bias(
        arr1(&[-0.75, 0., 1.25]).into_dyn(),
        arr1(&[0.5, -0.25, 0.]).into_dyn(),
    )
    .unwrap();
    bn.scale_err = arr1(&[0.125, 0.25, 0.0625]).into_dyn();
    bn.bias_err = arr1(&[0.0625, 0.125, 0.25]).into_dyn();
    let a = Array2::from_shape_vec((3, 2), vec![0.5, -0.25, -0.75, 0.5, 0.25, 0.125]).unwrap();
    let (g, x, map) = fixture(
        a,
        Some(arr1(&[0.25, -0.5, 0.125])),
        Some(bn),
        vec![-1., -0.5],
        vec![0.75, 1.],
    );
    let s = run(&g, &x, &map, &[0, 1, 2], &[0.25, 0.5, 1.]);
    let bn = s.source.bn.unwrap();
    let a = s.source.linear.weight.as_slice().unwrap();
    let b = s.source.linear.bias.as_ref().unwrap().as_slice().unwrap();
    assert!(s.bn_radius() > 0.);
    for xp in 0..4 {
        let xx = [
            if xp & 1 == 0 { -1. } else { 0.75 },
            if xp & 2 == 0 { -0.5 } else { 1. },
        ];
        let t: Vec<Q> = (0..3)
            .map(|i| q(b[i]) + q(a[2 * i]) * q(xx[0]) + q(a[2 * i + 1]) * q(xx[1]))
            .collect();
        for i in 0..3 {
            assert!(t[i].abs() <= q64(s.linear_abs[i]));
        }
        for parameters in 0..64 {
            for row in 0..4 {
                let mut actual = Q::zero();
                let mut center = Q::zero();
                for i in 0..3 {
                    let ss = q(bn.scale.as_slice().unwrap()[i]);
                    let bb = q(bn.bias.as_slice().unwrap()[i]);
                    let es = q(bn.scale_err.as_slice().unwrap()[i]);
                    let eb = q(bn.bias_err.as_slice().unwrap()[i]);
                    let sc = if parameters & (1 << (2 * i)) == 0 {
                        &ss - &es
                    } else {
                        &ss + &es
                    };
                    let bc = if parameters & (1 << (2 * i + 1)) == 0 {
                        &bb - &eb
                    } else {
                        &bb + &eb
                    };
                    let k = sign(row, i) * q(s.weights[i]);
                    actual += &k * (sc * &t[i] + bc);
                    center += k * (ss * &t[i] + bb);
                }
                let (lo, hi) = s.sign_bounds(row).unwrap();
                assert!(q64(lo) <= actual && actual <= q64(hi));
                assert!(actual.abs() <= q(s.bound()));
                assert!((actual - center).abs() <= q64(s.bn_radius()));
            }
        }
    }
    // Zero stored scale with positive radius is locally sound. This does NOT
    // remove the current full extractor's relative-error conversion refusal.
    assert_eq!(bn.scale.as_slice().unwrap()[1], 0.);
}

#[test]
fn local_l1_bn_triple_rounding_cancellation_and_subnormals_are_outward() {
    ny_core::require_f64_interval_proof_environment().unwrap();
    let w = 255f32 / 256.;
    let a = f32::from_bits(1f32.to_bits() + 1);
    let exact = q(w) * q(a) * q(a);
    let naive = f32_to_f64_exact(w) * f32_to_f64_exact(a) * f32_to_f64_exact(a);
    assert_ne!(q64(naive), exact);
    let got = Interval::point(w)
        .unwrap()
        .mul(Interval::point(a).unwrap())
        .unwrap()
        .mul(Interval::point(a).unwrap())
        .unwrap();
    assert!(q64(got.lo) <= exact && exact <= q64(got.hi));
    let neg = got.neg();
    assert!(q64(neg.lo) <= -exact.clone() && -exact <= q64(neg.hi));
    let cancel = got.add(neg).unwrap();
    assert!(cancel.lo <= 0. && cancel.hi >= 0.);
    let tiny = f32::from_bits(1);
    let product = Interval::point(tiny)
        .unwrap()
        .mul(Interval::point(tiny).unwrap())
        .unwrap()
        .mul(Interval::point(tiny).unwrap())
        .unwrap();
    let exact = q(tiny) * q(tiny) * q(tiny);
    assert!(q64(product.lo) <= exact && exact <= q64(product.hi) && product.hi > 0.);
    // Nonzero underflow may round to0: operand-based zero shortcuts only.
    let min = f64::from_bits(1);
    let under = Interval::checked(min, min)
        .unwrap()
        .mul(Interval { lo: 0.5, hi: 0.5 })
        .unwrap();
    assert!(q64(under.lo) <= q64(min) / q(2.) && q64(min) / q(2.) <= q64(under.hi));
}

fn exact_l1_box(a: &Array2<f32>, selected: &[usize]) -> Q {
    let e = a.ncols();
    let m = selected.len();
    let mut best = Q::zero();
    for r in 0..(1 << (m - 1)) {
        let mut v = Q::zero();
        for j in 0..e {
            let c = (0..m)
                .map(|i| sign(r, i) * q(a[[selected[i], j]]))
                .sum::<Q>();
            v += c.abs();
        }
        best = best.max(v);
    }
    best
}

#[test]
fn local_l1_original_four_diamonds_strictly_separate_pair_triplet_and_unsafe() {
    let mut a = Array2::zeros((9, 8));
    for r in 0..4 {
        a[[2 * r, 2 * r]] = 0.5;
        a[[2 * r, 2 * r + 1]] = 0.5;
        a[[2 * r + 1, 2 * r]] = 0.5;
        a[[2 * r + 1, 2 * r + 1]] = -0.5;
        a[[8, 2 * r]] = 1.;
    }
    let mut biases = Array1::zeros(9);
    biases[8] = 5.;
    let (g, x, map) = fixture(a.clone(), Some(biases), None, vec![-1.; 8], vec![1.; 8]);
    let s = run(&g, &x, &map, &[0, 1, 2, 3, 4, 5, 6, 7], &[1.; 8]);
    assert_eq!(s.sign_count(), 128);
    assert_eq!(exact_l1_box(&a, &[0, 1, 2, 3, 4, 5, 6, 7]), q(4.));
    assert!(s.bound() >= 4. && s.bound() < 4.001);
    let mut pair_best = q(-100.);
    let mut triplet_best = q(-100.);
    for i in 0..8 {
        for j in i + 1..8 {
            let bi = exact_l1_box(&a, &[i, j]);
            for k in 0..8 {
                for l in k + 1..8 {
                    if [i, j].contains(&k) || [i, j].contains(&l) {
                        continue;
                    }
                    let bj = exact_l1_box(&a, &[k, l]);
                    for x in [0.5, 1.] {
                        for y in [0.5, 1.] {
                            pair_best = pair_best
                                .max(q(-3.75) + q(x) * (q(2.) - &bi) + q(y) * (q(2.) - &bj));
                        }
                    }
                }
            }
            for k in j + 1..8 {
                let bt = exact_l1_box(&a, &[i, j, k]);
                for l in [0.5, 1.] {
                    triplet_best = triplet_best.max(q(-3.75) + q(l) * (q(3.) - &bt));
                }
            }
        }
    }
    assert_eq!(pair_best, q(-1.75));
    assert_eq!(triplet_best, q(-2.75));
    for shift in [0., -0.5] {
        let mut minimum = q(100.);
        for mask in 0..256 {
            let xx: Vec<Q> = (0..8)
                .map(|i| q(if mask & (1 << i) == 0 { -1. } else { 1. }))
                .collect();
            let z: Vec<Q> = (0..8)
                .map(|i| (0..8).map(|j| q(a[[i, j]]) * &xx[j]).sum())
                .collect();
            let stable = q(5.) + (0..4).map(|r| xx[2 * r].clone()).sum::<Q>();
            assert!(q(1.) <= stable && stable <= q(9.));
            // Independent original authored chain, not a supplied bound scalar.
            let f = stable - q(0.75) + q(shift)
                - q(2.) * z.iter().map(|z| z.clone().max(Q::zero())).sum::<Q>();
            assert_eq!(f, q(4.25 + shift) - z.iter().map(|z| z.abs()).sum::<Q>());
            minimum = minimum.min(f);
        }
        assert_eq!(minimum, q(0.25 + shift));
        // Algebra of exact post(+2)/pre(-1)/bias(-B) cancellation at lambda1.
        // There is intentionally no full backend or root authority in this kernel test.
        let lower = q(4.25 + shift) - q(s.bound());
        assert!(lower <= minimum);
        if shift == 0. {
            assert!(lower > q(0.24));
        } else {
            assert!(lower <= q(-0.25));
        }
    }
}

#[test]
fn local_l1_each_poll_expired_entry_and_exact_budget_edges_publish_nothing() {
    let (g, x, map) = fixture(Array2::eye(3), None, None, vec![-1.; 3], vec![1.; 3]);
    let paid = quote(3, 3).unwrap();
    let mut calls = 0;
    let s = evaluate(
        &g,
        &x,
        &map,
        "relu",
        &[0, 1, 2],
        &[1.; 3],
        CertifiedWeightError::EXACT,
        paid,
        end(),
        &mut || {
            calls += 1;
            true
        },
    )
    .unwrap();
    assert_eq!(s.quote(), paid);
    for stop in 1..=calls {
        let mut n = 0;
        assert!(matches!(
            evaluate(
                &g,
                &x,
                &map,
                "relu",
                &[0, 1, 2],
                &[1.; 3],
                CertifiedWeightError::EXACT,
                paid,
                end(),
                &mut || {
                    n += 1;
                    n != stop
                }
            ),
            Err(Refusal::Cancelled)
        ));
        assert_eq!(n, stop);
    }
    for under in [
        LocalBoxL1Quote {
            work: paid.work - 1,
            ..paid
        },
        LocalBoxL1Quote {
            scratch_bytes: paid.scratch_bytes - 1,
            ..paid
        },
    ] {
        assert!(matches!(
            evaluate(
                &g,
                &x,
                &map,
                "relu",
                &[0, 1, 2],
                &[1.; 3],
                CertifiedWeightError::EXACT,
                under,
                end(),
                &mut || true
            ),
            Err(Refusal::Budget)
        ));
    }
    let mut polls = 0;
    assert!(matches!(
        evaluate(
            &g,
            &x,
            &map,
            "relu",
            &[0, 1, 2],
            &[1.; 3],
            CertifiedWeightError::EXACT,
            paid,
            Instant::now(),
            &mut || {
                polls += 1;
                true
            }
        ),
        Err(Refusal::Deadline)
    ));
    assert_eq!(polls, 0);
    let mut cancelled = || false;
    let mut guard = Guard {
        deadline: end(),
        poll: &mut cancelled,
        failed: None,
    };
    assert_eq!(guard.check(), Err(Refusal::Cancelled));
    assert_eq!(guard.check(), Err(Refusal::Cancelled));
    assert_eq!(quote(32768, 512).unwrap().work, 2_149_711_872);
    assert_eq!(quote(65536, 8192).unwrap().work, 4_297_195_520);
    assert!(quote(65537, 3).is_none());
    assert!(quote(1, 8193).is_none());
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep the original fixture owner intact while mutating an independent candidate."
)]
fn local_l1_source_shape_weights_uncertainty_and_unsafe_values_refuse() {
    let (g, x, map) = fixture(Array2::eye(3), None, None, vec![-1.; 3], vec![1.; 3]);
    let paid = quote(3, 3).unwrap();
    for error in [
        CertifiedWeightError {
            weight_rel_err: f32::from_bits(1),
            bias_abs_err: 0.,
        },
        CertifiedWeightError {
            weight_rel_err: 0.,
            bias_abs_err: 0.125,
        },
        CertifiedWeightError {
            weight_rel_err: f32::NAN,
            bias_abs_err: 0.,
        },
    ] {
        assert!(matches!(
            evaluate(
                &g,
                &x,
                &map,
                "relu",
                &[0, 1, 2],
                &[1.; 3],
                error,
                paid,
                end(),
                &mut || true
            ),
            Err(Refusal::Source)
        ));
    }
    for n in [[0, 0, 2], [0, 2, 1], [0, 1, 3]] {
        assert!(evaluate(
            &g,
            &x,
            &map,
            "relu",
            &n,
            &[1.; 3],
            CertifiedWeightError::EXACT,
            paid,
            end(),
            &mut || true
        )
        .is_err());
    }
    for w in [0., -1., 0.1, f32::from_bits(1), f32::NAN] {
        assert!(matches!(
            evaluate(
                &g,
                &x,
                &map,
                "relu",
                &[0, 1, 2],
                &[w, 1., 1.],
                CertifiedWeightError::EXACT,
                paid,
                end(),
                &mut || true
            ),
            Err(Refusal::Values)
        ));
    }
    assert!(evaluate(
        &g,
        &x,
        &map,
        "linear",
        &[0, 1, 2],
        &[1.; 3],
        CertifiedWeightError::EXACT,
        paid,
        end(),
        &mut || true
    )
    .is_err());
    let mut missing = map.clone();
    missing.remove("linear");
    assert!(evaluate(
        &g,
        &x,
        &missing,
        "relu",
        &[0, 1, 2],
        &[1.; 3],
        CertifiedWeightError::EXACT,
        paid,
        end(),
        &mut || true
    )
    .is_err());
    let bad = BoundedTensor::new(
        arr1(&[-1.; 3]).into_dyn(),
        arr1(&[ny_core::CROWN_COEFF_MAX; 3]).into_dyn(),
    )
    .unwrap();
    assert!(evaluate(
        &g,
        &bad,
        &map,
        "relu",
        &[0, 1, 2],
        &[1.; 3],
        CertifiedWeightError::EXACT,
        paid,
        end(),
        &mut || true
    )
    .is_err());
    let mut malformed = map.clone();
    malformed.insert(
        "linear".into(),
        BoundedTensor::new(arr1(&[-1.; 2]).into_dyn(), arr1(&[1.; 2]).into_dyn()).unwrap(),
    );
    assert!(evaluate(
        &g,
        &x,
        &malformed,
        "relu",
        &[0, 1, 2],
        &[1.; 3],
        CertifiedWeightError::EXACT,
        paid,
        end(),
        &mut || true
    )
    .is_err());
}

#[test]
fn local_l1_bn_layout_radius_shape_and_noncontiguous_parameters_refuse() {
    use crate::layers::normalization::BatchNormChannelAxisHint;
    let original =
        BatchNormLayer::from_scale_bias(arr1(&[1.; 3]).into_dyn(), arr1(&[0.; 3]).into_dyn())
            .unwrap();
    let (_, x, map) = fixture(
        Array2::eye(3),
        None,
        Some(original.clone()),
        vec![-1.; 3],
        vec![1.; 3],
    );
    for code in 0..5 {
        let mut b = original.clone();
        match code {
            0 => b.scale_err = arr1(&[0.; 2]).into_dyn(),
            1 => b.bias_err[ndarray::IxDyn(&[1])] = -f32::from_bits(1),
            2 => b.channel_axis_hint = Some(BatchNormChannelAxisHint::Fixed(1)),
            3 => b.scale[ndarray::IxDyn(&[1])] = f32::NAN,
            _ => b.scale = arr1(&[1.; 3]).slice_move(ndarray::s![..;-1]).into_dyn(),
        }
        let mut g = GraphNetwork::new();
        g.add_node(GraphNode::from_input(
            "linear",
            Layer::Linear(LinearLayer::new(Array2::eye(3), None).unwrap()),
        ));
        g.add_node(GraphNode::new(
            "bn",
            Layer::BatchNorm(b),
            vec!["linear".into()],
        ));
        g.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["bn".into()],
        ));
        g.set_output("relu");
        // The original map supplies only shape here. Each malformed source must
        // refuse before a result exists; no inference on this bad graph is run.
        assert!(evaluate(
            &g,
            &x,
            &map,
            "relu",
            &[0, 1, 2],
            &[1.; 3],
            CertifiedWeightError::EXACT,
            quote(3, 3).unwrap(),
            end(),
            &mut || true
        )
        .is_err());
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
#[allow(unsafe_code)]
fn local_l1_non_ieee_refuses_before_subnormal_source_work() {
    std::thread::spawn(|| {
        struct Restore(u32);
        impl Drop for Restore {
            fn drop(&mut self) {
                // SAFETY: same-thread restoration of the saved valid MXCSR.
                unsafe {
                    std::arch::asm!("ldmxcsr [{}]",in(reg)&self.0,options(nostack,preserves_flags));
                }
            }
        }
        let (g, x, map) = fixture(Array2::eye(3), None, None, vec![-1.; 3], vec![1.; 3]);
        let paid = quote(3, 3).unwrap();
        let mut old = 0u32;
        // SAFETY: STMXCSR writes four bytes into this live u32.
        unsafe {
            std::arch::asm!("stmxcsr [{}]",in(reg)&mut old,options(nostack,preserves_flags));
        }
        let restore = Restore(old);
        let bad = old | (1 << 6) | (1 << 15);
        // SAFETY: only DAZ/FTZ change; restoration occurs even during unwinding.
        unsafe {
            std::arch::asm!("ldmxcsr [{}]",in(reg)&bad,options(nostack,preserves_flags));
        }
        assert!(matches!(
            evaluate(
                &g,
                &x,
                &map,
                "relu",
                &[0, 1, 2],
                &[1.; 3],
                CertifiedWeightError::EXACT,
                paid,
                end(),
                &mut || true
            ),
            Err(Refusal::Environment)
        ));
        drop(restore);
    })
    .join()
    .unwrap();
}

// Independent exact dual enumeration: no recurrence and no reuse of the
// implementation's signed-product/addition helpers.
fn unary_subset_oracle(s: &LocalBoxL1Arithmetic<'_>, pre: &BoundedTensor) -> Q {
    let total = 2 * s.sign_count();
    let mask = total - 1;
    let m = s.active;
    let mut original = vec![Q::zero(); total];
    for r in 0..s.sign_count() {
        let (l, u) = s.sign_bounds(r).unwrap();
        original[2 * r] = q64(u);
        original[(2 * r) ^ mask] = -q64(l);
    }
    let unary = |sign: usize, i: usize| {
        let n = s.neurons[i];
        if sign & (1 << i) == 0 {
            q(pre.upper().as_slice().unwrap()[n])
        } else {
            -q(pre.lower().as_slice().unwrap()[n])
        }
    };
    for sign in 0..total {
        let marginal = (0..m).map(|i| q(s.weights[i]) * unary(sign, i)).sum::<Q>();
        original[sign] = original[sign].clone().min(marginal);
    }
    let mut maximum = Q::zero();
    for sign in 0..total {
        let best = (0..total)
            .map(|start| {
                original[start].clone()
                    + (0..m)
                        .filter(|i| (sign ^ start) & (1usize << *i) != 0)
                        .map(|i| q(2.) * q(s.weights[i]) * unary(sign, i))
                        .sum::<Q>()
            })
            .min()
            .unwrap();
        maximum = maximum.max(best);
    }
    maximum
}

#[test]
fn local_l1_unary_subset_transform_encloses_exact_asymmetric_dual_family() {
    for m in 3..=5 {
        let a = Array2::from_shape_fn((m, 2), |(i, j)| {
            ((3 * i + 5 * j) as i32 % 7 - 3) as f32 / 8.
        });
        let bias = Array1::from_shape_fn(m, |i| -(i as f32 + 1.) / 4.);
        let (g, x, map) = fixture(
            a.clone(),
            Some(bias.clone()),
            None,
            vec![-1.; 2],
            vec![1.; 2],
        );
        let indices: Vec<_> = (0..m).collect();
        let weights: Vec<_> = (0..m).map(|i| (i + 1) as f32 / 8.).collect();
        let s = run(&g, &x, &map, &indices, &weights);
        // A consistent smaller box for this arithmetic-only dual-family test.
        // It is not passed to evaluate as an authenticated original root map.
        let inner_l = [-0.25, -0.5];
        let inner_u = [0.75, 0.25];
        let mut l = Vec::new();
        let mut u = Vec::new();
        for i in 0..m {
            let mut lo = q(bias[i]);
            let mut hi = lo.clone();
            for j in 0..2 {
                let c = q(a[[i, j]]);
                let (low, high) = if c < Q::zero() {
                    (inner_u[j], inner_l[j])
                } else {
                    (inner_l[j], inner_u[j])
                };
                lo += &c * q(low);
                hi += c * q(high);
            }
            let lf = lo.to_f32().unwrap();
            let uf = hi.to_f32().unwrap();
            assert_eq!(q(lf), lo);
            assert_eq!(q(uf), hi);
            l.push(lf);
            u.push(uf);
        }
        let pre =
            BoundedTensor::new(Array1::from(l).into_dyn(), Array1::from(u).into_dyn()).unwrap();
        let expected = unary_subset_oracle(&s, &pre);
        let mut live = || true;
        let mut guard = Guard {
            deadline: end(),
            poll: &mut live,
            failed: None,
        };
        let got = refine_unary_bound(
            &s.signs,
            s.sign_count(),
            &indices,
            &weights,
            &pre,
            &mut guard,
        )
        .unwrap();
        assert!(expected <= q64(got.bound));
        assert!(q64(got.bound) - &expected < q(1. / 1048576.));
        // Original asymmetric endpoints remain unchanged by the helper.
        assert!((0..s.sign_count()).any(|r| {
            let (lo, hi) = s.sign_bounds(r).unwrap();
            lo != -hi
        }));
    }
}

fn strip_source(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut prefix = Array2::zeros((4, 4));
    let mut rows = Array2::zeros((7, 4));
    for k in 0..2 {
        prefix[[2 * k, 2 * k]] = 0.75;
        prefix[[2 * k, 2 * k + 1]] = 0.25;
        prefix[[2 * k + 1, 2 * k]] = -0.75;
        prefix[[2 * k + 1, 2 * k + 1]] = 0.25;
        rows[[3 * k, 2 * k]] = 1.;
        rows[[3 * k + 1, 2 * k + 1]] = 1.;
        rows[[3 * k + 2, 2 * k]] = 1.;
        rows[[3 * k + 2, 2 * k + 1]] = 1.;
        rows[[6, 2 * k]] = 2.;
        rows[[6, 2 * k + 1]] = 2.;
    }
    let mut b = Array1::zeros(7);
    b[6] = 9.;
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "prefix",
        Layer::Linear(LinearLayer::new(prefix.clone(), None).unwrap()),
    ));
    g.add_node(GraphNode::new(
        "linear",
        Layer::Linear(LinearLayer::new(rows.clone(), Some(b)).unwrap()),
        vec!["prefix".into()],
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".into()],
    ));
    g.add_node(GraphNode::new(
        "output",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 7), vec![-2., -2., -2., -2., -2., -2., 1.]).unwrap(),
                Some(arr1(&[-4.75 + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("output");
    let x = BoundedTensor::new(arr1(&[-1.; 4]).into_dyn(), arr1(&[1.; 4]).into_dyn()).unwrap();
    let mut map = g.collect_node_bounds(&x).unwrap();
    let old = &map["linear"];
    let mut lower = old.lower().clone();
    let mut upper = old.upper().clone();
    for r in 0..6 {
        // Exact backward composition from actual authored source matrices.
        // A symmetric input box makes the coefficient absolute sum exact.
        let radius = (0..4)
            .map(|j| {
                (0..4)
                    .map(|k| q(rows[[r, k]]) * q(prefix[[k, j]]))
                    .sum::<Q>()
                    .abs()
            })
            .sum::<Q>();
        let rf = radius.to_f32().unwrap();
        assert_eq!(q(rf), radius);
        lower[ndarray::IxDyn(&[r])] = -rf;
        upper[ndarray::IxDyn(&[r])] = rf;
    }
    // Keep the stable carrier's original coarse IBP enclosure; no extra proof.
    assert!(lower[ndarray::IxDyn(&[6])] > 0.);
    map.insert("linear".into(), BoundedTensor::new(lower, upper).unwrap());
    (g, x, map)
}

#[test]
fn local_l1_unary_genuine_six_row_source_strict_fold_and_unsafe() {
    for shift in [0., -0.5] {
        let (g, x, map) = strip_source(shift);
        let s = run(&g, &x, &map, &[0, 1, 2, 3, 4, 5], &[1.; 6]);
        assert_eq!(s.source.linear_parent, "prefix");
        assert!(std::ptr::eq(s.source.input, &raw const map["prefix"]));
        assert!(std::ptr::eq(s.source.pre_box, &raw const map["linear"]));
        let original = (0..s.sign_count())
            .map(|r| {
                let (l, u) = s.sign_bounds(r).unwrap();
                (-l).max(u)
            })
            .fold(0f64, f64::max);
        assert!((8. ..8.001).contains(&original));
        assert_eq!(s.original_local_bound().to_bits(), original.to_bits());
        for i in 0..6 {
            assert_eq!(
                s.pre_lower()[i].to_bits(),
                map["linear"].lower()[ndarray::IxDyn(&[i])].to_bits()
            );
            assert_eq!(
                s.pre_upper()[i].to_bits(),
                map["linear"].upper()[ndarray::IxDyn(&[i])].to_bits()
            );
        }
        assert!(s.pre_lower()[6..]
            .iter()
            .chain(s.pre_upper()[6..].iter())
            .all(|v| v.to_bits() == 0));
        assert!(s.bound() >= 4. && s.bound() < 4.001);
        assert!(f64::from(s.bound()) < original);
        let Layer::Linear(prefix) = g.node("prefix").unwrap().layer() else {
            unreachable!()
        };
        let Layer::Linear(linear) = g.node("linear").unwrap().layer() else {
            unreachable!()
        };
        let Layer::Linear(output) = g.node("output").unwrap().layer() else {
            unreachable!()
        };
        let mut parent_a = vec![Q::zero(); 4];
        let mut constant = q(output.bias.as_ref().unwrap()[0]);
        let lo = map["linear"].lower().as_slice().unwrap();
        let hi = map["linear"].upper().as_slice().unwrap();
        for r in 0..7 {
            let c = q(output.weight[[0, r]]);
            let (slope, intercept) = if lo[r] >= 0. {
                (q(1.), q(0.))
            } else {
                assert!(hi[r] > 0. && c < Q::zero());
                let slope = q(hi[r]) / (q(hi[r]) - q(lo[r]));
                let intercept = -q(lo[r]) * &slope;
                (slope, intercept)
            };
            constant += &c * (&slope * q(linear.bias.as_ref().unwrap()[r]) + intercept);
            for j in 0..4 {
                parent_a[j] += &c * &slope * q(linear.weight[[r, j]]);
            }
        }
        let root_a: Vec<Q> = (0..4)
            .map(|j| {
                (0..4)
                    .map(|k| &parent_a[k] * q(prefix.weight[[k, j]]))
                    .sum()
            })
            .collect();
        let triangle = constant - root_a.iter().map(|c| c.abs()).sum::<Q>();
        assert_eq!(triangle, q(-0.75 + shift));
        assert_eq!(q(4.25 + shift) - q(5.), triangle); // Unary sum alone.
        assert!(q(4.25 + shift) - q64(original) < triangle);
        let mut true_min = q(100.);
        for point in 0..16 {
            let input: Vec<Q> = (0..4)
                .map(|j| q(if point & (1 << j) == 0 { -1. } else { 1. }))
                .collect();
            let parent: Vec<Q> = (0..4)
                .map(|i| (0..4).map(|j| q(prefix.weight[[i, j]]) * &input[j]).sum())
                .collect();
            let z: Vec<Q> = (0..7)
                .map(|r| {
                    q(linear.bias.as_ref().unwrap()[r])
                        + (0..4)
                            .map(|j| q(linear.weight[[r, j]]) * &parent[j])
                            .sum::<Q>()
                })
                .collect();
            assert!(z[6] >= q(1.));
            let value = q(output.bias.as_ref().unwrap()[0])
                + (0..7)
                    .map(|i| q(output.weight[[0, i]]) * z[i].clone().max(Q::zero()))
                    .sum::<Q>();
            assert_eq!(
                value,
                q(4.25 + shift) - z[..6].iter().map(|v| v.abs()).sum::<Q>()
            );
            true_min = true_min.min(value);
        }
        // Negative ReLUs plus a stable affine row are concave; box corners
        // therefore determine the genuine original network's exact minimum.
        assert_eq!(true_min, q(0.25 + shift));
        let candidate = q(4.25 + shift) - q(s.bound());
        assert!(candidate <= true_min && candidate > triangle);
        if shift == 0. {
            assert!(candidate > q(0.24));
        } else {
            assert!(candidate <= q(-0.25));
        }
    }
}

#[test]
fn local_l1_unary_signed_stable_subnormal_and_rounding_controls() {
    ny_core::require_f64_interval_proof_environment().unwrap();
    let (g, x, map) = fixture(
        Array2::from_shape_vec((3, 1), vec![1., -1., 0.5]).unwrap(),
        Some(arr1(&[2., -2., 0.])),
        None,
        vec![-0.5],
        vec![0.75],
    );
    let s = run(&g, &x, &map, &[0, 1, 2], &[0.25, 0.75, 0.5]);
    let actual = [-0.5, 0.75]
        .into_iter()
        .map(|v| {
            q(0.25) * (q(v) + q(2.)).abs()
                + q(0.75) * (-q(v) - q(2.)).abs()
                + q(0.5) * (q(0.5) * q(v)).abs()
        })
        .max()
        .unwrap();
    assert!(actual <= q(s.bound()) && q(s.bound()) - actual < q(1. / 1024.));
    assert!(map["linear"].lower()[ndarray::IxDyn(&[0])] > 0.);
    assert!(map["linear"].upper()[ndarray::IxDyn(&[1])] < 0.);
    for tiny in [f32::from_bits(1), -f32::from_bits(1)] {
        assert_eq!(
            q64(unary_product(255. / 256., tiny).unwrap()),
            q(255. / 256.) * q(tiny)
        );
    }
    for (a, b) in [(1., 2f64.powi(-54)), (-1., -2f64.powi(-54)), (1., -1.)] {
        assert!(q64(signed_up_add(a, b).unwrap()) >= q64(a) + q64(b));
    }
    let sub = f64::from_bits(1);
    let rounded: Vec<_> = [-sub, 0., sub]
        .into_iter()
        .map(|a| signed_up_add(a, 1.).unwrap())
        .collect();
    assert!(rounded.windows(2).all(|p| p[0] <= p[1]));
    let edge = f32::from_bits(ny_core::CROWN_COEFF_MAX.to_bits() - 1);
    assert_eq!(unary_product(2., edge), Err(Refusal::Arithmetic));
    assert_eq!(
        signed_up_add(f32_to_f64_exact(edge), f32_to_f64_exact(edge)),
        Err(Refusal::Arithmetic)
    );
}

#[test]
fn local_l1_unary_uses_true_post_bn_box_and_keeps_source_identity() {
    let mut bn = BatchNormLayer::from_scale_bias(
        arr1(&[1.; 3]).into_dyn(),
        arr1(&[2., -3., 0.5]).into_dyn(),
    )
    .unwrap();
    bn.scale_err = arr1(&[0.125; 3]).into_dyn();
    bn.bias_err = arr1(&[0.25, 0.5, 0.125]).into_dyn();
    let (g, x, map) = fixture(Array2::eye(3), None, Some(bn), vec![0.; 3], vec![0.; 3]);
    let s = run(&g, &x, &map, &[0, 1, 2], &[1., 0.5, 0.25]);
    assert!(std::ptr::eq(s.source.pre_box, &raw const map["bn"]));
    for i in 0..3 {
        assert_eq!(
            s.pre_lower()[i].to_bits(),
            map["bn"].lower()[ndarray::IxDyn(&[i])].to_bits()
        );
        assert_eq!(
            s.pre_upper()[i].to_bits(),
            map["bn"].upper()[ndarray::IxDyn(&[i])].to_bits()
        );
    }
    assert!(!std::ptr::eq(s.source.pre_box, &raw const map["linear"]));
    assert!(s.bn_radius() > 0.);
    let exact = q(2.25) + q(0.5) * q(3.5) + q(0.25) * q(0.625);
    assert!(exact <= q(s.bound()) && q(s.bound()) - exact < q(1. / 1024.));
    assert!(s.matches_source(&g, &x, &map, "relu"));
    let other_map = map.clone();
    assert!(!s.matches_source(&g, &x, &other_map, "relu"));
    let other_input = x.clone();
    assert!(!s.matches_source(&g, &other_input, &map, "relu"));
    assert!(!s.matches_source(&g, &x, &map, "bn"));
    let mut missing = map.clone();
    missing.remove("bn");
    assert!(matches!(
        evaluate(
            &g,
            &x,
            &missing,
            "relu",
            &[0, 1, 2],
            &[1., 0.5, 0.25],
            CertifiedWeightError::EXACT,
            quote(3, 3).unwrap(),
            end(),
            &mut || true
        ),
        Err(Refusal::Source)
    ));
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep the original fixture owner intact while mutating an independent candidate."
)]
fn local_l1_unary_malformed_bounds_and_contradictions_publish_nothing() {
    let (g, x, map) = fixture(Array2::zeros((3, 1)), None, None, vec![0.], vec![0.]);
    for (l, u, expected) in [
        (vec![f32::NAN, 0., 0.], vec![1.; 3], Refusal::Values),
        (vec![0.; 3], vec![f32::INFINITY, 1., 1.], Refusal::Values),
        (vec![1.; 3], vec![0.; 3], Refusal::Values),
        (
            vec![0.; 3],
            vec![ny_core::CROWN_COEFF_MAX; 3],
            Refusal::Values,
        ),
        // Inconsistent with the original zero source: refuse, not empty proof.
        (vec![1.; 3], vec![1.; 3], Refusal::Arithmetic),
    ] {
        let mut bad = map.clone();
        bad.insert(
            "linear".into(),
            BoundedTensor::new_unchecked(Array1::from(l).into_dyn(), Array1::from(u).into_dyn())
                .unwrap(),
        );
        assert!(
            matches!(evaluate(&g, &x, &bad, "relu", &[0, 1, 2], &[1.; 3], CertifiedWeightError::EXACT, quote(1, 3).unwrap(), end(), &mut || true), Err(e) if e == expected)
        );
    }
    let mut bad = map.clone();
    bad.insert(
        "linear".into(),
        BoundedTensor::new(
            arr1(&[-1.; 3]).slice_move(ndarray::s![..;-1]).into_dyn(),
            arr1(&[1.; 3]).slice_move(ndarray::s![..;-1]).into_dyn(),
        )
        .unwrap(),
    );
    assert!(matches!(
        evaluate(
            &g,
            &x,
            &bad,
            "relu",
            &[0, 1, 2],
            &[1.; 3],
            CertifiedWeightError::EXACT,
            quote(1, 3).unwrap(),
            end(),
            &mut || true
        ),
        Err(Refusal::Shape)
    ));
}

#[test]
fn local_l1_unary_all_new_polls_storage_and_added_work_are_bounded() {
    let (g, x, map) = fixture(Array2::eye(8), None, None, vec![-1.; 8], vec![1.; 8]);
    let indices = [0, 1, 2, 3, 4, 5, 6, 7];
    let weights = [1.; 8];
    let s = run(&g, &x, &map, &indices, &weights);
    let mut calls = 0;
    let mut live = || {
        calls += 1;
        true
    };
    let mut guard = Guard {
        deadline: end(),
        poll: &mut live,
        failed: None,
    };
    refine_unary_bound(
        &s.signs,
        s.sign_count(),
        &indices,
        &weights,
        &map["linear"],
        &mut guard,
    )
    .unwrap();
    assert!(calls > 256 && calls <= 320);
    for stop in 1..=calls {
        let mut n = 0;
        let mut poll = || {
            n += 1;
            n != stop
        };
        let mut guard = Guard {
            deadline: end(),
            poll: &mut poll,
            failed: None,
        };
        assert_eq!(
            refine_unary_bound(
                &s.signs,
                s.sign_count(),
                &indices,
                &weights,
                &map["linear"],
                &mut guard
            ),
            Err(Refusal::Cancelled)
        );
        assert_eq!(guard.check(), Err(Refusal::Cancelled));
        assert_eq!(n, stop);
    }
    assert!(unary_scratch_bytes() <= UNARY_REFINE_BYTES);
    assert_eq!(UNARY_REFINE_BYTES, 4096);
    assert_eq!(SCRATCH_BYTES, 65536);
    let paid = quote(8, 8).unwrap();
    let old_price = WORK_PER_COLUMN * 9 + FIXED_WORK;
    assert_eq!(paid.work, old_price + 1_048_576 + UNARY_OBSERVATION_WORK);
    assert!(matches!(
        evaluate(
            &g,
            &x,
            &map,
            "relu",
            &indices,
            &weights,
            CertifiedWeightError::EXACT,
            LocalBoxL1Quote {
                work: old_price,
                ..paid
            },
            end(),
            &mut || true
        ),
        Err(Refusal::Budget)
    ));
    let mut calls = 0;
    let mut poll = || {
        calls += 1;
        true
    };
    let mut expired = Guard {
        deadline: Instant::now(),
        poll: &mut poll,
        failed: None,
    };
    assert_eq!(
        refine_unary_bound(
            &s.signs,
            s.sign_count(),
            &indices,
            &weights,
            &map["linear"],
            &mut expired
        ),
        Err(Refusal::Deadline)
    );
    assert_eq!(calls, 0);
}

#[test]
fn local_l1_unary_metadata_keeps_f64_original_without_diagnostic_refusal() {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "prefix",
        Layer::Linear(LinearLayer::new(Array2::zeros((3, 1)), None).unwrap()),
    ));
    g.add_node(GraphNode::new(
        "linear",
        Layer::Linear(LinearLayer::new(Array2::eye(3), None).unwrap()),
        vec!["prefix".into()],
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".into()],
    ));
    g.set_output("relu");
    let x = BoundedTensor::new(arr1(&[0.]).into_dyn(), arr1(&[0.]).into_dyn()).unwrap();
    let mut map = g.collect_node_bounds(&x).unwrap();
    let edge = f32::from_bits(ny_core::CROWN_COEFF_MAX.to_bits() - 1);
    // Original prefix and selected outputs are identically zero. This coarse
    // parent box and exact selected box are both independently valid, but the
    // original local-box maximum rounds to the f32 sentinel. Telemetry must
    // retain f64 and must not reject the successfully refined certificate.
    map.insert(
        "prefix".into(),
        BoundedTensor::new(
            arr1(&[-edge, -256., -128.]).into_dyn(),
            arr1(&[edge, 256., 128.]).into_dyn(),
        )
        .unwrap(),
    );
    map.insert(
        "linear".into(),
        BoundedTensor::new(arr1(&[0.; 3]).into_dyn(), arr1(&[0.; 3]).into_dyn()).unwrap(),
    );
    let s = run(&g, &x, &map, &[0, 1, 2], &[1.; 3]);
    assert!(s.original_local_bound() > f32_to_f64_exact(edge));
    assert!(ny_core::is_crown_coeff_safe_f64(s.original_local_bound()));
    assert_eq!(
        f64_to_f32_up(s.original_local_bound()),
        ny_core::CROWN_COEFF_MAX
    );
    assert!(s.bound() >= 0. && s.bound() <= f32::MIN_POSITIVE);
    assert!(s
        .pre_lower()
        .iter()
        .chain(s.pre_upper().iter())
        .all(|v| v.to_bits() == 0));
    let original = (0..s.sign_count())
        .map(|r| {
            let (l, u) = s.sign_bounds(r).unwrap();
            (-l).max(u)
        })
        .fold(0f64, f64::max);
    assert_eq!(original.to_bits(), s.original_local_bound().to_bits());
}

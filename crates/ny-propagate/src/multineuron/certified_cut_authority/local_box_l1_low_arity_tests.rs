// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;

#[test]
fn low_arity_exact_rows_and_raw_bn_parameter_corners_are_enclosed() {
    for m in 1..=2 {
        let a = Array2::from_shape_fn((m, 2), |(i, j)| [[0.5, -0.25], [-0.75, 0.125]][i][j]);
        let bias = Array1::from_vec(vec![0.25, -0.5][..m].to_vec());
        let scale = [-0.75, 1.25];
        let shift = [0.5, -0.125];
        let se = [0.125, 0.0625];
        let be = [0.0625, 0.125];
        let mut bn = BatchNormLayer::from_scale_bias(
            Array1::from_vec(scale[..m].to_vec()).into_dyn(),
            Array1::from_vec(shift[..m].to_vec()).into_dyn(),
        )
        .unwrap();
        bn.scale_err = Array1::from_vec(se[..m].to_vec()).into_dyn();
        bn.bias_err = Array1::from_vec(be[..m].to_vec()).into_dyn();
        let (g, x, mut map) = fixture(
            a.clone(),
            Some(bias.clone()),
            Some(bn),
            vec![-1., -0.5],
            vec![0.75, 1.],
        );
        let weights = vec![1., 0.5][..m].to_vec();
        let mut points = Vec::new();
        let mut lo = vec![f32::INFINITY; m];
        let mut hi = vec![f32::NEG_INFINITY; m];
        for xp in 0..4 {
            let xx = [
                if xp & 1 == 0 { -1. } else { 0.75 },
                if xp & 2 == 0 { -0.5 } else { 1. },
            ];
            for parameters in 0..1usize << (2 * m) {
                let mut zz = Vec::new();
                for i in 0..m {
                    let t = q(bias[i]) + q(a[[i, 0]]) * q(xx[0]) + q(a[[i, 1]]) * q(xx[1]);
                    let sc = q(scale[i])
                        + if parameters & (1 << (2 * i)) == 0 {
                            -q(se[i])
                        } else {
                            q(se[i])
                        };
                    let bc = q(shift[i])
                        + if parameters & (1 << (2 * i + 1)) == 0 {
                            -q(be[i])
                        } else {
                            q(be[i])
                        };
                    let z = sc * t + bc;
                    let v = z.to_f32().unwrap();
                    assert_eq!(q(v), z);
                    lo[i] = lo[i].min(v);
                    hi[i] = hi[i].max(v);
                    zz.push(z);
                }
                points.push(zz);
            }
        }
        // These exact dyadic endpoints enclose every affine/BN corner of the
        // complete source, including the declared parameter uncertainty.
        map.insert(
            "bn".into(),
            BoundedTensor::new(
                Array1::from_vec(lo).into_dyn(),
                Array1::from_vec(hi).into_dyn(),
            )
            .unwrap(),
        );
        let ids = (0..m).collect::<Vec<_>>();
        let s = run(&g, &x, &map, &ids, &weights);
        assert_eq!(s.sign_count(), 1 << (m - 1));
        assert!(s.bn_radius() > 0.);
        for zz in points {
            for r in 0..s.sign_count() {
                let value = (0..m)
                    .map(|i| sign(r, i) * q(weights[i]) * &zz[i])
                    .sum::<Q>();
                let (l, u) = s.sign_bounds(r).unwrap();
                assert!(q64(l) <= value && value <= q64(u));
                assert!(value.abs() <= q(s.bound()));
            }
        }
    }
}

#[test]
fn low_arity_singleton_unary_refinement_keeps_original_source_correlation() {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "parent",
        Layer::Linear(
            LinearLayer::new(Array2::from_shape_vec((2, 1), vec![1., 1.]).unwrap(), None).unwrap(),
        ),
    ));
    g.add_node(GraphNode::new(
        "linear",
        Layer::Linear(
            LinearLayer::new(Array2::from_shape_vec((1, 2), vec![1., -1.]).unwrap(), None).unwrap(),
        ),
        vec!["parent".into()],
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".into()],
    ));
    g.set_output("relu");
    let x = BoundedTensor::new(arr1(&[-1.]).into_dyn(), arr1(&[1.]).into_dyn()).unwrap();
    let mut map = g.collect_node_bounds(&x).unwrap();
    // z=x-x=0 on the original root even though its parent box is [-1,1]^2.
    map.insert(
        "linear".into(),
        BoundedTensor::new(arr1(&[0.]).into_dyn(), arr1(&[0.]).into_dyn()).unwrap(),
    );
    let s = run(&g, &x, &map, &[0], &[1.]);
    assert!(s.original_local_bound() >= 2.);
    // The certified result rounds a nonnegative zero support outward to the
    // smallest normal f32, retaining the kernel's existing underflow policy.
    assert!(s.bound() >= 0. && s.bound() <= f32::MIN_POSITIVE);
    assert_eq!(s.sign_count(), 1);
}

#[test]
fn low_arity_minimal_dimension_exact_budget_and_each_poll_refuse_atomically() {
    for m in 1..=2 {
        let (g, x, map) = fixture(Array2::eye(m), None, None, vec![-1.; m], vec![1.; m]);
        let ids = (0..m).collect::<Vec<_>>();
        let weights = vec![1.; m];
        let paid = quote(m, m).unwrap();
        let mut polls = 0;
        let s = evaluate(
            &g,
            &x,
            &map,
            "relu",
            &ids,
            &weights,
            CertifiedWeightError::EXACT,
            paid,
            end(),
            &mut || {
                polls += 1;
                true
            },
        )
        .unwrap();
        assert_eq!(s.sign_count(), 1 << (m - 1));
        assert!(s.bound() >= m as f32 && s.bound() < m as f32 + 0.001);
        for stop in 1..=polls {
            let mut n = 0;
            assert!(matches!(
                evaluate(
                    &g,
                    &x,
                    &map,
                    "relu",
                    &ids,
                    &weights,
                    CertifiedWeightError::EXACT,
                    paid,
                    end(),
                    &mut || {
                        n += 1;
                        n < stop
                    }
                ),
                Err(Refusal::Cancelled)
            ));
        }
        for short in [
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
                    &ids,
                    &weights,
                    CertifiedWeightError::EXACT,
                    short,
                    end(),
                    &mut || true
                ),
                Err(Refusal::Budget)
            ));
        }
        assert!(quote(m, 0).is_none());
    }
}

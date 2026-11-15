// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::onnx_proto::*;
use std::time::Duration;

fn value(name: &str, width: usize) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: [1, width]
                        .into_iter()
                        .map(|n| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(n as i64)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}
fn parameter(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: dims.into(),
        data_type: 1,
        raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    }
}
fn model() -> ModelProto {
    ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            input: vec![value("input", 1)],
            output: vec![value("output", 1)],
            initializer: vec![
                parameter("w0", &[4, 1], &[1., -1., 1., -1.]),
                parameter("b0", &[4], &[0., 0., -1., -1.]),
                parameter("w1", &[1, 4], &[2., -3., -2., 3.]),
                parameter("b1", &[1], &[0.]),
            ],
            node: vec![
                NodeProto {
                    input: vec!["input".into(), "w0".into(), "b0".into()],
                    output: vec!["hidden".into()],
                    op_type: "Gemm".into(),
                    attribute: vec![AttributeProto {
                        name: "transB".into(),
                        r#type: 2,
                        i: Some(1),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                NodeProto {
                    input: vec!["hidden".into()],
                    output: vec!["relu".into()],
                    op_type: "Relu".into(),
                    ..Default::default()
                },
                NodeProto {
                    input: vec!["relu".into(), "w1".into(), "b1".into()],
                    output: vec!["output".into()],
                    op_type: "Gemm".into(),
                    attribute: vec![AttributeProto {
                        name: "transB".into(),
                        r#type: 2,
                        i: Some(1),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn parsed(m: &ModelProto) -> Option<Chain> {
    extract(
        &m.encode_to_vec(),
        false,
        &mut Poll {
            end: Instant::now() + Duration::from_secs(5),
            callback: &mut || true,
        },
    )
}
fn fixture() -> PreparedFloat32Chain {
    PreparedFloat32Chain {
        chain: parsed(&model()).unwrap(),
        source_sha256: [0; 32],
        optimized_sha256: [0; 32],
    }
}
fn formula() -> NonLinearFormula {
    NonLinearFormula::parse("(vnnlib-version <2.0>) (declare-network f (declare-input X real [1,1]) (declare-output Y real [1,1])) (assert (and (>= X[0,0] -32) (<= X[0,0] 32) (> (* X[0,0] Y[0,0]) 0)))").unwrap()
}
fn oracle_dot(w: &[f64], x: &[f32], b: f64, reverse: bool, fma: bool, ftz: bool) -> f32 {
    let flush = |v: f32| {
        if ftz && v.is_subnormal() {
            0.0f32.copysign(v)
        } else {
            v
        }
    };
    let mut sum = flush(b as f32);
    let indices: Vec<usize> = if reverse {
        (0..x.len()).rev().collect()
    } else {
        (0..x.len()).collect()
    };
    for i in indices {
        let a = flush(w[i] as f32);
        let b = flush(x[i]);
        sum = flush(if fma {
            a.mul_add(b, flush(sum))
        } else {
            flush(sum) + flush(a * b)
        });
    }
    sum
}

#[test]
fn f32_chain_shared_scalar_keeps_exact_real_extrema_and_beats_independent_intervals() {
    let p = fixture();
    let end = Instant::now() + Duration::from_secs(5);
    let bounds = p.enclose(&[-32.], &[32.], end, &mut || true).unwrap();
    assert_eq!(bounds.shared_scalar_heads, 1);
    assert!(
        bounds.lower[0] < -3.
            && bounds.lower[0] > -3.01
            && bounds.upper[0] > 2.
            && bounds.upper[0] < 2.01
    );
    let [Stage::Gemm(a), Stage::Relu, Stage::Gemm(b)] = p.chain.stages.as_slice() else {
        panic!()
    };
    let mut poll = Poll {
        end,
        callback: &mut || true,
    };
    let first = affine(a, &[OpenIv::closed(-32., 32.)], &mut poll)
        .unwrap()
        .into_iter()
        .map(relu)
        .collect::<Vec<_>>();
    let independent = affine(b, &first, &mut poll).unwrap();
    assert!(independent[0].hi - independent[0].lo > 100.);
    // Independent exact rational evaluation, including both saturation tails.
    for i in -1024..=1024 {
        let t = BigRational::new(i.into(), 32.into());
        let z = a
            .weights
            .iter()
            .zip(&a.bias)
            .map(|(&w, &c)| {
                (BigRational::from_float(w).unwrap() * &t + BigRational::from_float(c).unwrap())
                    .max(BigRational::zero())
            })
            .collect::<Vec<_>>();
        let actual = b
            .weights
            .iter()
            .zip(z)
            .fold(BigRational::zero(), |s, (&w, z)| {
                s + BigRational::from_float(w).unwrap() * z
            });
        assert!(
            BigRational::from_float(bounds.lower[0]).unwrap() <= actual
                && actual <= BigRational::from_float(bounds.upper[0]).unwrap()
        );
    }
}

#[test]
fn f32_chain_bounds_contain_reduction_orders_fmas_subnormals_and_flushing() {
    let p = fixture();
    let end = Instant::now() + Duration::from_secs(5);
    let [Stage::Gemm(a), Stage::Relu, Stage::Gemm(b)] = p.chain.stages.as_slice() else {
        panic!()
    };
    for t in [
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        0.,
        -0.,
        -32.,
        -1.,
        0.25,
        0.5,
        1.,
        32.,
        16777216.,
    ] {
        let v = f64::from(t);
        let out = p.enclose(&[v], &[v], end, &mut || true).unwrap();
        for reverse in [false, true] {
            for fma in [false, true] {
                for ftz in [false, true] {
                    let z = (0..a.outputs)
                        .map(|i| {
                            oracle_dot(&a.weights[i..=i], &[t], a.bias[i], reverse, fma, ftz)
                                .max(0.)
                        })
                        .collect::<Vec<_>>();
                    let y = f64::from(oracle_dot(&b.weights, &z, b.bias[0], reverse, fma, ftz));
                    assert!(
                        out.lower[0] <= y && y <= out.upper[0],
                        "{t} {y} [{},{}]",
                        out.lower[0],
                        out.upper[0]
                    );
                }
            }
        }
    }
    let mut poll = Poll {
        end,
        callback: &mut || true,
    };
    let x = [
        OpenIv::closed(16777216., 16777216.),
        OpenIv::closed(1., 1.),
        OpenIv::closed(-16777216., -16777216.),
    ];
    let w = [1., 1., 1.];
    let ideal = dot(&x, &w, 0., &mut poll).unwrap();
    let e = rounding_error(&x, &w, 0., &mut poll).unwrap();
    assert!(ideal.lo <= 1. && ideal.hi >= 1. && e > 1.);
    assert_eq!(
        oracle_dot(&w, &[16777216., 1., -16777216.], 0., false, false, false),
        0.
    );
    assert!(expand(ideal, e).unwrap().lo <= 0.);
    // A flushing multiply can lose much more than one minimum normal when
    // its input is subnormal; the |w|*eta term must cover that loss.
    let x = [OpenIv::closed(
        f64::from(f32::from_bits(0x007f_ffff)),
        f64::from(f32::from_bits(0x007f_ffff)),
    )];
    let e = rounding_error(&x, &[1024.], 0., &mut poll).unwrap();
    assert!(e >= 1024. * x[0].lo);
}

#[test]
fn f32_chain_multiple_asymmetric_outputs_and_nonunit_breakpoints_are_enclosed() {
    let a = Gemm {
        inputs: 1,
        outputs: 3,
        weights: vec![0.5, -2., 4.],
        bias: vec![-0.25, 1., -3.],
    };
    let b = Gemm {
        inputs: 3,
        outputs: 2,
        weights: vec![2., -0.5, 1., -3., 2., -0.25],
        bias: vec![0.25, -0.5],
    };
    let end = Instant::now() + Duration::from_secs(5);
    let mut poll = Poll {
        end,
        callback: &mut || true,
    };
    let bound = shared_head(&a, &b, OpenIv::closed(-2., 3.), &mut poll).unwrap();
    for i in -512..=768 {
        let t = i as f64 / 256.;
        let z = a
            .weights
            .iter()
            .zip(&a.bias)
            .map(|(&w, &c)| (w * t + c).max(0.))
            .collect::<Vec<_>>();
        for j in 0..2 {
            let y = b.bias[j]
                + b.weights[j * 3..j * 3 + 3]
                    .iter()
                    .zip(&z)
                    .map(|(w, z)| w * z)
                    .sum::<f64>();
            assert!(bound[j].lo <= y && y <= bound[j].hi);
        }
    }
    assert!(shared_head(
        &Gemm {
            inputs: 2,
            ..a.clone()
        },
        &b,
        OpenIv::closed(0., 1.),
        &mut poll
    )
    .is_none());
}

#[test]
fn f32_chain_original_wire_attributes_payloads_and_complete_wiring_fail_closed() {
    let mut variants = Vec::new();
    let mut m = model();
    m.opset_import.push(OperatorSetIdProto {
        domain: "ai.onnx".into(),
        version: 6,
    });
    variants.push(m);
    let mut m = model();
    m.graph.as_mut().unwrap().node[0].domain = "other".into();
    variants.push(m);
    let mut m = model();
    m.graph.as_mut().unwrap().node[0].input[0] = "missing".into();
    variants.push(m);
    let mut m = model();
    m.graph.as_mut().unwrap().node[2].output[0] = "wrong".into();
    variants.push(m);
    let mut m = model();
    m.graph.as_mut().unwrap().node[1].op_type = "Sigmoid".into();
    variants.push(m);
    let mut m = model();
    let g = m.graph.as_mut().unwrap();
    g.initializer.push(g.initializer[0].clone());
    variants.push(m);
    let mut m = model();
    m.graph
        .as_mut()
        .unwrap()
        .initializer
        .push(parameter("unused", &[1], &[1.]));
    variants.push(m);
    let mut m = model();
    m.graph.as_mut().unwrap().initializer[0].raw_data[0..4]
        .copy_from_slice(&f32::from_bits(1).to_le_bytes());
    variants.push(m);
    let mut m = model();
    m.graph.as_mut().unwrap().initializer[0].float_data = vec![1.; 4];
    variants.push(m);
    let mut m = model();
    m.graph.as_mut().unwrap().initializer[0]
        .external_data
        .push(StringStringEntryProto {
            key: "location".into(),
            value: "unread".into(),
        });
    variants.push(m);
    let mut m = model();
    m.graph.as_mut().unwrap().node[0]
        .attribute
        .push(AttributeProto {
            name: "transA".into(),
            r#type: 2,
            i: Some(1),
            ..Default::default()
        });
    variants.push(m);
    let mut m = model();
    let n = &mut m.graph.as_mut().unwrap().node[0];
    n.attribute.push(n.attribute[0].clone());
    variants.push(m);
    let mut m = model();
    m.graph.as_mut().unwrap().input[0]
        .r#type
        .as_mut()
        .unwrap()
        .tensor_type
        .as_mut()
        .unwrap()
        .elem_type = 11;
    variants.push(m);
    for (i, m) in variants.iter().enumerate() {
        assert!(parsed(m).is_none(), "variant {i}");
    }
    let mut bytes = model().encode_to_vec();
    // Unknown ModelProto field 25 (local functions), absent from ny's proto.
    bytes.extend_from_slice(&[0xca, 0x01, 0x00]);
    let end = Instant::now() + Duration::from_secs(5);
    assert!(extract(
        &bytes,
        false,
        &mut Poll {
            end,
            callback: &mut || true
        }
    )
    .is_none());
    let mut bytes = model().encode_to_vec();
    bytes.extend_from_slice(&[8, 9]);
    assert!(extract(
        &bytes,
        false,
        &mut Poll {
            end,
            callback: &mut || true
        }
    )
    .is_none());
}

#[test]
fn f32_chain_relu_covers_flushed_subnormals_at_graph_input_and_output() {
    let mut m = model();
    let g = m.graph.as_mut().unwrap();
    g.initializer.clear();
    g.node = vec![NodeProto {
        input: vec!["input".into()],
        output: vec!["output".into()],
        op_type: "Relu".into(),
        ..Default::default()
    }];
    let p = PreparedFloat32Chain {
        chain: parsed(&m).unwrap(),
        source_sha256: [0; 32],
        optimized_sha256: [0; 32],
    };
    let lo = f64::from(f32::from_bits(1));
    let hi = f64::from(f32::from_bits(0x007f_ffff));
    let b = p
        .enclose(
            &[lo],
            &[hi],
            Instant::now() + Duration::from_secs(5),
            &mut || true,
        )
        .unwrap();
    assert_eq!(b.lower, [0.]);
    assert!(b.upper[0] >= hi && b.shared_scalar_heads == 0);
    assert_eq!(relu(OpenIv::closed(-hi, -lo)), OpenIv::closed(0., 0.));
}

#[test]
fn f32_chain_optimizer_may_fuse_relu_but_cannot_change_one_parameter_bit() {
    let original = parsed(&model()).unwrap();
    let mut m = model();
    let g = m.graph.as_mut().unwrap();
    g.node[0].op_type = "FusedGemm".into();
    g.node[0].domain = "com.microsoft".into();
    g.node[0].output = vec!["relu".into()];
    g.node[0].attribute.push(AttributeProto {
        name: "activation".into(),
        r#type: 3,
        s: Some(b"Relu".to_vec()),
        ..Default::default()
    });
    g.node.remove(1);
    m.opset_import.push(OperatorSetIdProto {
        domain: "com.microsoft".into(),
        version: 1,
    });
    let end = Instant::now() + Duration::from_secs(5);
    assert!(parsed(&m).is_none());
    assert_eq!(
        extract(
            &m.encode_to_vec(),
            true,
            &mut Poll {
                end,
                callback: &mut || true
            }
        )
        .unwrap(),
        original
    );
    m.graph.as_mut().unwrap().initializer[0].raw_data[0..4]
        .copy_from_slice(&1.0f32.next_up().to_le_bytes());
    assert_ne!(
        extract(
            &m.encode_to_vec(),
            true,
            &mut Poll {
                end,
                callback: &mut || true
            }
        )
        .unwrap(),
        original
    );
    let mut m = model();
    m.graph.as_mut().unwrap().initializer[1].raw_data[0..4]
        .copy_from_slice(&(-0.0f32).to_le_bytes());
    assert_ne!(parsed(&m).unwrap(), original);
}

#[test]
fn f32_chain_every_extraction_and_enclosure_poll_and_invalid_domain_refuse() {
    let raw = model().encode_to_vec();
    let run = |fail: usize| {
        let mut n = 0;
        let end = Instant::now() + Duration::from_secs(5);
        let c = extract(
            &raw,
            false,
            &mut Poll {
                end,
                callback: &mut || {
                    n += 1;
                    n != fail
                },
            },
        );
        (c, n)
    };
    let (c, polls) = run(usize::MAX);
    assert!(c.is_some() && polls > 50);
    for fail in 1..=polls {
        assert!(run(fail).0.is_none(), "source poll {fail}");
    }
    let p = fixture();
    let run = |fail: usize| {
        let mut n = 0;
        let r = p.enclose(
            &[-32.],
            &[32.],
            Instant::now() + Duration::from_secs(5),
            &mut || {
                n += 1;
                n != fail
            },
        );
        (r, n)
    };
    let (r, polls) = run(usize::MAX);
    assert!(r.is_some() && polls > 30);
    for fail in 1..=polls {
        assert!(run(fail).0.is_none(), "enclosure poll {fail}");
    }
    for (lo, hi) in [
        (f64::NAN, 1.),
        (0., f64::INFINITY),
        (1., -1.),
        (-f64::MAX, f64::MAX),
        (f64::from(f32::MAX), f64::from(f32::MAX)),
    ] {
        assert!(p
            .enclose(
                &[lo],
                &[hi],
                Instant::now() + Duration::from_secs(5),
                &mut || true
            )
            .is_none());
    }
    assert!(p
        .enclose(
            &[],
            &[],
            Instant::now() + Duration::from_secs(5),
            &mut || true
        )
        .is_none());
    assert!(p
        .enclose(&[0.], &[1.], Instant::now(), &mut || true)
        .is_none());
}

#[cfg(feature = "ort")]
#[test]
fn f32_chain_actual_cpu_optimized_export_is_source_checked_before_enclosure() {
    let raw = model().encode_to_vec();
    let end = Instant::now() + Duration::from_secs(10);
    let f = formula();
    assert!(f.prepare_f32_predicate(end, &mut || true).is_some());
    let original = parsed(&model()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("check.onnx");
    let _s = crate::runtime::session_builder()
        .unwrap()
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .with_inter_threads(1)
        .unwrap()
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)
        .unwrap()
        .with_optimized_model_path(&path)
        .unwrap()
        .commit_from_memory(&raw)
        .unwrap();
    let optimized = std::fs::read(&path).unwrap();
    assert!(
        audit(
            &optimized,
            Schema::Model,
            &mut Poll {
                end,
                callback: &mut || true
            },
            &mut 0
        )
        .is_some(),
        "optimized wire audit failed: {:?}",
        ModelProto::decode(optimized.as_slice()).unwrap()
    );
    let replay = extract(
        &optimized,
        true,
        &mut Poll {
            end,
            callback: &mut || true,
        },
    )
    .unwrap_or_else(|| {
        panic!(
            "optimized extraction failed: {:?}",
            ModelProto::decode(optimized.as_slice()).unwrap()
        )
    });
    assert_eq!(original, replay);
    let p = PreparedFloat32Chain::prepare(&raw, &f, end, &mut || true)
        .expect("actual CPU export must preserve the supported Gemm/Relu chain");
    assert_eq!(p.source_sha256(), <[u8; 32]>::from(Sha256::digest(&raw)));
    assert_ne!(p.optimized_sha256(), [0; 32]);
    assert_eq!(
        p.enclose(&[-32.], &[32.], end, &mut || true)
            .unwrap()
            .shared_scalar_heads,
        1
    );
    let wrong=NonLinearFormula::parse("(vnnlib-version <2.0>) (declare-network f (declare-input X real [1]) (declare-output Y real [1])) (assert (> X[0] 0))").unwrap();
    assert!(PreparedFloat32Chain::prepare(&raw, &wrong, end, &mut || true).is_none());
    assert!(PreparedFloat32Chain::prepare(&raw, &f, Instant::now(), &mut || true).is_none());
    assert!(PreparedFloat32Chain::prepare(&raw, &f, end, &mut || false).is_none());
}

fn property(lo: &str, hi: &str, body: &str) -> NonLinearFormula {
    NonLinearFormula::parse(&format!("(vnnlib-version <2.0>) (declare-network f (declare-input X real [1,1]) (declare-output Y real [1,1])) (assert (and (>= X[0,0] {lo}) (<= X[0,0] {hi}))) (assert {body})")).unwrap()
}

#[test]
fn f32_chain_full_owner_checks_both_profiles_and_the_whole_original_formula() {
    let p = fixture();
    let end = Instant::now() + Duration::from_secs(5);
    let f = property("-32", "32", "(< Y[0,0] -4)");
    let mut a = Float32ChainAttempt::default();
    assert!(p.refute_formula(&f, end, &mut || true, &mut a).is_some());
    assert!(
        a.completed
            && a.visited == 1
            && a.dual_refuted == 1
            && a.split == 0
            && a.shared_scalar_heads == 1
    );
    let f = property("-1", "1", "(and (> Y[0,0] 0) (< Y[0,0] 0))");
    assert!(p.refute_formula(&f, end, &mut || true, &mut a).is_some());
    assert!(a.completed && a.real_open_refuted > 0 && a.f32_pieces > 1);
    let f=NonLinearFormula::parse("(vnnlib-version <2.0>) (declare-network f (declare-input X real [1]) (declare-output Y real [1])) (assert (and (>= X[0] -1) (<= X[0] 1))) (assert (< Y[0] -4))").unwrap();
    assert!(p.refute_formula(&f, end, &mut || true, &mut a).is_none());
    assert!(!a.completed && a.visited == 0);
}

#[test]
fn f32_chain_real_only_underflow_refutation_cannot_publish_a_complete_cover() {
    let f = property(
        "0",
        "1.1754943508222875e-38",
        "(and (> X[0,0] 0) (> Y[0,0] 0) (<= (* X[0,0] Y[0,0]) 0))",
    );
    let tiny = f64::from(f32::from_bits(1));
    assert_eq!(f.holds_at(&[tiny], &[2. * tiny]), Some(false));
    let p = fixture();
    let mut a = Float32ChainAttempt::default();
    let mut polls = 0;
    // The fixture computes 2*t near zero. Its actual Float32 product is zero,
    // so the original typed predicate has a concrete positive assignment.
    assert_eq!((f32::from_bits(1) * (2. * f32::from_bits(1))).to_bits(), 0);
    assert!(p
        .refute_formula(
            &f,
            Instant::now() + Duration::from_secs(5),
            &mut || {
                polls += 1;
                polls < 2000
            },
            &mut a
        )
        .is_none());
    assert!(!a.completed && a.visited > 0 && a.real_open_refuted > 0);
}

#[test]
fn f32_chain_full_owner_every_poll_and_late_or_cancelled_cover_refuse() {
    let f = property("-32", "32", "(< Y[0,0] -4)");
    let p = fixture();
    let run = |fail: usize| {
        let mut a = Float32ChainAttempt::default();
        let mut polls = 0;
        let r = p.refute_formula(
            &f,
            Instant::now() + Duration::from_secs(5),
            &mut || {
                polls += 1;
                polls != fail
            },
            &mut a,
        );
        (r, a, polls)
    };
    let (r, a, polls) = run(usize::MAX);
    assert!(r.is_some() && a.completed && polls > 50);
    for fail in 1..=polls {
        let (r, a, _) = run(fail);
        assert!(r.is_none() && !a.completed, "owner poll {fail}");
    }
    let mut a = Float32ChainAttempt::default();
    assert!(p
        .refute_formula(&f, Instant::now(), &mut || true, &mut a)
        .is_none());
    assert!(!a.completed);
}

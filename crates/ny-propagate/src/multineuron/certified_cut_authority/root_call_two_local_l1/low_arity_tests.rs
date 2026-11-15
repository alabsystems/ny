// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;

pub(crate) fn fixture(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut w = Array2::<f32>::zeros((6, 6));
    for (r, c, v) in [
        (0, 0, 0.5),
        (0, 1, 0.5),
        (1, 1, 0.5),
        (1, 2, 0.5),
        (2, 2, 0.5),
        (2, 0, -0.5),
        (3, 3, 0.5),
        (3, 4, 0.5),
        (4, 3, 0.5),
        (4, 4, -0.5),
        (5, 5, 1.),
    ] {
        w[[r, c]] = v;
    }
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 6), vec![-1024., -1024., -1024., -1., -1., -1. / 1024.])
                    .unwrap(),
                Some(arr1(&[2049.25 + 1. / 1024. + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.; 6]).into_dyn(), arr1(&[1.; 6]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}
fn proposal<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    old: (f32, f32),
    end: Instant,
) -> RawTwoLocalL1RootCutQuote<'a> {
    quote_raw_two_local_l1_root_cut(
        g,
        x,
        b,
        None,
        &[1.],
        old,
        "relu",
        [2, 3],
        [[3, 4, 0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0, 0, 0]],
        [
            [1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 0., 0., 0., 0., 0.],
        ],
        [0.5, 512.],
        limits(),
        end,
        &mut || true,
    )
    .unwrap()
}

#[test]
fn low_arity_original_three_plus_two_keeps_tiny_unselected_term_and_unsafe_truth() {
    let mut minimum = Q::from_integer(10000.into());
    let mut support = [0i32; 2];
    for mask in 0usize..64 {
        let v: [i32; 6] = std::array::from_fn(|i| if mask & (1 << i) == 0 { -1 } else { 1 });
        let first = [v[0] + v[1], v[1] + v[2], v[2] - v[0]];
        let pair = [v[3] + v[4], v[3] - v[4]];
        support[0] = support[0].max(first.iter().map(|z| z.abs()).sum());
        support[1] = support[1].max(pair.iter().map(|z| z.abs()).sum());
        let value = rat(2049.25 + 1. / 1024.)
            - rat(512.) * Q::from_integer(first.iter().map(|z| (*z).max(0)).sum::<i32>().into())
            - rat(0.5) * Q::from_integer(pair.iter().map(|z| (*z).max(0)).sum::<i32>().into())
            - rat(1. / 1024.) * Q::from_integer(v[5].max(0).into());
        minimum = minimum.min(value);
    }
    assert_eq!(support, [4, 2]);
    assert_eq!(minimum, rat(0.25));
    // A negative sum of ReLUs is concave: the complete original source minimum
    // has a cube-vertex representative. The tiny sixth term is not cropped.
    for shift in [0., -0.5] {
        let (g, x, b) = fixture(shift);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        assert!(old.0 <= -512.25 + shift && old.0 > -512.3 + shift);
        for boxed in [false, true] {
            let mut q = proposal(&g, &x, &b, old, end);
            if boxed {
                assert!(q.try_enable_head_box_floor(0., limits()));
            }
            assert!(q.try_enable_full_head_advice(limits()));
            let engine = BundleBackend::default();
            let out = evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).unwrap();
            assert_eq!(out.active_lens(), [2, 3]);
            assert_eq!(out.certificates(), 2);
            assert!(out.bounds().0 > 0.24 + shift && out.bounds().0 <= 0.25 + shift);
            assert!(rat(out.bounds().0) <= &minimum + rat(shift));
            assert_eq!(out.bounds().1.to_bits(), old.1.to_bits());
            assert_eq!(out.single3_candidate_lower(), None);
            assert_eq!(
                engine.bundle_calls.load(Ordering::SeqCst),
                usize::from(!boxed || shift < 0.)
            );
        }
    }
}

#[test]
fn low_arity_singleton_and_two_singletons_accept_minimal_complete_source() {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(
            LinearLayer::new(Array2::from_shape_vec((2, 1), vec![1., -1.]).unwrap(), None).unwrap(),
        ),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, 2), vec![-1., -1.]).unwrap(),
                Some(arr1(&[1.25])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1.]).into_dyn(), arr1(&[1.]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let q = quote_raw_local_l1_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        "relu",
        1,
        [0; 8],
        [1., 0., 0., 0., 0., 0., 0., 0.],
        0.5,
        limits(),
        end,
        &mut || true,
    )
    .unwrap();
    let single = SingleBackend::default();
    let result = evaluate_raw_local_l1_root_cut(q, &single, &mut || true).unwrap();
    assert!(result.bounds().0 > 0.24 && result.bounds().0 <= 0.25);
    assert_eq!(single.local_calls.load(Ordering::SeqCst), 1);
    let mut q = quote_raw_two_local_l1_root_cut(
        &g,
        &x,
        &b,
        None,
        &[1.],
        old,
        "relu",
        [1, 1],
        [[0; 8], [1, 0, 0, 0, 0, 0, 0, 0]],
        [[1., 0., 0., 0., 0., 0., 0., 0.]; 2],
        [0.5; 2],
        limits(),
        end,
        &mut || true,
    )
    .unwrap();
    // The original upper endpoint makes this deliberately unproven threshold
    // exceed the known minimum; both box and required prefix must complete.
    assert!(q.try_enable_head_box_floor(1., limits()));
    assert!(q.try_enable_full_head_advice(limits()));
    let engine = BundleBackend::default();
    let out = evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).unwrap();
    assert!(out.bounds().0 > 0.24 && out.bounds().0 <= 0.25);
    assert_eq!(out.bounds().1.to_bits(), old.1.to_bits());
    assert_eq!(out.active_lens(), [1, 1]);
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 1);
    assert!(out.head_box_candidate_lower().is_some());
}

#[test]
fn low_arity_required_prefix_failure_and_every_poll_publish_no_partial_bound() {
    let (g, x, b) = fixture(-0.5);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let make = || {
        let mut q = proposal(&g, &x, &b, old, end);
        assert!(q.try_enable_head_box_floor(0., limits()));
        assert!(q.try_enable_full_head_advice(limits()));
        q
    };
    let fail = BundleBackend::default();
    fail.refuse_bundle.store(1, Ordering::SeqCst);
    assert!(evaluate_raw_two_local_l1_root_cut(make(), &fail, &mut || true).is_none());
    assert_eq!(fail.bundle_calls.load(Ordering::SeqCst), 1);
    let mut polls = 0;
    assert!(
        evaluate_raw_two_local_l1_root_cut(make(), &BundleBackend::default(), &mut || {
            polls += 1;
            true
        })
        .is_some()
    );
    for stop in 1..=polls {
        let mut n = 0;
        assert!(
            evaluate_raw_two_local_l1_root_cut(make(), &BundleBackend::default(), &mut || {
                n += 1;
                n < stop
            })
            .is_none(),
            "stop={stop}/{polls}"
        );
    }
}

#[test]
fn low_arity_fresh_source_scale_seals_and_generation_remain_required() {
    let (g, x, b) = fixture(-0.5);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let mut q = proposal(&g, &x, &b, old, end);
    assert!(q.try_enable_full_head_advice(limits()));
    let other = proposal(&g, &x, &b, old, end);
    let full = extract_charged_raw_bn_chain(
        &g,
        &x,
        g.output_name(),
        &b,
        &b,
        None,
        q.base.extraction,
        end,
        &mut || true,
    )
    .unwrap();
    let seed = exact_seed(1, 1, vec![1.]);
    let beta: Vec<Vec<f32>> = full.node_abs.iter().map(|v| vec![0.; v.len()]).collect();
    let alpha = GraphAlphaState::new();
    let engine = BundleBackend::default();
    let base = ResidentCutCallContext::new(
        ResidentCutSnapshotGenerations::initial(),
        &g,
        &x,
        &alpha,
        &b,
        Some(&engine),
        &seed,
        "relu",
        [q.neurons[0][0], q.neurons[0][1]],
        &full.segments,
        &full.relu_names,
        &beta,
        &full.frontier_abs,
        &full.node_abs,
        x.lower().as_slice().unwrap(),
        x.upper().as_slice().unwrap(),
        end,
    );
    let context = TwoLocalContext::new(&q, base).unwrap();
    let make = |group: usize| {
        local_box_l1::evaluate(
            &g,
            &x,
            &b,
            "relu",
            &q.neurons[group][..q.active[group]],
            &q.weights[group][..q.active[group]],
            ny_core::CertifiedWeightError::EXACT,
            q.kernel[group],
            end,
            &mut || true,
        )
        .unwrap()
    };
    let mut fresh = FreshTwoLocalSupport {
        seal: &q.seal,
        arithmetic: [make(0), make(1)],
    };
    let mut yes = || true;
    let mut guard = Poll {
        end,
        callback: &mut yes,
        alive: true,
    };
    fresh.seal = &other.seal;
    assert!(context.choose_scale(&fresh, &mut guard).is_none());
    fresh.seal = &q.seal;
    fresh.arithmetic.swap(0, 1);
    assert!(context.choose_scale(&fresh, &mut guard).is_none());
    fresh.arithmetic.swap(0, 1);
    let mut choice = context.choose_scale(&fresh, &mut guard).unwrap();
    assert!(context.build(&fresh).is_err());
    choice.seal = &other.seal;
    assert!(context.build_with_scale(&fresh, Some(&choice)).is_err());
    choice.seal = &q.seal;
    let carrier = context.build_with_scale(&fresh, Some(&choice)).unwrap();
    let result = context.run(&carrier, &engine).unwrap();
    assert!(result.lower > -0.26 && result.lower <= -0.25);
    assert!(context.run(&carrier, &engine).is_err());
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 1);
}

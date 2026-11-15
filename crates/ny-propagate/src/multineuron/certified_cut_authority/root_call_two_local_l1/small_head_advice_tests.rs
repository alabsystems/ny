// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Two multiplier dimensions do not require a single3 comparison prefix.
use super::*;

pub(crate) fn fixture(shift: f32) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut w = Array2::<f32>::zeros((7, 5));
    w[[0, 0]] = 0.5;
    w[[0, 1]] = 0.5;
    w[[1, 0]] = 0.5;
    w[[1, 1]] = -0.5;
    w[[2, 0]] = 0.5;
    w[[2, 1]] = 0.5;
    w[[3, 2]] = 0.5;
    w[[3, 3]] = 0.5;
    w[[4, 3]] = 0.5;
    w[[4, 4]] = 0.5;
    w[[5, 4]] = 0.5;
    w[[5, 2]] = -0.5;
    w[[6, 0]] = 960.;
    w[[6, 1]] = 320.;
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(
            LinearLayer::new(w, Some(arr1(&[0.5, 0.5, 0.5, 0., 0., 0., 2048.]))).unwrap(),
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
                Array2::from_shape_vec((1, 7), vec![-1024., -1024., -1024., -1., -1., -1., 1.])
                    .unwrap(),
                Some(arr1(&[450. + shift])),
            )
            .unwrap(),
        ),
        vec!["relu".into()],
    ));
    g.set_output("out");
    let input = BoundedTensor::new(arr1(&[-1.; 5]).into_dyn(), arr1(&[1.; 5]).into_dyn()).unwrap();
    let bounds = g.collect_node_bounds(&input).unwrap();
    (g, input, bounds)
}
fn small_quote<'a>(
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
        [3, 3],
        [[0, 1, 2, 0, 0, 0, 0, 0], [3, 4, 5, 0, 0, 0, 0, 0]],
        [[1., 1., 1., 0., 0., 0., 0., 0.]; 2],
        [512., 0.5],
        limits(),
        end,
        &mut || true,
    )
    .unwrap()
}

#[test]
fn small_head_advice_original_source_interior_gain_unsafe_and_all_plan_orders() {
    let mut minimum2 = i32::MAX;
    let mut supports2 = [0; 2];
    for mask in 0usize..32 {
        let a: [i32; 5] = std::array::from_fn(|j| if mask & (1 << j) == 0 { -1 } else { 1 });
        let first = [a[0] + a[1] + 1, a[0] - a[1] + 1, a[0] + a[1] + 1];
        let second = [a[2] + a[3], a[3] + a[4], a[4] - a[2]];
        supports2[0] = supports2[0].max(first.iter().map(|v| v.abs()).sum());
        supports2[1] = supports2[1].max(second.iter().map(|v| v.abs()).sum());
        assert!(2048 + 960 * a[0] + 320 * a[1] >= 768);
        let value2 = 4996 + 1920 * a[0] + 640 * a[1]
            - 1024 * first.iter().map(|v| (*v).max(0)).sum::<i32>()
            - second.iter().map(|v| (*v).max(0)).sum::<i32>();
        minimum2 = minimum2.min(value2);
    }
    // The stable positive unit is affine on the whole box. The remaining
    // objective is concave, so the integer corner census proves the minimum.
    assert_eq!((minimum2, supports2), (384, [7, 4]));
    for shift in [0., -256.] {
        let (g, x, b) = fixture(shift);
        let end = Instant::now() + Duration::from_secs(60);
        let old = baseline(&g, &x, &b, end);
        assert!(
            old.0 <= -64.5 + shift && old.0 > -64.51 + shift,
            "baseline={old:?}, shift={shift}"
        );
        let mut q = small_quote(&g, &x, &b, old, end);
        assert!(q.try_enable_head_box_floor(0., limits()));
        let endpoint =
            evaluate_raw_two_local_l1_root_cut(q, &BundleBackend::default(), &mut || true).unwrap();
        assert_eq!(endpoint.bounds().0.to_bits(), old.0.to_bits());
        for bound in [
            endpoint.head_box_candidate_lower().unwrap(),
            endpoint.bundle_candidate_lower().unwrap(),
        ] {
            assert!(bound <= -320. + shift && bound > -320.1 + shift);
        }
        for order in 0..3 {
            let mut q = small_quote(&g, &x, &b, old, end);
            if order == 1 {
                assert!(q.try_enable_head_box_floor(0., limits()));
            }
            assert!(q.try_enable_full_head_advice(limits()));
            if order == 2 {
                assert!(q.try_enable_head_box_floor(0., limits()));
            }
            let engine = BundleBackend::default();
            // This backend has no single-local API. Advice has no authority to
            // introduce an undeclared comparison call or an ordinary fold.
            assert!(!engine.provides_bounded_local_l1_lower_cut());
            let out = evaluate_raw_two_local_l1_root_cut(q, &engine, &mut || true).unwrap();
            assert!(out.bounds().0 > 63.9 + shift && out.bounds().0 <= 64. + shift);
            assert!(out.bounds().0 <= minimum2 as f32 / 2. + shift);
            assert_eq!(out.bounds().1.to_bits(), old.1.to_bits());
            let advice = out.multiplier_advice().unwrap();
            assert_eq!(advice.original, [512., 0.5]);
            assert!(advice.chosen[0] > 255. && advice.chosen[0] < 257.);
            assert_eq!(advice.chosen[1], 0.5);
            assert!(advice.chosen_score > advice.original_score + 383.9);
            assert_eq!(out.single3_candidate_lower(), None);
            assert_eq!(
                engine.bundle_calls.load(Ordering::SeqCst),
                usize::from(order == 0 || shift < 0.)
            );
        }
    }
}

#[test]
fn small_head_advice_quote_addition_is_order_independent_atomic_and_generic() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let mut after_box = small_quote(&g, &x, &b, old, end);
    assert!(after_box.try_enable_head_box_floor(0., limits()));
    let before = (
        after_box.work(),
        after_box.scratch_bytes(),
        after_box.candidate_count(),
    );
    let extra = full_head_advice::quote(5, 7, 1).unwrap();
    let price = before.0 + extra.work;
    let scratch = before.1 + extra.scratch_bytes;
    for (work, heap, allowed) in [
        (price - 1, scratch, false),
        (price, scratch - 1, false),
        (price, scratch, true),
    ] {
        let mut q = small_quote(&g, &x, &b, old, end);
        assert!(q.try_enable_head_box_floor(0., limits()));
        assert_eq!(
            q.try_enable_full_head_advice(RawRootCutLimits {
                max_work: work,
                max_scratch_bytes: heap,
                ..limits()
            }),
            allowed
        );
        assert_eq!(
            (q.work(), q.scratch_bytes(), q.candidate_count()),
            if allowed {
                (price, scratch, before.2)
            } else {
                before
            }
        );
    }
    assert!(after_box.try_enable_full_head_advice(limits()));
    let mut before_box = small_quote(&g, &x, &b, old, end);
    assert!(before_box.try_enable_full_head_advice(limits()));
    assert!(before_box.try_enable_head_box_floor(0., limits()));
    assert_eq!(
        (
            after_box.work(),
            after_box.scratch_bytes(),
            after_box.candidate_count()
        ),
        (
            before_box.work(),
            before_box.scratch_bytes(),
            before_box.candidate_count()
        )
    );
    assert!(!after_box.try_enable_full_head_advice(limits()));
    assert!(full_head_advice::quote(1, 1, 1).is_none());
    let (g, x, b) = super::fixture(0.);
    let old = baseline(&g, &x, &b, end);
    let mut q = quote(&g, &x, &b, old, end);
    assert_eq!(q.active, [8, 8]);
    assert!(q.try_enable_full_head_advice(limits()));
    let out =
        evaluate_raw_two_local_l1_root_cut(q, &BundleBackend::default(), &mut || true).unwrap();
    assert!(out.bounds().0 > 0.24 && out.bounds().0 <= 0.25);
    assert_eq!(out.single3_candidate_lower(), None);
}

#[test]
fn small_head_advice_required_prefix_and_each_poll_refuse_without_partial_result() {
    let (g, x, b) = fixture(-256.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let make = || {
        let mut q = small_quote(&g, &x, &b, old, end);
        assert!(q.try_enable_head_box_floor(0., limits()));
        assert!(q.try_enable_full_head_advice(limits()));
        q
    };
    let refused = BundleBackend::default();
    refused.refuse_bundle.store(1, Ordering::SeqCst);
    assert!(evaluate_raw_two_local_l1_root_cut(make(), &refused, &mut || true).is_none());
    assert_eq!(refused.bundle_calls.load(Ordering::SeqCst), 1);
    let mut polls = 0;
    assert!(
        evaluate_raw_two_local_l1_root_cut(make(), &BundleBackend::default(), &mut || {
            polls += 1;
            true
        })
        .is_some()
    );
    for stop in 1..=polls {
        let mut seen = 0;
        assert!(
            evaluate_raw_two_local_l1_root_cut(make(), &BundleBackend::default(), &mut || {
                seen += 1;
                seen < stop
            })
            .is_none(),
            "stop={stop}/{polls}"
        );
    }
}

#[test]
fn small_head_advice_fresh_source_custody_and_advisory_score_never_publish_a_bound() {
    let (g, x, b) = fixture(0.);
    let end = Instant::now() + Duration::from_secs(60);
    let old = baseline(&g, &x, &b, end);
    let mut q = small_quote(&g, &x, &b, old, end);
    assert!(q.try_enable_full_head_advice(limits()));
    let other = small_quote(&g, &x, &b, old, end);
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
        [0, 1],
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
    let mut choice = context.choose_scale(&fresh, &mut guard).unwrap();
    assert!(context.build(&fresh).is_err());
    choice.seal = &other.seal;
    assert!(context.build_with_scale(&fresh, Some(&choice)).is_err());
    choice.seal = &q.seal;
    // A large finite scheduling score cannot become proof. The fresh carrier
    // and genuine descriptor fold still determine the returned scalar.
    choice.decision.chosen_score = 1e20;
    let carrier = context.build_with_scale(&fresh, Some(&choice)).unwrap();
    let result = context.run(&carrier, &engine).unwrap();
    assert!(result.lower > 63.9 && result.lower <= 64.);
    assert!(context.run(&carrier, &engine).is_err());
    assert_eq!(engine.bundle_calls.load(Ordering::SeqCst), 1);
}

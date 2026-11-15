// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::test_support::ExactToyBackend;
use super::*;
use crate::layers::{FlattenLayer, LinearLayer, ReLULayer, ReshapeLayer};
use crate::network::GraphNode;
use ndarray::{arr1, arr2};
use num_rational::BigRational as Q;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

fn exact(x: f32) -> Q {
    Q::from_float(f64::from(x)).unwrap()
}
fn limits() -> RawRootCutLimits {
    RawRootCutLimits {
        max_nodes: 16,
        max_scratch_bytes: 16 << 20,
        max_work: 100_000_000,
    }
}
fn fixture(shape_only: bool) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1., 0.], [1., 0.25], [3., 0.5]]),
                Some(arr1(&[0.5, -0.25, 4.])),
            )
            .unwrap(),
        ),
    ));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    let parent = if shape_only {
        g.add_node(GraphNode::new(
            "shape",
            Layer::Reshape(ReshapeLayer::new(vec![1, 3])),
            vec!["relu".into()],
        ));
        g.add_node(GraphNode::new(
            "flat",
            Layer::Flatten(FlattenLayer::new(1)),
            vec!["shape".into()],
        ));
        "flat"
    } else {
        "relu"
    };
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[-2., -4., 1.]]), Some(arr1(&[-0.4375]))).unwrap()),
        vec![parent.into()],
    ));
    g.set_output("out");
    let x = BoundedTensor::new(arr1(&[-1., -1.]).into_dyn(), arr1(&[1., 1.]).into_dyn()).unwrap();
    let b = g.collect_node_bounds(&x).unwrap();
    (g, x, b)
}
fn quote<'a>(
    g: &'a GraphNetwork,
    x: &'a BoundedTensor,
    b: &'a HashMap<String, BoundedTensor>,
    pair: [usize; 2],
    cap: RawRootCutLimits,
    end: Instant,
) -> RawRootCutQuote<'a> {
    quote_raw_root_cut(
        g,
        x,
        b,
        None,
        &[1.],
        (-0.14, 4.),
        "relu",
        pair,
        1.,
        cap,
        end,
        &mut || true,
    )
    .unwrap()
}

#[test]
fn weighted_fresh_chain_closes_where_uniform_half_and_full_remain_negative() {
    // Conv/shape-style flat indexing is exercised through genuine shape layers,
    // not by pretending the final Linear is directly adjacent to this ReLU.
    let (g, x, b) = fixture(true);
    let end = Instant::now() + Duration::from_secs(30);
    // Independent all-domain proof: negative of convex |z1|+2|z2| is concave,
    // so its minimum on this square is at a vertex. These four exact values
    // establish minF=1/16; no sampled nonlinear-interior oracle is needed.
    let mut maximum = exact(0.);
    for u in [-1., 1.] {
        for v in [-1., 1.] {
            let z1 = exact(u) + exact(0.5);
            let z2 = exact(u) + exact(v) / exact(4.) - exact(0.25);
            let abs1 = if z1 < exact(0.) { -z1 } else { z1 };
            let abs2 = if z2 < exact(0.) { -z2 } else { z2 };
            maximum = maximum.max(abs1 + exact(2.) * abs2);
        }
    }
    assert_eq!(exact(3.5625) - maximum, exact(0.0625));
    // The supplied old -0.14 is safely below the independently derived exact
    // triangle bound -11/80. Uniform half/full yield -3/16,-7/16 (or weaker
    // after outward f32 chord storage), hence cannot improve that old bound.
    assert!(exact(-0.14) < Q::new((-11).into(), 80.into()));
    let uniform = evaluate_raw_root_cut_inner(
        quote(&g, &x, &b, [0, 1], limits(), end),
        &ExactToyBackend::default(),
        &mut || true,
        AdviceMode::Uniform,
    )
    .unwrap();
    assert_eq!(uniform.bounds(), (-0.14, 4.));
    assert_eq!(uniform.cut_calls(), 2);
    for (pair, normal) in [([0, 1], [-0.5, -1., 1., 2.]), ([1, 0], [-1., -0.5, 2., 1.])] {
        let engine = ExactToyBackend::default();
        let result =
            evaluate_raw_root_cut(quote(&g, &x, &b, pair, limits(), end), &engine, &mut || {
                true
            })
            .unwrap();
        assert_eq!(result.cut_normal(), normal);
        assert_eq!(result.lambda_scale(), 2.);
        assert_eq!(result.bounds(), (0.0625, 4.));
        assert_eq!((result.support_calls(), result.cut_calls()), (1, 2));
        assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn weighted_q8_normal_is_exact_bounded_and_refuses_invalid_advice() {
    for k in [1u16, 127, 128, 129, 255, 256] {
        let got = q8_advice([f64::from(k), 256.]).unwrap();
        let w = Q::new(k.into(), 256.into());
        assert_eq!(exact(got.normal[0]), -w.clone());
        assert_eq!(exact(got.normal[2]), exact(2.) * w);
        assert_eq!(
            (got.normal[1], got.normal[3], got.lambda_scale),
            (-1., 2., 128.)
        );
    }
    assert_eq!(
        q8_advice([127.5, 256.]).unwrap().normal,
        [-0.5, -1., 1., 2.]
    );
    assert_eq!(q8_advice([4., 4.]).unwrap(), CutAdvice::uniform(2.));
    for bad in [
        [0., 1.],
        [-1., 1.],
        [f64::NAN, 1.],
        [1., f64::INFINITY],
        [1., 1024.],
        [1e-300, 1e-300],
        [1e30, 1e30],
    ] {
        assert!(q8_advice(bad).is_none(), "invalid advice {bad:?}");
    }
    assert!(4 * size_of::<CutAdvice>() + 2 * size_of::<CompletedRawRootCut>() <= ADVICE_BYTES);
}

#[test]
fn weighted_real_root_consumer_preserves_upper_and_single_query_attempt() {
    use crate::beta_crown::certified_root_cut::{try_tighten, QueryOwner};
    use crate::beta_crown::config::VerificationArtifactAuthority;
    let (g, x, nodes) = fixture(true);
    let end = Instant::now() + Duration::from_secs(30);
    let owner = QueryOwner::new(Some(end), true).unwrap();
    let defaults = crate::BetaCrownConfig::default();
    let config = crate::BetaCrownConfig {
        certified_root_cut_owner: Some(Arc::clone(&owner)),
        certified_root_cut_raw_source: true,
        verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
        ..defaults
    };
    let engine = ExactToyBackend::default();
    let mut bounds = [(-0.14, 4.)];
    assert_eq!(
        try_tighten(
            &config,
            &g,
            &x,
            &[vec![1.]],
            &[0.],
            &mut bounds,
            &nodes,
            None,
            Some(&engine),
            Some(end)
        ),
        Some(0)
    );
    assert_eq!(bounds, [(0.0625, 4.)]);
    let mut fresh = [(-0.14, 4.)];
    assert!(try_tighten(
        &config,
        &g,
        &x,
        &[vec![1.]],
        &[0.],
        &mut fresh,
        &nodes,
        None,
        Some(&engine),
        Some(end)
    )
    .is_none());
    assert_eq!(fresh, [(-0.14, 4.)]);
    assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn weighted_every_poll_and_late_second_fold_refuse_atomic_publication() {
    let (g, x, b) = fixture(true);
    let end = Instant::now() + Duration::from_secs(30);
    let make = || quote(&g, &x, &b, [0, 1], limits(), end);
    let mut count = 0;
    evaluate_raw_root_cut(make(), &ExactToyBackend::default(), &mut || {
        count += 1;
        true
    })
    .unwrap();
    for fail_at in 1..=count {
        let mut n = 0;
        let got = evaluate_raw_root_cut(make(), &ExactToyBackend::default(), &mut || {
            n += 1;
            n != fail_at
        });
        assert!(got.is_none(), "weighted transient poll {fail_at}/{count}");
        assert_eq!(n, fail_at);
    }
    let engine = ExactToyBackend::default();
    engine.refuse_cut.store(2, Ordering::SeqCst);
    assert!(evaluate_raw_root_cut(make(), &engine, &mut || true).is_none());
    assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn weighted_quote_pays_scan_on_fallback_and_exact_work_byte_boundaries() {
    let (g, x, b) = fixture(false);
    let end = Instant::now() + Duration::from_secs(30);
    let two = quote(&g, &x, &b, [0, 1], limits(), end);
    let expected = phase_work(two.prefix_quote, 4, 1).unwrap()
        + 16_000_000
        + phase_work(two.full_quote, 1, 2).unwrap()
        + ADVICE_FIXED_WORK
        + ADVICE_WORK_PER_OUTPUT
        + EXECUTION_OBSERVATION_WORK;
    assert_eq!(two.work(), expected);
    let exact_caps = RawRootCutLimits {
        max_work: two.work(),
        max_scratch_bytes: two.scratch_bytes(),
        ..limits()
    };
    assert_eq!(
        quote(&g, &x, &b, [0, 1], exact_caps, end).candidate_count(),
        2
    );
    let one = quote(
        &g,
        &x,
        &b,
        [0, 1],
        RawRootCutLimits {
            max_work: two.work() - 1,
            ..exact_caps
        },
        end,
    );
    assert_eq!(one.candidate_count(), 1);
    let one_work = one.work();
    let paid = quote(
        &g,
        &x,
        &b,
        [0, 1],
        RawRootCutLimits {
            max_work: one_work,
            ..exact_caps
        },
        end,
    );
    assert_eq!(
        evaluate_raw_root_cut(paid, &ExactToyBackend::default(), &mut || true)
            .unwrap()
            .bounds(),
        (0.0625, 4.)
    );
    for caps in [
        RawRootCutLimits {
            max_work: one_work - 1,
            ..exact_caps
        },
        RawRootCutLimits {
            max_scratch_bytes: two.scratch_bytes() - 1,
            ..exact_caps
        },
    ] {
        assert!(quote_raw_root_cut(
            &g,
            &x,
            &b,
            None,
            &[1.],
            (-0.14, 4.),
            "relu",
            [0, 1],
            1.,
            caps,
            end,
            &mut || true
        )
        .is_none());
    }
    // A reversed objective has no negative incoming pair. Its actual quote
    // still pays every advice unit; only the advice drops to the old family.
    let reversed = quote_raw_root_cut(
        &g,
        &x,
        &b,
        None,
        &[-1.],
        (-4., 4.),
        "relu",
        [0, 1],
        1.,
        exact_caps,
        end,
        &mut || true,
    )
    .unwrap();
    assert_eq!(reversed.work(), two.work());
    let mut poll = || true;
    let mut guard = Poll {
        end,
        callback: &mut poll,
        alive: true,
    };
    assert_eq!(
        choose_advice(&reversed, &mut guard),
        Some(CutAdvice::uniform(1.))
    );
}

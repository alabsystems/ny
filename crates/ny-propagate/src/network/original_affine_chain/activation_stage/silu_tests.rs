// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Authored local exact-real SiLU chains, without a deployed model claim.

use crate::layers::SiLULayer;

use super::*;

fn exp_positive(x: &Q) -> R {
    assert!(*x >= Q::zero() && *x <= integer(2));
    // All terms are nonnegative. After term64, successive remainder ratios
    // are <=x/66<1, so a geometric series bounds the entire positive tail.
    // This oracle uses no production transcendental or host exp operation.
    let mut term = integer(1);
    let mut sum = term.clone();
    for n in 1..=64 {
        term = term * x / integer(n);
        sum += &term;
    }
    let next = term * x / integer(65);
    let tail = next / (integer(1) - x / integer(66));
    R {
        lo: sum.clone(),
        hi: sum + tail,
    }
}

fn exp_point(x: &Q) -> R {
    if *x < Q::zero() {
        exp_positive(&-x).reciprocal()
    } else {
        exp_positive(x)
    }
}

fn dyadic_hull(x: &R) -> R {
    // Exact outward rational projection controls oracle integer growth when
    // an independently bounded GELU image feeds the following SiLU stage.
    let scale = BigInt::from(1) << 80usize;
    let bound = |v: &Q, upper: bool| {
        let scaled = v * Q::from_integer(scale.clone());
        let mut n = scaled.numer() / scaled.denom();
        if &n * scaled.denom() != *scaled.numer() {
            if upper && scaled > Q::zero() {
                n += 1;
            } else if !upper && scaled < Q::zero() {
                n -= 1;
            }
        }
        Q::new(n, scale.clone())
    };
    let result = R {
        lo: bound(&x.lo, false),
        hi: bound(&x.hi, true),
    };
    assert!(result.lo <= x.lo && result.hi >= x.hi);
    result
}

fn silu(x: &R) -> R {
    if x.magnitude() < fraction(1, 1 << 30) {
        // s(0)=1/2 and 0<s'(x)<=1/4 give
        // |x*s(x)-x/2|<=x²/4, including original binary32 subnormals.
        // Keep their exact dyadic values rather than coarsening them to the
        // fixed oracle grid or growing a tiny-argument exponential series.
        let radius = x.magnitude() * x.magnitude() / integer(4);
        let half = x.mul(&R::point(fraction(1, 2)));
        return R {
            lo: half.lo - &radius,
            hi: half.hi + radius,
        };
    }
    let x = dyadic_hull(x);
    assert!(x.lo >= integer(-2) && x.hi <= integer(2));
    let exponential = R {
        lo: exp_point(&-&x.hi).lo,
        hi: exp_point(&-&x.lo).hi,
    };
    x.mul(&R::point(integer(1)).add(&exponential).reciprocal())
}

fn authored_nodes(layers: &[Layer]) -> Vec<OriginalChainNode<'_>> {
    layers
        .iter()
        .enumerate()
        .map(|(index, layer)| OriginalChainNode {
            ids: OriginalAffineIds {
                lowering: Digest::of_bytes(b"authored-local-silu", b"unary-exact-real/v1"),
                node: Digest::of_bytes(b"authored-local-node", &index.to_le_bytes()),
                input: Digest::of_bytes(b"authored-local-edge", &index.to_le_bytes()),
                output: Digest::of_bytes(b"authored-local-edge", &(index + 1).to_le_bytes()),
            },
            layer,
        })
        .collect()
}

fn silu_layers(ledger: &WorkBudget) -> Vec<Layer> {
    vec![
        owner(
            array![[1.0, 0.0], [-0.25, 0.5]],
            array![0.0, -0.125],
            ledger,
        ),
        Layer::SiLU(SiLULayer::new()),
        owner(
            array![[1.5, -0.5], [0.1, 0.3], [-0.25, 1.0]],
            array![-0.125, 0.25, 0.125],
            ledger,
        ),
    ]
}

fn silu_input() -> BoundedTensor {
    BoundedTensor::new(
        array![[-1.5, -0.75], [-1.5, -0.5]].into_dyn(),
        array![[0.75, 0.5], [0.5, 0.75]].into_dyn(),
    )
    .unwrap()
}

fn contains_output(bounds: &BoundedTensor, values: &[R]) {
    for (i, value) in values.iter().enumerate() {
        assert!(q(bounds.lower().as_slice().unwrap()[i]) <= value.lo);
        assert!(q(bounds.upper().as_slice().unwrap()[i]) >= value.hi);
    }
}

#[test]
fn original_affine_chain_silu_signed_and_sparse_contain_independent_rational_chain() {
    let ledger = budget();
    let layers = silu_layers(&ledger);
    let nodes = authored_nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    let input = silu_input();
    let facts = context.forward(&input, options(&ledger)).unwrap();
    let demand = incoming(6);
    let signed = context
        .signed_linear(&facts, &demand, options(&ledger))
        .unwrap();
    let sparse = context
        .sparse_margins(
            &facts,
            &[
                OriginalLogitDifference {
                    target: 0,
                    competitor: 1,
                    position: 0,
                },
                OriginalLogitDifference {
                    target: 2,
                    competitor: 0,
                    position: 1,
                },
            ],
            options(&ledger),
        )
        .unwrap();
    assert_eq!(facts.num_nodes(), 3);
    assert_eq!(facts.bounds().shape(), &[2, 3]);
    assert_eq!(signed.bounds().lower_a().dim(), (2, 4));
    assert_eq!(signed.scope(), context.scope());
    let Layer::PointLinear(up) = &layers[0] else {
        unreachable!()
    };
    let Layer::PointLinear(down) = &layers[2] else {
        unreachable!()
    };
    // Sixteen corners, the midpoint, and an interior source whose first
    // preactivation is near SiLU's minimum. These are observations in ONE
    // native case, not a universal sampling proof or eighteen test cases.
    for mask in 0..18usize {
        let points: Vec<_> = (0..4)
            .map(|c| {
                let l = q(input.lower().as_slice().unwrap()[c]);
                let u = q(input.upper().as_slice().unwrap()[c]);
                R::point(if mask == 17 {
                    if c % 2 == 0 {
                        q(-1.28125)
                    } else {
                        Q::zero()
                    }
                } else if mask == 16 {
                    (l + u) / integer(2)
                } else if mask & (1 << c) == 0 {
                    l
                } else {
                    u
                })
            })
            .collect();
        let pre = affine(up, &points, 2);
        let activated: Vec<_> = pre.iter().map(silu).collect();
        let output = affine(down, &activated, 2);
        contains_output(facts.bounds(), &output);
        for row in 0..2 {
            assert!(
                row_value(signed.bounds(), row, &points, true).lo
                    <= row_value(&demand, row, &output, true).lo
            );
            assert!(
                row_value(signed.bounds(), row, &points, false).hi
                    >= row_value(&demand, row, &output, false).hi
            );
        }
        for (row, value) in [output[0].sub(&output[1]), output[5].sub(&output[3])]
            .iter()
            .enumerate()
        {
            assert!(row_value(sparse.bounds(), row, &points, true).lo <= value.lo);
            assert!(row_value(sparse.bounds(), row, &points, false).hi >= value.hi);
        }
    }
}

#[test]
fn original_affine_chain_mixed_gelu_silu_reports_keep_both_profiles_and_rational_images() {
    let ledger = budget();
    let layers = vec![
        owner(array![[1.0]], array![0.0], &ledger),
        Layer::GELU(GELULayer::new(GeluApproximation::Erf)),
        owner(array![[0.5]], array![0.125], &ledger),
        Layer::SiLU(SiLULayer::new()),
        owner(array![[1.0], [-0.5]], array![0.0, 0.25], &ledger),
    ];
    let nodes = authored_nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    assert_eq!(
        context.scope().assumptions,
        Digest::of_bytes(
            b"original-affine-chain-profile",
            ORIGINAL_CHAIN_GELU_SILU_EXACT_REAL_PROFILE.as_bytes()
        )
    );
    let input = BoundedTensor::new(array![[-0.5]].into_dyn(), array![[0.5]].into_dyn()).unwrap();
    let facts = context.forward(&input, options(&ledger)).unwrap();
    let mut reversed_nodes = nodes.clone();
    reversed_nodes[1].layer = &layers[3];
    reversed_nodes[3].layer = &layers[1];
    let reversed = PreparedOriginalAffineChain::prepare(&reversed_nodes, options(&ledger)).unwrap();
    assert_eq!(reversed.scope().model, context.scope().model);
    assert_eq!(reversed.scope().assumptions, context.scope().assumptions);
    assert_ne!(reversed.scope().coordinates, context.scope().coordinates);
    let before = ledger.receipt().unwrap();
    assert!(reversed
        .signed_linear(&facts, &incoming(2), options(&ledger))
        .is_err());
    assert_eq!(ledger.receipt().unwrap(), before);
    let demand = incoming(2);
    let report = context
        .signed_linear(&facts, &demand, options(&ledger))
        .unwrap();
    let sparse = context
        .sparse_margins(
            &facts,
            &[OriginalLogitDifference {
                target: 0,
                competitor: 1,
                position: 0,
            }],
            options(&ledger),
        )
        .unwrap();
    let constant = normal_constant();
    for v in [-0.5, 0.0, 0.5] {
        let points = vec![R::point(q(v))];
        let activated = gelu(q(v), &constant)
            .mul(&R::point(fraction(1, 2)))
            .add(&R::point(fraction(1, 8)));
        let activated = silu(&activated);
        let output = vec![
            activated.clone(),
            activated
                .mul(&R::point(fraction(-1, 2)))
                .add(&R::point(fraction(1, 4))),
        ];
        contains_output(facts.bounds(), &output);
        for row in 0..2 {
            assert!(
                row_value(report.bounds(), row, &points, true).lo
                    <= row_value(&demand, row, &output, true).lo
            );
            assert!(
                row_value(report.bounds(), row, &points, false).hi
                    >= row_value(&demand, row, &output, false).hi
            );
        }
        let difference = output[0].sub(&output[1]);
        assert!(row_value(sparse.bounds(), 0, &points, true).lo <= difference.lo);
        assert!(row_value(sparse.bounds(), 0, &points, false).hi >= difference.hi);
    }
}

#[test]
fn original_affine_chain_silu_scope_owner_order_and_generic_quarantine_are_independent() {
    let ledger = budget();
    let layers = silu_layers(&ledger);
    let nodes = authored_nodes(&layers);
    let silu = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    assert_eq!(
        silu.scope().assumptions,
        Digest::of_bytes(
            b"original-affine-chain-profile",
            ORIGINAL_CHAIN_SILU_EXACT_REAL_PROFILE.as_bytes()
        )
    );
    let gelu = Layer::GELU(GELULayer::new(GeluApproximation::Erf));
    let relu = Layer::ReLU(ReLULayer::new());
    for (activation, profile) in [
        (&gelu, ORIGINAL_CHAIN_GELU_EXACT_REAL_PROFILE),
        (&relu, ORIGINAL_CHAIN_EXACT_REAL_PROFILE),
    ] {
        let mut alternate = nodes.clone();
        alternate[1].layer = activation;
        let other = PreparedOriginalAffineChain::prepare(&alternate, options(&ledger)).unwrap();
        assert_eq!(other.scope().model, silu.scope().model);
        assert_ne!(other.scope().coordinates, silu.scope().coordinates);
        assert_eq!(
            other.scope().assumptions,
            Digest::of_bytes(b"original-affine-chain-profile", profile.as_bytes())
        );
    }
    assert_eq!(
        ORIGINAL_CHAIN_EXACT_REAL_PROFILE,
        "exact-real-original-binary32-affine/relu-outward-error-carrier/v1"
    );
    assert_eq!(
        ORIGINAL_CHAIN_GELU_EXACT_REAL_PROFILE,
        "exact-real-original-binary32-affine/relu/gelu-erf-outward-error-carrier/v1"
    );
    let facts = silu.forward(&silu_input(), options(&ledger)).unwrap();
    let same_bytes = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    assert_eq!(same_bytes.scope(), silu.scope());
    let before = ledger.receipt().unwrap();
    assert!(same_bytes
        .signed_linear(&facts, &incoming(6), options(&ledger))
        .is_err());
    assert_eq!(ledger.receipt().unwrap(), before);
    assert!(silu
        .signed_linear(&facts, &incoming(6), options(&ledger))
        .is_ok());
    let distinct_layers = silu_layers(&ledger);
    let distinct_nodes = authored_nodes(&distinct_layers);
    let distinct = PreparedOriginalAffineChain::prepare(&distinct_nodes, options(&ledger)).unwrap();
    assert_eq!(distinct.scope(), silu.scope());
    let before = ledger.receipt().unwrap();
    assert!(distinct
        .signed_linear(&facts, &incoming(6), options(&ledger))
        .is_err());
    assert_eq!(ledger.receipt().unwrap(), before);
    let other_ledger = budget();
    let before = other_ledger.receipt().unwrap();
    assert!(silu
        .signed_linear(&facts, &incoming(6), options(&other_ledger))
        .is_err());
    assert_eq!(other_ledger.receipt().unwrap(), before);
    let tanh = Layer::GELU(GELULayer::new(GeluApproximation::Tanh));
    let mut changed = nodes.clone();
    changed[1].layer = &tanh;
    assert!(PreparedOriginalAffineChain::prepare(&changed, options(&ledger)).is_err());
    changed = nodes.clone();
    changed[2].ids.input = changed[0].ids.input;
    assert!(PreparedOriginalAffineChain::prepare(&changed, options(&ledger)).is_err());
    changed = nodes.clone();
    changed.swap(0, 2);
    assert!(PreparedOriginalAffineChain::prepare(&changed, options(&ledger)).is_err());
    assert!(PreparedOriginalAffineChain::prepare(&nodes[..2], options(&ledger)).is_err());
    let mut changed = nodes.clone();
    changed[2].layer = &layers[1];
    assert!(PreparedOriginalAffineChain::prepare(&changed, options(&ledger)).is_err());
    let mut network = Network::new();
    for layer in &layers {
        network.add_layer(layer.clone());
    }
    assert!(network.propagate_crown(&silu_input()).is_err());
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("authored-up", layers[0].clone()));
    graph.add_node(GraphNode::new(
        "authored-silu",
        layers[1].clone(),
        vec!["authored-up".into()],
    ));
    graph.add_node(GraphNode::new(
        "authored-down",
        layers[2].clone(),
        vec!["authored-silu".into()],
    ));
    graph.set_output("authored-down");
    assert!(graph.propagate_crown(&silu_input()).is_err());
}

#[test]
fn original_affine_chain_silu_backward_rejects_genuine_foreign_profile_and_poison() {
    let ledger = budget();
    let pre =
        BoundedTensor::new(array![-1.5, -0.5].into_dyn(), array![0.5, 1.0].into_dyn()).unwrap();
    let genuine_gelu = forward(&pre, options(&ledger)).unwrap();
    let genuine_silu =
        forward_profile(&pre, MathematicalActivation::SiLu, options(&ledger)).unwrap();
    let demand = incoming(2);
    let before = ledger.receipt().unwrap();
    assert!(backward_profile(
        &demand,
        &pre,
        &genuine_gelu,
        MathematicalActivation::SiLu,
        options(&ledger)
    )
    .is_err());
    let after = ledger.receipt().unwrap();
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert!(after.work_reserved > before.work_reserved);
    assert!(backward(&demand, &pre, &genuine_silu, options(&ledger)).is_err());
    let output = backward_profile(
        &demand,
        &pre,
        &genuine_silu,
        MathematicalActivation::SiLu,
        options(&ledger),
    )
    .unwrap();
    assert!(output
        .bounds
        .lower_a_err()
        .unwrap()
        .row(0)
        .iter()
        .any(|&e| e > 0.0));
    assert!(output
        .bounds
        .upper_a_err()
        .unwrap()
        .row(0)
        .iter()
        .any(|&e| e > 0.0));
    assert!(output.bounds.lower_a().row(1).iter().all(|&a| a == 0.0));
    assert!(output.bounds.upper_a().row(1).iter().all(|&a| a == 0.0));
    assert!(output.bounds.lower_b()[1] < 0.0 && output.bounds.upper_b()[1] > 0.0);
    for values in [[-1.5, -0.5], [-1.28125, 0.0], [0.0, 0.0], [0.5, 1.0]] {
        let points: Vec<_> = values.into_iter().map(|x| R::point(q(x))).collect();
        let activated: Vec<_> = points.iter().map(silu).collect();
        for row in 0..2 {
            assert!(
                row_value(&output.bounds, row, &points, true).lo
                    <= row_value(&demand, row, &activated, true).lo
            );
            assert!(
                row_value(&output.bounds, row, &points, false).hi
                    >= row_value(&demand, row, &activated, false).hi
            );
        }
    }
    for poisoned in [-f32::from_bits(1), f32::NAN, f32::INFINITY] {
        let bad = LinearBounds::from_prevalidated_parts_with_optional_err(
            array![[0.0, 0.0]],
            array![0.0],
            array![[0.0, 0.0]],
            array![0.0],
            Some(array![[poisoned, 0.0]]),
            None,
        )
        .unwrap();
        let before = ledger.receipt().unwrap().live_persistent_bytes;
        assert!(backward_profile(
            &bad,
            &pre,
            &genuine_silu,
            MathematicalActivation::SiLu,
            options(&ledger)
        )
        .is_err());
        assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, before);
    }
    let changed =
        BoundedTensor::new(array![-1.25, -0.5].into_dyn(), array![0.5, 1.0].into_dyn()).unwrap();
    assert!(backward_profile(
        &demand,
        &changed,
        &genuine_silu,
        MathematicalActivation::SiLu,
        options(&ledger)
    )
    .is_err());
    // At zero both genuine profiles have identical boxes and image hashes.
    // Replacing only the LAST scalar envelope therefore isolates the typed
    // per-envelope profile check from the independent box/hash checks.
    let zeros =
        BoundedTensor::new(array![0.0, 0.0].into_dyn(), array![0.0, 0.0].into_dyn()).unwrap();
    let mut zero_silu =
        forward_profile(&zeros, MathematicalActivation::SiLu, options(&ledger)).unwrap();
    let mut zero_gelu = forward(&zeros, options(&ledger)).unwrap();
    assert!(backward_profile(
        &demand,
        &zeros,
        &zero_silu,
        MathematicalActivation::SiLu,
        options(&ledger)
    )
    .is_ok());
    assert_eq!(zero_silu.input, zero_gelu.input);
    assert_eq!(zero_silu.output, zero_gelu.output);
    std::mem::swap(
        &mut zero_silu.0.get_mut().unwrap().envelopes[1],
        &mut zero_gelu.0.get_mut().unwrap().envelopes[1],
    );
    let before = ledger.receipt().unwrap();
    assert!(backward_profile(
        &demand,
        &zeros,
        &zero_silu,
        MathematicalActivation::SiLu,
        options(&ledger)
    )
    .is_err());
    assert_eq!(
        ledger.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
}

#[test]
fn original_affine_chain_silu_exact_admission_capacity_and_last_alias_are_retained() {
    let source = BoundedTensor::new(array![-0.5].into_dyn(), array![0.5].into_dyn()).unwrap();
    let req = forward_request(&source).unwrap();
    let blocked = WorkBudget::new(
        None,
        req.scratch_bytes,
        req.persistent_bytes - 1,
        req.work,
        0,
        Duration::ZERO,
    )
    .unwrap();
    let before = blocked.receipt().unwrap();
    assert!(forward_profile(&source, MathematicalActivation::SiLu, options(&blocked)).is_err());
    let mut expected_refusal = before;
    expected_refusal.attempted += 1;
    expected_refusal.refused += 1;
    assert_eq!(blocked.receipt().unwrap(), expected_refusal);
    let exact = WorkBudget::new(
        None,
        req.scratch_bytes,
        req.persistent_bytes,
        req.work,
        0,
        Duration::ZERO,
    )
    .unwrap();
    let fact = forward_profile(&source, MathematicalActivation::SiLu, options(&exact)).unwrap();
    assert_eq!(exact.receipt().unwrap().work_reserved, req.work);
    assert!(
        fact.envelopes.capacity() * size_of::<ScalarEnvelope>()
            <= size_of::<ScalarEnvelope>() + 128
    );
    assert!(RetainedOwner::<ActivationFactData>::metadata_bytes().unwrap() <= META);
    drop(fact);
    assert_eq!(exact.receipt().unwrap().live_persistent_bytes, 0);
    assert!(forward_profile(&source, MathematicalActivation::SiLu, options(&exact)).is_err());
    assert_eq!(exact.receipt().unwrap().work_reserved, req.work);
    let ledger = budget();
    let layers = silu_layers(&ledger);
    let nodes = authored_nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    assert!(size_of::<Stage<'_>>() <= size_of::<PreparedOriginalAffine<'_>>() + size_of::<usize>());
    assert!(size_of::<PreparedOriginalAffine<'_>>() <= 64);
    let actual = META
        + context.stages.capacity() * size_of::<Stage<'_>>()
        + context.shared.owners.capacity() * size_of::<PointLinearLayer>();
    assert_eq!(context.shared.reservation.bytes(), actual);
    let facts = context.forward(&silu_input(), options(&ledger)).unwrap();
    let report = context
        .signed_linear(&facts, &incoming(6), options(&ledger))
        .unwrap();
    let last = report.clone();
    assert_eq!(
        last.bounds().lower_a().as_ptr(),
        report.bounds().lower_a().as_ptr()
    );
    let (forward_owner, chain_owner, fact_capacity, fact_bytes) =
        facts.retained_storage_observation();
    eprintln!("ORIGINAL_SILU_LAYOUT stage={} fact={} carrier={} activation_owner={} activation_data={} scalar={} fact_capacity={} forward_owner={} chain_owner={} stage_capacity={} owner_capacity={} chain_reservation={} report_reserved={} original_bytes={}",
        size_of::<Stage<'_>>(), size_of::<Fact>(), size_of::<Carrier>(),
        size_of::<OwnedActivationFact>(), size_of::<ActivationFactData>(), size_of::<ScalarEnvelope>(),
        fact_capacity, forward_owner, chain_owner, context.stages.capacity(),
        context.shared.owners.capacity(), actual, last.reserved_bytes(), last.original_owner_bytes());
    assert_eq!(fact_bytes, size_of::<Fact>());
    drop(report);
    drop(facts);
    drop(context);
    drop(nodes);
    drop(layers);
    let live = ledger.receipt().unwrap();
    assert_eq!(live.live_scratch_bytes, 0);
    assert_eq!(
        live.live_persistent_bytes,
        last.reserved_bytes() + last.original_owner_bytes()
    );
    drop(last);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().work_reserved, live.work_reserved);
}

#[test]
fn original_affine_chain_silu_cancellation_and_inherited_deadline_refuse_without_renewal() {
    let source = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[8]), -0.5),
        ArrayD::from_elem(IxDyn(&[8]), 0.5),
    )
    .unwrap();
    let req = forward_request(&source).unwrap();
    let cancellation = CancellationToken::default();
    let memory = MemoryAuthority::new(req.scratch_bytes + req.persistent_bytes);
    let ledger = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: req.scratch_bytes,
            persistent_bytes: req.persistent_bytes,
            max_work: req.work,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            INPUT_CANCEL.with(|hook| *hook.borrow_mut() = None);
        }
    }
    let _clear = Clear;
    INPUT_CANCEL.with(|hook| *hook.borrow_mut() = Some((4, cancellation.clone())));
    assert!(forward_profile(&source, MathematicalActivation::SiLu, options(&ledger)).is_err());
    assert!(cancellation.is_cancelled());
    INPUT_CANCEL.with(|hook| assert!(hook.borrow().is_none()));
    let refused = ledger.receipt().unwrap();
    assert_eq!(refused.admitted, 1);
    assert_eq!(refused.completed, 0);
    assert_eq!(refused.work_reserved, req.work);
    assert_eq!(refused.live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert!(forward_profile(&source, MathematicalActivation::SiLu, options(&ledger)).is_err());
    assert_eq!(ledger.receipt().unwrap().work_reserved, req.work);
    let ledger = budget();
    let pre = BoundedTensor::new(array![-0.5].into_dyn(), array![0.5].into_dyn()).unwrap();
    let mut fact = forward_profile(&pre, MathematicalActivation::SiLu, options(&ledger)).unwrap();
    fact.0.get_mut().expect("fresh unique fact").deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("test clock supports an expired deadline"),
    );
    let before = ledger.receipt().unwrap();
    assert!(backward_profile(
        &incoming(1),
        &pre,
        &fact,
        MathematicalActivation::SiLu,
        OriginalAffineOptions {
            deadline: Some(Instant::now() + Duration::from_secs(60)),
            ..options(&ledger)
        }
    )
    .is_err());
    let after = ledger.receipt().unwrap();
    assert!(after.work_reserved > before.work_reserved);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
}

#[test]
fn original_affine_chain_silu_zero_subnormal_and_global_hull_use_original_public_paths() {
    let ledger = budget();
    let layers = vec![
        owner(array![[1.0]], array![0.0], &ledger),
        Layer::SiLU(SiLULayer::new()),
        owner(array![[1.0]], array![0.0], &ledger),
    ];
    let nodes = authored_nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    let tiny = f32::from_bits(1);
    for (l, u) in [(0.0, 0.0), (-tiny, tiny), (-1.0e30, 1.0e30)] {
        let input = BoundedTensor::new(array![[l]].into_dyn(), array![[u]].into_dyn()).unwrap();
        let facts = context.forward(&input, options(&ledger)).unwrap();
        let demand = LinearBounds::from_prevalidated_parts_with_optional_err(
            array![[1.0]],
            array![0.0],
            array![[1.0]],
            array![0.0],
            None,
            None,
        )
        .unwrap();
        let report = context
            .signed_linear(&facts, &demand, options(&ledger))
            .unwrap();
        if u > 1.0e20 {
            // 0<sigmoid(x)<1 proves this universal image hull without trying
            // to evaluate exp(1e30), host math, or production transcendental.
            let Fact::SiLu(fact) = &facts.facts()[1] else {
                unreachable!()
            };
            assert_eq!(
                fact.envelopes[0].kind(),
                activation::EnvelopeKind::GlobalRange
            );
            assert_eq!(fact.envelopes[0].slope(), 0.0);
            assert!(q(facts.bounds().lower()[[0, 0]]) <= q(l));
            assert!(q(facts.bounds().upper()[[0, 0]]) >= q(u));
            assert!(row_value(report.bounds(), 0, &[R { lo: q(l), hi: q(u) }], true).lo <= q(l));
            assert!(row_value(report.bounds(), 0, &[R { lo: q(l), hi: q(u) }], false).hi >= q(u));
        } else {
            for x in [l, 0.0, u] {
                let points = vec![R::point(q(x))];
                let exact = silu(&points[0]);
                contains_output(facts.bounds(), std::slice::from_ref(&exact));
                assert!(row_value(report.bounds(), 0, &points, true).lo <= exact.lo);
                assert!(row_value(report.bounds(), 0, &points, false).hi >= exact.hi);
            }
        }
    }
}

// Exact existing scoped hardware guard recipe; only the calling test thread
// changes, and Drop restores its original control register on unwind.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct DirectedRounding {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl DirectedRounding {
    #[allow(unsafe_code)]
    fn enter() -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr",saved=out(reg)saved,options(nomem,nostack));
            let directed = (saved & !(3 << 22)) | (1 << 22);
            std::arch::asm!("msr fpcr, {directed}","isb",directed=in(reg)directed,options(nomem,nostack));
            Self { saved }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr((saved & !(3 << 13)) | (1 << 13));
            Self { saved }
        }
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl Drop for DirectedRounding {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("msr fpcr, {saved}","isb",saved=in(reg)self.saved,options(nomem,nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn original_affine_chain_silu_live_directed_rounding_refuses_new_and_retained_paths() {
    let ledger = budget();
    let pre = BoundedTensor::new(array![-0.5].into_dyn(), array![0.5].into_dyn()).unwrap();
    let scalar = forward_profile(&pre, MathematicalActivation::SiLu, options(&ledger)).unwrap();
    let layers = silu_layers(&ledger);
    let nodes = authored_nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    let input = silu_input();
    let facts = context.forward(&input, options(&ledger)).unwrap();
    let signed = incoming(6);
    assert!(context
        .signed_linear(&facts, &signed, options(&ledger))
        .is_ok());
    let before = ledger.receipt().unwrap();
    let directed = DirectedRounding::enter();
    assert!(!ny_core::has_f64_interval_proof_environment());
    assert!(forward_profile(&pre, MathematicalActivation::SiLu, options(&ledger)).is_err());
    assert!(backward_profile(
        &incoming(1),
        &pre,
        &scalar,
        MathematicalActivation::SiLu,
        options(&ledger)
    )
    .is_err());
    assert!(PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).is_err());
    assert!(context.forward(&input, options(&ledger)).is_err());
    assert!(context
        .signed_linear(&facts, &signed, options(&ledger))
        .is_err());
    assert!(context
        .sparse_margins(
            &facts,
            &[OriginalLogitDifference {
                target: 0,
                competitor: 1,
                position: 0,
            }],
            options(&ledger)
        )
        .is_err());
    drop(directed);
    assert!(ny_core::has_f64_interval_proof_environment());
    let after = ledger.receipt().unwrap();
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, 0);
    assert!(context
        .signed_linear(&facts, &signed, options(&ledger))
        .is_ok());
}

// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use ndarray::{array, Array1, Array2};
use num_bigint::BigInt;
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};
use std::time::Duration;

use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{LeaseClass, WorkBudget, WorkLimits, WorkReserve};
use crate::layers::{GELULayer, GeluApproximation, PointLinearBuildPolicy, ReLULayer};
use crate::{GraphNetwork, GraphNode, Network};

use super::*;

fn budget() -> WorkBudget {
    WorkBudget::new(None, 1 << 24, 1 << 26, 1 << 32, 4096, Duration::ZERO).unwrap()
}
fn options(budget: &WorkBudget) -> OriginalAffineOptions<'_> {
    OriginalAffineOptions {
        budget,
        class: LeaseClass::Optional,
        deadline: None,
    }
}
fn q(value: f32) -> Q {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 255;
    assert_ne!(exponent, 255);
    let fraction = bits & 0x7fffff;
    let (mantissa, power) = if exponent == 0 {
        (fraction, -149)
    } else {
        (fraction | (1 << 23), exponent as i32 - 127 - 23)
    };
    let mut integer = BigInt::from(mantissa);
    if bits >> 31 != 0 {
        integer = -integer;
    }
    if power >= 0 {
        Q::from_integer(integer << power as usize)
    } else {
        Q::new(integer, BigInt::from(1) << (-power) as usize)
    }
}
fn integer(n: i64) -> Q {
    Q::from_integer(BigInt::from(n))
}
fn fraction(n: i64, d: i64) -> Q {
    Q::new(BigInt::from(n), BigInt::from(d))
}

#[derive(Clone)]
struct R {
    lo: Q,
    hi: Q,
}
impl R {
    fn point(x: Q) -> Self {
        Self {
            lo: x.clone(),
            hi: x,
        }
    }
    fn add(&self, b: &Self) -> Self {
        Self {
            lo: &self.lo + &b.lo,
            hi: &self.hi + &b.hi,
        }
    }
    fn sub(&self, b: &Self) -> Self {
        Self {
            lo: &self.lo - &b.hi,
            hi: &self.hi - &b.lo,
        }
    }
    fn mul(&self, b: &Self) -> Self {
        let products = [
            &self.lo * &b.lo,
            &self.lo * &b.hi,
            &self.hi * &b.lo,
            &self.hi * &b.hi,
        ];
        Self {
            lo: products.iter().min().unwrap().clone(),
            hi: products.iter().max().unwrap().clone(),
        }
    }
    fn reciprocal(&self) -> Self {
        assert!(self.lo > Q::zero());
        Self {
            lo: self.hi.recip(),
            hi: self.lo.recip(),
        }
    }
    fn magnitude(&self) -> Q {
        self.lo.abs().max(self.hi.abs())
    }
}
fn atan(x: Q) -> R {
    let square = &x * &x;
    let mut power = x;
    let mut sum = Q::zero();
    for n in 0..64 {
        let term = &power / integer(2 * n + 1);
        if n % 2 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
        power *= &square;
    }
    R {
        lo: sum.clone(),
        hi: sum + power / integer(129),
    }
}
fn normal_constant() -> R {
    let pi = atan(fraction(1, 5))
        .mul(&R::point(integer(16)))
        .sub(&atan(fraction(1, 239)).mul(&R::point(integer(4))));
    let value = pi.mul(&R::point(integer(2)));
    let mut low = Q::zero();
    let mut high = integer(4);
    for _ in 0..96 {
        let mid = (&low + &high) / integer(2);
        if &mid * &mid <= value.lo {
            low = mid;
        } else {
            high = mid;
        }
    }
    let lo = low;
    low = Q::zero();
    high = integer(4);
    for _ in 0..96 {
        let mid = (&low + &high) / integer(2);
        if &mid * &mid >= value.hi {
            high = mid;
        } else {
            low = mid;
        }
    }
    R { lo, hi: high }.reciprocal()
}
fn gelu(x: Q, constant: &R) -> R {
    // Independent alternating integral of the normal density:
    // integral_0^t exp(-u²/2)du=sum (-1)^n t^(2n+1)/(2^n n!(2n+1)).
    // For t<=2 the terms decrease from n=0; the next term brackets the
    // 64-term remainder. No production exp/erf/Taylor/kernel is reused.
    let t = x.abs();
    assert!(t <= integer(2));
    let square = &t * &t;
    let mut term = t;
    let mut sum = Q::zero();
    for n in 0..64 {
        if n % 2 == 0 {
            sum += &term;
        } else {
            sum -= &term;
        }
        term = term * &square * integer(2 * n + 1)
            / (integer(2) * integer(n + 1) * integer(2 * n + 3));
    }
    let integral = R {
        lo: sum.clone(),
        hi: sum + term,
    };
    let integral = if x < Q::zero() {
        R::point(Q::zero()).sub(&integral)
    } else {
        integral
    };
    let probability = R::point(fraction(1, 2)).add(&constant.mul(&integral));
    R::point(x).mul(&probability)
}
fn affine(owner: &PointLinearLayer, values: &[R], positions: usize) -> Vec<R> {
    let mut result = Vec::new();
    for p in 0..positions {
        for row in 0..owner.out_features() {
            let mut value = R::point(owner.bias().map_or(Q::zero(), |b| q(b[row])));
            for c in 0..owner.in_features() {
                value = value.add(
                    &R::point(q(owner.weight()[[row, c]]))
                        .mul(&values[p * owner.in_features() + c]),
                );
            }
            result.push(value);
        }
    }
    result
}
fn row_value(bounds: &LinearBounds, row: usize, values: &[R], lower: bool) -> R {
    let (a, b, e) = if lower {
        (bounds.lower_a(), bounds.lower_b(), bounds.lower_a_err())
    } else {
        (bounds.upper_a(), bounds.upper_b(), bounds.upper_a_err())
    };
    let mut value = R::point(q(b[row]));
    for (c, x) in values.iter().enumerate() {
        value = value.add(&R::point(q(a[[row, c]])).mul(x));
        if let Some(e) = e {
            let charge = q(e[[row, c]]) * x.magnitude();
            value = value.add(&R::point(if lower { -charge } else { charge }));
        }
    }
    value
}
fn owner(weights: Array2<f32>, bias: Array1<f32>, ledger: &WorkBudget) -> Layer {
    Layer::PointLinear(
        PointLinearLayer::from_original_view(
            weights.view(),
            false,
            Some(bias.view().into_dyn()),
            PointLinearBuildPolicy {
                budget: ledger,
                deadline: None,
                max_resident_bytes: 1 << 26,
                max_scratch_bytes: 1 << 24,
                max_work: 1 << 32,
            },
        )
        .unwrap(),
    )
}
fn layers(ledger: &WorkBudget) -> Vec<Layer> {
    vec![
        owner(
            array![[0.5, -0.25], [0.1, 0.3]],
            array![0.125, -0.25],
            ledger,
        ),
        Layer::GELU(GELULayer::new(GeluApproximation::Erf)),
        owner(
            array![[1.5, -0.5], [0.1, 0.3], [-0.25, 1.0]],
            array![-0.125, 0.25, 0.125],
            ledger,
        ),
    ]
}
fn nodes(layers: &[Layer]) -> Vec<OriginalChainNode<'_>> {
    // Exact authored Pythia fragment topology/names, with synthetic small
    // original owners. Qualification on actual pinned weights is a separate
    // external original-model gate, never implied by this hermetic fixture.
    let names = ["block.5.mlp_up", "block.5.gelu_erf", "block.5.mlp_down"];
    let edges = [
        "block.5.post_attention_norm:0",
        "block.5.mlp_up:0",
        "block.5.gelu_erf:0",
        "block.5.mlp_down:0",
    ];
    layers
        .iter()
        .enumerate()
        .map(|(i, layer)| OriginalChainNode {
            ids: OriginalAffineIds {
                lowering: Digest::of_bytes(
                    b"synthetic-pythia-fragment",
                    b"exact-erf original unary topology",
                ),
                node: Digest::of_bytes(b"node", names[i].as_bytes()),
                input: Digest::of_bytes(b"edge", edges[i].as_bytes()),
                output: Digest::of_bytes(b"edge", edges[i + 1].as_bytes()),
            },
            layer,
        })
        .collect()
}
fn input() -> BoundedTensor {
    BoundedTensor::new(
        array![[-0.5, -0.75], [-0.25, -0.5]].into_dyn(),
        array![[0.75, 0.5], [0.5, 0.75]].into_dyn(),
    )
    .unwrap()
}
fn incoming(columns: usize) -> LinearBounds {
    LinearBounds::from_prevalidated_parts_with_optional_err(
        Array2::from_shape_fn((2, columns), |(r, c)| {
            if r == 0 {
                if c % 2 == 0 {
                    0.1
                } else {
                    -0.3
                }
            } else {
                0.0
            }
        }),
        array![0.125, 0.0],
        Array2::from_shape_fn((2, columns), |(r, c)| {
            if r == 0 {
                if c % 2 == 0 {
                    -0.7
                } else {
                    0.3
                }
            } else {
                0.0
            }
        }),
        array![-0.25, 0.0],
        Some(Array2::from_shape_fn((2, columns), |(r, c)| {
            if r == 0 {
                0.125
            } else {
                0.25 * (c + 1) as f32
            }
        })),
        Some(Array2::from_shape_fn((2, columns), |(r, c)| {
            if r == 0 {
                0.0625
            } else {
                0.5 * (c + 1) as f32
            }
        })),
    )
    .unwrap()
}

#[test]
fn original_affine_chain_gelu_signed_and_sparse_contain_rational_original_fragment() {
    let ledger = budget();
    let layers = layers(&ledger);
    let nodes = nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&nodes, options(&ledger)).unwrap();
    let source = input();
    let facts = context.forward(&source, options(&ledger)).unwrap();
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
    assert_eq!(facts.bounds().shape(), &[2, 3]);
    assert_eq!(signed.bounds().lower_a().dim(), (2, 4));
    assert_eq!(signed.scope(), context.scope());
    assert_eq!(facts.num_nodes(), 3);
    let constant = normal_constant();
    let Layer::PointLinear(up) = &layers[0] else {
        unreachable!()
    };
    let Layer::PointLinear(down) = &layers[2] else {
        unreachable!()
    };
    for mask in 0..17usize {
        let x: Vec<_> = (0..4)
            .map(|c| {
                let l = q(source.lower().as_slice().unwrap()[c]);
                let u = q(source.upper().as_slice().unwrap()[c]);
                if mask == 16 {
                    (l + u) / integer(2)
                } else if mask & (1 << c) == 0 {
                    l
                } else {
                    u
                }
            })
            .collect();
        let points: Vec<_> = x.iter().cloned().map(R::point).collect();
        let pre = affine(up, &points, 2);
        assert!(pre.iter().all(|p| p.lo == p.hi));
        let activated: Vec<_> = pre.into_iter().map(|p| gelu(p.lo, &constant)).collect();
        let y = affine(down, &activated, 2);
        for (i, value) in y.iter().enumerate() {
            assert!(q(facts.bounds().lower().as_slice().unwrap()[i]) <= value.lo);
            assert!(q(facts.bounds().upper().as_slice().unwrap()[i]) >= value.hi);
        }
        for row in 0..2 {
            assert!(
                row_value(signed.bounds(), row, &points, true).lo
                    <= row_value(&demand, row, &y, true).lo
            );
            assert!(
                row_value(signed.bounds(), row, &points, false).hi
                    >= row_value(&demand, row, &y, false).hi
            );
        }
        for (row, exact) in [y[0].sub(&y[1]), y[5].sub(&y[3])].iter().enumerate() {
            assert!(row_value(sparse.bounds(), row, &points, true).lo <= exact.lo);
            assert!(row_value(sparse.bounds(), row, &points, false).hi >= exact.hi);
        }
    }
}

#[test]
fn original_affine_chain_gelu_all_signed_error_planes_and_stored_product_gap_are_paid() {
    let ledger = budget();
    let pre =
        BoundedTensor::new(array![-1.0, -0.5].into_dyn(), array![0.5, 1.0].into_dyn()).unwrap();
    let fact = forward(&pre, options(&ledger)).unwrap();
    let demand = incoming(2);
    let output = backward(&demand, &pre, &fact, options(&ledger)).unwrap();
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
    let constant = normal_constant();
    for x in [[-1.0, -0.5], [-0.5, 0.5], [0.0, 0.0], [0.5, 1.0]] {
        let points: Vec<_> = x.into_iter().map(|v| R::point(q(v))).collect();
        let y: Vec<_> = points
            .iter()
            .map(|p| gelu(p.lo.clone(), &constant))
            .collect();
        for row in 0..2 {
            assert!(
                row_value(&output.bounds, row, &points, true).lo
                    <= row_value(&demand, row, &y, true).lo
            );
            assert!(
                row_value(&output.bounds, row, &points, false).hi
                    >= row_value(&demand, row, &y, false).hi
            );
        }
        if x[1] != 0.0 {
            // A zero nominal carrier with its actual error discarded would
            // publish bias0, which excludes this original nonzero support.
            assert!(row_value(&demand, 1, &y, true).lo < Q::zero());
            assert!(row_value(&demand, 1, &y, false).hi > Q::zero());
        }
    }
}

#[test]
fn original_affine_chain_gelu_subnormal_errors_poison_and_box_binding_are_checked() {
    let ledger = budget();
    let pre = BoundedTensor::new(array![-0.5].into_dyn(), array![0.5].into_dyn()).unwrap();
    let fact = forward(&pre, options(&ledger)).unwrap();
    let tiny = f32::from_bits(1);
    let valid = LinearBounds::from_prevalidated_parts_with_optional_err(
        array![[tiny]],
        array![0.0],
        array![[-tiny]],
        array![0.0],
        Some(array![[tiny]]),
        Some(array![[tiny]]),
    )
    .unwrap();
    let output = backward(&valid, &pre, &fact, options(&ledger)).unwrap();
    assert!(output.bounds.lower_a().iter().all(|&a| a == 0.0));
    assert!(output.bounds.upper_a().iter().all(|&a| a == 0.0));
    assert!(output.bounds.lower_a_err().unwrap()[[0, 0]] >= f32::MIN_POSITIVE);
    let constant = normal_constant();
    for v in [-0.5, 0.0, 0.5] {
        let x = vec![R::point(q(v))];
        let y = vec![gelu(q(v), &constant)];
        assert!(row_value(&output.bounds, 0, &x, true).lo <= row_value(&valid, 0, &y, true).lo);
        assert!(row_value(&output.bounds, 0, &x, false).hi >= row_value(&valid, 0, &y, false).hi);
    }
    for poisoned in [-tiny, f32::NAN, f32::INFINITY] {
        let bad = LinearBounds::from_prevalidated_parts_with_optional_err(
            array![[0.0]],
            array![0.0],
            array![[0.0]],
            array![0.0],
            Some(array![[poisoned]]),
            None,
        )
        .unwrap();
        let before = ledger.receipt().unwrap().live_persistent_bytes;
        assert!(backward(&bad, &pre, &fact, options(&ledger)).is_err());
        assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, before);
    }
    let changed = BoundedTensor::new(array![-0.25].into_dyn(), array![0.5].into_dyn()).unwrap();
    assert!(backward(&valid, &changed, &fact, options(&ledger)).is_err());
    let other = budget();
    let before = other.receipt().unwrap();
    assert!(backward(&valid, &pre, &fact, options(&other)).is_err());
    assert_eq!(other.receipt().unwrap(), before);
}

#[test]
fn original_affine_chain_gelu_explicit_erf_scope_preserves_old_profiles_and_quarantine() {
    let ledger = budget();
    let mut layers = layers(&ledger);
    let node_list = nodes(&layers);
    let erf = PreparedOriginalAffineChain::prepare(&node_list, options(&ledger)).unwrap();
    assert_eq!(
        erf.scope().assumptions,
        Digest::of_bytes(
            b"original-affine-chain-profile",
            ORIGINAL_CHAIN_GELU_EXACT_REAL_PROFILE.as_bytes()
        )
    );
    let relu = Layer::ReLU(ReLULayer::new());
    let mut old_nodes = node_list.clone();
    old_nodes[1].layer = &relu;
    let old = PreparedOriginalAffineChain::prepare(&old_nodes, options(&ledger)).unwrap();
    assert_eq!(
        old.scope().assumptions,
        Digest::of_bytes(
            b"original-affine-chain-profile",
            ORIGINAL_CHAIN_EXACT_REAL_PROFILE.as_bytes()
        )
    );
    assert_eq!(old.scope().model, erf.scope().model);
    assert_ne!(old.scope().coordinates, erf.scope().coordinates);
    let tanh = Layer::GELU(GELULayer::new(GeluApproximation::Tanh));
    let mut bad = node_list.clone();
    bad[1].layer = &tanh;
    assert!(PreparedOriginalAffineChain::prepare(&bad, options(&ledger)).is_err());
    let mut bad = node_list.clone();
    bad[2].ids.input = bad[0].ids.input;
    assert!(PreparedOriginalAffineChain::prepare(&bad, options(&ledger)).is_err());
    let facts = erf.forward(&input(), options(&ledger)).unwrap();
    assert!(old
        .signed_linear(&facts, &incoming(6), options(&ledger))
        .is_err());
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("up", layers[0].clone()));
    graph.add_node(GraphNode::new("gelu", layers[1].clone(), vec!["up".into()]));
    graph.add_node(GraphNode::new(
        "down",
        layers[2].clone(),
        vec!["gelu".into()],
    ));
    graph.set_output("down");
    assert!(graph.propagate_crown(&input()).is_err());
    let mut network = Network::new();
    for layer in &layers {
        network.add_layer(layer.clone());
    }
    assert!(network.propagate_crown(&input()).is_err());
    // Independent source metadata/profile changes bind a distinct chain.
    let mut renamed = node_list.clone();
    renamed[1].ids.node = Digest::of_bytes(b"other-authored-gelu", b"exact same math");
    let renamed = PreparedOriginalAffineChain::prepare(&renamed, options(&ledger)).unwrap();
    assert_ne!(renamed.scope().coordinates, erf.scope().coordinates);
    drop((erf, old, renamed, facts, node_list));
    layers.clear();
}

#[test]
fn original_affine_chain_gelu_admission_and_last_clone_keep_all_original_claims() {
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
    assert!(forward(&source, options(&blocked)).is_err());
    assert_eq!(blocked.receipt().unwrap().admitted, 0);
    assert_eq!(blocked.receipt().unwrap().live_persistent_bytes, 0);
    let exact = WorkBudget::new(
        None,
        req.scratch_bytes,
        req.persistent_bytes,
        req.work,
        0,
        Duration::ZERO,
    )
    .unwrap();
    let fact = forward(&source, options(&exact)).unwrap();
    assert_eq!(exact.receipt().unwrap().work_reserved, req.work);
    drop(fact);
    assert_eq!(exact.receipt().unwrap().live_persistent_bytes, 0);
    assert!(forward(&source, options(&exact)).is_err());
    assert_eq!(exact.receipt().unwrap().work_reserved, req.work);

    let ledger = budget();
    let layers = layers(&ledger);
    let node_list = nodes(&layers);
    let context = PreparedOriginalAffineChain::prepare(&node_list, options(&ledger)).unwrap();
    let facts = context.forward(&input(), options(&ledger)).unwrap();
    let report = context
        .signed_linear(&facts, &incoming(6), options(&ledger))
        .unwrap();
    let last = report.clone();
    drop(report);
    drop(facts);
    drop(context);
    drop(node_list);
    drop(layers);
    let live = ledger.receipt().unwrap();
    assert_eq!(live.live_scratch_bytes, 0);
    assert_eq!(
        live.live_persistent_bytes,
        last.reserved_bytes() + last.original_owner_bytes()
    );
    let spent = live.work_reserved;
    drop(last);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().work_reserved, spent);
}

#[test]
fn original_affine_chain_gelu_input_scan_cancellation_and_earlier_fact_deadline_are_retained() {
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
    assert!(forward(&source, options(&ledger)).is_err());
    assert!(cancellation.is_cancelled());
    INPUT_CANCEL.with(|hook| assert!(hook.borrow().is_none()));
    assert_eq!(ledger.receipt().unwrap().admitted, 1);
    assert_eq!(ledger.receipt().unwrap().completed, 0);
    assert_eq!(ledger.receipt().unwrap().work_reserved, req.work);
    assert_eq!(ledger.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert!(forward(&source, options(&ledger)).is_err());
    assert_eq!(ledger.receipt().unwrap().work_reserved, req.work);

    let ledger = budget();
    let pre = BoundedTensor::new(array![-0.5].into_dyn(), array![0.5].into_dyn()).unwrap();
    let mut fact = forward(&pre, options(&ledger)).unwrap();
    // Inject only an expired inherited deadline; do not mutate FP state or
    // sleep. A later caller deadline cannot extend the retained fact deadline.
    fact.0
        .get_mut()
        .expect("one freshly minted fact owner")
        .deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("test clock supports an expired deadline"),
    );
    let before = ledger.receipt().unwrap();
    assert!(backward(&incoming(1), &pre, &fact, options(&ledger)).is_err());
    let after = ledger.receipt().unwrap();
    assert!(after.work_reserved > before.work_reserved);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
}

#[path = "silu_tests.rs"]
mod silu;

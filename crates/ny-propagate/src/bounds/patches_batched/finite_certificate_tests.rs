// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exact signed controls for the actual finite fan-in and stored certificates.
use super::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};
use std::time::Duration;

fn q(x: f64) -> Q {
    Q::from_float(x).expect("finite binary64")
}
fn q32(x: f32) -> Q {
    q(f32_to_f64_exact(x))
}
fn dense(lower: &[f32], upper: &[f32], lb: f32, ub: f32) -> BatchedLinearBounds {
    assert_eq!(lower.len(), upper.len());
    let shape = IxDyn(&[1, 1, lower.len()]);
    BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(shape.clone(), lower.to_vec()).unwrap(),
        ArrayD::from_elem(IxDyn(&[1, 1]), lb),
        ArrayD::from_shape_vec(shape, upper.to_vec()).unwrap(),
        ArrayD::from_elem(IxDyn(&[1, 1]), ub),
        vec![1, lower.len()],
        vec![1, 1],
    )
}
fn deadline() -> Option<Instant> {
    Some(Instant::now() + Duration::from_secs(120))
}
fn merge(parts: &[BatchedLinearBounds]) -> Result<BatchedLinearBounds> {
    let limit = deadline();
    let mut carrier = BatchedCrownBounds::Dense(parts[0].clone());
    for part in &parts[1..] {
        carrier.merge_dense_checked_with_deadline(
            part.clone(),
            "finite_batched_certificate control",
            limit,
        )?;
    }
    assert!(matches!(&carrier, BatchedCrownBounds::FiniteDense64(_)));
    carrier.into_batched_dense_checked_with_deadline("finite_batched_certificate control", limit)
}
fn error(bounds: &BatchedLinearBounds, lower: bool, i: usize) -> Q {
    let plane = if lower {
        &bounds.lower_a_err
    } else {
        &bounds.upper_a_err
    };
    plane
        .as_ref()
        .map_or_else(Q::zero, |a| q32(a.as_slice().unwrap()[i]))
}
fn signed_support(bounds: &BatchedLinearBounds, lower: bool, x: &[f64]) -> Q {
    let a = if lower {
        bounds.lower_a()
    } else {
        bounds.upper_a()
    };
    let b = if lower {
        bounds.lower_b()
    } else {
        bounds.upper_b()
    };
    let mut result = q32(b.as_slice().unwrap()[0]);
    for (i, (&a, &x)) in a.iter().zip(x).enumerate() {
        result += q32(a) * q(x);
        let term = error(bounds, lower, i) * q(x).abs();
        if lower {
            result -= term;
        } else {
            result += term;
        }
    }
    result
}
fn originals(parts: &[BatchedLinearBounds], lower: bool, x: &[f64]) -> Q {
    parts
        .iter()
        .map(|part| signed_support(part, lower, x))
        .sum()
}
fn finite_planes(bounds: &BatchedLinearBounds) -> usize {
    usize::from(bounds.lower_a_err.is_some()) + usize::from(bounds.upper_a_err.is_some())
}
fn assert_original_support(parts: &[BatchedLinearBounds], stored: &BatchedLinearBounds, x: &[f64]) {
    assert!(signed_support(stored, true, x) <= originals(parts, true, x));
    assert!(signed_support(stored, false, x) >= originals(parts, false, x));
}

#[test]
fn finite_batched_certificate_exact_half_fanin_preserves_signed_negative_point() -> Result<()> {
    crate::tests::with_env_edits(|_env| {
        let parts = [
            dense(&[0.5], &[0.5], 0.0, 0.0),
            dense(&[0.5], &[0.5], 0.0, 0.0),
        ];
        let stored = merge(&parts)?;
        assert_original_support(&parts, &stored, &[-1.0]);
        assert_eq!(stored.lower_a()[[0, 0, 0]].to_bits(), 1.0_f32.to_bits());
        assert_eq!(stored.upper_a()[[0, 0, 0]].to_bits(), 1.0_f32.to_bits());
        assert_eq!(finite_planes(&stored), 0);
        // Historical directed center and absent E excluded the original −1;
        // the bias's one subnormal step cannot pay a 2^-24 coefficient gap.
        let old_lower = q32(next_down_f32(1.0)) * q(-1.0) + q32(next_down_f32(0.0));
        assert!(old_lower > q(-1.0));
        Ok(())
    })
}

#[test]
fn finite_batched_certificate_sparse_storage_gap_creates_only_required_plane() -> Result<()> {
    crate::tests::with_env_edits(|_env| {
        let tiny = (2.0_f32).powi(-25);
        let parts = [
            dense(&[1.0, -2.0, 0.0], &[1.0, -2.0, 0.0], -0.5, 0.5),
            dense(&[tiny, 0.0, 0.0], &[0.0, 0.0, 0.0], -0.25, 0.25),
        ];
        let stored = merge(&parts)?;
        assert_eq!(finite_planes(&stored), 1);
        let e = stored.lower_a_err.as_ref().unwrap();
        assert!(e[[0, 0, 0]] >= tiny);
        assert_eq!(e[[0, 0, 1]], 0.0);
        assert_eq!(e[[0, 0, 2]], 0.0);
        for x0 in [-1.0, 1.0] {
            for x1 in [-2.0, 0.5] {
                assert_original_support(&parts, &stored, &[x0, x1, -0.25]);
            }
        }
        Ok(())
    })
}

#[test]
fn finite_batched_certificate_incoming_signed_errors_and_biases_remain_finite() -> Result<()> {
    crate::tests::with_env_edits(|_env| {
        let mut first = dense(&[-1.0, 0.25], &[0.5, -2.0], -0.75, 0.125);
        first.lower_a_err =
            Some(ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.25, 0.0]).unwrap());
        let mut second = dense(&[0.5, -0.125], &[-0.25, 0.5], -0.125, 0.25);
        second.lower_a_err = Some(ArrayD::from_elem(IxDyn(&[1, 1, 2]), 0.125));
        second.upper_a_err = Some(ArrayD::from_elem(IxDyn(&[1, 1, 2]), 0.5));
        let parts = [first, second];
        let stored = merge(&parts)?;
        assert_eq!(finite_planes(&stored), 2);
        assert!(stored.lower_b()[[0, 0]].is_finite() && stored.upper_b()[[0, 0]].is_finite());
        for x0 in [-3.0, 1.0] {
            for x1 in [-0.5, 2.0] {
                assert_original_support(&parts, &stored, &[x0, x1]);
            }
        }
        let republished = merge(&[stored.clone(), dense(&[0.0, 0.0], &[0.0, 0.0], 0.0, 0.0)])?;
        assert_eq!(republished.lower_a_err, stored.lower_a_err);
        assert_eq!(republished.upper_a_err, stored.upper_a_err);
        for x0 in [-3.0, 1.0] {
            for x1 in [-0.5, 2.0] {
                assert_original_support(&parts, &republished, &[x0, x1]);
            }
        }
        Ok(())
    })
}

#[test]
fn finite_batched_certificate_error_only_zero_planes_and_compatible_reshape() -> Result<()> {
    crate::tests::with_env_edits(|_env| {
        let mut first = dense(&[0.0, -0.0], &[0.0, 0.0], -0.25, 0.5);
        first.lower_a_err =
            Some(ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.125, 0.0]).unwrap());
        first.upper_a_err = Some(ArrayD::from_elem(IxDyn(&[1, 1, 2]), -0.0));
        let mut second = dense(&[0.0, 0.0], &[0.0, 0.0], -0.125, 0.25);
        second.lower_a_err =
            Some(ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.25, 0.0]).unwrap());
        // This is the existing contiguous flattened compatibility contract.
        second.lower_a = second
            .lower_a
            .into_shape_with_order(IxDyn(&[1, 2]))
            .unwrap();
        second.upper_a = second
            .upper_a
            .into_shape_with_order(IxDyn(&[1, 2]))
            .unwrap();
        second.lower_b = second.lower_b.into_shape_with_order(IxDyn(&[1])).unwrap();
        second.upper_b = second.upper_b.into_shape_with_order(IxDyn(&[1])).unwrap();
        second.lower_a_err = second
            .lower_a_err
            .map(|a| a.into_shape_with_order(IxDyn(&[1, 2])).unwrap());
        second.input_shape = vec![2];
        second.output_shape = vec![1];
        let parts = [first, second];
        let stored = merge(&parts)?;
        assert_eq!(finite_planes(&stored), 1);
        assert!(stored.lower_a_err.as_ref().unwrap()[[0, 0, 0]] >= 0.375);
        for x in [[-2.0, 3.0], [1.0, -1.0]] {
            assert_original_support(&parts, &stored, &x);
        }
        Ok(())
    })
}

#[test]
fn finite_batched_certificate_cancellation_and_subnormal_centers_pay_full_original_gap(
) -> Result<()> {
    crate::tests::with_env_edits(|_env| {
        let big = (2.0_f32).powi(60);
        let parts = [
            dense(&[big], &[big], 0.0, 0.0),
            dense(&[1.0], &[1.0], 0.0, 0.0),
            dense(&[-big], &[-big], 0.0, 0.0),
        ];
        let stored = merge(&parts)?;
        assert_eq!(stored.lower_a()[[0, 0, 0]], 0.0);
        assert!(stored.lower_a_err.as_ref().unwrap()[[0, 0, 0]] >= 1.0);
        for x in [-1.0, 1.0] {
            assert_original_support(&parts, &stored, &[x]);
        }
        let sub = f32::from_bits(1);
        let parts = [
            dense(&[sub], &[-sub], 0.0, 0.0),
            dense(&[0.0], &[0.0], 0.0, 0.0),
        ];
        let stored = merge(&parts)?;
        assert_eq!(stored.lower_a()[[0, 0, 0]], 0.0);
        assert_eq!(stored.upper_a()[[0, 0, 0]], 0.0);
        for plane in [&stored.lower_a_err, &stored.upper_a_err] {
            let error = plane.as_ref().unwrap()[[0, 0, 0]];
            assert!(error >= f32::MIN_POSITIVE);
        }
        for x in [-1.0, 1.0] {
            assert_original_support(&parts, &stored, &[x]);
        }
        Ok(())
    })
}

#[test]
fn finite_batched_certificate_poison_and_malformed_incoming_errors_are_vacuous() -> Result<()> {
    crate::tests::with_env_edits(|_env| {
        for poison in [f32::NAN, f32::INFINITY, -f32::from_bits(1)] {
            let mut first = dense(&[1.0], &[1.0], -0.25, 0.5);
            first.lower_a_err = Some(ArrayD::from_elem(IxDyn(&[1, 1, 1]), poison));
            let stored = merge(&[first, dense(&[0.0], &[0.0], 0.0, 0.0)])?;
            assert_eq!(stored.lower_b()[[0, 0]], f32::NEG_INFINITY);
            assert_eq!(
                stored.lower_a_err.as_ref().unwrap()[[0, 0, 0]],
                f32::INFINITY
            );
            assert!(stored.upper_b()[[0, 0]].is_finite());
        }
        let mut first = dense(&[1.0], &[1.0], 0.0, 0.0);
        first.lower_a_err = Some(ArrayD::from_elem(IxDyn(&[2]), 0.0));
        let stored = merge(&[first, dense(&[0.0], &[0.0], 0.0, 0.0)])?;
        assert_eq!(stored.lower_b()[[0, 0]], f32::NEG_INFINITY);
        assert_eq!(
            stored.lower_a_err.as_ref().unwrap()[[0, 0, 0]],
            f32::INFINITY
        );
        Ok(())
    })
}

#[test]
fn finite_batched_certificate_original_f64_tiny_and_nonfinite_storage_controls() -> Result<()> {
    crate::tests::with_env_edits(|_env| {
        for source in [
            f64::from_bits(1),
            -f64::from_bits(1),
            1.0 + (2.0_f64).powi(-25),
        ] {
            let finite = FiniteBatchedLinearBounds64 {
                lower_a: ArrayD::from_elem(IxDyn(&[1, 1, 1]), source),
                upper_a: ArrayD::from_elem(IxDyn(&[1, 1, 1]), source),
                lower_b: ArrayD::from_elem(IxDyn(&[1, 1]), 0.0),
                upper_b: ArrayD::from_elem(IxDyn(&[1, 1]), 0.0),
                lower_a_err: None,
                upper_a_err: None,
                input_shape: vec![1, 1],
                output_shape: vec![1, 1],
            };
            let mut admission = BatchedCrownBounds::finite_sidecar_downcast_admission(&finite)?;
            let mut clock = PatchesMaterializationDeadline::new(deadline());
            let stored = finite.to_f32(&mut admission, &mut clock)?;
            assert_eq!(finite_planes(&stored), 2);
            for x in [-1.0, 1.0] {
                let original = q(source) * q(x);
                assert!(signed_support(&stored, true, &[x]) <= original);
                assert!(signed_support(&stored, false, &[x]) >= original);
            }
        }
        for source in [f64::NAN, f64::INFINITY, f64::MAX] {
            let finite = FiniteBatchedLinearBounds64 {
                lower_a: ArrayD::from_elem(IxDyn(&[1, 1, 1]), source),
                upper_a: ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0),
                lower_b: ArrayD::from_elem(IxDyn(&[1, 1]), 0.0),
                upper_b: ArrayD::from_elem(IxDyn(&[1, 1]), 0.0),
                lower_a_err: None,
                upper_a_err: None,
                input_shape: vec![1, 1],
                output_shape: vec![1, 1],
            };
            let mut admission = BatchedCrownBounds::finite_sidecar_downcast_admission(&finite)?;
            let mut clock = PatchesMaterializationDeadline::new(deadline());
            let stored = finite.to_f32(&mut admission, &mut clock)?;
            assert_eq!(stored.lower_a()[[0, 0, 0]], 0.0);
            assert_eq!(stored.upper_a()[[0, 0, 0]], 0.0);
            assert_eq!(stored.lower_b()[[0, 0]], f32::NEG_INFINITY);
            assert_eq!(stored.upper_b()[[0, 0]], f32::INFINITY);
            assert_eq!(
                stored.lower_a_err.as_ref().unwrap()[[0, 0, 0]],
                f32::INFINITY
            );
            assert_eq!(
                stored.upper_a_err.as_ref().unwrap()[[0, 0, 0]],
                f32::INFINITY
            );
        }
        Ok(())
    })
}

#[test]
fn finite_batched_certificate_worst_plane_admission_precedes_lazy_allocations() -> Result<()> {
    crate::tests::with_env_edits(|env| {
        let first = dense(&[1.0, 0.0], &[1.0, 0.0], 0.0, 0.0);
        let second = first.clone();
        let mut admission =
            BatchedCrownBounds::guard_dense_sidecar_staging(&first, &second, false)?;
        let mut clock = PatchesMaterializationDeadline::new(deadline());
        let finite = FiniteBatchedLinearBounds64::from_f32(&first, &mut admission, &mut clock)?;
        let output = BatchedCrownBounds::finite_sidecar_downcast_admission(&finite)?;
        let shape_bytes =
            (finite.input_shape.len() + finite.output_shape.len()) * size_of::<usize>();
        let centers = (finite.lower_a.len() + finite.upper_a.len()) * size_of::<f32>();
        let biases = (finite.lower_b.len() + finite.upper_b.len()) * size_of::<f32>();
        assert_eq!(
            output.required_bytes,
            finite.memory_bytes() + 2 * centers + biases + shape_bytes
        );
        // Both absent error planes are included before their lazy casts are known.
        assert!(finite.lower_a_err.is_none() && finite.upper_a_err.is_none());
        env.set("NY_DENSE_BUDGET_MB", "0");
        assert!(matches!(
            BatchedCrownBounds::finite_sidecar_downcast_admission(&finite),
            Err(NyError::CpuMemoryExceeded { .. })
        ));
        let mut carrier = BatchedCrownBounds::Dense(first.clone());
        assert!(matches!(
            carrier.merge_dense_checked_with_deadline(second, "finite admission", deadline()),
            Err(NyError::CpuMemoryExceeded { .. })
        ));
        let BatchedCrownBounds::Dense(retained) = carrier else {
            panic!("failed admission replaced source")
        };
        assert_eq!(retained.lower_a(), first.lower_a());
        Ok(())
    })
}

#[test]
fn finite_batched_certificate_mid_error_publication_and_expired_deadline_refuse() -> Result<()> {
    crate::tests::with_env_edits(|_env| {
        let mut first = dense(&[1.0], &[1.0], 0.0, 0.0);
        first.lower_a_err = Some(ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.25));
        let mut clock = PatchesMaterializationDeadline::new(deadline());
        let mut admission = BatchedCrownBounds::guard_dense_sidecar_staging(&first, &first, false)?;
        let finite = FiniteBatchedLinearBounds64::from_f32(&first, &mut admission, &mut clock)?;
        let before = finite.lower_a.clone();
        let mut admission = BatchedCrownBounds::finite_sidecar_downcast_admission(&finite)?;
        let mut interrupted = PatchesMaterializationDeadline::forced_at(
            "during finite lower certificate publication",
        );
        assert!(matches!(
            finite.to_f32(&mut admission, &mut interrupted),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(finite.lower_a, before);
        let mut carrier = BatchedCrownBounds::Dense(first.clone());
        assert!(matches!(
            carrier.merge_dense_checked_with_deadline(
                first.clone(),
                "expired finite",
                Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
            ),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(carrier, BatchedCrownBounds::Dense(_)));
        Ok(())
    })
}

#[test]
fn finite_batched_certificate_public_some_deadline_batched_dag_preserves_rational_vertices(
) -> Result<()> {
    crate::tests::with_env_edits(|_env| {
        use crate::layers::{AddLayer, Layer, LinearLayer};
        use crate::network::{GraphNetwork, GraphNode};
        use crate::types::BoundsProvenance;
        use ny_tensor::BoundedTensor;
        let mut graph = GraphNetwork::new();
        for name in ["left", "right"] {
            graph.add_node(GraphNode::from_input(
                name,
                Layer::Linear(LinearLayer::new(
                    arr2(&[[0.5_f32]]),
                    Some(arr1(&[-0.125_f32])),
                )?),
            ));
        }
        graph.add_node(GraphNode::binary(
            "sum",
            Layer::Add(AddLayer),
            "left",
            "right",
        ));
        graph.set_output("sum");
        let input = BoundedTensor::new(arr1(&[-2.0_f32]).into_dyn(), arr1(&[-1.0_f32]).into_dyn())?;
        let result = graph.propagate_crown_batched_with_relaxation_and_deadline(
            &input,
            crate::MulBinaryRelaxationMode::default(),
            deadline(),
        )?;
        assert!(matches!(result.provenance, BoundsProvenance::Crown));
        for x in [-2.0, -1.0] {
            let exact = q(x) - q(0.25);
            assert!(q32(result.bounds.lower()[[0]]) <= exact);
            assert!(q32(result.bounds.upper()[[0]]) >= exact);
        }
        Ok(())
    })
}

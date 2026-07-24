// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Public-API gates for the exact-decimal direct input-box extractor.

use std::path::{Path, PathBuf};

use num_rational::BigRational;
use ny_onnx::vnnlib::{
    load_vnnlib_with_certified_input_box, parse_vnnlib_with_certified_input_box,
};

fn exact(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite test value")
}

#[test]
fn non_dyadic_point_is_outward_and_remains_a_hint() {
    let content = "
        (declare-const X_0 Real)
        (assert (>= X_0 0.1))
        (assert (<= X_0 0.1))
    ";
    let (_, box_) = parse_vnnlib_with_certified_input_box(content).unwrap();
    let target = BigRational::new(1.into(), 10.into());
    assert_eq!(box_.declared_point(), &[true]);
    assert!(exact(box_.lower()[0]) <= target);
    assert!(exact(box_.upper()[0]) >= target);
    assert!(box_.lower()[0] < box_.upper()[0]);
}

#[test]
fn exact_dyadic_point_needs_no_gratuitous_width() {
    let content = "
        (declare-const X_0 Real)
        (assert (= X_0 0.5))
    ";
    let (_, box_) = parse_vnnlib_with_certified_input_box(content).unwrap();
    assert_eq!(box_.lower(), &[0.5]);
    assert_eq!(box_.upper(), &[0.5]);
    assert_eq!(box_.declared_point(), &[true]);
}

#[test]
fn reversed_atoms_and_multiple_bounds_tighten() {
    let content = "
        (declare-const X_0 Real)
        (assert (<= -2e-1 X_0))
        (assert (>= X_0 -0.1))
        (assert (>= 0.4 X_0))
        (assert (<= X_0 0.3))
    ";
    let (_, box_) = parse_vnnlib_with_certified_input_box(content).unwrap();
    assert!(exact(box_.lower()[0]) <= BigRational::new((-1).into(), 10.into()));
    assert!(exact(box_.upper()[0]) >= BigRational::new(3.into(), 10.into()));
    assert_eq!(box_.declared_point(), &[false]);
}

#[test]
fn unsupported_or_incomplete_input_surfaces_fail_closed() {
    let compound = "
        (declare-const X_0 Real)
        (assert (>= X_0 0.0))
        (assert (<= (+ X_0 0.0) 1.0))
    ";
    let error = parse_vnnlib_with_certified_input_box(compound).unwrap_err();
    assert!(error.to_string().contains("direct axis-aligned atom"));

    let incomplete = "
        (declare-const X_0 Real)
        (assert (>= X_0 0.0))
    ";
    let error = parse_vnnlib_with_certified_input_box(incomplete).unwrap_err();
    assert!(error.to_string().contains("missing an upper bound"));
}

#[test]
fn platform_maximum_declaration_index_fails_without_panicking() {
    for prefix in ["X_", "Y_"] {
        for suffix in ["", " Real"] {
            let content = format!("(declare-const {prefix}{}{suffix})", usize::MAX);
            let result =
                std::panic::catch_unwind(|| parse_vnnlib_with_certified_input_box(&content));
            assert!(result.is_ok(), "public parser boundary must not panic");
            let error = result.unwrap().unwrap_err();
            assert!(error.to_string().contains("index overflows"));
        }
    }
}

#[test]
fn real_metaroom_119_retains_161_varying_coordinates() {
    let path = std::env::var_os("NY_METAROOM_119_VNNLIB")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join(
                "../../benchmarks/vnncomp2025/benchmarks/metaroom_2023/vnnlib/\
                 spec_idx_119_eps_0.00000436.vnnlib",
            )
        });
    if !path.exists() {
        return;
    }
    let (spec, box_) = load_vnnlib_with_certified_input_box(path).unwrap();
    assert_eq!(spec.num_inputs, 5_376);
    assert_eq!(box_.len(), 5_376);
    assert_eq!(
        box_.declared_point().iter().filter(|&&point| point).count(),
        5_215
    );
    assert_eq!(
        box_.declared_point()
            .iter()
            .filter(|&&point| !point)
            .count(),
        161
    );
    assert!(box_
        .lower()
        .iter()
        .zip(box_.upper())
        .all(|(&lower, &upper)| lower.is_finite() && lower <= upper && upper.is_finite()));
}

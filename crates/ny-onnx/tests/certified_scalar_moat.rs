// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_onnx::vnnlib::{
    load_vnnlib_with_certified_scalar_moat, parse_vnnlib, parse_vnnlib_with_certified_scalar_moat,
};
use sha2::{Digest, Sha256};

const PROP0: &str = include_str!("fixtures/cgan_nch1_prop0.vnnlib");
const PROP1: &str = include_str!("fixtures/cgan_nch1_prop1.vnnlib");
const PROP2: &str = include_str!("fixtures/cgan_nch1_prop2.vnnlib");
const PROP3: &str = include_str!("fixtures/cgan_nch1_prop3.vnnlib");

fn property(output_body: &str) -> String {
    format!(
        "
        (declare-const X_0 Real)
        (declare-const X_1 Real)
        (declare-const X_2 Real)
        (declare-const X_3 Real)
        (declare-const X_4 Real)
        (declare-const Y_0 Real)
        (assert (>= X_0 -1.0))
        (assert (<= X_0 1.0))
        (assert (>= X_1 -1.0))
        (assert (<= X_1 1.0))
        (assert (>= X_2 -1.0))
        (assert (<= X_2 1.0))
        (assert (>= X_3 -1.0))
        (assert (<= X_3 1.0))
        (assert (>= X_4 -1.0))
        (assert (<= X_4 1.0))
        (assert {output_body})
        "
    )
}

fn parse(output_body: &str) -> ny_onnx::vnnlib::CertifiedScalarMoat {
    let (_, _, moat) =
        parse_vnnlib_with_certified_scalar_moat(&property(output_body)).expect("valid moat");
    moat
}

#[test]
fn official_2025_nch1_properties_match_sealed_hashes_and_exact_bit_oracle() {
    let cases = [
        (
            PROP0,
            "9ed547db5dd45a300e6e5e8b62439ad6f0a2268c0b27f6e654f43faea00154a8",
            0x3fe6_1760_7fff_ffff,
            0x3fe5_219e_0000_0001,
        ),
        (
            PROP1,
            "1072a88791e6f8238acf59b071c62b0c08c81ba374b8a9539af94b5606027634",
            0x3fd5_1ba2_1fff_ffff,
            0x3fd3_301d_0000_0001,
        ),
        (
            PROP2,
            "c8d593b9ee66d0cf522dbab6323d1fd8d2c56488310ef0b70fef302c6298ce77",
            0x3fe2_b671_3fff_ffff,
            0x3fe1_1cd7_a000_0000,
        ),
        (
            PROP3,
            "6d0c76c394c9b2b023cb442000b02ca14ef39d34f8bc42aaec1dacf071a10697",
            0x3fe2_4001_2000_0000,
            0x3fe1_4a3e_8000_0001,
        ),
    ];

    for (source, expected_hash, expected_high, expected_low) in cases {
        assert_eq!(format!("{:x}", Sha256::digest(source)), expected_hash);
        let (spec, input, moat) =
            parse_vnnlib_with_certified_scalar_moat(source).expect("sealed CGAN property");
        assert_eq!((spec.num_inputs, spec.num_outputs), (5, 1));
        assert_eq!(input.len(), 5);
        assert_eq!(moat.high_lower().to_bits(), expected_high);
        assert_eq!(moat.low_upper().to_bits(), expected_low);
    }
}

#[test]
fn file_loader_reads_the_same_sealed_surface() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cgan_nch1_prop1.vnnlib");
    let (_, input, moat) =
        load_vnnlib_with_certified_scalar_moat(path).expect("load sealed fixture");
    assert_eq!(input.len(), 5);
    assert_eq!(moat.high_lower().to_bits(), 0x3fd5_1ba2_1fff_ffff);
    assert_eq!(moat.low_upper().to_bits(), 0x3fd3_301d_0000_0001);
}

#[test]
fn non_dyadic_thresholds_move_exactly_one_ulp_outward() {
    let moat = parse("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))");
    assert_eq!(moat.high_lower().next_up().to_bits(), 0.1_f64.to_bits());
    assert_eq!(moat.low_upper().next_down().to_bits(), (-0.1_f64).to_bits());

    let dyadic = parse("(or (and (>= Y_0 0.5)) (and (<= Y_0 0.25)))");
    assert_eq!(dyadic.high_lower().to_bits(), 0.5_f64.to_bits());
    assert_eq!(dyadic.low_upper().to_bits(), 0.25_f64.to_bits());
}

#[test]
fn exact_equality_and_strict_output_atoms_fail_closed() {
    let equality = property("(or (and (>= Y_0 0.1)) (and (<= Y_0 0.1)))");
    assert!(parse_vnnlib_with_certified_scalar_moat(&equality).is_err());

    for output in [
        "(or (and (> Y_0 0.1)) (and (<= Y_0 -0.1)))",
        "(or (and (>= Y_0 0.1)) (and (< Y_0 -0.1)))",
    ] {
        let error = parse_vnnlib_with_certified_scalar_moat(&property(output)).unwrap_err();
        assert!(error.to_string().contains("non-strict"));
    }
}

#[test]
fn harmless_reordering_preserves_exact_threshold_bits() {
    let ordinary = parse("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))");
    let reordered = parse("(or (and (<= Y_0 -0.1)) (and (>= Y_0 0.1)))");
    assert_eq!(ordinary, reordered);

    let mut reordered_asserts = property("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))");
    let lower = "(assert (>= X_0 -1.0))";
    let upper = "(assert (<= X_0 1.0))";
    reordered_asserts = reordered_asserts
        .replacen(lower, "__LOWER__", 1)
        .replacen(upper, lower, 1)
        .replacen("__LOWER__", upper, 1);
    let (_, _, moat) = parse_vnnlib_with_certified_scalar_moat(&reordered_asserts).unwrap();
    assert_eq!(ordinary, moat);
}

#[test]
fn syntactic_mutations_outside_the_narrow_surface_fail_closed() {
    for output in [
        "(or (and (<= 0.1 Y_0)) (and (<= Y_0 -0.1)))",
        "(or (>= Y_0 0.1) (<= Y_0 -0.1))",
        "(or (and (>= Y_0 0.1) true) (and (<= Y_0 -0.1)))",
        "(and (or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1))))",
        "(or (and (= Y_0 0.1)) (and (<= Y_0 -0.1)))",
    ] {
        assert!(
            parse_vnnlib_with_certified_scalar_moat(&property(output)).is_err(),
            "mutated output unexpectedly admitted: {output}"
        );
    }

    let duplicate_declaration =
        property("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))") + "(declare-const X_0 Real)";
    assert!(parse_vnnlib_with_certified_scalar_moat(&duplicate_declaration).is_err());

    let extra_top_level = property("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))") + "(check-sat)";
    assert!(parse_vnnlib_with_certified_scalar_moat(&extra_top_level).is_err());
}

#[test]
fn duplicate_extra_and_split_output_clauses_fail_closed() {
    for output in [
        "(or (and (>= Y_0 0.1)) (and (>= Y_0 0.2)))",
        "(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)) (and (<= Y_0 -0.2)))",
    ] {
        assert!(parse_vnnlib_with_certified_scalar_moat(&property(output)).is_err());
    }

    let split = property("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))")
        + "(assert (or (and (>= Y_0 0.2)) (and (<= Y_0 -0.2))))";
    let error = parse_vnnlib_with_certified_scalar_moat(&split).unwrap_err();
    assert!(error.to_string().contains("exactly one output assertion"));
}

#[test]
fn wrong_input_or_output_indices_and_clause_local_inputs_fail_closed() {
    let two_outputs = property("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))").replace(
        "(declare-const Y_0 Real)",
        "(declare-const Y_0 Real)\n(declare-const Y_1 Real)",
    );
    assert!(parse_vnnlib_with_certified_scalar_moat(&two_outputs).is_err());

    let wrong_output = property("(or (and (>= Y_1 0.1)) (and (<= Y_1 -0.1)))");
    assert!(parse_vnnlib_with_certified_scalar_moat(&wrong_output).is_err());

    let four_inputs = property("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))")
        .replace("(declare-const X_4 Real)", "")
        .replace("(assert (>= X_4 -1.0))", "")
        .replace("(assert (<= X_4 1.0))", "");
    assert!(parse_vnnlib_with_certified_scalar_moat(&four_inputs).is_err());

    let clause_local = property("(or (and (>= X_0 -0.5) (>= Y_0 0.1)) (and (<= Y_0 -0.1)))");
    assert!(parse_vnnlib_with_certified_scalar_moat(&clause_local).is_err());
}

#[test]
fn noncanonical_scalar_variable_aliases_fail_closed() {
    let canonical = property("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))");
    for aliased in [
        canonical.replace("X_0", "X_00"),
        canonical.replace("Y_0", "Y_00"),
    ] {
        assert!(
            parse_vnnlib(&aliased).is_ok(),
            "ordinary parser behavior must remain unchanged"
        );
        assert!(
            parse_vnnlib_with_certified_scalar_moat(&aliased).is_err(),
            "certified parser unexpectedly admitted a noncanonical alias"
        );
    }
}

#[test]
fn malformed_nonfinite_and_resource_heavy_decimals_fail_closed() {
    for high in ["NaN", "inf", "1/3", "1e4097"] {
        let source = property(&format!("(or (and (>= Y_0 {high})) (and (<= Y_0 -0.1)))"));
        assert!(
            parse_vnnlib_with_certified_scalar_moat(&source).is_err(),
            "invalid decimal unexpectedly admitted: {high}"
        );
    }
}

#[test]
fn extraction_is_explicit_and_does_not_mutate_ordinary_authority_surface() {
    let source = property("(or (and (>= Y_0 0.1)) (and (<= Y_0 -0.1)))");
    let before = parse_vnnlib(&source).expect("ordinary parse before extraction");
    let (_, _, observed) =
        parse_vnnlib_with_certified_scalar_moat(&source).expect("explicit extraction");
    let after = parse_vnnlib(&source).expect("ordinary parse after extraction");

    assert_eq!(format!("{before:?}"), format!("{after:?}"));
    assert_eq!(observed.high_lower().next_up().to_bits(), 0.1_f64.to_bits());
    assert_eq!(
        observed.low_upper().next_down().to_bits(),
        (-0.1_f64).to_bits()
    );

    // The established parser remains independent: a normal non-CGAN property
    // still parses even though the explicit scalar-moat extractor refuses it.
    let ordinary_only = "
        (declare-const X_0 Real)
        (declare-const Y_0 Real)
        (assert (>= X_0 -1.0))
        (assert (<= X_0 1.0))
        (assert (>= Y_0 0.0))
    ";
    assert!(parse_vnnlib(ordinary_only).is_ok());
    assert!(parse_vnnlib_with_certified_scalar_moat(ordinary_only).is_err());
}

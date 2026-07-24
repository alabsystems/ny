// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

use trust::{cite, ensures, invariant, requires};

#[requires(value > 0)]
fn item_with_precondition(value: u32) -> u32 {
    value + 1
}

#[ensures(|result: &u32| *result == value * 2)]
fn item_with_postcondition(value: u32) -> u32 {
    value * 2
}

#[invariant(value <= 10)]
fn item_with_invariant(value: u32) -> u32 {
    value
}

#[cite(ny_contracts::example_theorem)]
fn item_with_citation() -> &'static str {
    "preserved"
}

#[test]
fn every_compatibility_attribute_preserves_its_item() {
    assert_eq!(item_with_precondition(4), 5);
    assert_eq!(item_with_postcondition(6), 12);
    assert_eq!(item_with_invariant(8), 8);
    assert_eq!(item_with_citation(), "preserved");
}

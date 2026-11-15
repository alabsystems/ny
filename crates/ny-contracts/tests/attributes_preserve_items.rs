// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

use trust::{cite, ensures, requires};

#[requires(value > 0)]
fn item_with_precondition(value: u32) -> u32 {
    value + 1
}

#[ensures(|result: &u32| *result == value * 2)]
fn item_with_postcondition(value: u32) -> u32 {
    value * 2
}

// `#[invariant(...)]` on an ITEM is no longer a contract surface. The compiler
// used to accept it and silently discard it — it produced no obligation and no
// diagnostic, so the test "proved" that an attribute stating nothing preserved
// its item. Since trust 11aae56c6f it is refused outright (an invariant must
// name the loop it constrains: `while c invariant p { … }` / `loop invariant p`).
// The case is gone, not rewritten: there is no item-level spelling that means
// what it claimed to.

#[cite(ny_contracts::example_theorem)]
fn item_with_citation() -> &'static str {
    "preserved"
}

#[test]
fn every_compatibility_attribute_preserves_its_item() {
    assert_eq!(item_with_precondition(4), 5);
    assert_eq!(item_with_postcondition(6), 12);
    assert_eq!(item_with_citation(), "preserved");
}

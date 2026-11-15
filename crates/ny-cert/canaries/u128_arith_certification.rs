#![feature(register_tool)]
#![register_tool(trust)]
#![allow(dead_code)]

// R165: WHICH u128 construct defeats kernel certification?
// The solver PROVES w128_guarded; it is the clean-CIC certificate that is missing
// ("solver trust-full-verifier reported Proved without exact kernel/native proof
// authority"). Isolate the construct.

// A: u128 ORDER only — no arithmetic at all.
pub fn a_order_only(x: u128, lo: u128) -> u128 {
    if x < lo { lo } else { assert!(x >= lo); x }
}

// B: u128 NON-wrapping arithmetic (checked, no overflow obligation).
pub fn b_checked(x: u128, y: u128) -> u128 {
    let c = x.checked_add(y).unwrap_or(0);
    if c < x { x } else { assert!(c >= x); c }
}

// C: u128 WRAPPING arithmetic — the range_i128 shape.
pub fn c_wrapping(x: u128, y: u128) -> u128 {
    let c = x.wrapping_add(y);
    if c < x { x } else { assert!(c >= x); c }
}

// D: u64 wrapping, widened to u128 only AFTER the arithmetic.
pub fn d_u64_then_widen(x: u64, y: u64) -> u128 {
    let c = x.wrapping_add(y);
    if c < x { u128::from(x) } else { assert!(c >= x); u128::from(c) }
}

#![feature(register_tool)]
#![register_tool(trust)]
#![allow(dead_code)]

// R163: does a WIDTH>64 wrapping op in the PATH CONDITION make a guarded assert
// unprovable, while the identical shape at width 64 proves?
//
// R162 claims range_i128's three arms are each trivially provable from their own
// guards, and fail only because every arm's path condition mentions `candidate`,
// built by a u128 wrapping_add the vcgen body-defs encoding cannot represent.
// If that is right, the ONLY difference below (u64 vs u128) flips the verdict.

// CONTROL: width 64. The assert follows directly from the negated guard.
pub fn w64_guarded(lo: u64, off: u64) -> u64 {
    let c = lo.wrapping_add(off);
    if c < lo {
        lo
    } else {
        assert!(c >= lo);
        c
    }
}

// EXPERIMENT: width 128, byte-for-byte the same shape.
pub fn w128_guarded(lo: u128, off: u128) -> u128 {
    let c = lo.wrapping_add(off);
    if c < lo {
        lo
    } else {
        assert!(c >= lo);
        c
    }
}

// SECOND CONTROL: width 128 but the assert does NOT depend on the wrapping value —
// isolates "128-bit arithmetic anywhere in the body" from "128-bit value in the
// path condition of the obligation".
pub fn w128_independent(lo: u128, off: u128) -> u128 {
    let c = lo.wrapping_add(off);
    assert!(lo == lo);
    c
}

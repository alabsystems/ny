#![feature(register_tool)]
#![register_tool(trust)]
#![allow(dead_code)]

// R142 CANARY — written BEFORE any fix, which is the whole point.
//
// `copy_nonoverlapping` with a SYMBOLIC element count is the #2 blocker on the
// ny-cert survivor pre-filter: 610 of 5123 attributable bundle-member drops
// (11.9%), measured in R140. It is refused at
// trust-ir-bridge/src/lower.rs::lower_copy_nonoverlapping, and the refusal's
// reasoning is CORRECT, not a gate to relax:
//
//   "the BMC lane's memory is an SMT array: a single Store at a symbolic index
//    leaves every OTHER index holding its pre-copy value, so a later read of an
//    element the real copy overwrote would return a STALE value — a false-proof
//    generator"
//
// That is not hypothetical. Trust's own history records a 64-byte copy into an
// 8-byte array verifying silently on the strength of an earlier version of that
// comment (the 810baa6c4c revert), and this is the same family as incident #4 in
// reports/false-proof-pattern-2026-08-11.md.
//
// So this file is not a demonstration that something is broken. It is the pin a
// future fix has to satisfy. ANY change that makes `symbolic_copy_then_read`
// lower MUST leave `stale_read_must_not_prove` UNPROVED. If it ever reports
// PROVED, the model has kept a pre-copy value the real program overwrote, and
// the fix is a false-proof generator regardless of how it was argued.
//
// The only sound shapes for a symbolic count are (a) refuse, as today, or (b) a
// bulk region havoc in BOTH backends. Note that lower_copy_nonoverlapping's own
// comment says the CHC lane already could do this via invalidate_store_targets,
// and that it is specifically the SMT-array lane that lacks the primitive — so a
// CHC-only fix would pass a CHC-only test while staying unsound in the other
// lane. Any candidate must be exercised in both.

// BASELINE: a literal count within the unroll budget lowers precisely today.
// This is the liveness control — if it stops proving, the harness broke and the
// anti-controls below are passing vacuously.
pub fn literal_copy_then_read(src: &[u8; 4]) -> u8 {
    let mut dst = [0u8; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), 4);
    }
    dst[0]
}

// THE SHAPE UNDER TEST. Refused today (non-literal count). A fix makes it lower.
pub fn symbolic_copy_then_read(src: &[u8], dst: &mut [u8; 8], n: usize) -> u8 {
    if n <= 8 && n <= src.len() {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n);
        }
    }
    dst[0]
}

// THE FALSE-PROOF ANTI-CONTROL. `dst` is zeroed, then a symbolic-count copy may
// overwrite dst[0] with any byte from `src`. Asserting dst[0] == 0 afterwards is
// FALSE of the real program whenever n >= 1 and src[0] != 0.
//
// It can only "prove" if the model let the store at a symbolic index miss dst[0]
// and returned its stale pre-copy zero. MUST stay unknown or failed, forever.
pub fn stale_read_must_not_prove(src: &[u8], n: usize) -> u8 {
    let mut dst = [0u8; 8];
    if n <= 8 && n <= src.len() {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n);
        }
    }
    assert!(dst[0] == 0);
    dst[0]
}

// SECOND ANTI-CONTROL, the out-of-bounds direction that produced the historical
// incident: a copy whose count exceeds the destination. `n` is bounded only by
// `src`, so it may exceed dst's 8 bytes. This must never verify as safe.
pub fn oob_copy_must_not_prove(src: &[u8], n: usize) -> u8 {
    let mut dst = [0u8; 8];
    if n <= src.len() {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n);
        }
    }
    dst[0]
}

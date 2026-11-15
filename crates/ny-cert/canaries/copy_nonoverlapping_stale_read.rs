//! R129 CANARY (false-proof incident #4 class): `copy_nonoverlapping` with a
//! NON-LITERAL element count. A model that havocs FEWER elements than the copy
//! can write — or none — lets `stale_after_copy`'s assert falsely PROVE that
//! `dst[3]` kept its pre-copy value. The required verdicts, in every world:
//!
//! * `stale_after_copy`: the `dst[3] == 1` assert must NEVER prove
//!   (`n` may be 4, so `dst[3]` may be 7). unknown/refuted are both sound.
//! * `untouched_neighbour`: the `keep == 9` assert SHOULD prove under a
//!   whole-destination-havoc model (the copy cannot write `keep`), and MAY
//!   stay unknown under the fail-closed refusal — but must never fail.
//!
//! Run: rustc -Ztrust-verify=on --crate-type lib <this file>

pub fn stale_after_copy(n: usize) -> u8 {
    let src = [7u8; 4];
    let mut dst = [1u8; 4];
    let n = if n > 4 { 4 } else { n };
    unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n) };
    assert!(dst[3] == 1);
    dst[3]
}

pub fn untouched_neighbour(n: usize) -> u8 {
    let src = [7u8; 4];
    let mut dst = [1u8; 4];
    let keep: u8 = 9;
    let n = if n > 4 { 4 } else { n };
    unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n) };
    assert!(keep == 9);
    keep
}

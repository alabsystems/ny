#![feature(register_tool)]
#![register_tool(trust)]
#![allow(dead_code)]

// R169 — the canary R168 SHOULD have had.
//
// My previous fixtures used `assert!`, producing kind=Assertion VCs. The real LCG
// blockers are kind=Postcondition rows from #[trust::ensures] (R154 CERTIFY_MISS
// histogram says so). So those fixtures exercised a different path and could never
// falsify a fix for the real case. This one reproduces the ACTUAL shape:
// range_i128's branch-clamp under an #[ensures] postcondition.

// TARGET: the exact structure of ny_cert::generate::Lcg::range_i128's tail —
// a 128-bit wrapping construction, then a three-arm branch clamp, under #[ensures].
#[trust::requires(lo <= hi)]
#[trust::ensures(move |r: &i128| lo <= *r && *r <= hi)]
pub fn clamp_u128(lo: i128, hi: i128, off: u128) -> i128 {
    let candidate = (lo as u128).wrapping_add(off) as i128;
    if candidate < lo {
        lo
    } else if candidate > hi {
        hi
    } else {
        candidate
    }
}

// CONTROL: identical clamp, NO 128-bit arithmetic. If this certifies and the one
// above does not, the discriminator is confirmed on the POSTCONDITION path.
#[trust::requires(lo <= hi)]
#[trust::ensures(move |r: &i64| lo <= *r && *r <= hi)]
pub fn clamp_u64(lo: i64, hi: i64, off: u64) -> i64 {
    let candidate = (lo as u64).wrapping_add(off) as i64;
    if candidate < lo {
        lo
    } else if candidate > hi {
        hi
    } else {
        candidate
    }
}

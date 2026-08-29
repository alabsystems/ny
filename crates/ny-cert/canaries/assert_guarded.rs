//! PERTURBATION: identical, but the caller GUARDS the divisor.
//! Must PROVE -- otherwise the probe above is red for the wrong reason.
#![allow(dead_code)]
pub fn divide(a: u32, d: u32) -> u32 { a / d }
pub fn caller(a: u32, d: u32) -> u32 { if d == 0 { 0 } else { divide(a, d) } }

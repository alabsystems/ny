//! INCIDENT #5 REGRESSION PROBE (same-crate bundled callee).
//! `divide` contains a reachable Assert (d != 0). `caller` calls it with an
//! unconstrained divisor and MUST therefore be refused. If `caller` verifies,
//! the incident-#5 mechanism -- an Assert in a bundled callee minting no
//! obligation for the caller -- is live for same-crate callees.
#![allow(dead_code)]
pub fn divide(a: u32, d: u32) -> u32 { a / d }
pub fn caller(a: u32, d: u32) -> u32 { divide(a, d) }

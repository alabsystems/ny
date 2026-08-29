//! Does the CASCADE reproduce minimally? `helper` is same-crate and lowers fine
//! on its own, but calls Vec::push -> RawVec (probe-fails). If `caller` reports
//! an absent-callee row naming `helper`, that is ny-cert's exact 1504-drop shape.
#![allow(dead_code)]
pub fn helper(v: &mut Vec<u32>, x: u32) -> u32 { v.push(x); v.len() as u32 }
pub fn caller(v: &mut Vec<u32>, x: u32) -> u32 { helper(v, x) }

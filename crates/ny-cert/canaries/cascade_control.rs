//! CONTROL: identical call shape, helper touches no allocator path.
//! Must PROVE, else the reproduction above is red for the wrong reason.
#![allow(dead_code)]
pub fn helper(x: u32) -> u32 { x.wrapping_add(1) }
pub fn caller(x: u32) -> u32 { helper(x) }

//! PROBE: does transmuting a possibly-null raw pointer into NonNull make the
//! verifier BELIEVE non-nullness? `p` is an unconstrained parameter and may be
//! null, so the assert MUST NOT be provable. If it verifies, ptr_to_nonnull is
//! a grant whose premise was asserted by comment rather than checked.
#![allow(dead_code)]
use std::ptr::NonNull;
pub unsafe fn probe(p: *mut u32) {
    let nn: NonNull<u32> = std::mem::transmute(p);
    assert!(!nn.as_ptr().is_null());
}

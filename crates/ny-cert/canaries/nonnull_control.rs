//! CONTROL: the same assert WITHOUT the NonNull transmute. Must also be
//! unprovable. If the control proves, the probe measures nothing.
#![allow(dead_code)]
pub unsafe fn control(p: *mut u32) {
    assert!(!p.is_null());
}

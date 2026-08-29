//! POSITIVE CONTROL / PERTURBATION — same TLS indirect-call shape, panic REMOVED.
//! Today it cannot be proved. If it is STILL unproved after the fix, then
//! "the negative canary stayed red" is VACUOUS (red for the wrong reason) and
//! proves nothing about the fix.
#![feature(register_tool)]
#![register_tool(trust)]
#![allow(dead_code)]
use std::cell::Cell;

thread_local! { static COUNT: Cell<u32> = const { Cell::new(0) }; }

pub trait Sink {
    #[trust::panic_free]
    fn absorb(&mut self, x: u32) -> u32;
}

pub struct Total;
impl Sink for Total {
    fn absorb(&mut self, x: u32) -> u32 {
        COUNT.try_with(|c| c.get().wrapping_add(x)).unwrap_or(0)
    }
}

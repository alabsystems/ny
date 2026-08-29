//! NEGATIVE CANARY — MUST BE REFUSED.
//! `absorb` is `#[trust::panic_free]` and its impl panics on a REACHABLE path
//! behind a TLS indirect call. If the indirect-call arg/sig fix ever lets this
//! verify, that is a false proof of exactly the incident #1/#2 shape.
#![feature(register_tool)]
#![register_tool(trust)]
#![allow(dead_code)]
use std::cell::Cell;

thread_local! { static FLAG: Cell<u32> = const { Cell::new(0) }; }

pub trait Sink {
    #[trust::panic_free]
    fn absorb(&mut self, x: u32) -> u32;
}

pub struct Boom;
impl Sink for Boom {
    fn absorb(&mut self, x: u32) -> u32 {
        FLAG.try_with(|f| {
            f.set(x);
            if f.get() == 7 {
                panic!("reachable panic behind a TLS indirect call");
            }
            f.get()
        })
        .unwrap_or(0)
    }
}

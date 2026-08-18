// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cpu-sound-f64-engine: the CPU f64 floor is a FLOOR, never a preemption.
//!
//! This recovers a pin the in-crate test had to give up. Its sibling in
//! `faer_parallelism.rs` records two failed attempts at asserting the
//! never-preempt property from inside the lib-test binary and now asserts only
//! the order-independent part, noting that never-preempt is "guaranteed by
//! construction rather than by this test". Construction is a good argument, but
//! it is not a test — this is the test.
//!
//! It works because it gets a PROCESS of its own, which is the one thing the
//! lib-test binary cannot offer. The factory and engine slots are process-global
//! `OnceLock`s: `set_sound_f64_gemm_factory` is `let _ = FACTORY.set(..)`, a
//! silent no-op once taken, and `with_engine`/`with_engine_deadline` materialize
//! `ENGINE` exactly once from whatever factory is registered at that instant.
//! `with_engine_deadline` sits on a dozen CROWN/BaB paths (`root.rs`,
//! `image.rs`, `crown_single.rs`, `atomic_cuda_*`), so among ~9800 lib tests
//! many reach it first and the sentinel never runs. Here nothing has, and the
//! opening assertion below fails loudly if that ever stops being true.
//!
//! So this file must stay the ONLY test in this binary. A second test sharing
//! the process would reintroduce exactly the race the sibling documents.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ny_propagate::faer_parallelism::install_cpu_sound_f64_gemm_engine_if_absent;
use ny_propagate::sound_f64_gemm::{
    force_engine_materialization_for_test, is_installed, set_sound_f64_gemm_factory,
};

/// The floor installs, is idempotent, and — the load-bearing part — never
/// displaces a factory that registered first. Codex review flagged that nothing
/// pinned either property.
///
/// The ordering claim is structural, not timing-dependent:
/// `install_cpu_sound_f64_gemm_engine_if_absent` tests FACTORY registration via
/// `is_installed()`, not engine materialization, so a CUDA build's lazily
/// registered cuBLAS factory is already visible to it and it returns without
/// touching it. This pins that mechanism.
#[test]
fn cpu_sound_f64_floor_installs_idempotently_and_never_preempts() {
    // Anti-vacuous-green guard. Every assertion below is only meaningful in a
    // pristine process; if something registered first, the test would be
    // asserting about a factory it does not own. Fail loudly instead.
    assert!(
        !is_installed(),
        "this test must own a pristine process — something registered a sound-f64 \
         factory before it ran, so the pin below would not be testing anything"
    );

    // A sentinel factory standing in for cuBLAS having registered first.
    let sentinel_seen = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&sentinel_seen);
    set_sound_f64_gemm_factory(move || {
        seen.store(true, Ordering::SeqCst);
        None
    });
    assert!(is_installed(), "the sentinel factory must register");

    // The floor must now be a no-op: first installation wins.
    install_cpu_sound_f64_gemm_engine_if_absent();
    install_cpu_sound_f64_gemm_engine_if_absent();
    assert!(
        is_installed(),
        "installer must leave the pre-existing factory installed"
    );

    // And it must still be the SENTINEL, not the faer floor: force
    // materialization and observe which factory ran.
    force_engine_materialization_for_test();
    assert!(
        sentinel_seen.load(Ordering::SeqCst),
        "the pre-registered factory must be the one materialized — the CPU floor \
         must never preempt an accelerator that registered first"
    );
}

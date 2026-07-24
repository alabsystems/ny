// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Process-/thread-local gate for the normalization → Linear "L2 / Cauchy–Schwarz"
//! tightening lever.
//!
//! ## What the lever does
//!
//! Normalization IBP (RMSNorm / LayerNorm Standard) attaches a per-slice
//! Euclidean-ball ([`ny_tensor::L2Constraint`]) to its output, and the
//! immediately-downstream `Linear` IBP intersects its decorrelated box bound with
//! the exact Cauchy–Schwarz row bound implied by that sphere. The intersection
//! **only ever tightens** — attaching or dropping the annotation, or skipping the
//! intersection, is always sound (it can only fall back to the box bound).
//!
//! ## Why it needs a gate
//!
//! A single plain `graph.propagate_ibp()` pass benefits from the lever and pays
//! its cost (allocating an origin-/beta-centred center vector + radius per
//! normalization output, cloning it through `BoundedTensor`, and an `O(out·in)`
//! nominal `W·center` per downstream Linear) exactly once. But alpha-/beta-CROWN
//! re-run IBP forward bound collection *many* times — once per reference-bound
//! pass, per transformer block, and per intermediate-bound recomputation — and
//! each pass re-pays that cost while the CROWN backward relaxations don't even
//! consume the sphere. On deep CROWN tests this was a 15×+ slowdown (effectively
//! a hang) for zero verification benefit.
//!
//! Gating restricts the (sound, tighten-only) intersection to the top-level plain
//! IBP pass and makes it **inert** during any iterative CROWN bound
//! recomputation. Disabling it anywhere is always sound; the only effect is the
//! loss of an optional tightening that CROWN never relied on.
//!
//! ## Mechanism & threading
//!
//! A `thread_local` `Cell<bool>`, **DEFAULT ON**, so a plain `propagate_ibp`
//! invoked on any thread keeps the lever. CROWN/alpha-CROWN/beta-CROWN entry
//! points wrap their work in [`L2LeverGuard::disabled`], an RAII guard that sets
//! the gate OFF for that scope and restores the previous value on drop
//! (panic-safe via `Drop`, and correctly nested via save/restore).
//!
//! The lever fires inside `Linear`/normalization IBP **on whichever thread runs
//! that IBP forward pass**. The CROWN-internal IBP forward passes that caused the
//! regression run on the *driver* thread inside the CROWN entry-point scope, so a
//! thread-local guard set there covers them. CROWN also fans out CROWN-backward
//! work onto rayon workers (`spsa.rs`, beta-CROWN input-split, etc.); a thread
//! local set on the driver is **not** inherited by a fresh rayon worker. Those
//! worker closures already construct a [`crate::faer_parallelism::RayonTaskGuard`]
//! (to force faer to `Par::Seq`), and that guard *also* disables this lever for
//! the worker-task scope — so any IBP forward that happens to run on a
//! CROWN-spawned rayon worker is covered too. A worker that is never touched
//! reads the default (ON), which is still sound.
//!
//! We use a thread-local (not a global `AtomicBool`) so that a plain
//! `graph.propagate_ibp()` running concurrently on another thread (e.g. an
//! independent verification job) is unaffected by a CROWN pass elsewhere.

use std::cell::Cell;

thread_local! {
    /// Whether the L2/Cauchy–Schwarz tightening lever is active on this thread.
    /// DEFAULT ON so the plain top-level IBP pass keeps the lever.
    static L2_LEVER_ACTIVE: Cell<bool> = const { Cell::new(true) };
}

/// Whether the L2/Cauchy–Schwarz lever should be applied on the current thread.
///
/// Read by normalization IBP (before attaching the [`ny_tensor::L2Constraint`])
/// and by Linear IBP (before the Cauchy–Schwarz intersection). When this returns
/// `false` the behavior is byte-identical to the pre-lever code: no constraint is
/// attached and no intersection is attempted.
#[inline]
pub(crate) fn l2_lever_active() -> bool {
    L2_LEVER_ACTIVE.with(Cell::get)
}

/// Set the gate, returning the previous value (for save/restore nesting).
#[inline]
fn set_l2_lever_active(active: bool) -> bool {
    L2_LEVER_ACTIVE.with(|c| c.replace(active))
}

/// RAII guard that turns the L2 lever OFF for its scope and restores the previous
/// value on drop. Panic-safe (restore runs in `Drop`) and correctly nested.
#[must_use = "hold the guard for the full CROWN scope; dropping it re-enables the lever"]
pub(crate) struct L2LeverGuard {
    previous: bool,
}

impl L2LeverGuard {
    /// Disable the lever for this scope (used at CROWN/alpha-/beta-CROWN entry).
    #[inline]
    pub(crate) fn disabled() -> Self {
        Self {
            previous: set_l2_lever_active(false),
        }
    }
}

impl Drop for L2LeverGuard {
    #[inline]
    fn drop(&mut self) {
        set_l2_lever_active(self.previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_on() {
        // Fresh thread default — spawn so we don't inherit a guard from another test.
        let active = std::thread::spawn(l2_lever_active).join().unwrap();
        assert!(active, "L2 lever must default ON so plain IBP keeps it");
    }

    #[test]
    fn guard_disables_and_restores() {
        std::thread::spawn(|| {
            assert!(l2_lever_active());
            {
                let _g = L2LeverGuard::disabled();
                assert!(!l2_lever_active());
            }
            assert!(l2_lever_active(), "must restore ON after guard drops");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn guard_is_correctly_nested() {
        std::thread::spawn(|| {
            let _outer = L2LeverGuard::disabled();
            assert!(!l2_lever_active());
            {
                let _inner = L2LeverGuard::disabled();
                assert!(!l2_lever_active());
            }
            // Restoring the inner guard restores the OUTER's value (still OFF),
            // not the global default.
            assert!(!l2_lever_active(), "nested restore must keep outer OFF");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn guard_restores_on_panic() {
        std::thread::spawn(|| {
            let result = std::panic::catch_unwind(|| {
                let _g = L2LeverGuard::disabled();
                assert!(!l2_lever_active());
                panic!("boom");
            });
            assert!(result.is_err());
            // Drop ran during unwind → gate restored to ON.
            assert!(
                l2_lever_active(),
                "panic-safety: lever restored after unwind"
            );
        })
        .join()
        .unwrap();
    }
}

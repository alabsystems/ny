// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-`GraphNetwork` identity token.
//!
//! HISTORY: this module used to also host the experimental Certified Cut-CROWN
//! C2 λ-fold (`docs/CERTIFIED_CUT_CROWN_DESIGN.md` §C2) — a process-global
//! `NY_CUT_FOLD` registry plus a `CutFoldPolicy` seam threaded into every ReLU
//! backward dispatch. That implementation was DELETED: its proof-authority gate
//! was a hard `false`, so the fold was statically unreachable from every bound
//! that can carry a verdict, and the arithmetic behind the gate was
//! experiment-grade (plain f32 additions, no directed rounding, no widened
//! `lower_a_err`). Keeping an unreachable non-enclosing bound transform inside
//! the CROWN backward was pure false-proof surface. Any future certified fold
//! must be derived with the outward-rounding IBP machinery and carry a proof,
//! not be re-armed from process-global mutable state.
//!
//! What survives is only the identity token [`CutFoldScope`], which the rest of
//! the crate uses to answer "is this the same graph instance?" — conflict-clause
//! replay scoping, the MIP leaf run scope, and the resident root-patch cache
//! key. It carries no cut semantics of its own.

use std::sync::atomic::{AtomicU64, Ordering};

/// Identity of ONE `GraphNetwork` instance.
///
/// Minted once per `GraphNetwork::new()`. `Clone`d graphs share the token while
/// they remain semantically identical (a configured clone has the same model,
/// and BaB sub-domain boxes are subsets of the parent box). Every structural
/// graph mutation and output retarget mints a fresh token, so a mutated clone is
/// never mistaken for its source model by a scope-keyed consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CutFoldScope(u64);

fn next_scope_counter(current: u64) -> u64 {
    current
        .checked_add(1)
        .expect("CutFoldScope identity space exhausted")
}

impl CutFoldScope {
    /// Mint a fresh, process-unique scope token.
    ///
    /// Exhaustion panics before modifying the counter.  Silent wrap would let a
    /// newly loaded graph reuse an old model's verdict-adjacent authority.
    pub fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let mut current = NEXT.load(Ordering::Relaxed);
        loop {
            let next = next_scope_counter(current);
            match NEXT.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Self(current),
                Err(observed) => current = observed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_counter_refuses_wrap_before_reuse() {
        assert_eq!(next_scope_counter(0), 1);
        assert!(std::panic::catch_unwind(|| next_scope_counter(u64::MAX)).is_err());
    }

    #[test]
    fn fresh_scopes_are_distinct() {
        assert_ne!(CutFoldScope::fresh(), CutFoldScope::fresh());
    }
}

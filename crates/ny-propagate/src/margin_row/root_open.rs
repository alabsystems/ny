// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-open-disjuncts (VIBECHECK_PLAN §6.V3): the margin-row lane's
//! certified ROOT per-class bounds, published for the attack-only
//! open-disjunct stage (`ny-cli` `beta_crown::verify::attack_open_disjuncts`).
//!
//! # What is published
//!
//! `dj[k]` is the lane's own certified lower bound on `Y_t - Y_{adv[k]}` over
//! the lane's FULL input box: exactly the vector `bab::root_eval` returns and
//! the tree consumes (outward rounding; `root_eval` already refuses a
//! non-finite component). Only evaluations of the OUTER root frame publish:
//!
//! * `bab::root_eval_store`, reached only through
//!   `rebake::certified_root_bound` — the alpha guard's published-frame
//!   baseline/after evaluations and every rebake round. All of them evaluate
//!   the lane's TOP-LEVEL root (built from the lane's `lo`/`hi`), and the
//!   guard's baseline is the EARLIEST certified root the lane produces
//!   (before the alpha ascent spends its slice);
//! * the outer frame (`EpochFrameRoute::Outer`) of `run_inner` /
//!   `run_classwise_inner`.
//!
//! Nested epoch subtrees are ReLU-split SUBDOMAINS — a class closed there is
//! not closed on the box — and never publish. Repeated publications for the
//! same `(t, adv)` merge by element-wise MAX: each one is a certified lower
//! bound on the same quantity over the same box, so their max is one too. A
//! publication for a different `(t, adv)` is ignored (first frame wins).
//!
//! # Who may read it
//!
//! Attack SCHEDULING only: which disjuncts the attack stage spends restarts
//! on, and in what order. Never a bound, never a gate, never a verdict. A
//! class whose certified root bound is strictly positive has no
//! counterexample anywhere in the box, so skipping it cannot lose a real
//! `sat`; a missing, late or even wrong publication can only aim the attack
//! badly (a lost +10), because every candidate still passes the unchanged
//! confirmation funnel (`re_evaluate_and_confirm`, the 1e-4 noise floor) and
//! the trusted-ORT gate before any `sat` is emitted.
//!
//! # Dark by default
//!
//! Nothing is recorded until [`arm_root_bound_publication`] is called — by
//! `vnncomp` only, and only when the typed preset key
//! `attack.open_disjunct_restrict` is true. Unarmed, every hook below is one
//! atomic load and returns.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

/// One instance's published certified root, as the attack stage reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct PublishedRootBounds {
    /// True class of the robustness property the lane proves.
    pub t: usize,
    /// Adversarial classes, aligned with [`Self::dj`].
    pub adv: Vec<usize>,
    /// Certified lower bound on `Y_t - Y_{adv[k]}` over the full input box
    /// (element-wise max over every accepted publication).
    pub dj: Vec<f64>,
    /// How many publications were merged into [`Self::dj`].
    pub publications: usize,
    /// When the first publication landed.
    pub first_published_at: Instant,
}

/// Lifecycle of the per-process publication slot.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RootBoundState {
    /// No armed lane has started (unarmed, lane not admitted, or not yet
    /// spawned). A consumer must not wait on this state.
    #[default]
    Idle,
    /// An armed lane is running and has not yet published its root.
    Pending,
    /// A certified root is available.
    Published(Arc<PublishedRootBounds>),
    /// The armed lane returned without ever publishing (compile/gate failure,
    /// deadline before the root, ...). A consumer must not wait on this state.
    Abandoned,
}

#[derive(Debug, Clone, Copy)]
enum Event<'a> {
    LaneEnter,
    LaneExit,
    Publish {
        t: usize,
        adv: &'a [usize],
        dj: &'a [f64],
        at: Instant,
    },
}

/// The whole state machine, pure so it is testable without the process
/// global. A malformed publication (empty, misaligned, or carrying a
/// non-finite component) is refused: fail closed means "no root", which the
/// consumer treats as "every disjunct is open".
fn transition(state: RootBoundState, event: Event<'_>) -> RootBoundState {
    match event {
        Event::LaneEnter => match state {
            // A second lane invocation over the same instance keeps what the
            // first one already certified.
            RootBoundState::Published(published) => RootBoundState::Published(published),
            RootBoundState::Idle | RootBoundState::Pending | RootBoundState::Abandoned => {
                RootBoundState::Pending
            }
        },
        Event::LaneExit => match state {
            RootBoundState::Pending => RootBoundState::Abandoned,
            other => other,
        },
        Event::Publish { t, adv, dj, at } => {
            if adv.is_empty() || adv.len() != dj.len() || dj.iter().any(|v| !v.is_finite()) {
                return state;
            }
            match state {
                RootBoundState::Published(published) => {
                    if published.t == t && published.adv.as_slice() == adv {
                        let merged: Vec<f64> = published
                            .dj
                            .iter()
                            .zip(dj)
                            .map(|(&have, &new)| have.max(new))
                            .collect();
                        RootBoundState::Published(Arc::new(PublishedRootBounds {
                            t,
                            adv: published.adv.clone(),
                            dj: merged,
                            publications: published.publications.saturating_add(1),
                            first_published_at: published.first_published_at,
                        }))
                    } else {
                        RootBoundState::Published(published)
                    }
                }
                RootBoundState::Idle | RootBoundState::Pending | RootBoundState::Abandoned => {
                    RootBoundState::Published(Arc::new(PublishedRootBounds {
                        t,
                        adv: adv.to_vec(),
                        dj: dj.to_vec(),
                        publications: 1,
                        first_published_at: at,
                    }))
                }
            }
        }
    }
}

static ARMED: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<RootBoundState> = Mutex::new(RootBoundState::Idle);

fn apply(event: Event<'_>) {
    let mut slot = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    let current = std::mem::take(&mut *slot);
    *slot = transition(current, event);
}

/// Arm the publication for this process. Called by `vnncomp` before any
/// margin-row lane can start, only when `attack.open_disjunct_restrict` is
/// true. There is no disarm: one scored instance is one process.
pub fn arm_root_bound_publication() {
    ARMED.store(true, Ordering::SeqCst);
}

/// Whether [`arm_root_bound_publication`] has been called.
#[must_use]
pub fn root_bound_publication_armed() -> bool {
    ARMED.load(Ordering::SeqCst)
}

/// Snapshot of the publication slot (attack scheduling only).
#[must_use]
pub fn root_bound_state() -> RootBoundState {
    STATE.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// Record one certified evaluation of the lane's OUTER root frame. No-op
/// unless armed. Callers are responsible for the outer-frame precondition
/// (see the module docs); this function never feeds anything back into the
/// lane.
pub(crate) fn publish_outer_root(t: usize, adv: &[usize], dj: &[f64]) {
    if !root_bound_publication_armed() {
        return;
    }
    apply(Event::Publish {
        t,
        adv,
        dj,
        at: Instant::now(),
    });
}

/// Brackets one lane invocation: `Pending` on entry, `Abandoned` on exit if
/// the lane never published. Inert (no state change at all) when unarmed.
pub(crate) struct LaneGuard {
    active: bool,
}

impl LaneGuard {
    pub(crate) fn enter() -> Self {
        let active = root_bound_publication_armed();
        if active {
            apply(Event::LaneEnter);
        }
        Self { active }
    }
}

impl Drop for LaneGuard {
    fn drop(&mut self) {
        if self.active {
            apply(Event::LaneExit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn publish<'a>(t: usize, adv: &'a [usize], dj: &'a [f64]) -> Event<'a> {
        Event::Publish {
            t,
            adv,
            dj,
            at: Instant::now(),
        }
    }

    fn published(state: &RootBoundState) -> &PublishedRootBounds {
        match state {
            RootBoundState::Published(p) => p,
            other => panic!("expected a published root, got {other:?}"),
        }
    }

    #[test]
    fn lane_exit_without_a_root_is_abandoned_not_pending() {
        let s = transition(RootBoundState::Idle, Event::LaneEnter);
        assert_eq!(s, RootBoundState::Pending);
        let s = transition(s, Event::LaneExit);
        assert_eq!(
            s,
            RootBoundState::Abandoned,
            "a consumer waiting on a lane that died must stop waiting"
        );
        // An exit with no matching enter changes nothing.
        assert_eq!(
            transition(RootBoundState::Idle, Event::LaneExit),
            RootBoundState::Idle
        );
    }

    #[test]
    fn a_published_root_survives_lane_exit_and_reentry() {
        let s = transition(RootBoundState::Idle, Event::LaneEnter);
        let s = transition(s, publish(3, &[0, 1, 2], &[-0.5, 0.25, -0.1]));
        let p = published(&s).clone();
        assert_eq!((p.t, p.adv.as_slice()), (3, [0usize, 1, 2].as_slice()));
        assert_eq!(p.dj, vec![-0.5, 0.25, -0.1]);
        assert_eq!(p.publications, 1);
        let s = transition(s, Event::LaneExit);
        assert_eq!(published(&s), &p);
        let s = transition(s, Event::LaneEnter);
        assert_eq!(published(&s), &p);
    }

    #[test]
    fn republishing_the_same_frame_merges_by_elementwise_max() {
        let s = transition(RootBoundState::Pending, publish(1, &[0, 2], &[-0.5, 0.1]));
        let first_at = published(&s).first_published_at;
        let s = transition(s, publish(1, &[0, 2], &[-0.7, 0.3]));
        let p = published(&s);
        // Both are certified lower bounds on the same margins over the same
        // box, so the max of the two is certified and never looser.
        assert_eq!(p.dj, vec![-0.5, 0.3]);
        assert_eq!(p.publications, 2);
        assert_eq!(p.first_published_at, first_at);
    }

    #[test]
    fn a_publication_for_a_different_frame_is_ignored() {
        let s = transition(RootBoundState::Pending, publish(1, &[0, 2], &[-0.5, 0.1]));
        let before = published(&s).clone();
        // A per-class tree (adv = [class]) or another true class must not
        // replace the joint root the stage maps clauses against.
        let s = transition(s, publish(1, &[2], &[9.0]));
        assert_eq!(published(&s), &before);
        let s = transition(s, publish(0, &[1, 2], &[9.0, 9.0]));
        assert_eq!(published(&s), &before);
    }

    #[test]
    fn malformed_publications_are_refused() {
        for (adv, dj) in [
            (&[][..], &[][..]),
            (&[0usize, 1][..], &[0.1][..]),
            (&[0usize, 1][..], &[0.1, f64::NAN][..]),
            (&[0usize, 1][..], &[f64::INFINITY, 0.1][..]),
        ] {
            assert_eq!(
                transition(RootBoundState::Pending, publish(2, adv, dj)),
                RootBoundState::Pending,
                "adv={adv:?} dj={dj:?} must not publish"
            );
        }
        // ...and a malformed republication cannot corrupt a good one.
        let good = transition(RootBoundState::Pending, publish(2, &[0, 1], &[0.1, -0.2]));
        let kept = transition(good.clone(), publish(2, &[0, 1], &[f64::NAN, 5.0]));
        assert_eq!(kept, good);
    }
}

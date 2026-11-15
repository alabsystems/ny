// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r4 — SAFE POINTS ARE ENFORCEABLE (ROUTER_PROGRAM §3.2 C8, §6.1
//! "cancellation", §6.4). PURE: no clock, no environment, no global.
//!
//! A capability declares where the router may suspend or shorten it
//! ([`SafePoints`] on its `CapabilityDecl`). Until R4 that declaration was
//! documentation; from R4 every acting site that would END a capability early
//! or hand it a SHORTER window than its own rule derives asks [`may_suspend`] /
//! [`may_shorten`] first and obeys the ruling:
//!
//! | declared | at [`At::Start`] | at [`At::Unit`] (a unit boundary) | at [`At::Inside`] (mid-unit) |
//! |---|---|---|---|
//! | `None` | refused | refused | refused |
//! | `BetweenUnits` | allowed | allowed | refused |
//!
//! `None` means NEVER: the router never suspends such a capability and never
//! hands it a window shorter than today's — it runs to the cap its own rule
//! derives (the detached margin-row workers, the graph ReLU-split BaB, the
//! STE-PGD lane's fixed anneal). A capability the registry does not declare
//! at all is treated as `None` ([`declared_or_never`]): undeclared is never a
//! licence.
//!
//! What is NOT a suspension: admitting or declining a capability before it
//! starts (CR-A, `rule.root_admission`), and extending a window (a raise is
//! never a shortening). Neither needs a safe point.
//!
//! Soundness: none of this touches a bound, a certificate or the publisher. A
//! refusal only ever keeps today's behaviour; an allowed suspension ends a
//! capability at a boundary where its own state is consistent (the lane's own
//! loop already polls its deadline there), so a stopped lane reports
//! "undecided", never a claim.

use serde::{Deserialize, Serialize};

use super::decl::SafePoints;

/// Where a capability is when the router asks to suspend or shorten it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum At {
    /// Before its first unit (its own start, when it claims its window).
    Start,
    /// At a unit boundary (between two restarts, waves, levels, iterations),
    /// with the index of the unit just completed.
    Unit(u64),
    /// Inside a unit: never a safe point.
    Inside,
}

impl At {
    /// Stable token for logs.
    #[must_use]
    pub fn token(self) -> String {
        match self {
            Self::Start => "start".to_owned(),
            Self::Unit(n) => format!("unit:{n}"),
            Self::Inside => "inside".to_owned(),
        }
    }
}

/// The ruling on one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ruling {
    /// The capability may be suspended / shortened here.
    Allowed,
    /// It may not; the string says why (`no_safe_points`, `not_a_safe_point`).
    Refused(&'static str),
}

impl Ruling {
    /// Whether the request is allowed.
    #[must_use]
    pub const fn allowed(self) -> bool {
        matches!(self, Self::Allowed)
    }

    /// Stable token for logs (`allowed` or the refusal reason).
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::Refused(why) => why,
        }
    }
}

/// Refusal: the capability declares no safe point (or is undeclared).
pub const REFUSED_NO_SAFE_POINTS: &str = "no_safe_points";
/// Refusal: the request is not at one of the capability's safe points.
pub const REFUSED_NOT_A_SAFE_POINT: &str = "not_a_safe_point";

/// May the router SUSPEND (end early) a capability declaring `declared`,
/// asked at `at`? `None` is never; `BetweenUnits` only at its own start or at
/// a unit boundary.
#[must_use]
pub const fn may_suspend(declared: SafePoints, at: At) -> Ruling {
    match declared {
        SafePoints::None => Ruling::Refused(REFUSED_NO_SAFE_POINTS),
        SafePoints::BetweenUnits => match at {
            At::Start | At::Unit(_) => Ruling::Allowed,
            At::Inside => Ruling::Refused(REFUSED_NOT_A_SAFE_POINT),
        },
    }
}

/// May the router hand a capability declaring `declared` a window of
/// `new_ns` where its own rule derives `today_ns`, asked at `at`? A window at
/// least today's is never a shortening (always allowed); a shorter one is a
/// suspension and follows [`may_suspend`].
#[must_use]
pub const fn may_shorten(declared: SafePoints, at: At, today_ns: u64, new_ns: u64) -> Ruling {
    if new_ns >= today_ns {
        Ruling::Allowed
    } else {
        may_suspend(declared, at)
    }
}

/// The declared safe points, or `None` when the capability is undeclared:
/// an undeclared capability is never suspended.
#[must_use]
pub const fn declared_or_never(declared: Option<SafePoints>) -> SafePoints {
    match declared {
        Some(points) => points,
        None => SafePoints::None,
    }
}

/// The stricter of two declarations: a lane slot several capabilities share
/// is only as suspendable as its least suspendable capability.
#[must_use]
pub const fn strictest(a: SafePoints, b: SafePoints) -> SafePoints {
    match (a, b) {
        (SafePoints::BetweenUnits, SafePoints::BetweenUnits) => SafePoints::BetweenUnits,
        _ => SafePoints::None,
    }
}

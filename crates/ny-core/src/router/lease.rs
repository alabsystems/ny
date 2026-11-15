// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r4 CR-L — the MEMORY LEASE arithmetic (ROUTER_PROGRAM §2.2
//! `HostEnvelope`, §3.2 C7, §8, F6). PURE: no clock, no environment, no
//! global; integer bytes, saturating everywhere.
//!
//! # Why
//!
//! The graph-heavy optional lanes (the detached margin-row workers, the
//! post-verifier attack tail and its graph loads, the ORT VJP refine wave) are
//! admitted today by a LIVE gate: `live cgroup/RLIMIT headroom >= 16 GiB`, and
//! `Unbounded => true`. Competition runs ny with NO cgroup, so on the scored
//! host that gate admits every lane unconditionally and nothing bounds their
//! AGGREGATE (CLAUDE.md: anon-rss 118 GB of 121 GiB, `global_oom`, on one row);
//! and under a measurement cgroup the same gate reads the live, shrinking
//! headroom, so its answer depends on how the run was launched and on what
//! the process happened to hold at that instant — not replayable, not the
//! competition's answer.
//!
//! # What
//!
//! One envelope per row, computed from the DECLARED host (`HostEnvelope`:
//! class, threads, `E_static = min(MemTotal, cgroup memory.max) − reserve`,
//! device), never from the live headroom. Every lane that allocates a big
//! working set obtains a LEASE from it before it starts:
//!
//! * the terminal (the internal verifier) holds a base lease of
//!   `E / TERMINAL_SHARE_DEN` for as long as it runs — the same half of the
//!   lease CR-K's depth lift already allows its resident queue
//!   ([`super::derive::QUEUE_LEASE_DEN`]);
//! * each optional heavy lane asks for its memory model's bytes
//!   ([`OPTIONAL_HEAVY_LEASE_BYTES`], the measured admission boundary of the
//!   live gate it replaces: a 12 GiB run entered the post-verifier tail with
//!   9.6 GiB of headroom and consumed all of it, plus 6.4 GiB of margin);
//! * #kfsb-production: the kFSB single-shot candidate fan-out inside the
//!   graph ReLU-split lanes asks for the same 16 GiB model but declares a
//!   minimum of ONE simulation's share at its resident peak
//!   ([`KFSB_FANOUT_LEASE_NEED`]): the first NARROWABLE holder — narrowed, it
//!   runs fewer simulations at once;
//! * a lease is granted when the envelope minus every OUTSTANDING lease covers
//!   it, NARROWED when only the lane's declared minimum fits (a lane whose
//!   working set is indivisible declares `min == want` and is never narrowed),
//!   and DENIED otherwise — the lane then refuses, exactly as it refuses today
//!   when the live gate says no.
//!
//! The sum of outstanding leases never exceeds the envelope: that is the
//! aggregate reservation contract the live gate's own comment says the heavy
//! lanes lacked ("Keep the already-sound verifier verdict until the heavy work
//! ... gains an aggregate reservation contract").
//!
//! # Never a verdict input
//!
//! A lease is ADMISSION CONTROL: it decides whether a lane runs (or with how
//! much memory a narrowable lane runs), never what any lane computes. No byte
//! count here reaches a bound, a tolerance, an error term or the publisher.
//! The existing live envelope stays as a FLOOR at every gate (the glue checks
//! it first): a lease can only ever refuse more than today, never admit a lane
//! the kernel-enforced envelope refuses.

use serde::{Deserialize, Serialize};

use super::derive;
use super::facts::{DeviceQual, HostClass, HostEnvelope};

/// The terminal's base lease is `envelope / TERMINAL_SHARE_DEN`: the half of
/// the static lease CR-K already grants the verifier's resident queue.
pub const TERMINAL_SHARE_DEN: u64 = derive::QUEUE_LEASE_DEN;

/// One optional heavy lane's memory model, bytes: the live gate's measured
/// admission boundary (`OPTIONAL_HEAVY_MIN_HEADROOM_BYTES`, 16 GiB).
pub const OPTIONAL_HEAVY_LEASE_BYTES: u64 = 16 << 30;

/// #kfsb-production: the resident peak of an advertised dense pair, as a
/// multiple of its bytes. The dense budget a CROWN pass consults is the
/// bytes of ONE advertised coefficient pair, whose materialization can hold
/// up to four times that (ny-propagate `crown_memory`: the 7-D Patches ->
/// Dense conversion's eight same-sized matrices against the pair's two), so
/// a lease of `L` resident bytes covers dense budgets summing to `L / 4`.
pub const DENSE_PAIR_PEAK_FACTOR: u64 = 4;

/// #kfsb-production: the least dense budget one kFSB candidate simulation is
/// given in a shared-budget fan-out (a fan-out runs fewer simulations at
/// once rather than give any of them less). One simulation is one
/// constrained CROWN pass seeded with ONE objective row.
pub const KFSB_FANOUT_MIN_SHARE_BYTES: u64 = 256 << 20;

/// #kfsb-production: the kFSB single-shot fan-out's memory model. It wants the
/// uniform optional-heavy model ([`OPTIONAL_HEAVY_LEASE_BYTES`]) and runs with
/// as little as ONE simulation's share at its resident peak — the first
/// NARROWABLE need: a narrowed lease runs fewer simulations at once.
pub const KFSB_FANOUT_LEASE_NEED: LeaseNeed = LeaseNeed {
    want_bytes: OPTIONAL_HEAVY_LEASE_BYTES,
    min_bytes: DENSE_PAIR_PEAK_FACTOR * KFSB_FANOUT_MIN_SHARE_BYTES,
};

/// #attack-concurrent: the concurrent attack lane's memory model, bytes. The
/// lane shares the verifier's model BY REFERENCE (no reload, no copy), so its
/// working set is one PGD restart batch of activations and gradients plus the
/// trusted gate's ONNX Runtime session at publication — megabytes on the MLP /
/// small-CNN rows the lane serves; 2 GiB is an order of magnitude above that
/// and far below [`OPTIONAL_HEAVY_LEASE_BYTES`], which models a graph RELOAD.
/// Indivisible (`min == want`): the attack cannot run narrowed.
pub const ATTACK_CONCURRENT_LEASE_BYTES: u64 = 2 << 30;

/// Number of lease holders.
pub const HOLDERS: usize = 7;

/// A lane that holds (or asks for) a memory lease. A FIXED set, so the
/// authority's ledger never allocates per holder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseHolder {
    /// The internal verifier: the terminal's base lease, held while it runs.
    Terminal,
    /// The detached concurrent margin-row worker (from spawn to its end).
    MarginRowConcurrent,
    /// The post-verifier inline margin-row lane.
    MarginRowInline,
    /// The post-verifier attack tail (its graph loads and the f64 oracle).
    PostVerifierTail,
    /// The batched ORT VJP refine wave.
    OrtRefineWave,
    /// #kfsb-production: the kFSB single-shot candidate fan-out inside the
    /// graph ReLU-split lanes (held by the lane while it runs; narrowable).
    KfsbFanout,
    /// #attack-concurrent: the concurrent upfront-attack lane (from its spawn
    /// to its join; indivisible).
    AttackConcurrent,
}

impl LeaseHolder {
    /// Every holder, in index order.
    pub const ALL: [Self; HOLDERS] = [
        Self::Terminal,
        Self::MarginRowConcurrent,
        Self::MarginRowInline,
        Self::PostVerifierTail,
        Self::OrtRefineWave,
        Self::KfsbFanout,
        Self::AttackConcurrent,
    ];

    /// Index into the per-holder arrays.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Stable token for logs and records.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Terminal => "terminal",
            Self::MarginRowConcurrent => "margin_row_concurrent",
            Self::MarginRowInline => "margin_row_inline",
            Self::PostVerifierTail => "post_verifier_tail",
            Self::OrtRefineWave => "ort_refine_wave",
            Self::KfsbFanout => "kfsb_fanout",
            Self::AttackConcurrent => "attack_concurrent",
        }
    }

    /// Parse a token.
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|holder| holder.token() == token)
    }
}

/// What a holder asks for: `want` bytes, and the least it can run with
/// (`min == want`: indivisible, never narrowed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LeaseNeed {
    /// Bytes the holder's memory model asks for.
    pub want_bytes: u64,
    /// The least it can run with (`<= want_bytes`).
    pub min_bytes: u64,
}

impl LeaseNeed {
    /// An indivisible need.
    #[must_use]
    pub const fn whole(bytes: u64) -> Self {
        Self {
            want_bytes: bytes,
            min_bytes: bytes,
        }
    }

    /// Whether the holder can run narrowed.
    #[must_use]
    pub const fn narrowable(self) -> bool {
        self.min_bytes < self.want_bytes
    }
}

/// The row's lease plan: the envelope and every holder's need. Logged and
/// hashed with the commitments; replayed bit for bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LeasePlan {
    /// `E_static` of the declared host, bytes.
    pub envelope_bytes: u64,
    /// The declared host class (logged).
    pub host: HostClass,
    /// Its worker threads (logged; width is a plan value, never a decision).
    pub threads: u32,
    /// Its device qualification (logged).
    pub device: DeviceQual,
    /// Whether the host was declared (a measurement target) or read.
    pub declared: bool,
    /// Every holder's need, in [`LeaseHolder::ALL`] order.
    pub needs: [LeaseNeed; HOLDERS],
}

impl LeasePlan {
    /// `holder`'s need.
    #[must_use]
    pub fn need(&self, holder: LeaseHolder) -> LeaseNeed {
        self.needs
            .get(holder.index())
            .copied()
            .unwrap_or(LeaseNeed::whole(u64::MAX))
    }

    /// The terminal's base lease, bytes.
    #[must_use]
    pub fn terminal_bytes(&self) -> u64 {
        self.need(LeaseHolder::Terminal).want_bytes
    }
}

/// The lease plan for a host, or `None` when its envelope is unknown (`0`):
/// with no envelope there is nothing to lease against, and the lanes keep
/// today's live gate.
#[must_use]
pub fn lease_plan(host: &HostEnvelope) -> Option<LeasePlan> {
    let envelope = host.envelope_bytes();
    if envelope == 0 {
        return None;
    }
    let terminal = envelope / TERMINAL_SHARE_DEN;
    let mut needs = [LeaseNeed::whole(OPTIONAL_HEAVY_LEASE_BYTES); HOLDERS];
    if let Some(slot) = needs.get_mut(LeaseHolder::Terminal.index()) {
        *slot = LeaseNeed::whole(terminal);
    }
    if let Some(slot) = needs.get_mut(LeaseHolder::KfsbFanout.index()) {
        *slot = KFSB_FANOUT_LEASE_NEED;
    }
    if let Some(slot) = needs.get_mut(LeaseHolder::AttackConcurrent.index()) {
        *slot = LeaseNeed::whole(ATTACK_CONCURRENT_LEASE_BYTES);
    }
    Some(LeasePlan {
        envelope_bytes: envelope,
        host: host.class,
        threads: host.threads,
        device: host.device,
        declared: host.declared,
        needs,
    })
}

/// One lease decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseDecision {
    /// The whole need fits.
    Granted {
        /// Bytes leased.
        bytes: u64,
    },
    /// Only the holder's declared minimum (or more) fits: it runs narrowed.
    Narrowed {
        /// Bytes leased (`min <= bytes < want`).
        bytes: u64,
        /// What it wanted.
        want: u64,
    },
    /// Nothing fits: the holder refuses (today's refusal path).
    Denied {
        /// Bytes still available when it asked.
        available: u64,
        /// What it wanted.
        want: u64,
    },
}

impl LeaseDecision {
    /// Whether the holder may run (granted or narrowed).
    #[must_use]
    pub const fn admits(self) -> bool {
        !matches!(self, Self::Denied { .. })
    }

    /// Bytes leased (0 when denied).
    #[must_use]
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Granted { bytes } | Self::Narrowed { bytes, .. } => bytes,
            Self::Denied { .. } => 0,
        }
    }

    /// Stable token for logs.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Granted { .. } => "granted",
            Self::Narrowed { .. } => "narrowed",
            Self::Denied { .. } => "denied",
        }
    }
}

/// Decide one request against `available` bytes. Pure and total: granted
/// when the whole need fits, narrowed to what is left when at least the
/// declared minimum fits (and the holder is narrowable), denied otherwise. A
/// zero-byte need is always granted (nothing to reserve).
#[must_use]
pub const fn decide(available: u64, need: LeaseNeed) -> LeaseDecision {
    let want = need.want_bytes;
    let min = if need.min_bytes < want {
        need.min_bytes
    } else {
        want
    };
    if want <= available {
        LeaseDecision::Granted { bytes: want }
    } else if min < want && min <= available {
        LeaseDecision::Narrowed {
            bytes: available,
            want,
        }
    } else {
        LeaseDecision::Denied { available, want }
    }
}

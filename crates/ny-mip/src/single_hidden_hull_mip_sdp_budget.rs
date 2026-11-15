// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Separate global-SDP engineering limits, within the SAME original clock.
use super::*;
use std::sync::{Arc, Mutex};
pub(super) const MAX_N: usize = 96;
pub(super) const MAX_WORK: usize = 256_000_000;
pub(super) const MAX_BYTES: usize = 2 << 20;
pub(super) const MAX_CANDIDATES: usize = 8;
#[derive(Debug, Default, Clone)]
pub(super) struct Counts {
    pub work: usize,
    pub iterations: usize,
    pub factors: usize,
    pub checked: usize,
    pub replays: usize,
    pub attempts: usize,
    pub active: usize,
    pub completed_replays: usize,
    pub reserved: usize,
}
#[derive(Debug, Clone)]
pub(super) struct Budget {
    pub authority: Instant,
    pub search_end: Instant,
    pub admitted_bytes: usize,
    pub proof_cost: usize,
    pub counts: Arc<Mutex<Counts>>,
}
impl Budget {
    pub(super) fn new(
        cuts: &HullRootMipCuts<'_>,
        authority: Instant,
        reserve: Duration,
        n: usize,
    ) -> Option<Self> {
        if n == 0 || n > MAX_N {
            return None;
        }
        let authority = authority.min(cuts.deadline);
        let authority =
            ny_core::instance_budget::deadline().map_or(authority, |d| d.min(authority));
        let search_end = authority
            .checked_sub(reserve.min(Duration::from_secs(2)))?
            .checked_sub(Duration::from_millis(75))?;
        if !live(search_end) {
            return None;
        }
        let p = cuts.problem;
        let cells = p.hidden_dim.checked_mul(p.input_dim)?;
        // Conservative simultaneous owned Vec/matrix/scratch payload, not RSS.
        let bytes = 128usize
            .checked_mul(n.checked_mul(n)?)?
            .checked_add(16 * cells)?
            .checked_add(64 * (p.hidden_dim + p.input_dim + n + 1))?;
        if bytes > MAX_BYTES {
            return None;
        }
        let triangle = n.checked_mul(n + 1)?.checked_mul(n + 2)? / 6;
        let proof_cost = triangle
            .checked_mul(6)?
            .checked_add(64 * n * n)?
            .checked_add(64 * (cells + p.hidden_dim + p.input_dim + 1))?;
        if proof_cost.checked_mul(2)? > MAX_WORK {
            return None;
        }
        Some(Self {
            authority,
            search_end,
            admitted_bytes: bytes,
            proof_cost,
            counts: Arc::new(Mutex::new(Counts {
                reserved: 2 * proof_cost,
                ..Counts::default()
            })),
        })
    }
    pub(super) fn charge(&self, work: usize, replay: bool) -> Option<()> {
        if !live(if replay {
            self.authority
        } else {
            self.search_end
        }) {
            return None;
        }
        let mut s = self.counts.lock().ok()?;
        let next = s.work.checked_add(work)?;
        if next > MAX_WORK || (!replay && next.checked_add(s.reserved)? > MAX_WORK) {
            return None;
        }
        s.work = next;
        if replay {
            s.reserved = s.reserved.saturating_sub(work);
        }
        Some(())
    }
    pub(super) fn proof(&self, replay: bool) -> Option<()> {
        if !live(if replay {
            self.authority
        } else {
            self.search_end
        }) {
            return None;
        }
        let mut c = self.counts.lock().ok()?;
        if (replay && c.replays >= 2) || (!replay && c.attempts >= MAX_CANDIDATES) {
            return None;
        }
        let next = c.work.checked_add(self.proof_cost)?;
        if next > MAX_WORK || (!replay && next.checked_add(c.reserved)? > MAX_WORK) {
            return None;
        }
        c.work = next;
        if replay {
            c.replays += 1;
            c.reserved = c.reserved.saturating_sub(self.proof_cost);
        } else {
            c.attempts += 1;
        }
        Some(())
    }
    pub(super) fn factor(&self, n: usize) -> Option<()> {
        self.charge(2 * n * n * n, false)?;
        self.counts.lock().ok()?.factors += 1;
        Some(())
    }
    pub(super) fn iteration(&self) -> Option<()> {
        self.charge(1, false)?;
        self.counts.lock().ok()?.iterations += 1;
        Some(())
    }
}

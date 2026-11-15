// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Explicit weighted-work units, not a primitive-operation or RSS bound.
use super::*;
use std::sync::{Arc, Mutex};

pub(in crate::single_hidden_hull::mip_cuts) const MAX_GROUP: usize = 8;
pub(in crate::single_hidden_hull::mip_cuts) const MAX_ATTEMPTS: usize = 32;
pub(in crate::single_hidden_hull::mip_cuts) const MAX_SEARCH_MASKS: usize = 8192;
pub(in crate::single_hidden_hull::mip_cuts) const MAX_SELECTED_MASKS: usize = 2048;
pub(in crate::single_hidden_hull::mip_cuts) const MAX_WORK: usize = 2_000_000;
pub(in crate::single_hidden_hull::mip_cuts) const MAX_BYTES: usize = 1 << 20;
#[derive(Debug, Default)]
pub(in crate::single_hidden_hull::mip_cuts) struct Counts {
    pub(in crate::single_hidden_hull::mip_cuts) work: usize,
    pub(in crate::single_hidden_hull::mip_cuts) masks: usize,
    pub(in crate::single_hidden_hull::mip_cuts) replay_masks: usize,
    pub(in crate::single_hidden_hull::mip_cuts) attempts: usize,
    pub(in crate::single_hidden_hull::mip_cuts) bytes: usize,
    pub(in crate::single_hidden_hull::mip_cuts) peak_bytes: usize,
    pub(in crate::single_hidden_hull::mip_cuts) reserved_work: usize,
    pub(in crate::single_hidden_hull::mip_cuts) reserved_masks: usize,
}
#[derive(Debug, Clone)]
pub(in crate::single_hidden_hull::mip_cuts) struct Budget {
    pub(in crate::single_hidden_hull::mip_cuts) authority: Instant,
    pub(in crate::single_hidden_hull::mip_cuts) search_end: Instant,
    pub(in crate::single_hidden_hull::mip_cuts) state: Arc<Mutex<Counts>>,
}
impl Budget {
    pub(in crate::single_hidden_hull::mip_cuts) fn new(
        authority: Instant,
        reserve: Duration,
    ) -> Option<Self> {
        let authority =
            ny_core::instance_budget::deadline().map_or(authority, |d| d.min(authority));
        let search_end = authority
            .checked_sub(reserve.min(Duration::from_secs(2)))?
            .checked_sub(Duration::from_millis(75))?;
        live(search_end).then(|| Self {
            authority,
            search_end,
            state: Arc::default(),
        })
    }
    pub(in crate::single_hidden_hull::mip_cuts) fn attempt(&self) -> Option<()> {
        if !live(self.search_end) {
            return None;
        }
        let mut s = self.state.lock().ok()?;
        if s.attempts >= MAX_ATTEMPTS {
            return None;
        }
        s.attempts += 1;
        Some(())
    }
    pub(in crate::single_hidden_hull::mip_cuts) fn charge(
        &self,
        work: usize,
        masks: usize,
        replay: bool,
    ) -> Option<()> {
        if !live(if replay {
            self.authority
        } else {
            self.search_end
        }) {
            return None;
        }
        let mut s = self.state.lock().ok()?;
        let next = s.work.checked_add(work)?;
        if replay {
            let m = s.replay_masks.checked_add(masks)?;
            if next > MAX_WORK || m > 2 * MAX_SELECTED_MASKS {
                return None;
            }
            s.reserved_work = s.reserved_work.saturating_sub(work);
            s.reserved_masks = s.reserved_masks.saturating_sub(masks);
            s.replay_masks = m;
        } else {
            let m = s.masks.checked_add(masks)?;
            if next.checked_add(s.reserved_work)? > MAX_WORK || m > MAX_SEARCH_MASKS {
                return None;
            }
            s.masks = m;
        }
        s.work = next;
        Some(())
    }
    pub(in crate::single_hidden_hull::mip_cuts) fn reserve(
        &self,
        work: usize,
        masks: usize,
    ) -> Option<()> {
        let mut s = self.state.lock().ok()?;
        let work = work.checked_mul(2)?;
        let masks = masks.checked_mul(2)?;
        if masks > 2 * MAX_SELECTED_MASKS
            || s.work.checked_add(work)? > MAX_WORK
            || s.replay_masks.checked_add(masks)? > 2 * MAX_SELECTED_MASKS
        {
            return None;
        }
        s.reserved_work = work;
        s.reserved_masks = masks;
        Some(())
    }
    pub(in crate::single_hidden_hull::mip_cuts) fn memory(&self, bytes: usize) -> Option<Permit> {
        let mut s = self.state.lock().ok()?;
        let total = s.bytes.checked_add(bytes)?;
        if total > MAX_BYTES {
            return None;
        }
        s.bytes = total;
        s.peak_bytes = s.peak_bytes.max(total);
        Some(Permit {
            budget: self.clone(),
            bytes,
        })
    }
}
#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts) struct Permit {
    pub(in crate::single_hidden_hull::mip_cuts) budget: Budget,
    bytes: usize,
}
impl Drop for Permit {
    fn drop(&mut self) {
        if let Ok(mut s) = self.budget.state.lock() {
            s.bytes = s.bytes.saturating_sub(self.bytes);
        }
    }
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #gather-score (boxlift charter Inc 4 — DARK, `NY_MO_GATHER_SCORE=1`):
//! advisory branch-candidate scores harvested from the wide-β lane's
//! ALREADY-PAID `A_lower` gather.
//!
//! Per β iteration the wide backward returns, for every domain, the
//! pre-relaxation lower-A values at the UNION of split columns across the
//! batch (`wide_gathers`). For domain `d`, columns split by SIBLING domains
//! are exactly the kFSB-grade branch candidates `d` has not split yet, and
//! `|A_lower[crit_row, col]|` is the classic sensitivity surrogate — at ZERO
//! added backward passes (the gather is materialized for the β gradient
//! anyway). This module stores those scores per domain, keyed by the domain's
//! SPLIT-SET fingerprint (a ReLU-split domain is identified by its history;
//! the β entries encode it on both the producer and consumer sides), and the
//! branch selector consults them behind the dark gate.
//!
//! ADVISORY-ONLY by construction: scores can only reorder which unstable
//! neuron is split next — the split itself remains an exact partition either
//! way, so verdicts and bounds stay sound regardless of score quality. A
//! cache miss, an empty intersection with the caller's unstable set, or the
//! gate being off all fall back to the shipped scorer byte-identically.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::beta_crown::state::GraphBetaState;

/// One harvested candidate: `(relu node name, neuron column, |A_lower|)`.
pub(crate) type GatherScoreRow = (String, u32, f32);

/// Hard cap on cached domains — a runaway BaB tree must not grow the cache
/// unboundedly. On overflow the cache CLEARS (advisory data; losing it only
/// reverts candidates to the shipped scorer).
const GATHER_SCORE_CACHE_CAP: usize = 8192;

/// Process-unique identity for a deferred-write frame.
///
/// Cache addresses are not sufficient: two nested stages for the same cache
/// have the same address and may finish or unwind out of stack order.  A
/// checked monotonic token makes exact-frame removal unambiguous. Exhaustion
/// fails closed instead of wrapping and reusing an identity.
static NEXT_GATHER_STAGE_TOKEN: AtomicU64 = AtomicU64::new(1);

fn next_gather_stage_token() -> u64 {
    NEXT_GATHER_STAGE_TOKEN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |token| {
            token.checked_add(1)
        })
        .expect("gather stage token space exhausted")
}

/// Dark gate + mode: `NY_MO_GATHER_SCORE=1` = raw `|A_lower|` ranking,
/// `=2` = relaxation-gap-weighted (`|A|·min(−l,u)` — the classic kFSB
/// surrogate; the weight is applied by the consumer, which owns the domain's
/// pre-activation bounds). Anything else (incl. unset) disables both capture
/// and consult. Mode 1 measured FLAT on the 7-row tier (2026-07-22 arm D:
/// 2477 −1.92 vs C −1.88, 966 −0.66 vs −0.64, 2050 −0.77 vs −0.64, 1761
/// −0.42 flat) — raw magnitude ignores how much relaxation slack a split
/// removes; mode 2 is the recorded successor.
pub(crate) fn gather_score_mode() -> Option<u8> {
    match std::env::var("NY_MO_GATHER_SCORE").ok().as_deref() {
        Some("1") => Some(1),
        Some("2") => Some(2),
        _ => None,
    }
}

/// Split-set fingerprint of a domain's β state: order-independent hash over
/// `(node_name, neuron_idx, sign)` of every entry. Two ReLU-split domains
/// with the same root and the same split set are the same search node, so
/// this identifies the domain for ADVISORY lookup purposes. (The root/empty
/// state hashes too — harmless: scores for the root are as valid as any.)
pub(crate) fn beta_split_fingerprint(state: &GraphBetaState) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut keys: Vec<(&str, usize, i8)> = state
        .entries
        .iter()
        .map(|e| {
            (
                e.node_name.as_str(),
                e.neuron_idx,
                if e.sign >= 0.0 { 1i8 } else { -1i8 },
            )
        })
        .collect();
    keys.sort_unstable();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    keys.len().hash(&mut h);
    for k in keys {
        k.hash(&mut h);
    }
    h.finish()
}

/// Verifier-lifetime advisory score cache (see module docs).
#[derive(Default)]
pub(crate) struct GatherScoreCache {
    inner: Mutex<HashMap<u64, Arc<[GatherScoreRow]>>>,
}

struct StagedGatherWrites {
    cache_address: usize,
    token: u64,
    writes: Vec<(u64, Arc<[GatherScoreRow]>)>,
}

thread_local! {
    /// Call-thread-local deferred publication. The wide β gather producer runs
    /// on the caller thread (device work may be asynchronous, publication is
    /// not), so concurrent verifier borrows cannot observe or steal another
    /// arm's writes.
    static STAGED_GATHER_WRITES: RefCell<Vec<StagedGatherWrites>> =
        const { RefCell::new(Vec::new()) };
}

/// Deferred advisory writes from exactly one H or W evaluation.
pub(crate) struct GatherScoreWriteSet {
    writes: Vec<(u64, Arc<[GatherScoreRow]>)>,
}

/// RAII stage: dropping without `finish` discards every speculative write.
#[must_use = "dropping an unfinished gather stage discards its writes"]
pub(crate) struct GatherScoreWriteStage<'a> {
    _cache: &'a GatherScoreCache,
    cache_address: usize,
    token: u64,
    active: bool,
}

impl GatherScoreWriteStage<'_> {
    pub(crate) fn finish(mut self) -> GatherScoreWriteSet {
        let frame = STAGED_GATHER_WRITES.with(|stages| {
            let mut stages = stages.borrow_mut();
            let position = stages
                .iter()
                .position(|stage| {
                    stage.cache_address == self.cache_address && stage.token == self.token
                })
                .expect("gather write stage must still be registered");
            stages.remove(position)
        });
        // Keep the guard active until exact-frame extraction succeeds. If the
        // invariant above ever fails, unwinding still lets Drop attempt exact
        // cleanup instead of silently leaking the frame.
        self.active = false;
        GatherScoreWriteSet {
            writes: frame.writes,
        }
    }
}

impl Drop for GatherScoreWriteStage<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        STAGED_GATHER_WRITES.with(|stages| {
            let mut stages = stages.borrow_mut();
            if let Some(position) = stages.iter().position(|stage| {
                stage.cache_address == self.cache_address && stage.token == self.token
            }) {
                stages.remove(position);
            }
        });
    }
}

impl GatherScoreCache {
    /// Insert a domain's harvested scores (advisory; clears everything on
    /// cap overflow rather than evicting — simplicity over retention).
    pub(crate) fn insert(&self, fingerprint: u64, rows: Vec<GatherScoreRow>) {
        if rows.is_empty() {
            return;
        }
        let rows: Arc<[GatherScoreRow]> = Arc::from(rows);
        let cache_address = std::ptr::from_ref(self).addr();
        let staged = STAGED_GATHER_WRITES.with(|stages| {
            let mut stages = stages.borrow_mut();
            let Some(stage) = stages
                .iter_mut()
                .rev()
                .find(|stage| stage.cache_address == cache_address)
            else {
                return false;
            };
            stage.writes.push((fingerprint, Arc::clone(&rows)));
            true
        });
        if staged {
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.len() >= GATHER_SCORE_CACHE_CAP {
            inner.clear();
        }
        inner.insert(fingerprint, rows);
    }

    /// Advisory lookup; `None` reverts the caller to the shipped scorer.
    pub(crate) fn get(&self, fingerprint: u64) -> Option<Arc<[GatherScoreRow]>> {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&fingerprint)
            .cloned()
    }

    /// Begin deferred publication for one paired arm.
    pub(crate) fn stage_writes(&self) -> GatherScoreWriteStage<'_> {
        let cache_address = std::ptr::from_ref(self).addr();
        let token = next_gather_stage_token();
        STAGED_GATHER_WRITES.with(|stages| {
            stages.borrow_mut().push(StagedGatherWrites {
                cache_address,
                token,
                writes: Vec::new(),
            });
        });
        GatherScoreWriteStage {
            _cache: self,
            cache_address,
            token,
            active: true,
        }
    }

    /// Commit one established arm exactly as its writes originally arrived.
    pub(crate) fn commit_all(&self, writes: &GatherScoreWriteSet) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self::commit_writes_locked(&mut inner, writes);
    }

    /// Atomically publish H, replacing only selected fingerprints with W.
    ///
    /// Readers take the same mutex, so none can observe the intermediate
    /// all-H image. No W write for a selected fingerprint means no W advice:
    /// the corresponding H entry is removed.
    pub(crate) fn commit_pair_selection(
        &self,
        established: &GatherScoreWriteSet,
        candidate: &GatherScoreWriteSet,
        selected_fingerprints: &[u64],
    ) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self::commit_writes_locked(&mut inner, established);
        for &fingerprint in selected_fingerprints {
            let selected_value = candidate
                .writes
                .iter()
                .rev()
                .find_map(|(key, rows)| (*key == fingerprint).then_some(rows));
            if let Some(value) = selected_value {
                if !inner.contains_key(&fingerprint) && inner.len() >= GATHER_SCORE_CACHE_CAP {
                    inner.clear();
                }
                inner.insert(fingerprint, Arc::clone(value));
            } else {
                inner.remove(&fingerprint);
            }
        }
    }

    fn commit_writes_locked(
        inner: &mut HashMap<u64, Arc<[GatherScoreRow]>>,
        writes: &GatherScoreWriteSet,
    ) {
        for (fingerprint, rows) in &writes.writes {
            if inner.len() >= GATHER_SCORE_CACHE_CAP {
                inner.clear();
            }
            inner.insert(*fingerprint, Arc::clone(rows));
        }
    }

    #[cfg(test)]
    pub(crate) fn bit_image(&self) -> Vec<(u64, Vec<(String, u32, u32)>)> {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut image: Vec<_> = inner
            .iter()
            .map(|(&key, rows)| {
                (
                    key,
                    rows.iter()
                        .map(|(name, column, score)| (name.clone(), *column, score.to_bits()))
                        .collect(),
                )
            })
            .collect();
        image.sort_unstable_by_key(|(key, _)| *key);
        image
    }
}

/// Pick the best-scored candidate among the caller's unstable set.
///
/// Returns `(node_name, neuron_idx, score)` of the highest-|A| candidate that
/// is present in `unstable`, or `None` when nothing intersects (fallback to
/// the shipped scorer). Deterministic tie-break: larger score wins, then the
/// earlier `unstable` position (stable across runs).
pub(crate) fn best_scored_candidate(
    rows: &[GatherScoreRow],
    unstable: &[(String, usize)],
) -> Option<(String, usize, f32)> {
    let mut score_of: HashMap<(&str, u32), f32> = HashMap::with_capacity(rows.len());
    for (name, col, s) in rows {
        let e = score_of.entry((name.as_str(), *col)).or_insert(*s);
        if *s > *e {
            *e = *s;
        }
    }
    let mut best: Option<(usize, f32)> = None;
    for (pos, (name, idx)) in unstable.iter().enumerate() {
        let Ok(col) = u32::try_from(*idx) else {
            continue;
        };
        if let Some(&s) = score_of.get(&(name.as_str(), col)) {
            if s.is_finite() && best.is_none_or(|(_, bs)| s > bs) {
                best = Some((pos, s));
            }
        }
    }
    best.map(|(pos, s)| {
        let (name, idx) = &unstable[pos];
        (name.clone(), *idx, s)
    })
}

/// Mode-2 pick: `raw |A| × weight(name, idx)`, where the consumer-supplied
/// weight is the candidate's relaxation slack (`min(−l, u)` of its
/// pre-activation bounds — the kFSB improvement surrogate). Candidates whose
/// weight is `None` (stable, missing bounds) are skipped; empty intersection
/// falls back to the shipped scorer as in mode 1.
pub(crate) fn best_weighted_candidate(
    rows: &[GatherScoreRow],
    unstable: &[(String, usize)],
    mut weight: impl FnMut(&str, usize) -> Option<f32>,
) -> Option<(String, usize, f32)> {
    let mut score_of: HashMap<(&str, u32), f32> = HashMap::with_capacity(rows.len());
    for (name, col, s) in rows {
        let e = score_of.entry((name.as_str(), *col)).or_insert(*s);
        if *s > *e {
            *e = *s;
        }
    }
    let mut best: Option<(usize, f32)> = None;
    for (pos, (name, idx)) in unstable.iter().enumerate() {
        let Ok(col) = u32::try_from(*idx) else {
            continue;
        };
        let Some(&raw) = score_of.get(&(name.as_str(), col)) else {
            continue;
        };
        let Some(w) = weight(name.as_str(), *idx) else {
            continue;
        };
        let s = raw * w;
        if s.is_finite() && best.is_none_or(|(_, bs)| s > bs) {
            best = Some((pos, s));
        }
    }
    best.map(|(pos, s)| {
        let (name, idx) = &unstable[pos];
        (name.clone(), *idx, s)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, idx: usize, sign: f32) -> crate::beta_crown::state::GraphBetaEntry {
        crate::beta_crown::state::GraphBetaEntry::new(name.to_string(), idx, 0.0, 0.0, sign)
            .expect("valid entry")
    }

    #[test]
    fn fingerprint_is_order_independent_and_sign_sensitive() {
        let a = GraphBetaState::from_entries(vec![entry("relu1", 3, 1.0), entry("relu2", 7, -1.0)]);
        let b = GraphBetaState::from_entries(vec![entry("relu2", 7, -1.0), entry("relu1", 3, 1.0)]);
        assert_eq!(beta_split_fingerprint(&a), beta_split_fingerprint(&b));

        let c =
            GraphBetaState::from_entries(vec![entry("relu1", 3, -1.0), entry("relu2", 7, -1.0)]);
        assert_ne!(beta_split_fingerprint(&a), beta_split_fingerprint(&c));
    }

    #[test]
    fn best_scored_candidate_intersects_and_ranks() {
        let rows = vec![
            ("relu1".to_string(), 3u32, 0.5f32),
            ("relu1".to_string(), 4u32, 2.5f32),
            ("relu2".to_string(), 9u32, 9.0f32),
        ];
        // relu2:9 not unstable -> best among the intersection is relu1:4.
        let unstable = vec![("relu1".to_string(), 3usize), ("relu1".to_string(), 4)];
        let got = best_scored_candidate(&rows, &unstable).expect("intersection");
        assert_eq!(got.0, "relu1");
        assert_eq!(got.1, 4);
        assert!((got.2 - 2.5).abs() < 1e-6);
        // No intersection -> None (fallback to the shipped scorer).
        let unstable = vec![("relu9".to_string(), 1usize)];
        assert!(best_scored_candidate(&rows, &unstable).is_none());
    }

    #[test]
    fn weighted_candidate_ranks_by_product_and_skips_none() {
        let rows = vec![
            ("relu1".to_string(), 3u32, 10.0f32), // big raw, tiny slack
            ("relu1".to_string(), 4u32, 2.0f32),  // small raw, big slack
            ("relu1".to_string(), 5u32, 8.0f32),  // weight None -> skipped
        ];
        let unstable = vec![
            ("relu1".to_string(), 3usize),
            ("relu1".to_string(), 4),
            ("relu1".to_string(), 5),
        ];
        let got = best_weighted_candidate(&rows, &unstable, |_, idx| match idx {
            3 => Some(0.01),
            4 => Some(5.0),
            _ => None,
        })
        .expect("weighted pick");
        assert_eq!(got.1, 4); // 2.0*5.0 = 10.0 > 10.0*0.01
    }

    #[test]
    fn cache_insert_get_and_cap_clear() {
        let cache = GatherScoreCache::default();
        cache.insert(42, vec![("r".into(), 1, 1.0)]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(43).is_none());
        cache.insert(42, Vec::new()); // empty rows are ignored
        assert!(cache.get(42).is_some());
    }

    #[test]
    fn paired_arm_transaction_reject_has_no_w_leak_and_select_commits_only_w_key() {
        let cache = GatherScoreCache::default();
        cache.insert(7, vec![("old".to_string(), 0, 0.25)]);
        let pre_arm_bits = cache.bit_image();

        let h_stage = cache.stage_writes();
        cache.insert(11, vec![("h".to_string(), 1, 1.5)]);
        cache.insert(12, vec![("h-peer".to_string(), 2, 2.5)]);
        cache.insert(13, vec![("h-without-w".to_string(), 5, 5.5)]);
        assert_eq!(
            cache.bit_image(),
            pre_arm_bits,
            "deferred H writes must not be globally visible"
        );
        let h_writes = h_stage.finish();

        let w_stage = cache.stage_writes();
        cache.insert(11, vec![("w".to_string(), 3, 3.5)]);
        cache.insert(12, vec![("w-rejected".to_string(), 4, 4.5)]);
        assert_eq!(
            cache.bit_image(),
            pre_arm_bits,
            "deferred W writes must not leak before selection"
        );
        let w_writes = w_stage.finish();

        // Production publication is one atomic H/W selection.
        cache.commit_pair_selection(&h_writes, &w_writes, &[11, 13]);
        let image = cache.bit_image();
        assert!(image.iter().any(|(key, rows)| {
            *key == 11 && rows == &[("w".to_string(), 3, 3.5_f32.to_bits())]
        }));
        assert!(image.iter().any(|(key, rows)| {
            *key == 12 && rows == &[("h-peer".to_string(), 2, 2.5_f32.to_bits())]
        }));
        assert!(
            image.iter().all(|(key, _)| *key != 13),
            "selected W without an arm-local write retained H advice"
        );
        assert!(
            image
                .iter()
                .all(|(_, rows)| rows.iter().all(|(name, _, _)| name != "w-rejected")),
            "a rejected W domain leaked advisory gather state"
        );
    }

    #[test]
    fn staged_writes_lifo_commit_to_their_exact_frames() {
        let cache = GatherScoreCache::default();
        let outer = cache.stage_writes();
        cache.insert(1, vec![("outer-before".into(), 1, 1.0)]);
        let inner = cache.stage_writes();
        cache.insert(2, vec![("inner".into(), 2, 2.0)]);
        let inner_writes = inner.finish();
        cache.insert(3, vec![("outer-after".into(), 3, 3.0)]);
        let outer_writes = outer.finish();

        cache.commit_all(&inner_writes);
        assert!(cache.get(2).is_some());
        assert!(cache.get(1).is_none());
        assert!(cache.get(3).is_none());
        cache.commit_all(&outer_writes);
        assert!(cache.get(1).is_some());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn same_cache_out_of_order_finish_cannot_steal_inner_frame() {
        let cache = GatherScoreCache::default();
        let outer = cache.stage_writes();
        cache.insert(10, vec![("outer".into(), 0, 1.0)]);
        let inner = cache.stage_writes();
        cache.insert(20, vec![("inner-before".into(), 0, 2.0)]);

        let outer_writes = outer.finish();
        cache.insert(21, vec![("inner-after".into(), 0, 2.1)]);
        let inner_writes = inner.finish();

        cache.commit_all(&outer_writes);
        assert!(cache.get(10).is_some());
        assert!(cache.get(20).is_none());
        assert!(cache.get(21).is_none());
        cache.commit_all(&inner_writes);
        assert!(cache.get(20).is_some());
        assert!(cache.get(21).is_some());
    }

    #[test]
    fn same_cache_out_of_order_drop_cannot_steal_inner_frame() {
        let cache = GatherScoreCache::default();
        let outer = cache.stage_writes();
        cache.insert(30, vec![("discarded-outer".into(), 0, 3.0)]);
        let inner = cache.stage_writes();
        cache.insert(40, vec![("inner-before".into(), 0, 4.0)]);

        drop(outer);
        cache.insert(41, vec![("inner-after".into(), 0, 4.1)]);
        let inner_writes = inner.finish();
        cache.commit_all(&inner_writes);

        assert!(cache.get(30).is_none());
        assert!(cache.get(40).is_some());
        assert!(cache.get(41).is_some());
    }

    #[test]
    fn different_cache_stages_never_capture_each_others_writes() {
        let left = GatherScoreCache::default();
        let right = GatherScoreCache::default();
        let left_stage = left.stage_writes();
        let right_stage = right.stage_writes();
        left.insert(50, vec![("left".into(), 0, 5.0)]);
        right.insert(60, vec![("right".into(), 0, 6.0)]);
        let left_writes = left_stage.finish();
        let right_writes = right_stage.finish();

        left.commit_all(&left_writes);
        right.commit_all(&right_writes);
        assert!(left.get(50).is_some());
        assert!(left.get(60).is_none());
        assert!(right.get(60).is_some());
        assert!(right.get(50).is_none());
    }

    #[test]
    fn staged_write_unwind_and_early_return_discard_only_their_frames() {
        fn early_return(cache: &GatherScoreCache) {
            let _stage = cache.stage_writes();
            cache.insert(70, vec![("early".into(), 0, 7.0)]);
        }

        let cache = GatherScoreCache::default();
        let outer = cache.stage_writes();
        cache.insert(71, vec![("outer-before".into(), 0, 7.1)]);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _inner = cache.stage_writes();
            cache.insert(72, vec![("panic".into(), 0, 7.2)]);
            panic!("exercise gather-stage unwind");
        }));
        assert!(unwind.is_err());
        early_return(&cache);
        cache.insert(73, vec![("outer-after".into(), 0, 7.3)]);
        let outer_writes = outer.finish();
        cache.commit_all(&outer_writes);

        assert!(cache.get(70).is_none());
        assert!(cache.get(72).is_none());
        assert!(cache.get(71).is_some());
        assert!(cache.get(73).is_some());
    }
}

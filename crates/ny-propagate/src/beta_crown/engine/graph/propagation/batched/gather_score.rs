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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::beta_crown::state::GraphBetaState;

/// One harvested candidate: `(relu node name, neuron column, |A_lower|)`.
pub(crate) type GatherScoreRow = (String, u32, f32);

/// Hard cap on cached domains — a runaway BaB tree must not grow the cache
/// unboundedly. On overflow the cache CLEARS (advisory data; losing it only
/// reverts candidates to the shipped scorer).
const GATHER_SCORE_CACHE_CAP: usize = 8192;

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

impl GatherScoreCache {
    /// Insert a domain's harvested scores (advisory; clears everything on
    /// cap overflow rather than evicting — simplicity over retention).
    pub(crate) fn insert(&self, fingerprint: u64, rows: Vec<GatherScoreRow>) {
        if rows.is_empty() {
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.len() >= GATHER_SCORE_CACHE_CAP {
            inner.clear();
        }
        inner.insert(fingerprint, Arc::from(rows));
    }

    /// Advisory lookup; `None` reverts the caller to the shipped scorer.
    pub(crate) fn get(&self, fingerprint: u64) -> Option<Arc<[GatherScoreRow]>> {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&fingerprint)
            .cloned()
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
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #postbab-seed: process-global export of the internal attack's BEST-MARGIN
//! candidate (the closest-to-violation point the internal graph-PGD search saw,
//! even when it never crossed the internal violation threshold).
//!
//! MOTIVATION (measured, soundnessbench): the resister instances burn the full
//! internal tier WITHOUT emitting a near-miss witness — the internal search
//! stalls a hair below the violation threshold, so `run_and_translate`'s
//! post-BaB ULP-jitter lane previously started from the box center and never
//! reached the planted-CE basin in the leftover budget. Exporting the best
//! internal point gives that lane the same seed quality the model_5 flip had
//! (an internal witness ULPs below the ORT threshold).
//!
//! SOUNDNESS: attack-only guidance, never a verdict carrier. The consumer
//! (`try_postbab_falsify`) only uses the point as a SEARCH SEED; any candidate
//! it produces still passes the UNCHANGED trusted-ORT + true-f64 acceptance
//! gate. A wrong or stale point can at worst waste otherwise-dead leftover
//! budget — it can never manufacture a false `sat`.
//!
//! CONVENTION: `margin` is the joint AND-clause hinge loss of
//! [`super::verify`]'s `joint_hinge_loss` — strictly negative until every
//! modeled conjunct of the property holds at the point (0 crossing = internal
//! counterexample), saturating to the positive min-margin once all hold.
//! Higher is better; the tracker keeps the maximum seen since the last reset.

use ndarray::ArrayD;
use std::sync::Mutex;

/// The closest-to-violation internal candidate since the last reset.
#[derive(Debug, Clone)]
pub(crate) struct BestMarginCandidate {
    /// Flattened input point, row-major (network input order = `X_i` order).
    pub(crate) point: Vec<f32>,
    /// Joint hinge margin at the point (see module docs; >= 0 => the internal
    /// forward considered every modeled conjunct satisfied).
    pub(crate) margin: f32,
}

/// Two independent slots: `[0]` = PLAIN search lanes, `[1]` = #exploit-recycle
/// lanes (jittered clones of the best point). Kept separate because the two
/// lineages park in DIFFERENT near-boundary jam points and the post-BaB
/// ULP-jitter lane is position-sensitive (measured: soundnessbench model_6
/// flips from the plain-lane seed, model_37 from the exploit-lane seed; a
/// single max-margin slot loses whichever the other lineage overwrote).
static BEST: Mutex<[Option<BestMarginCandidate>; 2]> = Mutex::new([None, None]);

/// Recover from a poisoned lock: the tracker is guidance-only, so a panicked
/// attack thread must never take the verdict path down with it.
fn lock() -> std::sync::MutexGuard<'static, [Option<BestMarginCandidate>; 2]> {
    BEST.lock().unwrap_or_else(|p| p.into_inner())
}

/// Clear the tracker (call before starting a fresh verification run).
pub(crate) fn reset_best_margin_candidate() {
    *lock() = [None, None];
}

fn record_slot(slot: usize, margin: f32, point: &ArrayD<f32>) {
    if !margin.is_finite() {
        return;
    }
    let mut best = lock();
    if best[slot].as_ref().is_none_or(|b| margin > b.margin) {
        best[slot] = Some(BestMarginCandidate {
            point: point.iter().copied().collect(),
            margin,
        });
    }
}

/// Record `point` if its joint margin beats the best PLAIN-lane candidate seen
/// since the last reset. Non-finite margins (no modeled conjunct) are ignored.
pub(crate) fn record_best_margin_candidate(margin: f32, point: &ArrayD<f32>) {
    record_slot(0, margin, point);
}

/// Record an #exploit-recycle lane's `point` (separate slot; see [`BEST`]).
pub(crate) fn record_best_margin_candidate_exploit(margin: f32, point: &ArrayD<f32>) {
    record_slot(1, margin, point);
}

/// Take (and clear) the candidates recorded since the last reset, best margin
/// first, deduplicated (identical points collapse to one).
pub(crate) fn take_best_margin_candidates() -> Vec<BestMarginCandidate> {
    let mut slots = lock();
    let mut out: Vec<BestMarginCandidate> = slots.iter_mut().filter_map(Option::take).collect();
    out.sort_by(|a, b| {
        b.margin
            .partial_cmp(&a.margin)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if out.len() == 2 && out[0].point == out[1].point {
        out.truncate(1);
    }
    out
}

/// Take (and clear) the single best candidate (compat shim over
/// [`take_best_margin_candidates`]).
#[cfg(test)]
pub(crate) fn take_best_margin_candidate() -> Option<BestMarginCandidate> {
    take_best_margin_candidates().into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    fn pt(vals: &[f32]) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&[vals.len()]), vals.to_vec()).unwrap()
    }

    #[test]
    fn keeps_maximum_margin_and_clears_on_take() {
        reset_best_margin_candidate();
        record_best_margin_candidate(-1.0, &pt(&[0.1, 0.2]));
        record_best_margin_candidate(-0.5, &pt(&[0.3, 0.4]));
        record_best_margin_candidate(-2.0, &pt(&[0.5, 0.6])); // worse: ignored
        record_best_margin_candidate(f32::NAN, &pt(&[9.0, 9.0])); // non-finite: ignored
        record_best_margin_candidate(f32::INFINITY, &pt(&[9.0, 9.0])); // non-finite: ignored
        let best = take_best_margin_candidate().expect("candidate recorded");
        assert_eq!(best.margin, -0.5);
        assert_eq!(best.point, vec![0.3, 0.4]);
        assert!(take_best_margin_candidate().is_none(), "take clears");
    }
}

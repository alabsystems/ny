// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Named weight storage backed by a hash map of dynamic-rank arrays.

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Opaque snapshot of one floating-point weight revision.
///
/// Only [`WeightStore`] can create a snapshot.  It includes an unforgeable
/// store-instance token as well as the per-name revision, so replacing a store
/// with a separately constructed or cloned store cannot preserve a proof tied
/// to the original instance.
#[derive(Debug, Clone)]
pub struct WeightRevision {
    store_identity: Arc<()>,
    name: String,
    revision: u64,
}

/// Named weight storage backed by a hash map of dynamic-rank arrays.
#[derive(Debug)]
pub struct WeightStore {
    weights: HashMap<String, ArrayD<f32>>,
    /// Frozen constants that are known only up to a RIGOROUS ENCLOSURE
    /// `[lower, upper]`, keyed by tensor name and DISJOINT from `weights`.
    ///
    /// A name lands here when a both-constant fold could not be carried out
    /// bit-exactly in binary32 (the `try_fold_matmul_exact` gate) but a sound
    /// outward-rounded interval carrier could be computed instead. The
    /// invariant is the only thing the store promises:
    ///
    /// > the exact real value of the folded tensor lies in `[lower, upper]`
    /// > elementwise.
    ///
    /// WHY A SEPARATE MAP, AND WHY [`Self::get`] DOES NOT SEE IT. Every
    /// existing consumer of a frozen constant in this workspace reads through
    /// [`Self::get`] and would compute with the value as if it were exact.
    /// Silently handing back a midpoint is precisely the unsound point-fold
    /// this carrier exists to avoid, so an interval-valued name is INVISIBLE to
    /// the point API: `get`, `contains_key`, `len`, `keys`, `iter` and
    /// `find_by_key` are unchanged and never observe it. A consumer that wants
    /// the enclosure must ask for it by name through [`Self::get_interval`],
    /// which is an explicit, auditable opt-in; every consumer that does not ask
    /// simply fails to find the constant and degrades to its existing
    /// unsupported/fail-closed path.
    ///
    /// A ZERO-WIDTH interval is never stored: [`Self::insert_interval`]
    /// canonicalizes `lower == upper` (bitwise) back into `weights` as an
    /// ordinary point constant, so "the fold was exact after all" stays
    /// indistinguishable from today.
    interval_weights: HashMap<String, (ArrayD<f32>, ArrayD<f32>)>,
    integers: HashMap<String, ArrayD<i64>>,
    integer_ranges: HashMap<String, (i64, i64)>,
    revision_tracking: Option<RevisionTracking>,
}

#[derive(Debug)]
struct RevisionTracking {
    store_identity: Arc<()>,
    revisions: HashMap<String, u64>,
    revision_overflowed: HashSet<String>,
}

impl WeightStore {
    /// Creates an empty weight store.
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            interval_weights: HashMap::new(),
            integers: HashMap::new(),
            integer_ranges: HashMap::new(),
            revision_tracking: None,
        }
    }

    /// Returns the weight tensor with the given name, if present.
    pub fn get(&self, name: &str) -> Option<&ArrayD<f32>> {
        self.weights.get(name)
    }

    /// Returns `true` if a weight with the given name exists.
    pub fn contains_key(&self, name: &str) -> bool {
        self.weights.contains_key(name) || self.integers.contains_key(name)
    }

    /// Inserts or replaces a weight tensor by name.
    ///
    /// A POINT value supersedes any interval carrier previously recorded for
    /// the same name: the two maps stay disjoint, and the exact value is
    /// strictly better information than an enclosure of it.
    pub fn insert(&mut self, name: String, weights: ArrayD<f32>) {
        self.bump_revision(&name);
        self.interval_weights.remove(&name);
        self.weights.insert(name, weights);
    }

    /// Returns the rigorous `(lower, upper)` enclosure recorded for `name`, if
    /// this name is an interval-valued frozen constant.
    ///
    /// Returns `None` for ordinary point constants — a caller that wants to
    /// treat both uniformly must handle the point case through [`Self::get`]
    /// explicitly. That asymmetry is deliberate: it makes "I forgot the
    /// interval case" a missing constant (fail closed) rather than a silently
    /// exact-looking midpoint.
    #[must_use]
    pub fn get_interval(&self, name: &str) -> Option<(&ArrayD<f32>, &ArrayD<f32>)> {
        self.interval_weights.get(name).map(|(l, u)| (l, u))
    }

    /// Whether `name` is an interval-valued frozen constant.
    #[must_use]
    pub fn contains_interval(&self, name: &str) -> bool {
        self.interval_weights.contains_key(name)
    }

    /// Number of interval-valued frozen constants held.
    ///
    /// This is zero for every model loaded with the interval carrier disarmed,
    /// which is what the dark-behaviour tests assert.
    #[must_use]
    pub fn interval_len(&self) -> usize {
        self.interval_weights.len()
    }

    /// Iterate over `(name, lower, upper)` triples of the interval constants.
    pub fn iter_intervals(&self) -> impl Iterator<Item = (&str, &ArrayD<f32>, &ArrayD<f32>)> {
        self.interval_weights
            .iter()
            .map(|(k, (l, u))| (k.as_str(), l, u))
    }

    /// Records a rigorous enclosure `[lower, upper]` for a frozen constant.
    ///
    /// The caller is asserting that the EXACT REAL value of the tensor lies in
    /// `[lower, upper]` elementwise. Everything that can be checked here is
    /// checked here, and a violation is REFUSED (returns `false`, stores
    /// nothing) rather than repaired:
    ///
    /// * `lower` and `upper` must have the same shape;
    /// * every endpoint must be finite (an infinite endpoint is not an
    ///   enclosure worth carrying — the caller should leave the node unfolded
    ///   and let it fail closed to an unbounded skip instead);
    /// * `lower[i] <= upper[i]` elementwise, with NaN rejected;
    /// * the name must not already be a point constant.
    ///
    /// A bitwise-identical `lower`/`upper` pair is CANONICALIZED into the point
    /// map: zero width is exactly the case the existing bit-exact fold already
    /// covers, and keeping it out of the interval map is what makes
    /// [`Self::interval_len`] a faithful "did the carrier fire" signal.
    ///
    /// Returns whether the tensor was stored (in either map).
    pub fn insert_interval(
        &mut self,
        name: String,
        lower: ArrayD<f32>,
        upper: ArrayD<f32>,
    ) -> bool {
        if lower.shape() != upper.shape() {
            return false;
        }
        if self.weights.contains_key(&name) {
            return false;
        }
        let mut zero_width = true;
        for (&l, &u) in lower.iter().zip(upper.iter()) {
            if !l.is_finite() || !u.is_finite() || l > u {
                return false;
            }
            if l != u {
                zero_width = false;
            }
        }
        if zero_width {
            // Exact after all: indistinguishable from the bit-exact path. The
            // test is NUMERIC equality rather than bitwise, so a `(-0.0, +0.0)`
            // pair — which denotes the single real value zero — collapses too;
            // `upper` is stored because it carries the non-negative sign.
            self.insert(name, upper);
            return true;
        }
        self.interval_weights.insert(name, (lower, upper));
        true
    }

    /// Enables opaque floating-point revision snapshots for future inserts.
    ///
    /// Tracking is opt-in so ordinary model loading pays no revision-map or
    /// duplicated-name cost. It can only be enabled before the first floating
    /// weight is inserted; repeated enabling is idempotent.
    #[must_use]
    pub fn enable_revision_tracking(&mut self) -> bool {
        if self.revision_tracking.is_some() {
            return true;
        }
        if !self.weights.is_empty() {
            return false;
        }
        self.revision_tracking = Some(RevisionTracking {
            store_identity: Arc::new(()),
            revisions: HashMap::new(),
            revision_overflowed: HashSet::new(),
        });
        true
    }

    /// Returns an opaque snapshot of the current revision for `name`.
    ///
    /// `None` means the name has never been inserted or that its revision
    /// counter overflowed.  Overflow therefore fails closed for consumers that
    /// bind immutable evidence to a particular weight revision.
    #[must_use]
    pub fn revision(&self, name: &str) -> Option<WeightRevision> {
        let tracking = self.revision_tracking.as_ref()?;
        if tracking.revision_overflowed.contains(name) {
            return None;
        }
        Some(WeightRevision {
            store_identity: Arc::clone(&tracking.store_identity),
            name: name.to_string(),
            revision: *tracking.revisions.get(name)?,
        })
    }

    /// Returns whether `snapshot` is still the current revision for `name` in
    /// this exact store instance.
    #[must_use]
    pub fn matches_revision(&self, name: &str, snapshot: &WeightRevision) -> bool {
        let Some(tracking) = &self.revision_tracking else {
            return false;
        };
        !tracking.revision_overflowed.contains(name)
            && Arc::ptr_eq(&tracking.store_identity, &snapshot.store_identity)
            && snapshot.name == name
            && tracking.revisions.get(name) == Some(&snapshot.revision)
    }

    /// Removes a floating-point weight tensor by name.
    pub fn remove(&mut self, name: &str) -> Option<ArrayD<f32>> {
        self.bump_revision(name);
        self.interval_weights.remove(name);
        self.weights.remove(name)
    }

    /// Stores an integer tensor by name without passing through f32.
    pub fn insert_integers(&mut self, name: String, values: ArrayD<i64>) {
        self.integers.insert(name, values);
    }

    /// Stores the representable integer range for a tensor's original integer dtype.
    pub fn insert_integer_range(&mut self, name: String, min: i64, max: i64) {
        self.integer_ranges.insert(name, (min, max));
    }

    /// Returns the integer tensor with the given name, if present.
    pub fn get_integers(&self, name: &str) -> Option<&ArrayD<i64>> {
        self.integers.get(name)
    }

    /// Returns the representable integer range for a tensor's original integer dtype.
    pub fn get_integer_range(&self, name: &str) -> Option<(i64, i64)> {
        self.integer_ranges.get(name).copied()
    }

    /// Validates that no stored weight tensors contain NaN values.
    ///
    /// NaN in weight matrices silently propagates through matrix
    /// multiplications and produces unsound verification results.
    /// Call at model-loading boundaries after all weights are loaded
    /// (including batch-norm fold and constant fold).
    ///
    /// Note: Inf values are permitted because ONNX Slice uses +Inf as a
    /// sentinel for "to end". Inf in actual weight matrices is caught
    /// downstream during bound construction.
    pub fn validate_no_nan(&self) -> Result<()> {
        for (name, tensor) in &self.weights {
            if tensor.iter().any(|v| v.is_nan()) {
                return Err(NyError::ModelLoad(format!(
                    "Weight tensor '{}' contains NaN values — corrupted or adversarial model",
                    name
                )));
            }
        }
        for (name, (lower, upper)) in &self.interval_weights {
            if lower.iter().chain(upper.iter()).any(|v| v.is_nan()) {
                return Err(NyError::ModelLoad(format!(
                    "Interval weight tensor '{}' contains NaN endpoints — corrupted or \
                     adversarial model",
                    name
                )));
            }
        }
        Ok(())
    }

    /// Returns the number of stored weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Returns `true` if no weights are stored.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Returns an iterator over weight names.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(|s| s.as_str())
    }

    /// Iterate over (name, weight) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ArrayD<f32>)> {
        self.weights.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Find a weight by predicate on key.
    pub fn find_by_key<F>(&self, predicate: F) -> Option<(&str, &ArrayD<f32>)>
    where
        F: Fn(&str) -> bool,
    {
        self.weights
            .iter()
            .find(|(k, _)| predicate(k))
            .map(|(k, v)| (k.as_str(), v))
    }

    fn bump_revision(&mut self, name: &str) {
        let Some(tracking) = &mut self.revision_tracking else {
            return;
        };
        if tracking.revision_overflowed.contains(name) {
            return;
        }
        let revision = tracking.revisions.entry(name.to_string()).or_insert(0);
        if let Some(next) = revision.checked_add(1) {
            *revision = next;
        } else {
            tracking.revision_overflowed.insert(name.to_string());
        }
    }
}

impl Clone for WeightStore {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            interval_weights: self.interval_weights.clone(),
            integers: self.integers.clone(),
            integer_ranges: self.integer_ranges.clone(),
            revision_tracking: self.revision_tracking.as_ref().map(|tracking| {
                RevisionTracking {
                    // A tracked clone is a distinct mutable store. Giving it a
                    // fresh identity prevents assigning an old clone over a
                    // model from restoring an immutable revision proof.
                    store_identity: Arc::new(()),
                    revisions: tracking.revisions.clone(),
                    revision_overflowed: tracking.revision_overflowed.clone(),
                }
            }),
        }
    }
}

impl Default for WeightStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn revisions_change_on_insert_remove_and_reinsert() {
        let mut store = WeightStore::new();
        assert!(store.enable_revision_tracking());
        assert!(store.revision("w").is_none());

        store.insert("w".to_string(), arr1(&[1.0_f32]).into_dyn());
        let inserted = store.revision("w").expect("insert creates revision");
        assert!(store.matches_revision("w", &inserted));

        store.insert("w".to_string(), arr1(&[1.0_f32]).into_dyn());
        assert!(!store.matches_revision("w", &inserted));
        let replaced = store.revision("w").expect("replace advances revision");

        store.remove("w");
        assert!(!store.matches_revision("w", &replaced));
        let removed = store.revision("w").expect("remove advances revision");

        store.insert("w".to_string(), arr1(&[1.0_f32]).into_dyn());
        assert!(!store.matches_revision("w", &removed));
    }

    #[test]
    fn cloned_or_independent_store_cannot_match_revision() {
        let mut store = WeightStore::new();
        assert!(store.enable_revision_tracking());
        store.insert("w".to_string(), arr1(&[1.0_f32]).into_dyn());
        store.insert("same_revision".to_string(), arr1(&[1.0_f32]).into_dyn());
        let revision = store.revision("w").expect("insert creates revision");
        assert!(!store.matches_revision("same_revision", &revision));

        let cloned = store.clone();
        assert!(!cloned.matches_revision("w", &revision));

        let mut independent = WeightStore::new();
        assert!(independent.enable_revision_tracking());
        independent.insert("w".to_string(), arr1(&[1.0_f32]).into_dyn());
        assert!(!independent.matches_revision("w", &revision));
    }

    #[test]
    fn revision_overflow_fails_closed() {
        let mut store = WeightStore::new();
        assert!(store.enable_revision_tracking());
        store
            .revision_tracking
            .as_mut()
            .expect("tracking enabled")
            .revisions
            .insert("w".to_string(), u64::MAX);
        store.insert("w".to_string(), arr1(&[1.0_f32]).into_dyn());
        assert!(store.revision("w").is_none());
    }

    #[test]
    fn ordinary_store_has_no_revision_overhead_or_retroactive_enable() {
        let mut store = WeightStore::new();
        store.insert("w".to_string(), arr1(&[1.0_f32]).into_dyn());
        assert!(store.revision_tracking.is_none());
        assert!(store.revision("w").is_none());
        assert!(!store.enable_revision_tracking());
    }

    #[test]
    fn validate_no_nan_accepts_finite_weights() {
        let mut ws = WeightStore::new();
        ws.insert("w".to_string(), arr1(&[1.0, 2.0, 3.0]).into_dyn());
        ws.insert("b".to_string(), arr1(&[0.0, -1.0]).into_dyn());
        ws.validate_no_nan().expect("finite weights should pass");
    }

    #[test]
    fn validate_no_nan_accepts_inf() {
        // Inf is permitted: ONNX Slice uses +Inf as "to end" sentinel.
        let mut ws = WeightStore::new();
        ws.insert("ends".to_string(), arr1(&[f32::INFINITY]).into_dyn());
        ws.insert("starts".to_string(), arr1(&[f32::NEG_INFINITY]).into_dyn());
        ws.validate_no_nan().expect("Inf should be permitted");
    }

    #[test]
    fn validate_no_nan_rejects_nan() {
        let mut ws = WeightStore::new();
        ws.insert("good".to_string(), arr1(&[1.0, 2.0]).into_dyn());
        ws.insert("bad".to_string(), arr1(&[1.0, f32::NAN, 3.0]).into_dyn());
        let err = ws.validate_no_nan().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("bad") && msg.contains("NaN"),
            "Error should name the tensor and mention NaN, got: {msg}"
        );
    }

    #[test]
    fn validate_no_nan_empty_store() {
        let ws = WeightStore::new();
        ws.validate_no_nan().expect("empty store should pass");
    }

    #[test]
    fn validate_no_nan_reports_model_load_error() {
        let mut ws = WeightStore::new();
        ws.insert("nan_weight".to_string(), arr1(&[f32::NAN]).into_dyn());
        let err = ws.validate_no_nan().unwrap_err();
        assert!(
            matches!(err, NyError::ModelLoad(_)),
            "Expected ModelLoad error variant, got: {err:?}"
        );
    }

    #[test]
    fn insert_integers_preserves_exact_integer_values() {
        let mut ws = WeightStore::new();
        let original = arr1(&[i64::MAX, 16_777_217, -16_777_217]).into_dyn();
        ws.insert_integers("shape".to_string(), original.clone());

        let stored = ws
            .get_integers("shape")
            .expect("integer tensor should be retrievable");
        assert_eq!(stored, &original);
        assert!(
            ws.get("shape").is_none(),
            "integer-only insert should not fabricate an f32 tensor"
        );
        assert!(
            ws.contains_key("shape"),
            "contains_key must acknowledge integer-only tensors"
        );
    }
}

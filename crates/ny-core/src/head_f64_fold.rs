// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mn-head-facet increment 1 — HEAD k-ReLU coupling-facet fold REGISTRY for the
//! per-subdomain CPU f64 critical-row recovery (`sound_f64_lower_bound`).
//!
//! The fold ENTRY + its global registry + the dark gate live here in `ny-core`
//! (mirroring [`crate::resident_cut_fold`]) so the writer (`ny-propagate`, root
//! builder) and the reader (`ny-propagate`'s f64 recovery lane) reach the same
//! registry through their shared `ny-core` dependency without a dependency cycle.
//!
//! A registered fold contributes, to the LOWER objective at the target head ReLU
//! (the network's FIRST ReLU in fold/backward order = the dense `Gemm→ReLU→Gemm`
//! head), the proven Lagrangian embedding of a set of `β_c ≥ 0`-weighted coupling
//! facets `Σ a_i·x_i + Σ g_i·y_i ≤ b_c` (valid over-approximation half-spaces of
//! the 2-neuron ReLU-graph reachable set):
//!   * `post[i].0`  — `+Σ_c β_c·g_i(c)` on the ReLU-OUTPUT (post-activation `y_i`)
//!                    frontier, added BEFORE the ReLU relaxation (rides it);
//!   * `pre[i].0`   — `+Σ_c β_c·a_i(c)` on the ReLU-INPUT (pre-activation `x_i`)
//!                    frontier, added AFTER the relaxation (direct on `x_i`);
//!   * `bias_shift` — `−Σ_c β_c·b_c` folded once into the lower bias.
//! Because every `facet_c − b_c ≤ 0` on the reachable set and `β_c ≥ 0`, the folded
//! objective `margin + Σ_c β_c·(facet_c − b_c) ≤ margin`, so the concretized value
//! LOWER-bounds the true margin (weak duality — the identical construction to a
//! single-neuron β-split). The f64 recovery `max`-merges it into `best_lo`, and a
//! monotone max can only RAISE the certified lower bound — never above the true
//! margin. Hence NO false lower bound (no false UNSAT) is representable.
//!
//! MOAT — CERTIFIED BUILD ERROR. Each product `β_c·g_i(c)` is EXACT in f64 (an
//! `f32×f32` product has ≤48 mantissa bits ≤ 53). The ONLY rounding is the f64
//! accumulation over multiple facets, tracked OUTWARD per-key at build time
//! (`post[i].1`, `pre[i].1`, `bias_err`) and injected into the recovery's
//! certified `err`/`berr` channels at application, so the returned f32 is a
//! rigorous lower bound of the objective computed with the EXACT `β_c·g_i`
//! multipliers — which is `≤` the true margin by the Lagrangian argument above.
//!
//! PRODUCTION AUTHORITY IS QUARANTINED: `NY_MN_HEAD_FACET=1` cannot arm this
//! registry until generated facet support is checker-certified. The default path
//! never enters the fold arm, so it is byte-identical (the fold is `None` ⇒ the
//! `sound_f64_branch` Activation arm is untouched).

// Blessed env access (2026-07-21): reads process-global env under the same
// deny-by-default ENV lint discipline as `resident_cut_fold` (read-only gate).
#![allow(unknown_lints)]
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// One HEAD coupling-facet fold at a FIXED shared `β` (the recovery `max`es over a
/// small β-grid of these, each independently sound). All coefficients + their
/// certified build error are pre-reduced to f64 sparse maps keyed by the head
/// ReLU's flat neuron index (dense head ⇒ the pool `neuron_idx` and the f64
/// recovery row column share the same C-order flat index — no col2im permutation
/// hazard, unlike the conv/stem lever).
#[derive(Debug, Clone, Default)]
pub struct HeadF64Fold {
    /// GLOBAL fold-order (output→input) activation index of the head ReLU. The
    /// recovery folds when `act_base + act_idx == target_act`. Re-validated at the
    /// recovery site against the domain's actual `relu_names` (see `relu_name`).
    pub target_act: usize,
    /// Head ReLU node name — the AUTHORITATIVE identity. The recovery re-resolves
    /// `target_act` from the domain's `relu_names` by this name and refuses to
    /// fold on any mismatch (a fold of one ReLU's facets onto another's neurons
    /// would be an invalid Lagrangian ⇒ potential false lower bound).
    pub relu_name: String,
    /// Head ReLU width (`num_neurons`). The recovery refuses to fold unless the
    /// target Activation layer has exactly this many neurons (belt-and-suspenders
    /// against a same-name width drift or an index out of range).
    pub head_width: usize,
    /// `neuron_idx → (Σ_c β_c·g_i(c), certified accumulation error ≥ |exact−value|)`.
    /// Post-activation (ReLU-OUTPUT `y_i`) adds, applied BEFORE the sign-select.
    pub post: HashMap<u32, (f64, f64)>,
    /// `neuron_idx → (Σ_c β_c·a_i(c), certified accumulation error)`. Pre-activation
    /// (ReLU-INPUT `x_i`) adds, applied AFTER the relaxation (direct on `x_i`).
    pub pre: HashMap<u32, (f64, f64)>,
    /// `−Σ_c β_c·b_c`, folded once into the lower bias (outward).
    pub bias_shift: f64,
    /// Certified accumulation error `≥ |exact − bias_shift|` on the bias fold.
    pub bias_err: f64,
}

impl HeadF64Fold {
    /// True when the fold carries no term at all (empty maps, zero bias/err). Such
    /// a fold applied at `target_act` performs ZERO arithmetic on the recovery row
    /// (every guarded add is skipped), so it is byte-identical to the no-fold path
    /// — the invariant oracle (a) relies on.
    pub fn is_empty(&self) -> bool {
        self.post.is_empty()
            && self.pre.is_empty()
            && self.bias_shift == 0.0
            && self.bias_err == 0.0
    }

    /// Sorted `(neuron, coeff, err)` post-activation terms (deterministic; for
    /// tests / logging — the hot path uses `post.get` directly).
    pub fn post_terms(&self) -> Vec<(u32, f64, f64)> {
        let mut v: Vec<(u32, f64, f64)> = self.post.iter().map(|(&k, &(c, e))| (k, c, e)).collect();
        v.sort_unstable_by_key(|&(k, ..)| k);
        v
    }

    /// Sorted `(neuron, coeff, err)` pre-activation terms.
    pub fn pre_terms(&self) -> Vec<(u32, f64, f64)> {
        let mut v: Vec<(u32, f64, f64)> = self.pre.iter().map(|(&k, &(c, e))| (k, c, e)).collect();
        v.sort_unstable_by_key(|&(k, ..)| k);
        v
    }
}

fn registry() -> &'static RwLock<Option<Arc<[HeadF64Fold]>>> {
    static REGISTRY: OnceLock<RwLock<Option<Arc<[HeadF64Fold]>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(None))
}

/// Production authority gate for the HEAD coupling-facet fold.
///
/// Hard-quarantined even when `NY_MN_HEAD_FACET=1` is present. The f64 fold
/// carries reduction error and target metadata, but its generated facet RHS
/// still comes from tolerance-based vertex deduplication and an uncertified f64
/// support dot product. Treating those normals as proposals is safe only after a
/// directed/exact support checker proves the stored f32 half-space contains all
/// four ReLU orthants.
pub fn head_f64_fold_enabled() -> bool {
    false
}

/// RESEARCH-MEASUREMENT reader arm (dark, default-off, `NY_MN_HEAD_F64_MEASURE_ARM=1`).
///
/// Separate from the production authority gate [`head_f64_fold_enabled`] (which
/// stays hard-quarantined at `false`). This arms ONLY the registry READER, and is
/// intended to be paired with `NY_MN_HEAD_F64_CERTIFIED_MEASURE=1`, under which the
/// registered facets come from the EXACT support checker
/// (`certified_coupling_facets_exact`) — precisely the "directed/exact support
/// checker proves the stored half-space contains all four ReLU orthants" that the
/// quarantine names as the re-authorization bar. Sound by construction: the fold is
/// monotone-`max`-merged into `best_lo[critical]` (can only RAISE the certified
/// lower bound, never above the true margin). This is a measurement lane, NOT a
/// production verdict re-authorization — production (`NY_MN_HEAD_FACET`) stays
/// `head_f64_fold_enabled()==false`.
pub fn head_f64_measure_arm_enabled() -> bool {
    matches!(
        std::env::var("NY_MN_HEAD_F64_MEASURE_ARM").ok().as_deref(),
        Some("1")
    )
}

/// Register (or replace) the HEAD fold β-grid. Called ONCE at root, before the
/// wide lane runs (mirrors `set_resident_cut_fold`).
pub fn set_head_f64_folds(folds: Vec<HeadF64Fold>) {
    if let Ok(mut guard) = registry().write() {
        *guard = Some(Arc::from(folds));
    }
}

/// Clear the HEAD fold registry (back to the byte-identical default path).
pub fn clear_head_f64_folds() {
    if let Ok(mut guard) = registry().write() {
        *guard = None;
    }
}

/// The active HEAD fold β-grid, or `None` (zero-cost untouched path) unless the
/// gate is set AND an entry is registered. Returns an `Arc` so the per-subdomain
/// recovery reads it without cloning the maps.
pub fn active_head_f64_folds() -> Option<Arc<[HeadF64Fold]>> {
    if !head_f64_fold_enabled() && !head_f64_measure_arm_enabled() {
        return None;
    }
    registry().read().ok()?.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_and_registry_semantics() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_MN_HEAD_FACET");
            let mut post = HashMap::new();
            post.insert(3u32, (0.5f64, 1e-16f64));
            let mut pre = HashMap::new();
            pre.insert(2u32, (0.75f64, 2e-16f64));
            set_head_f64_folds(vec![HeadF64Fold {
                target_act: 0,
                relu_name: "Relu_head".to_string(),
                head_width: 8,
                post,
                pre,
                bias_shift: -1.25,
                bias_err: 3e-16,
            }]);
            // Default (gate unset) ⇒ inert even with an entry registered.
            assert!(active_head_f64_folds().is_none());

            // The environment request cannot grant verdict authority.
            env.set("NY_MN_HEAD_FACET", "1");
            assert!(!head_f64_fold_enabled());
            assert!(active_head_f64_folds().is_none());

            clear_head_f64_folds();
            assert!(active_head_f64_folds().is_none());
        });
    }

    #[test]
    fn measure_arm_reader_arms_only_under_its_own_var() {
        ny_test_utils::env::with_env_edits(|env| {
            env.remove("NY_MN_HEAD_FACET");
            env.remove("NY_MN_HEAD_F64_MEASURE_ARM");
            let mut post = HashMap::new();
            post.insert(3u32, (0.5f64, 1e-16f64));
            set_head_f64_folds(vec![HeadF64Fold {
                target_act: 0,
                relu_name: "Relu_head".to_string(),
                head_width: 8,
                post,
                pre: HashMap::new(),
                bias_shift: -1.25,
                bias_err: 3e-16,
            }]);
            // Production authority stays quarantined; the production var does NOT
            // arm the reader (byte-identical to today).
            env.set("NY_MN_HEAD_FACET", "1");
            assert!(!head_f64_fold_enabled());
            assert!(!head_f64_measure_arm_enabled());
            assert!(active_head_f64_folds().is_none());

            // The dedicated measurement var arms the READER (and only the reader);
            // the production gate remains false.
            env.set("NY_MN_HEAD_F64_MEASURE_ARM", "1");
            assert!(head_f64_measure_arm_enabled());
            assert!(!head_f64_fold_enabled());
            assert!(active_head_f64_folds().is_some());

            clear_head_f64_folds();
            assert!(active_head_f64_folds().is_none());
        });
    }

    #[test]
    fn empty_fold_is_reported_empty() {
        let f = HeadF64Fold {
            target_act: 0,
            relu_name: "R".to_string(),
            head_width: 4,
            post: HashMap::new(),
            pre: HashMap::new(),
            bias_shift: 0.0,
            bias_err: 0.0,
        };
        assert!(f.is_empty());
        assert!(f.post_terms().is_empty());
        assert!(f.pre_terms().is_empty());
    }
}

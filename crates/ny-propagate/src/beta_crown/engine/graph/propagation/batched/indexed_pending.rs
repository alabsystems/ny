// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Vec-indexed pending-bounds storage for batched beta-CROWN backward propagation.
//!
//! This mirrors the graph-CROWN dispatch-plan pattern but stores per-domain
//! `LinearBounds` payloads for the batched beta-CROWN reverse traversal.

use std::collections::HashMap;

use ndarray::Array2;
use ny_core::{NyError, Result};

use crate::network::CrownDispatchPlan;
use crate::{GraphNetwork, LinearBounds};

/// #lsnc-fast-merge-err gate. The certified coefficient-error merge
/// (`fill_merged_err`) is a hot leaf on the SERIAL batched-CROWN backward
/// critical path of the graph input-split BaB (MEASURED on real lsnc_relu after
/// the relaxed_clip fast path landed: `merge_coeff_err` is the single largest
/// attackable backward primitive). The scalar reference walks the error matrix
/// with per-element `ndarray::Index` after allocating a zeroed `Array2` that it
/// immediately overwrites. `build_merged_err_fast` computes the SAME arithmetic
/// in the SAME row-major order over flat standard-layout slices and builds the
/// output `Vec` directly, so it is BIT-IDENTICAL (see
/// `test_merge_coeff_err_fast_scalar_parity`). Default ON; set
/// `NY_INPUT_SPLIT_FAST_MERGE_ERR=0` to force the scalar reference (the A/B +
/// parity baseline), mirroring `NY_RELAXED_CLIP_FAST` / `NY_INPUT_SPLIT_*`.
fn fast_merge_coeff_err_enabled() -> bool {
    !matches!(
        std::env::var("NY_INPUT_SPLIT_FAST_MERGE_ERR")
            .ok()
            .as_deref(),
        Some("0") | Some("false")
    )
}

/// f32 unit roundoff (round-to-nearest): 2^-24. Shared by the coefficient-error
/// merge in this module and the SoA batched-backward merge (#lsnc-batched-bwd).
pub(super) const U_F32: f64 = 5.960464477539063e-08;

/// Single element of the `#dag-merge-bias` outward-rounded bias merge (the 2Sum
/// body of `accumulate_idx`'s `merge_bias`). Extracted so the SoA batched
/// backward (`batched_bwd.rs`) applies the IDENTICAL per-element arithmetic —
/// bit-parity by construction. See the block comment at the call site in
/// [`IndexedPendingLinearBounds::accumulate_idx`] for the full soundness
/// argument (#vnncomp-aw-soundness).
#[inline]
pub(super) fn merge_bias_elem(o: f32, bv: f32, is_lower: bool) -> f32 {
    let conservative = if is_lower {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let a64 = o as f64;
    let b64 = bv as f64;
    let s = a64 + b64;
    if s.is_nan() {
        conservative
    } else {
        // 2Sum (Knuth): a64 + b64 == s + e exactly; e == 0 iff the
        // f64 add was exact. The exact real sum is therefore s + e.
        let bp = s - a64;
        let ap = s - bp;
        let e = (a64 - ap) + (b64 - bp);
        let c = s as f32;
        if is_lower {
            if c == f32::INFINITY {
                // s > f32::MAX ⇒ true sum > MAX; MAX ≤ true (sound).
                f32::MAX
            } else {
                // keep c iff c ≤ s + e (== true); d = c − s is exact.
                let d = c as f64 - s;
                if d <= e {
                    c
                } else {
                    ny_tensor::next_down_f32(c)
                }
            }
        } else if c == f32::NEG_INFINITY {
            // s < −f32::MAX ⇒ true sum < −MAX; MIN ≥ true (sound).
            f32::MIN
        } else {
            let d = c as f64 - s;
            if d >= e {
                c
            } else {
                ny_tensor::next_up_f32(c)
            }
        }
    }
}

/// Single element of the certified coefficient-error merge:
/// `round_up( existing + new + U_F32·|merged_coeff| )` with the non-finite
/// degrade to `+inf`. Shared by `fill_merged_err` / `build_merged_err_fast`
/// below and the SoA batched-backward merge (#lsnc-batched-bwd) so all legs are
/// bit-identical per element (guarded by
/// `test_merge_coeff_err_fast_scalar_parity`).
#[inline]
pub(super) fn merged_err_elem(existing: Option<f32>, new: Option<f32>, merged_coeff: f32) -> f32 {
    let mut acc = U_F32 * (merged_coeff as f64).abs();
    if let Some(ee) = existing {
        let v = ee as f64;
        acc = if v.is_finite() {
            acc + v
        } else {
            f64::INFINITY
        };
    }
    if let Some(ne) = new {
        let v = ne as f64;
        acc = if v.is_finite() {
            acc + v
        } else {
            f64::INFINITY
        };
    }
    if acc.is_finite() && acc >= 0.0 {
        let cast = acc as f32;
        let up = ny_tensor::next_up_f32(cast);
        if up.is_finite() {
            up
        } else {
            f32::INFINITY
        }
    } else {
        f32::INFINITY
    }
}

pub(super) struct IndexedPendingLinearBounds {
    storage: Vec<Option<Vec<Option<LinearBounds>>>>,
    name_to_idx: HashMap<String, usize>,
    network_input_idx: usize,
    n_domains: usize,
    input_accumulated: Vec<bool>,
}

impl IndexedPendingLinearBounds {
    pub(super) fn new(plan: &CrownDispatchPlan, n_domains: usize) -> Self {
        let capacity = plan.node_count() + 1;
        Self {
            storage: vec![None; capacity],
            name_to_idx: plan.name_to_idx.clone(),
            network_input_idx: plan.network_input_idx,
            n_domains,
            input_accumulated: vec![false; n_domains],
        }
    }

    pub(super) fn seed_idx(
        &mut self,
        idx: usize,
        domain_idx: usize,
        bounds: LinearBounds,
    ) -> Result<()> {
        self.validate_domain(domain_idx)?;
        let slot = self.storage[idx].get_or_insert_with(|| vec![None; self.n_domains]);
        slot[domain_idx] = Some(bounds);
        if idx == self.network_input_idx {
            self.input_accumulated[domain_idx] = true;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn seed_name(
        &mut self,
        name: &str,
        domain_idx: usize,
        bounds: LinearBounds,
    ) -> Result<()> {
        let idx = self.index_of(name)?;
        self.seed_idx(idx, domain_idx, bounds)
    }

    #[inline]
    pub(super) fn take_idx(&mut self, idx: usize) -> Option<Vec<Option<LinearBounds>>> {
        self.storage[idx].take()
    }

    #[inline]
    pub(super) fn take_network_input(&mut self) -> Option<Vec<Option<LinearBounds>>> {
        self.take_idx(self.network_input_idx)
    }

    #[inline]
    pub(super) fn input_accumulated(&self) -> &[bool] {
        &self.input_accumulated
    }

    pub(super) fn accumulate_name(
        &mut self,
        input_name: &str,
        new_bounds: LinearBounds,
        domain_idx: usize,
    ) -> Result<()> {
        let idx = self.index_of(input_name)?;
        self.accumulate_idx(idx, new_bounds, domain_idx)
    }

    pub(super) fn accumulate_idx(
        &mut self,
        idx: usize,
        new_bounds: LinearBounds,
        domain_idx: usize,
    ) -> Result<()> {
        self.validate_domain(domain_idx)?;
        if idx == self.network_input_idx {
            self.input_accumulated[domain_idx] = true;
        }

        let slot = self.storage[idx].get_or_insert_with(|| vec![None; self.n_domains]);
        if let Some(existing) = &mut slot[domain_idx] {
            let new_la = GraphNetwork::safe_add(existing.lower_a(), new_bounds.lower_a(), true);
            let new_ua = GraphNetwork::safe_add(existing.upper_a(), new_bounds.upper_a(), false);
            // #dag-merge-bias: the residual/DAG bias merge must round OUTWARD. Each
            // stored bias is already a sound directed bound (lower_b <= true <= upper_b);
            // a plain round-to-nearest f32 add of two biases can round the lower bias UP
            // / the upper bias DOWN past the true merged bias, which concretize then
            // trusts as exact (biases carry no error term) — a false-tight certificate.
            // The f64 add of two f32 operands is NOT always exact (operands whose binary
            // exponents differ by more than ~29 need >53 significant bits), so a 2Sum
            // recovers the exact real sum `s + e` and the f32 cast is then directed
            // against `s + e` rather than the rounded `s` (self-audit fix: the old
            // `c <= s` test could keep an f32 that exceeds the true sum by up to ~½ ULP
            // of f64). f32 overflow is clamped to ±MAX (next_down/up_f32 are no-ops on
            // ±inf, so an overflowed lower bias would otherwise stay +inf — unsound).
            // inf + (-inf) cancellation degrades to the conservative sentinel.
            // #vnncomp-aw-soundness.
            let merge_bias =
                |a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>, is_lower: bool| {
                    let conservative = if is_lower {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                    if a.len() != b.len() {
                        return ndarray::Array1::from_elem(a.len(), conservative);
                    }
                    let mut out = a.clone();
                    for (o, &bv) in out.iter_mut().zip(b.iter()) {
                        *o = merge_bias_elem(*o, bv, is_lower);
                    }
                    out
                };
            let new_lb = merge_bias(existing.lower_b(), new_bounds.lower_b(), true);
            let new_ub = merge_bias(existing.upper_b(), new_bounds.upper_b(), false);
            // Merge the certified coefficient error BEFORE overwriting the
            // coefficients (#vnncomp-aw-soundness). The f32 `safe_add` of two
            // error-carrying contributions sums their true real coefficients, so the
            // merged error is `err_existing + err_new` PLUS the f32 add roundoff
            // (bounded by 2^-24·|sum|, the f32 unit roundoff — this path accumulates
            // coefficients in f32). Dropping either input's error here was the bug:
            // a residual/DAG merge would silently lose the per-contribution error
            // even though `propagates_coeff_err` is declared true downstream.
            let merged_lower_err =
                Self::merge_coeff_err(existing.lower_a_err(), new_bounds.lower_a_err(), &new_la);
            let merged_upper_err =
                Self::merge_coeff_err(existing.upper_a_err(), new_bounds.upper_a_err(), &new_ua);
            *existing.lower_a_mut() = new_la;
            *existing.lower_b_mut() = new_lb;
            *existing.upper_a_mut() = new_ua;
            *existing.upper_b_mut() = new_ub;
            match (merged_lower_err, merged_upper_err) {
                (Some(le), Some(ue)) => existing.set_coeff_err(le, ue),
                (None, None) => {}
                // Only one side carried error: still must attach BOTH (the other
                // side gets a zero error matrix) so the carried side is never
                // silently dropped by set_coeff_err's pairing.
                (Some(le), None) => {
                    let ue = Array2::<f32>::zeros(existing.upper_a().raw_dim());
                    existing.set_coeff_err(le, ue);
                }
                (None, Some(ue)) => {
                    let le = Array2::<f32>::zeros(existing.lower_a().raw_dim());
                    existing.set_coeff_err(le, ue);
                }
            }
        } else {
            slot[domain_idx] = Some(new_bounds);
        }
        Ok(())
    }

    /// Merge two certified per-coefficient error matrices for an f32-accumulated
    /// coefficient merge (#vnncomp-aw-soundness).
    ///
    /// Returns `existing_err + new_err + roundoff`, where `roundoff[i,j] =
    /// 2^-24·|merged_coeff[i,j]|` bounds the f32 add's rounding error (f32 unit
    /// roundoff `u = 2^-24`, round-to-nearest). All terms are accumulated in f64
    /// to avoid the merge itself losing the small terms, then rounded UP to a
    /// sound f32. A non-finite result becomes `f32::INFINITY` (degrade the row).
    /// Returns `None` only when neither input carries error AND the roundoff is
    /// entirely zero (exact merge).
    fn merge_coeff_err(
        existing_err: Option<&Array2<f32>>,
        new_err: Option<&Array2<f32>>,
        merged_coeff: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        let shape = merged_coeff.raw_dim();
        let shapes_ok = |e: Option<&Array2<f32>>| match e {
            Some(a) => a.raw_dim() == shape,
            None => true,
        };
        // If an error matrix is present but mis-shaped, we cannot map it soundly:
        // degrade every coefficient so concretize widens the rows.
        if !shapes_ok(existing_err) || !shapes_ok(new_err) {
            return Some(Array2::<f32>::from_elem(shape, f32::INFINITY));
        }
        if existing_err.is_none() && new_err.is_none() {
            // No carried error. The f32 add still introduced roundoff; only skip
            // attaching error when the merged coefficient row is trivially exact
            // (all zero), which keeps the common identity/seed merge error-free.
            if merged_coeff.iter().all(|&v| v == 0.0) {
                return None;
            }
        }
        let out = if fast_merge_coeff_err_enabled() {
            // #lsnc-fast-merge-err: build the output directly from flat row-major
            // slices, skipping both the per-element `ndarray::Index` and the
            // zero-init that the scalar path immediately overwrites.
            Self::build_merged_err_fast(existing_err, new_err, merged_coeff, U_F32)
        } else {
            let mut out = Array2::<f32>::zeros(shape);
            Self::fill_merged_err(&mut out, existing_err, new_err, merged_coeff, U_F32);
            out
        };
        Some(out)
    }

    /// Flat-slice, zero-init-free reimplementation of [`fill_merged_err`] that
    /// returns the built matrix directly. BIT-IDENTICAL: the scalar path walks
    /// `[[i, j]]` in row-major order (i outer, j inner), which for standard-layout
    /// arrays is exactly the flat index `i*ncols + j`; every input is normalized to
    /// standard layout so the k-th flat element aligns across `merged_coeff`,
    /// `existing_err`, and `new_err`, and each element applies the identical f64
    /// accumulation, `>= 0.0` finiteness gate, and directed `next_up_f32` rounding.
    /// The zeroed `Array2` is dropped because every element was overwritten anyway.
    /// Guarded by `test_merge_coeff_err_fast_scalar_parity`. #lsnc-fast-merge-err.
    fn build_merged_err_fast(
        existing_err: Option<&Array2<f32>>,
        new_err: Option<&Array2<f32>>,
        merged_coeff: &Array2<f32>,
        u: f64,
    ) -> Array2<f32> {
        let shape = merged_coeff.raw_dim();
        let mc = merged_coeff.as_standard_layout();
        let mc_s = mc
            .as_slice()
            .expect("merged_coeff standard layout contiguous");
        let ee = existing_err.map(|a| a.as_standard_layout());
        let ee_s = ee.as_ref().map(|a| {
            a.as_slice()
                .expect("existing_err standard layout contiguous")
        });
        let ne = new_err.map(|a| a.as_standard_layout());
        let ne_s = ne
            .as_ref()
            .map(|a| a.as_slice().expect("new_err standard layout contiguous"));

        debug_assert!((u - U_F32).abs() == 0.0, "u must be the f32 unit roundoff");
        let mut out = Vec::with_capacity(mc_s.len());
        for k in 0..mc_s.len() {
            out.push(merged_err_elem(
                ee_s.map(|s| s[k]),
                ne_s.map(|s| s[k]),
                mc_s[k],
            ));
        }
        Array2::from_shape_vec(shape, out).expect("build_merged_err_fast row-major shape")
    }

    /// Fill `out[i,j] = round_up( existing[i,j] + new[i,j] + u·|merged[i,j]| )`.
    fn fill_merged_err(
        out: &mut Array2<f32>,
        existing_err: Option<&Array2<f32>>,
        new_err: Option<&Array2<f32>>,
        merged_coeff: &Array2<f32>,
        u: f64,
    ) {
        debug_assert!((u - U_F32).abs() == 0.0, "u must be the f32 unit roundoff");
        let n = merged_coeff.nrows();
        let m = merged_coeff.ncols();
        for i in 0..n {
            for j in 0..m {
                out[[i, j]] = merged_err_elem(
                    existing_err.map(|e| e[[i, j]]),
                    new_err.map(|e| e[[i, j]]),
                    merged_coeff[[i, j]],
                );
            }
        }
    }

    fn index_of(&self, name: &str) -> Result<usize> {
        self.name_to_idx.get(name).copied().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "IndexedPendingLinearBounds: unknown input '{}'",
                name
            ))
        })
    }

    fn validate_domain(&self, domain_idx: usize) -> Result<()> {
        if domain_idx >= self.n_domains {
            return Err(NyError::InvalidSpec(format!(
                "IndexedPendingLinearBounds: domain_idx {} out of range for {} domains",
                domain_idx, self.n_domains
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn get_name(&self, name: &str) -> Option<&[Option<LinearBounds>]> {
        self.name_to_idx
            .get(name)
            .and_then(|&idx| self.storage[idx].as_deref())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array2};

    use super::*;
    use crate::layers::{Layer, LinearLayer};
    use crate::{GraphNetwork, GraphNode, NETWORK_INPUT};

    /// `build_merged_err_fast` (the flat-slice, zero-init-free coefficient-error
    /// merge) must be BIT-IDENTICAL to the scalar `fill_merged_err` reference for
    /// every {existing_err, new_err} presence combination and across value regimes
    /// (finite, zero, ±inf, tiny/huge) on lsnc-shaped matrices — this is the
    /// certified error term, so any divergence is a soundness bug. Compares raw
    /// f32 bit patterns. #lsnc-fast-merge-err.
    #[ntest::timeout(30000)]
    #[test]
    fn test_merge_coeff_err_fast_scalar_parity() {
        const U_F32: f64 = 5.960464477539063e-08;
        let (n, m) = (39usize, 6usize); // lsnc-shaped: 39 spec rows × 6 input coeffs

        // Deterministic pseudo-random fixture (LCG) with occasional ±inf / zero
        // entries to exercise the finiteness gates and infinity degradation.
        let mut state: u64 = 0xD1B54A32D192ED03;
        let mut nextf = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / (u32::MAX as f32);
            match (state >> 20) & 0x1f {
                0 => f32::INFINITY,
                1 => f32::NEG_INFINITY,
                2 => 0.0,
                _ => (u - 0.5) * 1e3, // spread of magnitudes
            }
        };
        let mut mk = || {
            let mut a = Array2::<f32>::zeros((n, m));
            for v in a.iter_mut() {
                *v = nextf();
            }
            a
        };

        let merged_coeff = mk();
        let err_a = mk();
        let err_b = mk();

        for existing in [None, Some(&err_a)] {
            for new in [None, Some(&err_b)] {
                let mut scalar = Array2::<f32>::zeros(merged_coeff.raw_dim());
                IndexedPendingLinearBounds::fill_merged_err(
                    &mut scalar,
                    existing,
                    new,
                    &merged_coeff,
                    U_F32,
                );
                let fast = IndexedPendingLinearBounds::build_merged_err_fast(
                    existing,
                    new,
                    &merged_coeff,
                    U_F32,
                );
                assert_eq!(scalar.raw_dim(), fast.raw_dim());
                let s = scalar.as_slice().unwrap();
                let f = fast.as_slice().unwrap();
                for i in 0..s.len() {
                    assert_eq!(
                        s[i].to_bits(),
                        f[i].to_bits(),
                        "merge_coeff_err bit mismatch at {i} (existing={}, new={}): scalar={} fast={}",
                        existing.is_some(),
                        new.is_some(),
                        s[i],
                        f[i]
                    );
                }
            }
        }
    }

    fn make_plan(node_name: &str) -> CrownDispatchPlan {
        let mut graph = GraphNetwork::new();
        graph
            .try_add_node(GraphNode::from_input(
                node_name,
                Layer::Linear(
                    LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32])))
                        .expect("valid linear layer"),
                ),
            ))
            .expect("node should be added");
        graph.set_output(node_name);
        CrownDispatchPlan::build(&graph).expect("dispatch plan should build")
    }

    #[test]
    fn test_indexed_pending_seed_and_take_round_trip_4417() {
        let plan = make_plan("out");
        let output_idx = plan.output_node_idx;
        let mut pending = IndexedPendingLinearBounds::new(&plan, 2);
        let seeded = LinearBounds::identity(1);

        pending
            .seed_idx(output_idx, 1, seeded.clone())
            .expect("seed should succeed");

        let taken = pending
            .take_idx(output_idx)
            .expect("output slot should exist");
        assert!(taken[0].is_none(), "domain 0 should remain empty");
        let taken_seed = taken[1].as_ref().expect("domain 1 seed should exist");
        assert_eq!(
            taken_seed.lower_a, seeded.lower_a,
            "take_idx should preserve lower_a"
        );
        assert_eq!(
            taken_seed.lower_b, seeded.lower_b,
            "take_idx should preserve lower_b"
        );
        assert_eq!(
            taken_seed.upper_a, seeded.upper_a,
            "take_idx should preserve upper_a"
        );
        assert_eq!(
            taken_seed.upper_b, seeded.upper_b,
            "take_idx should preserve upper_b"
        );
        assert!(
            pending.take_idx(output_idx).is_none(),
            "slot should be empty after take_idx"
        );
    }

    #[test]
    fn test_indexed_pending_tracks_network_input_4417() {
        let plan = make_plan("out");
        let mut pending = IndexedPendingLinearBounds::new(&plan, 1);

        pending
            .accumulate_name(NETWORK_INPUT, LinearBounds::identity(1), 0)
            .expect("network-input accumulation should succeed");

        assert!(
            pending.input_accumulated()[0],
            "accumulating to _input must mark the domain as reaching network input"
        );
        assert!(
            pending.take_network_input().is_some(),
            "network-input slot should hold the accumulated bounds"
        );
    }

    /// #vnncomp-aw-soundness REPRO: the batched beta-CROWN accumulate path used to
    /// `safe_add` coefficients while DROPPING each contribution's certified
    /// coefficient error — a residual/DAG merge silently lost the per-contribution
    /// error even though the path declares `propagates_coeff_err = true`. After
    /// merging two error-carrying contributions the merged certified error must
    /// cover `|stored_f32_merged − true_real_merged|` for every coefficient.
    ///
    /// On the OLD code the merged bounds keep only `existing`'s STALE error (the
    /// new contribution's error vanishes), so a coefficient that genuinely cancels
    /// loses coverage → certified < gap → FAILS. The patched code carries
    /// `err1 + err2 + roundoff` → PASSES.
    #[test]
    fn test_indexed_pending_carries_coeff_err_across_merge_aw_soundness() {
        use ndarray::Array2;

        let plan = make_plan("node1");
        let mut pending = IndexedPendingLinearBounds::new(&plan, 1);

        // Two contributions on 3 inputs whose TRUE coefficients cancel; stored f32
        // differs from truth by a certified error.
        let true1 = [5.000_000_2_f64, -3.000_000_4, 1.000_000_05];
        let true2 = [-5.000_000_1_f64, 3.000_000_2, -1.000_000_02];
        let mk = |t: &[f64]| -> LinearBounds {
            let stored: Vec<f32> = t.iter().map(|&v| v as f32).collect();
            let err: Vec<f32> = (0..t.len())
                .map(|j| ((stored[j] as f64 - t[j]).abs() as f32) * 2.0 + 1e-7)
                .collect();
            let a = Array2::from_shape_vec((1, t.len()), stored).unwrap();
            let e = Array2::from_shape_vec((1, t.len()), err).unwrap();
            LinearBounds::new_or_conservative_with_err(
                a.clone(),
                arr1(&[0.0_f32]),
                a,
                arr1(&[0.0_f32]),
                e.clone(),
                e,
            )
            .unwrap()
        };

        pending.seed_name("node1", 0, mk(&true1)).unwrap();
        pending.accumulate_name("node1", mk(&true2), 0).unwrap();

        let merged = pending
            .get_name("node1")
            .and_then(|per_domain| per_domain[0].as_ref())
            .expect("merged bounds should exist");
        let merged_err = merged.lower_a_err().expect(
            "patched batched accumulate must carry a certified coefficient error \
             across a DAG merge (OLD code drops the new contribution's error)",
        );

        for j in 0..3 {
            let stored_merged = merged.lower_a()[[0, j]] as f64;
            let true_merged = true1[j] + true2[j];
            let gap = (stored_merged - true_merged).abs();
            let cert = merged_err[[0, j]] as f64;
            assert!(
                cert >= gap,
                "UNSOUND batched merge certificate at col {j}: certified {cert:.3e} < \
                 |stored_f32_merged − true_real_merged| {gap:.3e}"
            );
        }
    }

    #[test]
    fn test_indexed_pending_nan_safe_merge_parity_4417() {
        let plan = make_plan("node1");
        let mut pending = IndexedPendingLinearBounds::new(&plan, 1);
        let existing = LinearBounds {
            lower_a: arr2(&[[f32::NEG_INFINITY, 1.0_f32]]),
            lower_b: arr1(&[f32::NEG_INFINITY]),
            upper_a: arr2(&[[f32::INFINITY, 2.0_f32]]),
            upper_b: arr1(&[f32::INFINITY]),
            lower_a_err: None,
            upper_a_err: None,
        };
        let new_bounds = LinearBounds {
            lower_a: arr2(&[[f32::INFINITY, 3.0_f32]]),
            lower_b: arr1(&[f32::INFINITY]),
            upper_a: arr2(&[[f32::NEG_INFINITY, 4.0_f32]]),
            upper_b: arr1(&[f32::NEG_INFINITY]),
            lower_a_err: None,
            upper_a_err: None,
        };

        pending
            .seed_name("node1", 0, existing)
            .expect("initial seed should succeed");
        pending
            .accumulate_name("node1", new_bounds, 0)
            .expect("merge should succeed");

        let result = pending
            .get_name("node1")
            .and_then(|per_domain| per_domain[0].as_ref())
            .expect("merged bounds should exist");

        assert_eq!(result.lower_a[[0, 0]], f32::NEG_INFINITY);
        assert_eq!(result.lower_b[0], f32::NEG_INFINITY);
        assert_eq!(result.upper_a[[0, 0]], f32::INFINITY);
        assert_eq!(result.upper_b[0], f32::INFINITY);
        assert!((result.lower_a[[0, 1]] - 4.0_f32).abs() < 1e-6);
        assert!((result.upper_a[[0, 1]] - 6.0_f32).abs() < 1e-6);
    }
}

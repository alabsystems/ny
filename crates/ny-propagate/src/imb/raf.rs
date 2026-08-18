// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reduced-Affine-Form (RAF) forward for per-leaf prefix anchors — the port of the
//! numpy reference `raf_forward` (`imb_certify2.py` / `imb_diag_anchors.py`).
//!
//! Carries, per prefix neuron, its LINEAR dependence on the `n_free` free input
//! dims as a coefficient matrix `A` [dim × n_free] plus an interval remainder
//! `[Rl, Ru]` [dim]. The value enclosure over a leaf's free-dim box `[xl_f, xu_f]`
//! is `A·x_free + [Rl, Ru]`, concretized (numpy) as
//! `lo = Ap·xl + An·xu + Rl`, `hi = Ap·xu + An·xl + Ru` (`Ap=max(A,0)`,
//! `An=min(A,0)`). Because the affine form keeps the input CORRELATION (vs
//! independent intervals), a small leaf gets a MUCH tighter anchor — the per-leaf
//! gain the crown_root root anchor cannot provide.
//!
//! # Soundness
//!
//! The form `(A, [Rl,Ru])` is maintained as a valid ENCLOSURE of the true reachable
//! set as a function of `x_free`:
//! - **Linear layer** `f(v)=L·v+b`: `node_out ∈ (L·A)·x_free + (L·[Rl,Ru]+b)`. We
//!   compute `A' = point_ibp(A_col) − point_ibp(0)` (the layer's linear part via
//!   ny's per-layer IBP), bound the f32 error `|A' − L·A| ≤ a_err` from the two
//!   forwards' radii, propagate the remainder through the SAME sound `propagate_ibp`
//!   (`L·[Rl,Ru]+b`), and fold `Σ_k a_err[k]·max(|xl_f[k]|,|xu_f[k]|)` OUTWARD into
//!   the remainder (directed rounding). Induction holds with the *approximate* `A`
//!   as the coefficient — we never need `A` to be the exact derivative, only that
//!   `(A,[Rl,Ru])` encloses the node.
//! - **ReLU**: per neuron, `pos` keep; `neg` zero; `unstable` pick `lam1`
//!   (keep `A`, `Ru += |zl|`, sound since `v ≤ ReLU(v) ≤ v+|zl|`) or `lam0` (zero
//!   `A`, `[0, max(0,zu)]`), the smaller-slack of the two.
//! - Only KNOWN-AFFINE non-ReLU layers are propagated; any other op aborts the RAF
//!   (`None`) and the caller falls back to the (sound) root-crown anchor.
//!
//! Every produced box is intersected with the sound root-crown anchor by the
//! caller, so the result is a sound enclosure regardless. The whole path is
//! log-only (STAGE 1 returns baseline).

use std::collections::{HashMap, HashSet};

use ny_core::dd::{next_down_f64, next_up_f64};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use ndarray::{ArrayD, IxDyn};

use crate::bounds::{certified_affine_sum_f32, OutwardDirection};
use crate::layers::{BoundPropagation, Layer};
use crate::{GraphNetwork, NETWORK_INPUT};

/// A non-ReLU layer whose forward is affine (`f(v)=L·v+b`), so `point_ibp(col) −
/// point_ibp(0)` recovers its exact linear map. Anything else aborts the RAF.
fn is_affine_prefix_layer(layer: &Layer) -> bool {
    matches!(
        layer,
        Layer::Linear(_)
            | Layer::Conv1d(_)
            | Layer::Conv2d(_)
            | Layer::ConvTranspose1d(_)
            | Layer::ConvTranspose2d(_)
            | Layer::BatchNorm(_)
            | Layer::Reshape(_)
            | Layer::Flatten(_)
            | Layer::Transpose(_)
    )
}

/// Build a degenerate (point) `BoundedTensor` of `shape` from a flat buffer.
fn point_bt(flat: &[f32], shape: &[usize]) -> Option<BoundedTensor> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), flat.to_vec()).ok()?;
    BoundedTensor::new(arr.clone(), arr).ok()
}

/// Concretize `A·x_free + [Rl,Ru]` over `[xl_f, xu_f]` with OUTWARD directed
/// rounding — `lo = Ap·xl + An·xu + Rl`, `hi = Ap·xu + An·xl + Ru` (f64 accum).
fn raf_concretize(
    a_cols: &[Vec<f32>],
    rem_lo: &[f32],
    rem_hi: &[f32],
    xl_f: &[f32],
    xu_f: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let dim = rem_lo.len();
    let nf = a_cols.len();
    let mut lo = vec![0.0f32; dim];
    let mut hi = vec![0.0f32; dim];
    for i in 0..dim {
        let acc_lo = certified_affine_sum_f32(
            rem_lo[i],
            (0..nf).map(|k| {
                let a = a_cols[k][i];
                (a, if a >= 0.0 { xl_f[k] } else { xu_f[k] })
            }),
            OutwardDirection::Lower,
        );
        let acc_hi = certified_affine_sum_f32(
            rem_hi[i],
            (0..nf).map(|k| {
                let a = a_cols[k][i];
                (a, if a >= 0.0 { xu_f[k] } else { xl_f[k] })
            }),
            OutwardDirection::Upper,
        );
        lo[i] = next_down_f32(acc_lo as f32);
        hi[i] = next_up_f32(acc_hi as f32);
    }
    (lo, hi)
}

/// In-place RAF ReLU relaxation on the pre-activation form (numpy lines 58-66).
fn relu_relax(
    a_cols: &mut [Vec<f32>],
    rem_lo: &mut [f32],
    rem_hi: &mut [f32],
    xl_f: &[f32],
    xu_f: &[f32],
) {
    let (zl, zu) = raf_concretize(a_cols, rem_lo, rem_hi, xl_f, xu_f);
    let dim = rem_lo.len();
    let nf = a_cols.len();
    for i in 0..dim {
        if zl[i] >= 0.0 {
            // pos: ReLU is identity — keep A, [Rl,Ru].
        } else if zu[i] <= 0.0 {
            // neg: ReLU is 0.
            for col in a_cols.iter_mut() {
                col[i] = 0.0;
            }
            rem_lo[i] = 0.0;
            rem_hi[i] = 0.0;
        } else if zu[i] >= -zl[i] {
            // unstable lam1: keep A; ReLU(v) ∈ [v, v+|zl|] ⇒ Ru += |zl| (outward).
            rem_hi[i] = next_up_f32(rem_hi[i] - zl[i]); // -zl[i] = |zl[i]|
        } else {
            // unstable lam0: ReLU(v) ∈ [0, max(0,zu)] ⇒ zero A, [0, max(0,zu)].
            for col in a_cols.iter_mut() {
                col[i] = 0.0;
            }
            rem_lo[i] = 0.0;
            rem_hi[i] = next_up_f32(zu[i].max(0.0));
        }
        let _ = nf;
    }
}

/// RAF forward over one leaf's input box `leaf`. Returns, for each node in
/// `record` (the prefix ReLU-source / pre-activation nodes), the concretized RAF
/// box — a sound, correlation-preserving per-leaf enclosure. `None` if the prefix
/// is not a clean affine+ReLU chain (caller falls back to the root-crown anchor).
pub(super) fn raf_forward(
    prefix: &GraphNetwork,
    leaf: &BoundedTensor,
    free_dims: &[usize],
    record: &HashSet<String>,
) -> Option<HashMap<String, BoundedTensor>> {
    let exec = prefix.exec_order().ok()?;
    let flat = leaf.flatten();
    let in_lo = flat.lower().as_slice()?.to_vec();
    let in_hi = flat.upper().as_slice()?.to_vec();
    let in_shape = leaf.lower().shape().to_vec();
    let in_dim = in_lo.len();
    let nf = free_dims.len();
    let xl_f: Vec<f32> = free_dims.iter().map(|&d| in_lo[d]).collect();
    let xu_f: Vec<f32> = free_dims.iter().map(|&d| in_hi[d]).collect();

    // INIT at the network input: A = free-dim selector; remainder carries fixed dims.
    let free_set: HashSet<usize> = free_dims.iter().copied().collect();
    let mut a_cols: Vec<Vec<f32>> = vec![vec![0.0f32; in_dim]; nf];
    for (k, &d) in free_dims.iter().enumerate() {
        a_cols[k][d] = 1.0;
    }
    let mut rem_lo = vec![0.0f32; in_dim];
    let mut rem_hi = vec![0.0f32; in_dim];
    for i in 0..in_dim {
        if !free_set.contains(&i) {
            rem_lo[i] = in_lo[i];
            rem_hi[i] = in_hi[i];
        }
    }
    let mut cur_shape = in_shape;
    let mut prev = NETWORK_INPUT.to_string();

    let mut out: HashMap<String, BoundedTensor> = HashMap::new();

    for name in exec {
        let node = prefix.nodes.get(name)?;
        // Clean-chain requirement: each node consumes the immediately-previous one.
        if node.inputs.first().map(String::as_str) != Some(prev.as_str()) {
            return None;
        }

        if matches!(node.layer, Layer::ReLU(_)) {
            relu_relax(&mut a_cols, &mut rem_lo, &mut rem_hi, &xl_f, &xu_f);
            // shape unchanged
        } else {
            if !is_affine_prefix_layer(&node.layer) {
                return None; // unknown nonlinear op — abort RAF (caller falls back)
            }
            let dim = rem_lo.len();
            // f(0) = bias (± tiny outward rounding).
            let z_out = node
                .layer
                .propagate_ibp(&point_bt(&vec![0.0f32; dim], &cur_shape)?)
                .ok()?;
            let zf = z_out.flatten();
            let z_lo = zf.lower().as_slice()?.to_vec();
            let z_hi = zf.upper().as_slice()?.to_vec();
            let out_shape = z_out.lower().shape().to_vec();
            let out_dim = z_lo.len();
            // f64 center/radius of f(0) (the bias) to avoid center-arithmetic ULP loss.
            // Bit-identical (a+b)*0.5 anchors: f32-cast operands cannot overflow
            // f64, and the literal form is the file's center convention.
            #[allow(clippy::manual_midpoint)]
            let mid_0: Vec<f64> = (0..out_dim)
                .map(|i| 0.5 * (z_lo[i] as f64 + z_hi[i] as f64))
                .collect();
            let rad_0: Vec<f64> = (0..out_dim)
                .map(|i| 0.5 * (z_hi[i] as f64 - z_lo[i] as f64))
                .collect();

            // A' column k = point_ibp(A_col_k) − f(0); a_err bounds |A' − L·A_col|.
            let mut new_a = vec![vec![0.0f32; out_dim]; nf];
            let mut a_err = vec![vec![0.0f32; out_dim]; nf];
            for k in 0..nf {
                let o = node
                    .layer
                    .propagate_ibp(&point_bt(&a_cols[k], &cur_shape)?)
                    .ok()?;
                let of = o.flatten();
                let olo = of.lower();
                let olo = olo.as_slice()?;
                let ohi = of.upper();
                let ohi = ohi.as_slice()?;
                #[allow(clippy::manual_midpoint)]
                for i in 0..out_dim {
                    // Center in f64, then store as f32 (round-to-nearest).
                    let na = 0.5 * (olo[i] as f64 + ohi[i] as f64) - mid_0[i];
                    new_a[k][i] = na as f32;
                    // Forward-interval radius (f64) bounds |mid − true_coeff|; add the
                    // one-ULP storage rounding of new_a, and round the whole slack
                    // OUTWARD (next_up_f32) so a_err NEVER under-estimates the coeff
                    // error — even when the point-IBP forward is exact (rad=0), where
                    // round-to-nearest could otherwise collapse the slack to zero.
                    let rad = 0.5 * (ohi[i] as f64 - olo[i] as f64) + rad_0[i];
                    let na_abs = new_a[k][i].abs();
                    let store_ulp = (next_up_f32(na_abs) - na_abs) as f64;
                    a_err[k][i] = next_up_f32((rad + store_ulp).max(0.0) as f32);
                }
            }

            // Remainder: sound propagate_ibp of [Rl,Ru] (= L·[Rl,Ru]+b), then fold
            // the coeff-error slack OUTWARD.
            let rem_bt = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&cur_shape), rem_lo.clone()).ok()?,
                ArrayD::from_shape_vec(IxDyn(&cur_shape), rem_hi.clone()).ok()?,
            )
            .ok()?;
            let rem_out = node.layer.propagate_ibp(&rem_bt).ok()?;
            let rrf = rem_out.flatten();
            let mut rlo = rrf.lower().as_slice()?.to_vec();
            let mut rhi = rrf.upper().as_slice()?.to_vec();
            for i in 0..out_dim {
                let slack = certified_affine_sum_f32(
                    0.0,
                    (0..nf).map(|k| {
                        let magnitude = xl_f[k].abs().max(xu_f[k].abs());
                        (a_err[k][i], magnitude)
                    }),
                    OutwardDirection::Upper,
                );
                if slack.is_finite() && slack > 0.0 {
                    rlo[i] = next_down_f32(next_down_f64(rlo[i] as f64 - slack) as f32);
                    rhi[i] = next_up_f32(next_up_f64(rhi[i] as f64 + slack) as f32);
                }
            }

            a_cols = new_a;
            rem_lo = rlo;
            rem_hi = rhi;
            cur_shape = out_shape;
        }

        if record.contains(name) {
            let (lo, hi) = raf_concretize(&a_cols, &rem_lo, &rem_hi, &xl_f, &xu_f);
            let bt = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&cur_shape), lo).ok()?,
                ArrayD::from_shape_vec(IxDyn(&cur_shape), hi).ok()?,
            )
            .ok()?;
            out.insert(name.clone(), bt);
        }
        prev = name.clone();
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::raf_concretize;

    #[test]
    fn concretize_survives_cancellation_larger_than_final_f32_ulp() {
        let large = 2.0_f32.powi(50);
        let columns = vec![vec![large], vec![1.0], vec![-large]];
        let (lower, upper) = raf_concretize(
            &columns,
            &[0.0],
            &[0.0],
            &[large, 1.0, large],
            &[large, 1.0, large],
        );

        assert!(lower[0] <= 1.0);
        assert!(
            upper[0] >= 1.0,
            "RAF upper {} must enclose exact 2^100 + 1 - 2^100",
            upper[0]
        );
    }
}

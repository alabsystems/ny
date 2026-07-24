// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified Cut-CROWN increment C1.5: L1 cut GENERATION from a real conv
//! layer (`docs/CERTIFIED_CUT_CROWN_DESIGN.md` §C1).
//!
//! The first ReLU layer's pre-activations are EXACTLY affine in the network
//! input (`z = conv1(x) + b`), so joint k-ReLU cut bounds over the input box
//! are derivable analytically (per-coordinate corner max — no enumeration
//! over the 3072-dim box; only the ≤2^k activation patterns).
//!
//! Group choice: **same spatial position, top-k channels by upper bound** —
//! those neurons share their ENTIRE receptive field, the maximal-correlation
//! regime. Measured on the real cifar100 resnet_medium conv1 at the real
//! ε=0.0039 box (2026-07-10, `l1_cut_slack.py`): median 25.6% / p90 41% /
//! max 55% relative slack vs the per-neuron bound sum; 8.55 absolute units
//! of tightening from one k=3 group per position (root deficit ≈ 4.1).
//!
//! Soundness: bounds come from [`derive_cut_bound`] (outward-rounded,
//! mirrors the proven `multiReluCut_box_le` schema). Group selection is
//! advisory — a badly chosen group only yields a uselessly loose (still
//! valid) cut.

use super::multi_relu_cut::{derive_cut_bound, AffineRow, MultiReluCut, SplitState};
use crate::layers::Conv2dLayer;

/// A generated L1 cut with its group's layer coordinates.
#[derive(Debug, Clone)]
pub struct L1Cut {
    /// The cut (group's FLAT neuron indices within the conv output, weights, bound).
    pub cut: MultiReluCut,
    /// The affine pre-activation rows of the group (kept for split re-derivation).
    pub rows: Vec<AffineRow>,
}

impl L1Cut {
    /// Re-derive this cut's bound under the given per-group-neuron split
    /// states (C3, the strengthening flywheel). Returns the tightened bound;
    /// callers keep `min(old, new)` per subdomain.
    pub fn rederive(&self, splits: &[SplitState], xl: &[f32], xu: &[f32]) -> f32 {
        derive_cut_bound(&self.rows, &self.cut.cc, splits, xl, xu)
    }
}

/// Build the sparse affine row of one conv output neuron `(oc, oy, ox)` over
/// the flattened input (`c*ih*iw + y*iw + x` layout, matching NY's flatten).
fn conv_neuron_row(
    conv: &Conv2dLayer,
    in_hw: (usize, usize),
    oc: usize,
    oy: usize,
    ox: usize,
) -> AffineRow {
    let ksh = conv.kernel.shape();
    let (in_c_g, kh, kw) = (ksh[1], ksh[2], ksh[3]);
    let (ih, iw) = in_hw;
    let out_c = ksh[0];
    let g = conv.groups;
    let group_of_oc = oc / (out_c / g);
    let mut w: Vec<(u32, f32)> = Vec::with_capacity(in_c_g * kh * kw);
    for icg in 0..in_c_g {
        let ic = group_of_oc * in_c_g + icg; // absolute input channel
        for dy in 0..kh {
            for dx in 0..kw {
                let y =
                    (oy * conv.stride.0 + dy * conv.dilation.0) as isize - conv.padding.0 as isize;
                let x =
                    (ox * conv.stride.1 + dx * conv.dilation.1) as isize - conv.padding.1 as isize;
                if y < 0 || x < 0 || y as usize >= ih || x as usize >= iw {
                    continue; // zero padding
                }
                let widx = (ic * ih * iw + y as usize * iw + x as usize) as u32;
                let wv = conv.kernel[[oc, icg, dy, dx]];
                if wv != 0.0 {
                    w.push((widx, wv));
                }
            }
        }
    }
    w.sort_unstable_by_key(|&(j, _)| j);
    let r = conv.bias.as_ref().map_or(0.0, |b| b[oc]);
    AffineRow { w, r }
}

/// Affine box max via the per-coordinate rule (f64, upper).
fn row_box_max(row: &AffineRow, xl: &[f32], xu: &[f32]) -> f64 {
    let mut t = f64::from(row.r);
    for &(j, w) in &row.w {
        let (lo, hi) = (f64::from(xl[j as usize]), f64::from(xu[j as usize]));
        let w = f64::from(w);
        t += (w * lo).max(w * hi);
    }
    t
}

/// Conv output spatial dims under the layer's stride/padding/dilation.
fn conv_out_hw(conv: &Conv2dLayer, in_hw: (usize, usize)) -> (usize, usize) {
    let ksh = conv.kernel.shape();
    let (kh, kw) = (ksh[2], ksh[3]);
    let eff_kh = conv.dilation.0 * (kh - 1) + 1;
    let eff_kw = conv.dilation.1 * (kw - 1) + 1;
    let oh = (in_hw.0 + 2 * conv.padding.0).saturating_sub(eff_kh) / conv.stride.0 + 1;
    let ow = (in_hw.1 + 2 * conv.padding.1).saturating_sub(eff_kw) / conv.stride.1 + 1;
    (oh, ow)
}

/// Enumerate the UNSTABLE channels at output position `(oy, ox)`:
/// `(oc, row, lo, hi)` with exact affine box bounds `lo < 0 < hi`.
fn unstable_candidates_at(
    conv: &Conv2dLayer,
    in_hw: (usize, usize),
    oy: usize,
    ox: usize,
    xl: &[f32],
    xu: &[f32],
) -> Vec<(usize, AffineRow, f64, f64)> {
    let out_c = conv.kernel.shape()[0];
    let mut cand = Vec::new();
    for oc in 0..out_c {
        let row = conv_neuron_row(conv, in_hw, oc, oy, ox);
        if row.w.is_empty() {
            continue;
        }
        let hi = row_box_max(&row, xl, xu);
        if hi <= 0.0 {
            continue; // stably inactive
        }
        // lower = -max(-row)
        let neg = AffineRow {
            w: row.w.iter().map(|&(j, w)| (j, -w)).collect(),
            r: -row.r,
        };
        let lo = -row_box_max(&neg, xl, xu);
        if lo >= 0.0 {
            continue; // stably active — relu is exact, no relaxation to cut
        }
        cand.push((oc, row, lo, hi));
    }
    cand
}

/// Generate same-position top-`k` L1 cuts for every output position with at
/// least `k` unstable channels. `xl`/`xu` are the flattened input box.
///
/// Unstable = pre-activation lower < 0 < upper (exact, since L1 is affine).
/// `cc = 1` for every group member (the measured-slack configuration).
pub fn generate_l1_cuts(conv: &Conv2dLayer, xl: &[f32], xu: &[f32], k: usize) -> Vec<L1Cut> {
    debug_assert!((2..=8).contains(&k));
    let Some(in_hw) = conv.input_shape else {
        return Vec::new();
    };
    let (oh, ow) = conv_out_hw(conv, in_hw);

    let mut cuts = Vec::new();
    for oy in 0..oh {
        for ox in 0..ow {
            // Per-channel exact bounds at this position.
            let mut cand: Vec<(usize, AffineRow, f64)> =
                unstable_candidates_at(conv, in_hw, oy, ox, xl, xu)
                    .into_iter()
                    .map(|(oc, row, _lo, hi)| (oc, row, hi))
                    .collect();
            if cand.len() < k {
                continue;
            }
            // Top-k channels by upper bound (largest independent-sum first).
            cand.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            cand.truncate(k);
            let rows: Vec<AffineRow> = cand.iter().map(|(_, r, _)| r.clone()).collect();
            let cc = vec![1.0f32; k];
            let bound = derive_cut_bound(&rows, &cc, &vec![SplitState::Free; k], xl, xu);
            let neurons: Vec<u32> = cand
                .iter()
                .map(|(oc, _, _)| (oc * oh * ow + oy * ow + ox) as u32)
                .collect();
            cuts.push(L1Cut {
                cut: MultiReluCut { neurons, cc, bound },
                rows,
            });
        }
    }
    cuts
}

/// `cc` weighting for objective-signed groups (C2b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignedCcMode {
    /// `cc_i = 1` for every group member.
    Uniform,
    /// `cc_i = |a_i| / max_j |a_j|` within the group (max-normalized to 1) —
    /// fold mass proportional to the objective coefficient it cancels.
    PropAbsA,
}

/// Diagnostics of one objective-signed generation pass (C2b): how much of the
/// objective row's negative coefficient mass at the cut ReLU the groups
/// cover, and the first-order price/recovery ledger.
#[derive(Debug, Clone, Default)]
pub struct SignedCutDiag {
    /// `Σ min(a_i, 0)` over ALL neurons of the layer (stable + unstable).
    pub total_neg_mass: f64,
    /// `Σ min(a_i, 0)` restricted to UNSTABLE neurons (the only ones whose
    /// lower relaxation has slack for the fold to recover).
    pub unstable_neg_mass: f64,
    /// `Σ a_i` over the grouped neurons (all strictly negative).
    pub covered_neg_mass: f64,
    /// `Σ B` over the emitted cuts — the bias paid per unit λ.
    pub sum_b: f64,
    /// `Σ cc_i · u_i·(−l_i)/(u_i−l_i)` over grouped neurons: the upper-chord
    /// INTERCEPT recovered per unit λ while `a_i + λ·cc_i` stays < 0. First-
    /// order only — ignores the chord-slope propagation term — so it is a
    /// headroom indicator, not a bound-delta prediction.
    pub intercept_recovery_rate: f64,
    /// Groups whose members share one spatial position (tight shared-RF `B`).
    pub same_pos_groups: usize,
    /// Leftover groups assembled across positions (valid, looser `B`).
    pub cross_pos_groups: usize,
    /// Total neurons placed into emitted groups.
    pub grouped_neurons: usize,
    /// Unstable neurons with `a_i < 0` (the selection pool).
    pub negative_unstable_neurons: usize,
}

/// C2b: generate OBJECTIVE-SIGNED k-ReLU L1 cuts — groups chosen among
/// unstable neurons whose incoming lower-side objective coefficient `a_i`
/// (captured at the fold ReLU frontier for one spec row, flat
/// `oc·oh·ow + oy·ow + ox` layout) is strictly NEGATIVE. Those are exactly
/// the neurons where the lower backward pays the ReLU upper-chord intercept
/// `|a_i|·u(−l)/(u−l)`, which the `+λ·cc_i` fold cancels; folding onto
/// `a_i ≥ 0` neurons only adds mass and pays `λ·B` for nothing (the measured
/// C2 sign-blind negative — `docs/CERTIFIED_CUT_CROWN_DESIGN.md`).
///
/// Grouping: per spatial position, most-negative-`a` first, chunked into
/// disjoint groups of `k` (shared receptive field keeps `B` tight); the
/// per-position leftovers are pooled, re-sorted by `a`, and chunked into
/// cross-position groups (still valid — `derive_cut_bound` handles arbitrary
/// groups; `B` is just looser).
///
/// `positive_first_order_only`: keep a group only when its first-order
/// intercept recovery rate `Σ cc_i·u_i(−l_i)/(u_i−l_i)` exceeds its bias
/// price `B` (both per unit λ) — drop groups that lose at first order.
///
/// Soundness: unchanged from [`generate_l1_cuts`] — bounds come from
/// [`derive_cut_bound`] for the exact `cc` used; the objective row only
/// steers SELECTION. Diagnostics accumulate over emitted groups only.
pub fn generate_l1_cuts_signed(
    conv: &Conv2dLayer,
    xl: &[f32],
    xu: &[f32],
    k: usize,
    obj_a: &[f32],
    mode: SignedCcMode,
    positive_first_order_only: bool,
) -> (Vec<L1Cut>, SignedCutDiag) {
    debug_assert!((2..=8).contains(&k));
    let mut diag = SignedCutDiag::default();
    let Some(in_hw) = conv.input_shape else {
        return (Vec::new(), diag);
    };
    let (oh, ow) = conv_out_hw(conv, in_hw);
    let out_c = conv.kernel.shape()[0];
    if obj_a.len() != out_c * oh * ow {
        debug_assert!(
            false,
            "generate_l1_cuts_signed: obj_a len {} != conv output {}x{}x{}",
            obj_a.len(),
            out_c,
            oh,
            ow
        );
        return (Vec::new(), diag);
    }
    diag.total_neg_mass = obj_a.iter().map(|&a| f64::from(a).min(0.0)).sum();

    // Selection pool: unstable AND a < 0, bucketed per position.
    // (flat, row, lo, hi, a)
    type Cand = (u32, AffineRow, f64, f64, f64);
    let mut per_pos: Vec<Vec<Cand>> = vec![Vec::new(); oh * ow];
    for oy in 0..oh {
        for ox in 0..ow {
            let pos = oy * ow + ox;
            for (oc, row, lo, hi) in unstable_candidates_at(conv, in_hw, oy, ox, xl, xu) {
                let flat = oc * oh * ow + pos;
                let a = f64::from(obj_a[flat]);
                diag.unstable_neg_mass += a.min(0.0);
                if a < 0.0 {
                    per_pos[pos].push((flat as u32, row, lo, hi, a));
                }
            }
        }
    }
    diag.negative_unstable_neurons = per_pos.iter().map(Vec::len).sum();

    // Same-position groups first (most-negative-a first), leftovers pooled.
    let mut groups: Vec<(Vec<Cand>, bool)> = Vec::new(); // (members, same_pos)
    let mut leftovers: Vec<Cand> = Vec::new();
    for mut bucket in per_pos {
        bucket.sort_by(|x, y| x.4.partial_cmp(&y.4).unwrap_or(std::cmp::Ordering::Equal));
        let mut it = bucket.into_iter();
        loop {
            let chunk: Vec<Cand> = it.by_ref().take(k).collect();
            if chunk.len() == k {
                groups.push((chunk, true));
            } else {
                leftovers.extend(chunk);
                break;
            }
        }
    }
    leftovers.sort_by(|x, y| x.4.partial_cmp(&y.4).unwrap_or(std::cmp::Ordering::Equal));
    let mut it = leftovers.into_iter();
    loop {
        let chunk: Vec<Cand> = it.by_ref().take(k).collect();
        if chunk.len() == k {
            groups.push((chunk, false));
        } else {
            break; // drop the <k tail
        }
    }

    let mut cuts = Vec::new();
    for (members, same_pos) in groups {
        let rows: Vec<AffineRow> = members.iter().map(|(_, r, _, _, _)| r.clone()).collect();
        let cc: Vec<f32> = match mode {
            SignedCcMode::Uniform => vec![1.0; k],
            SignedCcMode::PropAbsA => {
                let amax = members
                    .iter()
                    .map(|(_, _, _, _, a)| a.abs())
                    .fold(0.0f64, f64::max);
                members
                    .iter()
                    .map(|(_, _, _, _, a)| (a.abs() / amax) as f32)
                    .collect()
            }
        };
        let bound = derive_cut_bound(&rows, &cc, &vec![SplitState::Free; k], xl, xu);
        // First-order ledger: chord intercept recovered vs bias paid, per unit λ.
        let rate: f64 = members
            .iter()
            .zip(&cc)
            .map(|((_, _, lo, hi, _), &c)| f64::from(c) * hi * (-lo) / (hi - lo))
            .sum();
        if positive_first_order_only && rate <= f64::from(bound) {
            continue;
        }
        diag.covered_neg_mass += members.iter().map(|(_, _, _, _, a)| a).sum::<f64>();
        diag.sum_b += f64::from(bound);
        diag.intercept_recovery_rate += rate;
        diag.grouped_neurons += k;
        if same_pos {
            diag.same_pos_groups += 1;
        } else {
            diag.cross_pos_groups += 1;
        }
        let neurons: Vec<u32> = members.iter().map(|(flat, _, _, _, _)| *flat).collect();
        cuts.push(L1Cut {
            cut: MultiReluCut { neurons, cc, bound },
            rows,
        });
    }
    (cuts, diag)
}

/// C3 diagnostic record for one same-position candidate group anchored on a
/// BaB-split first-ReLU neuron (`generate_l1_cuts_for_splits`).
#[derive(Debug, Clone)]
pub struct L1SplitGroupDiag {
    /// Per-member split state (split members `Active`/`Inactive`; the
    /// unstable neighbors filling the group are `Free`).
    pub states: Vec<SplitState>,
    /// Unconditional bound (all-`Free` derivation) for THIS group.
    pub root_b: f32,
    /// Bound re-derived under the split premises
    /// (`multiReluCut_split_box_le` — the C3 strengthening).
    pub strengthened_b: f32,
    /// Split members in the group.
    pub n_split: usize,
    /// Split members that are unstable on the box (a box-stable split member
    /// contributes a premise but no relaxation slack of its own).
    pub n_split_unstable: usize,
    /// The group's cut payload (neurons in flat layout, `cc = 1`, bound =
    /// `root_b`) with its affine rows — ready for a per-domain fold.
    pub cut: L1Cut,
}

/// C3 diagnostic generation: for every conv-output position holding at least
/// one SPLIT neuron, build the same-position candidate group = the split
/// neuron(s) at that position plus the top unstable neighbors (by upper
/// bound) up to `k` members — exactly the [`generate_l1_cuts`] grouping
/// restricted to positions the subdomain actually split — and derive both the
/// root bound and the split-strengthened bound.
///
/// `splits` are `(flat neuron index, is_active)` premises on the FIRST ReLU
/// (flat `oc·oh·ow + oy·ow + ox` layout). Positions whose group would have a
/// single member (no unstable neighbor and one split) are skipped — a k=1
/// "joint" cut is just the neuron's own bound. Duplicate flats keep the first
/// premise; out-of-range flats are ignored.
///
/// Soundness: unchanged from [`generate_l1_cuts`] — both bounds come from
/// [`derive_cut_bound`]; split selection only steers grouping.
pub fn generate_l1_cuts_for_splits(
    conv: &Conv2dLayer,
    xl: &[f32],
    xu: &[f32],
    k: usize,
    splits: &[(u32, bool)],
) -> Vec<L1SplitGroupDiag> {
    debug_assert!((2..=8).contains(&k));
    let Some(in_hw) = conv.input_shape else {
        return Vec::new();
    };
    let (oh, ow) = conv_out_hw(conv, in_hw);
    let out_c = conv.kernel.shape()[0];
    let plane = oh * ow;

    // Bucket split premises per output position (dedup by flat index).
    let mut per_pos: std::collections::BTreeMap<usize, Vec<(u32, bool)>> =
        std::collections::BTreeMap::new();
    for &(flat, act) in splits {
        let f = flat as usize;
        if f >= out_c * plane {
            continue;
        }
        let bucket = per_pos.entry(f % plane).or_default();
        if !bucket.iter().any(|&(g, _)| g == flat) {
            bucket.push((flat, act));
        }
    }

    let mut out = Vec::new();
    for (pos, mut split_members) in per_pos {
        let (oy, ox) = (pos / ow, pos % ow);
        split_members.truncate(k);
        let cand = unstable_candidates_at(conv, in_hw, oy, ox, xl, xu);

        // Split members first (rows built directly — exact even when the
        // member is box-stable), then unstable non-split neighbors by upper.
        let mut neurons: Vec<u32> = Vec::with_capacity(k);
        let mut rows: Vec<AffineRow> = Vec::with_capacity(k);
        let mut states: Vec<SplitState> = Vec::with_capacity(k);
        let mut n_split_unstable = 0usize;
        for &(flat, act) in &split_members {
            let oc = flat as usize / plane;
            let row = match cand.iter().find(|(c, _, _, _)| *c == oc) {
                Some((_, r, _, _)) => {
                    n_split_unstable += 1;
                    r.clone()
                }
                None => conv_neuron_row(conv, in_hw, oc, oy, ox),
            };
            neurons.push(flat);
            rows.push(row);
            states.push(if act {
                SplitState::Active
            } else {
                SplitState::Inactive
            });
        }
        let mut neighbors: Vec<&(usize, AffineRow, f64, f64)> = cand
            .iter()
            .filter(|(oc, _, _, _)| {
                !split_members
                    .iter()
                    .any(|&(f, _)| f as usize / plane == *oc)
            })
            .collect();
        neighbors.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        for (oc, row, _, _) in neighbors.into_iter().take(k.saturating_sub(neurons.len())) {
            neurons.push((oc * plane + pos) as u32);
            rows.push(row.clone());
            states.push(SplitState::Free);
        }
        if neurons.len() < 2 {
            continue; // no joint cut at this position
        }

        let cc = vec![1.0f32; neurons.len()];
        let root_b = derive_cut_bound(&rows, &cc, &vec![SplitState::Free; rows.len()], xl, xu);
        let strengthened_b = derive_cut_bound(&rows, &cc, &states, xl, xu);
        out.push(L1SplitGroupDiag {
            n_split: split_members.len(),
            n_split_unstable,
            root_b,
            strengthened_b,
            states,
            cut: L1Cut {
                cut: MultiReluCut {
                    neurons,
                    cc,
                    bound: root_b,
                },
                rows,
            },
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, ArrayD, IxDyn};

    /// 2-channel 2x2-kernel conv on a 3x3 single-channel input, stride 1:
    /// channels see the same patch per position ⇒ joint cut ≤ independent sum,
    /// and both dominate the exact ReLU sum on a grid.
    #[test]
    fn generated_cuts_sound_and_no_looser_than_independent() {
        let mut kernel = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 2, 2]));
        // ch0 = x00 + x01 - x10; ch1 = -x00 + x01 + x11
        kernel[[0, 0, 0, 0]] = 1.0;
        kernel[[0, 0, 0, 1]] = 1.0;
        kernel[[0, 0, 1, 0]] = -1.0;
        kernel[[1, 0, 0, 0]] = -1.0;
        kernel[[1, 0, 0, 1]] = 1.0;
        kernel[[1, 0, 1, 1]] = 1.0;
        let conv = Conv2dLayer {
            kernel,
            bias: Some(Array1::from(vec![0.1f32, -0.1])),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            input_shape: Some((3, 3)),
        };
        let n = 9;
        let xl = vec![-1.0f32; n];
        let xu = vec![1.0f32; n];
        let cuts = generate_l1_cuts(&conv, &xl, &xu, 2);
        assert!(!cuts.is_empty(), "expected cuts at 2x2 output positions");

        for c in &cuts {
            // Independent bound: sum of per-neuron uppers.
            let indep: f64 = c
                .rows
                .iter()
                .map(|r| row_box_max(r, &xl, &xu).max(0.0))
                .sum();
            assert!(
                f64::from(c.cut.bound) <= indep + 1e-6,
                "joint bound {} must be ≤ independent {indep}",
                c.cut.bound
            );

            // Soundness vs exact ReLU sum on a coarse grid over the 4 live dims.
            let live: Vec<u32> = {
                let mut s: Vec<u32> = c
                    .rows
                    .iter()
                    .flat_map(|r| r.w.iter().map(|&(j, _)| j))
                    .collect();
                s.sort_unstable();
                s.dedup();
                s
            };
            let steps = 4;
            let mut idx = vec![0usize; live.len()];
            loop {
                let mut x = vec![0.0f64; n];
                for (d, &j) in live.iter().enumerate() {
                    x[j as usize] = -1.0 + 2.0 * (idx[d] as f64) / (steps as f64);
                }
                let v: f64 = c
                    .rows
                    .iter()
                    .map(|r| {
                        let z: f64 = f64::from(r.r)
                            + r.w
                                .iter()
                                .map(|&(j, w)| f64::from(w) * x[j as usize])
                                .sum::<f64>();
                        z.max(0.0)
                    })
                    .sum();
                assert!(
                    v <= f64::from(c.cut.bound) + 1e-9,
                    "cut violated at {x:?}: {v} > {}",
                    c.cut.bound
                );
                // odometer
                let mut d = 0;
                loop {
                    if d == idx.len() {
                        break;
                    }
                    idx[d] += 1;
                    if idx[d] <= steps {
                        break;
                    }
                    idx[d] = 0;
                    d += 1;
                }
                if d == idx.len() {
                    break;
                }
            }
        }
    }

    /// C2b signed generation: only negative-`a` unstable neurons are grouped
    /// (same-position first, leftovers cross-position), the diagnostics ledger
    /// adds up, cc modes behave, the first-order filter drops losing groups,
    /// and every emitted cut is SOUND vs the exact ReLU sum on a grid.
    #[test]
    fn signed_generation_selects_negative_a_and_stays_sound() {
        let mut kernel = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 2, 2]));
        // ch0 = x00 + x01 - x10 + 0.1; ch1 = -x00 + x01 + x11 - 0.1
        kernel[[0, 0, 0, 0]] = 1.0;
        kernel[[0, 0, 0, 1]] = 1.0;
        kernel[[0, 0, 1, 0]] = -1.0;
        kernel[[1, 0, 0, 0]] = -1.0;
        kernel[[1, 0, 0, 1]] = 1.0;
        kernel[[1, 0, 1, 1]] = 1.0;
        let conv = Conv2dLayer {
            kernel,
            bias: Some(Array1::from(vec![0.1f32, -0.1])),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            input_shape: Some((3, 3)),
        };
        let n = 9;
        let xl = vec![-1.0f32; n];
        let xu = vec![1.0f32; n];
        // 2x2 output, flat = oc*4 + pos. Both channels are unstable at every
        // position on this box (checked by unstable_candidates_at coverage).

        // Case 1: only ch0 negative — one negative-a neuron per position, so
        // NO same-position pair exists; the 4 leftovers form 2 cross-position
        // pairs (most-negative first).
        let mut a = vec![1.0f32; 8];
        for p in 0..4 {
            a[p] = -(1.0 + p as f32); // ch0
        }
        let (cuts, diag) =
            generate_l1_cuts_signed(&conv, &xl, &xu, 2, &a, SignedCcMode::Uniform, false);
        assert_eq!(diag.same_pos_groups, 0);
        assert_eq!(diag.cross_pos_groups, 2);
        assert_eq!(cuts.len(), 2);
        assert_eq!(diag.grouped_neurons, 4);
        assert_eq!(diag.negative_unstable_neurons, 4);
        assert!(
            (diag.total_neg_mass + 10.0).abs() < 1e-6,
            "{}",
            diag.total_neg_mass
        );
        assert!((diag.unstable_neg_mass + 10.0).abs() < 1e-6);
        assert!((diag.covered_neg_mass + 10.0).abs() < 1e-6);
        assert!(diag.sum_b > 0.0 && diag.intercept_recovery_rate > 0.0);
        for c in &cuts {
            assert!(
                c.cut.neurons.iter().all(|&i| i < 4),
                "only ch0 (negative-a) neurons may be grouped: {:?}",
                c.cut.neurons
            );
        }

        // Case 2: everything negative ⇒ 4 same-position pairs; PropAbsA cc is
        // max-normalized to 1 and strictly positive.
        let a2: Vec<f32> = (0..8).map(|i| -1.0 - 0.25 * i as f32).collect();
        let (cuts2, diag2) =
            generate_l1_cuts_signed(&conv, &xl, &xu, 2, &a2, SignedCcMode::PropAbsA, false);
        assert_eq!(diag2.same_pos_groups, 4);
        assert_eq!(diag2.cross_pos_groups, 0);
        for c in &cuts2 {
            assert!(c.cut.cc.iter().all(|&w| w > 0.0 && w <= 1.0));
            assert!(c.cut.cc.iter().any(|&w| (w - 1.0).abs() < 1e-6));
        }

        // First-order filter: on this geometry every group's chord-intercept
        // rate is below its B, so the filtered set must shrink (here: empty).
        let (cuts_f, diag_f) =
            generate_l1_cuts_signed(&conv, &xl, &xu, 2, &a2, SignedCcMode::Uniform, true);
        assert!(cuts_f.len() < cuts2.len());
        assert_eq!(diag_f.grouped_neurons, cuts_f.len() * 2);

        // Soundness of every emitted cut (both cases) vs the exact ReLU sum.
        for c in cuts.iter().chain(&cuts2) {
            let live: Vec<u32> = {
                let mut s: Vec<u32> = c
                    .rows
                    .iter()
                    .flat_map(|r| r.w.iter().map(|&(j, _)| j))
                    .collect();
                s.sort_unstable();
                s.dedup();
                s
            };
            let steps = 4;
            let mut idx = vec![0usize; live.len()];
            loop {
                let mut x = vec![0.0f64; n];
                for (d, &j) in live.iter().enumerate() {
                    x[j as usize] = -1.0 + 2.0 * (idx[d] as f64) / (steps as f64);
                }
                let v: f64 = c
                    .rows
                    .iter()
                    .zip(&c.cut.cc)
                    .map(|(r, &w)| {
                        let z: f64 = f64::from(r.r)
                            + r.w
                                .iter()
                                .map(|&(j, wj)| f64::from(wj) * x[j as usize])
                                .sum::<f64>();
                        f64::from(w) * z.max(0.0)
                    })
                    .sum();
                assert!(
                    v <= f64::from(c.cut.bound) + 1e-9,
                    "signed cut violated at {x:?}: {v} > {}",
                    c.cut.bound
                );
                let mut d = 0;
                loop {
                    if d == idx.len() {
                        break;
                    }
                    idx[d] += 1;
                    if idx[d] <= steps {
                        break;
                    }
                    idx[d] = 0;
                    d += 1;
                }
                if d == idx.len() {
                    break;
                }
            }
        }
    }

    /// C3 diagnostic generation: groups anchor on the split neuron, states
    /// mark it Active/Inactive, and the strengthened bound matches the Lean
    /// `split_strengthening_strict` witness (root B = 1, Act ⇒ B = 1/2).
    #[test]
    fn split_group_diags_match_strengthening_witness() {
        let mut kernel = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 1, 1]));
        kernel[[0, 0, 0, 0]] = 1.0;
        kernel[[1, 0, 0, 0]] = -1.0;
        let conv = Conv2dLayer {
            kernel,
            bias: Some(Array1::from(vec![0.0f32, -0.5])),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            input_shape: Some((1, 1)),
        };
        let (xl, xu) = (vec![-1.0f32], vec![1.0f32]);
        // z0 = x (flat 0), z1 = -x - 1/2 (flat 1) on [-1,1]: exactly the Lean
        // witness geometry. Split premise: z1 ACTIVE.
        let diags = generate_l1_cuts_for_splits(&conv, &xl, &xu, 2, &[(1, true)]);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(
            d.cut.cut.neurons,
            vec![1, 0],
            "split member leads the group"
        );
        assert_eq!(d.states, vec![SplitState::Active, SplitState::Free]);
        assert_eq!(d.n_split, 1);
        assert_eq!(d.n_split_unstable, 1);
        assert!(
            (f64::from(d.root_b) - 1.0).abs() < 1e-5,
            "root B = {}",
            d.root_b
        );
        assert!(
            (f64::from(d.strengthened_b) - 0.5).abs() < 1e-5,
            "strengthened B = {}",
            d.strengthened_b
        );
        assert!(d.strengthened_b < d.root_b);
        // Inactive premise on z0: patterns exclude 0 ⇒ B = max(0, 1/2) = 1/2.
        let diags2 = generate_l1_cuts_for_splits(&conv, &xl, &xu, 2, &[(0, false)]);
        assert_eq!(diags2.len(), 1);
        assert!((f64::from(diags2[0].strengthened_b) - 0.5).abs() < 1e-5);
        // Out-of-range flats are ignored; a lone split with no neighbor at a
        // 1-channel conv position yields no group.
        assert!(generate_l1_cuts_for_splits(&conv, &xl, &xu, 2, &[(99, true)]).is_empty());
    }

    /// Split re-derivation tightens (C3 flywheel contract).
    #[test]
    fn rederive_under_split_is_monotone() {
        let mut kernel = ArrayD::<f32>::zeros(IxDyn(&[2, 1, 1, 1]));
        kernel[[0, 0, 0, 0]] = 1.0;
        kernel[[1, 0, 0, 0]] = -1.0;
        let conv = Conv2dLayer {
            kernel,
            bias: Some(Array1::from(vec![0.0f32, -0.5])),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            input_shape: Some((1, 1)),
        };
        let (xl, xu) = (vec![-1.0f32], vec![1.0f32]);
        let cuts = generate_l1_cuts(&conv, &xl, &xu, 2);
        assert_eq!(cuts.len(), 1);
        let c = &cuts[0];
        // This is exactly the Lean split_strengthening_strict geometry:
        // z0 = x, z1 = -x - 1/2 on [-1,1]: root B = 1, Act on z1 ⇒ B = 1/2.
        // Group ordering is by upper bound: u(z0)=1 > u(z1)=1/2 ⇒ index 1 is z1.
        assert!(
            (f64::from(c.cut.bound) - 1.0).abs() < 1e-5,
            "root B = {}",
            c.cut.bound
        );
        let b2 = c.rederive(&[SplitState::Free, SplitState::Active], &xl, &xu);
        assert!((f64::from(b2) - 0.5).abs() < 1e-5, "strengthened B = {b2}");
        assert!(b2 < c.cut.bound);
    }
}

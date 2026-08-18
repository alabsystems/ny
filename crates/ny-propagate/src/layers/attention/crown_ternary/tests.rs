// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness + tightening tests for the fused self-attention ternary CROWN
//! backward. The soundness proptest is the GATE: over thousands of sampled
//! inputs in the joint Q/K/V box, the concretized linear bound MUST enclose the
//! true attention output (0 violations), for standard AND causal masking.

use super::*;
use crate::layers::attention::{AttentionMask, SelfAttentionLayer};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Build a BoundedTensor box from center + radius, given shape.
fn box_from_center(shape: &[usize], centers: &[f32], radii: &[f32]) -> BoundedTensor {
    let lo: Vec<f32> = centers
        .iter()
        .zip(radii.iter())
        .map(|(&c, &r)| c - r)
        .collect();
    let hi: Vec<f32> = centers
        .iter()
        .zip(radii.iter())
        .map(|(&c, &r)| c + r)
        .collect();
    let l = ArrayD::from_shape_vec(IxDyn(shape), lo).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(shape), hi).unwrap();
    BoundedTensor::new(l, u).unwrap()
}

/// True attention forward (concrete) for one (sampled) Q/K/V point. Returns the
/// flattened Y. Mirrors the math the surrogate linearizes.
fn attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dims: &SliceDims,
    scale: f32,
    mask: AttentionMask,
    window: Option<usize>,
) -> Vec<f32> {
    let SliceDims {
        n_slices,
        sq,
        sk,
        dk,
        dv,
    } = *dims;
    let mut out = vec![0.0f32; n_slices * sq * dv];
    for slice in 0..n_slices {
        let qb = slice * sq * dk;
        let kb = slice * sk * dk;
        let vb = slice * sk * dv;
        let yb = slice * sq * dv;
        for i in 0..sq {
            // scores over visible keys
            let mut scores = vec![f64::NEG_INFINITY; sk];
            let mut maxs = f64::NEG_INFINITY;
            for kk in 0..sk {
                let visible = match mask {
                    AttentionMask::Standard => true,
                    AttentionMask::Causal => kk <= i && window.map(|w| i - kk <= w).unwrap_or(true),
                };
                if !visible {
                    continue;
                }
                let mut dot = 0.0f64;
                for d in 0..dk {
                    dot += q[qb + i * dk + d] as f64 * k[kb + kk * dk + d] as f64;
                }
                let s = scale as f64 * dot;
                scores[kk] = s;
                if s > maxs {
                    maxs = s;
                }
            }
            let mut probs = vec![0.0f64; sk];
            let mut sum = 0.0f64;
            for kk in 0..sk {
                if scores[kk].is_finite() {
                    let e = (scores[kk] - maxs).exp();
                    probs[kk] = e;
                    sum += e;
                }
            }
            for kk in 0..sk {
                probs[kk] /= sum;
            }
            for j in 0..dv {
                let mut y = 0.0f64;
                for kk in 0..sk {
                    y += probs[kk] * v[vb + kk * dv + j] as f64;
                }
                out[yb + i * dv + j] = y as f32;
            }
        }
    }
    out
}

/// Concretize the Nary result over the joint input box: returns per-output
/// `(lower, upper)`. `node_lb` here is identity (Y -> Y), so the result is the
/// CROWN bound on the attention output itself.
#[allow(clippy::too_many_arguments)]
fn concretize_nary(
    bounds: &[Option<LinearBounds>],
    bias_lo: &Array1<f32>,
    bias_hi: &Array1<f32>,
    qbox: &BoundedTensor,
    kbox: &BoundedTensor,
    vbox: &BoundedTensor,
) -> (Vec<f32>, Vec<f32>) {
    let n_out = bias_lo.len();
    let boxes = [qbox, kbox, vbox];
    let mut lower = vec![0.0f64; n_out];
    let mut upper = vec![0.0f64; n_out];
    for o in 0..n_out {
        lower[o] = bias_lo[o] as f64;
        upper[o] = bias_hi[o] as f64;
    }
    for (bi, lb) in bounds.iter().enumerate() {
        let Some(lb) = lb else { continue };
        let bx = boxes[bi].flatten();
        let in_l = bx.lower().as_slice().unwrap().to_vec();
        let in_u = bx.upper().as_slice().unwrap().to_vec();
        let la = lb.lower_a();
        let ua = lb.upper_a();
        for o in 0..n_out {
            for j in 0..lb.num_inputs() {
                let lc = la[[o, j]] as f64;
                let uc = ua[[o, j]] as f64;
                // lower path: min of lc*x over box
                if lc >= 0.0 {
                    lower[o] += lc * in_l[j] as f64;
                } else {
                    lower[o] += lc * in_u[j] as f64;
                }
                if uc >= 0.0 {
                    upper[o] += uc * in_u[j] as f64;
                } else {
                    upper[o] += uc * in_l[j] as f64;
                }
            }
        }
    }
    (
        lower.iter().map(|&v| v as f32).collect(),
        upper.iter().map(|&v| v as f32).collect(),
    )
}

/// Identity upstream node_lb over the flattened Y of size `n_y`.
fn identity_node_lb(n_y: usize) -> LinearBounds {
    LinearBounds::new(
        Array2::eye(n_y),
        Array1::zeros(n_y),
        Array2::eye(n_y),
        Array1::zeros(n_y),
    )
    .unwrap()
}

struct RandConfig {
    n_slices: usize,
    sq: usize,
    sk: usize,
    dk: usize,
    dv: usize,
    mask: AttentionMask,
    window: Option<usize>,
}

/// Run the full soundness check for one random config + box: sample `n_samples`
/// points uniformly from the joint box, assert the concretized bound encloses
/// every sample's true attention output. Returns the average bound width and the
/// IBP-equal-or-tighter check.
fn check_config_sound(cfg: &RandConfig, seed: u64, n_samples: usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let dims = SliceDims {
        n_slices: cfg.n_slices,
        sq: cfg.sq,
        sk: cfg.sk,
        dk: cfg.dk,
        dv: cfg.dv,
    };
    let q_shape: Vec<usize> = if cfg.n_slices > 1 {
        vec![cfg.n_slices, cfg.sq, cfg.dk]
    } else {
        vec![cfg.sq, cfg.dk]
    };
    let k_shape: Vec<usize> = if cfg.n_slices > 1 {
        vec![cfg.n_slices, cfg.sk, cfg.dk]
    } else {
        vec![cfg.sk, cfg.dk]
    };
    let v_shape: Vec<usize> = if cfg.n_slices > 1 {
        vec![cfg.n_slices, cfg.sk, cfg.dv]
    } else {
        vec![cfg.sk, cfg.dv]
    };

    let n_q = cfg.n_slices * cfg.sq * cfg.dk;
    let n_k = cfg.n_slices * cfg.sk * cfg.dk;
    let n_v = cfg.n_slices * cfg.sk * cfg.dv;
    let n_y = cfg.n_slices * cfg.sq * cfg.dv;

    // Random centers + radii. Keep radii modest so softmax stays well-conditioned.
    let gen_cr = |n: usize, rng: &mut StdRng, crange: f32, rmax: f32| -> (Vec<f32>, Vec<f32>) {
        let c: Vec<f32> = (0..n).map(|_| rng.random_range(-crange..crange)).collect();
        let r: Vec<f32> = (0..n).map(|_| rng.random_range(0.0..rmax)).collect();
        (c, r)
    };
    let (qc, qr) = gen_cr(n_q, &mut rng, 1.0, 0.3);
    let (kc, kr) = gen_cr(n_k, &mut rng, 1.0, 0.3);
    let (vc, vr) = gen_cr(n_v, &mut rng, 1.5, 0.5);

    let qbox = box_from_center(&q_shape, &qc, &qr);
    let kbox = box_from_center(&k_shape, &kc, &kr);
    let vbox = box_from_center(&v_shape, &vc, &vr);

    let scale = 1.0 / (cfg.dk as f32).sqrt();
    let layer = match cfg.mask {
        AttentionMask::Standard => SelfAttentionLayer::new(AttentionMask::Standard, Some(scale)),
        AttentionMask::Causal => {
            let l = SelfAttentionLayer::new(AttentionMask::Causal, Some(scale));
            match cfg.window {
                Some(w) => l.with_window_size(w),
                None => l,
            }
        }
    };

    let node_lb = identity_node_lb(n_y);
    let (bounds, bias_lo, bias_hi) = layer
        .propagate_crown_ternary(&node_lb, &qbox, &kbox, &vbox)
        .expect("ternary CROWN should succeed for well-conditioned config");

    let (cl, cu) = concretize_nary(&bounds, &bias_lo, &bias_hi, &qbox, &kbox, &vbox);

    // === SOUNDNESS GATE: enclose every sample ===
    let sample_pt = |centers: &[f32], radii: &[f32], rng: &mut StdRng| -> Vec<f32> {
        centers
            .iter()
            .zip(radii.iter())
            .map(|(&c, &r)| c + rng.random_range(-1.0..1.0) * r)
            .collect()
    };
    let tol = 1e-3f32; // f32 / linearization rounding slack
    for _ in 0..n_samples {
        let qs = sample_pt(&qc, &qr, &mut rng);
        let ks = sample_pt(&kc, &kr, &mut rng);
        let vs = sample_pt(&vc, &vr, &mut rng);
        let y = attention_forward(&qs, &ks, &vs, &dims, scale, cfg.mask, cfg.window);
        for o in 0..n_y {
            assert!(
                y[o] >= cl[o] - tol && y[o] <= cu[o] + tol,
                "SOUNDNESS VIOLATION seed={seed} out={o}: true={} not in [{}, {}] (mask={:?})",
                y[o],
                cl[o],
                cu[o],
                cfg.mask,
            );
        }
    }

    // === Tighter-or-equal vs IBP, AFTER the framework's CROWN∩IBP intersection ===
    // The per-node CROWN bound uses a provably-pointwise-sound (wide) margin and
    // may be looser than IBP at this node; the graph engine ALWAYS intersects the
    // concretized CROWN output with the sound IBP forward bound
    // (`tighten_crown_output_with_provenance`). We mirror that intersection here:
    // the shipped result is tighter-or-equal to IBP by construction and still
    // encloses every sampled true output (both CROWN and IBP enclose the truth,
    // so does their intersection).
    let ibp = layer
        .propagate_ibp_ternary(&qbox, &kbox, &vbox)
        .unwrap()
        .flatten();
    let ibp_lo = ibp.lower();
    let ibp_hi = ibp.upper();
    let ibp_lo = ibp_lo.as_slice().unwrap();
    let ibp_hi = ibp_hi.as_slice().unwrap();
    for o in 0..n_y {
        let int_lo = cl[o].max(ibp_lo[o]);
        let int_hi = cu[o].min(ibp_hi[o]);
        assert!(
            int_lo >= ibp_lo[o] - tol && int_hi <= ibp_hi[o] + tol,
            "intersected CROWN looser than IBP at out={o}: [{int_lo}, {int_hi}] vs IBP [{}, {}]",
            ibp_lo[o],
            ibp_hi[o],
        );
        assert!(
            int_lo <= int_hi + tol,
            "intersection inverted at out={o}: [{int_lo}, {int_hi}]"
        );
    }
}

#[test]
fn soundness_standard_small() {
    for seed in 0..30u64 {
        check_config_sound(
            &RandConfig {
                n_slices: 1,
                sq: 3,
                sk: 3,
                dk: 2,
                dv: 2,
                mask: AttentionMask::Standard,
                window: None,
            },
            seed,
            400,
        );
    }
}

#[test]
fn soundness_causal_small() {
    for seed in 100..130u64 {
        check_config_sound(
            &RandConfig {
                n_slices: 1,
                sq: 4,
                sk: 4,
                dk: 2,
                dv: 3,
                mask: AttentionMask::Causal,
                window: None,
            },
            seed,
            400,
        );
    }
}

#[test]
fn soundness_causal_windowed() {
    for seed in 200..220u64 {
        check_config_sound(
            &RandConfig {
                n_slices: 1,
                sq: 5,
                sk: 5,
                dk: 2,
                dv: 2,
                mask: AttentionMask::Causal,
                window: Some(2),
            },
            seed,
            400,
        );
    }
}

#[test]
fn soundness_multi_slice_standard() {
    for seed in 300..320u64 {
        check_config_sound(
            &RandConfig {
                n_slices: 2,
                sq: 3,
                sk: 3,
                dk: 2,
                dv: 2,
                mask: AttentionMask::Standard,
                window: None,
            },
            seed,
            300,
        );
    }
}

#[test]
fn soundness_larger_dims() {
    for seed in 400..410u64 {
        check_config_sound(
            &RandConfig {
                n_slices: 1,
                sq: 6,
                sk: 6,
                dk: 4,
                dv: 4,
                mask: AttentionMask::Standard,
                window: None,
            },
            seed,
            300,
        );
    }
}

/// Soundness gate with EXPLICIT per-input radii (drives the directional
/// O(radius²) margin's T1/T2/T3 enclosure in the adversarial V-binding /
/// QK-binding regimes the symmetric margin never exercised). Asserts the
/// concretized CROWN bound encloses every sampled true output (0 violations) and
/// is tighter-or-equal to IBP after intersection.
#[allow(clippy::too_many_arguments)]
fn check_radii_sound(
    seed: u64,
    sq: usize,
    sk: usize,
    dk: usize,
    dv: usize,
    mask: AttentionMask,
    window: Option<usize>,
    qrad: f32,
    krad: f32,
    vrad: f32,
    n_samples: usize,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let dims = SliceDims {
        n_slices: 1,
        sq,
        sk,
        dk,
        dv,
    };
    let n_q = sq * dk;
    let n_k = sk * dk;
    let n_v = sk * dv;
    let n_y = sq * dv;
    let gen_cr = |n: usize, rng: &mut StdRng, cr: f32, r: f32| -> (Vec<f32>, Vec<f32>) {
        (
            (0..n).map(|_| rng.random_range(-cr..cr)).collect(),
            (0..n).map(|_| rng.random_range(0.0..r.max(1e-6))).collect(),
        )
    };
    let (qc, qr) = gen_cr(n_q, &mut rng, 1.0, qrad);
    let (kc, kr) = gen_cr(n_k, &mut rng, 1.0, krad);
    let (vc, vr) = gen_cr(n_v, &mut rng, 1.5, vrad);
    let qbox = box_from_center(&[sq, dk], &qc, &qr);
    let kbox = box_from_center(&[sk, dk], &kc, &kr);
    let vbox = box_from_center(&[sk, dv], &vc, &vr);
    let scale = 1.0 / (dk as f32).sqrt();
    let layer = match mask {
        AttentionMask::Standard => SelfAttentionLayer::new(AttentionMask::Standard, Some(scale)),
        AttentionMask::Causal => {
            let l = SelfAttentionLayer::new(AttentionMask::Causal, Some(scale));
            match window {
                Some(w) => l.with_window_size(w),
                None => l,
            }
        }
    };
    let node_lb = identity_node_lb(n_y);
    let (bounds, blo, bhi) = layer
        .propagate_crown_ternary(&node_lb, &qbox, &kbox, &vbox)
        .expect("ternary CROWN should succeed");
    let (cl, cu) = concretize_nary(&bounds, &blo, &bhi, &qbox, &kbox, &vbox);
    let tol = 1e-3f32;
    for _ in 0..n_samples {
        let s = |c: &[f32], r: &[f32], rng: &mut StdRng| -> Vec<f32> {
            c.iter()
                .zip(r)
                .map(|(&c, &r)| c + rng.random_range(-1.0..1.0) * r)
                .collect()
        };
        let qs = s(&qc, &qr, &mut rng);
        let ks = s(&kc, &kr, &mut rng);
        let vs = s(&vc, &vr, &mut rng);
        let y = attention_forward(&qs, &ks, &vs, &dims, scale, mask, window);
        for o in 0..n_y {
            assert!(
                y[o] >= cl[o] - tol && y[o] <= cu[o] + tol,
                "DIRECTIONAL-MARGIN SOUNDNESS VIOLATION seed={seed} out={o}: \
                 true={} not in [{}, {}] (q={qrad} k={krad} v={vrad} mask={mask:?})",
                y[o],
                cl[o],
                cu[o],
            );
        }
    }
    // Tighter-or-equal to IBP after intersection.
    let ibp = layer
        .propagate_ibp_ternary(&qbox, &kbox, &vbox)
        .unwrap()
        .flatten();
    let il = ibp.lower();
    let ih = ibp.upper();
    let il = il.as_slice().unwrap();
    let ih = ih.as_slice().unwrap();
    for o in 0..n_y {
        let int_lo = cl[o].max(il[o]);
        let int_hi = cu[o].min(ih[o]);
        assert!(
            int_lo >= il[o] - tol && int_hi <= ih[o] + tol && int_lo <= int_hi + tol,
            "not tighter-or-equal seed={seed} o={o}: [{int_lo}, {int_hi}] vs IBP [{}, {}]",
            il[o],
            ih[o],
        );
    }
}

/// Adversarial directional-margin soundness: V perturbation ≫ Q/K (V-binding),
/// the reverse (QK-binding), and balanced wide boxes (margin clamp engages),
/// across standard + causal masks. This is the regime where an under-bounded
/// T1/T2/T3 enclosure would surface as an enclosure violation.
#[test]
fn soundness_directional_margin_adversarial() {
    let regimes = [
        // (qrad, krad, vrad)
        (0.01f32, 0.01f32, 0.8f32), // V-binding (large V, tiny QK)
        (0.5, 0.5, 0.01),           // QK-binding (large QK, tiny V)
        (0.4, 0.4, 0.6),            // balanced wide (clamp engages)
        (0.2, 0.05, 0.3),           // asymmetric Q≫K
        (0.05, 0.2, 0.3),           // asymmetric K≫Q
    ];
    for seed in 0..12u64 {
        for &(qr, kr, vr) in &regimes {
            check_radii_sound(
                seed,
                4,
                4,
                3,
                3,
                AttentionMask::Standard,
                None,
                qr,
                kr,
                vr,
                600,
            );
            check_radii_sound(
                seed + 50,
                4,
                4,
                3,
                3,
                AttentionMask::Causal,
                None,
                qr,
                kr,
                vr,
                600,
            );
            check_radii_sound(
                seed + 100,
                5,
                5,
                2,
                2,
                AttentionMask::Causal,
                Some(2),
                qr,
                kr,
                vr,
                500,
            );
        }
    }
}

/// Build a BoundedTensor box from explicit per-coordinate lower/upper.
fn box_from_lo_hi(shape: &[usize], lo: &[f32], hi: &[f32]) -> BoundedTensor {
    let l = ArrayD::from_shape_vec(IxDyn(shape), lo.to_vec()).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(shape), hi.to_vec()).unwrap();
    BoundedTensor::new(l, u).unwrap()
}

/// Run the soundness gate for one EXPLICIT joint box (lo/hi per input). Samples
/// `n_samples` interior points (plus all `2^?`-ish corners are exercised by the
/// proptest separately) and asserts the concretized CROWN bound encloses every
/// sampled true attention output. Also checks the explicit `extra_pts` (e.g. a
/// known worst-case corner). Returns nothing; panics on any enclosure violation.
#[allow(clippy::too_many_arguments)]
fn assert_box_sound(
    layer: &SelfAttentionLayer,
    dims: &SliceDims,
    scale: f32,
    mask: AttentionMask,
    window: Option<usize>,
    qlo: &[f32],
    qhi: &[f32],
    klo: &[f32],
    khi: &[f32],
    vlo: &[f32],
    vhi: &[f32],
    extra_pts: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    seed: u64,
    n_samples: usize,
) {
    let q_shape = vec![dims.sq, dims.dk];
    let k_shape = vec![dims.sk, dims.dk];
    let v_shape = vec![dims.sk, dims.dv];
    let qbox = box_from_lo_hi(&q_shape, qlo, qhi);
    let kbox = box_from_lo_hi(&k_shape, klo, khi);
    let vbox = box_from_lo_hi(&v_shape, vlo, vhi);
    let n_y = dims.sq * dims.dv;
    let node_lb = identity_node_lb(n_y);
    let (bounds, blo, bhi) = layer
        .propagate_crown_ternary(&node_lb, &qbox, &kbox, &vbox)
        .expect("ternary CROWN should succeed");
    let (cl, cu) = concretize_nary(&bounds, &blo, &bhi, &qbox, &kbox, &vbox);

    // IBP for the tighter-or-equal check (after the framework's CROWN∩IBP).
    let ibp = layer
        .propagate_ibp_ternary(&qbox, &kbox, &vbox)
        .unwrap()
        .flatten();
    let il = ibp.lower();
    let ih = ibp.upper();
    let il = il.as_slice().unwrap();
    let ih = ih.as_slice().unwrap();

    let tol = 1e-3f32;
    // Tolerance for the underflow-regime witness is dominated by f32 magnitude
    // (outputs up to ~50), so use a small RELATIVE slack too.
    let enclosed = |y: f32, lo: f32, hi: f32| -> bool {
        let slack = tol + 1e-4 * y.abs();
        y >= lo - slack && y <= hi + slack
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let sample = |lo: &[f32], hi: &[f32], rng: &mut StdRng| -> Vec<f32> {
        lo.iter()
            .zip(hi)
            .map(|(&l, &h)| l + rng.random_range(0.0..1.0) * (h - l))
            .collect()
    };
    let check_pt = |qs: &[f32], ks: &[f32], vs: &[f32], cl: &[f32], cu: &[f32]| {
        let y = attention_forward(qs, ks, vs, dims, scale, mask, window);
        for o in 0..n_y {
            assert!(
                enclosed(y[o], cl[o], cu[o]),
                "UNDERFLOW-REGIME SOUNDNESS VIOLATION o={o}: true={} not in [{}, {}] \
                 (mask={mask:?})",
                y[o],
                cl[o],
                cu[o],
            );
        }
    };

    for _ in 0..n_samples {
        let qs = sample(qlo, qhi, &mut rng);
        let ks = sample(klo, khi, &mut rng);
        let vs = sample(vlo, vhi, &mut rng);
        check_pt(&qs, &ks, &vs, &cl, &cu);
    }
    for (qs, ks, vs) in extra_pts {
        check_pt(qs, ks, vs, &cl, &cu);
    }

    // Tighter-or-equal to IBP after intersection (the shipped contract).
    for o in 0..n_y {
        let int_lo = cl[o].max(il[o]);
        let int_hi = cu[o].min(ih[o]);
        assert!(
            int_lo >= il[o] - tol && int_hi <= ih[o] + tol && int_lo <= int_hi + tol,
            "not tighter-or-equal o={o}: [{int_lo}, {int_hi}] vs IBP [{}, {}]",
            il[o],
            ih[o],
        );
    }
}

/// REGRESSION for the CONFIRMED false-proof hole: a VISIBLE key whose CENTER
/// softmax prob underflows to exactly 0.0 in f64 (`score − max_score < ≈ −745`)
/// was dropped from the error margin (the old `pc[m]==0.0` guard conflated it
/// with a genuinely masked key), making the certified bound too narrow and
/// EXCLUDING the reachable true output — a FALSE certificate.
///
/// Witness (from the adversarial audit): seq_k=3, dk=1, scale=1,
/// Kc = [0, 0, −800], Qc = [1], Vc = [1, 2, 50]; box Q∈[−1,3], K[2]∈[−1300,−300]
/// (others at center). Center pc=[.5,.5,0] (key 2 underflowed), surrogate Yc=1.5.
/// At the LEGITIMATE corner Q=−1, K[2]=−1300: scores=[−1,−1,1300] → P=[0,0,1] →
/// Y_true=50.0. The buggy bound (centered at 1.5 with a tiny O(radius²) margin)
/// certified upper ≈ 38.5 and EXCLUDED 50.0. After the fix the underflowed
/// visible key is in the margin (T1), the margin clamps to the simplex-aware IBP
/// gap (IBP upper here is 50.0), and the certified bound ENCLOSES Y_true=50.0.
#[test]
fn false_proof_underflowed_visible_key_witness() {
    let scale = 1.0f32;
    let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(scale));
    let dims = SliceDims {
        n_slices: 1,
        sq: 1,
        sk: 3,
        dk: 1,
        dv: 1,
    };
    // Q ∈ [−1, 3]  (center 1, radius 2)
    let qlo = [-1.0f32];
    let qhi = [3.0f32];
    // K rows: K[0]=0, K[1]=0 fixed; K[2] ∈ [−1300, −300] (center −800).
    let klo = [0.0f32, 0.0, -1300.0];
    let khi = [0.0f32, 0.0, -300.0];
    // V fixed at [1, 2, 50].
    let vlo = [1.0f32, 2.0, 50.0];
    let vhi = [1.0f32, 2.0, 50.0];

    // The known worst-case legitimate corner: Q=−1, K[2]=−1300 → Y_true=50.0.
    let worst = (
        vec![-1.0f32],
        vec![0.0f32, 0.0, -1300.0],
        vec![1.0f32, 2.0, 50.0],
    );

    // Sanity: the true output at the corner really is 50.0 (key 2 wins).
    let y_corner = attention_forward(
        &worst.0,
        &worst.1,
        &worst.2,
        &dims,
        scale,
        AttentionMask::Standard,
        None,
    );
    assert!(
        (y_corner[0] - 50.0).abs() < 1e-3,
        "witness corner true output should be 50.0, got {}",
        y_corner[0]
    );

    // The certified bound MUST enclose Y_true=50.0 (and every interior sample).
    assert_box_sound(
        &layer,
        &dims,
        scale,
        AttentionMask::Standard,
        None,
        &qlo,
        &qhi,
        &klo,
        &khi,
        &vlo,
        &vhi,
        std::slice::from_ref(&worst),
        12345,
        4000,
    );

    // Explicit assertion on the SHIPPED (CROWN∩IBP) certified interval: its upper
    // must be ≥ 50.0 so the corner is enclosed (the buggy upper was ≈38.5).
    let qbox = box_from_lo_hi(&[1, 1], &qlo, &qhi);
    let kbox = box_from_lo_hi(&[3, 1], &klo, &khi);
    let vbox = box_from_lo_hi(&[3, 1], &vlo, &vhi);
    let node_lb = identity_node_lb(1);
    let (bounds, blo, bhi) = layer
        .propagate_crown_ternary(&node_lb, &qbox, &kbox, &vbox)
        .unwrap();
    let (cl, cu) = concretize_nary(&bounds, &blo, &bhi, &qbox, &kbox, &vbox);
    let ibp = layer
        .propagate_ibp_ternary(&qbox, &kbox, &vbox)
        .unwrap()
        .flatten();
    let ih = ibp.upper();
    let ih = ih.as_slice().unwrap();
    let il = ibp.lower();
    let il = il.as_slice().unwrap();
    let shipped_hi = cu[0].min(ih[0]);
    let shipped_lo = cl[0].max(il[0]);
    assert!(
        shipped_hi >= 50.0 - 1e-2,
        "FALSE PROOF: certified upper {shipped_hi} excludes reachable Y_true=50.0 \
         (CROWN raw upper={}, IBP upper={})",
        cu[0],
        ih[0]
    );
    assert!(
        shipped_lo <= 50.0,
        "certified lower {shipped_lo} above reachable 50.0?!"
    );
}

/// Causal variant of the underflow witness: the underflowed key is the DIAGONAL
/// (self) key, which is always visible under the causal mask, so it must be in
/// the margin exactly as in the standard case.
#[test]
fn false_proof_underflowed_visible_key_causal() {
    let scale = 1.0f32;
    let layer = SelfAttentionLayer::new(AttentionMask::Causal, Some(scale));
    // sq=sk=2, dk=1, dv=1. Row i=1 sees keys {0,1}. Make key 1 (diagonal, visible)
    // underflow at center but rise to win over the box.
    let dims = SliceDims {
        n_slices: 1,
        sq: 2,
        sk: 2,
        dk: 1,
        dv: 1,
    };
    // Q rows: Q[0]=0 (unused-ish), Q[1] ∈ [−1, 3] (center 1).
    let qlo = [0.0f32, -1.0];
    let qhi = [0.0f32, 3.0];
    // K rows: K[0]=0 fixed; K[1] ∈ [−1300, −300] (center −800) → row1 key1 center
    // score = 1·−800 = −800 (underflows vs key0 score 0).
    let klo = [0.0f32, -1300.0];
    let khi = [0.0f32, -300.0];
    // V rows: V[0]=1, V[1]=50.
    let vlo = [1.0f32, 50.0];
    let vhi = [1.0f32, 50.0];
    // Corner: Q[1]=−1, K[1]=−1300 → row1 score key1 = (−1)(−1300)=1300 ≫ key0=0 →
    // P=[0,1] → Y[1]=50.
    let worst = (
        vec![0.0f32, -1.0],
        vec![0.0f32, -1300.0],
        vec![1.0f32, 50.0],
    );
    let y_corner = attention_forward(
        &worst.0,
        &worst.1,
        &worst.2,
        &dims,
        scale,
        AttentionMask::Causal,
        None,
    );
    assert!(
        (y_corner[1] - 50.0).abs() < 1e-3,
        "causal witness row1 true output should be 50.0, got {}",
        y_corner[1]
    );
    assert_box_sound(
        &layer,
        &dims,
        scale,
        AttentionMask::Causal,
        None,
        &qlo,
        &qhi,
        &klo,
        &khi,
        &vlo,
        &vhi,
        std::slice::from_ref(&worst),
        222,
        4000,
    );
}

/// EXTENDED soundness proptest deliberately driving the >745 score-gap /
/// underflow regime the bounded-radius proptest never reached. For each config we
/// place ONE (or a few) key center scores ≫ 745 BELOW the row max (so their
/// center softmax prob underflows to 0.0 in f64) while giving Q and K WIDE radii
/// so that key's box score can RISE past the row max and WIN the softmax (P→1).
/// This is exactly the class that produced the false certificate. We then sample
/// the box (interior + the score-maximizing corner per underflowed key) and
/// assert 0 enclosure violations, for standard AND causal masks.
fn check_underflow_regime(seed: u64, mask: AttentionMask, window: Option<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let sq = 3usize;
    let sk = 3usize;
    let dk = 1usize; // dk=1 makes the score = scale·Q·K easy to drive into underflow
    let dv = 2usize;
    let scale = 1.0f32;
    let dims = SliceDims {
        n_slices: 1,
        sq,
        sk,
        dk,
        dv,
    };
    let layer = match mask {
        AttentionMask::Standard => SelfAttentionLayer::new(AttentionMask::Standard, Some(scale)),
        AttentionMask::Causal => {
            let l = SelfAttentionLayer::new(AttentionMask::Causal, Some(scale));
            match window {
                Some(w) => l.with_window_size(w),
                None => l,
            }
        }
    };

    // Build a box where K[under] center is hugely negative (underflow) but its
    // upper makes the score rise past the others, and Q can flip sign so the
    // product Q·K[under] becomes large positive. Centers chosen so the row max is
    // some normal key and the underflowed key's CENTER score is < −745.
    let under = sk - 1; // last key underflows at center
                        // Q rows: center near +1, but box reaches −1.5 (sign flip).
    let mut qlo = vec![0.0f32; sq * dk];
    let mut qhi = vec![0.0f32; sq * dk];
    for i in 0..sq {
        let c = rng.random_range(0.5..1.5);
        let r = rng.random_range(1.5..3.0);
        qlo[i] = c - r;
        qhi[i] = c + r;
    }
    // K rows: normal keys near small values; the underflow key hugely negative
    // center with a wide radius so its score can become large-positive when Q<0.
    let mut klo = vec![0.0f32; sk * dk];
    let mut khi = vec![0.0f32; sk * dk];
    for k in 0..sk {
        if k == under {
            let c = rng.random_range(-1000.0..-800.0); // center score |Q·K|≈ ≫745
            let r = rng.random_range(400.0..700.0);
            klo[k] = c - r;
            khi[k] = c + r;
        } else {
            let c = rng.random_range(-0.5..0.5);
            let r = rng.random_range(0.0..0.3);
            klo[k] = c - r;
            khi[k] = c + r;
        }
    }
    // V rows: spread so the underflowed key's V row is FAR (big margin if it wins).
    let mut vlo = vec![0.0f32; sk * dv];
    let mut vhi = vec![0.0f32; sk * dv];
    for k in 0..sk {
        for j in 0..dv {
            let c = if k == under {
                rng.random_range(20.0..60.0) // big-magnitude V for the underflow key
            } else {
                rng.random_range(-2.0..2.0)
            };
            let r = rng.random_range(0.0..0.5);
            vlo[k * dv + j] = c - r;
            vhi[k * dv + j] = c + r;
        }
    }

    // Confirm the regime is actually triggered: at the CENTER, key `under`'s
    // softmax prob underflows to 0 for some visible row; over the box its score
    // can exceed the row max. (Otherwise the test would be vacuous.)
    let qc: Vec<f32> = qlo
        .iter()
        .zip(&qhi)
        .map(|(&l, &h)| f32::midpoint(l, h))
        .collect();
    let kc: Vec<f32> = klo
        .iter()
        .zip(&khi)
        .map(|(&l, &h)| f32::midpoint(l, h))
        .collect();

    // Score-maximizing corner for the underflow key in each row: pick Q sign to
    // maximize Q·K[under]; K[under] at whichever end maximizes the product.
    let mut extra_pts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
    let vc: Vec<f32> = vlo
        .iter()
        .zip(&vhi)
        .map(|(&l, &h)| f32::midpoint(l, h))
        .collect();
    {
        // K corner that makes K[under] most negative AND most positive.
        for &kend in &[klo[under * dk], khi[under * dk]] {
            let mut kpt = kc.clone();
            kpt[under * dk] = kend;
            // Q corner that maximizes Q·K[under] for each row independently.
            let mut qpt = qc.clone();
            for i in 0..sq {
                qpt[i * dk] = if kend >= 0.0 {
                    qhi[i * dk]
                } else {
                    qlo[i * dk]
                };
            }
            extra_pts.push((qpt, kpt, vc.clone()));
        }
    }

    assert_box_sound(
        &layer,
        &dims,
        scale,
        mask,
        window,
        &qlo,
        &qhi,
        &klo,
        &khi,
        &vlo,
        &vhi,
        &extra_pts,
        seed.wrapping_mul(2_654_435_761),
        3000,
    );
}

/// The extended underflow soundness gate: standard + causal + windowed, many
/// seeds. 0 enclosure violations REQUIRED (this is the false-proof class).
#[test]
fn soundness_underflow_large_score_gap() {
    for seed in 0..40u64 {
        check_underflow_regime(seed, AttentionMask::Standard, None);
        check_underflow_regime(seed + 1000, AttentionMask::Causal, None);
        check_underflow_regime(seed + 2000, AttentionMask::Causal, Some(2));
    }
}

// =========================================================================
// MASK-SOUNDNESS GATE: windowed-causal cross-attention (sq>sk) false-proof
// =========================================================================

/// FAITHFUL causal forward for a single output `Y[i,j]`, using the SOUND forward
/// `active_range` (NOT the crown `visible` predicate):
///   active_end   = min(i+1, sk)
///   active_start = window ? max(0, active_end − (w+1)) : 0
/// Softmax over `[active_start, active_end)` of `scale·⟨Q[i],K[k]⟩`, then `· V[k,j]`.
/// This is the reference the certified bound MUST enclose. The shared
/// `attention_forward` helper instead mirrors the crown `visible` set (it tests the
/// surrogate's own mask); for windowed-causal sq>sk the two DIFFER on rows i>=sk —
/// which is exactly the false-proof hole. Single-slice; `dk`/`dv` indexing matches
/// the flattened `[sq,dk]`/`[sk,dk]`/`[sk,dv]` test boxes.
#[allow(clippy::too_many_arguments)]
fn faithful_causal_forward_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    i: usize,
    sk: usize,
    dk: usize,
    scale: f32,
    window: Option<usize>,
) -> f32 {
    faithful_causal_forward_elem(q, k, v, i, 0, sk, dk, 1, scale, window)
}

#[allow(clippy::too_many_arguments)]
fn faithful_causal_forward_elem(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    i: usize,
    j: usize,
    sk: usize,
    dk: usize,
    dv: usize,
    scale: f32,
    window: Option<usize>,
) -> f32 {
    let active_end = (i + 1).min(sk);
    let active_start = match window {
        Some(w) => active_end.saturating_sub(w + 1),
        None => 0,
    };
    let mut maxs = f64::NEG_INFINITY;
    let mut scores = vec![0.0f64; sk];
    for kk in active_start..active_end {
        let mut dot = 0.0f64;
        for d in 0..dk {
            dot += q[i * dk + d] as f64 * k[kk * dk + d] as f64;
        }
        let s = scale as f64 * dot;
        scores[kk] = s;
        if s > maxs {
            maxs = s;
        }
    }
    let mut sum = 0.0f64;
    for kk in active_start..active_end {
        sum += (scores[kk] - maxs).exp();
    }
    let mut y = 0.0f64;
    for kk in active_start..active_end {
        let p = (scores[kk] - maxs).exp() / sum;
        y += p * v[kk * dv + j] as f64;
    }
    y as f32
}

/// Faithful causal forward over the WHOLE flattened Y (all rows, all dv), using
/// `active_range`. Standard mask falls back to the shared `attention_forward`
/// (which is already faithful for the standard, unmasked case).
fn faithful_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dims: &SliceDims,
    scale: f32,
    mask: AttentionMask,
    window: Option<usize>,
) -> Vec<f32> {
    if mask == AttentionMask::Standard {
        return attention_forward(q, k, v, dims, scale, mask, window);
    }
    let SliceDims { sq, sk, dk, dv, .. } = *dims;
    let mut out = vec![0.0f32; sq * dv];
    for i in 0..sq {
        for j in 0..dv {
            out[i * dv + j] =
                faithful_causal_forward_elem(q, k, v, i, j, sk, dk, dv, scale, window);
        }
    }
    out
}

/// CONFIRMED HIGH-SEVERITY false-proof witness (auditor-supplied). For
/// WINDOWED-causal CROSS-attention with sq > sk, the crown `visible(i,k)` set
/// (`k<=i && i−k<=w`) is STRICTER than the sound forward `active_range`
/// (`active_end=min(i+1,sk)`, `active_start=active_end−(w+1)`) on rows `i >= sk`:
/// the forward attends keys `[active_start, sk)` but crown keeps only `[i−w, i]`,
/// and since `i>=sk` these differ, so crown DROPS genuinely-visible keys from the
/// surrogate AND the T1/T2/T3 error margin, certifying a too-narrow interval that
/// EXCLUDES the reachable true output.
///
/// Witness: sk=3, w=1, sq=4, dk=dv=1, K centers all 0, V=[1,100,1]. Row i=3:
///   forward active_range = [max(0,min(4,3)−2), min(4,3)) = [1,3) = {1,2}
///   crown visible        = {k : k<=3 && 3−k<=1} = {2}
/// With K=0 the row-3 scores over the forward-visible keys {1,2} are both 0, so
/// softmax=[.5,.5] and Y[3] = .5·V[1] + .5·V[2] = .5·100 + .5·1 = 50.5. The
/// ungated crown would see only key 2 and certify [1.0, 1.0] — a FALSE proof that
/// EXCLUDES 50.5.
///
/// With the mask-soundness gate, `propagate_crown_ternary` for this config now
/// RETURNS UnsupportedOp (the dispatch falls back to the sound IBP). This test
/// asserts BOTH: (1) the crown path is gated to Unsupported, and (2) the sound
/// IBP fallback ENCLOSES the true Y[3]=50.5 — never the false [1.0,1.0].
#[test]
fn mask_gate_windowed_causal_cross_attention_witness() {
    let scale = 1.0f32;
    let layer = SelfAttentionLayer::new(AttentionMask::Causal, Some(scale)).with_window_size(1);
    // sq=4 > sk=3 (cross-attention), dk=dv=1.
    let sq = 4usize;
    let sk = 3usize;
    // K centers all 0 (concrete). Q centers all 1 (concrete) → all scores 0.
    let q_vals = vec![1.0f32; sq];
    let k_vals = vec![0.0f32; sk];
    // V = [1, 100, 1].
    let v_vals = vec![1.0f32, 100.0, 1.0];
    let qbox = box_from_lo_hi(&[sq, 1], &q_vals, &q_vals);
    let kbox = box_from_lo_hi(&[sk, 1], &k_vals, &k_vals);
    let vbox = box_from_lo_hi(&[sk, 1], &v_vals, &v_vals);

    // Sanity: the FAITHFUL forward (using the SOUND `active_range`, NOT the crown
    // `visible` predicate) at this point yields Y[3]=50.5. NOTE: the shared
    // `attention_forward` test helper deliberately replicates the crown `visible`
    // mask (so the soundness sweeps test the surrogate's own mask), which for this
    // very witness would WRONGLY give Y[3]=1.0 — that is precisely the false
    // certificate. We therefore compute the faithful forward inline here.
    let y3 = faithful_causal_forward_row(&q_vals, &k_vals, &v_vals, 3, sk, 1, scale, Some(1));
    assert!(
        (y3 - 50.5).abs() < 1e-3,
        "faithful forward row 3 (active_range keys {{1,2}}) must be 50.5, got {y3}"
    );

    // (1) GATE: the crown ternary path must return UnsupportedOp for this
    //     windowed-causal sq>sk config. CRITICALLY it must NOT return Ok(bound):
    //     an emitted bound here would (using the too-strict crown `visible` set
    //     {2}) certify the false [1.0, 1.0] that EXCLUDES the reachable 50.5. The
    //     gate fires in `parse_slice_dims`, BEFORE any surrogate is built, so no
    //     false bound can escape.
    let n_y = sq;
    let node_lb = identity_node_lb(n_y);
    let res = layer.propagate_crown_ternary(&node_lb, &qbox, &kbox, &vbox);
    let err = res.expect_err("windowed-causal sq>sk must be gated, NOT emit a crown bound");
    assert!(
        matches!(err, NyError::UnsupportedOp(_)),
        "expected the mask-soundness UnsupportedOp gate, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("sq>sk") || msg.contains("active_range") || msg.contains("windowed-causal"),
        "gate message should explain the windowed-causal sq>sk mismatch, got: {msg}"
    );

    // (2) NO FALSE CERTIFICATE downstream. The dispatch converts this UnsupportedOp
    //     to a graceful IBP fallback. The SOUND forward (CausalSoftmax) itself only
    //     supports seq_q ≤ seq_k, so for this sq>sk config the IBP fallback REFUSES
    //     to certify (InvalidSpec) rather than emit anything — the framework cannot
    //     produce ANY interval here, least of all the false [1.0, 1.0]. Either way
    //     the reachable truth 50.5 is never excluded by a shipped certificate.
    let ibp = layer.propagate_ibp_ternary(&qbox, &kbox, &vbox);
    match ibp {
        Err(NyError::InvalidSpec(m)) => assert!(
            m.contains("seq_q") && m.contains("seq_k"),
            "expected the forward seq_q<=seq_k refusal, got: {m}"
        ),
        Ok(t) => {
            // If a future forward DOES support sq>sk causal, the IBP it returns must
            // ENCLOSE the true 50.5 — never the false [1.0, 1.0]. (Defensive.)
            let f = t.flatten();
            let il = f.lower();
            let ih = f.upper();
            let il = il.as_slice().unwrap();
            let ih = ih.as_slice().unwrap();
            assert!(
                il[3] <= 50.5 + 1e-2 && ih[3] >= 50.5 - 1e-2,
                "any IBP that DOES certify sq>sk causal must enclose 50.5, got [{}, {}]",
                il[3],
                ih[3]
            );
        }
        Err(other) => panic!("unexpected IBP error for witness: {other:?}"),
    }
}

/// Companion to the witness: the GATE must fire for windowed-causal whenever
/// sq>sk, and must NOT fire for the provably-matched classes (standard, any
/// sq==sk, windowless-causal at any sq/sk, and windowed-causal with sq<=sk).
/// This pins the EXACT sound class boundary so a future widening of `visible()`
/// support cannot silently re-open the hole.
#[test]
fn mask_gate_fires_exactly_on_windowed_causal_sq_gt_sk() {
    // Build a concrete (zero-radius) box for the given shape and run crown; return
    // whether it was gated (Err UnsupportedOp) or emitted a bound (Ok).
    let try_crown = |mask: AttentionMask, window: Option<usize>, sq: usize, sk: usize| -> bool {
        let scale = 1.0f32;
        let mut layer = SelfAttentionLayer::new(mask, Some(scale));
        if let Some(w) = window {
            layer = layer.with_window_size(w);
        }
        let dk = 1usize;
        let dv = 1usize;
        // Mild well-separated values so the un-gated path would succeed numerically.
        let qv: Vec<f32> = (0..sq * dk).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let kv: Vec<f32> = (0..sk * dk).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let vv: Vec<f32> = (0..sk * dv).map(|i| (i as f32) - 1.0).collect();
        let qbox = box_from_lo_hi(&[sq, dk], &qv, &qv);
        let kbox = box_from_lo_hi(&[sk, dk], &kv, &kv);
        let vbox = box_from_lo_hi(&[sk, dv], &vv, &vv);
        let node_lb = identity_node_lb(sq * dv);
        matches!(
            layer.propagate_crown_ternary(&node_lb, &qbox, &kbox, &vbox),
            Err(NyError::UnsupportedOp(_))
        )
    };

    // GATE FIRES: windowed-causal with sq>sk (the only unsound class).
    assert!(
        try_crown(AttentionMask::Causal, Some(1), 4, 3),
        "w=1 sq=4>sk=3 must gate"
    );
    assert!(
        try_crown(AttentionMask::Causal, Some(2), 5, 3),
        "w=2 sq=5>sk=3 must gate"
    );
    assert!(
        try_crown(AttentionMask::Causal, Some(0), 3, 2),
        "w=0 sq=3>sk=2 must gate"
    );

    // GATE DOES NOT FIRE (provably-matched, win preserved):
    //  - standard, any sq/sk:
    assert!(
        !try_crown(AttentionMask::Standard, None, 4, 3),
        "standard sq!=sk must NOT gate"
    );
    assert!(
        !try_crown(AttentionMask::Standard, None, 4, 4),
        "standard sq==sk must NOT gate"
    );
    //  - windowed-causal with sq==sk (self-attention — the end-to-end win):
    assert!(
        !try_crown(AttentionMask::Causal, Some(1), 4, 4),
        "windowed sq==sk must NOT gate"
    );
    assert!(
        !try_crown(AttentionMask::Causal, Some(2), 5, 5),
        "windowed sq==sk must NOT gate"
    );
    //  - windowed-causal with sq<sk (rows all i<sk ⇒ active_end=i+1 ⇒ MATCH):
    assert!(
        !try_crown(AttentionMask::Causal, Some(1), 3, 5),
        "windowed sq<sk must NOT gate"
    );
    //  - windowless-causal, ANY sq/sk (active_start=0=crown start always):
    assert!(
        !try_crown(AttentionMask::Causal, None, 4, 3),
        "windowless-causal sq>sk must NOT gate"
    );
    assert!(
        !try_crown(AttentionMask::Causal, None, 5, 5),
        "windowless-causal sq==sk must NOT gate"
    );
    assert!(
        !try_crown(AttentionMask::Causal, None, 3, 6),
        "windowless-causal sq<sk must NOT gate"
    );
}

/// Cross-mask soundness sweep across ALL mask classes and BOTH sq==sk and sq!=sk.
/// Wherever the crown path EMITS a bound, the concretized bound (intersected with
/// IBP, as the framework ships it) MUST enclose the true forward at every sampled
/// point (0 violations); wherever the path is GATED, we assert the sound IBP
/// fallback encloses the truth instead. This is the regression that the gate is
/// the SMALLEST sound boundary: matched classes still produce (sound) bounds, the
/// one unsound class falls back.
#[test]
fn soundness_all_mask_classes_emit_or_fallback() {
    // (mask, window, sq, sk, forward_supports). The SOUND forward (CausalSoftmax)
    // only supports seq_q ≤ seq_k, so causal configs with sq>sk are forward-INVALID
    // (the IBP refuses with InvalidSpec — no certificate is producible, sound). We
    // still drive crown on them to assert the GATE fires (no false bound), but skip
    // the IBP-enclosure check (there is no IBP). Standard attention has no such
    // constraint, so standard cross-attention (sq≠sk) is forward-valid and emits.
    let configs: &[(AttentionMask, Option<usize>, usize, usize, bool)] = &[
        (AttentionMask::Standard, None, 4, 4, true), // standard self
        (AttentionMask::Standard, None, 4, 3, true), // standard cross sq>sk (valid)
        (AttentionMask::Standard, None, 3, 5, true), // standard cross sq<sk (valid)
        (AttentionMask::Causal, None, 4, 4, true),   // windowless self
        (AttentionMask::Causal, None, 3, 6, true),   // windowless cross sq<sk (valid)
        (AttentionMask::Causal, Some(2), 5, 5, true), // windowed self (THE win class)
        (AttentionMask::Causal, Some(1), 4, 4, true), // windowed self
        (AttentionMask::Causal, Some(2), 3, 5, true), // windowed cross sq<sk (matched, valid)
        (AttentionMask::Causal, None, 5, 3, false),  // windowless cross sq>sk (fwd-invalid)
        (AttentionMask::Causal, Some(1), 5, 3, false), // windowed cross sq>sk (GATED + fwd-invalid)
        (AttentionMask::Causal, Some(2), 6, 4, false), // windowed cross sq>sk (GATED + fwd-invalid)
    ];
    let dk = 2usize;
    let dv = 2usize;
    let scale = 1.0f32 / (dk as f32).sqrt();
    let mut gated = 0usize;
    let mut emitted = 0usize;
    let mut fwd_refused = 0usize;
    for (ci, &(mask, window, sq, sk, fwd_ok)) in configs.iter().enumerate() {
        let mut layer = SelfAttentionLayer::new(mask, Some(scale));
        if let Some(w) = window {
            layer = layer.with_window_size(w);
        }
        let dims = SliceDims {
            n_slices: 1,
            sq,
            sk,
            dk,
            dv,
        };
        let mut rng = StdRng::seed_from_u64(7000 + ci as u64);
        let n_q = sq * dk;
        let n_k = sk * dk;
        let n_v = sk * dv;
        let n_y = sq * dv;
        let gen_cr = |n: usize, rng: &mut StdRng, c: f32, r: f32| -> (Vec<f32>, Vec<f32>) {
            (
                (0..n).map(|_| rng.random_range(-c..c)).collect(),
                (0..n).map(|_| rng.random_range(0.0..r)).collect(),
            )
        };
        let (qc, qr) = gen_cr(n_q, &mut rng, 1.0, 0.3);
        let (kc, kr) = gen_cr(n_k, &mut rng, 1.0, 0.3);
        let (vc, vr) = gen_cr(n_v, &mut rng, 1.5, 0.5);
        let qbox = box_from_center(&[sq, dk], &qc, &qr);
        let kbox = box_from_center(&[sk, dk], &kc, &kr);
        let vbox = box_from_center(&[sk, dv], &vc, &vr);
        let node_lb = identity_node_lb(n_y);

        let tol = 1e-3f32;
        let crown = layer.propagate_crown_ternary(&node_lb, &qbox, &kbox, &vbox);
        let ibp_res = layer.propagate_ibp_ternary(&qbox, &kbox, &vbox);

        // Forward-validity sanity: causal sq>sk must be refused by the IBP.
        if !fwd_ok {
            assert!(
                matches!(ibp_res, Err(NyError::InvalidSpec(_))),
                "cfg={ci} ({mask:?} w={window:?} sq={sq} sk={sk}) should be forward-refused"
            );
            fwd_refused += 1;
        }

        match crown {
            Ok((bounds, blo, bhi)) => {
                emitted += 1;
                // A crown bound was emitted ⇒ the config MUST be forward-valid
                // (the gate forbids the only unsound class) and IBP must enclose.
                assert!(
                    fwd_ok,
                    "cfg={ci} emitted a crown bound but is forward-invalid"
                );
                let ibp = ibp_res.as_ref().unwrap().flatten();
                let il = ibp.lower();
                let ih = ibp.upper();
                let il = il.as_slice().unwrap();
                let ih = ih.as_slice().unwrap();
                let (cl, cu) = concretize_nary(&bounds, &blo, &bhi, &qbox, &kbox, &vbox);
                // The SHIPPED bound is CROWN∩IBP. It must enclose the FAITHFUL
                // (active_range) forward at every sample — an INDEPENDENT check
                // against the sound truth, not the surrogate's own mask value.
                let mut srng = StdRng::seed_from_u64(99_000 + ci as u64);
                for _ in 0..1500 {
                    let s = |c: &[f32], r: &[f32], rng: &mut StdRng| -> Vec<f32> {
                        c.iter()
                            .zip(r)
                            .map(|(&c, &r)| c + rng.random_range(-1.0..1.0) * r)
                            .collect()
                    };
                    let qs = s(&qc, &qr, &mut srng);
                    let ks = s(&kc, &kr, &mut srng);
                    let vs = s(&vc, &vr, &mut srng);
                    let y = faithful_forward(&qs, &ks, &vs, &dims, scale, mask, window);
                    for o in 0..n_y {
                        let lo = cl[o].max(il[o]);
                        let hi = cu[o].min(ih[o]);
                        assert!(
                            y[o] >= lo - tol && y[o] <= hi + tol,
                            "EMITTED-BOUND SOUNDNESS VIOLATION cfg={ci} ({mask:?} w={window:?} \
                             sq={sq} sk={sk}) o={o}: true={} not in shipped [{lo}, {hi}]",
                            y[o],
                        );
                    }
                }
            }
            Err(NyError::UnsupportedOp(_)) => {
                gated += 1;
                // GATED config: NO false bound escaped. The dispatch falls back to
                // IBP, which for these (forward-invalid sq>sk) configs ALSO refuses —
                // so the framework certifies NOTHING here (sound: never a false [1,1]).
                assert!(
                    !fwd_ok,
                    "cfg={ci} ({mask:?} w={window:?} sq={sq} sk={sk}) was gated but is \
                     forward-valid — the gate should be the SMALLEST sound boundary"
                );
            }
            Err(NyError::InvalidSpec(_)) => {
                // Non-gated forward-invalid (e.g. windowless causal sq>sk): crown's
                // own IBP call refuses. Sound (no bound emitted). Not the gate, but
                // also no false certificate.
                assert!(
                    !fwd_ok,
                    "cfg={ci} unexpectedly InvalidSpec though forward-valid"
                );
            }
            Err(other) => panic!("unexpected error for cfg={ci}: {other:?}"),
        }
    }
    // The sweep must exercise all branches.
    assert!(
        emitted >= 7,
        "expected the forward-valid configs to emit sound bounds, got {emitted}"
    );
    assert_eq!(
        gated, 2,
        "exactly the 2 windowed-causal sq>sk configs must hit the gate, got {gated}"
    );
    assert_eq!(
        fwd_refused, 3,
        "exactly the 3 causal sq>sk configs are forward-refused, got {fwd_refused}"
    );
}

/// Concrete inputs (zero radius) should give bounds that pin the exact output.
#[test]
fn concrete_inputs_pin_output() {
    let cfg = RandConfig {
        n_slices: 1,
        sq: 3,
        sk: 3,
        dk: 2,
        dv: 2,
        mask: AttentionMask::Standard,
        window: None,
    };
    let dims = SliceDims {
        n_slices: 1,
        sq: 3,
        sk: 3,
        dk: 2,
        dv: 2,
    };
    let scale = 1.0 / 2.0_f32.sqrt();
    let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(scale));
    let qc = vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.1];
    let kc = vec![0.2, -0.1, 0.3, 0.2, -0.4, 0.1];
    let vc = vec![1.0, -0.5, 0.3, 0.7, -0.2, 0.9];
    let zeros = vec![0.0f32; 6];
    let qbox = box_from_center(&[3, 2], &qc, &zeros);
    let kbox = box_from_center(&[3, 2], &kc, &zeros);
    let vbox = box_from_center(&[3, 2], &vc, &zeros);
    let n_y = 6;
    let node_lb = identity_node_lb(n_y);
    let (bounds, blo, bhi) = layer
        .propagate_crown_ternary(&node_lb, &qbox, &kbox, &vbox)
        .unwrap();
    let (cl, cu) = concretize_nary(&bounds, &blo, &bhi, &qbox, &kbox, &vbox);
    let y = attention_forward(&qc, &kc, &vc, &dims, scale, cfg.mask, cfg.window);
    for o in 0..n_y {
        assert!(
            (cl[o] - y[o]).abs() < 1e-3 && (cu[o] - y[o]).abs() < 1e-3,
            "concrete out={o}: bound=[{}, {}] true={}",
            cl[o],
            cu[o],
            y[o]
        );
    }
}

/// Tightening report across regimes. Reports the average CROWN∩IBP vs IBP width
/// per regime (honest measurement of where the linear bound helps at the node
/// level). Always asserts soundness + tighter-or-equal-after-intersection.
///
/// HONEST node-local finding (directional O(radius²) margin): at the ISOLATED
/// attention node the surrogate's first-order box-concretization (the Jacobian
/// interval product on Q/K) dominates the bound width, so the raw node CROWN is
/// wider than the simplex-aware IBP and the CROWN∩IBP node width ties IBP. The
/// directional margin's win is GRAPH-level: distinct, tight directional biases
/// survive backsubstitution through downstream layers, where the
/// `end_to_end_*` tests measure ratio ≈ 0.70 (strictly beats IBP). The
/// `small_box_*` regimes here show the raw node CROWN approaching IBP as the
/// radius shrinks (the O(radius²) margin → 0), the node-level signature of the
/// graph-level win.
#[test]
fn tightening_vs_ibp_report() {
    // (label, q_radius, k_radius, v_radius)
    let regimes = [
        ("balanced", 0.15f32, 0.15f32, 0.4f32),
        ("v_dominant_qk_tight", 0.01, 0.01, 0.5),
        ("qk_dominant_v_tight", 0.3, 0.3, 0.01),
        // Small robustness-radius boxes (the regime certified verification uses):
        ("small_box_balanced", 0.03, 0.03, 0.05),
        ("small_box_qk", 0.05, 0.05, 0.01),
    ];
    for (label, qrad, krad, vrad) in regimes {
        run_tightening_regime(label, qrad, krad, vrad);
    }
}

fn run_tightening_regime(label: &str, qrad: f32, krad: f32, vrad: f32) {
    let s = 16usize;
    let dk = 8usize;
    let dv = 8usize;
    let mut rng = StdRng::seed_from_u64(7);
    let dims = SliceDims {
        n_slices: 1,
        sq: s,
        sk: s,
        dk,
        dv,
    };
    let n_q = s * dk;
    let n_v = s * dv;
    let n_y = s * dv;
    let gen_cr = |n: usize, rng: &mut StdRng, cr: f32, rmax: f32| -> (Vec<f32>, Vec<f32>) {
        (
            (0..n).map(|_| rng.random_range(-cr..cr)).collect(),
            (0..n).map(|_| rng.random_range(0.0..rmax)).collect(),
        )
    };
    let (qc, qr) = gen_cr(n_q, &mut rng, 1.0, qrad);
    let (kc, kr) = gen_cr(n_q, &mut rng, 1.0, krad);
    let (vc, vr) = gen_cr(n_v, &mut rng, 1.0, vrad);
    let qbox = box_from_center(&[s, dk], &qc, &qr);
    let kbox = box_from_center(&[s, dk], &kc, &kr);
    let vbox = box_from_center(&[s, dv], &vc, &vr);
    let scale = 1.0 / (dk as f32).sqrt();
    let layer = SelfAttentionLayer::new(AttentionMask::Standard, Some(scale));
    let node_lb = identity_node_lb(n_y);
    let (bounds, blo, bhi) = layer
        .propagate_crown_ternary(&node_lb, &qbox, &kbox, &vbox)
        .unwrap();
    let (cl, cu) = concretize_nary(&bounds, &blo, &bhi, &qbox, &kbox, &vbox);
    let ibp = layer
        .propagate_ibp_ternary(&qbox, &kbox, &vbox)
        .unwrap()
        .flatten();
    let il = ibp.lower();
    let ih = ibp.upper();
    let il = il.as_slice().unwrap();
    let ih = ih.as_slice().unwrap();

    // Soundness gate.
    for _ in 0..2000 {
        let qs: Vec<f32> = qc
            .iter()
            .zip(&qr)
            .map(|(&c, &r)| c + rng.random_range(-1.0..1.0) * r)
            .collect();
        let ks: Vec<f32> = kc
            .iter()
            .zip(&kr)
            .map(|(&c, &r)| c + rng.random_range(-1.0..1.0) * r)
            .collect();
        let vs: Vec<f32> = vc
            .iter()
            .zip(&vr)
            .map(|(&c, &r)| c + rng.random_range(-1.0..1.0) * r)
            .collect();
        let y = attention_forward(&qs, &ks, &vs, &dims, scale, AttentionMask::Standard, None);
        for o in 0..n_y {
            assert!(
                y[o] >= cl[o] - 1e-3 && y[o] <= cu[o] + 1e-3,
                "soundness {label} o={o}: true={} bound=[{}, {}]",
                y[o],
                cl[o],
                cu[o]
            );
        }
    }

    let mut ibp_w = 0.0f64;
    let mut int_w = 0.0f64;
    let mut crown_w = 0.0f64;
    let mut tighter = 0usize;
    for o in 0..n_y {
        let int_lo = cl[o].max(il[o]);
        let int_hi = cu[o].min(ih[o]);
        assert!(
            int_lo >= il[o] - 1e-3 && int_hi <= ih[o] + 1e-3,
            "not tighter-or-equal {label} o={o}"
        );
        let iw = (ih[o] - il[o]) as f64;
        let tw = (int_hi - int_lo).max(0.0) as f64;
        ibp_w += iw;
        int_w += tw;
        crown_w += (cu[o] - cl[o]).max(0.0) as f64;
        if tw + 1e-6 < iw {
            tighter += 1;
        }
    }
    eprintln!(
        "tightening[{label}]: IBP_w={:.5} CROWN_w(raw)={:.5} CROWN∩IBP_w={:.5} factor={:.4} | {tighter}/{n_y} strictly tighter",
        ibp_w / n_y as f64,
        crown_w / n_y as f64,
        int_w / n_y as f64,
        if ibp_w > 0.0 { int_w / ibp_w } else { 1.0 },
    );
}

/// End-to-end graph CROWN through a FUSED SelfAttention node (added via
/// `add_node`, so it stays fused exactly as nn-verify emits it). Builds
/// `x -> {q,k,v identity} -> SelfAttention -> Linear readout` and compares the
/// dense graph CROWN output to IBP. Asserts: (1) CROWN ⊆ IBP (sound — the engine
/// intersects with the IBP forward bound), (2) the result encloses sampled true
/// outputs, (3) the directional O(radius²) margin STRICTLY beats IBP end-to-end
/// (ratio < 1) — the linear map + tight directional bias compose backward
/// through the readout, where the per-node bound merely ties IBP.
#[test]
fn end_to_end_graph_crown_vs_ibp() {
    use crate::layers::{LinearLayer, ReshapeLayer};
    use crate::network::{GraphNetwork, GraphNetworkCrownExt, GraphNode};
    use crate::{Layer, MulBinaryRelaxationMode};

    // Enable the opt-in fused-attention CROWN ternary backward for this test.
    // Serialized + restored via the blessed env choke point (clippy env wall).
    let _env_lock = ny_test_utils::env::lock_env();
    let _gate = ny_test_utils::env::ScopedEnvVar::set("NY_ATTN_CROWN_TERNARY", "1");

    // Self-attention with Q=K=V=x (single head, seq=4, head_dim=3).
    let s = 4usize;
    let d = 3usize;
    let n = s * d;
    let scale = 1.0 / (d as f32).sqrt();

    let mut graph = GraphNetwork::new();
    // Identity projections so attention sees three graph edges from the input.
    let eye = Array2::<f32>::eye(n);
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
    ));
    // q/k/v carry shape [n]; reshape to [s, d] for attention.
    for name in ["q", "k", "v"] {
        graph.add_node(GraphNode::new(
            format!("{name}r"),
            Layer::Reshape(ReshapeLayer::new(vec![s as i64, d as i64])),
            vec![name.to_string()],
        ));
    }
    // FUSED attention (add_node — NOT decomposed).
    graph.add_node(GraphNode::new(
        "attn",
        Layer::SelfAttention(SelfAttentionLayer::new(
            AttentionMask::Standard,
            Some(scale),
        )),
        vec!["qr".to_string(), "kr".to_string(), "vr".to_string()],
    ));
    // Downstream readout: flatten attn [s,d] -> [n], then a linear to 2 outputs.
    graph.add_node(GraphNode::new(
        "attn_flat",
        Layer::Reshape(ReshapeLayer::new(vec![n as i64])),
        vec!["attn".to_string()],
    ));
    let mut w = Array2::<f32>::zeros((2, n));
    for j in 0..n {
        w[[0, j]] = if j % 2 == 0 { 1.0 } else { -1.0 };
        w[[1, j]] = 0.5;
    }
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(w.clone(), None).unwrap()),
        vec!["attn_flat".to_string()],
    ));
    graph.set_output("out");

    // Input box: x in [center ± 0.1].
    let mut rng = StdRng::seed_from_u64(3);
    let xc: Vec<f32> = (0..n).map(|_| rng.random_range(-0.5..0.5)).collect();
    let xr: Vec<f32> = (0..n).map(|_| rng.random_range(0.05..0.15)).collect();
    let xbox = box_from_center(&[n], &xc, &xr);

    // The dense graph CROWN forward+backward goes through our ternary dispatch
    // arm for the fused SelfAttention node (added via add_node, NOT decomposed).
    // `crown_backward_with_relaxation` is the path nn-verify uses: it collects
    // node IBP bounds via the ternary-aware `collect_node_bounds` and intersects
    // the concretized CROWN output with the IBP forward bound.
    let crown = GraphNetworkCrownExt::crown_backward_with_relaxation(
        &graph,
        &xbox,
        None,
        MulBinaryRelaxationMode::default(),
    )
    .unwrap();
    let cl = crown.lower();
    let cu = crown.upper();
    let cl = cl.as_slice().unwrap();
    let cu = cu.as_slice().unwrap();

    // Ternary-aware end-to-end IBP (the fallback CROWN beats / ties).
    let node_bounds = graph.collect_node_bounds(&xbox).unwrap();
    let ibp_out = node_bounds.get("out").unwrap().flatten();
    let il = ibp_out.lower();
    let ih = ibp_out.upper();
    let il = il.as_slice().unwrap();
    let ih = ih.as_slice().unwrap();
    // CROWN must be tighter-or-equal to IBP end-to-end (engine intersects).
    for o in 0..2 {
        assert!(
            cl[o] >= il[o] - 2e-3 && cu[o] <= ih[o] + 2e-3,
            "end-to-end CROWN looser than IBP o={o}: crown=[{}, {}] ibp=[{}, {}]",
            cl[o],
            cu[o],
            il[o],
            ih[o]
        );
    }

    // Soundness: sample inputs, run the true network forward, enclose.
    let dims = SliceDims {
        n_slices: 1,
        sq: s,
        sk: s,
        dk: d,
        dv: d,
    };
    for _ in 0..2000 {
        let xs: Vec<f32> = xc
            .iter()
            .zip(&xr)
            .map(|(&c, &r)| c + rng.random_range(-1.0..1.0) * r)
            .collect();
        let y = attention_forward(&xs, &xs, &xs, &dims, scale, AttentionMask::Standard, None);
        // out = w @ y_flat
        for o in 0..2 {
            let mut z = 0.0f64;
            for j in 0..n {
                z += w[[o, j]] as f64 * y[j] as f64;
            }
            let z = z as f32;
            assert!(
                z >= cl[o] - 2e-3 && z <= cu[o] + 2e-3,
                "end-to-end soundness o={o}: true={z} crown=[{}, {}]",
                cl[o],
                cu[o]
            );
        }
    }

    let crown_w: f64 = (0..2).map(|o| (cu[o] - cl[o]) as f64).sum::<f64>() / 2.0;
    let ibp_w: f64 = (0..2).map(|o| (ih[o] - il[o]) as f64).sum::<f64>() / 2.0;
    eprintln!(
        "end_to_end_graph_crown_vs_ibp: fused-attention graph CROWN succeeded (was IBP-fallback). \
         IBP avg width={ibp_w:.5} CROWN avg width={crown_w:.5} ratio={:.4} (sound, encloses 2000 samples)",
        if ibp_w > 0.0 { crown_w / ibp_w } else { 1.0 },
    );
    // TIGHTENING GATE: directional margin strictly beats IBP end-to-end.
    assert!(
        crown_w + 1e-4 < ibp_w,
        "directional margin failed to beat IBP end-to-end: CROWN avg {crown_w:.5} \
         vs IBP avg {ibp_w:.5}"
    );
}

/// Honest end-to-end characterization with a nonlinearity (ReLU) AFTER
/// attention: `x -> {q,k,v} -> SelfAttention -> ReLU -> Linear`. The fused
/// ternary backward un-blocks CROWN continuation past attention (pre-change it
/// returned `UnsupportedOp`, aborting the WHOLE graph to IBP). This test asserts
/// SOUNDNESS (encloses samples) and tighter-or-equal-vs-IBP, and REPORTS the
/// ratio. Finding (directional O(radius²) margin): the end-to-end ratio is
/// STRICTLY < 1 (≈0.70) — the tight directional bias channel survives
/// backsubstitution through the downstream ReLU+Linear, so the composed CROWN
/// bound beats the simplex-aware IBP end-to-end. (The original symmetric
/// IBP-sized margin produced ratio≈1.0 — a tie; the directional margin is the
/// genuine improvement.) Soundness + tighter-or-equal are asserted; the ratio is
/// reported and lightly gated (`strictly_tighter ≥ 1`).
#[test]
fn end_to_end_crown_continues_past_attention() {
    use crate::layers::{LinearLayer, ReLULayer, ReshapeLayer};
    use crate::network::{GraphNetwork, GraphNetworkCrownExt, GraphNode};
    use crate::{Layer, MulBinaryRelaxationMode};

    // Enable the opt-in fused-attention CROWN ternary backward for this test.
    // Serialized + restored via the blessed env choke point (clippy env wall).
    let _env_lock = ny_test_utils::env::lock_env();
    let _gate = ny_test_utils::env::ScopedEnvVar::set("NY_ATTN_CROWN_TERNARY", "1");

    let s = 4usize;
    let d = 3usize;
    let n = s * d;
    let scale = 1.0 / (d as f32).sqrt();

    let mut graph = GraphNetwork::new();
    let eye = Array2::<f32>::eye(n);
    for name in ["q", "k", "v"] {
        graph.add_node(GraphNode::from_input(
            name,
            Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            format!("{name}r"),
            Layer::Reshape(ReshapeLayer::new(vec![s as i64, d as i64])),
            vec![name.to_string()],
        ));
    }
    graph.add_node(GraphNode::new(
        "attn",
        Layer::SelfAttention(SelfAttentionLayer::new(
            AttentionMask::Standard,
            Some(scale),
        )),
        vec!["qr".to_string(), "kr".to_string(), "vr".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "attn_flat",
        Layer::Reshape(ReshapeLayer::new(vec![n as i64])),
        vec!["attn".to_string()],
    ));
    // Post-attention nonlinearity — this is where CROWN beats IBP.
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["attn_flat".to_string()],
    ));
    let mut w = Array2::<f32>::zeros((2, n));
    for j in 0..n {
        w[[0, j]] = if j % 2 == 0 { 1.0 } else { -1.0 };
        w[[1, j]] = 0.7;
    }
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(w.clone(), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("out");

    let mut rng = StdRng::seed_from_u64(9);
    let xc: Vec<f32> = (0..n).map(|_| rng.random_range(-0.4..0.4)).collect();
    let xr: Vec<f32> = (0..n).map(|_| rng.random_range(0.05..0.12)).collect();
    let xbox = box_from_center(&[n], &xc, &xr);

    let crown = GraphNetworkCrownExt::crown_backward_with_relaxation(
        &graph,
        &xbox,
        None,
        MulBinaryRelaxationMode::default(),
    )
    .unwrap();
    let cl = crown.lower();
    let cu = crown.upper();
    let cl = cl.as_slice().unwrap();
    let cu = cu.as_slice().unwrap();

    let node_bounds = graph.collect_node_bounds(&xbox).unwrap();
    let ibp_out = node_bounds.get("out").unwrap().flatten();
    let il = ibp_out.lower();
    let ih = ibp_out.upper();
    let il = il.as_slice().unwrap();
    let ih = ih.as_slice().unwrap();

    // Soundness: enclose sampled true outputs (attention -> relu -> w).
    let dims = SliceDims {
        n_slices: 1,
        sq: s,
        sk: s,
        dk: d,
        dv: d,
    };
    for _ in 0..2000 {
        let xs: Vec<f32> = xc
            .iter()
            .zip(&xr)
            .map(|(&c, &r)| c + rng.random_range(-1.0..1.0) * r)
            .collect();
        let y = attention_forward(&xs, &xs, &xs, &dims, scale, AttentionMask::Standard, None);
        for o in 0..2 {
            let mut z = 0.0f64;
            for j in 0..n {
                z += w[[o, j]] as f64 * (y[j].max(0.0)) as f64;
            }
            let z = z as f32;
            assert!(
                z >= cl[o] - 2e-3 && z <= cu[o] + 2e-3,
                "soundness o={o}: true={z} crown=[{}, {}]",
                cl[o],
                cu[o]
            );
        }
    }

    // CROWN must be tighter-or-equal to IBP end-to-end (the engine intersects).
    let mut strictly_tighter = 0;
    for o in 0..2 {
        assert!(
            cl[o] >= il[o] - 2e-3 && cu[o] <= ih[o] + 2e-3,
            "CROWN looser than IBP o={o}: crown=[{}, {}] ibp=[{}, {}]",
            cl[o],
            cu[o],
            il[o],
            ih[o]
        );
        if (cu[o] - cl[o]) + 1e-4 < (ih[o] - il[o]) {
            strictly_tighter += 1;
        }
    }
    let crown_w: f64 = (0..2).map(|o| (cu[o] - cl[o]) as f64).sum::<f64>() / 2.0;
    let ibp_w: f64 = (0..2).map(|o| (ih[o] - il[o]) as f64).sum::<f64>() / 2.0;
    eprintln!(
        "crown_continues_past_attention: IBP avg width={ibp_w:.5} CROWN avg width={crown_w:.5} \
         ratio={:.4} | {strictly_tighter}/2 outputs strictly tighter (sound + memory-bounded; \
         directional O(radius²) margin BEATS the simplex-aware IBP end-to-end)",
        if ibp_w > 0.0 { crown_w / ibp_w } else { 1.0 },
    );
    // TIGHTENING GATE: the directional O(radius²) margin must STRICTLY beat IBP
    // end-to-end here (the win the foundation's symmetric margin could not get).
    assert!(
        crown_w + 1e-4 < ibp_w,
        "directional margin failed to beat IBP end-to-end: CROWN avg {crown_w:.5} \
         vs IBP avg {ibp_w:.5}"
    );
    assert!(
        strictly_tighter >= 1,
        "expected at least one output strictly tighter than IBP, got {strictly_tighter}"
    );
}

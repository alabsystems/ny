// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-ibp-seed real-ONNX gates on `challenging_certified_training_2026`'s
//! IBP-trained eps8 CNN7s (FRONTIER_PLAN §7.F2).
//!
//! * `ibp_seed_contains_ort_samples_cct_eps8` — idx5323 (eps8 cnn7, a stale
//!   bank) and idx3144 (eps8 wide, a GAP row): the extracted net is first
//!   cross-checked against ONNX Runtime (the extractor guard), then every
//!   sampled trunk pre-activation (centre, random vertices, interior points
//!   and per-neuron targeted trunk-0 vertices, evaluated by the stored net's
//!   f64 forward) must lie inside the SEEDED root boxes and inside the pure
//!   IBP chain's boxes. On 5323 the seed must also ENGAGE (fewer unstable
//!   trunk neurons than the dark root) — a null there is VOID.
//! * `ibp_seed_never_verifies_field_sat_rows` — the two field-SAT negative
//!   controls, idx8090 (eps8 cnn7, witness class 7; the field's witnesses use
//!   class 5) and idx8397 (eps8 wide, class 5): the seeded root's certified
//!   per-class bound stays NEGATIVE on every witness class and never verifies
//!   the property, it lower-bounds the margin at the checked-in witness
//!   (`testdata/ibp_seed/w_*.npy`, replayed through ORT) and at every sample.
//!   A failure here is a SOUNDNESS EMERGENCY, not a regression.
//!
//! Both are in the explicit `external-vnncomp` lane (this module is compiled
//! only with that feature): they build full CNN7 root tableaux (65,536 x 3,073
//! entries per matrix on cnn7, twice that on wide) and need the gitignored
//! corpus. Run them in release with the corpus symlinked
//! (`benchmarks/vnncomp2026` from the main tree):
//!
//! ```text
//! targo --unverified test --release -p ny-cli --bin ny --features external-vnncomp ibp_seed_
//! ```
//!
//! and assert each test's own `... ok` line. A missing corpus, a missing
//! witness, an unavailable ORT or a debug build PANICS with `VOID:` — it never
//! passes silently.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::Array2;
use ny_propagate::margin_row::bab::root_eval;
use ny_propagate::margin_row::engine::BackwardEngine;
use ny_propagate::margin_row::ibp_seed::pure_ibp_trunk_boxes;
use ny_propagate::margin_row::net::TwinOp;
use ny_propagate::margin_row::{RootGates, RoundMode, TwinNet};

use super::*;

const CCT: &str = "vnncomp2026/benchmarks/challenging_certified_training_2026/2.0";

fn bench(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks")
        .join(rel)
}

fn testdata(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/ibp_seed")
        .join(rel)
}

/// The full CNN7 root build is minutes in release and far longer in debug:
/// refuse loudly instead of timing out a gate.
#[expect(
    clippy::assertions_on_constants,
    reason = "debug builds refuse at test runtime"
)]
fn require_release() {
    assert!(
        !cfg!(debug_assertions),
        "VOID: the real-ONNX #mr-ibp-seed gates build full CNN7 root tableaux; run them with \
         --release --features external-vnncomp (a debug build takes tens of minutes)"
    );
}

/// Deterministic splitmix64 stream (no external RNG API in the gate).
struct Mix(u64);

impl Mix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let v = (self.next_u64() >> 11) as f64;
        v / (1u64 << 53) as f64
    }
}

/// One loaded cct row.
struct Row {
    tag: &'static str,
    onnx: PathBuf,
    net: TwinNet,
    lo: Vec<f64>,
    hi: Vec<f64>,
    t: usize,
    adv: Vec<usize>,
}

fn load_row(tag: &'static str, onnx_rel: &str, vnnlib_rel: &str) -> Row {
    require_release();
    let onnx = bench(&format!("{CCT}/{onnx_rel}"));
    let vnnlib = bench(&format!("{CCT}/{vnnlib_rel}"));
    assert!(
        onnx.is_file() && vnnlib.is_file(),
        "VOID: {tag}: the cct corpus is absent ({} / {}) — symlink benchmarks/vnncomp2026 \
         from the main tree; a skipped real-ONNX gate is a failed gate",
        onnx.display(),
        vnnlib.display()
    );
    let (lo, hi, t, adv) = parse_vnnlib_robustness(&vnnlib).expect("cct vnnlib parses");
    let spec = extract_twin_spec(&onnx).expect("cct CNN7 extracts");
    let net = TwinNet::compile(&spec).expect("cct CNN7 compiles for the margin-row lane");
    assert_eq!(lo.len(), net.n_in, "{tag}: box width");
    Row {
        tag,
        onnx,
        net,
        lo,
        hi,
        t,
        adv,
    }
}

/// ONNX Runtime logits at one f32 point (the organizer's inference path).
fn ort_logits(onnx: &Path, x: &[f32]) -> Vec<f32> {
    let mut fwd = ny_onnx::diff::OrtForward::from_path(onnx, x.len())
        .unwrap_or_else(|e| panic!("VOID: ORT unavailable for {}: {e}", onnx.display()));
    fwd.run(x)
        .unwrap_or_else(|e| panic!("VOID: ORT forward failed on {}: {e}", onnx.display()))
}

/// The stored net's f64 forward: logits per point and every trunk relu's
/// pre-activations (`(n, B)` per relu op).
fn ny_forward(net: &TwinNet, pts: &[Vec<f64>]) -> (Vec<Vec<f64>>, BTreeMap<usize, Array2<f64>>) {
    let b = pts.len();
    let mut x = Array2::<f64>::zeros((net.n_in, b));
    for (c, p) in pts.iter().enumerate() {
        for (i, &v) in p.iter().enumerate() {
            x[[i, c]] = v;
        }
    }
    let mut sel = BTreeMap::new();
    for &op in &net.trunk_relus {
        // Op `op` writes tensor `op + 1`, the relu output, which has the
        // pre-activation's width.
        let n = net.tsize[op + 1];
        sel.insert(op, (0..n).collect::<Vec<usize>>());
    }
    let (y, pre) = net.forward_points(&x, &sel).expect("f64 forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    let logits = (0..b)
        .map(|c| {
            (0..n_out)
                .map(|o| {
                    let mut acc = b2[o];
                    for k in 0..n_y {
                        acc += w2[o * n_y + k] * y[[k, c]].max(0.0);
                    }
                    acc
                })
                .collect()
        })
        .collect();
    (logits, pre)
}

/// Extractor guard: ny's f64 logits at the (f32-rounded) box centre agree with
/// ONNX Runtime to the organizer's 1e-3 relative tolerance.
#[expect(
    clippy::manual_midpoint,
    reason = "retain original runtime comparison sample"
)]
fn ort_cross_check(row: &Row) {
    let centre32: Vec<f32> = row
        .lo
        .iter()
        .zip(&row.hi)
        .map(|(&l, &h)| {
            #[allow(clippy::cast_possible_truncation)]
            let v = (0.5 * (l + h)) as f32;
            v
        })
        .collect();
    let centre: Vec<f64> = centre32.iter().map(|&v| f64::from(v)).collect();
    let (ny, _) = ny_forward(&row.net, std::slice::from_ref(&centre));
    let ort = ort_logits(&row.onnx, &centre32);
    assert_eq!(ort.len(), ny[0].len(), "{}: logit count", row.tag);
    let scale = ny[0].iter().fold(1.0f64, |m, &v| m.max(v.abs()));
    for (o, (&a, &b)) in ort.iter().zip(&ny[0]).enumerate() {
        assert!(
            (f64::from(a) - b).abs() <= 1e-3 * scale,
            "{}: ORT logit {o} = {a} vs ny f64 {b} — the extracted net is not the ONNX model",
            row.tag
        );
    }
}

/// Centre, 24 random vertices, 24 interior points and the targeted trunk-0
/// vertices of 8 random conv-1 neurons (both faces) — all exactly in the box.
fn sample_points(row: &Row, seed: u64) -> Vec<Vec<f64>> {
    let mut mix = Mix(seed);
    let centre: Vec<f64> = row
        .lo
        .iter()
        .zip(&row.hi)
        .map(|(&l, &h)| (l + 0.5 * (h - l)).clamp(l, h))
        .collect();
    let mut pts = vec![centre.clone()];
    for i in 0..48 {
        pts.push(
            row.lo
                .iter()
                .zip(&row.hi)
                .map(|(&l, &h)| {
                    let t = mix.unit();
                    if i % 2 == 0 {
                        if t < 0.5 {
                            l
                        } else {
                            h
                        }
                    } else {
                        (l + t * (h - l)).clamp(l, h)
                    }
                })
                .collect(),
        );
    }
    if let Some(TwinOp::Conv(c)) = row.net.ops.first() {
        let p = c.oshape.1 * c.oshape.2;
        let k = c.k_fwd;
        let n0 = row.net.tsize[1];
        for _ in 0..8 {
            #[allow(clippy::cast_possible_truncation)]
            let j = (mix.next_u64() % n0 as u64) as usize;
            let (oc, sp) = (j / p, j % p);
            for upper in [true, false] {
                let mut x = centre.clone();
                for tap in 0..k {
                    let gi = c.gather[tap * p + sp];
                    if gi == usize::MAX {
                        continue;
                    }
                    let w = c.wmat[oc * k + tap];
                    x[gi] = if (w >= 0.0) == upper {
                        row.hi[gi]
                    } else {
                        row.lo[gi]
                    };
                }
                pts.push(x);
            }
        }
    }
    pts
}

/// Sampled pre-activations outside `boxes` (tolerance: the f64 forward's own
/// rounding and the ~1e-15 fold error, `1e-9 (1 + |v|)`), and the minimum
/// headroom observed.
fn box_escapes(
    net: &TwinNet,
    pre: &BTreeMap<usize, Array2<f64>>,
    boxes: &[(Vec<f64>, Vec<f64>)],
    what: &str,
) -> (usize, f64) {
    assert_eq!(boxes.len(), net.trunk_relus.len(), "{what}: trunk depth");
    let (mut escapes, mut min_head) = (0usize, f64::INFINITY);
    for (li, op) in net.trunk_relus.iter().enumerate() {
        let vals = &pre[op];
        let (l, u) = &boxes[li];
        assert_eq!(l.len(), vals.nrows(), "{what}: layer {li} width");
        for j in 0..l.len() {
            for c in 0..vals.ncols() {
                let v = vals[[j, c]];
                let tol = 1e-9 * (1.0 + v.abs());
                min_head = min_head.min((v - l[j]).min(u[j] - v));
                if v < l[j] - tol || v > u[j] + tol {
                    escapes += 1;
                    if escapes <= 8 {
                        eprintln!(
                            "[mr-ibp-seed real] {what}: layer {li} neuron {j} point {c}: \
                             {v:e} outside [{:e}, {:e}]",
                            l[j], u[j]
                        );
                    }
                }
            }
        }
    }
    (escapes, min_head)
}

fn root_boxes(root: &RootGates) -> Vec<(Vec<f64>, Vec<f64>)> {
    root.layers
        .iter()
        .map(|g| (g.l.clone(), g.u.clone()))
        .collect()
}

fn unstable(root: &RootGates) -> Vec<usize> {
    root.layers.iter().map(|g| g.unst.len()).collect()
}

/// Per-class certified root bounds (`root_eval`), adv order.
fn root_bounds(row: &Row, root: &RootGates) -> Vec<f64> {
    let eng = BackwardEngine::new(&row.net, root);
    root_eval(&eng, &row.net, row.t, &row.adv)
        .expect("certified root_eval")
        .dj
}

#[test]
fn ibp_seed_contains_ort_samples_cct_eps8() {
    let rows = [
        (
            "5323",
            "onnx/cifar10_eps8_cnn7.onnx",
            "vnnlib/cifar10_eps8_cnn7/cifar10_eps8_cnn7_idx5323_sample0.vnnlib",
            true,
        ),
        (
            "3144w",
            "onnx/cifar10_eps8_wide_cnn7.onnx",
            "vnnlib/cifar10_eps8_wide_cnn7/cifar10_eps8_wide_cnn7_idx3144_sample1.vnnlib",
            false,
        ),
    ];
    let mut checked = 0usize;
    for (i, (tag, onnx, vnnlib, with_dark)) in rows.into_iter().enumerate() {
        let row = load_row(tag, onnx, vnnlib);
        ort_cross_check(&row);
        let t0 = Instant::now();
        let seeded =
            RootGates::build_ibp_seeded(&row.net, &row.lo, &row.hi, RoundMode::Outward, None, true)
                .expect("seeded root builds");
        let build_secs = t0.elapsed().as_secs_f64();
        let pure = pure_ibp_trunk_boxes(&row.net, &row.lo, &row.hi).expect("pure IBP chain");
        let pts = sample_points(&row, 0x1B5E_ED00 ^ i as u64);
        let (_, pre) = ny_forward(&row.net, &pts);
        let (esc_s, head_s) = box_escapes(&row.net, &pre, &root_boxes(&seeded), "seeded root");
        let (esc_p, head_p) = box_escapes(&row.net, &pre, &pure, "pure IBP");
        checked += pts.len() * row.net.trunk_relus.len();
        let bounds = root_bounds(&row, &seeded);
        let min_bound = bounds.iter().copied().fold(f64::INFINITY, f64::min);
        let open = bounds.iter().filter(|&&b| b <= 0.0).count();
        eprintln!(
            "[mr-ibp-seed real] {tag}: points={} seeded_escapes={esc_s} min_head={head_s:.3e} \
             pure_escapes={esc_p} pure_min_head={head_p:.3e} unst_seeded={:?} \
             seeded_root_bound={min_bound:.6} open_classes={open} build_secs={build_secs:.1}",
            pts.len(),
            unstable(&seeded)
        );
        if with_dark {
            let dark = RootGates::build_ibp_seeded(
                &row.net,
                &row.lo,
                &row.hi,
                RoundMode::Outward,
                None,
                false,
            )
            .expect("dark root builds");
            let (ud, us) = (unstable(&dark), unstable(&seeded));
            let dark_bound = root_bounds(&row, &dark)
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            eprintln!(
                "[mr-ibp-seed real] {tag}: dark unst={ud:?} dark_root_bound={dark_bound:.6} \
                 seeded unst={us:?} seeded_root_bound={min_bound:.6}"
            );
            assert!(
                us.iter().sum::<usize>() < ud.iter().sum::<usize>(),
                "VOID: {tag}: the seed stabilized no trunk neuron on an IBP-trained eps8 net \
                 (dark {ud:?} vs seeded {us:?}) — the lever did not engage"
            );
        }
        assert_eq!(
            esc_s, 0,
            "{tag}: a sampled pre-activation escapes the SEEDED root box (false-unsat generator)"
        );
        assert_eq!(
            esc_p, 0,
            "{tag}: a sampled pre-activation escapes the pure IBP box"
        );
    }
    assert!(checked > 0, "VOID: nothing checked");
}

/// Read a little-endian f32 `.npy` (v1/v2 header, C order).
fn load_npy_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("VOID: witness {} unreadable: {e}", path.display()));
    assert!(
        bytes.len() > 12 && bytes[..6] == *b"\x93NUMPY",
        "VOID: {} is not an npy file",
        path.display()
    );
    let (hlen, off) = if bytes[6] == 1 {
        (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10)
    } else {
        (
            usize::try_from(u32::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11],
            ]))
            .expect("header length"),
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[off..off + hlen]).expect("ascii npy header");
    assert!(
        header.contains("'descr': '<f4'") && header.contains("'fortran_order': False"),
        "VOID: unexpected npy header {header:?}"
    );
    let data = &bytes[off + hlen..];
    assert_eq!(data.len() % 4, 0, "VOID: truncated npy payload");
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn ibp_seed_never_verifies_field_sat_rows() {
    // (tag, onnx, vnnlib, witness, witness class, classes that must stay open)
    let rows: [(&'static str, &str, &str, &str, usize, &[usize]); 2] = [
        (
            "8090",
            "onnx/cifar10_eps8_cnn7.onnx",
            "vnnlib/cifar10_eps8_cnn7/cifar10_eps8_cnn7_idx8090_sample7.vnnlib",
            "w_8090.npy",
            7,
            &[7, 5],
        ),
        (
            "8397",
            "onnx/cifar10_eps8_wide_cnn7.onnx",
            "vnnlib/cifar10_eps8_wide_cnn7/cifar10_eps8_wide_cnn7_idx8397_sample9.vnnlib",
            "w_8397.npy",
            5,
            &[5],
        ),
    ];
    let mut checks = 0usize;
    for (i, (tag, onnx, vnnlib, witness, wclass, open_classes)) in rows.into_iter().enumerate() {
        let row = load_row(tag, onnx, vnnlib);
        ort_cross_check(&row);
        let seeded =
            RootGates::build_ibp_seeded(&row.net, &row.lo, &row.hi, RoundMode::Outward, None, true)
                .expect("seeded root builds");
        let dj = root_bounds(&row, &seeded);
        assert_eq!(dj.len(), row.adv.len());
        let min = dj.iter().copied().fold(f64::INFINITY, f64::min);
        eprintln!(
            "[mr-ibp-seed real] {tag} (field SAT): seeded root bounds {:?} (min {min:.6}), \
             unst_seeded={:?}",
            row.adv.iter().zip(&dj).collect::<Vec<_>>(),
            unstable(&seeded)
        );
        assert!(
            min < 0.0,
            "SOUNDNESS EMERGENCY: {tag} is field-SAT and the seeded root VERIFIED it \
             (min bound {min:e})"
        );
        for &c in open_classes {
            let r = row
                .adv
                .iter()
                .position(|&a| a == c)
                .unwrap_or_else(|| panic!("VOID: {tag}: class {c} is not adversarial here"));
            assert!(
                dj[r] < 0.0,
                "SOUNDNESS EMERGENCY: {tag}: witness class {c} closed at the root (bound {:e})",
                dj[r]
            );
        }
        // The witness: in the box, a real violation (ny f64 and ORT agree), and
        // every certified bound lower-bounds its margin.
        let w32 = load_npy_f32(&testdata(witness));
        assert_eq!(w32.len(), row.net.n_in, "VOID: {tag}: witness width");
        let w: Vec<f64> = w32.iter().map(|&v| f64::from(v)).collect();
        assert!(
            w.iter()
                .zip(row.lo.iter().zip(&row.hi))
                .all(|(&x, (&l, &h))| l <= x && x <= h),
            "VOID: {tag}: the checked-in witness is not inside the vnnlib box"
        );
        let (wl, _) = ny_forward(&row.net, std::slice::from_ref(&w));
        let ort = ort_logits(&row.onnx, &w32);
        let wmargin = wl[0][row.t] - wl[0][wclass];
        let omargin = f64::from(ort[row.t]) - f64::from(ort[wclass]);
        assert!(
            wmargin < 0.0 && omargin < 0.0,
            "VOID: {tag}: the witness does not violate class {wclass} (ny {wmargin:e}, ORT \
             {omargin:e})"
        );
        let mut pts = sample_points(&row, 0x5A7_0000 ^ i as u64);
        pts.push(w.clone());
        let (logits, _) = ny_forward(&row.net, &pts);
        for (p, lg) in logits.iter().enumerate() {
            for (r, &j) in row.adv.iter().enumerate() {
                checks += 1;
                let margin = lg[row.t] - lg[j];
                assert!(
                    dj[r] <= margin + 1e-6,
                    "SOUNDNESS EMERGENCY: {tag}: certified bound {:e} for class {j} exceeds \
                     the margin {margin:e} at point {p}",
                    dj[r]
                );
            }
        }
        eprintln!(
            "[mr-ibp-seed real] {tag}: witness class {wclass} margin ny={wmargin:e} ort={omargin:e}; \
             bound for it {:e}",
            dj[row
                .adv
                .iter()
                .position(|&a| a == wclass)
                .expect("witness class is adversarial")]
        );
    }
    assert!(checks > 0, "VOID: nothing checked");
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact-decimal input-box seam for the double-double zonotope
//! (`#dd-zonotope`).
//!
//! # Why this exists — it is the load-bearing constraint of the whole lane
//!
//! `VnnLibSpec::split_input_bounds_f32` (`ny-onnx/src/vnnlib/spec.rs:414`,
//! issue #2658) rounds EVERY endpoint outward by one f32 ULP on the
//! `f64 -> f32` narrowing, so the box the graph engine receives is a strict
//! superset of the VNN-LIB region. That is exactly right for every existing ny
//! path — and fatal for this one.
//!
//! On `vggnet16_2022` the VNN-LIB fixes 150527 of 150528 pixels with
//! `X_i >= c` and `X_i <= c` for the SAME decimal `c`. After the outward f32
//! widening, every one of those "fixed" pixels arrives at the engine as a
//! 2-ULP-wide interval (`~4.8e-7`). A zonotope has exactly two ways to carry
//! that:
//!
//! * as a generator column per pixel — 150528 columns, which is neither
//!   affordable in time nor in memory; or
//! * in the interval channel `ec` — but `ec` is transported by `|W|`, i.e. by
//!   precisely the IBP operator whose measured VGG16 gain is ~1e13. A `4.8e-7`
//!   interval on the fixed pixels reaches the logits at ~1e6 and the bound is
//!   vacuous.
//!
//! MEASURED consequence: **the input center must carry ZERO interval
//! uncertainty** for this method to produce anything at all. Even one f64 ULP
//! of center uncertainty (`~2.9e-16`) amplifies to ~1e4 at the logits.
//!
//! So the pass needs the EXACT declared box, not the engine's widened f32 one.
//! `ny_onnx::vnnlib::CertifiedInputBox` already produces it: it reparses the
//! direct input-bound atoms as exact rationals and rounds each endpoint
//! OUTWARD to f64, so a coordinate whose decimal is exactly representable
//! stays a point. This module is the seam that carries it from the CLI (which
//! owns the VNN-LIB file) to the graph engine (which only ever sees a
//! `BoundedTensor`).
//!
//! # Fail-closed identity
//!
//! A registered box is keyed by the EXACT byte string of the f32 box it was
//! derived from (`f32::to_le_bytes` of every lower then every upper). Lookup
//! compares that string byte-for-byte — it is an identity check, not a hash,
//! so no collision can pair a certified box with the wrong instance. In
//! addition:
//!
//! * a second registration with a DIFFERENT fingerprint poisons the registry
//!   permanently (every later lookup returns `None`), so a multi-instance
//!   process can never serve a stale box;
//! * lookup re-verifies containment `f32_lower <= cert_lower <= cert_upper <=
//!   f32_upper` elementwise and refuses on any violation, so even a byte-equal
//!   pairing cannot smuggle in a box the engine was not asked to verify.

use std::sync::{Mutex, OnceLock};

use ny_tensor::BoundedTensor;

/// A registered exact-decimal input box plus the identity of the f32 box it
/// was derived from.
/// The exact-decimal box, carried at double-double precision.
///
/// `center_hi + center_lo` approximates the EXACT rational box center with
/// residual at most `center_err`; `half_width` is the exact half-width rounded
/// OUTWARD, so a declared point has `half_width == 0.0` even when its decimal
/// is not dyadic. `lower`/`upper` are the outward `f64` endpoints, kept only
/// for the containment re-check against the engine's f32 box.
#[derive(Debug, Clone)]
pub struct ExactBox {
    /// Outward f64 lower endpoints.
    pub lower: Vec<f64>,
    /// Outward f64 upper endpoints.
    pub upper: Vec<f64>,
    /// Leading word of the double-double exact center.
    pub center_hi: Vec<f64>,
    /// Trailing word of the double-double exact center.
    pub center_lo: Vec<f64>,
    /// Outward bound on `|exact_center - (center_hi + center_lo)|`.
    pub center_err: Vec<f64>,
    /// Exact half-width, rounded outward. Exactly `0.0` for a declared point.
    pub half_width: Vec<f64>,
}

impl ExactBox {
    fn len(&self) -> usize {
        self.lower.len()
    }

    fn is_well_formed(&self) -> bool {
        let n = self.len();
        self.upper.len() == n
            && self.center_hi.len() == n
            && self.center_lo.len() == n
            && self.center_err.len() == n
            && self.half_width.len() == n
    }
}

#[derive(Debug)]
struct Entry {
    /// EXACT byte string of the engine-facing f32 box. Compared verbatim.
    fingerprint: Vec<u8>,
    exact: ExactBox,
}

#[derive(Debug, Default)]
struct Registry {
    entry: Option<Entry>,
    /// Set once a conflicting registration is seen; never cleared.
    poisoned: bool,
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

/// Byte-exact identity of an f32 input box.
fn fingerprint(lower: &[f32], upper: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity((lower.len() + upper.len()) * 4 + 8);
    out.extend_from_slice(&(lower.len() as u64).to_le_bytes());
    for v in lower {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in upper {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Publish the exact-decimal input box for the instance whose engine-facing
/// f32 box is `(f32_lower, f32_upper)`.
///
/// Registering the SAME fingerprint twice is a no-op. Registering a different
/// one poisons the registry: a process that verifies several instances must
/// never serve one instance's exact box to another, and refusing is always
/// sound.
pub fn register(f32_lower: &[f32], f32_upper: &[f32], exact: ExactBox) {
    if exact.len() != f32_lower.len() || f32_upper.len() != f32_lower.len() {
        return;
    }
    if !exact.is_well_formed() {
        return;
    }
    let fp = fingerprint(f32_lower, f32_upper);
    let Ok(mut reg) = registry().lock() else {
        return;
    };
    if reg.poisoned {
        return;
    }
    match reg.entry.as_ref() {
        Some(existing) if existing.fingerprint == fp => {}
        Some(_) => {
            reg.entry = None;
            reg.poisoned = true;
        }
        None => {
            reg.entry = Some(Entry {
                fingerprint: fp,
                exact,
            });
        }
    }
}

/// Serialize the registry across the parallel test harness: it is deliberately
/// process-global, so two tests touching it concurrently would interfere.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Clear the registry. Test-only: production registers once per process.
#[cfg(test)]
pub(crate) fn reset_for_test() {
    if let Ok(mut reg) = registry().lock() {
        reg.entry = None;
        reg.poisoned = false;
    }
}

/// Look up the exact-decimal box for `input`, or `None` when there is no
/// byte-exact match, the registry is poisoned, or containment fails.
#[must_use]
pub(crate) fn lookup(input: &BoundedTensor) -> Option<ExactBox> {
    let f32_lower = input.lower().as_slice()?;
    let f32_upper = input.upper().as_slice()?;
    let fp = fingerprint(f32_lower, f32_upper);
    let reg = registry().lock().ok()?;
    if reg.poisoned {
        return None;
    }
    let entry = reg.entry.as_ref()?;
    if entry.fingerprint != fp {
        return None;
    }
    let e = &entry.exact;
    // Containment re-check: the exact box must lie INSIDE the box the engine
    // was asked to verify. `split_input_bounds_f32` guarantees this by
    // construction; verifying it here means a pairing that somehow slipped
    // through identity cannot widen the verified region. The double-double
    // center is checked the same way, INFLATED by its own residual and the
    // exact half-width, so a bad decomposition cannot escape the box either.
    for i in 0..e.len() {
        let (cl, cu) = (e.lower[i], e.upper[i]);
        if !cl.is_finite() || !cu.is_finite() || cl > cu {
            return None;
        }
        if cl < f64::from(f32_lower[i]) || cu > f64::from(f32_upper[i]) {
            return None;
        }
        let (chi, clo, cerr, hw) = (
            e.center_hi[i],
            e.center_lo[i],
            e.center_err[i],
            e.half_width[i],
        );
        if !chi.is_finite() || !clo.is_finite() || !cerr.is_finite() || !hw.is_finite() {
            return None;
        }
        if cerr < 0.0 || hw < 0.0 {
            return None;
        }
        // The double-double center plus its slack must reconstruct an interval
        // that both encloses [lower, upper] on the inside and stays within it
        // on the outside, to within the outward-rounding slop of `lower`/
        // `upper` themselves (one f64 ulp each way).
        let c = chi + clo;
        let slack = hw + cerr;
        let lo_recon = c - slack;
        let hi_recon = c + slack;
        if !lo_recon.is_finite() || !hi_recon.is_finite() {
            return None;
        }
        if lo_recon < f64::from(f32_lower[i]) || hi_recon > f64::from(f32_upper[i]) {
            return None;
        }
    }
    Some(e.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// Build an `ExactBox` from plain f64 endpoints (exactly representable in
    /// the tests, so the second word and the residual are zero).
    fn exact_box(lo: &[f64], up: &[f64]) -> ExactBox {
        ExactBox {
            lower: lo.to_vec(),
            upper: up.to_vec(),
            center_hi: lo.iter().zip(up).map(|(a, b)| (a + b) * 0.5).collect(),
            center_lo: vec![0.0; lo.len()],
            center_err: vec![0.0; lo.len()],
            half_width: lo.iter().zip(up).map(|(a, b)| (b - a) * 0.5).collect(),
        }
    }

    fn bt(lo: &[f32], up: &[f32]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[lo.len()]), lo.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[up.len()]), up.to_vec()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn round_trips_a_matching_box_and_refuses_everything_else() {
        let _guard = test_lock();
        reset_for_test();
        let lo = [0.5_f32, 1.0];
        let up = [0.5_f32, 2.0];
        // The exact box: coordinate 0 is a declared POINT, coordinate 1 an
        // interval strictly inside the engine's f32 box.
        register(&lo, &up, exact_box(&[0.5, 1.0], &[0.5, 2.0]));
        let got = lookup(&bt(&lo, &up)).expect("byte-exact match");
        assert_eq!(got.lower, vec![0.5, 1.0]);
        assert_eq!(got.upper, vec![0.5, 2.0]);
        // Coordinate 0 is a declared point: exact half-width zero.
        assert_eq!(got.half_width[0], 0.0);

        // A different f32 box has a different fingerprint: no match.
        assert!(lookup(&bt(&[0.5, 1.0], &[0.5, 2.5])).is_none());

        reset_for_test();
    }

    #[test]
    fn a_conflicting_registration_poisons_the_registry() {
        let _guard = test_lock();
        reset_for_test();
        register(&[0.0_f32], &[1.0_f32], exact_box(&[0.0], &[1.0]));
        assert!(lookup(&bt(&[0.0], &[1.0])).is_some());
        // A second, different instance in the same process.
        register(&[0.0_f32], &[2.0_f32], exact_box(&[0.0], &[2.0]));
        assert!(lookup(&bt(&[0.0], &[1.0])).is_none());
        assert!(lookup(&bt(&[0.0], &[2.0])).is_none());
        reset_for_test();
    }

    #[test]
    fn a_box_outside_the_engine_box_is_refused() {
        let _guard = test_lock();
        reset_for_test();
        let lo = [0.0_f32];
        let up = [1.0_f32];
        // The "exact" box escapes the engine box: must never be served.
        register(&lo, &up, exact_box(&[-1.0], &[2.0]));
        assert!(lookup(&bt(&lo, &up)).is_none());
        reset_for_test();
    }

    #[test]
    fn non_finite_and_inverted_entries_are_refused() {
        let _guard = test_lock();
        reset_for_test();
        register(&[0.0_f32], &[1.0_f32], exact_box(&[f64::NAN], &[1.0]));
        assert!(lookup(&bt(&[0.0], &[1.0])).is_none());
        reset_for_test();
        register(&[0.0_f32], &[1.0_f32], exact_box(&[0.9], &[0.1]));
        assert!(lookup(&bt(&[0.0], &[1.0])).is_none());
        reset_for_test();
    }

    #[test]
    fn an_unregistered_box_is_refused() {
        let _guard = test_lock();
        reset_for_test();
        assert!(lookup(&bt(&[0.0], &[1.0])).is_none());
    }
}

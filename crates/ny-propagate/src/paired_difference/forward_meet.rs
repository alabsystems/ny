// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Intersections with interval images and the exact relation f = g + delta.
//!
//! These helpers only SHRINK already certified boxes using independently
//! certified enclosures. Forward affine forms remain valid when their concrete
//! boxes improve; no coefficient or plane is reconstructed from a met bound.

use super::*;

fn valid_box(lo: &[f64], hi: &[f64]) -> bool {
    lo.len() == hi.len()
        && lo
            .iter()
            .zip(hi)
            .all(|(&l, &h)| l.is_finite() && h.is_finite() && l <= h)
}

fn valid_state(state: &Enclosure) -> bool {
    let width = state.f_lo.len();
    width == state.g_lo.len()
        && width == state.d_lo.len()
        && valid_box(&state.f_lo, &state.f_hi)
        && valid_box(&state.g_lo, &state.g_hi)
        && valid_box(&state.d_lo, &state.d_hi)
}

/// Exact error-free TwoSum residual for a - b. The stored coefficients are
/// finite and capped by the engine, and all intermediate arithmetic is checked
/// for overflow. The residual is charged with outward interval multiplication,
/// so it is never silently omitted when the rounded subtraction loses bits.
fn difference(a: f64, b: f64) -> Option<(f64, f64)> {
    let negative_b = -b;
    let sum = a + negative_b;
    let virtual_b = sum - a;
    let error = (a - (sum - virtual_b)) + (negative_b - virtual_b);
    [sum, virtual_b, error]
        .iter()
        .all(|v| v.is_finite())
        .then_some((sum, error))
}

fn term(c: f64, lo: f64, hi: f64) -> (f64, f64) {
    if c >= 0.0 {
        (mul_down(c, lo), mul_up(c, hi))
    } else {
        (mul_down(c, hi), mul_up(c, lo))
    }
}

fn add_term(acc_lo: &mut f64, acc_hi: &mut f64, c: f64, lo: f64, hi: f64) {
    let (l, h) = term(c, lo, hi);
    *acc_lo = add_down(*acc_lo, l);
    *acc_hi = add_up(*acc_hi, h);
}

/// Three deterministic outward interval-contractor sweeps. Each face follows
/// from one exact identity and other currently rigorous faces:
///
/// f ∈ [g_lo + d_lo, g_hi + d_hi],
/// g ∈ [f_lo - d_hi, f_hi - d_lo],
/// d ∈ [f_lo - g_hi, f_hi - g_lo].
///
/// Using newly tightened faces in subsequent updates is sound by induction.
/// The iteration cap affects tightness only, never soundness or coverage.
pub(super) fn consistency(target: &mut Enclosure) -> Option<()> {
    if !valid_state(target) {
        return None;
    }
    for _ in 0..3 {
        for j in 0..target.f_lo.len() {
            target.f_lo[j] = target.f_lo[j].max(add_down(target.g_lo[j], target.d_lo[j]));
            target.f_hi[j] = target.f_hi[j].min(add_up(target.g_hi[j], target.d_hi[j]));
            target.g_lo[j] = target.g_lo[j].max(add_down(target.f_lo[j], -target.d_hi[j]));
            target.g_hi[j] = target.g_hi[j].min(add_up(target.f_hi[j], -target.d_lo[j]));
            target.d_lo[j] = target.d_lo[j].max(add_down(target.f_lo[j], -target.g_hi[j]));
            target.d_hi[j] = target.d_hi[j].min(add_up(target.f_hi[j], -target.g_lo[j]));
        }
        if !valid_state(target) {
            return None;
        }
    }
    Some(())
}

/// Meet an affine stage's forward-form box with three independent interval
/// images. The difference uses the exact identity
///
/// Wf*f - Wg*g = (Wf - Wg)*f + Wg*delta.
///
/// A rounded difference coefficient is represented as its rounded central
/// value PLUS its exact TwoSum residual, and both terms are enclosed. This
/// retains small authored weight/bias differences without approximate affine
/// composition, and remains sound even for widely separated binary exponents.
pub(super) fn affine_meet(stage: &Stage, source: &Enclosure, target: &mut Enclosure) -> Option<()> {
    let Stage::Affine {
        rows,
        cols,
        wf,
        wg,
        bf,
        bg,
    } = stage
    else {
        return None;
    };
    if !valid_state(source)
        || !valid_state(target)
        || source.f_lo.len() != *cols
        || target.f_lo.len() != *rows
        || wf.len() != rows.checked_mul(*cols)?
        || wg.len() != wf.len()
        || bf.len() != *rows
        || bg.len() != *rows
        || wf
            .iter()
            .chain(wg)
            .chain(bf)
            .chain(bg)
            .any(|v| !v.is_finite())
    {
        return None;
    }
    for i in 0..*rows {
        let (mut fl, mut fh) = (bf[i], bf[i]);
        let (mut gl, mut gh) = (bg[i], bg[i]);
        let (db, db_error) = difference(bf[i], bg[i])?;
        let (mut dl, mut dh) = (add_down(db, db_error), add_up(db, db_error));
        for j in 0..*cols {
            let a = wf[i * cols + j];
            let b = wg[i * cols + j];
            add_term(&mut fl, &mut fh, a, source.f_lo[j], source.f_hi[j]);
            add_term(&mut gl, &mut gh, b, source.g_lo[j], source.g_hi[j]);
            let (dw, dw_error) = difference(a, b)?;
            add_term(&mut dl, &mut dh, b, source.d_lo[j], source.d_hi[j]);
            add_term(&mut dl, &mut dh, dw, source.f_lo[j], source.f_hi[j]);
            add_term(&mut dl, &mut dh, dw_error, source.f_lo[j], source.f_hi[j]);
        }
        if [fl, fh, gl, gh, dl, dh].iter().any(|v| !v.is_finite()) {
            return None;
        }
        target.f_lo[i] = target.f_lo[i].max(fl);
        target.f_hi[i] = target.f_hi[i].min(fh);
        target.g_lo[i] = target.g_lo[i].max(gl);
        target.g_hi[i] = target.g_hi[i].min(gh);
        target.d_lo[i] = target.d_lo[i].max(dl);
        target.d_hi[i] = target.d_hi[i].min(dh);
    }
    consistency(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(
        fl: Vec<f64>,
        fh: Vec<f64>,
        gl: Vec<f64>,
        gh: Vec<f64>,
        dl: Vec<f64>,
        dh: Vec<f64>,
    ) -> Enclosure {
        Enclosure {
            f_lo: fl,
            f_hi: fh,
            g_lo: gl,
            g_hi: gh,
            d_lo: dl,
            d_hi: dh,
            hulls: None,
        }
    }

    #[test]
    fn signed_consistency_recovers_tower_bounds_and_preserves_points() {
        let mut box_state = state(
            vec![-10.0],
            vec![10.0],
            vec![1.0],
            vec![2.0],
            vec![0.2],
            vec![0.3],
        );
        consistency(&mut box_state).unwrap();
        assert!(box_state.f_lo[0] > 1.19 && box_state.f_hi[0] < 2.31);
        for i in 0..101 {
            for j in 0..101 {
                let g = 1.0 + f64::from(i) / 100.0;
                let d = 0.25 + f64::from(j) / 4096.0;
                let f = g + d;
                assert!(box_state.f_lo[0] <= f && f <= box_state.f_hi[0]);
                assert!(box_state.g_lo[0] <= g && g <= box_state.g_hi[0]);
                assert!(box_state.d_lo[0] <= d && d <= box_state.d_hi[0]);
            }
        }
    }

    #[test]
    fn affine_meet_retains_small_difference_and_encloses_reachable_points() {
        let stage = Stage::Affine {
            rows: 1,
            cols: 2,
            wf: vec![2.0, -1.0],
            wg: vec![2.015625, -0.984375],
            bf: vec![0.125],
            bg: vec![0.140625],
        };
        let source = state(
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        );
        let mut target = state(
            vec![-100.0],
            vec![100.0],
            vec![-100.0],
            vec![100.0],
            vec![-100.0],
            vec![100.0],
        );
        affine_meet(&stage, &source, &mut target).unwrap();
        assert!(target.d_lo[0] > -0.064 && target.d_hi[0] < -0.030);
        // Dyadic coefficients and points make this reference evaluation EXACT
        // in f64, so the assertion does not confuse floating forward error
        // with the mathematical real-valued network being enclosed.
        for i in 0..65 {
            for j in 0..65 {
                let x = f64::from(i) / 64.0;
                let y = 1.0 + f64::from(j) / 64.0;
                let f = 2.0 * x - y + 0.125;
                let g = 2.015625 * x - 0.984375 * y + 0.140625;
                assert!(target.f_lo[0] <= f && f <= target.f_hi[0]);
                assert!(target.g_lo[0] <= g && g <= target.g_hi[0]);
                assert!(target.d_lo[0] <= f - g && f - g <= target.d_hi[0]);
            }
        }
    }

    #[test]
    fn difference_residual_is_retained_across_widely_separated_exponents() {
        let (central, residual) = difference(1.0, 2.0_f64.powi(-80)).unwrap();
        assert_eq!(central, 1.0);
        assert_eq!(residual, -2.0_f64.powi(-80));
        let lo = add_down(central, residual);
        let hi = add_up(central, residual);
        assert!(lo < 1.0 && hi >= 1.0);
    }
}

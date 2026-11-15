// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounded one-objective arithmetic primitive. This module does not establish
//! that a supplied cut belongs to the original graph/domain; the private
//! ny-propagate call-context certificate must establish that before dispatch.
//! No shadow observation is accepted, no beta phase constraint is used, and no
//! upper result is exported. Root-only scheduling/admission is separate.

use super::*;
use std::time::Instant;

fn check<E: GemmEngine + ?Sized>(eng: &E, deadline: Instant) -> Result<()> {
    eng.poll_crown_backward_deadline()?;
    if Instant::now() >= deadline {
        return Err(NyError::DeadlineExceeded(
            "certified lower cut expired".into(),
        ));
    }
    Ok(())
}

fn safe_slice<E: GemmEngine + ?Sized>(eng: &E, values: &[f32]) -> Result<()> {
    for chunk in values.chunks(BOUNDED_ROWS_VALIDATION_CHUNK) {
        eng.poll_crown_backward_deadline()?;
        if chunk.iter().any(|&v| !ny_core::is_crown_coeff_safe(v)) {
            return Err(NyError::SoundnessRefusal(
                "certified lower cut: unsafe source value/sentinel".into(),
            ));
        }
    }
    Ok(())
}

fn safe_layers<E: GemmEngine + ?Sized>(eng: &E, layers: &[GpuCrownLayer]) -> Result<()> {
    certified_weight_charge::validate_layers(layers)?;
    for layer in layers {
        eng.poll_crown_backward_deadline()?;
        match layer {
            GpuCrownLayer::Linear { weight, bias, .. } => {
                safe_slice(eng, weight)?;
                if let Some(bias) = bias {
                    safe_slice(eng, bias)?;
                }
            }
            GpuCrownLayer::Conv2d {
                weight_col,
                bias_expanded,
                ..
            } => {
                safe_slice(eng, weight_col)?;
                if let Some(bias) = bias_expanded {
                    safe_slice(eng, bias)?;
                }
            }
            GpuCrownLayer::Activation { .. } | GpuCrownLayer::ActivationReluDualAlpha { .. } => {
                let activation = ActivationView::from_layer(layer).ok_or_else(|| {
                    NyError::InternalError("certified lower cut: activation view mismatch".into())
                })?;
                for values in activation.tables {
                    safe_slice(eng, values)?;
                }
            }
            _ => {
                return Err(NyError::UnsupportedOp(
                    "certified lower cut: unsupported layer".into(),
                ))
            }
        }
    }
    Ok(())
}

fn safe_frontier<E: GemmEngine + ?Sized>(eng: &E, f: &Frontier) -> Result<()> {
    let centers = [&f.lower_a, &f.upper_a, &f.lb, &f.ub];
    let errors = [&f.lower_err, &f.upper_err, &f.lb_err, &f.ub_err];
    for values in centers {
        for chunk in values.chunks(BOUNDED_ROWS_VALIDATION_CHUNK) {
            eng.poll_crown_backward_deadline()?;
            if chunk.iter().any(|&x| !ny_core::is_crown_coeff_safe_f64(x)) {
                return Err(NyError::SoundnessRefusal(
                    "certified lower cut: unsafe completed frontier".into(),
                ));
            }
        }
    }
    for values in errors {
        for chunk in values.chunks(BOUNDED_ROWS_VALIDATION_CHUNK) {
            eng.poll_crown_backward_deadline()?;
            if chunk
                .iter()
                .any(|&x| !ny_core::is_crown_coeff_safe_f64(x) || x < 0.0)
            {
                return Err(NyError::SoundnessRefusal(
                    "certified lower cut: invalid completed error enclosure".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Production entry retains the existing throughput gate. It receives a real
/// one-row seed; no duplicated/padded rows bypass the old 2..=8 validator.
/// The caller must use the explicit deadline-bounded CUDA adapter.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentLowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    check(eng, deadline)?;
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::Single(carrier),
        deadline,
        true,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn checked_fold<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentLowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    // Tests isolate numeric correctness below the production throughput gate.
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::Single(carrier),
        deadline,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn checked_fold_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: LowerCutRef<'_>,
    deadline: Instant,
    enforce_size_gate: bool,
) -> Result<f32> {
    check(eng, deadline)?;
    ny_core::require_f64_interval_proof_environment()?;
    validate_deadline_bounded_rows_resnet_request_range(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        &[],
        &[],
        1,
        1,
    )?;
    if enforce_size_gate && resnet_max_macs(segments, 1) < MIN_RESIDENT_MACS {
        return Err(NyError::UnsupportedOp(
            "certified lower cut: below unchanged resident size gate".into(),
        ));
    }
    // This entry starts at an exact objective, never at a rounded suffix whose
    // source-error lanes have been discarded. Semantic identity is checked by
    // the private caller; matching lower/upper seed bytes are necessary here.
    if seed
        .lower_a
        .iter()
        .zip(seed.upper_a.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits())
        || seed
            .lower_b
            .iter()
            .zip(seed.upper_b.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(NyError::SoundnessRefusal(
            "certified lower cut: non-exact objective seed".into(),
        ));
    }
    for v in [
        &*seed.lower_a,
        &*seed.upper_a,
        &*seed.lower_b,
        &*seed.upper_b,
        input_lower,
        input_upper,
    ] {
        safe_slice(eng, v)?;
    }
    for segment in segments {
        match segment {
            GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => safe_layers(eng, l)?,
            GpuResnetSegment::ResidualProj(f, p) => {
                safe_layers(eng, f)?;
                safe_layers(eng, p)?;
            }
        }
    }
    let widths = resident_activation_widths(segments)?;
    let width = *widths.get(carrier.target_activation()).ok_or_else(|| {
        NyError::InvalidSpec("certified lower cut: absent target activation".into())
    })?;
    match carrier {
        LowerCutRef::Single(carrier) => {
            carrier.validate_for_call(widths.len(), width, 1, deadline)?;
            if !carrier.has_nonzero_multiplier() {
                return Err(NyError::UnsupportedOp(
                    "certified lower cut: zero proposal".into(),
                ));
            }
            for row in carrier.rows() {
                let bias = row.bias();
                for c in row
                    .pre()
                    .iter()
                    .chain(row.post())
                    .chain(std::iter::once(&bias))
                {
                    safe_slice(eng, &[c.value(), c.source_abs_error()])?;
                }
            }
        }
        LowerCutRef::TwoPair(carrier) => {
            carrier.validate_for_call(widths.len(), width, 1, deadline)?;
            if !carrier.has_nonzero_multiplier() {
                return Err(NyError::UnsupportedOp(
                    "certified two-pair lower cut: zero proposal".into(),
                ));
            }
            let bias = carrier.bias();
            for c in carrier
                .pre()
                .iter()
                .chain(carrier.post())
                .chain(std::iter::once(&bias))
            {
                safe_slice(eng, &[c.value(), c.source_abs_error()])?;
            }
        }
        LowerCutRef::Triplet(carrier) => {
            carrier.validate_for_call(widths.len(), width, 1, deadline)?;
            if !carrier.has_nonzero_multiplier() {
                return Err(NyError::UnsupportedOp(
                    "certified triplet lower cut: zero proposal".into(),
                ));
            }
            let bias = carrier.bias();
            for c in carrier
                .pre()
                .iter()
                .chain(carrier.post())
                .chain(std::iter::once(&bias))
            {
                safe_slice(eng, &[c.value(), c.source_abs_error()])?;
            }
        }
        LowerCutRef::LocalL1(carrier) => {
            carrier.validate_for_call(widths.len(), width, 1, deadline)?;
            let bias = carrier.bias();
            for c in carrier
                .pre()
                .iter()
                .chain(carrier.post())
                .chain(std::iter::once(&bias))
            {
                safe_slice(eng, &[c.value(), c.source_abs_error()])?;
            }
        }
        LowerCutRef::TwoLocalL1(carrier) => {
            carrier.validate_for_call(widths.len(), width, 1, deadline)?;
            let bias = carrier.bias();
            // All32 fixed coefficient channels plus the aggregate bias are checked,
            // including canonical inactive slots, before any full-fold mutation.
            for c in carrier
                .pre()
                .iter()
                .flatten()
                .chain(carrier.post().iter().flatten())
                .chain(std::iter::once(&bias))
            {
                safe_slice(eng, &[c.value(), c.source_abs_error()])?;
            }
        }
    }
    check(eng, deadline)?;
    let mut aux = Some(ActAux::checked_cut(carrier));
    let f = resnet_fold_f64_core(eng, segments, seed, &mut aux)?;
    check(eng, deadline)?;
    if !aux
        .as_ref()
        .is_some_and(|a| a.lower_cut_applied && a.cursor == widths.len())
    {
        return Err(NyError::SoundnessRefusal(
            "certified lower cut: incomplete target fold".into(),
        ));
    }
    // Unconditional and independent of NY_CUDA_CONCRETIZE_SENTINEL_GUARD.
    safe_frontier(eng, &f)?;
    let result = concretize_frontier(&f, 1, input_lower, input_upper);
    validate_deadline_bounded_rows_resnet_result(1, &result)?;
    if !ny_core::is_crown_coeff_safe(result.lower_bounds[0]) {
        return Err(NyError::SoundnessRefusal(
            "certified lower cut: unsafe final bound".into(),
        ));
    }
    check(eng, deadline)?;
    Ok(result.lower_bounds[0])
}

/// Separate fixed-one-row capability for a summed two-pair carrier. The
/// private caller authenticates both facets and their outward aggregate bias;
/// this arithmetic entry does not turn transport into source authority.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_two_pair<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentTwoPairLowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    check(eng, deadline)?;
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::TwoPair(carrier),
        deadline,
        true,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn checked_two_pair<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentTwoPairLowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    // Numeric controls only; production never bypasses the resident size gate.
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::TwoPair(carrier),
        deadline,
        false,
    )
}

/// Separate fixed-one-row capability for a three-neuron carrier. The
/// private caller authenticates the relation and its outward bias;
/// this arithmetic entry does not turn transport into source authority.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_triplet<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentTripletLowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    check(eng, deadline)?;
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::Triplet(carrier),
        deadline,
        true,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn checked_triplet<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentTripletLowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    // Numeric controls only; production never bypasses the resident size gate.
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::Triplet(carrier),
        deadline,
        false,
    )
}

/// Separate fixed-one-row capability for a local weighted-L1 carrier. The
/// private caller authenticates the relation and its outward bias;
/// this arithmetic entry does not turn transport into source authority.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_local_l1<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentLocalL1LowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    check(eng, deadline)?;
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::LocalL1(carrier),
        deadline,
        true,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn checked_local_l1<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentLocalL1LowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    // Numeric controls only; production never bypasses the resident size gate.
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::LocalL1(carrier),
        deadline,
        false,
    )
}

/// Checked two-local full fold only; no support4 entry or permission required.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_two_local_l1<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentTwoLocalL1LowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    check(eng, deadline)?;
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::TwoLocalL1(carrier),
        deadline,
        true,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn checked_two_local_l1<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    carrier: &ResidentTwoLocalL1LowerCutCarrier,
    deadline: Instant,
) -> Result<f32> {
    // Numeric controls only; production never bypasses the resident size gate.
    checked_fold_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        LowerCutRef::TwoLocalL1(carrier),
        deadline,
        false,
    )
}

/// Exact-four-row support surface; no carrier, beta or shadow transport.
/// Ordinary bounded-row and single-row permissions remain unchanged.
pub(crate) fn support4<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    deadline: Instant,
) -> Result<GpuCrownResult> {
    support4_impl(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        deadline,
        true,
    )
}

fn support4_impl<E: GemmEngine + ?Sized>(
    eng: &E,
    segments: &[GpuResnetSegment],
    seed: &GpuCrownSeed,
    input_lower: &[f32],
    input_upper: &[f32],
    deadline: Instant,
    size_gate: bool,
) -> Result<GpuCrownResult> {
    check(eng, deadline)?;
    ny_core::require_f64_interval_proof_environment()?;
    validate_deadline_bounded_rows_resnet_request_range(
        eng,
        segments,
        seed,
        input_lower,
        input_upper,
        &[],
        &[],
        4,
        4,
    )?;
    if size_gate && resnet_max_macs(segments, 4) < MIN_RESIDENT_MACS {
        return Err(NyError::UnsupportedOp(
            "checked support below resident size gate".into(),
        ));
    }
    if seed
        .lower_a
        .iter()
        .zip(seed.upper_a.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits())
        || seed
            .lower_b
            .iter()
            .zip(seed.upper_b.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(NyError::SoundnessRefusal(
            "support objective seed differs".into(),
        ));
    }
    for v in [
        &*seed.lower_a,
        &*seed.upper_a,
        &*seed.lower_b,
        &*seed.upper_b,
        input_lower,
        input_upper,
    ] {
        safe_slice(eng, v)?;
    }
    for segment in segments {
        match segment {
            GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => safe_layers(eng, l)?,
            GpuResnetSegment::ResidualProj(f, p) => {
                safe_layers(eng, f)?;
                safe_layers(eng, p)?;
            }
        }
    }
    check(eng, deadline)?;
    let f = resnet_fold_f64_core(eng, segments, seed, &mut None)?;
    safe_frontier(eng, &f)?;
    let result = concretize_frontier(&f, 4, input_lower, input_upper);
    validate_deadline_bounded_rows_resnet_result(4, &result)?;
    for v in [&*result.lower_bounds, &*result.upper_bounds] {
        safe_slice(eng, v)?;
    }
    check(eng, deadline)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::Duration;

    #[derive(Default)]
    struct Cpu {
        polls: AtomicUsize,
        fail: AtomicUsize,
        calls: AtomicUsize,
    }
    impl GemmEngine for Cpu {
        fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
            panic!("the primitive must retain its f64 error path")
        }
        fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut out = vec![0.; m * n];
            for i in 0..m {
                for j in 0..n {
                    for h in 0..k {
                        out[i * n + j] += a[i * k + h] * b[h * n + j];
                    }
                }
            }
            Ok(out)
        }
        fn poll_crown_backward_deadline(&self) -> Result<()> {
            let p = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
            if p == self.fail.load(Ordering::SeqCst) {
                Err(NyError::DeadlineExceeded("injected typed boundary".into()))
            } else {
                Ok(())
            }
        }
    }
    fn linear(w: &[f32], b: Option<&[f32]>, o: usize, i: usize) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from(w),
            bias: b.map(Arc::from),
            out_features: o,
            in_features: i,
            cert_err: Default::default(),
        }
    }
    fn fixture(
        error: f32,
    ) -> (
        Vec<GpuResnetSegment>,
        GpuCrownSeed,
        ResidentLowerCutCarrier,
        Instant,
    ) {
        let segments = vec![
            GpuResnetSegment::Chain(vec![linear(&[1.], Some(&[1.25]), 1, 1)]),
            GpuResnetSegment::ResidualProj(
                vec![
                    linear(&[-2., -2.], None, 1, 2),
                    GpuCrownLayer::Activation {
                        lower_slope: vec![0.; 2],
                        upper_slope: vec![0.5; 2],
                        lower_intercept: vec![0.; 2],
                        upper_intercept: vec![0.5; 2],
                        num_neurons: 2,
                    },
                    linear(&[0.5, 0.5, 0.5, -0.5], None, 2, 2),
                ],
                vec![linear(&[1., 0.], None, 1, 2)],
            ),
        ];
        let seed = GpuCrownSeed {
            lower_a: Arc::from([1.]),
            upper_a: Arc::from([1.]),
            lower_b: Arc::from([0.]),
            upper_b: Arc::from([0.]),
            current_dim: 1,
            num_specs: 1,
        };
        let end = Instant::now() + Duration::from_secs(30);
        let ch = |x, e| ResidentLowerCutChannel::try_new(x, e).unwrap();
        // Exact support of[-1,-1,2,2] over the diamond is1; this core fixture
        // supplies the resulting transport directly, not a production authority.
        let carrier = ResidentLowerCutCarrier::try_new(
            0,
            2,
            [0, 1],
            vec![ny_core::ResidentLowerCutRow::try_new(
                vec![1.],
                [ch(-1., 0.), ch(-1., 0.)],
                [ch(2., 0.), ch(2., 0.)],
                ch(-1., error),
            )
            .unwrap()],
            end,
        )
        .unwrap();
        (segments, seed, carrier, end)
    }
    #[test]
    fn lower_cut_strict_diamond_residual_improves_and_charges_source_error() {
        let (segments, seed, carrier, end) = fixture(0.);
        let eng = Cpu::default();
        let old = resnet_backward_f64_core(&eng, &segments, &seed, &[-1.; 2], &[1.; 2], &mut None)
            .unwrap();
        assert!(old.lower_bounds[0] <= -0.75 && old.lower_bounds[0] > -0.751);
        let v = checked_fold(&eng, &segments, &seed, &[-1.; 2], &[1.; 2], &carrier, end).unwrap();
        assert!(v > 0.249 && v <= 0.25);
        // Independent all-domain identity: |(u+v)/2|+|(u-v)/2|=max(|u|,|v|).
        // Its maximum on[-1,1]^2 is1, attained at(1,1): the true minimum is1/4.
        let (s, q, c, t) = fixture(0.125);
        let charged = checked_fold(&eng, &s, &q, &[-1.; 2], &[1.; 2], &c, t).unwrap();
        assert!(charged <= 0.125 && charged > 0.124);
        assert!(matches!(
            run(&eng, &segments, &seed, &[-1.; 2], &[1.; 2], &carrier, end),
            Err(NyError::UnsupportedOp(_))
        ));
    }
    #[test]
    fn lower_cut_refuses_sentinel_even_if_later_zero_could_hide_it() {
        let (mut s, q, c, t) = fixture(0.);
        if let GpuResnetSegment::Chain(l) = &mut s[0] {
            l[0] = linear(&[0.], Some(&[ny_core::CROWN_COEFF_MAX]), 1, 1);
        }
        let e = Cpu::default();
        assert!(checked_fold(&e, &s, &q, &[-1.; 2], &[1.; 2], &c, t).is_err());
        assert_eq!(e.calls.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn lower_cut_refuses_invalid_weight_error_wrong_shape_and_expired_transport() {
        let (mut s, q, c, t) = fixture(0.);
        if let GpuResnetSegment::Chain(l) = &mut s[0] {
            if let GpuCrownLayer::Linear { cert_err, .. } = &mut l[0] {
                cert_err.weight_rel_err = -f32::EPSILON;
            }
        }
        let e = Cpu::default();
        assert!(checked_fold(&e, &s, &q, &[-1.; 2], &[1.; 2], &c, t).is_err());
        assert_eq!(e.calls.load(Ordering::SeqCst), 0);
        let (s, mut q, c, t) = fixture(0.);
        q.num_specs = 2;
        assert!(checked_fold(&e, &s, &q, &[-1.; 2], &[1.; 2], &c, t).is_err());
        q.num_specs = 1;
        assert!(checked_fold(
            &e,
            &s,
            &q,
            &[-1.; 2],
            &[1.; 2],
            &c,
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test clock supports an expired deadline")
        )
        .is_err());
    }
    #[test]
    fn lower_cut_final_poll_failure_returns_no_scalar() {
        let (s, q, c, t) = fixture(0.);
        let e = Cpu::default();
        checked_fold(&e, &s, &q, &[-1.; 2], &[1.; 2], &c, t).unwrap();
        let polls = e.polls.load(Ordering::SeqCst);
        let failed = Cpu::default();
        failed.fail.store(polls, Ordering::SeqCst);
        assert!(matches!(
            checked_fold(&failed, &s, &q, &[-1.; 2], &[1.; 2], &c, t),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(failed.calls.load(Ordering::SeqCst) > 0);
    }
    fn support_fixture() -> (Vec<GpuResnetSegment>, GpuCrownSeed, Instant) {
        let s = vec![GpuResnetSegment::Chain(vec![linear(
            &[0.5, 0.5, 0.5, -0.5],
            None,
            2,
            2,
        )])];
        let a: Arc<[f32]> = Arc::from([1., 0., 0., 1., 1., 1., 1., -1.]);
        let b: Arc<[f32]> = Arc::from([0.; 4]);
        (
            s,
            GpuCrownSeed {
                lower_a: a.clone(),
                upper_a: a,
                lower_b: b.clone(),
                upper_b: b,
                num_specs: 4,
                current_dim: 2,
            },
            Instant::now() + Duration::from_secs(30),
        )
    }
    #[test]
    fn support4_contains_four_exact_diamond_objectives_and_preserves_size_gate() {
        let (s, q, t) = support_fixture();
        let e = Cpu::default();
        let v = support4_impl(&e, &s, &q, &[-1.; 2], &[1.; 2], t, false).unwrap();
        // All four exact ranges are [-1,1]: z1,z2,u=z1+z2,v=z1-z2.
        for i in 0..4 {
            assert!(v.lower_bounds[i] <= -1. && v.lower_bounds[i] > -1.001);
            assert!(v.upper_bounds[i] >= 1. && v.upper_bounds[i] < 1.001);
        }
        assert!(e.calls.load(Ordering::SeqCst) > 0);
        assert!(matches!(
            support4(&e, &s, &q, &[-1.; 2], &[1.; 2], t),
            Err(NyError::UnsupportedOp(_))
        ));
    }
    #[test]
    fn support4_rejects_unsafe_source_asymmetric_seed_and_final_cancellation() {
        let (s, q, t) = support_fixture();
        let e = Cpu::default();
        support4_impl(&e, &s, &q, &[-1.; 2], &[1.; 2], t, false).unwrap();
        let count = e.polls.load(Ordering::SeqCst);
        let failed = Cpu::default();
        failed.fail.store(count, Ordering::SeqCst);
        assert!(matches!(
            support4_impl(&failed, &s, &q, &[-1.; 2], &[1.; 2], t, false),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(failed.calls.load(Ordering::SeqCst) > 0);
        let (mut s, mut q, t) = support_fixture();
        q.upper_a = Arc::from([0.; 8]);
        let e = Cpu::default();
        assert!(support4_impl(&e, &s, &q, &[-1.; 2], &[1.; 2], t, false).is_err());
        assert_eq!(e.calls.load(Ordering::SeqCst), 0);
        q.upper_a = q.lower_a.clone();
        if let GpuResnetSegment::Chain(l) = &mut s[0] {
            l[0] = linear(&[0.; 4], Some(&[ny_core::CROWN_COEFF_MAX, 0.]), 2, 2);
        }
        assert!(support4_impl(&e, &s, &q, &[-1.; 2], &[1.; 2], t, false).is_err());
        assert_eq!(e.calls.load(Ordering::SeqCst), 0);
    }

    fn two_pair_fixture() -> (Vec<GpuResnetSegment>, GpuCrownSeed, Instant) {
        // Genuine pure chain on [-1,1]^4. The first four neurons are two
        // independent diamonds; the fifth is the stable x+u+4 carrier.
        let segments = vec![GpuResnetSegment::Chain(vec![
            linear(&[-2., -2., -2., -2., 1.], Some(&[-1.75]), 1, 5),
            GpuCrownLayer::Activation {
                lower_slope: vec![0., 0., 0., 0., 1.],
                upper_slope: vec![0.5, 0.5, 0.5, 0.5, 1.],
                lower_intercept: vec![0.; 5],
                upper_intercept: vec![0.5, 0.5, 0.5, 0.5, 0.],
                num_neurons: 5,
            },
            linear(
                &[
                    0.5, 0.5, 0., 0., 0.5, -0.5, 0., 0., 0., 0., 0.5, 0.5, 0., 0., 0.5, -0.5, 1.,
                    0., 1., 0.,
                ],
                Some(&[0., 0., 0., 0., 4.]),
                5,
                4,
            ),
        ])];
        let seed = GpuCrownSeed {
            lower_a: Arc::from([1.]),
            upper_a: Arc::from([1.]),
            lower_b: Arc::from([0.]),
            upper_b: Arc::from([0.]),
            current_dim: 1,
            num_specs: 1,
        };
        (segments, seed, Instant::now() + Duration::from_secs(30))
    }

    fn two_pair_carrier(
        end: Instant,
        pre_error: f32,
        post_error: f32,
        bias_error: f32,
    ) -> ResidentTwoPairLowerCutCarrier {
        ResidentTwoPairLowerCutCarrier::try_new(
            0,
            5,
            [[0, 1], [2, 3]],
            [1.; 2],
            [ResidentLowerCutChannel::try_new(-1., pre_error).unwrap(); 4],
            [ResidentLowerCutChannel::try_new(2., post_error).unwrap(); 4],
            ResidentLowerCutChannel::try_new(-2., bias_error).unwrap(),
            end,
        )
        .unwrap()
    }

    #[test]
    fn two_pair_strict_diamonds_beat_every_single_pair_half_full() {
        let (s, q, t) = two_pair_fixture();
        let e = Cpu::default();
        let old = resnet_backward_f64_core(&e, &s, &q, &[-1.; 4], &[1.; 4], &mut None).unwrap();
        assert!(old.lower_bounds[0] <= -1.75 && old.lower_bounds[0] > -1.751);
        for first in 0..4 {
            for second in first + 1..4 {
                // Same-diamond exact support is 1. Cross-diamond projection is
                // the full square, whose exact support is 2.
                let b = if first / 2 == second / 2 { 1. } else { 2. };
                for lambda in [0.5, 1.] {
                    let ch = |v| ResidentLowerCutChannel::try_new(v, 0.).unwrap();
                    let c = ResidentLowerCutCarrier::try_new(
                        0,
                        5,
                        [first, second],
                        vec![ny_core::ResidentLowerCutRow::try_new(
                            vec![lambda],
                            [ch(-lambda); 2],
                            [ch(2. * lambda); 2],
                            ch(-lambda * b),
                        )
                        .unwrap()],
                        t,
                    )
                    .unwrap();
                    let v = checked_fold(&e, &s, &q, &[-1.; 4], &[1.; 4], &c, t).unwrap();
                    let exact = -1.75 + lambda * (2. - b);
                    assert!(v <= exact && v > exact - 0.001);
                    assert!(v < 0.);
                }
            }
        }
        let c = two_pair_carrier(t, 0., 0., 0.);
        let v = checked_two_pair(&e, &s, &q, &[-1.; 4], &[1.; 4], &c, t).unwrap();
        // All-domain identity: F=9/4-max(|x|,|y|)-max(|u|,|v|).
        // Its true minimum is 1/4. Both exact facets cancel all four ReLUs.
        assert!(v > 0.249 && v <= 0.25);
        assert!(e.calls.load(Ordering::SeqCst) > 0);
        assert!(matches!(
            run_two_pair(&e, &s, &q, &[-1.; 4], &[1.; 4], &c, t),
            Err(NyError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn two_pair_charges_all_pre_post_and_aggregate_bias_errors() {
        let (s, q, t) = two_pair_fixture();
        let epsilon = 1. / 128.;
        let bias_error = 1. / 32.;
        let c = two_pair_carrier(t, epsilon, epsilon, bias_error);
        let v = checked_two_pair(&Cpu::default(), &s, &q, &[-1.; 4], &[1.; 4], &c, t).unwrap();
        // Every operation below is exact dyadic arithmetic. The aggregate
        // nominal Lagrangian is 1/4; independent signed channel perturbations
        // have pointwise minimum 1/4-eps*sum(|z|+ReLU(z))-bias_error.
        // This is concave, so its box minimum is attained at a vertex.
        let mut exact_min = f32::INFINITY;
        for mask in 0..16 {
            let x: [f32; 4] = std::array::from_fn(|i| if mask & (1 << i) == 0 { -1. } else { 1. });
            let z = [
                f32::midpoint(x[0], x[1]),
                (x[0] - x[1]) / 2.,
                f32::midpoint(x[2], x[3]),
                (x[2] - x[3]) / 2.,
            ];
            let magnitude: f32 = z.iter().map(|z| z.abs() + z.max(0.)).sum();
            exact_min = exact_min.min(0.25 - epsilon * magnitude - bias_error);
        }
        assert_eq!(exact_min, 0.1875);
        assert!(v <= exact_min && v > 0.15);
        // Bias must be added once, not once per pair.
        let only_bias = two_pair_carrier(t, 0., 0., 0.125);
        let b =
            checked_two_pair(&Cpu::default(), &s, &q, &[-1.; 4], &[1.; 4], &only_bias, t).unwrap();
        assert!(b <= 0.125 && b > 0.124);
        // Individually exercise every channel at an exact positive point.
        // z=h=(1/2,1/2,1/2,1/2), so one radius eps must subtract eps/2.
        // Unlike the all-box test, independent coordinate elimination has no
        // slack here that could hide a missing channel's error charge.
        let (mut point_layers, point_seed, _) = two_pair_fixture();
        if let GpuResnetSegment::Chain(l) = &mut point_layers[0] {
            l[1] = GpuCrownLayer::Activation {
                lower_slope: vec![1.; 5],
                upper_slope: vec![1.; 5],
                lower_intercept: vec![0.; 5],
                upper_intercept: vec![0.; 5],
                num_neurons: 5,
            };
        }
        let point = [1., 0., 1., 0.];
        for post in [false, true] {
            for position in 0..4 {
                let mut pre = [ResidentLowerCutChannel::try_new(-1., 0.).unwrap(); 4];
                let mut after = [ResidentLowerCutChannel::try_new(2., 0.).unwrap(); 4];
                if post {
                    after[position] = ResidentLowerCutChannel::try_new(2., epsilon).unwrap();
                } else {
                    pre[position] = ResidentLowerCutChannel::try_new(-1., epsilon).unwrap();
                }
                let c = ResidentTwoPairLowerCutCarrier::try_new(
                    0,
                    5,
                    [[0, 1], [2, 3]],
                    [1.; 2],
                    pre,
                    after,
                    ResidentLowerCutChannel::try_new(-2., 0.).unwrap(),
                    t,
                )
                .unwrap();
                let v = checked_two_pair(
                    &Cpu::default(),
                    &point_layers,
                    &point_seed,
                    &point,
                    &point,
                    &c,
                    t,
                )
                .unwrap();
                let exact = 0.25 - epsilon / 2.;
                // The inherited sign-safe slope-error rule uses |ls|+|us|=2
                // even for this stable cell, so post error may be twice as
                // wide. Missing that channel entirely would violate exact.
                let conservative = 0.25 - if post { epsilon } else { epsilon / 2. };
                assert!(v <= exact && v > conservative - 0.001);
            }
        }
    }

    #[test]
    fn two_pair_sparse_order_keeps_upper_frontier_unchanged() {
        assert!(size_of::<Option<LowerCutRef<'_>>>() <= 3 * size_of::<usize>());
        let (s, q, t) = two_pair_fixture();
        let ch = |v| ResidentLowerCutChannel::try_new(v, 0.).unwrap();
        let ordered = ResidentTwoPairLowerCutCarrier::try_new(
            0,
            5,
            [[4, 1], [3, 0]],
            [1., 1.],
            [ch(-1.), ch(-2.), ch(-3.), ch(-4.)],
            [ch(5.), ch(6.), ch(7.), ch(8.)],
            ch(-9.),
            t,
        )
        .unwrap();
        let arm = LowerCutRef::TwoPair(&ordered);
        let mut a = [0.; 5];
        let mut error = [0.; 5];
        arm.apply_coefficients(&mut a, &mut error, 1, 5, true)
            .unwrap();
        assert_eq!(a, [8., 6., 0., 7., 5.]);
        arm.apply_coefficients(&mut a, &mut error, 1, 5, false)
            .unwrap();
        assert_eq!(a, [4., 4., 0., 4., 4.]);
        let mut bias = [0.];
        let mut bias_error = [0.];
        arm.apply_bias(&mut bias, &mut bias_error).unwrap();
        assert_eq!(bias, [-9.]);
        let e = Cpu::default();
        let before = resnet_fold_f64_core(&e, &s, &q, &mut None).unwrap();
        let c = two_pair_carrier(t, 0., 0., 0.);
        let mut aux = Some(ActAux::checked_cut(LowerCutRef::TwoPair(&c)));
        let after = resnet_fold_f64_core(&e, &s, &q, &mut aux).unwrap();
        for (left, right) in [
            (&before.upper_a, &after.upper_a),
            (&before.upper_err, &after.upper_err),
            (&before.ub, &after.ub),
            (&before.ub_err, &after.ub_err),
        ] {
            assert_eq!(
                left.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                right.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        assert!(aux.unwrap().lower_cut_applied);
    }

    #[test]
    fn two_pair_refuses_bad_source_seed_target_zero_and_deadline_before_dispatch() {
        let (s, q, t) = two_pair_fixture();
        let c = two_pair_carrier(t, 0., 0., 0.);
        let e = Cpu::default();
        let mut wrong_seed = q.clone();
        wrong_seed.upper_a = Arc::from([0.]);
        assert!(checked_two_pair(&e, &s, &wrong_seed, &[-1.; 4], &[1.; 4], &c, t).is_err());
        wrong_seed = q.clone();
        wrong_seed.num_specs = 2;
        assert!(checked_two_pair(&e, &s, &wrong_seed, &[-1.; 4], &[1.; 4], &c, t).is_err());
        for (activation, width, multipliers) in
            [(1, 5, [1., 1.]), (0, 6, [1., 1.]), (0, 5, [0., 0.])]
        {
            let invalid = ResidentTwoPairLowerCutCarrier::try_new(
                activation,
                width,
                c.ordered_pairs(),
                multipliers,
                *c.pre(),
                *c.post(),
                c.bias(),
                t,
            )
            .unwrap();
            assert!(checked_two_pair(&e, &s, &q, &[-1.; 4], &[1.; 4], &invalid, t).is_err());
        }
        assert!(matches!(
            checked_two_pair(
                &e,
                &s,
                &q,
                &[-1.; 4],
                &[1.; 4],
                &c,
                t + Duration::from_secs(1)
            ),
            Err(NyError::SoundnessRefusal(_))
        ));
        assert!(matches!(
            checked_two_pair(
                &e,
                &s,
                &q,
                &[-1.; 4],
                &[1.; 4],
                &c,
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("test clock supports an expired deadline")
            ),
            Err(NyError::DeadlineExceeded(_))
        ));
        let (mut bad, _, _) = two_pair_fixture();
        if let GpuResnetSegment::Chain(l) = &mut bad[0] {
            if let GpuCrownLayer::Linear { cert_err, .. } = &mut l[0] {
                cert_err.bias_abs_err = -f32::EPSILON;
            }
        }
        assert!(checked_two_pair(&e, &bad, &q, &[-1.; 4], &[1.; 4], &c, t).is_err());
        let (mut bad, _, _) = two_pair_fixture();
        if let GpuResnetSegment::Chain(l) = &mut bad[0] {
            l[0] = linear(&[0.; 5], Some(&[ny_core::CROWN_COEFF_MAX]), 1, 5);
        }
        assert!(checked_two_pair(&e, &bad, &q, &[-1.; 4], &[1.; 4], &c, t).is_err());
        let unsafe_channel = ResidentTwoPairLowerCutCarrier::try_new(
            0,
            5,
            c.ordered_pairs(),
            [1.; 2],
            *c.pre(),
            *c.post(),
            ResidentLowerCutChannel::try_new(ny_core::CROWN_COEFF_MAX, 0.).unwrap(),
            t,
        )
        .unwrap();
        assert!(checked_two_pair(&e, &s, &q, &[-1.; 4], &[1.; 4], &unsafe_channel, t).is_err());
        assert_eq!(e.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn two_pair_every_typed_poll_and_final_barrier_refuse_without_scalar() {
        let (s, q, t) = two_pair_fixture();
        let c = two_pair_carrier(t, 0., 0., 0.);
        let e = Cpu::default();
        checked_two_pair(&e, &s, &q, &[-1.; 4], &[1.; 4], &c, t).unwrap();
        let count = e.polls.load(Ordering::SeqCst);
        assert!(count > 0 && e.calls.load(Ordering::SeqCst) > 0);
        for failure in 1..=count {
            let e = Cpu::default();
            e.fail.store(failure, Ordering::SeqCst);
            assert!(matches!(
                checked_two_pair(&e, &s, &q, &[-1.; 4], &[1.; 4], &c, t),
                Err(NyError::DeadlineExceeded(_))
            ));
            if failure == count {
                assert!(e.calls.load(Ordering::SeqCst) > 0);
            }
        }
    }

    #[test]
    fn two_pair_accepts_boundary_zero_and_fully_stable_targets() {
        for positive in [false, true] {
            let (mut s, q, t) = two_pair_fixture();
            if let GpuResnetSegment::Chain(l) = &mut s[0] {
                l[1] = GpuCrownLayer::Activation {
                    lower_slope: vec![if positive { 1. } else { 0. }; 5],
                    upper_slope: vec![if positive { 1. } else { 0. }; 5],
                    lower_intercept: vec![0.; 5],
                    upper_intercept: vec![0.; 5],
                    num_neurons: 5,
                };
                if let GpuCrownLayer::Activation {
                    lower_slope,
                    upper_slope,
                    ..
                } = &mut l[1]
                {
                    lower_slope[4] = 1.;
                    upper_slope[4] = 1.;
                }
            }
            let base = two_pair_carrier(t, 0., 0., 0.);
            let c = ResidentTwoPairLowerCutCarrier::try_new(
                0,
                5,
                base.ordered_pairs(),
                [1.; 2],
                *base.pre(),
                *base.post(),
                ResidentLowerCutChannel::try_new(if positive { -2. } else { 0. }, 0.).unwrap(),
                t,
            )
            .unwrap();
            let point = if positive { [1., 0., 1., 0.] } else { [0.; 4] };
            let exact = if positive { 0.25 } else { 2.25 };
            let v = checked_two_pair(&Cpu::default(), &s, &q, &point, &point, &c, t).unwrap();
            assert!(v <= exact && v > exact - 0.001);
        }
    }
}

#[cfg(test)]
#[path = "dual_alpha_tests.rs"]
mod dual_alpha_tests;

#[cfg(test)]
#[path = "triplet_tests.rs"]
mod triplet_tests;

#[cfg(test)]
#[path = "local_l1_tests.rs"]
mod local_l1_tests;

#[cfg(test)]
#[path = "two_local_l1_tests.rs"]
mod two_local_l1_tests;

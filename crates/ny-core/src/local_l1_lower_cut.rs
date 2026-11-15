// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fixed-capacity arithmetic transport for one local weighted-L1 lower cut.
//!
//! This is not proof authority. The private caller authenticates the local
//! certificate, original source/domain/bounds, objective, generation and clock.
//! Channels are already reduced centers with outward source-error radii. No
//! certificate, source identity, upper channel or support4 permission lives here.

use std::time::Instant;

use crate::{NyError, ResidentLowerCutChannel, Result};

/// Lower-only transport for 1 through 8 strictly ordered target neurons.
///
/// Active channel positions follow the corresponding indices. All remaining
/// indices and channel fields must be canonical positive zero. The single bias
/// is added once. Fixed private fields carry no proof/verdict token, and this
/// value deliberately does not implement Clone.
#[derive(Debug, PartialEq)]
pub struct ResidentLocalL1LowerCutCarrier {
    target_activation: usize,
    target_width: usize,
    active_len: usize,
    ordered_neurons: [usize; 8],
    multiplier: f32,
    pre: [ResidentLowerCutChannel; 8],
    post: [ResidentLowerCutChannel; 8],
    bias: ResidentLowerCutChannel,
    deadline: Instant,
}

impl ResidentLocalL1LowerCutCarrier {
    /// Validate declarations without deriving a certificate or reading the clock.
    ///
    /// The semantic caller must establish the IEEE proof environment BEFORE
    /// reducing or constructing these channel arguments: neither this constructor
    /// nor the backend can restore an earlier flushed error radius. Multiplier
    /// and padding checks use bits, preserving positive subnormal declarations.
    /// This new transport refuses both signed zeros as multipliers.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        target_activation: usize,
        target_width: usize,
        active_len: usize,
        ordered_neurons: [usize; 8],
        multiplier: f32,
        pre: [ResidentLowerCutChannel; 8],
        post: [ResidentLowerCutChannel; 8],
        bias: ResidentLowerCutChannel,
        deadline: Instant,
    ) -> Result<Self> {
        if target_width == 0 || !(1..=8).contains(&active_len) {
            return Err(NyError::InvalidSpec(
                "local L1 lower cut requires a nonzero width and 1 through 8 active neurons".into(),
            ));
        }
        for position in 0..active_len {
            let index = ordered_neurons[position];
            if index >= target_width || (position != 0 && index <= ordered_neurons[position - 1]) {
                return Err(NyError::InvalidSpec(
                    "local L1 lower-cut indices must be strictly increasing and inside the width"
                        .into(),
                ));
            }
        }
        for position in active_len..8 {
            if ordered_neurons[position] != 0
                || pre[position].value().to_bits() != 0
                || pre[position].source_abs_error().to_bits() != 0
                || post[position].value().to_bits() != 0
                || post[position].source_abs_error().to_bits() != 0
            {
                return Err(NyError::InvalidSpec(
                    "local L1 lower-cut inactive slots must be canonical positive zero".into(),
                ));
            }
        }
        let bits = multiplier.to_bits();
        if bits == 0 || bits >= 0x7f80_0000 {
            return Err(NyError::NumericalInstability(
                "local L1 lower-cut multiplier must be finite and strictly positive".into(),
            ));
        }
        Ok(Self {
            target_activation,
            target_width,
            active_len,
            ordered_neurons,
            multiplier,
            pre,
            post,
            bias,
            deadline,
        })
    }

    /// Bind the fixed shape and exact deadline to one live arithmetic call.
    /// The backend separately checks the source, seed, errors and IEEE contracts.
    pub fn validate_for_call(
        &self,
        activation_count: usize,
        activation_width: usize,
        expected_rows: usize,
        deadline: Instant,
    ) -> Result<()> {
        if self.target_activation >= activation_count {
            return Err(NyError::InvalidSpec(
                "local L1 lower-cut target is outside the resident activation count".into(),
            ));
        }
        if self.target_width != activation_width || expected_rows != 1 {
            return Err(NyError::InvalidSpec(
                "local L1 lower cut requires matching width and exactly one objective row".into(),
            ));
        }
        if self.deadline != deadline {
            return Err(NyError::SoundnessRefusal(
                "local L1 lower-cut deadline is not the exact call-local deadline".into(),
            ));
        }
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "local L1 lower-cut deadline expired before backend mutation".into(),
            ));
        }
        Ok(())
    }

    /// Exact resident activation ordinal.
    pub const fn target_activation(&self) -> usize {
        self.target_activation
    }
    /// Exact target ReLU width.
    pub const fn target_width(&self) -> usize {
        self.target_width
    }
    /// Number of active channel positions, in 1..=8.
    pub const fn active_len(&self) -> usize {
        self.active_len
    }
    /// Strict active order followed by canonical zero padding.
    pub const fn ordered_neurons(&self) -> &[usize; 8] {
        &self.ordered_neurons
    }
    /// Positive declaration; channel consistency remains the private caller's duty.
    pub const fn multiplier(&self) -> f32 {
        self.multiplier
    }
    /// Preactivation channels, applied after slope propagation.
    pub const fn pre(&self) -> &[ResidentLowerCutChannel; 8] {
        &self.pre
    }
    /// Postactivation channels, applied before ReLU sign selection.
    pub const fn post(&self) -> &[ResidentLowerCutChannel; 8] {
        &self.post
    }
    /// One bias center and outward error radius.
    pub const fn bias(&self) -> ResidentLowerCutChannel {
        self.bias
    }
    /// Mandatory original call deadline.
    pub const fn deadline(&self) -> Instant {
        self.deadline
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, GpuResnetSegment,
        ResidentTripletLowerCutCarrier,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::Duration;

    fn channel(value: f32, error: f32) -> ResidentLowerCutChannel {
        ResidentLowerCutChannel::try_new(value, error).unwrap()
    }
    fn candidate(
        active: usize,
        width: usize,
        multiplier: f32,
        deadline: Instant,
    ) -> Result<ResidentLocalL1LowerCutCarrier> {
        let mut indices = [0; 8];
        let mut pre = [channel(0., 0.); 8];
        let mut post = pre;
        for position in 0..active.min(8) {
            indices[position] = 2 * position;
            pre[position] = channel(-(position as f32 + 1.), 0.125);
            post[position] = channel(2. * (position as f32 + 1.), 0.25);
        }
        ResidentLocalL1LowerCutCarrier::try_new(
            1,
            width,
            active,
            indices,
            multiplier,
            pre,
            post,
            channel(-4., 0.5),
            deadline,
        )
    }

    #[test]
    fn local_l1_transport_preserves_each_active_length_and_all_channels() {
        let deadline = Instant::now() + Duration::from_secs(30);
        for active in 1..=8 {
            let c = candidate(active, 16, 0.5, deadline).unwrap();
            assert_eq!(c.active_len(), active);
            assert_eq!(c.target_activation(), 1);
            assert_eq!(c.target_width(), 16);
            assert_eq!(c.multiplier(), 0.5);
            assert_eq!(c.deadline(), deadline);
            assert_eq!(c.bias(), channel(-4., 0.5));
            for position in 0..8 {
                if position < active {
                    assert_eq!(c.ordered_neurons()[position], 2 * position);
                    assert_eq!(c.pre()[position], channel(-(position as f32 + 1.), 0.125));
                    assert_eq!(
                        c.post()[position],
                        channel(2. * (position as f32 + 1.), 0.25)
                    );
                } else {
                    assert_eq!(c.ordered_neurons()[position], 0);
                    assert_eq!(c.pre()[position], channel(0., 0.));
                    assert_eq!(c.post()[position], channel(0., 0.));
                }
            }
            assert!(c.validate_for_call(2, 16, 1, deadline).is_ok());
        }
        // Prospective owner accounting must include the concrete carrier, not
        // assume it is as small as the old three/four-channel transports.
        assert!(size_of::<ResidentLocalL1LowerCutCarrier>() <= 320);
    }

    #[test]
    fn local_l1_transport_rejects_count_order_alias_range_and_padding() {
        let t = Instant::now() + Duration::from_secs(30);
        for active in [0, 9, usize::MAX] {
            assert!(matches!(
                candidate(active, 16, 1., t),
                Err(NyError::InvalidSpec(_))
            ));
        }
        assert!(candidate(3, 0, 1., t).is_err());
        assert!(candidate(8, 14, 1., t).is_err());
        let good = candidate(3, 16, 1., t).unwrap();
        for position in 0..3 {
            for invalid in [16, usize::MAX] {
                let mut indices = *good.ordered_neurons();
                indices[position] = invalid;
                assert!(ResidentLocalL1LowerCutCarrier::try_new(
                    1,
                    16,
                    3,
                    indices,
                    1.,
                    *good.pre(),
                    *good.post(),
                    good.bias(),
                    t
                )
                .is_err());
            }
        }
        for indices in [[0, 0, 4, 0, 0, 0, 0, 0], [4, 2, 0, 0, 0, 0, 0, 0]] {
            assert!(ResidentLocalL1LowerCutCarrier::try_new(
                1,
                16,
                3,
                indices,
                1.,
                *good.pre(),
                *good.post(),
                good.bias(),
                t
            )
            .is_err());
        }
        for position in 3..8 {
            for slot in 0..5 {
                let mut indices = *good.ordered_neurons();
                let mut pre = *good.pre();
                let mut post = *good.post();
                match slot {
                    0 => indices[position] = 1,
                    1 => pre[position] = channel(f32::from_bits(1), 0.),
                    2 => pre[position] = channel(0., f32::from_bits(1)),
                    3 => post[position] = channel(-1., 0.),
                    _ => post[position] = channel(0., 0.125),
                }
                assert!(ResidentLocalL1LowerCutCarrier::try_new(
                    1,
                    16,
                    3,
                    indices,
                    1.,
                    pre,
                    post,
                    good.bias(),
                    t
                )
                .is_err());
            }
        }
    }

    #[test]
    fn local_l1_positive_multiplier_is_checked_by_bits() {
        let t = Instant::now() + Duration::from_secs(30);
        for bad in [
            0.,
            -0.,
            -1.,
            f32::from_bits(0x8000_0001),
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert!(matches!(
                candidate(3, 16, bad, t),
                Err(NyError::NumericalInstability(_))
            ));
        }
        let tiny = candidate(3, 16, f32::from_bits(1), t).unwrap();
        assert_eq!(tiny.multiplier().to_bits(), 1);
    }

    #[test]
    fn local_l1_call_requires_exact_width_activation_and_one_row() {
        let t = Instant::now() + Duration::from_secs(30);
        let c = candidate(3, 16, 1., t).unwrap();
        for (count, width, rows) in [
            (0, 16, 1),
            (1, 16, 1),
            (2, 15, 1),
            (2, 17, 1),
            (2, 16, 0),
            (2, 16, 2),
            (2, 16, usize::MAX),
        ] {
            assert!(matches!(
                c.validate_for_call(count, width, rows, t),
                Err(NyError::InvalidSpec(_))
            ));
        }
    }

    #[test]
    fn local_l1_expired_deadline_cannot_be_replaced() {
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("test instant supports a one-second lookback");
        let c = candidate(3, 16, 1., expired).unwrap();
        assert!(matches!(
            c.validate_for_call(2, 16, 1, expired),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(
            c.validate_for_call(2, 16, 1, expired + Duration::from_secs(30)),
            Err(NyError::SoundnessRefusal(_))
        ));
    }

    #[derive(Default)]
    struct OldCapabilities {
        calls: AtomicUsize,
    }
    impl GpuCrownBackward for OldCapabilities {
        fn crown_backward_gpu(
            &self,
            _: &[GpuCrownLayer],
            _: &[f32],
            _: usize,
            _: &[f32],
            _: &[f32],
        ) -> Result<GpuCrownResult> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(GpuCrownResult::default())
        }
        fn provides_bounded_lower_cut(&self) -> bool {
            true
        }
        fn provides_bounded_two_pair_lower_cut(&self) -> bool {
            true
        }
        fn provides_bounded_triplet_lower_cut(&self) -> bool {
            true
        }
        fn provides_bounded_support4(&self) -> bool {
            true
        }
        fn crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
            &self,
            _: &[GpuResnetSegment],
            _: &GpuCrownSeed,
            _: &[f32],
            _: &[f32],
            _: &ResidentTripletLowerCutCarrier,
            _: Instant,
        ) -> Result<f32> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(123.)
        }
        fn crown_backward_gpu_resnet_support4_with_deadline(
            &self,
            _: &[GpuResnetSegment],
            _: &GpuCrownSeed,
            _: &[f32],
            _: &[f32],
            _: Instant,
        ) -> Result<GpuCrownResult> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(GpuCrownResult::default())
        }
    }

    #[test]
    fn local_l1_default_refuses_without_support_or_old_cut_delegation() {
        let engine = OldCapabilities::default();
        let backend: &dyn GpuCrownBackward = &engine;
        assert!(backend.provides_bounded_triplet_lower_cut());
        assert!(backend.provides_bounded_support4());
        assert!(!backend.provides_bounded_local_l1_lower_cut());
        let t = Instant::now() + Duration::from_secs(30);
        let c = candidate(3, 16, 1., t).unwrap();
        let q = GpuCrownSeed {
            lower_a: Arc::from([1.]),
            upper_a: Arc::from([1.]),
            lower_b: Arc::from([0.]),
            upper_b: Arc::from([0.]),
            num_specs: 1,
            current_dim: 1,
        };
        assert!(matches!(
            backend.crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
                &[],
                &q,
                &[-1.],
                &[1.],
                &c,
                t
            ),
            Err(NyError::UnsupportedOp(_))
        ));
        assert_eq!(engine.calls.load(Ordering::Relaxed), 0);
    }
}

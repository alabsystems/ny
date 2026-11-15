// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fixed one-row arithmetic transport for one three-neuron ReLU relation.
//!
//! This is not proof authority. The private caller authenticates the exact
//! relation, original source/domain/bounds, objective, generation and deadline.
//! Channels are already reduced centers with outward source-error charges.
//! No facet weights, certificate, upper channel or source identity live here.
//! There is no conversion from a single-pair or two-pair carrier.

use std::time::Instant;

use crate::{NyError, ResidentLowerCutChannel, Result};

/// Lower-only transport for exactly three distinct ordered target neurons.
///
/// The three pre/post channel positions follow `ordered_neurons` literally.
/// The one bias channel is added once. Private fixed-size fields carry no
/// proof or verdict token; this value deliberately does not implement Clone.
#[derive(Debug, PartialEq)]
pub struct ResidentTripletLowerCutCarrier {
    target_activation: usize,
    target_width: usize,
    ordered_neurons: [usize; 3],
    multiplier: f32,
    pre: [ResidentLowerCutChannel; 3],
    post: [ResidentLowerCutChannel; 3],
    bias: ResidentLowerCutChannel,
    deadline: Instant,
}

impl ResidentTripletLowerCutCarrier {
    /// Validate declarations, without deriving a relation or reading the clock.
    ///
    /// The semantic caller must establish the IEEE proof environment BEFORE
    /// reducing or constructing these channel arguments: this constructor
    /// cannot restore a source-error radius flushed earlier. The multiplier
    /// declaration uses bits, preserving positive subnormals and refusing
    /// negative nonzero values under DAZ. Signed zero is normalized; zero is
    /// representable but the backend must refuse it before arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        target_activation: usize,
        target_width: usize,
        ordered_neurons: [usize; 3],
        mut multiplier: f32,
        pre: [ResidentLowerCutChannel; 3],
        post: [ResidentLowerCutChannel; 3],
        bias: ResidentLowerCutChannel,
        deadline: Instant,
    ) -> Result<Self> {
        if target_width == 0 {
            return Err(NyError::InvalidSpec(
                "triplet lower-cut target width must be nonzero".into(),
            ));
        }
        for (position, &index) in ordered_neurons.iter().enumerate() {
            if index >= target_width || ordered_neurons[..position].contains(&index) {
                return Err(NyError::InvalidSpec(
                    "triplet lower-cut indices must be distinct and inside the target width".into(),
                ));
            }
        }
        let bits = multiplier.to_bits();
        let magnitude = bits & 0x7fff_ffff;
        if magnitude >= 0x7f80_0000 || (bits >> 31 != 0 && magnitude != 0) {
            return Err(NyError::NumericalInstability(
                "triplet lower-cut multiplier must be finite and nonnegative".into(),
            ));
        }
        if magnitude == 0 {
            multiplier = 0.0;
        }
        Ok(Self {
            target_activation,
            target_width,
            ordered_neurons,
            multiplier,
            pre,
            post,
            bias,
            deadline,
        })
    }

    /// Bind the fixed shape and exact deadline to one live arithmetic call.
    ///
    /// The backend separately checks the multiplier, full source, seed, error
    /// and IEEE contracts. Shape agreement alone does not authorize a proof.
    pub fn validate_for_call(
        &self,
        activation_count: usize,
        activation_width: usize,
        expected_rows: usize,
        deadline: Instant,
    ) -> Result<()> {
        if self.target_activation >= activation_count {
            return Err(NyError::InvalidSpec(
                "triplet lower-cut target is outside the resident activation count".into(),
            ));
        }
        if self.target_width != activation_width || expected_rows != 1 {
            return Err(NyError::InvalidSpec(
                "triplet lower-cut requires matching width and exactly one objective row".into(),
            ));
        }
        if self.deadline != deadline {
            return Err(NyError::SoundnessRefusal(
                "triplet lower-cut deadline is not the exact call-local deadline".into(),
            ));
        }
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "triplet lower-cut deadline expired before backend mutation".into(),
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
    /// Literal neuron/channel order; never sorted by this transport.
    pub const fn ordered_neurons(&self) -> [usize; 3] {
        self.ordered_neurons
    }
    /// Declared nonnegative multiplier; semantic channel consistency is private.
    pub const fn multiplier(&self) -> f32 {
        self.multiplier
    }
    /// Three preactivation channels, applied after slope propagation.
    pub const fn pre(&self) -> &[ResidentLowerCutChannel; 3] {
        &self.pre
    }
    /// Three postactivation channels, applied before ReLU sign selection.
    pub const fn post(&self) -> &[ResidentLowerCutChannel; 3] {
        &self.post
    }
    /// One bias center and its outward source-error radius.
    pub const fn bias(&self) -> ResidentLowerCutChannel {
        self.bias
    }
    /// Mandatory original call deadline.
    pub const fn deadline(&self) -> Instant {
        self.deadline
    }
    /// Whether the validated declaration has a nonzero magnitude.
    pub fn has_nonzero_multiplier(&self) -> bool {
        self.multiplier.to_bits() != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, GpuResnetSegment,
        ResidentLowerCutCarrier, ResidentTwoPairLowerCutCarrier,
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
        width: usize,
        indices: [usize; 3],
        multiplier: f32,
        deadline: Instant,
    ) -> Result<ResidentTripletLowerCutCarrier> {
        ResidentTripletLowerCutCarrier::try_new(
            1,
            width,
            indices,
            multiplier,
            [channel(-1., 0.); 3],
            [channel(2., 0.); 3],
            channel(-2., 0.),
            deadline,
        )
    }

    #[test]
    fn triplet_transport_keeps_order_and_every_channel_error() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let pre = [channel(-1., 0.125), channel(-2., 0.25), channel(-3., 0.5)];
        let post = [channel(2., 0.0625), channel(4., 0.125), channel(6., 0.25)];
        let bias = channel(-3., 0.03125);
        let c = ResidentTripletLowerCutCarrier::try_new(
            1,
            9,
            [8, 2, 6],
            0.5,
            pre,
            post,
            bias,
            deadline,
        )
        .unwrap();
        assert_eq!(c.ordered_neurons(), [8, 2, 6]);
        assert_eq!(*c.pre(), pre);
        assert_eq!(*c.post(), post);
        assert_eq!(c.bias(), bias);
        assert_eq!(c.multiplier(), 0.5);
        assert_eq!(c.target_activation(), 1);
        assert_eq!(c.target_width(), 9);
        assert_eq!(c.deadline(), deadline);
        assert!(c.validate_for_call(2, 9, 1, deadline).is_ok());
    }

    #[test]
    fn triplet_transport_refuses_every_alias_and_out_of_range_index() {
        let deadline = Instant::now() + Duration::from_secs(30);
        assert!(matches!(
            candidate(0, [0, 1, 2], 1., deadline),
            Err(NyError::InvalidSpec(_))
        ));
        for first in 0..3 {
            for second in first + 1..3 {
                let mut indices = [0, 1, 2];
                indices[second] = indices[first];
                assert!(matches!(
                    candidate(3, indices, 1., deadline),
                    Err(NyError::InvalidSpec(_))
                ));
            }
            for invalid in [3, usize::MAX] {
                let mut indices = [0, 1, 2];
                indices[first] = invalid;
                assert!(matches!(
                    candidate(3, indices, 1., deadline),
                    Err(NyError::InvalidSpec(_))
                ));
            }
        }
    }

    #[test]
    fn triplet_multiplier_bits_preserve_subnormal_and_refuse_invalid_values() {
        let deadline = Instant::now() + Duration::from_secs(30);
        for invalid in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1.,
            f32::from_bits(0x8000_0001),
        ] {
            assert!(matches!(
                candidate(3, [0, 1, 2], invalid, deadline),
                Err(NyError::NumericalInstability(_))
            ));
        }
        let zero = candidate(3, [0, 1, 2], -0., deadline).unwrap();
        assert_eq!(zero.multiplier().to_bits(), 0);
        assert!(!zero.has_nonzero_multiplier());
        assert!(zero.validate_for_call(2, 3, 1, deadline).is_ok());
        let tiny = candidate(3, [0, 1, 2], f32::from_bits(1), deadline).unwrap();
        assert_eq!(tiny.multiplier().to_bits(), 1);
        assert!(tiny.has_nonzero_multiplier());
    }

    #[test]
    fn triplet_call_requires_exact_activation_width_one_row_and_deadline() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let c = candidate(3, [0, 1, 2], 1., deadline).unwrap();
        for (count, width, rows) in [
            (0, 3, 1),
            (1, 3, 1),
            (2, 2, 1),
            (2, 4, 1),
            (2, 3, 0),
            (2, 3, 2),
            (2, 3, usize::MAX),
        ] {
            assert!(matches!(
                c.validate_for_call(count, width, rows, deadline),
                Err(NyError::InvalidSpec(_))
            ));
        }
        assert!(matches!(
            c.validate_for_call(2, 3, 1, deadline + Duration::from_nanos(1)),
            Err(NyError::SoundnessRefusal(_))
        ));
    }

    #[test]
    fn triplet_expiry_cannot_be_replaced_with_a_fresh_deadline() {
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("test instant supports a one-second lookback");
        let c = candidate(3, [0, 1, 2], 1., expired).unwrap();
        assert!(matches!(
            c.validate_for_call(2, 3, 1, expired),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(
            c.validate_for_call(2, 3, 1, Instant::now() + Duration::from_secs(30)),
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
        fn crown_backward_gpu_resnet_lower_cut_with_deadline(
            &self,
            _: &[GpuResnetSegment],
            _: &GpuCrownSeed,
            _: &[f32],
            _: &[f32],
            _: &ResidentLowerCutCarrier,
            _: Instant,
        ) -> Result<f32> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(123.)
        }
        fn crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
            &self,
            _: &[GpuResnetSegment],
            _: &GpuCrownSeed,
            _: &[f32],
            _: &[f32],
            _: &ResidentTwoPairLowerCutCarrier,
            _: Instant,
        ) -> Result<f32> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(123.)
        }
    }

    #[test]
    fn triplet_default_refuses_without_delegating_to_any_old_capability() {
        let engine = OldCapabilities::default();
        let backend: &dyn GpuCrownBackward = &engine;
        assert!(backend.provides_bounded_lower_cut());
        assert!(backend.provides_bounded_two_pair_lower_cut());
        assert!(!backend.provides_bounded_triplet_lower_cut());
        let deadline = Instant::now() + Duration::from_secs(30);
        let c = candidate(3, [0, 1, 2], 1., deadline).unwrap();
        let q = GpuCrownSeed {
            lower_a: Arc::from([1.]),
            upper_a: Arc::from([1.]),
            lower_b: Arc::from([0.]),
            upper_b: Arc::from([0.]),
            num_specs: 1,
            current_dim: 1,
        };
        assert!(matches!(
            backend.crown_backward_gpu_resnet_triplet_lower_cut_with_deadline(
                &[],
                &q,
                &[-1.],
                &[1.],
                &c,
                deadline
            ),
            Err(NyError::UnsupportedOp(_))
        ));
        assert_eq!(engine.calls.load(Ordering::Relaxed), 0);
    }
}

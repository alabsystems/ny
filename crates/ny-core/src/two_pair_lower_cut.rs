// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fixed one-row arithmetic transport for two disjoint ReLU pairs.
//!
//! This is not proof authority. A private caller must authenticate both exact
//! facets, their common original source/domain/bound map, objective, generations
//! and call deadline before constructing and synchronously consuming a carrier.
//! The channels are already reduced centers with outward source-error charges;
//! this module neither derives those channels nor certifies their relationship
//! to the multipliers. Backend arithmetic and semantic proof replay are separate
//! requirements. There is no conversion from the existing single-pair carrier.

use std::time::Instant;

use crate::{NyError, ResidentLowerCutChannel, Result};

/// Lower-only, one-objective transport for exactly two disjoint ordered pairs.
///
/// Channel slots are `pair0.first`, `pair0.second`, `pair1.first`, `pair1.second`.
/// The pair and channel order is retained literally, never sorted here. The
/// single bias channel contains the outward reduction of BOTH facet biases.
/// All fields are private and fixed-size. No upper channel, proof certificate,
/// cached identity, or verdict is represented. This value is not `Clone`.
#[derive(Debug, PartialEq)]
pub struct ResidentTwoPairLowerCutCarrier {
    target_activation: usize,
    target_width: usize,
    ordered_pairs: [[usize; 2]; 2],
    multipliers: [f32; 2],
    pre: [ResidentLowerCutChannel; 4],
    post: [ResidentLowerCutChannel; 4],
    bias: ResidentLowerCutChannel,
    deadline: Instant,
}

impl ResidentTwoPairLowerCutCarrier {
    /// Validate finite arithmetic declarations and four distinct target indices.
    ///
    /// Channel constructors already require finite centers and finite,
    /// nonnegative source errors. The private caller must establish the IEEE
    /// proof environment BEFORE reducing or constructing these channel
    /// arguments; this constructor cannot repair earlier flushed error charges.
    /// Multipliers must be finite and nonnegative;
    /// signed zero is normalized. Zero multipliers are representable, but an
    /// executing backend must separately require `has_nonzero_multiplier()`.
    /// A private two-facet proof builder may impose the stronger both-positive
    /// requirement. Construction does not validate a proof or read the clock.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        target_activation: usize,
        target_width: usize,
        ordered_pairs: [[usize; 2]; 2],
        mut multipliers: [f32; 2],
        pre: [ResidentLowerCutChannel; 4],
        post: [ResidentLowerCutChannel; 4],
        bias: ResidentLowerCutChannel,
        deadline: Instant,
    ) -> Result<Self> {
        if target_width == 0 {
            return Err(NyError::InvalidSpec(
                "two-pair lower-cut target width must be nonzero".into(),
            ));
        }
        let indices = [
            ordered_pairs[0][0],
            ordered_pairs[0][1],
            ordered_pairs[1][0],
            ordered_pairs[1][1],
        ];
        for (position, &index) in indices.iter().enumerate() {
            if index >= target_width || indices[..position].contains(&index) {
                return Err(NyError::InvalidSpec(
                    "two-pair lower-cut indices must be distinct and inside the target width"
                        .into(),
                ));
            }
        }
        for multiplier in &mut multipliers {
            // Declaration validation uses bits, so a negative subnormal cannot
            // become an accepted zero under a caller's DAZ setting. The actual
            // arithmetic backend must still enforce its IEEE environment guard.
            let bits = multiplier.to_bits();
            let magnitude = bits & 0x7fff_ffff;
            if magnitude >= 0x7f80_0000 || (bits >> 31 != 0 && magnitude != 0) {
                return Err(NyError::NumericalInstability(
                    "two-pair lower-cut multipliers must be finite and nonnegative".into(),
                ));
            }
            if magnitude == 0 {
                *multiplier = 0.0;
            }
        }
        Ok(Self {
            target_activation,
            target_width,
            ordered_pairs,
            multipliers,
            pre,
            post,
            bias,
            deadline,
        })
    }

    /// Bind the arithmetic shape to one live backend call, not a semantic proof.
    ///
    /// The backend must also require at least one nonzero multiplier and check
    /// its full source, seed, error, IEEE and capability contracts. An expired
    /// deadline is refused before mutation; no fresh deadline can substitute.
    pub fn validate_for_call(
        &self,
        activation_count: usize,
        activation_width: usize,
        expected_rows: usize,
        deadline: Instant,
    ) -> Result<()> {
        if self.target_activation >= activation_count {
            return Err(NyError::InvalidSpec(
                "two-pair lower-cut target is outside the resident activation count".into(),
            ));
        }
        if self.target_width != activation_width {
            return Err(NyError::InvalidSpec(
                "two-pair lower-cut width does not match the resident target".into(),
            ));
        }
        if expected_rows != 1 {
            return Err(NyError::InvalidSpec(
                "two-pair lower-cut transport requires exactly one objective row".into(),
            ));
        }
        if self.deadline != deadline {
            return Err(NyError::SoundnessRefusal(
                "two-pair lower-cut deadline is not the exact call-local deadline".into(),
            ));
        }
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "two-pair lower-cut deadline expired before backend mutation".into(),
            ));
        }
        Ok(())
    }

    /// Target activation in the exact resident fold order.
    pub const fn target_activation(&self) -> usize {
        self.target_activation
    }

    /// Declared width of the target ReLU.
    pub const fn target_width(&self) -> usize {
        self.target_width
    }

    /// Literal pair order; channel positions follow the flattened pair order.
    pub const fn ordered_pairs(&self) -> [[usize; 2]; 2] {
        self.ordered_pairs
    }

    /// One nonnegative multiplier for each pair's authenticated facet.
    pub const fn multipliers(&self) -> &[f32; 2] {
        &self.multipliers
    }

    /// Four preactivation channels, applied after ReLU slope propagation.
    pub const fn pre(&self) -> &[ResidentLowerCutChannel; 4] {
        &self.pre
    }

    /// Four postactivation channels, applied before ReLU sign selection.
    pub const fn post(&self) -> &[ResidentLowerCutChannel; 4] {
        &self.post
    }

    /// Single aggregate bias center and its outward source-error charge.
    pub const fn bias(&self) -> ResidentLowerCutChannel {
        self.bias
    }

    /// Mandatory call-local deadline.
    pub const fn deadline(&self) -> Instant {
        self.deadline
    }

    /// Whether at least one finite nonnegative declared multiplier is nonzero.
    pub fn has_nonzero_multiplier(&self) -> bool {
        self.multipliers.iter().any(|value| value.to_bits() != 0)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::{
        GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, GpuResnetSegment,
        ResidentLowerCutCarrier,
    };

    fn channel(value: f32, error: f32) -> ResidentLowerCutChannel {
        ResidentLowerCutChannel::try_new(value, error).expect("finite test channel")
    }

    fn candidate(
        width: usize,
        pairs: [[usize; 2]; 2],
        multipliers: [f32; 2],
        deadline: Instant,
    ) -> Result<ResidentTwoPairLowerCutCarrier> {
        ResidentTwoPairLowerCutCarrier::try_new(
            1,
            width,
            pairs,
            multipliers,
            [channel(-1.0, 0.0); 4],
            [channel(2.0, 0.0); 4],
            channel(-2.0, 0.0),
            deadline,
        )
    }

    #[test]
    fn two_pair_transport_preserves_order_and_every_channel_error() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let pre = [
            channel(-1.0, 0.125),
            channel(-2.0, 0.25),
            channel(-3.0, 0.5),
            channel(-4.0, 1.0),
        ];
        let post = [
            channel(2.0, 0.0625),
            channel(4.0, 0.125),
            channel(6.0, 0.25),
            channel(8.0, 0.5),
        ];
        let bias = channel(-3.0, 0.03125);
        let carrier = ResidentTwoPairLowerCutCarrier::try_new(
            1,
            9,
            [[8, 2], [6, 0]],
            [0.5, 2.0],
            pre,
            post,
            bias,
            deadline,
        )
        .unwrap();
        assert_eq!(carrier.ordered_pairs(), [[8, 2], [6, 0]]);
        assert_eq!(carrier.pre(), &pre);
        assert_eq!(carrier.post(), &post);
        assert_eq!(carrier.bias(), bias);
        assert_eq!(carrier.multipliers(), &[0.5, 2.0]);
        assert_eq!(carrier.target_activation(), 1);
        assert_eq!(carrier.target_width(), 9);
        assert_eq!(carrier.deadline(), deadline);
        assert!(carrier.validate_for_call(2, 9, 1, deadline).is_ok());
    }

    #[test]
    fn two_pair_transport_refuses_all_index_aliases_and_out_of_range_indices() {
        let deadline = Instant::now() + Duration::from_secs(30);
        assert!(matches!(
            candidate(0, [[0, 1], [2, 3]], [1.0; 2], deadline),
            Err(NyError::InvalidSpec(_))
        ));
        for first in 0..4 {
            for second in first + 1..4 {
                let mut indices = [0, 1, 2, 3];
                indices[second] = indices[first];
                let pairs = [[indices[0], indices[1]], [indices[2], indices[3]]];
                assert!(matches!(
                    candidate(4, pairs, [1.0; 2], deadline),
                    Err(NyError::InvalidSpec(_))
                ));
            }
            for invalid in [4, usize::MAX] {
                let mut indices = [0, 1, 2, 3];
                indices[first] = invalid;
                let pairs = [[indices[0], indices[1]], [indices[2], indices[3]]];
                assert!(matches!(
                    candidate(4, pairs, [1.0; 2], deadline),
                    Err(NyError::InvalidSpec(_))
                ));
            }
        }
    }

    #[test]
    fn two_pair_multiplier_declarations_preserve_subnormals_and_refuse_invalid_bits() {
        let deadline = Instant::now() + Duration::from_secs(30);
        for position in 0..2 {
            for invalid in [
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -1.0,
                f32::from_bits(0x8000_0001),
            ] {
                let mut multipliers = [1.0; 2];
                multipliers[position] = invalid;
                assert!(matches!(
                    candidate(4, [[0, 1], [2, 3]], multipliers, deadline),
                    Err(NyError::NumericalInstability(_))
                ));
            }
        }
        let zero = candidate(4, [[0, 1], [2, 3]], [-0.0, 0.0], deadline).unwrap();
        assert_eq!((*zero.multipliers()).map(f32::to_bits), [0, 0]);
        assert!(!zero.has_nonzero_multiplier());
        // Shape validity is not execution permission or proof authority.
        assert!(zero.validate_for_call(2, 4, 1, deadline).is_ok());
        for position in 0..2 {
            let mut multipliers = [0.0; 2];
            multipliers[position] = f32::from_bits(1);
            let tiny = candidate(4, [[0, 1], [2, 3]], multipliers, deadline).unwrap();
            assert_eq!(tiny.multipliers()[position].to_bits(), 1);
            assert!(tiny.has_nonzero_multiplier());
        }
    }

    #[test]
    fn two_pair_call_binding_requires_exact_activation_width_one_row_and_deadline() {
        let deadline = Instant::now() + Duration::from_secs(30);
        let carrier = candidate(4, [[0, 1], [2, 3]], [1.0; 2], deadline).unwrap();
        for (count, width, rows) in [
            (1, 4, 1),
            (0, 4, 1),
            (2, 3, 1),
            (2, 5, 1),
            (2, 4, 0),
            (2, 4, 2),
            (2, 4, usize::MAX),
        ] {
            assert!(matches!(
                carrier.validate_for_call(count, width, rows, deadline),
                Err(NyError::InvalidSpec(_))
            ));
        }
        assert!(matches!(
            carrier.validate_for_call(2, 4, 1, deadline + Duration::from_nanos(1)),
            Err(NyError::SoundnessRefusal(_))
        ));
        assert!(carrier.validate_for_call(2, 4, 1, deadline).is_ok());
    }

    #[test]
    fn two_pair_expired_carrier_cannot_be_rebound_to_a_fresh_deadline() {
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("test instant supports a one-second lookback");
        let carrier = candidate(4, [[0, 1], [2, 3]], [1.0; 2], expired).unwrap();
        assert!(matches!(
            carrier.validate_for_call(2, 4, 1, expired),
            Err(NyError::DeadlineExceeded(_))
        ));
        let fresh = Instant::now() + Duration::from_secs(30);
        assert!(matches!(
            carrier.validate_for_call(2, 4, 1, fresh),
            Err(NyError::SoundnessRefusal(_))
        ));
    }

    #[derive(Default)]
    struct OldPairOnly {
        ordinary_calls: AtomicUsize,
        old_pair_calls: AtomicUsize,
    }

    impl GpuCrownBackward for OldPairOnly {
        fn crown_backward_gpu(
            &self,
            _layers: &[GpuCrownLayer],
            _spec: &[f32],
            _num_specs: usize,
            _input_lower: &[f32],
            _input_upper: &[f32],
        ) -> Result<GpuCrownResult> {
            self.ordinary_calls.fetch_add(1, Ordering::Relaxed);
            Ok(GpuCrownResult::default())
        }

        fn provides_bounded_lower_cut(&self) -> bool {
            true
        }

        fn crown_backward_gpu_resnet_lower_cut_with_deadline(
            &self,
            _segments: &[GpuResnetSegment],
            _seed: &GpuCrownSeed,
            _input_lower: &[f32],
            _input_upper: &[f32],
            _carrier: &ResidentLowerCutCarrier,
            _deadline: Instant,
        ) -> Result<f32> {
            self.old_pair_calls.fetch_add(1, Ordering::Relaxed);
            Ok(123.0)
        }
    }

    #[test]
    fn two_pair_default_capability_refuses_without_old_or_ordinary_delegation() {
        let engine = OldPairOnly::default();
        let backend: &dyn GpuCrownBackward = &engine;
        assert!(backend.provides_bounded_lower_cut());
        assert!(!backend.provides_bounded_two_pair_lower_cut());
        let deadline = Instant::now() + Duration::from_secs(30);
        let carrier = candidate(4, [[0, 1], [2, 3]], [1.0; 2], deadline).unwrap();
        let seed = GpuCrownSeed {
            lower_a: Arc::from([1.0_f32]),
            upper_a: Arc::from([1.0_f32]),
            lower_b: Arc::from([0.0_f32]),
            upper_b: Arc::from([0.0_f32]),
            num_specs: 1,
            current_dim: 1,
        };
        assert!(matches!(
            backend.crown_backward_gpu_resnet_two_pair_lower_cut_with_deadline(
                &[],
                &seed,
                &[-1.0],
                &[1.0],
                &carrier,
                deadline,
            ),
            Err(NyError::UnsupportedOp(_))
        ));
        assert_eq!(engine.ordinary_calls.load(Ordering::Relaxed), 0);
        assert_eq!(engine.old_pair_calls.load(Ordering::Relaxed), 0);
    }
}

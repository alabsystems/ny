// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Arithmetic transport for two disjoint local weighted-L1 inequalities.
//! This is not proof authority. The private caller authenticates both local
//! certificates, one original source/domain/map/objective, generation and clock.
//! No support4 permission or source proof is supplied by this type.
use crate::{NyError, ResidentLowerCutChannel, Result};
use std::time::Instant;

/// Two fixed-capacity groups, each containing 1 through 8 sorted indices.
/// Group labels are retained; no cross-group sorting or renormalization occurs.
/// Only active prefixes must be disjoint. Inactive indices/channels are +0.
/// The single aggregate bias is injected once. This value is deliberately not Clone.
#[derive(Debug, PartialEq)]
pub struct ResidentTwoLocalL1LowerCutCarrier {
    target_activation: usize,
    target_width: usize,
    active_lens: [usize; 2],
    ordered_neurons: [[usize; 8]; 2],
    multipliers: [f32; 2],
    pre: [[ResidentLowerCutChannel; 8]; 2],
    post: [[ResidentLowerCutChannel; 8]; 2],
    bias: ResidentLowerCutChannel,
    deadline: Instant,
}
impl ResidentTwoLocalL1LowerCutCarrier {
    /// Validate declarations without deriving certificates or reading the clock.
    /// Establish IEEE BEFORE reducing or constructing any channels; backend checks
    /// cannot recover a radius flushed by an earlier channel constructor. Both
    /// multipliers are bitwise positive finite, including positive subnormals.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        target_activation: usize,
        target_width: usize,
        active_lens: [usize; 2],
        ordered_neurons: [[usize; 8]; 2],
        multipliers: [f32; 2],
        pre: [[ResidentLowerCutChannel; 8]; 2],
        post: [[ResidentLowerCutChannel; 8]; 2],
        bias: ResidentLowerCutChannel,
        deadline: Instant,
    ) -> Result<Self> {
        if target_width == 0 || active_lens.iter().any(|n| !(1..=8).contains(n)) {
            return Err(NyError::InvalidSpec(
                "two local L1 groups require nonzero width and 1 through 8 neurons each".into(),
            ));
        }
        for group in 0..2 {
            for position in 0..active_lens[group] {
                let index = ordered_neurons[group][position];
                if index >= target_width
                    || (position != 0 && index <= ordered_neurons[group][position - 1])
                {
                    return Err(NyError::InvalidSpec(
                        "two local L1 active indices must be strictly increasing and in range"
                            .into(),
                    ));
                }
            }
            for position in active_lens[group]..8 {
                if ordered_neurons[group][position] != 0
                    || pre[group][position].value().to_bits() != 0
                    || pre[group][position].source_abs_error().to_bits() != 0
                    || post[group][position].value().to_bits() != 0
                    || post[group][position].source_abs_error().to_bits() != 0
                {
                    return Err(NyError::InvalidSpec(
                        "two local L1 inactive slots must be canonical positive zero".into(),
                    ));
                }
            }
            let bits = multipliers[group].to_bits();
            if bits == 0 || bits >= 0x7f80_0000 {
                return Err(NyError::NumericalInstability(
                    "two local L1 multipliers must be positive finite".into(),
                ));
            }
        }
        // Padding can equal a genuine active index0. Compare active prefixes only.
        for index in &ordered_neurons[0][..active_lens[0]] {
            if ordered_neurons[1][..active_lens[1]].contains(index) {
                return Err(NyError::InvalidSpec(
                    "two local L1 active groups must be disjoint".into(),
                ));
            }
        }
        Ok(Self {
            target_activation,
            target_width,
            active_lens,
            ordered_neurons,
            multipliers,
            pre,
            post,
            bias,
            deadline,
        })
    }
    /// Match one exact objective row, target width, activation and live call deadline.
    pub fn validate_for_call(
        &self,
        activation_count: usize,
        activation_width: usize,
        expected_rows: usize,
        deadline: Instant,
    ) -> Result<()> {
        if self.target_activation >= activation_count
            || self.target_width != activation_width
            || expected_rows != 1
        {
            return Err(NyError::InvalidSpec(
                "two local L1 target width/activation/one-row shape mismatch".into(),
            ));
        }
        if self.deadline != deadline {
            return Err(NyError::SoundnessRefusal(
                "two local L1 deadline differs from original call".into(),
            ));
        }
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "two local L1 deadline expired before mutation".into(),
            ));
        }
        Ok(())
    }
    /// Exact activation ordinal.
    pub const fn target_activation(&self) -> usize {
        self.target_activation
    }
    /// Exact activation width.
    pub const fn target_width(&self) -> usize {
        self.target_width
    }
    /// Both active prefix lengths; group labels are immutable.
    pub const fn active_lens(&self) -> &[usize; 2] {
        &self.active_lens
    }
    /// Each sorted active prefix followed by canonical zero padding.
    pub const fn ordered_neurons(&self) -> &[[usize; 8]; 2] {
        &self.ordered_neurons
    }
    /// Positive declarations; semantic consistency is the private caller's duty.
    pub const fn multipliers(&self) -> &[f32; 2] {
        &self.multipliers
    }
    /// Channels added after slopes.
    pub const fn pre(&self) -> &[[ResidentLowerCutChannel; 8]; 2] {
        &self.pre
    }
    /// Channels added before sign selection.
    pub const fn post(&self) -> &[[ResidentLowerCutChannel; 8]; 2] {
        &self.post
    }
    /// Aggregate center/radius for both original bias products, added once.
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
        ResidentLocalL1LowerCutCarrier,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::Duration;
    fn ch(v: f32, e: f32) -> ResidentLowerCutChannel {
        ResidentLowerCutChannel::try_new(v, e).unwrap()
    }
    fn candidate(
        lens: [usize; 2],
        scales: [f32; 2],
        t: Instant,
    ) -> Result<ResidentTwoLocalL1LowerCutCarrier> {
        let mut ids = [[0; 8]; 2];
        let mut pre = [[ch(0., 0.); 8]; 2];
        let mut post = pre;
        for g in 0..2 {
            for i in 0..lens[g].min(8) {
                ids[g][i] = 2 * i + g;
                pre[g][i] = ch(-(i as f32 + 1.), 0.125);
                post[g][i] = ch(2. * (i as f32 + 1.), 0.25);
            }
        }
        ResidentTwoLocalL1LowerCutCarrier::try_new(
            1,
            16,
            lens,
            ids,
            scales,
            pre,
            post,
            ch(-7., 0.5),
            t,
        )
    }
    fn rebuilt(
        c: &ResidentTwoLocalL1LowerCutCarrier,
        ids: [[usize; 8]; 2],
        pre: [[ResidentLowerCutChannel; 8]; 2],
        post: [[ResidentLowerCutChannel; 8]; 2],
    ) -> Result<ResidentTwoLocalL1LowerCutCarrier> {
        ResidentTwoLocalL1LowerCutCarrier::try_new(
            1,
            16,
            *c.active_lens(),
            ids,
            *c.multipliers(),
            pre,
            post,
            c.bias(),
            c.deadline(),
        )
    }
    #[test]
    fn two_local_l1_transport_preserves_all_counts_group_labels_and_channels() {
        let t = Instant::now() + Duration::from_secs(30);
        for a in 1..=8 {
            for b in 1..=8 {
                let c = candidate([a, b], [0.5, 2.], t).unwrap();
                assert_eq!(c.active_lens(), &[a, b]);
                assert_eq!(c.multipliers(), &[0.5, 2.]);
                assert_eq!(
                    (c.target_activation(), c.target_width(), c.deadline()),
                    (1, 16, t)
                );
                assert_eq!(c.bias(), ch(-7., 0.5));
                for g in 0..2 {
                    for i in 0..8 {
                        if i < c.active_lens()[g] {
                            assert_eq!(c.ordered_neurons()[g][i], 2 * i + g);
                            assert_eq!(c.pre()[g][i], ch(-(i as f32 + 1.), 0.125));
                            assert_eq!(c.post()[g][i], ch(2. * (i as f32 + 1.), 0.25));
                        } else {
                            assert_eq!(c.ordered_neurons()[g][i], 0);
                            assert_eq!(c.pre()[g][i], ch(0., 0.));
                            assert_eq!(c.post()[g][i], ch(0., 0.));
                        }
                    }
                }
                assert!(c.validate_for_call(2, 16, 1, t).is_ok());
            }
        }
        assert!(size_of::<ResidentTwoLocalL1LowerCutCarrier>() <= 640);
    }
    #[test]
    fn two_local_l1_active_disjointness_ignores_padding_and_refuses_aliases() {
        let t = Instant::now() + Duration::from_secs(30);
        let c = candidate([8, 3], [1., 1.], t).unwrap();
        assert_eq!(c.ordered_neurons()[0][0], 0);
        assert_eq!(c.ordered_neurons()[1][3], 0);
        for g in 0..2 {
            for bad in [0, 9, usize::MAX] {
                let mut lens = [3, 3];
                lens[g] = bad;
                assert!(candidate(lens, [1., 1.], t).is_err());
            }
        }
        let c = candidate([3, 3], [1., 1.], t).unwrap();
        for g in 0..2 {
            for i in 0..3 {
                for bad in [16, usize::MAX] {
                    let mut ids = *c.ordered_neurons();
                    ids[g][i] = bad;
                    assert!(rebuilt(&c, ids, *c.pre(), *c.post()).is_err());
                }
            }
        }
        for ids in [
            [[0, 0, 4, 0, 0, 0, 0, 0], [1, 3, 5, 0, 0, 0, 0, 0]],
            [[4, 2, 0, 0, 0, 0, 0, 0], [1, 3, 5, 0, 0, 0, 0, 0]],
            [[0, 2, 4, 0, 0, 0, 0, 0], [0, 3, 5, 0, 0, 0, 0, 0]],
        ] {
            assert!(rebuilt(&c, ids, *c.pre(), *c.post()).is_err());
        }
    }
    #[test]
    fn two_local_l1_every_inactive_slot_is_canonical_positive_zero() {
        let t = Instant::now() + Duration::from_secs(30);
        let c = candidate([3, 4], [1., 1.], t).unwrap();
        for g in 0..2 {
            for i in c.active_lens()[g]..8 {
                for kind in 0..5 {
                    let mut ids = *c.ordered_neurons();
                    let mut pre = *c.pre();
                    let mut post = *c.post();
                    match kind {
                        0 => ids[g][i] = 1,
                        1 => pre[g][i] = ch(-1., 0.),
                        2 => pre[g][i] = ch(0., 0.125),
                        3 => post[g][i] = ch(f32::from_bits(1), 0.),
                        _ => post[g][i] = ch(0., 0.25),
                    }
                    assert!(rebuilt(&c, ids, pre, post).is_err());
                }
            }
        }
    }
    #[test]
    fn two_local_l1_each_multiplier_is_positive_finite_by_bits() {
        let t = Instant::now() + Duration::from_secs(30);
        for g in 0..2 {
            for bad in [
                0.,
                -0.,
                -1.,
                f32::from_bits(0x80000001),
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ] {
                let mut scales = [1., 1.];
                scales[g] = bad;
                assert!(matches!(
                    candidate([3, 3], scales, t),
                    Err(NyError::NumericalInstability(_))
                ));
            }
        }
        let c = candidate([3, 3], [f32::from_bits(1), f32::from_bits(2)], t).unwrap();
        assert_eq!((*c.multipliers()).map(f32::to_bits), [1, 2]);
    }
    #[test]
    fn two_local_l1_exact_one_row_target_and_unrenewable_deadline() {
        let t = Instant::now() + Duration::from_secs(30);
        let c = candidate([3, 3], [1., 1.], t).unwrap();
        for (count, width, rows) in [
            (0, 16, 1),
            (1, 16, 1),
            (2, 15, 1),
            (2, 17, 1),
            (2, 16, 0),
            (2, 16, 2),
        ] {
            assert!(matches!(
                c.validate_for_call(count, width, rows, t),
                Err(NyError::InvalidSpec(_))
            ));
        }
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("test instant supports a one-second lookback");
        let c = candidate([3, 3], [1., 1.], expired).unwrap();
        assert!(matches!(
            c.validate_for_call(2, 16, 1, expired),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(
            c.validate_for_call(2, 16, 1, t),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    #[derive(Default)]
    struct Old {
        calls: AtomicUsize,
    }
    impl GpuCrownBackward for Old {
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
        fn provides_bounded_local_l1_lower_cut(&self) -> bool {
            true
        }
        fn provides_bounded_support4(&self) -> bool {
            true
        }
        fn crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
            &self,
            _: &[GpuResnetSegment],
            _: &GpuCrownSeed,
            _: &[f32],
            _: &[f32],
            _: &ResidentLocalL1LowerCutCarrier,
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
    fn two_local_l1_default_permission_never_delegates_to_old_capabilities() {
        let e = Old::default();
        let b: &dyn GpuCrownBackward = &e;
        assert!(b.provides_bounded_local_l1_lower_cut() && b.provides_bounded_support4());
        assert!(!b.provides_bounded_two_local_l1_lower_cut());
        let t = Instant::now() + Duration::from_secs(30);
        let c = candidate([3, 3], [1., 1.], t).unwrap();
        let q = GpuCrownSeed {
            lower_a: Arc::from([1.]),
            upper_a: Arc::from([1.]),
            lower_b: Arc::from([0.]),
            upper_b: Arc::from([0.]),
            num_specs: 1,
            current_dim: 1,
        };
        assert!(matches!(
            b.crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
                &[],
                &q,
                &[-1.],
                &[1.],
                &c,
                t
            ),
            Err(NyError::UnsupportedOp(_))
        ));
        assert_eq!(e.calls.load(Ordering::Relaxed), 0);
    }
}

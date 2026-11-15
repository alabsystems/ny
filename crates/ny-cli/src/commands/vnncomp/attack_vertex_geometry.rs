// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-vertex-geometry: the two typed levers that reshape the wrapper's
//! upfront exact-gradient falsification lane
//! (`gradient_guided_falsify_with_schedule`).
//!
//! - `attack.upfront_step_geometry: global_max_eps` replaces the lane's
//!   per-dimension APGD rule with alpha-beta-CROWN's global scalar sign step
//!   on single-output threshold objectives (ml4acopf `300_ieee` prop2).
//! - `attack.corner_seed_restarts: N` starts restarts `1..=N` at box corners
//!   (lower, upper, then random vertices) right after the historical
//!   centre-seeded restart 0 (safenlp `ruarobot/1800`, whose centre is a
//!   strict local minimum of the violation margin).
//!
//! Both are DARK by default: with neither key set the lane runs byte-for-byte
//! as before and this module prints nothing.
//!
//! SOUNDNESS. Both levers only change which candidate points the search
//! PROPOSES. Acceptance is untouched: every candidate still goes through the
//! trusted ORT forward + zero-tolerance `property_violated_f64` inside the
//! lane, and the caller's unchanged `gate_sat_with_trusted_oracle`. Every
//! proposed point is inside the lane's inward-rounded search box: a corner
//! copies box endpoints verbatim, and a global step is projected by
//! `clamp_to_box`. Neither lever can produce `unsat`.

use crate::commands::beta_crown::BetaCrownPresetSnapshot;
use crate::preset::UpfrontStepGeometry;

use super::{clamp_to_box, SimpleRng};

/// Steps per restart under `global_max_eps`. abc's restarts hit by about step
/// 6 and the off-box simulation of this rule by a median step of 16-19, so 40
/// leaves headroom while spending the rest of the slice on fresh starts.
pub(super) const GLOBAL_MAX_EPS_STEPS_PER_RESTART: usize = 40;

/// abc's `ExponentialLR` factor for the attack step (`pgd_lr_decay` 0.99).
pub(super) const GLOBAL_MAX_EPS_LR_DECAY: f32 = 0.99;

/// `alpha0 = max_d(width_d) / 8`, which is abc's `max_eps / 4` with
/// `max_eps = max_d(width_d) / 2` (`attack_interface.py:102,110-112`).
pub(super) const GLOBAL_MAX_EPS_ALPHA0_DIVISOR: f32 = 8.0;

/// Separate xorshift seed for the random-vertex corner slots, so vertex coins
/// never shift the lane's historical uniform restart stream.
pub(super) const UPFRONT_CORNER_VERTEX_SEED: u64 = 0xC0A7_5EED_0B0C_0002;

/// The two #attack-vertex-geometry requests, frozen from the preset snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct UpfrontAttackShape {
    pub(super) geometry: UpfrontStepGeometry,
    pub(super) corner_seeds: usize,
}

impl UpfrontAttackShape {
    /// Read both keys from the frozen preset. A missing or invalid snapshot is
    /// the dark default (an invalid snapshot never reaches the lane anyway).
    pub(super) fn from_snapshot(snapshot: Option<&BetaCrownPresetSnapshot>) -> Self {
        let Some(preset) = snapshot.and_then(BetaCrownPresetSnapshot::loaded) else {
            return Self::default();
        };
        Self {
            geometry: preset.attack.upfront_step_geometry.unwrap_or_default(),
            corner_seeds: preset.attack.corner_seed_restarts.unwrap_or(0),
        }
    }

    /// Whether either lever is requested (and so whether receipts print).
    pub(super) fn armed(self) -> bool {
        self.geometry != UpfrontStepGeometry::Apgd || self.corner_seeds > 0
    }
}

/// `alpha0 = max_d(width_d) / 8` over the finite, positive widths; `None`
/// for a box with no such dimension (the lane returns before that anyway).
pub(super) fn global_max_eps_alpha0(width: &[f32]) -> Option<f32> {
    let max_width = width
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .fold(0.0_f32, f32::max);
    (max_width > 0.0).then_some(max_width / GLOBAL_MAX_EPS_ALPHA0_DIVISOR)
}

/// The step size at step `step` of a restart: `alpha0 * 0.99^step`.
pub(super) fn global_max_eps_alpha(alpha0: f32, step: usize) -> f32 {
    alpha0 * GLOBAL_MAX_EPS_LR_DECAY.powi(i32::try_from(step).unwrap_or(i32::MAX))
}

/// One coordinate of the global sign step: `clip(x + alpha * sign, lo, hi)`.
///
/// The SAME scalar `alpha` for every dimension. That is the whole mechanism:
/// `alpha0` exceeds the width of most ml4acopf dimensions (median width 0.07
/// against `alpha0 = 0.255`), so those flip from vertex to vertex on the
/// gradient sign while the few wide dimensions move gradually. The result is
/// always inside `[lo, hi]` for `lo <= hi`.
pub(super) fn global_sign_coord_step(x: f32, direction: f32, alpha: f32, lo: f32, hi: f32) -> f32 {
    clamp_to_box(x + alpha * direction, lo, hi)
}

/// The corner slot of restart `restart_idx`, or `None` when that restart keeps
/// its historical seed.
///
/// Restart 0 ALWAYS keeps its historical seed (the witness, else the box
/// centre), so the first restart of an armed run is byte-identical to today's
/// and any banked `sat` its descent finds is kept. The `N` corners take
/// restarts `1..=N`. Restart 1 is what they displace first, and on the
/// two-output safenlp nets restart 1 is a verbatim replay of restart 0 (its
/// DLR row falls back to the margin row below three outputs), so the corners
/// cost no distinct historical start there; later historical restarts keep
/// their order and their seeds, one slot later per displaced one.
pub(super) fn corner_restart_slot(restart_idx: usize, corner_seeds: usize) -> Option<usize> {
    let slot = restart_idx.checked_sub(1)?;
    (slot < corner_seeds).then_some(slot)
}

/// A uniform point of the box, drawn exactly like `restart_seed`'s random arm.
/// `global_max_eps` starts every restart here (abc's uniform initialization,
/// no centre restart).
pub(super) fn global_step_uniform_seed(
    box_lo: &[f32],
    box_hi: &[f32],
    rng: &mut SimpleRng,
) -> Vec<f32> {
    (0..box_lo.len())
        .map(|d| {
            let width = (box_hi[d] - box_lo[d]).max(0.0);
            clamp_to_box(box_lo[d] + rng.next_f32() * width, box_lo[d], box_hi[d])
        })
        .collect()
}

/// The corner seed for `slot` over the lane's search box, vertex coins drawn
/// from `rng` (the dedicated corner stream).
pub(super) fn upfront_corner_seed(
    slot: usize,
    box_lo: &[f32],
    box_hi: &[f32],
    rng: &mut SimpleRng,
) -> Vec<f32> {
    ny_propagate::corner_seed_point(slot, box_lo, box_hi, || rng.next_u32() & 1 == 0)
}

/// Why `global_max_eps` did or did not drive the restart loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GeometryDecision {
    /// The key asks for the historical APGD rule (nothing to report).
    NotArmed,
    /// The lane returned before its restart loop (admission, budget, an
    /// unsupported net, an earlier stage's hit, or the deadline).
    NotReached,
    /// The wide exact-VJP pre-wave found the candidate first.
    PrewaveHit,
    /// The restart loop ran under the global scalar step.
    Engaged,
    /// The row ascends the margin/DLR ensemble, not the raw margin.
    LossScheduleNotMarginOnly,
    /// A binarized net keeps its own smooth sign-PGD recipe.
    SignNet,
    /// No finite positive box width to scale `alpha0` from.
    DegenerateBox,
}

impl GeometryDecision {
    pub(super) const fn stable_token(self) -> &'static str {
        match self {
            Self::NotArmed => "not_armed",
            Self::NotReached => "not_reached",
            Self::PrewaveHit => "prewave_hit",
            Self::Engaged => "engaged",
            Self::LossScheduleNotMarginOnly => "loss_schedule_not_margin_only",
            Self::SignNet => "sign_net",
            Self::DegenerateBox => "degenerate_box",
        }
    }
}

/// Pure engagement rule for `global_max_eps`.
pub(super) fn global_max_eps_decision(
    geometry: UpfrontStepGeometry,
    margin_only: bool,
    sign_net: bool,
    alpha0: Option<f32>,
) -> GeometryDecision {
    match geometry {
        UpfrontStepGeometry::Apgd => GeometryDecision::NotArmed,
        UpfrontStepGeometry::GlobalMaxEps if !margin_only => {
            GeometryDecision::LossScheduleNotMarginOnly
        }
        UpfrontStepGeometry::GlobalMaxEps if sign_net => GeometryDecision::SignNet,
        UpfrontStepGeometry::GlobalMaxEps if alpha0.is_none() => GeometryDecision::DegenerateBox,
        UpfrontStepGeometry::GlobalMaxEps => GeometryDecision::Engaged,
    }
}

/// Per-consultation engagement receipt for both levers.
///
/// ENGAGEMENT OBSERVABLE. Dropping the receipt prints, UNCONDITIONALLY
/// whenever the matching key is armed, one fixed-schema stderr line per
/// armed lever (and the same text as a flight event, because
/// `ny benchmarks run` discards stderr):
///
/// ```text
/// [upfront-geom] geometry=global_max_eps alpha0=<f|none> restart=<k|none> step=<n|none> hit=<0|1> engaged=<0|1> reason=<token> restarts=<n> grad_steps=<n> max_steps_per_restart=40 consultation=<token> schema=ny-cli/upfront-geom/v1
/// [pgd-init] lane=wrapper_upfront corner_seeds=<N> seeded=<n> hit=<0|1> hit_restart=<k|none> hit_by_corner=<0|1> consultation=<token> schema=ny-cli/pgd-init/v1
/// ```
///
/// Because it prints from `Drop`, every exit path of the consultation
/// (including the dozen early returns before the restart loop) reports, so an
/// armed run that never engaged (`engaged=0 reason=not_reached`) is
/// distinguishable from one that engaged and missed (`engaged=1 hit=0`). A
/// null without the line is VOID.
#[derive(Debug)]
pub(super) struct UpfrontAttackShapeReceipt {
    shape: UpfrontAttackShape,
    consultation: &'static str,
    decision: GeometryDecision,
    alpha0: Option<f32>,
    restarts: usize,
    grad_steps: usize,
    corners_seeded: usize,
    current_restart_is_corner: bool,
    hit: Option<(usize, usize)>,
    hit_by_corner: bool,
}

impl UpfrontAttackShapeReceipt {
    pub(super) fn new(shape: UpfrontAttackShape, consultation: &'static str) -> Self {
        let decision = match shape.geometry {
            UpfrontStepGeometry::Apgd => GeometryDecision::NotArmed,
            UpfrontStepGeometry::GlobalMaxEps => GeometryDecision::NotReached,
        };
        Self {
            shape,
            consultation,
            decision,
            alpha0: None,
            restarts: 0,
            grad_steps: 0,
            corners_seeded: 0,
            current_restart_is_corner: false,
            hit: None,
            hit_by_corner: false,
        }
    }

    /// Print the receipt for a consultation that never ran (the allocator
    /// skipped the lane). Prints only when a lever is armed.
    pub(super) fn emit_unconsulted(shape: UpfrontAttackShape, consultation: &'static str) {
        if shape.armed() {
            drop(Self::new(shape, consultation));
        }
    }

    pub(super) fn shape(&self) -> UpfrontAttackShape {
        self.shape
    }

    pub(super) fn record_decision(&mut self, decision: GeometryDecision, alpha0: Option<f32>) {
        self.decision = decision;
        self.alpha0 = alpha0;
    }

    pub(super) fn note_prewave_hit(&mut self) {
        if self.decision == GeometryDecision::NotReached {
            self.decision = GeometryDecision::PrewaveHit;
        }
    }

    pub(super) fn note_restart(&mut self, corner_slot: Option<usize>) {
        self.restarts += 1;
        self.current_restart_is_corner = corner_slot.is_some();
        if corner_slot.is_some() {
            self.corners_seeded += 1;
        }
    }

    pub(super) fn note_gradient_step(&mut self) {
        self.grad_steps += 1;
    }

    pub(super) fn note_hit(&mut self, restart: usize, step: usize) {
        self.hit = Some((restart, step));
        self.hit_by_corner = self.current_restart_is_corner;
    }

    fn optional<T: std::fmt::Display>(value: Option<T>) -> String {
        value.map_or_else(|| "none".to_string(), |value| value.to_string())
    }

    pub(super) fn geometry_line(&self) -> Option<String> {
        if self.shape.geometry == UpfrontStepGeometry::Apgd {
            return None;
        }
        Some(format!(
            "[upfront-geom] geometry={} alpha0={} restart={} step={} hit={} engaged={} \
             reason={} restarts={} grad_steps={} max_steps_per_restart={} consultation={} \
             schema=ny-cli/upfront-geom/v1",
            self.shape.geometry.stable_token(),
            self.alpha0
                .map_or_else(|| "none".to_string(), |alpha0| format!("{alpha0:.6e}")),
            Self::optional(self.hit.map(|(restart, _)| restart)),
            Self::optional(self.hit.map(|(_, step)| step)),
            u8::from(self.hit.is_some()),
            u8::from(self.decision == GeometryDecision::Engaged),
            self.decision.stable_token(),
            self.restarts,
            self.grad_steps,
            GLOBAL_MAX_EPS_STEPS_PER_RESTART,
            self.consultation,
        ))
    }

    pub(super) fn corner_line(&self) -> Option<String> {
        if self.shape.corner_seeds == 0 {
            return None;
        }
        Some(format!(
            "[pgd-init] lane=wrapper_upfront corner_seeds={} seeded={} hit={} hit_restart={} \
             hit_by_corner={} consultation={} schema=ny-cli/pgd-init/v1",
            self.shape.corner_seeds,
            self.corners_seeded,
            u8::from(self.hit.is_some()),
            Self::optional(self.hit.map(|(restart, _)| restart)),
            u8::from(self.hit.is_some() && self.hit_by_corner),
            self.consultation,
        ))
    }
}

impl Drop for UpfrontAttackShapeReceipt {
    fn drop(&mut self) {
        if let Some(line) = self.geometry_line() {
            eprintln!("{line}");
            crate::flight::note(
                "upfront_attack_geometry",
                crate::flight::FlightStatus::Ran,
                Some(line),
            );
        }
        if let Some(line) = self.corner_line() {
            eprintln!("{line}");
            crate::flight::note(
                "upfront_attack_corner_seeds",
                crate::flight::FlightStatus::Ran,
                Some(line),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pure schedule against alpha-beta-CROWN's recipe
    /// (`attack_interface.py:102,110-112` + `ExponentialLR(0.99)`):
    /// `alpha_t = 0.99^t * max_eps / 4` with `max_eps = max_d(width_d) / 2`.
    #[test]
    fn upfront_global_step_matches_abc_schedule() {
        // The 300_ieee prop2 box: max width 2.04 (max_eps 1.02, alpha 0.255).
        let width = [0.07_f32, 2.04, 0.0, 0.5, f32::NAN, 0.03];
        let alpha0 = global_max_eps_alpha0(&width).expect("a positive finite width exists");
        let max_eps = 2.04_f32 / 2.0;
        assert!((alpha0 - max_eps / 4.0).abs() <= f32::EPSILON * max_eps);
        assert!((alpha0 - 0.255).abs() < 1e-6);
        for step in [0_usize, 1, 6, 19, 39] {
            let expected = f64::from(alpha0) * 0.99_f64.powi(step as i32);
            let got = f64::from(global_max_eps_alpha(alpha0, step));
            assert!(
                (got - expected).abs() <= 1e-5 * expected,
                "step {step}: {got} vs {expected}"
            );
        }
        assert_eq!(global_max_eps_alpha(alpha0, 0), alpha0);
        assert_eq!(global_max_eps_alpha0(&[0.0, -1.0, f32::NAN]), None);
        assert_eq!(global_max_eps_alpha0(&[]), None);
        assert_eq!(GLOBAL_MAX_EPS_STEPS_PER_RESTART, 40);

        // ONE scalar for every dimension: a narrow dimension jumps to its
        // vertex, a wide one moves by exactly alpha.
        let alpha = global_max_eps_alpha(alpha0, 0);
        assert_eq!(global_sign_coord_step(0.5, 1.0, alpha, 0.47, 0.54), 0.54);
        assert_eq!(global_sign_coord_step(0.5, -1.0, alpha, 0.47, 0.54), 0.47);
        assert_eq!(global_sign_coord_step(0.0, 1.0, alpha, -1.02, 1.02), alpha);
        assert_eq!(global_sign_coord_step(0.3, 0.0, alpha, -1.02, 1.02), 0.3);
    }

    /// SAMPLED CONTAINMENT: the global step never leaves `[lo, hi]`, for any
    /// start inside the box, direction and step size.
    #[test]
    fn upfront_global_step_stays_inside_the_box() {
        let mut rng = SimpleRng::new(0x5EED_600D);
        for _ in 0..20_000 {
            let lo = (rng.next_f32() - 0.5) * 200.0;
            let hi = lo + rng.next_f32() * 3.0;
            let x = lo + rng.next_f32() * (hi - lo);
            let x = clamp_to_box(x, lo, hi);
            let direction = match rng.next_u32() % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
            let alpha = global_max_eps_alpha(rng.next_f32() * 10.0, (rng.next_u32() % 64) as usize);
            let next = global_sign_coord_step(x, direction, alpha, lo, hi);
            assert!(lo <= next && next <= hi, "{next} escaped [{lo}, {hi}]");
        }
    }

    /// SAMPLED CONTAINMENT for the seeds the two levers introduce: uniform
    /// starts and corners are points of the search box, corners exactly.
    #[test]
    fn upfront_corner_and_uniform_seeds_are_box_members() {
        let mut box_rng = SimpleRng::new(0xB0C5);
        let mut seed_rng = SimpleRng::new(UPFRONT_CORNER_VERTEX_SEED);
        for case in 0..300 {
            let dim = 1 + case % 31;
            let lo: Vec<f32> = (0..dim)
                .map(|_| (box_rng.next_f32() - 0.5) * 20.0)
                .collect();
            let hi: Vec<f32> = lo
                .iter()
                .enumerate()
                .map(|(d, &l)| {
                    if d % 4 == 0 {
                        l
                    } else {
                        l + box_rng.next_f32()
                    }
                })
                .collect();
            let uniform = global_step_uniform_seed(&lo, &hi, &mut seed_rng);
            for slot in 0..5 {
                let corner = upfront_corner_seed(slot, &lo, &hi, &mut seed_rng);
                for d in 0..dim {
                    assert!(corner[d] == lo[d] || corner[d] == hi[d], "case {case}");
                }
            }
            for d in 0..dim {
                assert!(lo[d] <= uniform[d] && uniform[d] <= hi[d], "case {case}");
            }
        }
    }

    /// Restart 0 (witness, else centre) is never displaced; the corners take
    /// restarts 1..=N; dark (N = 0) never yields a slot.
    #[test]
    fn corner_restart_slots_follow_the_historical_first_restart() {
        for restart in 0..8 {
            assert_eq!(corner_restart_slot(restart, 0), None, "dark: no slots");
        }
        assert_eq!(corner_restart_slot(0, 2), None, "restart 0 keeps its seed");
        assert_eq!(
            corner_restart_slot(1, 2),
            Some(0),
            "restart 1 = lower corner"
        );
        assert_eq!(
            corner_restart_slot(2, 2),
            Some(1),
            "restart 2 = upper corner"
        );
        assert_eq!(corner_restart_slot(3, 2), None);
        assert_eq!(corner_restart_slot(3, 3), Some(2), "slot 2 = a vertex");
        assert_eq!(
            corner_restart_slot(usize::MAX, usize::MAX),
            Some(usize::MAX - 1)
        );
    }

    /// Engagement is scoped: only the raw-margin (single-output threshold)
    /// schedule on a non-Sign net with a real box takes the global step.
    #[test]
    fn global_max_eps_engages_only_on_the_margin_only_schedule() {
        use GeometryDecision as D;
        use UpfrontStepGeometry as G;
        assert_eq!(
            global_max_eps_decision(G::Apgd, true, false, Some(0.1)),
            D::NotArmed
        );
        assert_eq!(
            global_max_eps_decision(G::GlobalMaxEps, true, false, Some(0.1)),
            D::Engaged
        );
        assert_eq!(
            global_max_eps_decision(G::GlobalMaxEps, false, false, Some(0.1)),
            D::LossScheduleNotMarginOnly
        );
        assert_eq!(
            global_max_eps_decision(G::GlobalMaxEps, true, true, Some(0.1)),
            D::SignNet
        );
        assert_eq!(
            global_max_eps_decision(G::GlobalMaxEps, true, false, None),
            D::DegenerateBox
        );
    }

    /// DARK BY DEFAULT: an absent key, an explicit `apgd` and `0` corners are
    /// the unarmed shape, and an unarmed receipt prints no line at all.
    #[test]
    fn attack_vertex_geometry_is_dark_by_default() {
        assert_eq!(UpfrontStepGeometry::default(), UpfrontStepGeometry::Apgd);
        let dark = UpfrontAttackShape::from_snapshot(None);
        assert_eq!(dark, UpfrontAttackShape::default());
        assert!(!dark.armed());

        let mut receipt = UpfrontAttackShapeReceipt::new(dark, "ordinary");
        receipt.note_restart(None);
        receipt.note_gradient_step();
        receipt.note_hit(0, 3);
        assert_eq!(receipt.geometry_line(), None);
        assert_eq!(receipt.corner_line(), None);

        let omitted: crate::preset::PresetConfig =
            serde_yaml::from_str("attack:\n  upfront_attack: true\n").unwrap();
        assert_eq!(omitted.attack.upfront_step_geometry, None);
        assert_eq!(omitted.attack.corner_seed_restarts, None);
        let snapshot = BetaCrownPresetSnapshot::Loaded(Box::new(omitted));
        assert!(!UpfrontAttackShape::from_snapshot(Some(&snapshot)).armed());
    }

    /// The yaml spellings parse to the shape the lane reads, and an unknown
    /// geometry token is a parse error rather than a silent default.
    #[test]
    fn attack_vertex_geometry_keys_parse_from_yaml() {
        let armed: crate::preset::PresetConfig = serde_yaml::from_str(
            "attack:\n  upfront_step_geometry: global_max_eps\n  corner_seed_restarts: 2\n",
        )
        .unwrap();
        let snapshot = BetaCrownPresetSnapshot::Loaded(Box::new(armed));
        let shape = UpfrontAttackShape::from_snapshot(Some(&snapshot));
        assert_eq!(shape.geometry, UpfrontStepGeometry::GlobalMaxEps);
        assert_eq!(shape.corner_seeds, 2);
        assert!(shape.armed());

        let apgd: crate::preset::PresetConfig =
            serde_yaml::from_str("attack:\n  upfront_step_geometry: apgd\n").unwrap();
        assert_eq!(
            apgd.attack.upfront_step_geometry,
            Some(UpfrontStepGeometry::Apgd)
        );
        assert!(serde_yaml::from_str::<crate::preset::PresetConfig>(
            "attack:\n  upfront_step_geometry: global\n"
        )
        .is_err());
    }

    /// The receipt reports engagement, the hit coordinates, and whether a
    /// corner restart made the hit — the fields a measurement reads.
    #[test]
    fn upfront_attack_shape_receipt_lines_carry_engagement_and_hit() {
        let shape = UpfrontAttackShape {
            geometry: UpfrontStepGeometry::GlobalMaxEps,
            corner_seeds: 2,
        };
        let mut receipt = UpfrontAttackShapeReceipt::new(shape, "ordinary");
        let not_reached = receipt.geometry_line().unwrap();
        assert!(not_reached.starts_with("[upfront-geom] geometry=global_max_eps "));
        assert!(not_reached.contains(" engaged=0 reason=not_reached "));
        assert!(not_reached.contains(" hit=0 "));

        receipt.record_decision(GeometryDecision::Engaged, Some(0.255));
        receipt.note_restart(Some(0));
        receipt.note_gradient_step();
        receipt.note_restart(Some(1));
        receipt.note_gradient_step();
        receipt.note_restart(None);
        receipt.note_gradient_step();
        receipt.note_hit(2, 18);
        let geometry = receipt.geometry_line().unwrap();
        assert!(geometry.contains(" restart=2 step=18 hit=1 engaged=1 reason=engaged "));
        assert!(geometry.contains(" restarts=3 grad_steps=3 max_steps_per_restart=40 "));
        assert!(geometry.contains(" alpha0=2.550000e-1 "));
        let corner = receipt.corner_line().unwrap();
        assert!(corner.starts_with("[pgd-init] lane=wrapper_upfront corner_seeds=2 seeded=2 "));
        assert!(corner.contains(" hit=1 hit_restart=2 hit_by_corner=0 "));

        let mut corner_hit = UpfrontAttackShapeReceipt::new(
            UpfrontAttackShape {
                geometry: UpfrontStepGeometry::Apgd,
                corner_seeds: 2,
            },
            "ordinary",
        );
        corner_hit.note_restart(Some(0));
        corner_hit.note_restart(Some(1));
        corner_hit.note_hit(1, 4);
        assert_eq!(corner_hit.geometry_line(), None, "geometry not armed");
        assert!(corner_hit
            .corner_line()
            .unwrap()
            .contains(" hit=1 hit_restart=1 hit_by_corner=1 "));

        let mut prewave = UpfrontAttackShapeReceipt::new(shape, "probe_first");
        prewave.note_prewave_hit();
        assert!(prewave
            .geometry_line()
            .unwrap()
            .contains(" engaged=0 reason=prewave_hit "));
    }

    /// Scope pin: each key is armed ONLY where it was measured (2026-09-26, W6B on
    /// ny_wave6_cf376ec20): the step geometry in vnncomp25/ml4acopf_2024.yaml (the
    /// three 300_ieee prop2 rows sat 3/3), the corner seeds in
    /// vnncomp26/safenlp_2024.yaml (ruarobot/1800 sat 3/3); both must be present there.
    #[test]
    fn no_shipped_preset_arms_the_attack_vertex_geometry_keys() {
        let configs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
        let mut armed = Vec::new();
        let mut scanned = 0usize;
        for year in std::fs::read_dir(&configs).expect("list configs/") {
            let year = year.expect("configs entry").path();
            let is_vnncomp = year
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("vnncomp"));
            if !is_vnncomp || !year.is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(&year).expect("list preset dir") {
                let path = entry.expect("preset entry").path();
                if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                    continue;
                }
                let preset = crate::preset::load_preset(&path)
                    .unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
                scanned += 1;
                let dir = year
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                let geometry_allowed = dir == "vnncomp25" && name == "ml4acopf_2024.yaml";
                let corners_allowed = dir == "vnncomp26" && name == "safenlp_2024.yaml";
                if geometry_allowed {
                    assert!(
                        preset.attack.upfront_step_geometry.is_some(),
                        "vnncomp25/ml4acopf_2024.yaml must arm the measured step geometry"
                    );
                }
                if corners_allowed {
                    assert_eq!(
                        preset.attack.corner_seed_restarts,
                        Some(2),
                        "vnncomp26/safenlp_2024.yaml must arm the measured corner seeds"
                    );
                }
                if (preset.attack.upfront_step_geometry.is_some() && !geometry_allowed)
                    || (preset.attack.corner_seed_restarts.is_some() && !corners_allowed)
                {
                    armed.push(path.display().to_string());
                }
            }
        }
        assert!(scanned > 0, "the scan must actually read shipped presets");
        assert!(
            armed.is_empty(),
            "attack.upfront_step_geometry / attack.corner_seed_restarts are set outside their \
             measured presets: {armed:?}"
        );
    }
}

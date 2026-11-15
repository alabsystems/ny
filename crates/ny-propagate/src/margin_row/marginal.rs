//! `#budget-governor` — marginal-value yielding, shared by every budget-consuming
//! phase of the margin-row lane.
//!
//! # The defect, measured 2026-09-17
//!
//! Each phase takes a share of the instance budget set by ITS OWN constant,
//! decided in isolation. Nothing ever asks what another second would buy here
//! versus somewhere else. Four measurements of the same shape:
//!
//! | phase                  | spends            | what the last of it buys |
//! |------------------------|-------------------|--------------------------|
//! | alpha ascent tail      | ~5 s (it 13-20)   | **1.6e-4** of root bound |
//! | `backward_interm`      | 14.6 s (topk 1024)| **0.0057** of root bound |
//! | two-clock reserve      | **9 s of 100 s**  | nothing (lane cut at 91s)|
//! | tinyimagenet collect   | ~36% of budget    | 4-5 of 40 nodes          |
//!
//! Meanwhile the BaB tree — the phase that actually closes rows — costs 1.57 s
//! per expansion and on near-miss rows sits ONE closure from `stop=queue_empty`.
//! Returning ~5 s of the ascent tail converted `idx_1280` and `idx_7886`
//! (timeout -> unsat at the official 100 s, moat clean).
//!
//! # Why a STOPPING rule, not an allocator
//!
//! The phases run in a fixed order and cannot be resumed once ended, so the only
//! decision available is "stop now and hand the remainder downstream". A
//! stopping rule needs no common currency — which matters, because the ascent
//! improves a continuous `f64` bound while the tree closes discrete domains and
//! `backward_interm` tightens intermediate boxes. Any scheme that forced those
//! into shared units would be inventing an exchange rate nobody measured.
//!
//! # The rule: RELATIVE SELF-DECAY
//!
//! A phase yields when its marginal rate falls below `ratio` of its OWN best
//! observed rate. Unit-free, scale-free, and identical for every phase — so it
//! cannot degenerate into a per-benchmark constant, which is the failure mode of
//! `alpha_iters: 12`.
//!
//! Verified against BOTH measured profiles at `ratio = 0.15`:
//!
//! * alpha ascent (per-iteration root gain): best 0.007936, so the floor is
//!   0.00119; iteration 11 gains 0.003737 (continue), iteration 12 gains
//!   0.000907 -> **YIELD AT 12**, reproducing the setting that converted two rows.
//! * `backward_interm` (rows tightened per second): 1024/6.77s = 151.3,
//!   301/8.32s = 36.2, 219/10.42s = 21.0; floor 22.7 -> **YIELD AT LAYER 3**,
//!   reproducing the measured stop.
//!
//! # Soundness — and the limit of the "schedule-only" claim
//!
//! It CANNOT produce a wrong verdict. Every bound this lane publishes is computed
//! by the unchanged certified path, and the root alpha ascent additionally
//! re-scores each iterate and reverts under `cfg.guard`.
//!
//! ⚠ **But "schedule-only, so it touches nothing" is TOO STRONG in this lane, and
//! an earlier draft of this file said exactly that.** Adversarial review found
//! `bab::closure_step` computes `remaining = cfg.deadline - Instant::now()` and
//! passes it into `closure::observe`: **remaining wall FEEDS the closure ladder's
//! escalation pricing.** So changing how much time a phase leaves behind changes
//! which sound relaxations the tree subsequently chooses. Every such relaxation
//! is certified, so no verdict can be WRONG — but the set of verdicts CAN move,
//! in either direction.
//!
//! ⇒ The honest statement: **a wrong yield cannot cost correctness, but it can
//! cost a PROOF.** A row that currently proves with a thin margin can become
//! `unknown` under a looser root or a differently-priced ladder. That is a MOAT
//! question, and it is why `ratio` ships `None` and why arming requires a full
//! moat sweep over every banked `unsat`, not a spot check.
//!
//! ⚠ **`backward_interm` AS A WHOLE is NOT an eligible yield site**, despite
//! having the same marginal shape. Two reasons, both found by review: (1) its
//! early-exit latch (`self.exhausted = true`) is already measured to cost a
//! verdict; (2) the time it frees does NOT reach the tree — intermediate phases
//! size themselves as `cfg.secs.min(0.4 * remaining)`, so an earlier stop merely
//! inflates the ASCENT's cap, feeding the very phase this governor exists to
//! shrink. Wiring it would need an explicit credit that shrinks non-terminal
//! caps. The ascent is eligible because it is the LAST phase before the tree, so
//! its freed seconds reach the sink directly — which is why `alpha_iters:12`
//! converted two rows.
//!
//! # `#ia-yield`: the ONE site inside `backward_interm` that IS eligible
//!
//! Measured 2026-09-17 over 20 independent cifar100 gap rows (the `gap_*` logs
//! of the 49-row sweep), `#interm-alpha`'s own marginal profile is the sharpest
//! and most reproducible collapse on the board — identical in shape on 20 of 20
//! rows:
//!
//! | layer | median gain | median secs | rate            |
//! |-------|-------------|-------------|-----------------|
//! | 1     | 1136        | 4.45 s      | **255.6 /s**    |
//! | 3     | 140         | 7.05 s      | **20.4 /s**     |
//! | 5     | 41          | 7.49 s      | **5.7 /s**      |
//!
//! The dose costs `ia_secs` **15.7-20.9 s of a `bi_secs` 33.9-42.0 s root
//! build — 48.9% of it, on every row** — and the phase still ends
//! `BUDGET-EXHAUSTED` with columns uncovered (`cols=256/342`). The final segment
//! buys **3.1% of the tightenings for 39% of the time**.
//!
//! Neither objection above applies to gating THIS block:
//!
//! 1. **No latch.** It disarms only the `#interm-alpha` candidate loop. The
//!    baseline shrink-only pass keeps running, `self.exhausted` is untouched,
//!    and the phase ends exactly where its own budget or `#bi-yield` ends it.
//! 2. **The freed time does not leak into the ascent — and it is not even
//!    trying to.** It is spent INSIDE `backward_interm`'s unchanged budget,
//!    where `out_of_time()` now fires later, so more columns and more layers of
//!    the CHEAP baseline pass fit in the same seconds. Any residue that does
//!    return early reaches an ascent that is governed by YIELD, not by its cap
//!    (measured `secs=17.17` against a 20.0 s cap), so it passes through to the
//!    tree instead of inflating the phase this governor exists to shrink.
//!
//! ⇒ **The ascent governor is what makes this site eligible.** Objection (2) is
//! an argument about a budget-bound ascent; once the ascent yields on marginal
//! value it no longer absorbs upstream savings, and the two levers compose
//! rather than cancel. Ship inert, arm only behind a full moat sweep.

/// `#ia-yield` ratio for the `#interm-alpha` candidate loop inside
/// `backward_interm`. `None` ships the phase byte-identical.
pub(crate) const IA_GOVERNOR_RATIO: Option<f64> = None;

/// Layers observed before `#ia-yield` may fire. Two: the first layer sets the
/// high-water rate and can never itself trigger a stop, so the earliest possible
/// yield is after the SECOND observed layer — which is exactly where the
/// measured profile collapses (255.6 /s -> 20.4 /s, a ratio of 0.080).
pub(crate) const IA_GOVERNOR_MIN_SAMPLES: usize = 2;

/// Marginal-value yield decision for one phase.
///
/// Feed it a gain per unit of work (or per second — the rule is scale-free, so
/// the unit only has to be consistent WITHIN a phase) and it answers whether the
/// phase should stop and hand its remaining budget downstream.
#[derive(Debug, Clone)]
pub(crate) struct MarginalYield {
    /// Best rate observed so far in this phase.
    best: f64,
    /// Observations recorded.
    samples: usize,
    /// Observations required before the rule may fire, so a slow start or a
    /// single early stall cannot end a phase prematurely.
    min_samples: usize,
    /// Yield when `rate < ratio * best`. `None` disables the rule entirely and
    /// the phase runs exactly as it does today.
    ratio: Option<f64>,
    /// Sample index at which the rule fired, for the phase's observable.
    fired_at: Option<usize>,
}

impl MarginalYield {
    pub(crate) fn new(ratio: Option<f64>, min_samples: usize) -> Self {
        Self {
            best: 0.0,
            samples: 0,
            min_samples,
            ratio,
            fired_at: None,
        }
    }

    /// Record one marginal observation and decide whether to yield.
    ///
    /// A non-finite rate is recorded as no progress rather than poisoning
    /// `best` — a NaN high-water mark would disable the rule for the rest of
    /// the phase, which is the opposite of failing closed.
    pub(crate) fn observe(&mut self, rate: f64) -> bool {
        self.samples += 1;
        let rate = if rate.is_finite() { rate } else { 0.0 };
        if rate > self.best {
            self.best = rate;
        }
        let Some(ratio) = self.ratio else {
            return false;
        };
        if self.samples < self.min_samples || !ratio.is_finite() || ratio <= 0.0 {
            return false;
        }
        // `best <= 0` means the phase has never made progress. Yielding then is
        // correct — there is nothing to protect — but only once past the
        // minimum sample count, which the guard above already enforces.
        if self.best <= 0.0 {
            self.fired_at = Some(self.samples);
            return true;
        }
        if rate < ratio * self.best {
            self.fired_at = Some(self.samples);
            return true;
        }
        false
    }

    /// The yield floor this phase is currently applying (`ratio * best`).
    pub(crate) fn floor(&self) -> f64 {
        self.ratio.map_or(0.0, |r| r * self.best)
    }

    pub(crate) fn best(&self) -> f64 {
        self.best
    }

    pub(crate) fn samples(&self) -> usize {
        self.samples
    }

    #[cfg(test)]
    pub(crate) fn fired_at(&self) -> Option<usize> {
        self.fired_at
    }

    #[cfg(test)]
    pub(crate) fn armed(&self) -> bool {
        self.ratio.is_some()
    }

    /// The armed ratio (`None` when disarmed), for the phase's observable.
    pub(crate) fn ratio(&self) -> Option<f64> {
        self.ratio
    }
}

/// Where a phase's yield ratio came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GovernorSource {
    /// The shipped compile-time constant (`None`: disarmed, today's phase).
    Constant,
    /// The acting optimizer's stop authority (`optimizer.mode: on`,
    /// #router-r3 `rule.governor` / `rule.ia_yield`).
    Router,
    /// An explicit caller value (tests and research harnesses).
    Explicit,
}

/// #router-r3 `rule.governor`: the alpha ascent's yield rule, from `explicit`
/// (a caller override), else the stop authority `authority` when the acting
/// optimizer armed it, else the shipped constants ([`GOVERNOR_RATIO`] =
/// `None`: byte-identical to today). Returns `(ratio, min_samples, source)`.
pub(crate) fn ascent_governor_from(
    authority: &ny_core::stop_authority::StopAuthority,
    explicit: Option<f64>,
) -> (Option<f64>, usize, GovernorSource) {
    if let Some(ratio) = explicit {
        return (Some(ratio), GOVERNOR_MIN_SAMPLES, GovernorSource::Explicit);
    }
    match authority.governor() {
        Some(rule) => (
            Some(rule.ratio),
            usize::try_from(rule.min_samples).unwrap_or(GOVERNOR_MIN_SAMPLES),
            GovernorSource::Router,
        ),
        None => (
            GOVERNOR_RATIO,
            GOVERNOR_MIN_SAMPLES,
            GovernorSource::Constant,
        ),
    }
}

/// [`ascent_governor_from`] against the process's stop authority.
pub(crate) fn ascent_governor(explicit: Option<f64>) -> (Option<f64>, usize, GovernorSource) {
    ascent_governor_from(ny_core::stop_authority::global(), explicit)
}

/// #router-r3 `rule.ia_yield`: the `#interm-alpha` candidate loop's yield
/// rule, from the stop authority when the acting optimizer armed it, else the
/// shipped constants ([`IA_GOVERNOR_RATIO`] = `None`: byte-identical).
pub(crate) fn ia_governor_from(
    authority: &ny_core::stop_authority::StopAuthority,
) -> (Option<f64>, usize, GovernorSource) {
    match authority.ia_yield() {
        Some(rule) => (
            Some(rule.ratio),
            usize::try_from(rule.min_samples).unwrap_or(IA_GOVERNOR_MIN_SAMPLES),
            GovernorSource::Router,
        ),
        None => (
            IA_GOVERNOR_RATIO,
            IA_GOVERNOR_MIN_SAMPLES,
            GovernorSource::Constant,
        ),
    }
}

/// [`ia_governor_from`] against the process's stop authority.
pub(crate) fn ia_governor() -> (Option<f64>, usize, GovernorSource) {
    ia_governor_from(ny_core::stop_authority::global())
}

/// #router-r3: the phase `to` starts with `deadline` ahead (`None`:
/// unbounded). Every early stop the acting optimizer recorded since the last
/// call is handed to it and printed (`[optimizer] hand-on ...`). Prints and
/// returns nothing unless the stop authority is armed.
pub(crate) fn hand_on_to(
    to: &str,
    deadline: Option<std::time::Instant>,
) -> Vec<ny_core::stop_authority::StopHandOn> {
    let authority = ny_core::stop_authority::global();
    if !authority.is_armed() {
        return Vec::new();
    }
    let window = deadline.map(|d| d.saturating_duration_since(std::time::Instant::now()));
    let hand_ons = authority.downstream_start(to, window);
    for h in &hand_ons {
        eprintln!(
            "[optimizer] hand-on rule={} from={} to={} freed_ms={} window_ms={} credited_ms={} \
             realized={}",
            h.rule,
            h.from.token(),
            h.to,
            h.freed_ms,
            h.window_ms
                .map_or_else(|| "unbounded".to_owned(), |w| w.to_string()),
            h.credited_ms,
            u8::from(h.realized)
        );
    }
    hand_ons
}

/// Shipped default: `None` — every phase runs exactly as it does today, and the
/// disarmed path is byte-identical rather than merely equivalent.
pub(crate) const GOVERNOR_RATIO: Option<f64> = None;

/// Observations required before any phase may yield.
pub(crate) const GOVERNOR_MIN_SAMPLES: usize = 5;

#[cfg(test)]
mod tests {
    use super::{
        MarginalYield, GOVERNOR_MIN_SAMPLES, GOVERNOR_RATIO, IA_GOVERNOR_MIN_SAMPLES,
        IA_GOVERNOR_RATIO,
    };

    /// MEASURED per-iteration root-bound gain, cifar100 `idx_1280`, shipped
    /// preset, official 100 s, clean binary. Iterations 1..=20.
    const ASCENT: [f64; 20] = [
        0.007936, 0.007501, 0.007143, 0.006315, 0.005781, 0.005370, 0.004998, 0.004740, 0.004245,
        0.003998, 0.003737, 0.000907, 0.000160, 0.000055, 0.000029, 0.000026, 0.000001, -0.000021,
        -0.000008, 0.000000,
    ];

    /// MEASURED `backward_interm` layer rates on cifar100 resnet_large, in rows
    /// tightened per second: 1024/6.77s, 301/8.32s, 219/10.42s.
    const BACKWARD_INTERM: [f64; 3] = [151.25, 36.18, 21.02];

    fn first_yield(rates: &[f64], ratio: Option<f64>, min_samples: usize) -> Option<usize> {
        let mut y = MarginalYield::new(ratio, min_samples);
        for (i, &r) in rates.iter().enumerate() {
            if y.observe(r) {
                return Some(i + 1);
            }
        }
        None
    }

    #[test]
    fn the_measured_ascent_profile_yields_at_iteration_12() {
        // 12 is the setting that CONVERTED idx_1280 and idx_7886 (timeout ->
        // unsat at the official budget). The rule must reproduce it WITHOUT
        // naming an iteration count.
        assert_eq!(first_yield(&ASCENT, Some(0.15), 5), Some(12));
    }

    #[test]
    fn the_measured_backward_interm_profile_yields_at_layer_3() {
        // best 151.25 -> floor 22.69; layer 2 at 36.18 continues, layer 3 at
        // 21.02 stops. Matches the stop measured for `#bi-yield`.
        assert_eq!(first_yield(&BACKWARD_INTERM, Some(0.15), 2), Some(3));
    }

    #[test]
    fn one_ratio_fits_both_phases_which_is_the_whole_claim() {
        // THE UNIFICATION. Two phases with different objectives (continuous
        // f64 bound gain vs discrete rows-per-second) and rates four orders of
        // magnitude apart, governed by the SAME scale-free ratio. If this ever
        // needs different ratios per phase, the design has degenerated into the
        // per-benchmark constant it exists to avoid.
        for ratio in [0.15, 0.20] {
            assert_eq!(
                first_yield(&ASCENT, Some(ratio), 5),
                Some(12),
                "ascent @{ratio}"
            );
            assert_eq!(
                first_yield(&BACKWARD_INTERM, Some(ratio), 2),
                Some(3),
                "bi @{ratio}"
            );
        }
    }

    #[test]
    fn it_does_not_yield_while_the_phase_is_still_paying() {
        // Ascent iterations 5..=11 gain 0.0058..0.0037 against a 0.00119 floor.
        let mut y = MarginalYield::new(Some(0.15), 5);
        for (i, &r) in ASCENT.iter().enumerate().take(11) {
            assert!(!y.observe(r), "must not yield at iteration {}", i + 1);
        }
    }

    #[test]
    fn the_minimum_sample_guard_prevents_a_premature_yield() {
        // A single terrible early sample must not end a phase: a slow start is
        // not convergence.
        let mut y = MarginalYield::new(Some(0.15), 5);
        assert!(!y.observe(100.0));
        assert!(
            !y.observe(0.0),
            "sample 2 is below the floor but under the minimum"
        );
        assert!(!y.observe(0.0));
        assert!(!y.observe(0.0));
        assert!(y.observe(0.0), "allowed from the minimum on");
    }

    #[test]
    fn a_non_finite_rate_does_not_poison_the_high_water_mark() {
        // A NaN or inf `best` would disable the rule for the rest of the phase —
        // failing OPEN, the opposite of what is wanted.
        let mut y = MarginalYield::new(Some(0.15), 2);
        y.observe(10.0);
        y.observe(f64::NAN);
        y.observe(f64::INFINITY);
        assert!(y.best().is_finite(), "best stayed finite, got {}", y.best());
        assert!(
            (y.best() - 10.0).abs() < 1e-12,
            "inf/NaN recorded as no progress"
        );
        assert!(
            y.observe(0.1),
            "the rule still fires against the finite best"
        );
    }

    #[test]
    fn a_phase_that_never_progresses_yields_once_past_the_minimum() {
        // best stays 0: there is nothing to protect, so hand the budget on.
        let mut y = MarginalYield::new(Some(0.15), 3);
        assert!(!y.observe(0.0));
        assert!(!y.observe(0.0));
        assert!(y.observe(0.0));
    }

    #[test]
    fn disarmed_never_yields_so_every_phase_runs_exactly_as_today() {
        assert_eq!(first_yield(&ASCENT, None, 5), None);
        assert_eq!(first_yield(&BACKWARD_INTERM, None, 2), None);
        let mut y = MarginalYield::new(None, 5);
        for &r in ASCENT.iter() {
            assert!(!y.observe(r));
        }
        assert!(!y.armed());
        assert!(y.fired_at().is_none());
    }

    #[test]
    fn a_degenerate_ratio_fails_open() {
        for bad in [f64::NAN, 0.0, -1.0, f64::INFINITY] {
            assert_eq!(
                first_yield(&ASCENT, Some(bad), 5),
                None,
                "ratio {bad} must not yield"
            );
        }
    }

    #[test]
    fn the_dose_response_over_the_measured_profiles_is_pinned() {
        // So a future change to the rule is visible rather than silent.
        assert_eq!(first_yield(&ASCENT, Some(0.10), 5), Some(13));
        assert_eq!(first_yield(&ASCENT, Some(0.15), 5), Some(12));
        assert_eq!(first_yield(&ASCENT, Some(0.25), 5), Some(12));
        assert_eq!(
            first_yield(&BACKWARD_INTERM, Some(0.10), 2),
            None,
            "0.10 is below bi's decay"
        );
        assert_eq!(first_yield(&BACKWARD_INTERM, Some(0.25), 2), Some(2));
    }

    #[test]
    fn the_observable_fields_report_what_fired_and_where() {
        let mut y = MarginalYield::new(Some(0.15), 5);
        for &r in ASCENT.iter().take(12) {
            y.observe(r);
        }
        assert_eq!(y.fired_at(), Some(12));
        assert_eq!(y.samples(), 12);
        assert!((y.best() - 0.007936).abs() < 1e-12);
        assert!((y.floor() - 0.15 * 0.007936).abs() < 1e-12);
    }

    #[test]
    fn the_governor_ships_disarmed() {
        assert!(
            GOVERNOR_RATIO.is_none(),
            "None ships: every phase runs exactly as today and the disarmed path \
is byte-identical, not merely equivalent"
        );
        assert_eq!(GOVERNOR_MIN_SAMPLES, 5);
    }

    // ---------------------------------------------------------------------
    //  #ia-yield: the MEASURED `#interm-alpha` profile
    // ---------------------------------------------------------------------

    /// Median extra_tightened-per-second by layer, over the 20 cifar100 gap rows
    /// of the 2026-09-17 49-row sweep. Layers 1, 3, 5 of resnet_large/medium.
    /// Derived as (per-layer `extra_tightened`) / (differenced cumulative
    /// `ia_secs`) from each row's `[interm-alpha] layer=` lines; the SHAPE is
    /// identical on 20 of 20 rows, which is why a ratio rule and not a constant.
    const INTERM_ALPHA: [f64; 3] = [255.6, 20.4, 5.7];

    #[test]
    fn ia_yield_replays_the_measured_interm_alpha_profile() {
        // 20.4 / 255.6 = 0.0798, well under the 0.15 floor, so the rule fires on
        // the SECOND observed layer -- layer 3 -- and layer 5 never runs its
        // candidates. That is the measured median saving of 7.50 s.
        assert_eq!(first_yield(&INTERM_ALPHA, IA_GOVERNOR_RATIO, 2), None);
        assert_eq!(first_yield(&INTERM_ALPHA, Some(0.15), 2), Some(2));
    }

    #[test]
    fn ia_yield_dose_response_is_monotone_and_pinned() {
        // A slack floor cannot fire at all; a tight one fires a layer earlier.
        // Pinned so a future edit to `observe` shows up as a fixture break
        // rather than as a silently different schedule.
        assert_eq!(first_yield(&INTERM_ALPHA, Some(0.01), 2), None);
        assert_eq!(first_yield(&INTERM_ALPHA, Some(0.05), 2), Some(3));
        assert_eq!(first_yield(&INTERM_ALPHA, Some(0.15), 2), Some(2));
        assert_eq!(first_yield(&INTERM_ALPHA, Some(0.50), 2), Some(2));
    }

    #[test]
    fn ia_yield_first_layer_can_never_end_the_candidates() {
        // The high-water rate is set by the first observation, so a layer that
        // happens to tighten nothing at all must still not stop the lever before
        // a second layer has priced it.
        let mut y = MarginalYield::new(IA_GOVERNOR_RATIO.or(Some(0.15)), IA_GOVERNOR_MIN_SAMPLES);
        assert!(!y.observe(0.0), "layer 1 sets the baseline; it cannot fire");
        assert_eq!(y.samples(), 1);
    }

    #[test]
    fn ia_yield_ships_disarmed() {
        assert!(
            IA_GOVERNOR_RATIO.is_none(),
            "#ia-yield ships inert: `interm_alpha` runs every layer exactly as \
today until a moat sweep clears an armed ratio"
        );
        assert_eq!(IA_GOVERNOR_MIN_SAMPLES, 2);
    }

    // ---------------------------------------------------------------------
    //  #router-r3: the rules' ratios come from the acting optimizer's stop
    //  authority; unarmed they are exactly the shipped constants.
    // ---------------------------------------------------------------------

    use super::{ascent_governor_from, ia_governor_from, GovernorSource};
    use ny_core::stop_authority::{RatioRule, StopAuthority, StopDirectives};

    fn armed(governor: bool, ia: bool) -> StopAuthority {
        let a = StopAuthority::new();
        a.arm(StopDirectives {
            governor: governor.then_some(RatioRule {
                ratio: 0.15,
                min_samples: 5,
            }),
            ia_yield: ia.then_some(RatioRule {
                ratio: 0.15,
                min_samples: 2,
            }),
            attack_defer: None,
            root_admission: true,
        })
        .expect("arm");
        a
    }

    #[test]
    fn router_r3_unarmed_governors_are_exactly_the_shipped_constants() {
        let unarmed = StopAuthority::new();
        assert_eq!(
            ascent_governor_from(&unarmed, None),
            (
                GOVERNOR_RATIO,
                GOVERNOR_MIN_SAMPLES,
                GovernorSource::Constant
            )
        );
        assert_eq!(
            ia_governor_from(&unarmed),
            (
                IA_GOVERNOR_RATIO,
                IA_GOVERNOR_MIN_SAMPLES,
                GovernorSource::Constant
            )
        );
        // ... and a phase built from them never yields: today's phase.
        let (r, m, _) = ascent_governor_from(&unarmed, None);
        assert_eq!(first_yield(&ASCENT, r, m), None);
        let (r, m, _) = ia_governor_from(&unarmed);
        assert_eq!(first_yield(&INTERM_ALPHA, r, m), None);
    }

    /// The acting rules replay the MEASURED profiles: the ascent yields at
    /// iteration 12 (the converting setting), the interm-alpha candidates on
    /// the second layer.
    #[test]
    fn router_r3_armed_governors_replay_the_measured_profiles() {
        let a = armed(true, true);
        let (r, m, src) = ascent_governor_from(&a, None);
        assert_eq!(src, GovernorSource::Router);
        assert_eq!(m, 5);
        assert_eq!(first_yield(&ASCENT, r, m), Some(12));
        let (r, m, src) = ia_governor_from(&a);
        assert_eq!(src, GovernorSource::Router);
        assert_eq!(m, 2);
        assert_eq!(first_yield(&INTERM_ALPHA, r, m), Some(2));
        // Each rule is its own switch.
        let gov_only = armed(true, false);
        assert_eq!(ia_governor_from(&gov_only).2, GovernorSource::Constant);
        let ia_only = armed(false, true);
        assert_eq!(
            ascent_governor_from(&ia_only, None).2,
            GovernorSource::Constant
        );
        // An explicit caller value wins (research harness, tests).
        assert_eq!(
            ascent_governor_from(&StopAuthority::new(), Some(0.25)).2,
            GovernorSource::Explicit
        );
    }

    /// Synthetic progress streams: a phase still paying is never stopped; a
    /// collapsed one is stopped at the first unit under the floor; a phase that
    /// never progresses yields once past its minimum.
    #[test]
    fn router_r3_governor_synthetic_streams() {
        let a = armed(true, true);
        let (r, m, _) = ascent_governor_from(&a, None);
        // Geometric decay at 0.9/iter never falls under 0.15 x best within 15.
        let paying: Vec<f64> = (0..15).map(|i| 0.01 * 0.9f64.powi(i)).collect();
        assert_eq!(first_yield(&paying, r, m), None);
        // A cliff after a plateau of 6: stops at the cliff.
        let cliff = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0001, 0.01];
        assert_eq!(first_yield(&cliff, r, m), Some(7));
        // A cliff inside the minimum-sample guard waits for the guard.
        let early = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(first_yield(&early, r, m), Some(5));
        // Rejected iterates (negative deltas) never raise the high-water mark.
        let rejected = [-0.01, -0.02, -0.01, -0.03, -0.01, -0.01];
        assert_eq!(first_yield(&rejected, r, m), Some(5));
    }
}

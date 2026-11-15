// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Configuration for PGD attacks.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use ny_tensor::BoundedTensor;

use super::optimizer::{auto_alpha, AdamClippingParams, PgdAlphaMode, PgdOptimizer, PgdStepState};

/// Default GAMA guidance weight λ₀ (Sriramanan et al., NeurIPS 2020, §4:
/// λ = 50 for softmax-space guidance; annealed linearly to 0).
pub const GAMA_LAMBDA_DEFAULT: f32 = 50.0;

/// Per-run policy for the exact point-VJP graph PGD routes.
///
/// VNN-COMP constructs this from its entry-time frozen [`ny_levers::LeverSet`]
/// and carries it through [`PgdConfig`]. Interactive/compatibility entry points
/// may instead call [`Self::legacy_env_over`] once at their configuration
/// boundary. The attack consumers themselves never reopen process-global
/// environment state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactVjpPgdConfig {
    /// Prefer exact-gradient conjunctive routing over generic batched SPSA.
    pub prefer_exact_batched: bool,
    /// Attempt the point-Jacobian fast gradient in sequential graph PGD.
    pub fast_gradient: bool,
    /// Admit both batched exact-VJP drivers.
    pub batch_enabled: bool,
    /// Requested wide restart width. `None` keeps the route-specific default.
    pub batch_width: Option<usize>,
    /// Uniform seed-replay lanes in the first conjunctive wave.
    pub legacy_lanes: usize,
    /// Output-diversified initialization steps per conjunctive wave.
    pub odi_steps: usize,
    /// ODI step scale in units of the input half-range.
    pub odi_eta: f32,
    /// Joint-margin steps per conjunctive wave. `None` inherits `num_steps`.
    pub wave_steps: Option<usize>,
    /// Rate-projection lane-recycling window; zero disables.
    pub lane_window: usize,
    /// Enable endgame exploit recycling.
    pub exploit: bool,
    /// Tau-hinge threshold for odd conjunctive lanes; zero disables. The
    /// disjunctive focus wave (#disj-odi, [`Self::disj_focus_lanes`]) reuses
    /// it as its per-lane hinge.
    pub hinge_tau: f32,
    /// #disj-odi: run [`Self::odi_steps`] output-diversified-initialization
    /// steps (scale [`Self::odi_eta`]) on the DYNAMIC lanes of every
    /// disjunctive exact-VJP wave. Default false: the disjunctive driver never
    /// consults the ODI fields and stays byte-identical.
    pub disj_odi: bool,
    /// #disj-odi: dangerous-class focus lanes the disjunctive exact-VJP
    /// driver appends after its restart waves (ODI init, tau-hinge margin of
    /// the pinned class, momentum-sign steps on a cosine schedule). Zero
    /// disables.
    pub disj_focus_lanes: usize,
}

impl Default for ExactVjpPgdConfig {
    fn default() -> Self {
        Self {
            prefer_exact_batched: true,
            fast_gradient: true,
            batch_enabled: true,
            batch_width: None,
            legacy_lanes: 0,
            odi_steps: 8,
            odi_eta: 1.0,
            wave_steps: None,
            lane_window: 0,
            exploit: true,
            hinge_tau: 0.0,
            disj_odi: false,
            disj_focus_lanes: 0,
        }
    }
}

impl ExactVjpPgdConfig {
    /// Resolve this policy from the exact immutable set used for the run's
    /// lever receipt.
    pub fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        Self::from_values(|decl| {
            levers
                .resolved(decl)
                .map(|resolved| resolved.value.clone())
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        })
    }

    /// Resolve the historical process environment over an already-typed
    /// interactive policy exactly once.
    ///
    /// This preserves all legacy parsing/default behavior, including
    /// route-specific contextual defaults and default-on gates for which only
    /// the exact token `0` kills. Scored VNN-COMP runs do not call this method;
    /// they use [`Self::from_levers`].
    pub fn legacy_env_over(self) -> Result<Self, String> {
        use ny_levers::decls::exact_vjp_pgd as d;

        self.validate()?;
        let registry = ny_levers::collect(&[&d::EXACT_VJP_PGD_LEVERS])
            .map_err(|error| format!("invalid exact-VJP lever registry: {error}"))?;
        let raw = ny_levers::RawLeverInputs::capture(&registry);
        let levers =
            ny_levers::LeverSet::resolve_layered(&registry, &raw, |decl| self.config_value(decl))
                .map_err(|error| format!("invalid typed exact-VJP compatibility policy: {error}"))?;
        Self::from_levers(&levers)
    }

    /// Validate direct programmatic construction at the same typed boundary as
    /// declaration/preset materialization.
    pub fn validate(self) -> Result<(), String> {
        if self.batch_width == Some(0) {
            return Err("exact-VJP batch width must be >= 1 when configured".to_string());
        }
        if self.wave_steps == Some(0) {
            return Err("exact-VJP wave steps must be >= 1 when configured".to_string());
        }
        if !self.odi_eta.is_finite() || self.odi_eta <= 0.0 {
            return Err(format!(
                "exact-VJP ODI eta must be finite and > 0, got {}",
                self.odi_eta
            ));
        }
        if !self.hinge_tau.is_finite() || self.hinge_tau < 0.0 {
            return Err(format!(
                "exact-VJP hinge tau must be finite and >= 0, got {}",
                self.hinge_tau
            ));
        }
        Ok(())
    }

    /// Effective conjunctive width before any adapter capacity retry.
    pub const fn conjunctive_batch_width(self) -> usize {
        match self.batch_width {
            Some(width) => width,
            None => 64,
        }
    }

    /// Effective disjunctive width.
    pub const fn disjunctive_batch_width(self) -> usize {
        match self.batch_width {
            Some(width) => width,
            None => 32,
        }
    }

    /// Effective conjunctive steps per wave.
    pub const fn effective_wave_steps(self, pgd_steps: usize) -> usize {
        match self.wave_steps {
            Some(steps) => steps,
            None => pgd_steps,
        }
    }

    fn config_value(self, decl: &'static ny_levers::LeverDecl) -> Option<ny_levers::LeverValue> {
        use ny_levers::decls::exact_vjp_pgd as d;
        use ny_levers::LeverValue;

        if std::ptr::eq(decl, &raw const d::EXACT_BATCHED) {
            Some(LeverValue::Bool(self.prefer_exact_batched))
        } else if std::ptr::eq(decl, &raw const d::FAST_GRAD) {
            Some(LeverValue::Bool(self.fast_gradient))
        } else if std::ptr::eq(decl, &raw const d::VJP_BATCH) {
            Some(LeverValue::Bool(self.batch_enabled))
        } else if std::ptr::eq(decl, &raw const d::VJP_K) {
            self.batch_width.map(|value| LeverValue::U64(value as u64))
        } else if std::ptr::eq(decl, &raw const d::LEGACY_LANES) {
            Some(LeverValue::U64(self.legacy_lanes as u64))
        } else if std::ptr::eq(decl, &raw const d::ODI_STEPS) {
            Some(LeverValue::U64(self.odi_steps as u64))
        } else if std::ptr::eq(decl, &raw const d::ODI_ETA) {
            Some(LeverValue::F64(f64::from(self.odi_eta)))
        } else if std::ptr::eq(decl, &raw const d::VJP_STEPS) {
            self.wave_steps.map(|value| LeverValue::U64(value as u64))
        } else if std::ptr::eq(decl, &raw const d::LANE_WINDOW) {
            Some(LeverValue::U64(self.lane_window as u64))
        } else if std::ptr::eq(decl, &raw const d::EXPLOIT) {
            Some(LeverValue::Bool(self.exploit))
        } else if std::ptr::eq(decl, &raw const d::HINGE_TAU) {
            Some(LeverValue::F64(f64::from(self.hinge_tau)))
        } else if std::ptr::eq(decl, &raw const d::DISJ_ODI) {
            Some(LeverValue::Bool(self.disj_odi))
        } else if std::ptr::eq(decl, &raw const d::DISJ_FOCUS_LANES) {
            Some(LeverValue::U64(self.disj_focus_lanes as u64))
        } else {
            None
        }
    }

    fn from_values<F>(mut value: F) -> Result<Self, String>
    where
        F: FnMut(&'static ny_levers::LeverDecl) -> Result<ny_levers::LeverValue, String>,
    {
        use ny_levers::decls::exact_vjp_pgd as d;
        use ny_levers::LeverValue;

        let boolean =
            |decl: &'static ny_levers::LeverDecl, value: LeverValue| -> Result<bool, String> {
                match value {
                    LeverValue::Bool(value) => Ok(value),
                    other => Err(format!(
                        "{} resolved to incompatible value {other:?}",
                        decl.name
                    )),
                }
            };
        let optional_usize = |decl: &'static ny_levers::LeverDecl,
                              value: LeverValue|
         -> Result<Option<usize>, String> {
            match value {
                LeverValue::Unset => Ok(None),
                LeverValue::U64(value) => usize::try_from(value).map(Some).map_err(|_| {
                    format!("{} resolved above this platform's usize range", decl.name)
                }),
                other => Err(format!(
                    "{} resolved to incompatible value {other:?}",
                    decl.name
                )),
            }
        };
        let required_usize =
            |decl: &'static ny_levers::LeverDecl, value: LeverValue| -> Result<usize, String> {
                optional_usize(decl, value)?.ok_or_else(|| {
                    format!(
                        "{} unexpectedly resolved without a concrete value",
                        decl.name
                    )
                })
            };
        let scalar_f32 =
            |decl: &'static ny_levers::LeverDecl, value: LeverValue| -> Result<f32, String> {
                match value {
                    LeverValue::F64(value) => {
                        let narrowed = value as f32;
                        if narrowed.is_finite() && f64::from(narrowed) == value {
                            Ok(narrowed)
                        } else {
                            Err(format!(
                                "{} did not resolve to an exact finite f32",
                                decl.name
                            ))
                        }
                    }
                    other => Err(format!(
                        "{} resolved to incompatible value {other:?}",
                        decl.name
                    )),
                }
            };

        let policy = Self {
            prefer_exact_batched: boolean(&d::EXACT_BATCHED, value(&d::EXACT_BATCHED)?)?,
            fast_gradient: boolean(&d::FAST_GRAD, value(&d::FAST_GRAD)?)?,
            batch_enabled: boolean(&d::VJP_BATCH, value(&d::VJP_BATCH)?)?,
            batch_width: optional_usize(&d::VJP_K, value(&d::VJP_K)?)?,
            legacy_lanes: required_usize(&d::LEGACY_LANES, value(&d::LEGACY_LANES)?)?,
            odi_steps: required_usize(&d::ODI_STEPS, value(&d::ODI_STEPS)?)?,
            odi_eta: scalar_f32(&d::ODI_ETA, value(&d::ODI_ETA)?)?,
            wave_steps: optional_usize(&d::VJP_STEPS, value(&d::VJP_STEPS)?)?,
            lane_window: required_usize(&d::LANE_WINDOW, value(&d::LANE_WINDOW)?)?,
            exploit: boolean(&d::EXPLOIT, value(&d::EXPLOIT)?)?,
            hinge_tau: scalar_f32(&d::HINGE_TAU, value(&d::HINGE_TAU)?)?,
            disj_odi: boolean(&d::DISJ_ODI, value(&d::DISJ_ODI)?)?,
            disj_focus_lanes: required_usize(&d::DISJ_FOCUS_LANES, value(&d::DISJ_FOCUS_LANES)?)?,
        };
        policy.validate()?;
        Ok(policy)
    }
}

/// Default floor on each attacked disjunct's share of the open-disjunct stage
/// wall, in seconds (Vibecheck `phase26_pgd_per_spec_min_per_spec: 0.5`).
pub const OPEN_DISJUNCT_DEFAULT_MIN_SECS: f32 = 0.5;
/// Default wall cap of the open-disjunct stage's attack (after its root
/// wait), in seconds.
pub const OPEN_DISJUNCT_DEFAULT_MAX_SECS: f32 = 10.0;
/// Default restarts per attacked disjunct (Vibecheck tinyimagenet
/// `pgd_restarts: 100`).
pub const OPEN_DISJUNCT_DEFAULT_RESTARTS: usize = 100;
/// Default share of the time remaining at stage start that the whole stage
/// (root wait + attack) may consume.
pub const OPEN_DISJUNCT_DEFAULT_MAX_FRAC: f32 = 0.25;

/// Typed policy of the root-pruned, per-disjunct, many-restart attack stage
/// (#attack-open-disjuncts, VIBECHECK_PLAN §6.V3).
///
/// Preset keys (all under `attack:`): `open_disjunct_restrict`,
/// `open_disjunct_min_secs`, `open_disjunct_max_secs`,
/// `open_disjunct_restarts`, `open_disjunct_root_wait_secs`,
/// `open_disjunct_max_frac`. The stage runs
/// after the disjunctive attack phase of a multi-clause disjunction: it takes
/// the margin-row lane's certified root per-class bounds (when published),
/// drops every disjunct the root closed, orders the rest by root lower bound
/// (lowest first), and assigns restart `r` to open disjunct `r % n_open`, each
/// restart ascending its OWN disjunct's margin with the exact batched VJP,
/// from uniform (even cycles) or OSI (odd cycles) starts.
///
/// SOUNDNESS: attack-only. It changes where candidates are proposed, never how
/// one is accepted: every candidate passes the unchanged
/// `re_evaluate_and_confirm` (noise-scaled 1e-4 floor, attack box, per-clause
/// box) and then the trusted-ORT `gate_sat_with_trusted_oracle`. It cannot
/// produce `unsat`. Dark by default: `restrict: false` means the stage does
/// not exist and the root publication is never armed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenDisjunctAttackConfig {
    /// Master switch (`attack.open_disjunct_restrict`).
    pub restrict: bool,
    /// Floor on each attacked disjunct's share of the stage wall, seconds
    /// (`attack.open_disjunct_min_secs`). When `n_open * min_secs` exceeds
    /// the stage budget, only the lowest-root-bound disjuncts that fit are
    /// attacked; none is given less than this share.
    pub min_secs: f32,
    /// Wall cap of the ATTACK, seconds (`attack.open_disjunct_max_secs`),
    /// counted from the end of the root wait (the wait is capped separately by
    /// `root_wait_secs`); the whole stage is further clamped by `max_frac` and
    /// the instance deadline.
    pub max_secs: f32,
    /// Restarts per attacked disjunct (`attack.open_disjunct_restarts`); the
    /// stage ends early when all of them have run.
    pub restarts: usize,
    /// Longest wait for the lane's certified root, seconds
    /// (`attack.open_disjunct_root_wait_secs`). Zero uses a root only if it
    /// is already published; a lane that is not running is never waited on.
    pub root_wait_secs: f32,
    /// Share of the time remaining to the verifier's overall deadline at
    /// stage start that the WHOLE stage — root wait plus attack — may use
    /// (`attack.open_disjunct_max_frac`, in `(0, 1]`). This is what keeps one
    /// arming safe across budget tiers (cct runs 30/120/550 s rows): the
    /// absolute seconds above are caps, and `1 - max_frac` of the clock is
    /// what the BaB hand-off always keeps. The root wait never runs past this
    /// limit less one `min_secs` floor, so a lane that never publishes cannot
    /// turn the stage into a pure wait.
    pub max_frac: f32,
}

impl Default for OpenDisjunctAttackConfig {
    fn default() -> Self {
        Self {
            restrict: false,
            min_secs: OPEN_DISJUNCT_DEFAULT_MIN_SECS,
            max_secs: OPEN_DISJUNCT_DEFAULT_MAX_SECS,
            restarts: OPEN_DISJUNCT_DEFAULT_RESTARTS,
            root_wait_secs: 0.0,
            max_frac: OPEN_DISJUNCT_DEFAULT_MAX_FRAC,
        }
    }
}

impl OpenDisjunctAttackConfig {
    /// Longest any of the stage's time keys may be, seconds. Longer than every
    /// VNN-COMP instance budget (the longest is 1800 s), and small enough that
    /// every `Duration`/`Instant` conversion downstream is exact and cannot
    /// overflow.
    pub const MAX_SECS_CAP: f32 = 3600.0;

    /// Reject values that cannot describe a stage: non-finite, negative or
    /// absurdly large times, a zero per-disjunct floor (it would let an
    /// unbounded number of disjuncts share the wall), or zero restarts.
    pub fn validate(self) -> Result<(), String> {
        for (name, value) in [
            ("attack.open_disjunct_min_secs", self.min_secs),
            ("attack.open_disjunct_max_secs", self.max_secs),
            ("attack.open_disjunct_root_wait_secs", self.root_wait_secs),
        ] {
            if !value.is_finite() || !(0.0..=Self::MAX_SECS_CAP).contains(&value) {
                return Err(format!(
                    "{name} must be finite and in [0, {}], got {value}",
                    Self::MAX_SECS_CAP
                ));
            }
        }
        if self.min_secs <= 0.0 {
            return Err(format!(
                "attack.open_disjunct_min_secs must be > 0, got {}",
                self.min_secs
            ));
        }
        if self.restarts == 0 {
            return Err("attack.open_disjunct_restarts must be >= 1".to_string());
        }
        if !(self.max_frac > 0.0 && self.max_frac <= 1.0) {
            return Err(format!(
                "attack.open_disjunct_max_frac must be in (0, 1], got {}",
                self.max_frac
            ));
        }
        Ok(())
    }
}

/// Initialization strategy for PGD restarts.
///
/// Reference: alpha-beta-CROWN `attack_interface.py:29-35`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PgdInitialization {
    #[default]
    Uniform,
    Osi,
}

/// Configuration for PGD attack.
#[derive(Debug, Clone)]
pub struct PgdConfig {
    pub num_restarts: usize,
    pub num_steps: usize,
    /// Legacy scalar fallback retained for compatibility with older tests and
    /// call sites. New code should prefer `alpha_mode`.
    pub step_size: f32,
    pub spsa_delta: f32,
    pub seed: u64,
    pub parallel: bool,
    pub deadline: Option<Instant>,
    pub restart_when_stuck: bool,
    pub initialization: PgdInitialization,
    pub osi_steps: usize,
    pub optimizer: PgdOptimizer,
    pub alpha_mode: PgdAlphaMode,
    pub adam: AdamClippingParams,
    /// GAMA guidance weight λ₀ (`attack_mode: diversed_GAMA_PGD`).
    ///
    /// `Some(λ₀)` makes relational-target attack steps ascend the GAMA loss
    /// `softmax_margin + λ·‖P − softmax‖²` (Sriramanan et al., NeurIPS 2020)
    /// with λ annealed linearly from λ₀ to 0. Conjunctive graph attacks,
    /// including the generic restart-batched SPSA route and constant-threshold
    /// objectives, retain their raw property margin and add only the same
    /// annealed guidance term.
    /// `None` (default) keeps the raw-margin objective. Attack-only: it can
    /// only produce counterexample candidates, never affect a sound verdict.
    /// Reference: alpha-beta-CROWN `attack_mode: diversed_GAMA_PGD` →
    /// `GAMA_loss=True` (`attack_interface.py:29-35`).
    pub gama_lambda: Option<f32>,
    /// Straight-through-estimator surrogate gradient for `Layer::Sign`
    /// (#surrogate-sign, `attack: surrogate_sign_gradient`).
    ///
    /// `true` makes ATTACK gradient estimation treat `d/dx sign(x)` as 1
    /// (plain STE: probes are evaluated through a network whose Sign layers
    /// are replaced by the identity). The default `false` keeps the legacy
    /// `tanh(β·x)` smooth relaxation, which saturates to a zero gradient once
    /// BNN pre-activations leave `[-1, 1]` scale (traffic_signs QConv nets).
    /// Violation checks always use the TRUE Sign forward, and every candidate
    /// is re-validated downstream, so this can never affect a sound verdict.
    pub surrogate_sign_gradient: bool,
    /// Dense deterministic grid sweep for low-effective-dimension boxes
    /// (#dense-sweep, `attack: dense_low_dim_sweep`).
    ///
    /// `true` runs a pre-PGD phase when at most `dense_sweep_max_dims` input
    /// dims have nonzero width (cctsdb_yolo has 2): a uniform grid over the
    /// varying dims plus top-k local refinement, batched through the existing
    /// concrete forward machinery. Attack-only: any hit is a candidate for
    /// the normal witness path.
    pub dense_low_dim_sweep: bool,
    /// Maximum number of nonzero-width input dims for the dense sweep to run.
    pub dense_sweep_max_dims: usize,
    /// Total forward-evaluation budget for the dense sweep (grid + refinement).
    pub dense_sweep_points: usize,
    /// Box-corner seeds for the first N restarts (#attack-vertex-geometry,
    /// `attack: corner_seed_restarts`).
    ///
    /// Restart `k < N` (the restart whose seed is `seed + k`) starts at
    /// `corner_seed_point(k, ..)`
    /// ([`corner_seed_point`](crate::pgd_attack::corner_seed_point)):
    /// the lower corner, then the upper corner, then random box vertices.
    /// Every restart `k >= N` keeps its historical Uniform/OSI seed and RNG
    /// stream exactly, so `0` (the default) is byte-identical to the
    /// pre-corner schedule. Attack-only: a corner is a point of the box and
    /// every candidate is re-validated downstream.
    pub corner_seed_restarts: usize,
    /// Frozen per-run policy for batched exact point-VJP graph attacks.
    pub exact_vjp: ExactVjpPgdConfig,
    /// Cooperative cancellation (#attack-concurrent): `Some(token)` makes
    /// [`Self::past_deadline`] (and every attack loop that polls its deadline
    /// through [`Self::past_deadline`] / [`Self::cancelled`]) report "stop"
    /// as soon as the token is cancelled, so a concurrent attack lane stops at
    /// its next step or restart boundary when the proof it races decides.
    ///
    /// `None` (the default, and every caller except the concurrent attack
    /// lane) keeps every loop byte-identical: the deadline test is exactly the
    /// historical `deadline.is_some_and(|d| now >= d)`. Attack-only: stopping
    /// a search early can only withhold a candidate, never mint a verdict.
    pub cancel: Option<PgdCancelToken>,
    /// #attack-setup-regrant: the search slice a falsification PHASE declared,
    /// re-granted ONCE from the instant its search can start when the phase's
    /// own setup (dispatch prelude, batched-VJP plan construction) already
    /// consumed [`Self::deadline`] — see [`PgdSetupRegrant`].
    ///
    /// `None` (the default, and every caller except the global disjunctive
    /// PGD phase) keeps every driver byte-identical. Attack-only: it can only
    /// move WHEN candidate generation stops, never mint a verdict.
    pub setup_regrant: Option<PgdSetupRegrant>,
}

/// #attack-setup-regrant: what a driver may re-grant when its slice expired
/// before its first search step.
///
/// The disjunctive PGD slice is an instant anchored at the BaB ledger start, so
/// everything that runs before the first wave — the BaB-dispatch prelude and
/// the batched exact-VJP plan construction — is charged against it. On a large
/// model that setup can exceed the whole slice and the falsifier then performs
/// ZERO steps (`exact_vjp_execution inactive_reason=deadline_before_start`).
/// A driver that receives this re-grants, once, `min(now + slice, cap)` at
/// the moment its search is ready — the slice the phase DECLARED, spent on
/// search rather than on setup — and only when the original deadline has
/// already passed there, so a falsifier that already runs keeps its exact
/// historical deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PgdSetupRegrant {
    /// The declared search slice (the phase's fraction, measured from the
    /// phase start; already clamped by the ledger's caps).
    pub slice: Duration,
    /// Never past this instant (the caller's overall deadline).
    pub cap: Instant,
}

impl PgdSetupRegrant {
    /// The re-granted deadline at `now`, or `None` when nothing may be
    /// re-granted: the original `deadline` is still live (or unbounded), or
    /// the cap leaves no time after `now`.
    #[must_use]
    pub fn regranted_deadline(self, deadline: Option<Instant>, now: Instant) -> Option<Instant> {
        let expired = deadline.is_some_and(|limit| now >= limit);
        if !expired {
            return None;
        }
        let fresh = now
            .checked_add(self.slice)
            .map_or(self.cap, |end| end.min(self.cap));
        (fresh > now).then_some(fresh)
    }
}

/// Cooperative cancellation handle for a PGD attack (#attack-concurrent).
///
/// One shared flag: [`Self::cancel`] latches it, every clone observes it.
/// Cancellation is a token, never a deadline write — the attack's own
/// deadline is unchanged and a cancelled attack simply stops at its next poll.
#[derive(Debug, Clone, Default)]
pub struct PgdCancelToken(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl PgdCancelToken {
    /// A fresh, uncancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Latch the cancellation. Idempotent.
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Whether [`Self::cancel`] has been called on this token or any clone.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl Default for PgdConfig {
    fn default() -> Self {
        Self {
            num_restarts: 100,
            num_steps: 50,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true,
            deadline: None,
            restart_when_stuck: false,
            initialization: PgdInitialization::Uniform,
            osi_steps: 20,
            optimizer: PgdOptimizer::AdamClipping,
            alpha_mode: PgdAlphaMode::Auto,
            adam: AdamClippingParams::default(),
            gama_lambda: None,
            surrogate_sign_gradient: false,
            dense_low_dim_sweep: false,
            dense_sweep_max_dims: 3,
            dense_sweep_points: 32_768,
            corner_seed_restarts: 0,
            exact_vjp: ExactVjpPgdConfig::default(),
            cancel: None,
            setup_regrant: None,
        }
    }
}

impl PgdConfig {
    /// Whether the attack must stop now: its deadline has passed, or its
    /// cancellation token (#attack-concurrent) was cancelled. With
    /// `cancel: None` this is exactly the historical deadline test.
    pub fn past_deadline(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false) || self.cancelled()
    }

    /// Whether this attack's cancellation token (#attack-concurrent) has been
    /// cancelled. Always `false` with `cancel: None`.
    #[must_use]
    pub fn cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(PgdCancelToken::is_cancelled)
    }

    pub fn fast() -> Self {
        Self {
            num_restarts: 10,
            num_steps: 20,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: false,
            deadline: None,
            restart_when_stuck: false,
            initialization: PgdInitialization::Uniform,
            osi_steps: 20,
            optimizer: PgdOptimizer::SignedGradient,
            alpha_mode: PgdAlphaMode::Scalar(0.01),
            adam: AdamClippingParams::default(),
            gama_lambda: None,
            ..Default::default()
        }
    }

    pub fn thorough() -> Self {
        Self {
            num_restarts: 1000,
            num_steps: 100,
            step_size: 0.005,
            spsa_delta: 0.0005,
            seed: 42,
            parallel: true,
            deadline: None,
            restart_when_stuck: false,
            initialization: PgdInitialization::Uniform,
            osi_steps: 20,
            optimizer: PgdOptimizer::AdamClipping,
            alpha_mode: PgdAlphaMode::Auto,
            adam: AdamClippingParams::default(),
            gama_lambda: None,
            ..Default::default()
        }
    }

    pub fn acas_xu() -> Self {
        Self {
            num_restarts: 5000,
            num_steps: 50,
            step_size: 0.01,
            spsa_delta: 0.001,
            seed: 42,
            parallel: true,
            deadline: None,
            restart_when_stuck: true,
            initialization: PgdInitialization::Uniform,
            osi_steps: 20,
            optimizer: PgdOptimizer::AdamClipping,
            alpha_mode: PgdAlphaMode::Auto,
            adam: AdamClippingParams::default(),
            gama_lambda: None,
            ..Default::default()
        }
    }

    pub(crate) fn create_step_state(&self, input_bounds: &BoundedTensor) -> PgdStepState {
        PgdStepState::from_config(
            self.optimizer,
            self.alpha_mode,
            self.step_size,
            self.adam,
            input_bounds,
            input_bounds.shape(),
        )
    }

    pub fn base_alpha(&self, input_bounds: &BoundedTensor) -> f32 {
        match self.alpha_mode {
            PgdAlphaMode::Auto => auto_alpha(input_bounds),
            PgdAlphaMode::Scalar(alpha) => alpha,
            PgdAlphaMode::InputRangeScaled(alpha) => {
                input_bounds
                    .upper()
                    .iter()
                    .zip(input_bounds.lower().iter())
                    .map(|(u, l)| (u - l).abs())
                    .fold(0.0_f32, f32::max)
                    * alpha
            }
        }
    }

    pub fn suggested_spsa_delta(&self, input_bounds: &BoundedTensor) -> f32 {
        (self.base_alpha(input_bounds) * 0.1).max(0.001)
    }
}

#[cfg(test)]
mod cancel_token_tests {
    use super::*;
    use std::time::Duration;

    /// #attack-concurrent: with `cancel: None` (every caller but the
    /// concurrent attack lane) `past_deadline` is exactly the historical
    /// deadline test; a cancelled token stops the attack whatever its
    /// deadline, and every clone observes one cancellation.
    #[test]
    #[expect(
        clippy::redundant_clone,
        reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
    )]
    fn past_deadline_without_a_token_is_the_historical_deadline_test() {
        let mut config = PgdConfig::default();
        assert!(config.cancel.is_none(), "dark by default");
        assert!(!config.past_deadline() && !config.cancelled());
        config.deadline = Some(Instant::now() + Duration::from_secs(60));
        assert!(!config.past_deadline());
        config.deadline = Some(Instant::now());
        assert!(config.past_deadline());

        let token = PgdCancelToken::new();
        let armed = PgdConfig {
            deadline: Some(Instant::now() + Duration::from_secs(60)),
            cancel: Some(token.clone()),
            ..PgdConfig::default()
        };
        assert!(!armed.past_deadline() && !armed.cancelled());
        let remote = token.clone();
        remote.cancel();
        assert!(token.is_cancelled());
        assert!(armed.cancelled() && armed.past_deadline());
        assert!(
            armed.clone().past_deadline(),
            "a config clone shares the token"
        );
    }
}

#[cfg(test)]
mod exact_vjp_policy_tests {
    use super::*;
    use ny_levers::decls::exact_vjp_pgd::EXACT_VJP_PGD_LEVERS;

    fn registry() -> ny_levers::LeverRegistry {
        ny_levers::collect(&[&EXACT_VJP_PGD_LEVERS]).unwrap()
    }

    #[test]
    fn defaults_keep_route_specific_contextual_width_and_steps() {
        let policy = ExactVjpPgdConfig::default();
        assert!(policy.fast_gradient);
        assert_eq!(policy.conjunctive_batch_width(), 64);
        assert_eq!(policy.disjunctive_batch_width(), 32);
        assert_eq!(policy.effective_wave_steps(1_000), 1_000);
        policy.validate().unwrap();
    }

    /// #disj-odi: both keys are dark by default, resolve per frozen set, and a
    /// set that names neither resolves to exactly the historical policy.
    #[test]
    fn disj_odi_keys_are_dark_by_default_and_resolve_per_run() {
        use ny_levers::decls::exact_vjp_pgd::{DISJ_FOCUS_LANES, DISJ_ODI};

        let policy = ExactVjpPgdConfig::default();
        assert!(!policy.disj_odi);
        assert_eq!(policy.disj_focus_lanes, 0);

        let armed = ny_levers::LeverSet::freeze_with(&registry(), |name| {
            if name == DISJ_ODI.name {
                Some("1".to_string())
            } else if name == DISJ_FOCUS_LANES.name {
                Some("32".to_string())
            } else {
                None
            }
        });
        let dark = ny_levers::LeverSet::freeze_with(&registry(), |_| None);
        let armed = ExactVjpPgdConfig::from_levers(&armed).unwrap();
        let dark = ExactVjpPgdConfig::from_levers(&dark).unwrap();
        assert!(armed.disj_odi);
        assert_eq!(armed.disj_focus_lanes, 32);
        assert_eq!(
            ExactVjpPgdConfig {
                disj_odi: false,
                disj_focus_lanes: 0,
                ..armed
            },
            dark,
            "arming the two keys must change nothing else in the policy"
        );
        assert_eq!(dark, ExactVjpPgdConfig::default());
    }

    #[test]
    fn two_frozen_sets_produce_distinct_run_scoped_policies() {
        let first = ny_levers::LeverSet::freeze_with(&registry(), |name| match name {
            "NY_PGD_EXACT_BATCHED" => Some("0".to_string()),
            "NY_PGD_FAST_GRAD" => Some("0".to_string()),
            "NY_PGD_VJP_K" => Some("52".to_string()),
            "NY_PGD_ODI_ETA" => Some("0.5".to_string()),
            "NY_PGD_HINGE_TAU" => Some("0.0001".to_string()),
            _ => None,
        });
        let second = ny_levers::LeverSet::freeze_with(&registry(), |name| match name {
            "NY_PGD_EXACT_BATCHED" => Some("1".to_string()),
            "NY_PGD_FAST_GRAD" => Some("1".to_string()),
            "NY_PGD_VJP_K" => Some("96".to_string()),
            "NY_PGD_ODI_ETA" => Some("2".to_string()),
            _ => None,
        });

        let first = ExactVjpPgdConfig::from_levers(&first).unwrap();
        let second = ExactVjpPgdConfig::from_levers(&second).unwrap();
        assert!(!first.prefer_exact_batched);
        assert!(!first.fast_gradient);
        assert_eq!(first.batch_width, Some(52));
        assert_eq!(first.odi_eta, 0.5);
        assert_eq!(first.hinge_tau, 0.0001_f32);
        assert!(second.prefer_exact_batched);
        assert!(second.fast_gradient);
        assert_eq!(second.batch_width, Some(96));
        assert_eq!(second.odi_eta, 2.0);
        assert_eq!(second.hinge_tau, 0.0);
        assert_ne!(first, second, "per-run policy must not latch process-wide");
    }

    #[test]
    fn legacy_env_over_preserves_fast_gradient_parser_and_layering() {
        let _env_lock = ny_test_utils::env::lock_env();
        let registry = registry();
        let _unset: Vec<_> = registry
            .all()
            .iter()
            .map(|decl| ny_test_utils::env::ScopedEnvVar::unset(decl.name))
            .collect();
        let typed_off = ExactVjpPgdConfig {
            fast_gradient: false,
            ..ExactVjpPgdConfig::default()
        };

        assert!(!typed_off.legacy_env_over().unwrap().fast_gradient);
        {
            let _zero = ny_test_utils::env::ScopedEnvVar::set("NY_PGD_FAST_GRAD", "0");
            assert!(
                !ExactVjpPgdConfig::default()
                    .legacy_env_over()
                    .unwrap()
                    .fast_gradient
            );
        }
        {
            let _one = ny_test_utils::env::ScopedEnvVar::set("NY_PGD_FAST_GRAD", "1");
            assert!(typed_off.legacy_env_over().unwrap().fast_gradient);
        }
        {
            let _malformed = ny_test_utils::env::ScopedEnvVar::set("NY_PGD_FAST_GRAD", "false");
            assert!(
                typed_off.legacy_env_over().unwrap().fast_gradient,
                "a malformed present environment value suppresses typed false and restores the declaration's legacy-on default"
            );
        }
    }

    #[test]
    fn direct_typed_policy_rejects_values_the_declarations_reject() {
        for invalid in [
            ExactVjpPgdConfig {
                batch_width: Some(0),
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                wave_steps: Some(0),
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                odi_eta: f32::NAN,
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                odi_eta: 0.0,
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                hinge_tau: -1.0,
                ..ExactVjpPgdConfig::default()
            },
            ExactVjpPgdConfig {
                hinge_tau: f32::INFINITY,
                ..ExactVjpPgdConfig::default()
            },
        ] {
            assert!(invalid.validate().is_err(), "accepted {invalid:?}");
        }
    }
}

#[cfg(test)]
mod setup_regrant_tests {
    use super::*;

    /// #attack-setup-regrant: a deadline that already passed is re-granted the
    /// declared slice from `now`, clamped to the cap; a live or unbounded
    /// deadline is never touched; a cap at or before `now` grants nothing.
    #[test]
    fn regrant_fires_only_on_an_expired_deadline_and_respects_the_cap() {
        let now = Instant::now();
        let slice = Duration::from_millis(1_800);
        let far_cap = now + Duration::from_secs(20);
        let regrant = PgdSetupRegrant {
            slice,
            cap: far_cap,
        };
        let expired = now
            .checked_sub(Duration::from_millis(500))
            .expect("half a second before now is representable");
        assert_eq!(
            regrant.regranted_deadline(Some(expired), now),
            Some(now + slice)
        );
        // Exactly at the deadline counts as expired (the drivers' own test).
        assert_eq!(
            regrant.regranted_deadline(Some(now), now),
            Some(now + slice)
        );
        // Live and unbounded deadlines keep their historical value.
        assert_eq!(
            regrant.regranted_deadline(Some(now + Duration::from_millis(1)), now),
            None
        );
        assert_eq!(regrant.regranted_deadline(None, now), None);
        // The cap binds.
        let near_cap = PgdSetupRegrant {
            slice,
            cap: now + Duration::from_millis(300),
        };
        assert_eq!(
            near_cap.regranted_deadline(Some(expired), now),
            Some(now + Duration::from_millis(300))
        );
        // A spent cap grants nothing.
        let spent = PgdSetupRegrant { slice, cap: now };
        assert_eq!(spent.regranted_deadline(Some(expired), now), None);
        // A zero slice grants nothing.
        let empty = PgdSetupRegrant {
            slice: Duration::ZERO,
            cap: far_cap,
        };
        assert_eq!(empty.regranted_deadline(Some(expired), now), None);
    }

    /// Dark by default on every constructor.
    #[test]
    fn setup_regrant_is_dark_by_default() {
        assert!(PgdConfig::default().setup_regrant.is_none());
    }
}

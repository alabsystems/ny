// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r5 — the `ny-cli` half of acting set 4: CAPABILITIES ROUTED, NOT
//! ARMED (ROUTER_PROGRAM §3.5, §10 R5).
//!
//! `ny_core::router::route::commit` decides, purely and name-blind, which
//! capabilities CR-RS routes on the row (only at SR.seal) and whether CR-G's
//! gap-order contract funds multi-objective classes;
//! `ny_core::route_authority` carries those decisions to the consumers. This
//! module opens and closes the session and CONFIGURES the routed
//! capabilities at their own configuration boundaries:
//!
//! | site | helper | capabilities |
//! |---|---|---|
//! | `router::start` (scored entry, after the lever freeze) | [`pinned_keys`] — the typed keys the row pinned (its file, or a non-default lever source) | every route key |
//! | `router::start` (acting) | [`arm`] — one `[router] route` line per routable capability, then arm the authority | — |
//! | the verifier's configuration boundary (beside CR-K) | [`apply_to_verifier_config`] | `cgan-child-zono`, `fwd-root-screen`, `la-warm-frontier`, `vit-cover-certify`, `kfsb-single-shot` |
//! | the margin-row lane's frozen runtime policy | [`apply_to_margin_row`] | `bi-window-joint-refresh`, `wave-beta-bab`, `wave-beta-bab-cuda` |
//! | the frozen-dual tier's own arming latch (`ny-propagate`) | `zono_state::armed() \|\| routed()` | `frozen-dual-bab` |
//! | the frozen-dual tier's class loop (`ny-propagate`) | CR-G's funding | — |
//! | `finish_after_publication` | [`finish`] — applied settings, consults, tiers, DISARM | — |
//!
//! Every helper is today's value in, today's value out unless the authority
//! routed that capability: unarmed (`off`, `shadow`, `on` with nothing
//! sealed — every row today) each one is a single atomic load. A setting is
//! applied only where the row did not pin the key: the pure commit already
//! refused a route whose key the row pinned (`pin:<key>`, and pin debt when
//! the capability is sealed), so the adapters never override a pin.

use std::sync::{Mutex, PoisonError};

use ny_core::route_authority::{self, AppliedSetting, RouteAuthority, RouteSnapshot};
use ny_core::router::decl::{ids, CapId};
use ny_core::router::route::R5Commitments;
use serde::{Deserialize, Serialize};

use super::registry;

/// The R5 half of the flight `router` record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RouteRecord {
    /// Whether the authority armed (something routed, or CR-G on).
    pub(crate) armed: bool,
    /// The authority at the end of the row.
    pub(crate) snapshot: Option<RouteSnapshot>,
}

static STATE: Mutex<Option<RouteRecord>> = Mutex::new(None);

fn state() -> std::sync::MutexGuard<'static, Option<RouteRecord>> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Where a route key is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Adapter {
    /// The verifier's configuration boundary ([`apply_to_verifier_config`]).
    VerifierConfig,
    /// The margin-row lane's frozen runtime policy ([`apply_to_margin_row`]).
    MarginRowPolicy,
    /// The capability's own arming read site in `ny-propagate`.
    LaneLatch,
}

/// The adapter that applies `key` (`None`: no consumer would ever see the
/// route; a test pins that every registry route key has one).
pub(crate) fn adapter_for(key: &str) -> Option<Adapter> {
    match key {
        "solver.alpha_crown.cgan_child_zono_refresh.enabled"
        | "bab.forward_root_screen"
        | "bab.la_warm_frontier"
        | "bab.mo_cover_certify"
        | "bab.mo_containment_fence"
        | "bab.mo_root_ort_adjudication"
        | "bab.kfsb_single_shot" => Some(Adapter::VerifierConfig),
        "margin_row.bi_window" | "margin_row.tree_mode" => Some(Adapter::MarginRowPolicy),
        "margin_row.frozen_dual_bab" => Some(Adapter::LaneLatch),
        _ => None,
    }
}

/// The paths a typed key may be written under in a category file (the
/// solver-level α-CROWN section has a `bab.` twin).
fn key_paths(key: &str) -> Vec<String> {
    let mut paths = vec![key.to_owned()];
    if let Some(rest) = key.strip_prefix("solver.alpha_crown.") {
        paths.push(format!("bab.alpha_crown.{rest}"));
    }
    paths
}

/// Whether `serialized` (a category file, as serialized) sets `path`.
pub(crate) fn file_sets(serialized: &serde_json::Value, path: &str) -> bool {
    path.split('.')
        .try_fold(serialized, |node, part| node.as_object()?.get(part))
        .is_some_and(|value| !value.is_null())
}

/// The lever declaration whose axis delivers `key`, if any.
pub(crate) fn lever_for_key(key: &str) -> Option<&'static ny_levers::LeverDecl> {
    let axis = ny_levers::space::axes()
        .iter()
        .chain(ny_levers::space::unsafe_axes())
        .find(|axis| matches!(axis.deliver, ny_levers::space::Deliver::PresetKey(k) if k == key))?;
    ny_levers::all().get(axis.name)
}

/// The typed keys of every registered route that the row PINNED: set by its
/// category file (under any path the key may be written at), or resolved by
/// the frozen lever set from a non-default source. Sorted, deduplicated. A
/// file that cannot be serialized pins every key (fail closed).
pub(crate) fn pinned_keys(
    file: Option<&crate::preset::PresetConfig>,
    levers: Option<&ny_levers::LeverSet>,
) -> Vec<String> {
    let serialized = file.map(serde_json::to_value);
    let mut out = Vec::new();
    for decl in registry::REGISTRY {
        let Some(routable) = decl.routable else {
            continue;
        };
        for setting in routable.route.settings {
            let key = setting.key;
            let by_file = match &serialized {
                None => false,
                Some(Err(_)) => true,
                Some(Ok(value)) => key_paths(key).iter().any(|path| file_sets(value, path)),
            };
            let by_lever = levers.is_some_and(|levers| {
                lever_for_key(key)
                    .and_then(|lever| levers.resolved(lever))
                    .is_some_and(|resolved| resolved.source != ny_levers::Source::Default)
            });
            if by_file || by_lever {
                out.push(key.to_owned());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// `[router] act rule=<id> ...` lines plus one `[router] route` line per
/// routable capability.
pub(crate) fn decision_lines(commitments: &R5Commitments) -> Vec<String> {
    let mut lines = super::acting::act_lines_r5(commitments);
    lines.extend(commitments.routes.iter().map(|d| {
        format!(
            "[router] route cap={} sr={} routed={} blocked_by={} pin_debt={} reasons={} \
             settings={}",
            d.cap,
            d.sr.token(),
            u8::from(d.routed),
            d.blocked_by().unwrap_or("-"),
            u8::from(d.pin_debt),
            if d.reasons.is_empty() {
                "-".to_owned()
            } else {
                d.reasons.join(",")
            },
            d.settings
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(","),
        )
    }));
    lines
}

/// CR-RS / CR-G: arm the process's route authority from the commitments.
/// Called by `router::start` (inside `fail_open`) only when the router ACTS.
/// Returns the lines to print.
pub(crate) fn arm(commitments: &R5Commitments) -> Vec<String> {
    let (record, lines) = arm_with(route_authority::global(), commitments);
    *state() = Some(record);
    lines
}

pub(crate) fn arm_with(
    authority: &RouteAuthority,
    commitments: &R5Commitments,
) -> (RouteRecord, Vec<String>) {
    let mut lines = decision_lines(commitments);
    let routes: Vec<(String, Vec<(String, String)>)> = commitments
        .routed()
        .map(|d| (d.cap.clone(), d.settings.clone()))
        .collect();
    let routed_ids: Vec<String> = routes.iter().map(|(cap, _)| cap.clone()).collect();
    authority.arm(routes, commitments.gap_contract);
    let armed = authority.is_armed();
    lines.push(if armed {
        format!(
            "[router] route armed rule=CR-RS routes=[{}] gap_contract={}",
            if routed_ids.is_empty() {
                "-".to_owned()
            } else {
                routed_ids.join(",")
            },
            u8::from(commitments.gap_contract),
        )
    } else {
        "[router] route idle rule=CR-RS reason=nothing_routed_and_no_contract".to_owned()
    });
    (
        RouteRecord {
            armed,
            snapshot: None,
        },
        lines,
    )
}

/// Close the row's route session: snapshot, DISARM, and the lines to print.
/// `None` when R5 did not commit here.
pub(crate) fn finish() -> Option<(RouteRecord, Vec<String>)> {
    let mut record = state().take()?;
    let authority = route_authority::global();
    let snapshot = authority.snapshot();
    authority.disarm();
    // Each applied setting printed its own line where it was applied.
    let lines = vec![done_line(record.armed, &snapshot)];
    record.snapshot = Some(snapshot);
    Some((record, lines))
}

/// `[router] route applied cap= key= from= to= [blocked_by=]`.
pub(crate) fn applied_line(applied: &AppliedSetting) -> String {
    format!(
        "[router] route applied cap={} key={} from={} to={} source=router{}",
        applied.cap,
        applied.key,
        applied.from,
        applied.to,
        applied
            .blocked_by
            .as_deref()
            .map_or_else(String::new, |why| format!(" blocked_by={why}")),
    )
}

/// The `[router] route done ...` summary line.
pub(crate) fn done_line(armed: bool, snapshot: &RouteSnapshot) -> String {
    let consults = snapshot
        .routes
        .iter()
        .map(|r| format!("{}:{}", r.cap, r.consults))
        .collect::<Vec<_>>()
        .join(",");
    let sum = |f: fn(&route_authority::GapTier) -> u64| snapshot.tiers.iter().map(f).sum::<u64>();
    format!(
        "[router] route done armed={} routed={} consults=[{}] applied={} gap_contract={} tiers={} \
         fair={} contract={} declined={} cut={} dropped={}",
        u8::from(armed),
        snapshot.routes.len(),
        if consults.is_empty() { "-" } else { &consults },
        snapshot.applied.len(),
        u8::from(snapshot.gap_contract),
        snapshot.tiers.len(),
        sum(|t| t.fair),
        sum(|t| t.contract),
        sum(|t| t.declined),
        sum(|t| t.cut),
        snapshot.dropped,
    )
}

/// Apply one routed capability's settings to the verifier's configuration
/// (pure: the caller supplies the settings). Returns what it applied.
///
/// Every write is an explicit `config.<field> = …` assignment, so the preset
/// override frontier (`preset_arrival_ratchet::preset_override_frontier_is_exact`)
/// sees and declares each one: these fields are what a category file sets, and
/// a route writes them only where the row did not pin the key (the commit
/// refused a pinned route).
pub(crate) fn apply_config_settings(
    config: &mut ny_propagate::BetaCrownConfig,
    cap: CapId,
    settings: &[(String, String)],
) -> Vec<AppliedSetting> {
    let mut out = Vec::new();
    for (key, value) in settings {
        if adapter_for(key) != Some(Adapter::VerifierConfig) {
            continue;
        }
        let parsed = parse_bool(value);
        let (from, to, blocked) = match key.as_str() {
            "solver.alpha_crown.cgan_child_zono_refresh.enabled" => {
                // Review fix (#router-r5): written as the WHOLE field the
                // resolver fills (`preset::apply` writes
                // `config.cgan_child_zono_refresh = refresh`), so the preset
                // override frontier sees and declares this site like the
                // other five; a nested `.enabled =` write was invisible to it.
                let mut refresh = config.alpha_config.cgan_child_zono_refresh;
                let from = refresh.enabled;
                if let Some(on) = parsed {
                    refresh.enabled = on;
                }
                config.alpha_config.cgan_child_zono_refresh = refresh;
                bool_outcome(from, parsed)
            }
            "bab.forward_root_screen" => {
                let from = config.forward_root_screen;
                if let Some(on) = parsed {
                    config.forward_root_screen = on;
                }
                bool_outcome(from, parsed)
            }
            "bab.la_warm_frontier" => {
                let from = config.la_warm_frontier;
                if let Some(on) = parsed {
                    config.la_warm_frontier = on;
                }
                bool_outcome(from, parsed)
            }
            "bab.mo_containment_fence" => {
                let from = config.mo_containment_fence;
                if let Some(on) = parsed {
                    config.mo_containment_fence = on;
                }
                bool_outcome(from, parsed)
            }
            "bab.mo_root_ort_adjudication" => {
                let from = config.mo_root_ort_adjudication;
                if let Some(on) = parsed {
                    config.mo_root_ort_adjudication = on;
                }
                bool_outcome(from, parsed)
            }
            // #kfsb-production: the single-shot kFSB scorer (advisory branch
            // selection; the committed children are bounded unchanged).
            "bab.kfsb_single_shot" => {
                let from = config.kfsb_single_shot;
                if let Some(on) = parsed {
                    config.kfsb_single_shot = on;
                }
                bool_outcome(from, parsed)
            }
            "bab.mo_cover_certify" => {
                let from = config.mo_cover_certify.label().to_owned();
                match ny_propagate::MoCoverCertify::parse(value) {
                    Some(mode) => {
                        config.mo_cover_certify = mode;
                        (from, mode.label().to_owned(), None)
                    }
                    None => (from.clone(), from, Some("bad_value".to_owned())),
                }
            }
            _ => continue,
        };
        out.push(AppliedSetting {
            cap: cap.as_str().to_owned(),
            key: key.clone(),
            from,
            to,
            blocked_by: blocked,
        });
    }
    out
}

/// Apply one routed capability's settings to the margin-row lane's frozen
/// runtime policy (pure: the caller supplies the settings). Explicit writes,
/// as above.
pub(crate) fn apply_margin_row_settings(
    policy: &mut ny_propagate::margin_row::MarginRowRuntimePolicy,
    cap: CapId,
    settings: &[(String, String)],
) -> Vec<AppliedSetting> {
    let mut out = Vec::new();
    for (key, value) in settings {
        if adapter_for(key) != Some(Adapter::MarginRowPolicy) {
            continue;
        }
        let (from, to, blocked) = match key.as_str() {
            "margin_row.bi_window" => {
                let from = policy.backward_interm.window;
                let parsed = parse_bool(value);
                if let Some(on) = parsed {
                    policy.backward_interm.window = on;
                }
                let (from, to, blocked) = bool_outcome(from, parsed);
                // The window crops the `backward_interm` passes: with that
                // phase off on this row the route is inert (recorded).
                let blocked = blocked.or_else(|| {
                    (!policy.backward_interm.enabled)
                        .then(|| "inert:backward_interm_off".to_owned())
                });
                (from, to, blocked)
            }
            "margin_row.tree_mode" => {
                use ny_propagate::margin_row::TreeMode;
                let from = policy.wave.mode.token().to_owned();
                match TreeMode::from_token(value) {
                    Some(mode) => {
                        policy.wave.mode = mode;
                        (from, mode.token().to_owned(), None)
                    }
                    None => (from.clone(), from, Some("bad_value".to_owned())),
                }
            }
            _ => continue,
        };
        out.push(AppliedSetting {
            cap: cap.as_str().to_owned(),
            key: key.clone(),
            from,
            to,
            blocked_by: blocked,
        });
    }
    out
}

/// A boolean route value (`true` / `false`; anything else is refused).
fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// `(from, to, blocked_by)` of a boolean setting: applied when it parsed,
/// refused (`bad_value`, unchanged) when it did not.
fn bool_outcome(from: bool, parsed: Option<bool>) -> (String, String, Option<String>) {
    match parsed {
        Some(on) => (from.to_string(), on.to_string(), None),
        None => (
            from.to_string(),
            from.to_string(),
            Some("bad_value".to_owned()),
        ),
    }
}

/// The capabilities whose route the verifier's configuration boundary applies.
const VERIFIER_CONFIG_CAPS: [CapId; 4] = [
    ids::CGAN_CHILD_ZONO,
    ids::FWD_ROOT_SCREEN,
    ids::LA_WARM_FRONTIER,
    ids::VIT_COVER_CERTIFY,
];

/// The capabilities whose route the margin-row runtime policy applies.
const MARGIN_ROW_CAPS: [CapId; 3] = [
    ids::BI_WINDOW_JOINT_REFRESH,
    ids::WAVE_BETA_BAB_CUDA,
    ids::WAVE_BETA_BAB,
];

/// CR-RS at the verifier's configuration boundary: apply every routed
/// configuration capability's settings. A no-op (one atomic load) unless the
/// router routed one; pins were excluded by the commit.
pub(crate) fn apply_to_verifier_config(config: &mut ny_propagate::BetaCrownConfig) {
    let authority = route_authority::global();
    if !authority.is_armed() {
        return;
    }
    let applied = ny_core::router::guard::fail_open("route_config", || {
        let mut applied = Vec::new();
        for cap in VERIFIER_CONFIG_CAPS {
            if let Some(settings) = authority.settings(cap) {
                applied.extend(apply_config_settings(config, cap, &settings));
            }
        }
        applied
    });
    record(authority, applied);
}

/// CR-RS at the margin-row lane: apply every routed margin-row capability's
/// settings to the lane's frozen runtime policy. A no-op (one atomic load)
/// unless the router routed one.
pub(crate) fn apply_to_margin_row(policy: &mut ny_propagate::margin_row::MarginRowRuntimePolicy) {
    let authority = route_authority::global();
    if !authority.is_armed() {
        return;
    }
    let applied = ny_core::router::guard::fail_open("route_margin_row", || {
        let mut applied = Vec::new();
        for cap in MARGIN_ROW_CAPS {
            if let Some(settings) = authority.settings(cap) {
                applied.extend(apply_margin_row_settings(policy, cap, &settings));
            }
        }
        applied
    });
    record(authority, applied);
}

fn record(
    authority: &RouteAuthority,
    applied: Result<Vec<AppliedSetting>, ny_core::router::guard::RouterFault>,
) {
    match applied {
        Ok(applied) => {
            for entry in applied {
                eprintln!("{}", applied_line(&entry));
                authority.note_applied(entry);
            }
        }
        Err(fault) => eprintln!("{}", super::telemetry::fault_line(&fault)),
    }
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-v3-bankable: the certified SLACK an `InfeasibleDomain` close carries.
//!
//! The graph lane decides a region EMPTY when one of its split literals
//! contradicts the lane's certified bound on that pre-activation. The error it
//! raises (`NyError::InfeasibleDomain(String)`) used to be the whole
//! certificate — an error return with no number, which M2c's
//! certified-infeasibility rule forbids as a close. Every producer on the
//! constrained forward writes its bound values into the message, and this
//! module is the ONE place that knows those formats: the producers build their
//! messages here, and [`slack_from_message`] turns a message back into the
//! slack:
//!
//! * active literal `z >= 0` over a certified `u < 0`  → slack `-u`
//!   (`… but pre_u=<u> < 0`);
//! * inactive literal `z <= 0` over a certified `l > 0` → slack `l`
//!   (`… but pre_l=<l> > 0`);
//! * a constraint application that inverted a certified interval
//!   `[l, u]` with `l > u` → slack `l - u` (`…: [<l>, <u>]` at the end).
//!
//! The values are written with Rust's shortest round-trip `Display`, so the
//! parse recovers the exact binary32 value, and the slack is computed in `f64`
//! (exact for these differences of two binary32 values' magnitude range). A
//! message without one of these forms yields `None`: no slack, no evidence.
//! Pinned by the round-trip tests below and by
//! `mo_cover_emptiness_messages_round_trip_their_slack`.

/// The active-literal infeasibility message (`pre_u < 0`).
pub(crate) fn active_message(node: &str, neuron: usize, pre_u: f32) -> String {
    format!("active ReLU constraint at node '{node}' idx={neuron} but pre_u={pre_u} < 0")
}

/// The inactive-literal infeasibility message (`pre_l > 0`).
pub(crate) fn inactive_message(node: &str, neuron: usize, pre_l: f32) -> String {
    format!("inactive ReLU constraint at node '{node}' idx={neuron} but pre_l={pre_l} > 0")
}

/// The inverted-interval message of a ReLU node whose tightened bounds
/// crossed: names the widest inversion `[lower, upper]` (`lower > upper`).
pub(crate) fn interaction_message(node: &str, neuron: usize, lower: f32, upper: f32) -> String {
    format!(
        "constraint interaction at ReLU node '{node}': tightened bounds have lower > upper \
         at idx={neuron}: [{lower}, {upper}]"
    )
}

fn parse_after(message: &str, key: &str) -> Option<f32> {
    let start = message.rfind(key)? + key.len();
    let rest = &message[start..];
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    rest[..end].parse::<f32>().ok()
}

fn parse_trailing_interval(message: &str) -> Option<(f32, f32)> {
    let trimmed = message.trim_end();
    let body = trimmed.strip_suffix(']')?;
    let open = body.rfind('[')?;
    let (lower, upper) = body[open + 1..].split_once(',')?;
    Some((lower.trim().parse().ok()?, upper.trim().parse().ok()?))
}

/// The certified emptiness slack an infeasibility message carries, or `None`
/// when the message has none of the pinned forms, or the slack is not a
/// finite positive number.
#[must_use]
pub(crate) fn slack_from_message(message: &str) -> Option<f64> {
    let slack = if let Some(upper) = parse_after(message, "pre_u=") {
        -f64::from(upper)
    } else if let Some(lower) = parse_after(message, "pre_l=") {
        f64::from(lower)
    } else if let Some((lower, upper)) = parse_trailing_interval(message) {
        f64::from(lower) - f64::from(upper)
    } else {
        return None;
    };
    (slack.is_finite() && slack > 0.0).then_some(slack)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mo_cover_emptiness_messages_round_trip_their_slack() {
        // The active and inactive forms: the exact binary32 value comes back.
        for u in [-1.0e-7_f32, -0.001_234_5, -3.5, -f32::MIN_POSITIVE] {
            let message = active_message("relu1", 7, u);
            assert_eq!(
                slack_from_message(&message),
                Some(-f64::from(u)),
                "{message}"
            );
        }
        for l in [1.0e-7_f32, 0.25, 12.5] {
            let message = inactive_message("relu1", 7, l);
            assert_eq!(
                slack_from_message(&message),
                Some(f64::from(l)),
                "{message}"
            );
        }
        // The inverted-interval form.
        let message = interaction_message("relu2", 3, 0.5, 0.25);
        assert_eq!(slack_from_message(&message), Some(0.25));
        // `NyError`'s Display prefix does not disturb the parse.
        let wrapped = format!("Infeasible domain: {}", active_message("r", 0, -0.5));
        assert_eq!(slack_from_message(&wrapped), Some(0.5));
        // The lookups.rs pre-constraint form ends in the interval too.
        assert_eq!(
            slack_from_message("pre-constraint at relu='r' idx=2: [0.75, -0.25]"),
            Some(1.0)
        );
    }

    #[test]
    fn mo_cover_emptiness_slack_refuses_what_is_not_a_certificate() {
        // No slack form at all.
        assert_eq!(slack_from_message("late empty child"), None);
        assert_eq!(slack_from_message("certified empty child"), None);
        // A non-positive slack is not evidence (a zero-width contact is not
        // an empty region; a sign-flipped value is a malformed message).
        assert_eq!(slack_from_message(&active_message("r", 0, 0.0)), None);
        assert_eq!(slack_from_message(&active_message("r", 0, 0.5)), None);
        assert_eq!(slack_from_message(&inactive_message("r", 0, -0.5)), None);
        assert_eq!(
            slack_from_message(&interaction_message("r", 0, 0.25, 0.5)),
            None
        );
        // Non-finite values never become a slack.
        assert_eq!(
            slack_from_message(&active_message("r", 0, f32::NEG_INFINITY)),
            None
        );
        assert_eq!(slack_from_message(&active_message("r", 0, f32::NAN)), None);
        assert_eq!(slack_from_message("x: [inf, 0]"), None);
        assert_eq!(slack_from_message("x: [NaN, 0]"), None);
    }
}

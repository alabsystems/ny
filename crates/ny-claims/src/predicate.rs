// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Predicates: what a re-derived value must satisfy.
//!
//! Deliberately BOUNDED comparisons only. There is no `always_true`, no
//! "matches anything", and no existential ("some line mentions the lane") —
//! an unfalsifiable predicate is the prose form of the disease, and a claim
//! that cannot fail is not a claim. [`Predicate::Present`] is the weakest form
//! allowed and it is still falsifiable: the thing can be absent.
//!
//! Absence is never silently truthy. Every predicate but [`Predicate::Absent`]
//! fails on [`Value::Absent`], so a missing number can never satisfy a claim.

use serde::{Deserialize, Serialize};

use crate::extract::Value;

/// The outcome of testing one value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Judgement {
    pub holds: bool,
    /// Human-readable comparison, e.g. `observed -0.176890 within 1e-6 of -0.176890`.
    pub rendered: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Predicate {
    /// Exact string/number equality after normalisation.
    Equals {
        value: serde_json::Value,
    },
    /// `|observed - expected| <= tolerance`. Tolerance is mandatory: a bare
    /// float equality on a measured bound is a claim that will flap.
    NumberWithin {
        expected: f64,
        tolerance: f64,
    },
    AtLeast {
        min: f64,
    },
    AtMost {
        max: f64,
    },
    Contains {
        substring: String,
    },
    /// The value exists. The weakest allowed predicate.
    Present,
    /// The value does NOT exist. Used by claims that assert something is
    /// missing — e.g. "the arming variable was not exported".
    Absent,
}

impl Predicate {
    pub fn test(&self, observed: &Value) -> Judgement {
        match self {
            Self::Absent => Judgement {
                holds: observed.is_absent(),
                rendered: format!("observed {} must be absent", observed.render()),
            },
            _ if observed.is_absent() => Judgement {
                holds: false,
                rendered: format!("observed <absent>; {} cannot hold", self.describe()),
            },
            Self::Present => Judgement {
                holds: true,
                rendered: format!("observed {} is present", observed.render()),
            },
            Self::Equals { value } => {
                let holds = match (observed, value) {
                    (Value::Str(text), serde_json::Value::String(expected)) => text == expected,
                    (Value::Num(number), serde_json::Value::Number(expected)) => {
                        expected.as_f64().is_some_and(|expected| {
                            (number - expected).abs() <= f64::EPSILON * expected.abs().max(1.0)
                        })
                    }
                    // Cross-type: compare rendered forms, so `"0"` and `0`
                    // agree rather than silently failing on a CSV that typed
                    // its field differently from the claim's author.
                    (observed, expected) => {
                        normalise(&observed.render()) == normalise(&expected.to_string())
                    }
                };
                Judgement {
                    holds,
                    rendered: format!("observed {} == {}", observed.render(), value),
                }
            }
            Self::NumberWithin {
                expected,
                tolerance,
            } => match observed.as_number() {
                Some(number) => Judgement {
                    holds: (number - expected).abs() <= *tolerance,
                    rendered: format!(
                        "observed {number} within {tolerance} of {expected} (delta {:.3e})",
                        (number - expected).abs()
                    ),
                },
                None => Judgement {
                    holds: false,
                    rendered: format!("observed {} is not numeric", observed.render()),
                },
            },
            Self::AtLeast { min } => match observed.as_number() {
                Some(number) => Judgement {
                    holds: number >= *min,
                    rendered: format!("observed {number} >= {min}"),
                },
                None => Judgement {
                    holds: false,
                    rendered: format!("observed {} is not numeric", observed.render()),
                },
            },
            Self::AtMost { max } => match observed.as_number() {
                Some(number) => Judgement {
                    holds: number <= *max,
                    rendered: format!("observed {number} <= {max}"),
                },
                None => Judgement {
                    holds: false,
                    rendered: format!("observed {} is not numeric", observed.render()),
                },
            },
            Self::Contains { substring } => {
                let text = match observed {
                    Value::Str(text) => text.clone(),
                    other => other.render(),
                };
                Judgement {
                    holds: text.contains(substring),
                    rendered: format!("observed {text:?} contains {substring:?}"),
                }
            }
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Equals { value } => format!("== {value}"),
            Self::NumberWithin {
                expected,
                tolerance,
            } => format!("within {tolerance} of {expected}"),
            Self::AtLeast { min } => format!(">= {min}"),
            Self::AtMost { max } => format!("<= {max}"),
            Self::Contains { substring } => format!("contains {substring:?}"),
            Self::Present => "present".to_string(),
            Self::Absent => "absent".to_string(),
        }
    }
}

fn normalise(text: &str) -> String {
    text.trim().trim_matches('"').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The load-bearing rule: an absent observation satisfies nothing. A
    /// predicate that quietly passed on a missing number would reproduce
    /// exactly the failure this crate exists to close.
    #[test]
    fn absence_satisfies_nothing_except_the_absence_predicate() {
        let absent = Value::Absent;
        for predicate in [
            Predicate::Present,
            Predicate::AtLeast { min: 0.0 },
            Predicate::AtMost { max: 1e9 },
            Predicate::Contains {
                substring: String::new(),
            },
            Predicate::Equals {
                value: serde_json::json!("x"),
            },
            Predicate::NumberWithin {
                expected: 0.0,
                tolerance: 1e9,
            },
        ] {
            assert!(
                !predicate.test(&absent).holds,
                "{predicate:?} must not hold on an absent value"
            );
        }
        assert!(Predicate::Absent.test(&absent).holds);
    }

    #[test]
    fn a_string_verdict_compares_exactly() {
        let predicate = Predicate::Equals {
            value: serde_json::json!("unsat"),
        };
        assert!(predicate.test(&Value::Str("unsat".into())).holds);
        assert!(!predicate.test(&Value::Str("timeout".into())).holds);
    }

    #[test]
    fn a_measured_bound_compares_with_a_tolerance() {
        let predicate = Predicate::NumberWithin {
            expected: -0.176_890,
            tolerance: 1e-6,
        };
        assert!(predicate.test(&Value::Num(-0.176_890)).holds);
        assert!(!predicate.test(&Value::Num(-0.176_9)).holds);
    }

    #[test]
    fn a_numeric_string_and_a_number_compare_equal() {
        let predicate = Predicate::Equals {
            value: serde_json::json!(0),
        };
        assert!(predicate.test(&Value::Num(0.0)).holds);
    }
}

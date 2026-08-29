// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Extractors: the closed, typed op set that re-derives a claimed number from
//! the artifacts it was claimed from.
//!
//! # Why a closed op set and not a regex or a shell snippet
//!
//! An extractor IS the evidence — it is the only thing standing between a
//! sentence and a fact — so it has to be auditable by someone who did not
//! write it, and it has to be reproducible on a machine that is not this one.
//! A shell snippet is neither: it is unbounded, it can read files the claim
//! never cited, and it silently depends on whatever `grep` is on the path. An
//! unbounded regex is only slightly better: partial matches and greedy
//! captures fail *quietly*, and quiet failure is the whole disease.
//!
//! So the op set is closed and each op is implemented once, here, with tests.
//! It is small because the workspace's log and receipt conventions are already
//! regular: `key=value` tokens (`route=`, `root_bound=`, `gpu_lines=`, `id=`,
//! `stamp=`), one-instance-per-row CSV banks, and JSON flight receipts. Every
//! extraction in the shipped ledger is expressible in it.
//!
//! # Every extraction returns a CITATION
//!
//! Not just a value: the artifact and the 1-based line it came from. The
//! campaign's own method rules ask for `file:line` on every finding; making
//! the extractor produce it means a receipt cannot report a number without
//! also reporting where to look, and a reader can disagree with the machine.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

/// Which occurrence of a repeated key to take, or how to aggregate them.
///
/// [`Occurrence::Unique`] is the default on purpose: a key that was expected
/// once and appears twice with different values is a defect in the artifact,
/// and defaulting to `First` would hide it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Occurrence {
    #[default]
    Unique,
    First,
    Last,
    /// Numeric max over all occurrences. The right aggregation for an
    /// engagement witness across repetitions of an A/B: "did the lane execute
    /// on ANY run" is `max(gpu_lines) >= 1`, and it stays honest when one of
    /// nine runs engaged.
    Max,
    /// Numeric min over all occurrences — "did the lane execute on EVERY run".
    Min,
    /// How many occurrences there are.
    Count,
}

/// A value pulled out of an artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Num(f64),
    Str(String),
    /// The artifact was read successfully and the thing is NOT THERE.
    ///
    /// A first-class outcome, never an error: "the observable is absent" is
    /// the finding in reversals 4 and 5, and an extractor that returned `Err`
    /// for it would let a caller conflate it with "the file is missing".
    Absent,
}

impl Value {
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Num(number) => Some(*number),
            Self::Str(text) => text.trim().parse::<f64>().ok(),
            Self::Absent => None,
        }
    }

    pub fn render(&self) -> String {
        match self {
            Self::Num(number) => format!("{number}"),
            Self::Str(text) => format!("{text:?}"),
            Self::Absent => "<absent>".to_string(),
        }
    }

    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }
}

/// Where a value came from, so a receipt can print `path:line`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    pub artifact: String,
    /// 1-based, when the op has a single source line.
    pub line: Option<usize>,
    /// How the op found it, e.g. `key=root_bound occurrence=unique`.
    pub detail: String,
}

impl Citation {
    pub fn render(&self) -> String {
        match self.line {
            Some(line) => format!("{}:{}", self.artifact, line),
            None => self.artifact.clone(),
        }
    }
}

/// An extracted value with its citation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Extracted {
    pub value: Value,
    pub citation: Citation,
}

/// The closed op set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Extractor {
    /// A comma-separated field of the first line containing `row_contains`.
    ///
    /// `row_contains` doubles as the INSTANCE GUARD: the campaign's wrong-ROW
    /// trap (a sweep that ran a banked sibling and read as a conversion) is a
    /// row-selection bug, so selection is by content, never by index.
    CsvField {
        artifact: String,
        row_contains: String,
        column: usize,
    },
    /// The value of a `key=value` token. Value runs to the next whitespace,
    /// comma or closing bracket.
    KeyValue {
        artifact: String,
        key: String,
        /// Restrict to lines containing this substring.
        ///
        /// This is the INSTANCE GUARD in its second form. A results file holds
        /// one line per row; reading `root_bound=` out of the whole file and
        /// taking the first hit is how a number gets attributed to the wrong
        /// row (the campaign's wrong-ROW trap: a sweep that ran a banked
        /// sibling and read as a conversion). With the filter set, a claim
        /// about `idx_8600` physically cannot read `idx_6659`'s bound.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        in_lines_containing: Option<String>,
        #[serde(default)]
        occurrence: Occurrence,
    },
    /// How many lines contain `needle`.
    CountLinesContaining { artifact: String, needle: String },
    /// The whole line containing `needle`.
    LineContaining {
        artifact: String,
        needle: String,
        #[serde(default)]
        occurrence: Occurrence,
    },
    /// A JSON pointer (RFC 6901) into a JSON artifact, e.g.
    /// `/host/hostname` or `/ambient_env/NY_MARGIN_ROW_GPU_BACKWARD`.
    JsonPointer { artifact: String, pointer: String },
    /// The `status` of a named flight-recorder event.
    FlightEventStatus { artifact: String, method: String },
    /// A `key=value` token inside the `reason` of a named flight event. This
    /// is how `route=` is read out of a benchmark's own receipt, which is the
    /// check that reversal 4 skipped.
    FlightEventField {
        artifact: String,
        method: String,
        key: String,
    },
}

impl Extractor {
    /// The artifact this op reads. Every op reads exactly one, so a claim's
    /// cited-artifact set can be checked against what its extractors actually
    /// open — a claim cannot quietly depend on a file it did not cite.
    pub fn artifact(&self) -> &str {
        match self {
            Self::CsvField { artifact, .. }
            | Self::KeyValue { artifact, .. }
            | Self::CountLinesContaining { artifact, .. }
            | Self::LineContaining { artifact, .. }
            | Self::JsonPointer { artifact, .. }
            | Self::FlightEventStatus { artifact, .. }
            | Self::FlightEventField { artifact, .. } => artifact,
        }
    }

    /// Re-derive the value from the artifact as it exists NOW.
    ///
    /// `Err` means the artifact could not be read or the op is malformed —
    /// i.e. the claim is UNGROUNDED. A successfully-read artifact that does
    /// not contain the thing yields `Ok(Value::Absent)`.
    pub fn run(&self, root: &Path) -> Result<Extracted> {
        let path = resolve(root, self.artifact())?;
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read cited artifact {}", path.display()))?;
        let artifact = self.artifact().to_string();
        match self {
            Self::CsvField {
                row_contains,
                column,
                ..
            } => Ok(csv_field(&artifact, &text, row_contains, *column)),
            Self::KeyValue {
                key,
                in_lines_containing,
                occurrence,
                ..
            } => Ok(key_value(
                &artifact,
                &text,
                key,
                in_lines_containing.as_deref(),
                *occurrence,
            )),
            Self::CountLinesContaining { needle, .. } => {
                let count = text.lines().filter(|line| line.contains(needle)).count();
                Ok(Extracted {
                    value: Value::Num(count as f64),
                    citation: Citation {
                        artifact,
                        line: None,
                        detail: format!("count of lines containing {needle:?}"),
                    },
                })
            }
            Self::LineContaining {
                needle, occurrence, ..
            } => Ok(line_containing(&artifact, &text, needle, *occurrence)),
            Self::JsonPointer { pointer, .. } => json_pointer(&artifact, &text, pointer),
            Self::FlightEventStatus { method, .. } => flight_event(&artifact, &text, method, None),
            Self::FlightEventField { method, key, .. } => {
                flight_event(&artifact, &text, method, Some(key))
            }
        }
    }
}

/// Reject absolute paths and `..` escapes: a claim must cite something inside
/// the repo, or nobody else can re-check it.
fn resolve(root: &Path, relative: &str) -> Result<PathBuf> {
    let candidate = Path::new(relative);
    if candidate.is_absolute() {
        return Err(anyhow!(
            "artifact path must be repo-relative, got the absolute path {relative:?}; \
             a claim citing a path outside the repo cannot be re-checked"
        ));
    }
    if candidate
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(anyhow!(
            "artifact path must stay inside the repo, got {relative:?}"
        ));
    }
    Ok(root.join(candidate))
}

fn typed(text: &str) -> Value {
    match text.trim().parse::<f64>() {
        Ok(number) => Value::Num(number),
        Err(_) => Value::Str(text.trim().to_string()),
    }
}

fn csv_field(artifact: &str, text: &str, row_contains: &str, column: usize) -> Extracted {
    for (index, line) in text.lines().enumerate() {
        if !line.contains(row_contains) {
            continue;
        }
        let value = line.split(',').nth(column).map_or(Value::Absent, typed);
        return Extracted {
            value,
            citation: Citation {
                artifact: artifact.to_string(),
                line: Some(index + 1),
                detail: format!("row containing {row_contains:?}, column {column}"),
            },
        };
    }
    Extracted {
        value: Value::Absent,
        citation: Citation {
            artifact: artifact.to_string(),
            line: None,
            detail: format!("no row contains {row_contains:?}"),
        },
    }
}

/// Find every `key=value` token. The value ends at whitespace, `,`, `]`, `)`
/// or `;` — the delimiters the workspace's own log lines actually use
/// (`route=Historical solver_timeout=456s`, `gpu_lines=0,`, `[features=mip ...]`).
fn key_hits(text: &str, key: &str, in_lines_containing: Option<&str>) -> Vec<(usize, String)> {
    let needle = format!("{key}=");
    let mut hits = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if in_lines_containing.is_some_and(|filter| !line.contains(filter)) {
            continue;
        }
        let mut cursor = 0;
        while let Some(found) = line[cursor..].find(&needle) {
            let start = cursor + found;
            // Only a token START counts: `root_bound=` must not match inside
            // `child_root_bound=`.
            let boundary_ok = start == 0
                || !line.as_bytes()[start - 1].is_ascii_alphanumeric()
                    && line.as_bytes()[start - 1] != b'_';
            let value_start = start + needle.len();
            if boundary_ok {
                let value: String = line[value_start..]
                    .chars()
                    .take_while(|character| {
                        !character.is_whitespace() && !matches!(character, ',' | ']' | ')' | ';')
                    })
                    .collect();
                hits.push((index + 1, value));
            }
            cursor = value_start;
        }
    }
    hits
}

fn key_value(
    artifact: &str,
    text: &str,
    key: &str,
    in_lines_containing: Option<&str>,
    occurrence: Occurrence,
) -> Extracted {
    let hits = key_hits(text, key, in_lines_containing);
    let scope = in_lines_containing
        .map(|filter| format!(" in lines containing {filter:?}"))
        .unwrap_or_default();
    aggregate(
        artifact,
        hits,
        occurrence,
        &format!("key={key}{scope} occurrence={occurrence:?}"),
    )
}

fn line_containing(artifact: &str, text: &str, needle: &str, occurrence: Occurrence) -> Extracted {
    let hits: Vec<(usize, String)> = text
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains(needle))
        .map(|(index, line)| (index + 1, line.trim().to_string()))
        .collect();
    aggregate(
        artifact,
        hits,
        occurrence,
        &format!("line containing {needle:?} occurrence={occurrence:?}"),
    )
}

fn aggregate(
    artifact: &str,
    hits: Vec<(usize, String)>,
    occurrence: Occurrence,
    detail: &str,
) -> Extracted {
    let cite = |line: Option<usize>, detail: String| Citation {
        artifact: artifact.to_string(),
        line,
        detail,
    };
    if matches!(occurrence, Occurrence::Count) {
        return Extracted {
            value: Value::Num(hits.len() as f64),
            citation: cite(None, format!("{detail} -> {} occurrences", hits.len())),
        };
    }
    if hits.is_empty() {
        return Extracted {
            value: Value::Absent,
            citation: cite(None, format!("{detail} -> no occurrence")),
        };
    }
    match occurrence {
        Occurrence::Unique => {
            let distinct: std::collections::BTreeSet<&String> =
                hits.iter().map(|(_, value)| value).collect();
            if distinct.len() > 1 {
                // Deliberately Absent, not "the first one". A key expected to
                // be unique that is not unique means the artifact does not say
                // what the claim thinks it says.
                return Extracted {
                    value: Value::Absent,
                    citation: cite(
                        Some(hits[0].0),
                        format!(
                            "{detail} -> {} conflicting values {:?}; occurrence=unique refuses to pick",
                            distinct.len(),
                            distinct
                        ),
                    ),
                };
            }
            Extracted {
                value: typed(&hits[0].1),
                citation: cite(Some(hits[0].0), format!("{detail} ({} hits)", hits.len())),
            }
        }
        Occurrence::First => Extracted {
            value: typed(&hits[0].1),
            citation: cite(Some(hits[0].0), detail.to_string()),
        },
        Occurrence::Last => {
            let last = hits.last().expect("non-empty");
            Extracted {
                value: typed(&last.1),
                citation: cite(Some(last.0), detail.to_string()),
            }
        }
        Occurrence::Max | Occurrence::Min => {
            let mut best: Option<(usize, f64)> = None;
            for (line, raw) in &hits {
                let Ok(number) = raw.trim().parse::<f64>() else {
                    return Extracted {
                        value: Value::Absent,
                        citation: cite(
                            Some(*line),
                            format!("{detail} -> non-numeric value {raw:?}"),
                        ),
                    };
                };
                let better = match (best, occurrence) {
                    (None, _) => true,
                    (Some((_, current)), Occurrence::Max) => number > current,
                    (Some((_, current)), _) => number < current,
                };
                if better {
                    best = Some((*line, number));
                }
            }
            let (line, number) = best.expect("non-empty");
            Extracted {
                value: Value::Num(number),
                citation: cite(Some(line), format!("{detail} over {} hits", hits.len())),
            }
        }
        Occurrence::Count => unreachable!("handled above"),
    }
}

fn json_pointer(artifact: &str, text: &str, pointer: &str) -> Result<Extracted> {
    let document: serde_json::Value = serde_json::from_str(text)
        .with_context(|| format!("parse {artifact} as JSON for pointer {pointer}"))?;
    let value = match document.pointer(pointer) {
        None => Value::Absent,
        Some(serde_json::Value::Number(number)) => Value::Num(number.as_f64().unwrap_or(f64::NAN)),
        Some(serde_json::Value::String(text)) => Value::Str(text.clone()),
        Some(serde_json::Value::Bool(flag)) => Value::Str(flag.to_string()),
        Some(serde_json::Value::Null) => Value::Absent,
        Some(other) => Value::Str(other.to_string()),
    };
    Ok(Extracted {
        value,
        citation: Citation {
            artifact: artifact.to_string(),
            line: None,
            detail: format!("json pointer {pointer}"),
        },
    })
}

/// Read a flight-recorder event by method name: its `status`, or a `key=value`
/// token out of its `reason`.
fn flight_event(artifact: &str, text: &str, method: &str, key: Option<&str>) -> Result<Extracted> {
    let document: serde_json::Value = serde_json::from_str(text)
        .with_context(|| format!("parse {artifact} as a flight receipt"))?;
    let events = document
        .get("events")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("{artifact} has no `events` array; not a flight receipt"))?;
    let detail = match key {
        Some(key) => format!("flight event {method:?} field {key}="),
        None => format!("flight event {method:?} status"),
    };
    let absent = |detail: String| Extracted {
        value: Value::Absent,
        citation: Citation {
            artifact: artifact.to_string(),
            line: None,
            detail,
        },
    };
    let Some(event) = events
        .iter()
        .find(|event| event.get("method").and_then(serde_json::Value::as_str) == Some(method))
    else {
        // An event that is not in the receipt is the honest ABSENT: the
        // recorder is documented to omit lanes that ran and lost, so a caller
        // must never read this as "the lane did not run".
        return Ok(absent(format!("{detail} -> no such event in the receipt")));
    };
    let Some(key) = key else {
        let status = event
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        return Ok(Extracted {
            value: Value::Str(status.to_string()),
            citation: Citation {
                artifact: artifact.to_string(),
                line: None,
                detail,
            },
        });
    };
    let reason = event
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let hits = key_hits(reason, key, None);
    if hits.is_empty() {
        return Ok(absent(format!(
            "{detail} -> not present in the event reason"
        )));
    }
    Ok(Extracted {
        value: typed(&hits[0].1),
        citation: Citation {
            artifact: artifact.to_string(),
            line: None,
            detail,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) -> String {
        std::fs::write(dir.join(name), body).expect("write fixture");
        name.to_string()
    }

    #[test]
    fn a_csv_row_is_selected_by_content_and_cited_by_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = write(
            dir.path(),
            "bank.csv",
            "b,onnx,vnnlib/idx_1.vnnlib,0,timeout,98.1,tag\n\
             b,onnx,vnnlib/idx_2.vnnlib,0,unsat,45.0,tag\n",
        );
        let extracted = Extractor::CsvField {
            artifact,
            row_contains: "idx_2".to_string(),
            column: 4,
        }
        .run(dir.path())
        .expect("extract");
        assert_eq!(extracted.value, Value::Str("unsat".to_string()));
        assert_eq!(extracted.citation.line, Some(2));
    }

    #[test]
    fn a_missing_row_is_absent_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = write(dir.path(), "bank.csv", "b,onnx,v,0,unsat,1.0,tag\n");
        let extracted = Extractor::CsvField {
            artifact,
            row_contains: "idx_nope".to_string(),
            column: 4,
        }
        .run(dir.path())
        .expect("a readable artifact must not error");
        assert!(extracted.value.is_absent());
    }

    /// The engagement-witness aggregation that reversal 5 needed: nine runs,
    /// every one reporting `gpu_lines=0`, must reduce to 0 and not to "well,
    /// some line mentioned gpu_lines".
    #[test]
    fn max_over_repetitions_is_the_engagement_aggregation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = write(
            dir.path(),
            "results.csv",
            "row,r1,unsat,expansions=21,root_bound=-0.267429,gpu_lines=0,,kill=0\n\
             row,r2,unsat,expansions=21,root_bound=-0.267429,gpu_lines=0,,kill=0\n",
        );
        let extracted = Extractor::KeyValue {
            artifact: artifact.clone(),
            key: "gpu_lines".to_string(),
            in_lines_containing: None,
            occurrence: Occurrence::Max,
        }
        .run(dir.path())
        .expect("extract");
        assert_eq!(extracted.value, Value::Num(0.0));

        let counted = Extractor::KeyValue {
            artifact,
            key: "gpu_lines".to_string(),
            in_lines_containing: None,
            occurrence: Occurrence::Count,
        }
        .run(dir.path())
        .expect("extract");
        assert_eq!(counted.value, Value::Num(2.0));
    }

    /// A key expected once that appears twice with different values is a
    /// defect in the artifact, and `unique` must refuse rather than pick.
    #[test]
    fn unique_refuses_to_pick_between_conflicting_values() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = write(dir.path(), "log.txt", "route=A\nroute=B\n");
        let extracted = Extractor::KeyValue {
            artifact,
            key: "route".to_string(),
            in_lines_containing: None,
            occurrence: Occurrence::Unique,
        }
        .run(dir.path())
        .expect("extract");
        assert!(extracted.value.is_absent(), "{extracted:?}");
        assert!(extracted.citation.detail.contains("conflicting"));
    }

    /// `root_bound=` must not match inside `child_root_bound=`.
    #[test]
    fn a_key_matches_only_at_a_token_boundary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = write(
            dir.path(),
            "log.txt",
            "child_root_bound=-9.0 root_bound=-0.17\n",
        );
        let extracted = Extractor::KeyValue {
            artifact,
            key: "root_bound".to_string(),
            in_lines_containing: None,
            occurrence: Occurrence::Unique,
        }
        .run(dir.path())
        .expect("extract");
        assert_eq!(extracted.value, Value::Num(-0.17));
    }

    #[test]
    fn an_absolute_path_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let error = Extractor::KeyValue {
            artifact: "/etc/hostname".to_string(),
            key: "x".to_string(),
            in_lines_containing: None,
            occurrence: Occurrence::First,
        }
        .run(dir.path())
        .expect_err("absolute paths must be refused");
        assert!(format!("{error:#}").contains("repo-relative"), "{error:#}");
    }

    #[test]
    fn a_flight_event_field_reads_the_route_marker() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = write(
            dir.path(),
            "row.flight.json",
            r#"{"events":[{"method":"internal_timeout_route","status":"ran",
                "reason":"route=Historical solver_timeout=456s scored_budget=480s"}]}"#,
        );
        let extracted = Extractor::FlightEventField {
            artifact: artifact.clone(),
            method: "internal_timeout_route".to_string(),
            key: "route".to_string(),
        }
        .run(dir.path())
        .expect("extract");
        assert_eq!(extracted.value, Value::Str("Historical".to_string()));

        let missing = Extractor::FlightEventField {
            artifact,
            method: "a_lane_that_never_logged".to_string(),
            key: "route".to_string(),
        }
        .run(dir.path())
        .expect("extract");
        assert!(missing.value.is_absent());
    }
}

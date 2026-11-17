// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Pinned 2026 ToolResult timeout policy for supplied organizer field rows.

use super::*;

#[derive(Debug, Default, Serialize)]
pub(crate) struct FieldTimeoutNormalization {
    pub(crate) tools: BTreeMap<String, ToolTimeoutNormalization>,
    pub(crate) forced_timeouts: usize,
    pub(crate) removed_categories: usize,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct ToolTimeoutNormalization {
    source_csv: String,
    source_csv_sha256: String,
    startup_overhead_seconds: f64,
    forced_timeouts: usize,
    removed_categories: Vec<String>,
}

/// `single_overhead=True` in pinned process_results.py: CPU category lists are
/// empty, so every raw decided row contributes to one minimum per tool. This
/// scan deliberately precedes harness and track/category filtering.
fn overhead(path: &Path) -> Result<(f64, String)> {
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("read organizer aggregate {}", path.display()))?;
    let mut minimum: Option<f64> = None;
    for (offset, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = parse_csv_line(line)?;
        if is_results_header(&fields) {
            continue;
        }
        if CsvSchema::for_fields(&fields, path, offset + 1)? != CsvSchema::Official6 {
            bail!(
                "{}: field timeout normalization requires six-column organizer rows",
                path.display()
            );
        }
        if Verdict::parse(&fields[4])?.solved() {
            let seconds = validate_nonnegative_finite(&fields[5], "run_time", path, offset + 1)?;
            minimum = Some(minimum.map_or(seconds, |previous| previous.min(seconds)));
        }
    }
    Ok((
        minimum.unwrap_or(0.0),
        format!("{:x}", Sha256::digest(body.as_bytes())),
    ))
}

fn validate_network_version(key: &InstanceKey, entry: &RecordedVerdict) -> Result<()> {
    let mut version: Option<String> = None;
    for (_, path) in model_identity_entries(&entry.network_field)? {
        let parts = path.split('/').collect::<Vec<_>>();
        let current = parts
            .windows(3)
            .find_map(|parts| {
                (parts[0] == "benchmarks"
                    && parts[1] == key.benchmark
                    && matches!(parts[2], "1.0" | "2.0"))
                .then(|| parts[2].to_owned())
            })
            .with_context(|| {
                format!(
            "{}: organizer field normalization needs benchmarks/{}/<1.0|2.0>/ network paths",
            entry.source, key.benchmark
        )
            })?;
        if version
            .as_ref()
            .is_some_and(|previous| previous != &current)
        {
            bail!("{}: relational networks mix VNN-LIB versions", entry.source);
        }
        version = Some(current);
    }
    let version = version.context("empty organizer network field")?;
    if entry.vnnlib_version.as_ref() != Some(&version) {
        bail!(
            "{}: organizer network/property versions disagree",
            entry.source
        );
    }
    Ok(())
}

fn same_aggregate_rows(aggregate: &ResultRows, supplied: &ResultRows) -> bool {
    aggregate.len() == supplied.len()
        && aggregate.iter().all(|(key, row)| {
            supplied.get(key).is_some_and(|other| {
                row.verdict == other.verdict
                    && row.seconds == other.seconds
                    && row.prepare_seconds == other.prepare_seconds
                    && row.network_field == other.network_field
                    && row.property_field == other.property_field
                    && row.organizer_row_identity == other.organizer_row_identity
            })
        })
}

/// Version-specific source paths are required even though this caller's budget
/// table already refuses conflicting budgets across matching versioned lists.
/// Cross-tool positional equivalence and payload provenance are separate gates.
pub(super) fn normalize(
    root: &Path,
    field: &mut OfficialField,
    budgets: &BudgetTable,
) -> Result<FieldTimeoutNormalization> {
    let mut report = FieldTimeoutNormalization::default();
    for (tool, rows) in field.iter_mut() {
        // This is the pinned scorer's ../*/results.csv glob. A partial set of
        // per-category CSVs cannot establish that tool's global minimum.
        let source = root.join(tool).join("results.csv");
        let metadata = std::fs::symlink_metadata(&source).with_context(|| {
            format!(
                "field timeout normalization requires complete organizer aggregate {}",
                source.display()
            )
        })?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            bail!(
                "organizer aggregate is not a regular file: {}",
                source.display()
            );
        }
        let (startup, source_sha256) = overhead(&source)?;
        let aggregate = read_results(&source)?;
        if format!("{:x}", Sha256::digest(std::fs::read(&source)?)) != source_sha256 {
            bail!(
                "organizer aggregate changed while reading: {}",
                source.display()
            );
        }
        if !same_aggregate_rows(&aggregate, rows) {
            bail!(
                "{}: supplied field differs from its complete organizer aggregate",
                source.display()
            );
        }
        let mut detail = ToolTimeoutNormalization {
            source_csv: source.display().to_string(),
            source_csv_sha256: source_sha256,
            startup_overhead_seconds: startup,
            ..Default::default()
        };
        for (key, entry) in rows.iter_mut() {
            validate_network_version(key, entry)?;
            let budget = budgets.get(&key.semantic_key()).with_context(|| {
                format!(
                    "{}: organizer field identity has no original instance budget",
                    entry.source
                )
            })?;
            // Preserve raw time and the original six-field check identity.
            // Only the in-memory scoring verdict/authority changes.
            if entry.verdict.solved() && entry.seconds - startup > *budget as f64 {
                entry.verdict = Verdict::Undecided;
                entry.counterexample = None;
                detail.forced_timeouts += 1;
            }
        }
        let categories = rows
            .keys()
            .map(|key| key.benchmark.clone())
            .collect::<BTreeSet<_>>();
        let decided = rows
            .iter()
            .filter(|(_, entry)| entry.verdict.solved())
            .map(|(key, _)| key.benchmark.clone())
            .collect::<BTreeSet<_>>();
        detail.removed_categories = categories.difference(&decided).cloned().collect();
        rows.retain(|key, _| decided.contains(&key.benchmark));
        report.forced_timeouts += detail.forced_timeouts;
        report.removed_categories += detail.removed_categories.len();
        report.tools.insert(tool.clone(), detail);
    }
    field.retain(|_, rows| !rows.is_empty());
    Ok(report)
}

#[cfg(test)]
#[path = "vnncomp_score_field_timeouts_tests.rs"]
mod tests;

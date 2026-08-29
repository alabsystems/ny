// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny claims …` — shared by the `ny` subcommand and the standalone
//! `ny-claims` binary, which are the same code for a reason: the check that
//! runs in a git hook and the check that runs in CI must not be able to
//! disagree.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

use crate::ledger::{lint_all, load_dir};
use crate::verify::{evaluate, tally, Outcome, Status};
use crate::world::{BinaryFacts, World};

/// Default ledger location: a top-level directory, deliberately not under
/// `docs/`. A claim is not documentation.
pub const DEFAULT_LEDGER: &str = "claims";

#[derive(Debug, Clone, Args)]
pub struct ClaimsArgs {
    /// Repo root. Artifact paths are resolved against it.
    #[arg(long, global = true)]
    pub root: Option<PathBuf>,
    /// Ledger directory, relative to the root.
    #[arg(long, global = true, default_value = DEFAULT_LEDGER)]
    pub ledger: PathBuf,
    #[command(subcommand)]
    pub action: ClaimsAction,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ClaimsAction {
    /// Re-derive every claim from its artifacts and print TRUE / FALSE /
    /// STALE / VACUOUS / UNGROUNDED. Exits non-zero unless every claim is TRUE.
    Verify {
        /// Only this claim id.
        #[arg(long)]
        id: Option<String>,
        #[arg(long)]
        json: bool,
        /// Treat "no verifier binary to compare against" as a failure rather
        /// than as silence. For CI, where an unbuilt checkout is itself a bug.
        #[arg(long)]
        require_binary: bool,
        /// Print the full receipt for every claim, not only the failing ones.
        #[arg(long)]
        verbose: bool,
    },
    /// Admissibility only: is every record a well-formed claim? Fast, no
    /// artifact reads. This is the pre-commit gate.
    Lint,
    /// Recompute the sha256 pins of a claim's cited artifacts and rewrite the
    /// record. Run this when you have DELIBERATELY re-measured, never to make
    /// a red claim go green.
    Pin { id: String },
    /// Print one claim's record and its receipt.
    Show { id: String },
    /// Scan prose for `[[claim:<id>]]` references and report each one's
    /// current status, so a doc that cites a STALE claim is findable without
    /// reading the doc.
    Refs {
        /// Directories to scan, relative to the root.
        #[arg(default_values_t = [String::from("docs")])]
        paths: Vec<String>,
    },
}

/// Entry point. `binary` is the caller's best evidence about the current
/// verifier: the in-`ny` host passes its own running stamp, the standalone
/// binary passes `None` and lets [`World::probe`] ask `target/release/ny`.
pub fn run(args: &ClaimsArgs, binary: Option<BinaryFacts>) -> Result<i32> {
    let root = match &args.root {
        Some(root) => root.clone(),
        None => discover_root()?,
    };
    let ledger = root.join(&args.ledger);
    let world = match binary {
        Some(facts) => World::with_binary(root.clone(), Some(facts)),
        None => World::probe(root.clone()),
    };

    match &args.action {
        ClaimsAction::Lint => {
            let loaded = load_dir(&ledger)?;
            let findings = lint_all(&loaded);
            if findings.is_empty() {
                println!("claims lint: {} records admissible", loaded.len());
                return Ok(0);
            }
            for finding in &findings {
                println!("INADMISSIBLE {}: {}", finding.id, finding.message);
            }
            println!("\n{} inadmissible record(s)", findings.len());
            Ok(1)
        }
        ClaimsAction::Verify {
            id,
            json,
            require_binary,
            verbose,
        } => {
            let loaded = load_dir(&ledger)?;
            let findings = lint_all(&loaded);
            let selected: Vec<_> = loaded
                .iter()
                .filter(|entry| id.as_ref().is_none_or(|id| entry.claim.id == *id))
                .collect();
            if selected.is_empty() {
                anyhow::bail!("no claim matched {id:?} in {}", ledger.display());
            }
            if *require_binary && world.binary.is_none() {
                anyhow::bail!(
                    "--require-binary: {}/target/release/ny could not identify itself; \
                     every binary-sensitive claim is unverifiable",
                    root.display()
                );
            }
            let outcomes: Vec<Outcome> = selected
                .iter()
                .map(|entry| evaluate(&entry.claim, &world))
                .collect();
            if *json {
                println!("{}", serde_json::to_string_pretty(&outcomes)?);
            } else {
                render(&world, &outcomes, &findings, *verbose);
            }
            let bad = outcomes.iter().filter(|o| !o.status.is_ok()).count() + findings.len();
            Ok(i32::from(bad > 0))
        }
        ClaimsAction::Show { id } => {
            let loaded = load_dir(&ledger)?;
            let entry = loaded
                .iter()
                .find(|entry| entry.claim.id == *id)
                .with_context(|| format!("no claim {id:?} in {}", ledger.display()))?;
            println!("{}", serde_json::to_string_pretty(&entry.claim)?);
            let outcome = evaluate(&entry.claim, &world);
            println!("\n{}", render_one(&outcome, true));
            Ok(i32::from(!outcome.status.is_ok()))
        }
        ClaimsAction::Pin { id } => {
            let loaded = load_dir(&ledger)?;
            let entry = loaded
                .iter()
                .find(|entry| entry.claim.id == *id)
                .with_context(|| format!("no claim {id:?} in {}", ledger.display()))?;
            let mut claim = entry.claim.clone();
            for artifact in &mut claim.artifacts {
                let found = crate::verify::hash_file(&root.join(&artifact.path))?;
                if found != artifact.sha256 {
                    println!("{}: {} -> {}", artifact.path, artifact.sha256, found);
                    artifact.sha256 = found;
                }
            }
            std::fs::write(&entry.path, serde_json::to_string_pretty(&claim)? + "\n")?;
            println!("pinned {}", entry.path.display());
            Ok(0)
        }
        ClaimsAction::Refs { paths } => {
            let loaded = load_dir(&ledger)?;
            let mut exit = 0;
            let mut seen = 0_usize;
            for relative in paths {
                for (file, line_number, id) in scan_refs(&root.join(relative))? {
                    seen += 1;
                    let display = file
                        .strip_prefix(&root)
                        .unwrap_or(&file)
                        .display()
                        .to_string();
                    match loaded.iter().find(|entry| entry.claim.id == id) {
                        None => {
                            println!("DANGLING   {display}:{line_number} -> [[claim:{id}]]");
                            exit = 1;
                        }
                        Some(entry) => {
                            let outcome = evaluate(&entry.claim, &world);
                            println!(
                                "{:<10} {display}:{line_number} -> {id}",
                                outcome.status.label()
                            );
                            if !outcome.status.is_ok() {
                                exit = 1;
                            }
                        }
                    }
                }
            }
            println!("\n{seen} claim reference(s) scanned");
            Ok(exit)
        }
    }
}

fn discover_root() -> Result<PathBuf> {
    let mut current = std::env::current_dir()?;
    loop {
        if current.join(".git").exists() && current.join("Cargo.toml").exists() {
            return Ok(current);
        }
        if !current.pop() {
            anyhow::bail!("could not find the repo root above the current directory; pass --root");
        }
    }
}

/// Find `[[claim:<id>]]` in every text file under `dir`.
fn scan_refs(dir: &Path) -> Result<Vec<(PathBuf, usize, String)>> {
    let mut hits = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        if current.is_dir() {
            for entry in std::fs::read_dir(&current)? {
                stack.push(entry?.path());
            }
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&current) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            let mut cursor = 0;
            while let Some(found) = line[cursor..].find("[[claim:") {
                let start = cursor + found + "[[claim:".len();
                let Some(end) = line[start..].find("]]") else {
                    break;
                };
                hits.push((
                    current.clone(),
                    index + 1,
                    line[start..start + end].to_string(),
                ));
                cursor = start + end;
            }
        }
    }
    hits.sort();
    Ok(hits)
}

fn render(
    world: &World,
    outcomes: &[Outcome],
    findings: &[crate::ledger::LintFinding],
    verbose: bool,
) {
    println!("ny claims verify");
    println!("  root    {}", world.root.display());
    println!("  host    {}", world.host);
    match &world.binary {
        Some(facts) => println!(
            "  binary  {} stamp={} [{}]",
            facts.version,
            &facts.stamp[..16.min(facts.stamp.len())],
            facts.source
        ),
        None => println!("  binary  <not built or cannot self-identify>"),
    }
    println!();

    for finding in findings {
        println!("INADMISSIBLE {}: {}", finding.id, finding.message);
    }
    if !findings.is_empty() {
        println!();
    }

    for outcome in outcomes {
        println!(
            "{}",
            render_one(outcome, verbose || !outcome.status.is_ok())
        );
    }

    let counts = tally(outcomes);
    let summary = ["TRUE", "FALSE", "STALE", "VACUOUS", "UNGROUNDED"]
        .iter()
        .map(|label| format!("{label} {}", counts.get(label).copied().unwrap_or(0)))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{summary}   ({} claims)", outcomes.len());
}

fn render_one(outcome: &Outcome, detail: bool) -> String {
    let mut out = format!("{:<10} {}\n", outcome.status.label(), outcome.id);
    out.push_str(&format!("           {}\n", outcome.assertion));
    if !detail {
        return out;
    }
    out.push_str(&format!("           why: {}\n", outcome.headline));
    if let (Some(observed), Some(citation)) = (&outcome.observed, &outcome.citation) {
        out.push_str(&format!("           observed {observed} from {citation}\n"));
    }
    for witness in &outcome.witnesses {
        out.push_str(&format!(
            "           witness {} {}: {} at {}\n",
            if witness.holds { "OK  " } else { "FAIL" },
            witness.name,
            witness.rendered,
            witness.citation
        ));
        if !witness.why.is_empty() {
            out.push_str(&format!("                   ({})\n", witness.why));
        }
    }
    for drift in &outcome.drifts {
        out.push_str(&format!("           drift {}\n", drift.render()));
    }
    if outcome.status == Status::Stale || outcome.provenance_grade != "intrinsic" {
        out.push_str(&format!(
            "           provenance {}\n",
            outcome.provenance_grade
        ));
    }
    if let Some(reason) = &outcome.retracted {
        out.push_str(&format!("           RETRACTED: {reason}\n"));
    }
    out
}

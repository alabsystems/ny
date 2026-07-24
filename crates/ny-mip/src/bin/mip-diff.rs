// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Backend differential + certificate harness over a dumped `.milp` corpus
// (gates G0/LG0/LG3).
//
// Usage:
//   NY_MIP_DUMP=corpus/ ny beta-crown ...       # capture production instances
//   mip-diff corpus/*.milp                      # default: ay (lib) vs ay-proc
//   mip-diff --timeout 60 corpus/               # a directory expands to *.milp
//   mip-diff --certify corpus/                  # LG3: ay vs its own certificates
//
// Diff mode: solves with two backends and compares verdicts. Sat-vs-Unsat
// between backends is a DISAGREEMENT (exit 1) — one of the solvers is
// wrong about a bit-identical problem. Timeout/Error on either side is
// recorded but is not a disagreement. Wall time per backend feeds the
// baseline ledger (docs/AY_MIP_P0.md).
//
// Certify mode (LG3, replaces the deleted HiGHS oracle): solves with the
// production ay backend and holds every verdict to its own evidence — an
// UNSAT must carry a VERIFIED exact certificate (Farkas or case-split,
// checked at the seam), a SAT witness is re-checked downstream anyway.
// Reports certification coverage; exits 1 on any hard failure: a
// certificate that FAILED verification (the seam surfaces it as an error,
// never as a bare UNSAT), a solver error, or an unloadable file.
// Certificate ABSENCE is reported, not fatal: some exact-but-
// uncertifiable trees exist until P4 completes the factory.
//
// Backends: `ay` (in-process ay-milp library, the production default),
// `ay-proc` (frozen P0 subprocess lane, needs the `ay` binary via
// $NY_AY/$PATH).

use ny_mip::{dump, MilpProblem, MipBackend, MipConfig, MipResult, MipSolver};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage: mip-diff [--timeout <secs>] [--backends <a>,<b>] [--certify] <file.milp | dir>...\n\
         backends: ay | ay-proc (default: ay,ay-proc)"
    );
    std::process::exit(2)
}

fn parse_backend(name: &str) -> MipBackend {
    match name {
        "ay" => MipBackend::Ay,
        "ay-proc" => MipBackend::AyProc,
        _ => usage(),
    }
}

fn backend_name(b: MipBackend) -> &'static str {
    match b {
        MipBackend::Ay => "ay",
        MipBackend::AyProc => "ay-proc",
    }
}

fn main() {
    let mut timeout_secs = 300.0_f64;
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut pair = (MipBackend::Ay, MipBackend::AyProc);
    let mut certify = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--timeout" => {
                let Some(v) = args.next().and_then(|s| s.parse().ok()) else {
                    usage()
                };
                timeout_secs = v;
            }
            "--certify" => certify = true,
            "--backends" => {
                let Some(v) = args.next() else { usage() };
                let mut parts = v.splitn(2, ',');
                let (Some(a), Some(b)) = (parts.next(), parts.next()) else {
                    usage()
                };
                pair = (parse_backend(a), parse_backend(b));
            }
            "--help" | "-h" => usage(),
            _ => inputs.push(PathBuf::from(arg)),
        }
    }
    if inputs.is_empty() {
        usage();
    }

    let files = collect_files(&inputs);
    if files.is_empty() {
        eprintln!("mip-diff: no .milp files found");
        std::process::exit(2);
    }

    if certify {
        run_certify(&files, timeout_secs);
    }

    let (left, right) = pair;
    println!(
        "{:<40} {:>6} {:>6}  {:>9} {:>9}  {:>9} {:>9}",
        "instance",
        "cols",
        "rows",
        backend_name(left),
        "t(s)",
        backend_name(right),
        "t(s)"
    );
    let mut disagreements = 0usize;
    for path in &files {
        match run_one(path, timeout_secs, pair) {
            Ok(disagreed) => disagreements += usize::from(disagreed),
            Err(e) => {
                println!("{:<40} load error: {e}", short_name(path));
                disagreements += 1; // an unloadable corpus file fails the gate
            }
        }
    }
    println!(
        "\n{} instance(s), {} disagreement(s)",
        files.len(),
        disagreements
    );
    if disagreements > 0 {
        std::process::exit(1);
    }
}

/// LG3 certify mode: every ay UNSAT must carry verified evidence.
fn run_certify(files: &[PathBuf], timeout_secs: f64) -> ! {
    let mut sat = 0usize;
    let mut unsat_certified = 0usize;
    let mut unsat_bare = 0usize;
    let mut inconclusive = 0usize;
    let mut failures = 0usize;
    println!("{:<40} {:>10}", "instance", "verdict");
    for path in files {
        let (verdict, detail) = match std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|text| dump::from_milp_text(&text).map_err(|e| e.to_string()))
        {
            Ok(problem) => {
                let (result, _) = solve(&problem, MipBackend::Ay, timeout_secs);
                match result {
                    MipResult::Sat { .. } => {
                        sat += 1;
                        ("sat", None)
                    }
                    MipResult::Unsat { certified: true } => {
                        unsat_certified += 1;
                        ("unsat+cert", None)
                    }
                    MipResult::Unsat { certified: false } => {
                        unsat_bare += 1;
                        ("unsat", None)
                    }
                    MipResult::Timeout => {
                        inconclusive += 1;
                        ("timeout", None)
                    }
                    MipResult::Error(e) => {
                        failures += 1;
                        ("error", Some(e))
                    }
                }
            }
            Err(e) => {
                failures += 1;
                ("load-error", Some(e))
            }
        };
        match detail {
            Some(detail) => println!("{:<40} {verdict:>10}  {detail}", short_name(path)),
            None => println!("{:<40} {verdict:>10}", short_name(path)),
        }
    }
    println!(
        "\n{} instance(s): {sat} sat, {unsat_certified} unsat certified, \
         {unsat_bare} unsat bare, {inconclusive} inconclusive, {failures} failures",
        files.len()
    );
    // A certificate that fails verification arrives from the seam as
    // `MipResult::Error` (ay_lib::map_outcome), so it lands in `failures`
    // with the other hard errors; only certificate ABSENCE degrades to
    // `unsat bare`.
    std::process::exit(if failures > 0 { 1 } else { 0 })
}

fn run_one(
    path: &Path,
    timeout_secs: f64,
    (left, right): (MipBackend, MipBackend),
) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let problem = dump::from_milp_text(&text).map_err(|e| e.to_string())?;
    let (cols, rows) = (problem.num_cols(), problem.num_rows());

    let (lres, t_left) = solve(&problem, left, timeout_secs);
    let (rres, t_right) = solve(&problem, right, timeout_secs);

    let disagreed = matches!(
        (&lres, &rres),
        (MipResult::Sat { .. }, MipResult::Unsat { .. })
            | (MipResult::Unsat { .. }, MipResult::Sat { .. })
    );
    println!(
        "{:<40} {cols:>6} {rows:>6}  {:>9} {t_left:>9.3}  {:>9} {t_right:>9.3}{}",
        short_name(path),
        verdict(&lres),
        verdict(&rres),
        if disagreed { "  <-- DISAGREEMENT" } else { "" }
    );
    Ok(disagreed)
}

fn solve(problem: &MilpProblem, backend: MipBackend, timeout_secs: f64) -> (MipResult, f64) {
    let parts = ny_mip::MipParts {
        problem: problem.clone(),
        input_vars: vec![],
        output_vars: vec![],
        binary_vars: vec![],
        binary_widths: vec![],
        num_cols: problem.num_cols(),
    };
    let config = MipConfig {
        backend,
        parallel_split: 1, // serial: compare raw solver strength, not racing
        timeout_secs,
        ..MipConfig::default()
    };
    let start = Instant::now();
    let result = MipSolver::new(parts, config)
        .check_feasibility()
        .unwrap_or_else(|e| MipResult::Error(e.to_string()));
    (result, start.elapsed().as_secs_f64())
}

fn verdict(r: &MipResult) -> &'static str {
    match r {
        MipResult::Sat { .. } => "sat",
        MipResult::Unsat { certified: true } => "unsat+cert",
        MipResult::Unsat { certified: false } => "unsat",
        MipResult::Timeout => "timeout",
        MipResult::Error(_) => "error",
    }
}

fn collect_files(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let Ok(entries) = std::fs::read_dir(input) else {
                continue;
            };
            let mut batch: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "milp"))
                .collect();
            batch.sort();
            files.extend(batch);
        } else {
            files.push(input.clone());
        }
    }
    files
}

fn short_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Feasible-incumbent WCNF frontend; NY's pinned in-process MIP engine only.
#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use ny_mip::maxsat::{Outcome, Wcnf};

fn run(start: Instant) -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("usage: ny-maxsat INSTANCE.wcnf [--timeout SECONDS]");
        return Ok(());
    }
    if args.len() != 1 && args.len() != 3 {
        return Err("usage: ny-maxsat INSTANCE.wcnf [--timeout SECONDS]".into());
    }
    let seconds = if args.len() == 3 {
        if args[1] != "--timeout" {
            return Err("expected --timeout".into());
        }
        args[2].to_str().ok_or("non-UTF8 timeout")?.parse::<f64>()?
    } else {
        60.0
    };
    if !seconds.is_finite() || !(0.001..=86_400.0).contains(&seconds) {
        return Err("timeout must be finite, in [0.001,86400] seconds".into());
    }
    let deadline = start
        .checked_add(Duration::from_secs_f64(seconds))
        .ok_or("deadline overflow")?;
    let file = File::open(PathBuf::from(&args[0]))?;
    let result =
        Wcnf::read(BufReader::new(file), deadline).and_then(|formula| formula.solve(deadline));
    let mut output = std::io::stdout().lock();
    writeln!(
        output,
        "c ny-maxsat: NY MIP frontend; feasible incumbents only"
    )?;
    writeln!(
        output,
        "c backend ny-mip/ay-milp 0fb75da29369a25e9c5d6a296f299eec2bb61f35; no subprocess"
    )?;
    match result {
        Ok(Outcome::Incumbent(candidate)) => {
            let model: String = candidate
                .assignment
                .into_iter()
                .map(|v| if v { '1' } else { '0' })
                .collect();
            // Prepare the complete frame before the final deadline check. Even
            // formatting a large model is charged to this invocation's budget.
            let frame = format!("o {}\ns SATISFIABLE\nv {model}\n", candidate.cost);
            if model.is_empty() || Instant::now() >= deadline {
                writeln!(output, "c no in-budget nonempty MSE bit vector")?;
                writeln!(output, "s UNKNOWN")?;
            } else {
                // Construct one complete candidate frame before publication;
                // an interrupted pipe remains incomplete for the independent grader.
                output.write_all(frame.as_bytes())?;
            }
        }
        Ok(Outcome::Unknown(reason)) => {
            writeln!(output, "c {}", reason.replace(['\n', '\r'], " "))?;
            writeln!(output, "s UNKNOWN")?;
        }
        Err(error) => {
            writeln!(output, "c {}", error.to_string().replace(['\n', '\r'], " "))?;
            writeln!(output, "s UNKNOWN")?;
        }
    }
    output.flush()?;
    Ok(())
}

fn main() -> ExitCode {
    let start = Instant::now();
    match run(start) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ny-maxsat: {error}");
            ExitCode::from(4)
        }
    }
}

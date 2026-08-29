// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Standalone `ny-claims`: identical code to `ny claims`, built in seconds.
//!
//! It exists because of a measured fact about this repo — an incremental
//! release build of `ny` takes **19 minutes** on this host. A verification
//! gate that costs a 19-minute link is a gate that gets skipped, and a skipped
//! gate is the documentation this crate replaces. So the pre-commit path links
//! nothing but this leaf crate, and the `ny claims` subcommand exists for
//! people who already have the verifier in their hand.
//!
//! The one thing this binary must NOT do is guess the verifier's identity: it
//! asks `target/release/ny --build-stamp` and reports honestly when there is
//! no answer.

use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "ny-claims",
    about = "Re-derive every empirical claim from the artifacts it was made from."
)]
struct Cli {
    #[command(flatten)]
    claims: ny_claims::cli::ClaimsArgs,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let code = ny_claims::cli::run(&cli.claims, None)?;
    std::process::exit(code);
}

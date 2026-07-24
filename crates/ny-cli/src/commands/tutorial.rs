// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny tutorial` — learn to verify neural networks, interactively.
//!
//! Design principles (borrowed from the `ay` tutorial):
//! - Never patronizing. No "great job!", no baby talk.
//! - Show the verifier's real work — actual verdicts, actual exit codes.
//! - Honest. If a method cannot decide something, say so; never a guess.
//! - A thin wrapper over the real `ny` engine: `ny tutorial demo` runs the
//!   same binary the reader would run.

use std::path::Path;
use std::process::Command as ProcCommand;

use anyhow::{Context, Result};
use clap::Subcommand;

// Tiny embedded fixtures (f(x) = |x|, 111-byte network) so `ny tutorial demo`
// runs with no files on disk and no dependency on the checkout layout.
const CROSSING_RELU_NNET: &[u8] = include_bytes!("tutorial/crossing_relu.nnet");
const CROSSING_RELU_SAFE: &[u8] = include_bytes!("tutorial/crossing_relu_safe.vnnlib");
const CROSSING_RELU_UNSAFE: &[u8] = include_bytes!("tutorial/crossing_relu_unsafe.vnnlib");

/// Interactive lessons for neural-network verification.
#[derive(Debug, Subcommand)]
pub(crate) enum TutorialTopic {
    /// Fundamentals of neural-network verification, in five short lessons
    Basics,
    /// Verify real robustness properties and wire ny into CI
    Robustness,
    /// Evidence you can check without trusting ny
    Certificates,
    /// A one-line map of every ny command family
    Features,
    /// Run the verifier live on a tiny embedded network
    Demo,
}

/// Entry point for `ny tutorial [TOPIC]`.
pub(crate) fn run(topic: Option<&TutorialTopic>) -> Result<()> {
    match topic {
        None => {
            print_welcome();
            Ok(())
        }
        Some(TutorialTopic::Basics) => {
            basics();
            Ok(())
        }
        Some(TutorialTopic::Robustness) => {
            robustness();
            Ok(())
        }
        Some(TutorialTopic::Certificates) => {
            certificates();
            Ok(())
        }
        Some(TutorialTopic::Features) => {
            features();
            Ok(())
        }
        Some(TutorialTopic::Demo) => demo(),
    }
}

fn rule() {
    println!("{}", "─".repeat(74));
}

fn heading(title: &str) {
    println!();
    rule();
    println!("  {title}");
    rule();
}

fn print_welcome() {
    heading("ny — learn to verify neural networks");
    println!(
        "
  A verifier answers one question about a model: can it EVER violate a
  property? You give ny a network and a property; it returns one of three
  answers, and it never guesses:

    verified     the property holds for every input in the region
    falsified    the property can be violated — here is a concrete input
    unknown      ny could not settle it within its methods or limits

  Courses — each is a short read, and every command shown is real:

    ny tutorial basics         fundamentals, in five lessons
    ny tutorial robustness     verify real robustness properties; wire ny into CI
    ny tutorial certificates   evidence you can check without trusting ny
    ny tutorial features       a one-line map of every ny command family
    ny tutorial demo           run the verifier live on a tiny embedded network

  New here?   ny tutorial basics
  Impatient?  ny tutorial demo
"
    );
}

fn basics() {
    heading("Basics · 1/5 — a property is a promise about outputs");
    println!(
        "
  `crossing_relu` is a tiny network that computes f(x) = |x|. A property is a
  promise we want to hold for the WHOLE input region, not just a few samples.
  Here the promise is: \"the output stays below 2 for every x in [-1, 1].\"
  Since |x| never exceeds 1 on that box, the promise holds — but a verifier has
  to PROVE it for every point, not test a handful."
    );

    heading("Basics · 2/5 — VNN-LIB states the region and the violation");
    println!(
        "
  Properties are written in VNN-LIB. Two things go in: the input region, and
  the violation to rule out. By convention the file asserts the NEGATION of the
  property — the bad event the verifier must show is impossible:

      (assert (>= X_0 -1.0))   (assert (<= X_0 1.0))   ; region: x in [-1, 1]
      (assert (>= Y_0  2.0))                            ; violation: output >= 2

  If ny can prove that violation is unreachable, the property is verified."
    );

    heading("Basics · 3/5 — bounds without splitting (CROWN)");
    println!(
        "
  The fast, incomplete methods (IBP, CROWN, alpha-CROWN) push a linear
  over-approximation of the output through the network:

      ny verify tests/models/crossing_relu.nnet \\
        -p tests/models/crossing_relu_safe.vnnlib --method crown

  If the over-approximation already fits under the bound, the property is
  proved. If it is too loose, ny reports `unknown` (exit 2) — it will not call
  something safe on a bound it cannot justify."
    );

    heading("Basics · 4/5 — complete verification (beta-CROWN)");
    println!(
        "
  `beta-crown` splits the input and the ReLUs until it either proves the bound
  everywhere or finds a violating input:

      ny beta-crown tests/models/crossing_relu.nnet \\
        -p tests/models/crossing_relu_safe.vnnlib      # → VERIFIED, exit 0

  Now ask for a bound the network CAN exceed — |x| reaches 1 at x = ±1, so
  \"output < 0.5\" is false:

      ny beta-crown tests/models/crossing_relu.nnet \\
        -p tests/models/crossing_relu_unsafe.vnnlib    # → VIOLATED, exit 1

  Exit codes: 0 proved, 1 falsified, 2 unknown, 3 timeout."
    );

    heading("Basics · 5/5 — ny does not ask you to trust it");
    println!(
        "
  A falsified verdict comes with a concrete input, which ny re-runs through the
  model to confirm the violation before printing it — you can re-run it too. A
  verified verdict, on eligible networks, comes with an exact-rational
  certificate an external checker can replay.

  See it run:      ny tutorial demo
  See the proof:   ny tutorial certificates
"
    );
}

fn robustness() {
    heading("Robustness — the property people actually want");
    println!(
        "
  \"Local robustness\" asks: within a small perturbation of this input, can the
  network's decision change? The perturbation is the region; the decision flip
  is the violation. ny ships fixtures for the standard shape:

      # an MNIST image, L-infinity ball of radius eps around it
      ny beta-crown mnist.onnx -p tests/models/mnist_robustness_eps0.010_label0.vnnlib

  Smaller eps is easier to prove; larger eps is where verifiers earn their keep.
  `ny verify --method alpha` is the quick incomplete pass; `ny beta-crown` is
  the complete one that also returns counterexamples.

  ACAS Xu — the classic aircraft collision-avoidance benchmark — runs under the
  competition protocol end to end:

      ny vnncomp v1 acasxu_2023 model.onnx prop.vnnlib out.txt 116

  In CI, drive the build off the exit code (0 proved, 1 falsified, 2 unknown,
  3 timeout), or use the Python pytest plugin:

      from ny_pytest import assert_verified
      assert_verified(\"model.onnx\", \"prop.vnnlib\")     # fails the test on a counterexample
"
    );
}

fn certificates() {
    heading("Certificates — checking ny without trusting ny");
    println!(
        "
  Two kinds of evidence back a verdict:

  A COUNTEREXAMPLE is self-evidencing. ny re-executes the reported input through
  the model and shows it violates the property before printing it. Re-run it in
  any inference engine and you will see the same violation.

  A PROOF is what backs a `verified` verdict. On a sequential fully-connected
  ReLU network (up to 8192 hidden neurons), ny re-derives the CROWN bound in
  exact rational arithmetic and writes an entailment + Farkas certificate:

      ny beta-crown model.nnet -p prop.vnnlib --emit-certificate model.cert.json

  The certificate is self-checked before it is written, then stands on its own —
  the claim, the exact-rational Farkas multipliers, and the linear premises that
  empty the unsafe region. No floating point, no reference to ny's solver state.
  ny's external kernel checker, Clean, replays these end to end (published
  alongside ny; until then the Lean corpus in crates/ny-cert/proofs/lean/ is the
  reproducible kernel-check path).

  Related evidence:
      ny lipschitz model.onnx        sound certified global Lipschitz bound
      ny gt verify spec model.onnx --escalate smt   exact SMT check via the ay solver
"
    );
}

fn features() {
    heading("Features — a map of the ny command families");
    println!(
        "
  Verification
      ny verify        incomplete bounds (ibp | crown | alpha | beta)
      ny beta-crown    complete branch-and-bound; returns proofs or counterexamples
      ny vnncomp       run one VNN-COMP instance under the competition protocol
      ny gt            verify against a geometric ground-truth spec (exact SMT escalation)

  Understand a model
      ny inspect       structure and operator list
      ny lipschitz     certified global Lipschitz upper bound
      ny sensitivity   which layers amplify input noise
      ny profile-bounds  how bound width grows through the network
      ny coverage      per-operator soundness classification

  Compare and port
      ny diff          where two ONNX models diverge, layer by layer
      ny compare       equivalence between two models
      ny weights       inspect and diff weights (ONNX, SafeTensors)
      ny quantize-check  float16 / int8 quantization safety

  Run `ny <command> --help` for the flags of any command.
"
    );
}

fn demo() -> Result<()> {
    heading("Demo — the real verifier on a tiny embedded network");
    let dir = std::env::temp_dir().join(format!("ny-tutorial-{}", std::process::id()));
    std::fs::create_dir_all(&dir).context("create tutorial scratch directory")?;
    let model = dir.join("crossing_relu.nnet");
    let safe = dir.join("crossing_relu_safe.vnnlib");
    let unsafe_prop = dir.join("crossing_relu_unsafe.vnnlib");
    std::fs::write(&model, CROSSING_RELU_NNET).context("write demo model")?;
    std::fs::write(&safe, CROSSING_RELU_SAFE).context("write demo safe property")?;
    std::fs::write(&unsafe_prop, CROSSING_RELU_UNSAFE).context("write demo unsafe property")?;

    let exe = std::env::current_exe().context("locate the running ny binary")?;
    println!(
        "
  f(x) = |x| on the box x in [-1, 1]. First a property that HOLDS (output < 2),
  then one that does NOT (output < 0.5, but |x| reaches 1). ny runs both below —
  this is the same `ny beta-crown` you would type yourself.
"
    );
    run_and_show(&exe, &model, &safe, "output < 2  (should verify)");
    run_and_show(&exe, &model, &unsafe_prop, "output < 0.5  (should falsify)");
    let _ = std::fs::remove_dir_all(&dir);
    println!(
        "
  That is the whole loop: a property, a verdict, and — for the counterexample —
  a concrete input ny re-checked before printing. Next: ny tutorial certificates
"
    );
    Ok(())
}

fn run_and_show(exe: &Path, model: &Path, prop: &Path, label: &str) {
    println!("  ── {label} ──");
    println!("  $ ny beta-crown crossing_relu.nnet -p {}", short(prop));
    let output = ProcCommand::new(exe)
        .arg("beta-crown")
        .arg(model)
        .arg("-p")
        .arg(prop)
        .output();
    match output {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout);
            // Echo just the human "--- Result ---" section, indented.
            let mut in_result = false;
            for line in text.lines() {
                if line.trim_start().starts_with("--- Result") {
                    in_result = true;
                }
                if in_result {
                    println!("      {line}");
                }
            }
            let code = out.status.code().unwrap_or(-1);
            let verdict = match code {
                0 => "verified",
                1 => "falsified",
                2 => "unknown",
                3 => "timeout",
                _ => "error",
            };
            println!("      (exit {code} → {verdict})");
        }
        Err(err) => {
            println!("      (could not launch the verifier: {err})");
            println!("      run it yourself once ny is on PATH:");
            println!(
                "        ny beta-crown tests/models/crossing_relu.nnet -p tests/models/{}",
                prop.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("prop.vnnlib")
            );
        }
    }
    println!();
}

fn short(p: &Path) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("prop.vnnlib")
        .to_string()
}

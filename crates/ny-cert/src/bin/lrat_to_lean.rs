//! `lrat_to_lean` — transcribe a DIMACS CNF + LRAT refutation into a
//! self-contained Lean 4 instance file for NY's `NyProof` overlay.
//!
//! The sat_relu Route A driver (`ny-cli` `cnf_route.rs`) recovers a CNF from
//! the gadget network and has `ay` solve it; on UNSAT, `ay` emits a proof
//! artifact.  `ay` emits **LRAT natively** (`ay solve x.cnf --proof x.lrat
//! --proof-format lrat`), so no DRAT→LRAT conversion step is needed — this
//! tool parses the LRAT text format directly and fails closed on anything it
//! does not understand (RAT steps with negative hints, unknown clause ids,
//! missing empty clause).
//!
//! The emitted Lean file contains:
//!   * `F : Formula` — the DIMACS clauses as a `List (List (ℕ × Bool))` literal;
//!   * `steps : List RStep` — the LRAT addition lines, hints resolved from
//!     LRAT clause ids to 0-based indices into the growing clause database
//!     (original clauses first, then each derived clause in order; deletion
//!     lines are dropped — sound, the checker looks hints up explicitly);
//!   * `check_ok : checkRefutation F steps = true := by decide` — the kernel
//!     replays the whole refutation (plain `decide`; NEVER `native_decide`,
//!     which would add the `Lean.ofReduceBool` axiom and fail the audit gate);
//!   * `instance_unsat : ¬ ∃ σ, satFormula σ F` via
//!     `RupChecker.checkRefutation_sound`;
//!   * `instance_safe` — the real-arithmetic gadget verdict via
//!     `SatReluVerdict.safe_of_unsat` with `s = Finset.Icc 1 n_vars` and the
//!     clause-variables-⊆-s side condition discharged by `decide`.
//!
//! This transcription is TRUSTED PLUMBING in the sense of the overlay notes:
//! it does no reasoning, only syntax.  Everything semantic is re-checked by
//! the Lean kernel when the emitted file is built.
//!
//! Usage:
//!   lrat_to_lean <input.cnf> <input.lrat> <output.lean> <ModuleName> [--fast]
//!
//! `ModuleName` becomes a namespace under `Crownproof` while the source belongs
//! to NY's `NyProof` library (e.g. `SatReluDemo_v10c26` is written to
//! `NyProof/SatReluDemo_v10c26.lean`).
//!
//! With `--fast` the emitted file targets `RupCheckerFast.checkRefutationFast`
//! (bitmask assignment + trie clause database — same `Formula`/`RStep`
//! literals, ~2 orders of magnitude faster kernel replay, same 3-axiom
//! soundness theorem).  Without it the output is byte-identical to the
//! original mode targeting `RupChecker.checkRefutation`.

use std::fmt::Write as _;
use std::process::ExitCode;

/// A clause: DIMACS signed literals (non-zero).
type Clause = Vec<i64>;

/// One LRAT addition step with hints resolved to 0-based database indices.
struct Step {
    clause: Clause,
    hints: Vec<usize>,
}

struct Cnf {
    n_vars: usize,
    clauses: Vec<Clause>,
}

fn parse_dimacs(text: &str) -> Result<Cnf, String> {
    let mut n_vars: Option<usize> = None;
    let mut declared_clauses: Option<usize> = None;
    let mut clauses = Vec::new();
    let mut current: Clause = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') || line.starts_with('%') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('p') {
            // total: iterator destructure (not `.collect()` + `toks[k]`): the
            // identical "exactly 3 tokens, first is `cnf`" acceptance with no
            // bulk alloc or slice indexing; anything else fails closed.
            let mut toks = rest.split_whitespace();
            let (Some("cnf"), Some(nv), Some(nc), None) =
                (toks.next(), toks.next(), toks.next(), toks.next())
            else {
                return Err(format!("malformed DIMACS header: {line}"));
            };
            n_vars = Some(nv.parse().map_err(|e| format!("bad n_vars: {e}"))?);
            declared_clauses = Some(nc.parse().map_err(|e| format!("bad n_clauses: {e}"))?);
            continue;
        }
        for tok in line.split_whitespace() {
            let lit: i64 = tok
                .parse()
                .map_err(|e| format!("bad DIMACS literal {tok:?}: {e}"))?;
            if lit == 0 {
                clauses.push(std::mem::take(&mut current));
            } else {
                current.push(lit);
            }
        }
    }
    if !current.is_empty() {
        return Err("DIMACS ends inside an unterminated clause".to_string());
    }
    let n_vars = n_vars.ok_or("DIMACS file has no `p cnf` header")?;
    if let Some(m) = declared_clauses {
        if m != clauses.len() {
            return Err(format!(
                "DIMACS header declares {m} clauses but {} were read",
                clauses.len()
            ));
        }
    }
    for cl in &clauses {
        for &lit in cl {
            let v = lit.unsigned_abs() as usize;
            if v == 0 || v > n_vars {
                return Err(format!("literal {lit} out of range 1..={n_vars}"));
            }
        }
    }
    Ok(Cnf { n_vars, clauses })
}

/// Parse LRAT text: addition lines `<id> <lit>* 0 <hint-id>* 0`, deletion
/// lines `<id> d <id>* 0` (dropped).  Hints are resolved from LRAT clause ids
/// to 0-based indices into the database `original clauses ++ prior additions`.
/// Fails closed on negative hints (RAT — not RUP), unknown ids, or a proof
/// that never derives the empty clause.
fn parse_lrat(text: &str, n_original: usize) -> Result<Vec<Step>, String> {
    // LRAT id -> database index. Original clause id k (1-based) -> k-1.
    // Explicit insert loop (not `.collect()`): the original-clause count is
    // input-derived, so a bulk collect raises an unbounded-alloc obligation;
    // identical map.
    let mut id_to_idx: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for i in 0..n_original {
        id_to_idx.insert(i as u64 + 1, i);
    }
    let mut db_len = n_original;
    let mut steps: Vec<Step> = Vec::new();
    let mut saw_empty = false;
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        if saw_empty {
            break; // checker accepts at the empty clause; ignore trailing lines
        }
        // Explicit Vec::new()+push (not `.collect()`): the token count is
        // input-derived, so a bulk collect raises an unbounded-alloc
        // obligation; identical Vec.
        let mut toks: Vec<&str> = Vec::new();
        for tok in line.split_whitespace() {
            toks.push(tok);
        }
        if toks.len() < 2 {
            return Err(format!("LRAT line {}: too short: {line}", lineno + 1));
        }
        // total: `first`/`get(1)` (not `toks[0]`/`toks[1]`): `toks.len() >= 2`
        // was just checked, so both are `Some`; the fallback fails closed.
        let id_tok = *toks
            .first()
            .ok_or_else(|| format!("LRAT line {}: too short: {line}", lineno + 1))?;
        let id: u64 = id_tok
            .parse()
            .map_err(|e| format!("LRAT line {}: bad id {id_tok:?}: {e}", lineno + 1))?;
        if toks.get(1) == Some(&"d") {
            continue; // deletion — keep clauses in the database (sound)
        }
        // literals up to the first 0
        let mut clause: Clause = Vec::new();
        let mut pos = 1;
        loop {
            let tok = toks
                .get(pos)
                .ok_or_else(|| format!("LRAT line {}: missing literal terminator", lineno + 1))?;
            let lit: i64 = tok
                .parse()
                .map_err(|e| format!("LRAT line {}: bad literal {tok:?}: {e}", lineno + 1))?;
            pos += 1;
            if lit == 0 {
                break;
            }
            clause.push(lit);
        }
        // hints up to the second 0
        let mut hints: Vec<usize> = Vec::new();
        loop {
            let tok = toks
                .get(pos)
                .ok_or_else(|| format!("LRAT line {}: missing hint terminator", lineno + 1))?;
            let hint: i64 = tok
                .parse()
                .map_err(|e| format!("LRAT line {}: bad hint {tok:?}: {e}", lineno + 1))?;
            pos += 1;
            if hint == 0 {
                break;
            }
            if hint < 0 {
                return Err(format!(
                    "LRAT line {}: negative hint {hint} — RAT step, not RUP; \
                     this importer fails closed on RAT (CDCL DRUP-style proofs \
                     from `ay` are RUP-only)",
                    lineno + 1
                ));
            }
            let idx = *id_to_idx.get(&(hint as u64)).ok_or_else(|| {
                format!(
                    "LRAT line {}: hint id {hint} not in the live clause database",
                    lineno + 1
                )
            })?;
            hints.push(idx);
        }
        if pos != toks.len() {
            return Err(format!(
                "LRAT line {}: trailing tokens after hint terminator",
                lineno + 1
            ));
        }
        saw_empty = clause.is_empty();
        steps.push(Step { clause, hints });
        id_to_idx.insert(id, db_len);
        db_len += 1;
    }
    if !saw_empty {
        return Err("LRAT proof never derives the empty clause".to_string());
    }
    Ok(steps)
}

fn lean_clause(clause: &[i64]) -> String {
    // Explicit Vec::new()+push (not `.collect()`): input-derived literal
    // count; a bulk collect raises an unbounded-alloc obligation. Identical
    // output.
    let mut lits: Vec<String> = Vec::new();
    for &l in clause {
        lits.push(format!("({}, {})", l.unsigned_abs(), l > 0));
    }
    format!("[{}]", lits.join(", "))
}

/// Make a string safe for inclusion in a Lean block comment: Lean block
/// comments NEST, so a path like `/tmp/x/-y` would open an unterminated
/// nested comment.
fn comment_safe(s: &str) -> String {
    s.replace("/-", "/ -").replace("-/", "- /")
}

fn emit_lean(
    cnf: &Cnf,
    steps: &[Step],
    module: &str,
    cnf_src: &str,
    lrat_src: &str,
    fast: bool,
) -> String {
    let mut out = String::new();
    let (cnf_src, lrat_src) = (comment_safe(cnf_src), comment_safe(lrat_src));
    let n = cnf.n_vars;
    let m = cnf.clauses.len();
    let (checker, sound, extra_import, max_rec_depth) = if fast {
        (
            "RupCheckerFast.checkRefutationFast",
            "RupCheckerFast.checkRefutationFast_sound",
            "\nimport NyProof.RupCheckerFast",
            65536,
        )
    } else {
        (
            "RupChecker.checkRefutation",
            "RupChecker.checkRefutation_sound",
            "",
            8192,
        )
    };
    let _ = write!(
        out,
        "/-\n\
         MACHINE-GENERATED by `ny-cert`'s `lrat_to_lean` — do not edit.\n\
         \n\
         sat_relu Route A instance transcript:\n\
         * DIMACS source : {cnf_src}  (p cnf {n} {m})\n\
         * LRAT source   : {lrat_src}  ({} addition steps kept)\n\
         \n\
         The kernel replays the full LRAT refutation with PLAIN `decide`\n\
         (no `native_decide`, no extra axioms) through\n\
         `{checker}`, then composes\n\
         `{sound_tail}` (CNF-unsat) with\n\
         `SatReluVerdict.safe_of_unsat` (gadget real-arithmetic safety).\n\
         -/\n\
         import NyProof.SatReluVerdict{extra_import}\n\
         \n\
         namespace Crownproof\n\
         \n\
         namespace {module}\n\
         \n\
         open RupImport.RUP RupChecker\n\
         \n\
         set_option maxRecDepth {max_rec_depth}\n\
         \n",
        steps.len(),
        sound_tail = sound.split('.').next_back().unwrap_or(sound),
    );

    out.push_str("/-- The recovered DIMACS formula, literal-for-literal. -/\ndef F : Formula :=\n");
    for (i, cl) in cnf.clauses.iter().enumerate() {
        let sep = if i == 0 { "  [ " } else { "  , " };
        let _ = writeln!(out, "{sep}{}", lean_clause(cl));
    }
    out.push_str("  ]\n\n");

    out.push_str(
        "/-- The LRAT refutation: derived clause + database-index hints per step. -/\n\
         def steps : List RStep :=\n",
    );
    for (i, st) in steps.iter().enumerate() {
        let sep = if i == 0 { "  [ " } else { "  , " };
        // Explicit push (not `.collect()`): input-derived hint count — same
        // unbounded bulk-alloc obligation as `lean_clause`. Identical output.
        let mut hints: Vec<String> = Vec::new();
        for h in &st.hints {
            hints.push(h.to_string());
        }
        let _ = writeln!(
            out,
            "{sep}⟨{}, [{}]⟩",
            lean_clause(&st.clause),
            hints.join(", ")
        );
    }
    out.push_str("  ]\n\n");

    let check_call = if fast {
        "RupCheckerFast.checkRefutationFast F steps"
    } else {
        "checkRefutation F steps"
    };
    let _ = write!(
        out,
        "set_option maxHeartbeats 40000000 in\n\
         /-- KERNEL replay of the refutation — plain `decide`, no `native_decide`. -/\n\
         theorem check_ok : {check_call} = true := by decide\n\
         \n\
         /-- The recovered CNF is unsatisfiable (kernel-checked LRAT replay +\n\
         `rup_sound` composition). -/\n\
         theorem instance_unsat : ¬ ∃ σ, satFormula σ F :=\n\
         \x20\x20{sound} F steps check_ok\n\
         \n\
         /-- The gadget's variable set: DIMACS variables `1..{n}`. -/\n\
         def s : Finset ℕ := Finset.Icc 1 {n}\n\
         \n\
         set_option maxHeartbeats 4000000 in\n\
         /-- END-TO-END VERDICT: no point of the `[0,1]` box reaches the sat_relu\n\
         gadget's unsafe region `{{Y₀ ≥ 1 ∧ Y₁ ≤ 0}}` — real-arithmetic SAFE. -/\n\
         theorem instance_safe :\n\
         \x20\x20\x20\x20∀ x : ℕ → ℝ, (∀ j ∈ s, 0 ≤ x j ∧ x j ≤ 1) →\n\
         \x20\x20\x20\x20\x20\x20¬ (1 ≤ SatRelu.Y0 (SatReluVerdict.clausesOf F) x ∧\n\
         \x20\x20\x20\x20\x20\x20\x20\x20\x20SatRelu.Y1 s x ≤ 0) :=\n\
         \x20\x20SatReluVerdict.safe_of_unsat F s (by decide) instance_unsat\n\
         \n\
         end {module}\n\
         \n\
         end Crownproof\n\
         \n\
         /-! ## Trust-base check -/\n\
         \n\
         #print axioms Crownproof.{module}.check_ok\n\
         #print axioms Crownproof.{module}.instance_unsat\n\
         #print axioms Crownproof.{module}.instance_safe\n"
    );
    out
}

fn run() -> Result<(), String> {
    // Explicit push (not `.collect()`): argv length is input-derived — same
    // unbounded bulk-alloc obligation as the parsers above. Identical Vec.
    let mut args: Vec<String> = Vec::new();
    for a in std::env::args() {
        args.push(a);
    }
    // total: `get` (not `args[5]` / `args[1..=4]`): guarded by the arg-count
    // checks, so the `None` arms are unreachable and fail closed (usage error)
    // rather than index.
    let fast = args.len() == 6 && args.get(5).is_some_and(|a| a == "--fast");
    if !(args.len() == 5 || fast) {
        return Err(format!(
            "usage: {} <input.cnf> <input.lrat> <output.lean> <ModuleName> [--fast]",
            args.first().map_or("lrat_to_lean", String::as_str)
        ));
    }
    let (Some(cnf_path), Some(lrat_path), Some(out_path), Some(module)) =
        (args.get(1), args.get(2), args.get(3), args.get(4))
    else {
        return Err(
            "usage: lrat_to_lean <input.cnf> <input.lrat> <output.lean> <ModuleName> [--fast]"
                .to_owned(),
        );
    };
    if module.is_empty()
        || !module
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        || !module
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_uppercase())
    {
        return Err(format!(
            "ModuleName {module:?} must be an uppercase-initial Lean identifier \
             ([A-Z][A-Za-z0-9_]*)"
        ));
    }
    let cnf_text =
        std::fs::read_to_string(cnf_path).map_err(|e| format!("read {cnf_path}: {e}"))?;
    let lrat_text =
        std::fs::read_to_string(lrat_path).map_err(|e| format!("read {lrat_path}: {e}"))?;
    let cnf = parse_dimacs(&cnf_text)?;
    let steps = parse_lrat(&lrat_text, cnf.clauses.len())?;
    let lean = emit_lean(&cnf, &steps, module, cnf_path, lrat_path, fast);
    std::fs::write(out_path, lean).map_err(|e| format!("write {out_path}: {e}"))?;
    eprintln!(
        "lrat_to_lean: {} vars, {} clauses, {} LRAT steps -> {}{}",
        cnf.n_vars,
        cnf.clauses.len(),
        steps.len(),
        out_path,
        if fast { " (fast checker)" } else { "" }
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lrat_to_lean: error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{emit_lean, parse_dimacs, parse_lrat, Cnf, Step};

    const CNF: &str = "c comment\np cnf 2 2\n1 -2 0\n-1 0\n";

    #[test]
    fn parses_dimacs() {
        let cnf = parse_dimacs(CNF).expect("parse");
        assert_eq!(cnf.n_vars, 2);
        assert_eq!(cnf.clauses, vec![vec![1, -2], vec![-1]]);
    }

    #[test]
    fn rejects_clause_count_mismatch() {
        assert!(parse_dimacs("p cnf 2 3\n1 0\n").is_err());
    }

    #[test]
    fn parses_lrat_with_deletion_and_id_mapping() {
        // ids 1,2 are the originals; id 4 derives [-2]; id 5 deletes; id 6 empty.
        let lrat = "4 -2 0 1 2 0\n5 d 1 0\n6 0 4 2 0\n";
        let steps = parse_lrat(lrat, 2).expect("parse");
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].clause, vec![-2]);
        assert_eq!(steps[0].hints, vec![0, 1]);
        assert_eq!(steps[1].clause, Vec::<i64>::new());
        // hint id 4 -> database index 2 (first derived clause)
        assert_eq!(steps[1].hints, vec![2, 1]);
    }

    #[test]
    fn rejects_rat_hints_and_missing_empty_clause() {
        assert!(parse_lrat("3 -2 0 -1 2 0\n", 2).is_err());
        assert!(parse_lrat("3 -2 0 1 2 0\n", 2).is_err());
    }

    #[test]
    fn emitted_instances_import_the_ny_overlay() {
        let cnf = Cnf {
            n_vars: 1,
            clauses: vec![vec![1]],
        };
        let steps = vec![Step {
            clause: Vec::new(),
            hints: vec![0],
        }];
        let regular = emit_lean(&cnf, &steps, "Demo", "x.cnf", "x.lrat", false);
        assert!(regular.contains("import NyProof.SatReluVerdict\n"));
        assert!(!regular.contains("import Crownproof.SatReluVerdict"));

        let fast = emit_lean(&cnf, &steps, "Demo", "x.cnf", "x.lrat", true);
        assert!(fast.contains("import NyProof.SatReluVerdict"));
        assert!(fast.contains("import NyProof.RupCheckerFast"));
        assert!(!fast.contains("import Crownproof.RupCheckerFast"));
    }
}

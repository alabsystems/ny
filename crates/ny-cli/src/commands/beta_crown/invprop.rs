// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ny_propagate::AlphaCrownConfig;
use tracing::{info, warn};

pub(super) fn maybe_set_alpha_output_constraints(
    alpha_config: &mut AlphaCrownConfig,
    vnnlib_spec: Option<&ny_onnx::vnnlib::VnnLibSpec>,
) -> Result<()> {
    if let Some(spec) = vnnlib_spec {
        if spec.output_constraints.is_empty() && spec.output_constraint_clauses.is_empty() {
            warn!(
                "VNN-LIB property has no output constraints; skipping INVPROP output constraints"
            );
            return Ok(());
        }
        if spec.has_multi_constraint_disjunction() {
            warn!(
                "VNN-LIB property has multi-constraint disjunctions; skipping INVPROP output constraints"
            );
            return Ok(());
        }
        alpha_config.output_constraints = Some(spec.to_output_constraints()?);
    }
    Ok(())
}

/// Parse a truthy env var ("1"/"true"/"on"/"yes", case-insensitive).
fn env_truthy(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| {
        let v = v.trim();
        v == "1"
            || v.eq_ignore_ascii_case("true")
            || v.eq_ignore_ascii_case("on")
            || v.eq_ignore_ascii_case("yes")
    })
}

pub(super) fn maybe_enable_invprop(
    alpha_config: &mut AlphaCrownConfig,
    invprop: bool,
    invprop_apply: Vec<String>,
    invprop_share_gammas: bool,
) -> Result<()> {
    // INGRESS ON BY DEFAULT: supported conjunction queries initialize the
    // exact-zero output seed and its clause-local metadata. The proof-producing
    // gamma treatment remains default-dark below. A nonzero gamma produces bounds
    // valid only inside the candidate violation region, so feasible conditioned
    // iterates are never published as global output boxes. Only a typed finite-
    // inversion certificate may discharge the region; disjunctions fail closed
    // below. Opt out per-run with `NY_INVPROP=0`; `--invprop` also forces ingress.
    let enable = env_truthy("NY_INVPROP").unwrap_or(true) || invprop;
    if !enable {
        return Ok(());
    }

    // Gamma optimization (the assume-violation output-seed ascent) is an
    // experimental treatment. Keep it default-dark until an authenticated,
    // receipt-bound category factorial demonstrates a net solve gain: on deep
    // convolutional models each probe can cost another CPU backward pass.
    // `NY_INVPROP_OPTIMIZE=1` opts in explicitly; the zero-gamma default remains
    // the byte-identical inert baseline.
    let optimize = env_truthy("NY_INVPROP_OPTIMIZE").unwrap_or(false);
    let gamma_lr = std::env::var("NY_INVPROP_LR")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| *v > 0.0);

    alpha_config.invprop.enabled = true;
    alpha_config.invprop.optimize_gammas = optimize;
    if let Some(lr) = gamma_lr {
        alpha_config.invprop.gamma_lr = lr;
    } else if alpha_config.invprop.gamma_lr <= 0.0 {
        alpha_config.invprop.gamma_lr = 0.5;
    }
    alpha_config.invprop.share_gammas = invprop_share_gammas;
    alpha_config.invprop.apply_output_constraints_to = if invprop_apply.is_empty() {
        vec!["all".to_string()]
    } else {
        invprop_apply
    };

    if optimize {
        info!(
            "INVPROP enabled (assume-violation output-seed dual, gamma ascent lr={})",
            alpha_config.invprop.gamma_lr
        );
    } else {
        info!("INVPROP enabled with inert zero-gamma seed (set NY_INVPROP_OPTIMIZE=1 to opt in)");
    }

    match alpha_config.output_constraints.as_ref() {
        None => {
            // Constraints are attached by `maybe_set_alpha_output_constraints`
            // immediately BEFORE this call, so `None` here is final, not a
            // not-yet-populated state. Leaving `enabled = true` made the config
            // self-report an active channel that every consumer then skipped
            // (they all require `output_constraints.is_some()`), which reads in
            // logs as "INVPROP enabled" next to "no output constraints" — the
            // pair that sent a live lsnc_relu diagnosis down the wrong path.
            // Report the channel as what it is: off.
            warn!("INVPROP requested but no output constraints were provided (use --property); disabling INVPROP for this query");
            alpha_config.invprop.enabled = false;
            alpha_config.invprop.optimize_gammas = false;
        }
        Some(oc) if !oc.is_conjunction => {
            // Fail closed: a disjunctive violation must not be dualized as one
            // conjunction. Disable the channel rather than error out so the run
            // proceeds soundly without INVPROP.
            warn!(
                "INVPROP: top-level disjunction detected; disabling INVPROP for this query (conjunction-only)"
            );
            alpha_config.invprop.enabled = false;
            alpha_config.invprop.optimize_gammas = false;
        }
        Some(_) => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wires_output_constraints_from_vnnlib() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= Y_0 Y_1))
"#;

        let spec = ny_onnx::vnnlib::parse_vnnlib(content).unwrap();

        let mut alpha_config = AlphaCrownConfig::default();
        assert!(alpha_config.output_constraints.is_none());

        maybe_set_alpha_output_constraints(&mut alpha_config, Some(&spec)).unwrap();

        let oc = alpha_config.output_constraints.as_ref().unwrap();
        assert_eq!(oc.num_constraints(), 1);
        assert_eq!(oc.output_dim(), 2);
        assert!(oc.is_conjunction);
    }

    #[test]
    fn invprop_defaults_to_all_layers_when_enabled() {
        // A conjunctive property, so the channel actually stays enabled: INVPROP
        // with no output constraints is inert and now reports itself as off, so
        // exercising the "all layers" default requires real constraints.
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (<= Y_0 Y_1))
"#;
        let spec = ny_onnx::vnnlib::parse_vnnlib(content).unwrap();

        let mut alpha_config = AlphaCrownConfig::default();
        assert!(!alpha_config.invprop.enabled);
        assert!(alpha_config.output_constraints.is_none());

        maybe_set_alpha_output_constraints(&mut alpha_config, Some(&spec)).unwrap();
        maybe_enable_invprop(&mut alpha_config, true, Vec::new(), false).unwrap();

        assert!(alpha_config.invprop.enabled);
        assert_eq!(
            alpha_config.invprop.apply_output_constraints_to,
            vec!["all".to_string()]
        );
        assert!(!alpha_config.invprop.share_gammas);
        assert!(
            !alpha_config.invprop.optimize_gammas,
            "gamma optimization must remain default-dark until a sealed factorial promotes it"
        );
    }

    #[test]
    fn invprop_rejects_disjunction_properties() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (or (<= Y_0 0.0) (<= Y_1 0.0)))
"#;
        let spec = ny_onnx::vnnlib::parse_vnnlib(content).unwrap();
        assert!(spec.is_disjunction);
        assert!(spec.has_multi_constraint_disjunction());

        // The multi-constraint disjunction guard skips setting output constraints,
        // so output_constraints remains None (disjunctions are handled by per-clause
        // dispatch instead of INVPROP).
        let mut alpha_config = AlphaCrownConfig::default();
        maybe_set_alpha_output_constraints(&mut alpha_config, Some(&spec)).unwrap();
        assert!(alpha_config.output_constraints.is_none());

        // With no output constraints, maybe_enable_invprop warns but does not
        // error — and reports the channel as OFF rather than claiming it is
        // enabled. This is the live lsnc_relu shape (multi-constraint
        // disjunction => constraints skipped): the old `enabled = true` here was
        // inert at every consumer but made logs read "INVPROP enabled" beside
        // "no output constraints were provided".
        maybe_enable_invprop(&mut alpha_config, true, Vec::new(), false).unwrap();
        assert!(!alpha_config.invprop.enabled);
        assert!(!alpha_config.invprop.optimize_gammas);
    }

    #[test]
    fn skips_output_constraints_for_empty_property() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0))
(assert (<= X_0 1))
"#;

        let spec = ny_onnx::vnnlib::parse_vnnlib(content).unwrap();
        assert!(spec.output_constraints.is_empty());

        let mut alpha_config = AlphaCrownConfig::default();
        maybe_set_alpha_output_constraints(&mut alpha_config, Some(&spec)).unwrap();
        assert!(alpha_config.output_constraints.is_none());
    }

    #[test]
    fn skips_output_constraints_for_multi_constraint_disjunction() {
        let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (or (and (<= Y_0 0.0) (<= Y_1 0.1))
            (and (<= Y_0 0.2) (<= Y_1 0.3))))
"#;
        let spec = ny_onnx::vnnlib::parse_vnnlib(content).unwrap();
        assert!(spec.has_multi_constraint_disjunction());

        let mut alpha_config = AlphaCrownConfig::default();
        maybe_set_alpha_output_constraints(&mut alpha_config, Some(&spec)).unwrap();
        assert!(alpha_config.output_constraints.is_none());
    }

    #[test]
    fn wires_programmatic_clause_only_conjunction() {
        use ny_onnx::vnnlib::OutputConstraint;

        let mut spec = ny_onnx::vnnlib::VnnLibSpec::new();
        spec.num_outputs = 2;
        spec.output_constraint_clauses = vec![vec![OutputConstraint::LessEq(0, 1)]];
        assert!(spec.output_constraints.is_empty());
        assert!(!spec.is_disjunction);

        let mut alpha_config = AlphaCrownConfig::default();
        maybe_set_alpha_output_constraints(&mut alpha_config, Some(&spec)).unwrap();

        let constraints = alpha_config
            .output_constraints
            .expect("a clause-only conjunction must reach INVPROP");
        assert!(constraints.is_conjunction);
        assert_eq!(constraints.num_constraints(), 1);
        assert_eq!(constraints.output_dim(), 2);
    }
}

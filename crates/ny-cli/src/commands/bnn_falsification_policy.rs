// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Frozen policy for the two scored binarized-network falsification lanes.

/// Frozen trust-region selection for the LP-guided sign-space lane.
///
/// `FullBox` is deliberately distinct from the explicit `box` treatment: the
/// former is the historical unset policy, while the latter begins with a 1/8
/// box-radius restriction and expands only as needed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum SignSpaceTrustRegionPolicy {
    #[default]
    FullBox,
    Box,
    Tight,
    Linf,
}

impl SignSpaceTrustRegionPolicy {
    pub(crate) const fn stable_token(self) -> &'static str {
        match self {
            Self::FullBox => "full_box",
            Self::Box => "box",
            Self::Tight => "tight",
            Self::Linf => "linf",
        }
    }

    pub(crate) fn from_value(raw: Option<&str>) -> Option<Self> {
        match raw {
            None => Some(Self::FullBox),
            Some("box") => Some(Self::Box),
            Some("tight") => Some(Self::Tight),
            Some("linf") => Some(Self::Linf),
            Some(_) => None,
        }
    }
}

/// The complete BNN falsification policy consumed by the scored wrapper.
///
/// This value is derived once from the receipt-owning [`ny_levers::LeverSet`]
/// and then carried into every private, committed-cap, and scheduled lane
/// invocation. No method on this type consults process environment or preset
/// compatibility cells.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BnnFalsificationPolicy {
    pub(crate) sign_space_enabled: bool,
    pub(crate) minimal_move: bool,
    pub(crate) trust_region: SignSpaceTrustRegionPolicy,
    pub(crate) ste_pgd_enabled: bool,
}

impl BnnFalsificationPolicy {
    pub(crate) fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::dark_probes::{
            BNN_SIGN_SPACE_LANE, BNN_SIGN_SPACE_MINIMAL_MOVE, BNN_SIGN_SPACE_TRUST_REGION,
            BNN_STE_PGD_LANE,
        };
        use ny_levers::LeverValue;

        let boolean = |decl: &'static ny_levers::LeverDecl| match levers.get(decl) {
            LeverValue::Bool(value) => Ok(value),
            other => Err(format!(
                "{} resolved to non-boolean value {other:?}",
                decl.name
            )),
        };
        let trust_region = match levers.get(&BNN_SIGN_SPACE_TRUST_REGION) {
            LeverValue::Unset => SignSpaceTrustRegionPolicy::FullBox,
            LeverValue::Text(value) => SignSpaceTrustRegionPolicy::from_value(Some(&value))
                .ok_or_else(|| {
                    format!(
                        "{} resolved to unknown arm {value:?}",
                        BNN_SIGN_SPACE_TRUST_REGION.name
                    )
                })?,
            other => {
                return Err(format!(
                    "{} resolved to non-enum value {other:?}",
                    BNN_SIGN_SPACE_TRUST_REGION.name
                ));
            }
        };
        Ok(Self {
            sign_space_enabled: boolean(&BNN_SIGN_SPACE_LANE)?,
            minimal_move: boolean(&BNN_SIGN_SPACE_MINIMAL_MOVE)?,
            trust_region,
            ste_pgd_enabled: boolean(&BNN_STE_PGD_LANE)?,
        })
    }

    /// Resolve the lower typed-preset/default layer without consulting the
    /// process environment. This is execution evidence for the counterfactual
    /// that an explicit environment override replaced.
    pub(crate) fn from_config_values(
        sign_space: Option<bool>,
        minimal_move: Option<bool>,
        trust_region: Option<&str>,
        ste_pgd: Option<bool>,
    ) -> Result<Self, String> {
        let trust_region = SignSpaceTrustRegionPolicy::from_value(trust_region)
            .ok_or_else(|| format!("unknown typed BNN trust-region arm {trust_region:?}"))?;
        Ok(Self {
            sign_space_enabled: sign_space.unwrap_or(false),
            minimal_move: minimal_move.unwrap_or(false),
            trust_region,
            ste_pgd_enabled: ste_pgd.unwrap_or(false),
        })
    }

    /// Render the singular policy record consumed by the search validator.
    ///
    /// Keeping the complete schema in one pure function makes field drift
    /// testable without depending on the process-global flight recorder.
    pub(crate) fn flight_reason(self, lower: Self, mip_compiled: bool) -> String {
        format!(
            "schema=ny-cli/bnn-falsification-policy/v1 mip_compiled={mip_compiled} \
             sign_space_enabled={} minimal_move={} trust_region={} ste_pgd_enabled={} \
             lower_sign_space_enabled={} lower_minimal_move={} lower_trust_region={} \
             lower_ste_pgd_enabled={}",
            self.sign_space_enabled,
            self.minimal_move,
            self.trust_region.stable_token(),
            self.ste_pgd_enabled,
            lower.sign_space_enabled,
            lower.minimal_move,
            lower.trust_region.stable_token(),
            lower.ste_pgd_enabled,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{BnnFalsificationPolicy, SignSpaceTrustRegionPolicy};

    fn frozen_set(
        raw: &[(&str, &str)],
        config: &[(&str, ny_levers::LeverValue)],
    ) -> ny_levers::LeverSet {
        let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |name| {
            raw.iter()
                .find_map(|(candidate, value)| (*candidate == name).then(|| (*value).into()))
        });
        ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
            config
                .iter()
                .find_map(|(candidate, value)| (*candidate == decl.name).then(|| value.clone()))
        })
        .expect("BNN test configuration must be admissible")
    }

    #[test]
    fn typed_lower_policy_preserves_unset_as_full_box() {
        let policy =
            BnnFalsificationPolicy::from_config_values(Some(true), None, None, Some(false))
                .unwrap();
        assert!(policy.sign_space_enabled);
        assert!(!policy.minimal_move);
        assert_eq!(policy.trust_region, SignSpaceTrustRegionPolicy::FullBox);
        assert!(!policy.ste_pgd_enabled);
    }

    #[test]
    fn typed_lower_policy_rejects_an_unknown_trust_region() {
        assert!(
            BnnFalsificationPolicy::from_config_values(None, None, Some("wide"), None).is_err()
        );
    }

    #[test]
    fn scored_policy_is_frozen_from_one_layered_receipt() {
        use ny_levers::decls::dark_probes::{
            BNN_SIGN_SPACE_LANE, BNN_SIGN_SPACE_MINIMAL_MOVE, BNN_SIGN_SPACE_TRUST_REGION,
            BNN_STE_PGD_LANE,
        };
        use ny_levers::LeverValue;

        let config = [
            (BNN_SIGN_SPACE_LANE.name, LeverValue::Bool(true)),
            (BNN_SIGN_SPACE_MINIMAL_MOVE.name, LeverValue::Bool(false)),
            (
                BNN_SIGN_SPACE_TRUST_REGION.name,
                LeverValue::Text("box".to_string()),
            ),
            (BNN_STE_PGD_LANE.name, LeverValue::Bool(true)),
        ];
        let frozen = BnnFalsificationPolicy::from_levers(&frozen_set(&[], &config)).unwrap();
        assert_eq!(
            frozen,
            BnnFalsificationPolicy {
                sign_space_enabled: true,
                minimal_move: false,
                trust_region: SignSpaceTrustRegionPolicy::Box,
                ste_pgd_enabled: true,
            }
        );

        // A later snapshot can resolve every axis differently, but it cannot
        // mutate the plain policy already carried into the scored call graph.
        let later = BnnFalsificationPolicy::from_levers(&frozen_set(
            &[
                (BNN_SIGN_SPACE_LANE.name, "0"),
                (BNN_SIGN_SPACE_MINIMAL_MOVE.name, "1"),
                (BNN_SIGN_SPACE_TRUST_REGION.name, "linf"),
                (BNN_STE_PGD_LANE.name, "0"),
            ],
            &config,
        ))
        .unwrap();
        assert_eq!(
            later,
            BnnFalsificationPolicy {
                sign_space_enabled: false,
                minimal_move: true,
                trust_region: SignSpaceTrustRegionPolicy::Linf,
                ste_pgd_enabled: false,
            }
        );
        assert!(frozen.sign_space_enabled);
        assert_eq!(frozen.trust_region, SignSpaceTrustRegionPolicy::Box);
    }

    #[test]
    fn policy_flight_reason_is_exact_and_distinguishes_unset() {
        let effective = BnnFalsificationPolicy {
            sign_space_enabled: false,
            minimal_move: true,
            trust_region: SignSpaceTrustRegionPolicy::Linf,
            ste_pgd_enabled: true,
        };
        let lower = BnnFalsificationPolicy {
            sign_space_enabled: true,
            minimal_move: false,
            trust_region: SignSpaceTrustRegionPolicy::FullBox,
            ste_pgd_enabled: true,
        };
        assert_eq!(
            effective.flight_reason(lower, true),
            "schema=ny-cli/bnn-falsification-policy/v1 mip_compiled=true \
             sign_space_enabled=false minimal_move=true trust_region=linf \
             ste_pgd_enabled=true lower_sign_space_enabled=true \
             lower_minimal_move=false lower_trust_region=full_box \
             lower_ste_pgd_enabled=true"
        );
    }
}

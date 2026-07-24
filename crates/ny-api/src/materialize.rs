// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Stable device-to-bounds boundary for external integrations.
//!
//! ny's public verification APIs consume [`BoundedTensor`] values:
//! host-resident `f32` lower/upper bounds with validation enforced at
//! construction time. Public bounds sources may be external device-backed or
//! lazy wrappers, or in-repo certified carriers such as
//! `composition::BoundCertificate` behind the `propagate` feature, but they
//! must all materialize concrete host bounds before crossing the verification
//! boundary.
//!
//! Contract:
//! - Resolve any lazy/device work in the external runtime.
//! - Materialize lower/upper `f32` bounds on the host.
//! - Construct a [`BoundedTensor`] with [`BoundedTensor::new`] or
//!   [`BoundedTensor::new_allow_infinite`].
//! - Pass the resulting bounds into ny verification APIs.
//!
//! ny may execute internal IBP/CROWN phases on GPU, but those device
//! details stay internal. The stable exchange type at the public boundary
//! remains [`BoundedTensor`].

use ny_core::Result;
use ny_tensor::{BoundedScalar, BoundedTensor, GenericBounds};

/// Source of verification bounds for the public ny API surface.
///
/// Implementers may be external device-backed/lazy tensor wrappers or internal
/// certified carriers such as `BoundCertificate` under the `propagate`
/// feature. The implementation is responsible for producing a validated host
/// [`BoundedTensor`] while hiding storage details from downstream spec
/// construction.
pub trait VerificationBoundsSource {
    /// Materialize host `f32` bounds for verification.
    fn materialize_bounds(&self) -> Result<BoundedTensor>;
}

impl VerificationBoundsSource for BoundedTensor {
    fn materialize_bounds(&self) -> Result<BoundedTensor> {
        Ok(self.clone())
    }
}

impl<T: BoundedScalar> VerificationBoundsSource for GenericBounds<T> {
    fn materialize_bounds(&self) -> Result<BoundedTensor> {
        Ok(self.as_f32().clone())
    }
}

impl<T: VerificationBoundsSource + ?Sized> VerificationBoundsSource for &T {
    fn materialize_bounds(&self) -> Result<BoundedTensor> {
        (*self).materialize_bounds()
    }
}

/// Certificates are first-class bounds sources for pipeline chaining.
///
/// This lets `SpecBuilder::try_input_source(&certificate)` work directly,
/// so consumers don't need to know which field contains the reusable bounds.
#[cfg(feature = "propagate")]
impl VerificationBoundsSource for ny_propagate::composition::certificate::BoundCertificate {
    fn materialize_bounds(&self) -> Result<BoundedTensor> {
        Ok(self.output_bounds().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::VerificationBoundsSource;
    use ndarray::arr1;
    use ny_tensor::{BoundedTensor, GenericBounds};

    #[test]
    fn bounded_tensor_source_returns_equivalent_host_bounds() {
        let bounds = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.5_f32]).into_dyn(),
            arr1(&[2.0_f32, 1.5_f32]).into_dyn(),
        )
        .expect("valid test bounds");

        let materialized = bounds.materialize_bounds().expect("host bounds clone");

        assert_eq!(materialized.lower(), bounds.lower());
        assert_eq!(materialized.upper(), bounds.upper());
    }

    #[test]
    fn generic_bounds_source_returns_inner_f32_bounds() {
        let bounds = GenericBounds::<f64>::new(
            arr1(&[-1.0_f64, 0.5_f64]).into_dyn(),
            arr1(&[2.0_f64, 1.5_f64]).into_dyn(),
        )
        .expect("valid generic bounds");

        let materialized = bounds.materialize_bounds().expect("generic -> host bounds");

        assert_eq!(materialized.lower(), bounds.as_f32().lower());
        assert_eq!(materialized.upper(), bounds.as_f32().upper());
    }

    #[test]
    fn reference_source_forwards_to_underlying_materializer() {
        let bounds = BoundedTensor::new(
            arr1(&[0.0_f32, 1.0_f32]).into_dyn(),
            arr1(&[0.0_f32, 2.0_f32]).into_dyn(),
        )
        .expect("valid test bounds");
        let source = &bounds;

        let materialized = source
            .materialize_bounds()
            .expect("reference impl should forward");

        assert_eq!(materialized.lower(), bounds.lower());
        assert_eq!(materialized.upper(), bounds.upper());
    }
}

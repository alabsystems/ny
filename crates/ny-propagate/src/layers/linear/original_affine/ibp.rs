// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::super::bias::{add_f64_down, f64_to_f32_down_no_subnormal, f64_to_f32_up_no_subnormal};
use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

fn geometry(outputs: usize, inputs: usize, shape: &[usize]) -> Result<(usize, usize)> {
    OriginalAffineView::validation_request(outputs, inputs, false)?;
    if shape.is_empty() || shape.len() > MAX_RANK || shape.last() != Some(&inputs) {
        return Err(invalid("IBP rank/input feature mismatch"));
    }
    let mut positions = 1usize;
    for &dimension in &shape[..shape.len() - 1] {
        positions = mul(positions, dimension)?;
    }
    let input = cap_elements(mul(positions, inputs)?)?;
    let output = cap_elements(mul(positions, outputs)?)?;
    Ok((input, output))
}

pub(crate) fn ibp_request(outputs: usize, inputs: usize, shape: &[usize]) -> Result<WorkRequest> {
    let (input, output) = geometry(outputs, inputs, shape)?;
    let macs = mul(output, inputs)?;
    let mut request = output_request(
        mul(output, 2)?,
        add(
            add(mul(macs, 32)?, mul(input, 8)?)?,
            add(mul(output, 24)?, mul(shape.len(), 16)?)?,
        )?,
    )?;
    request.persistent_bytes = add(request.persistent_bytes, add(mul(shape.len(), 16)?, 512)?)?;
    Ok(request)
}

/// Mathematical-real W*x+b enclosure over the original f32 coefficient bits.
/// No device/diagnostic arithmetic is used as proof authority. Finite source
/// endpoints are the initial profile; unsupported infinities refuse.
pub(crate) fn ibp(
    parameters: OriginalAffineView<'_>,
    input: &BoundedTensor,
    phase: &mut OriginalAffinePhase,
) -> Result<BoundedTensor> {
    phase.entry()?;
    let (input_len, output_len) = geometry(
        parameters.out_features(),
        parameters.in_features(),
        input.shape(),
    )?;
    if input.lower().shape() != input.upper().shape()
        || input.lower().len() != input_len
        || input.upper().len() != input_len
    {
        return Err(invalid("IBP lower/upper source shape mismatch"));
    }
    phase.spend(as_work(mul(input.shape().len(), 8)?)?)?;
    let lower = input
        .lower()
        .as_slice()
        .ok_or_else(|| invalid("IBP requires standard-layout source bounds"))?;
    let upper = input
        .upper()
        .as_slice()
        .ok_or_else(|| invalid("IBP requires standard-layout source bounds"))?;
    for (&lo, &hi) in lower.iter().zip(upper) {
        phase.spend(8)?;
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(numeric("nonfinite/inverted IBP input endpoints"));
        }
    }
    phase.retain_bytes(add(mul(input.shape().len(), 16)?, 512)?)?;
    let mut out_lower = phase.output_vec(output_len)?;
    let mut out_upper = phase.output_vec(output_len)?;
    let outputs = parameters.out_features();
    let features = parameters.in_features();
    for output in 0..output_len {
        let position = output / outputs;
        let row = output % outputs;
        let mut lo = parameters.bias_at(row);
        let mut hi = lo;
        for column in 0..features {
            phase.spend(32)?;
            let w = f32_to_f64_exact(parameters.weight[[row, column]]);
            let l = f32_to_f64_exact(lower[position * features + column]);
            let u = f32_to_f64_exact(upper[position * features + column]);
            let (a, b) = if w >= 0.0 { (l, u) } else { (u, l) };
            // Products of two binary32 values are exact in binary64, including
            // bit-decoded original subnormals. Every sum is rounded outward.
            lo = add_f64_down(lo, w * a);
            hi = add_f64_up(hi, w * b);
        }
        phase.spend(16)?;
        out_lower[output] = f64_to_f32_down_no_subnormal(lo);
        out_upper[output] = f64_to_f32_up_no_subnormal(hi);
    }
    let mut shape = input.shape().to_vec();
    *shape
        .last_mut()
        .ok_or_else(|| invalid("empty output shape"))? = outputs;
    let lower = ArrayD::from_shape_vec(IxDyn(&shape), out_lower)
        .map_err(|_| invalid("IBP output shape"))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&shape), out_upper)
        .map_err(|_| invalid("IBP output shape"))?;
    // Price the constructor's three complete validation scans before it runs;
    // its cooperative checkpoints use this same phase.
    phase.spend(as_work(mul(output_len, 4)?)?)?;
    let result = BoundedTensor::new_allow_infinite_with_poll(lower, upper, || phase.checkpoint())?;
    phase.checkpoint()?;
    Ok(result)
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Literal expected bits come from the archived independent rational oracle.
//! Qualification exercises the same maps, allocations, copies and module used
//! by production, including zero owners and boundaries inside channel/spatial rows.

use super::*;
use ny_core::packed_root_conv::Run;

pub(super) struct Fixture {
    dims: [usize; 6],
    m: Vec<f32>,
    d: Vec<f32>,
    weights: Vec<f32>,
    gather: Vec<usize>,
    offsets: Vec<usize>,
    runs: Vec<Run>,
    pub(super) expected: [Vec<u32>; 2],
}

impl Fixture {
    pub(super) fn request(&self) -> Request<'_> {
        let [source_stride, output_stride, output_spatial, output_channels, spatial_taps, taps] =
            self.dims;
        Request {
            m: &self.m,
            d: &self.d,
            weights: &self.weights,
            gather: &self.gather,
            offsets: &self.offsets,
            runs: &self.runs,
            source_stride,
            output_stride,
            output_spatial,
            output_channels,
            spatial_taps,
            taps,
        }
    }
}

/// Independent closed-form oracle for wide owner windows. Every product and
/// partial sum is an exactly representable dyadic. Let A=(col%16+1)*(sp%31+1)
/// and J=sp*912+col+1. Summing channel factors gives 320*A+J; nine taps give
/// E=9*(320*A+J)<=3,037,248<2^24. For each owner, M products and partial sums
/// are integer multiples of W/8; D uses |W|/4. Distinct signed power-of-two
/// channel weights range from 2^-64 to 2^63; all values stay finite and normal.
/// Expected values use channel-major coordinates directly, without consulting
/// gather, runs, inverse maps, or the production contraction implementation.
/// This supplements the literal cancellation/FMA/subnormal fixtures.
pub(super) fn wide_fixture(spatial: usize) -> Fixture {
    assert!((2..=196).contains(&spatial));
    const CHANNELS: usize = 128;
    const STRIDE: usize = 912;
    const TAPS: usize = CHANNELS * 9;
    let mut m = Vec::with_capacity(CHANNELS * spatial * STRIDE);
    let mut d = Vec::with_capacity(m.capacity());
    for ic in 0..CHANNELS {
        for sp in 0..spatial {
            for col in 0..STRIDE {
                let a = ((col % 16 + 1) * (sp % 31 + 1) * (ic % 4 + 1)
                    + if ic == 0 { sp * STRIDE + col + 1 } else { 0 })
                    as f32;
                m.push(a * 0.125);
                d.push(a * if col % 2 == 0 { 0.25 } else { -0.25 });
            }
        }
    }
    let mut weights = Vec::with_capacity(CHANNELS * TAPS);
    let mut expected = [Vec::with_capacity(m.len()), Vec::with_capacity(m.len())];
    for oc in 0..CHANNELS {
        let w = f32::from_bits(((oc + 63) as u32) << 23) * if oc % 2 == 0 { 1.0 } else { -1.0 };
        weights.extend(std::iter::repeat_n(w, TAPS));
        for sp in 0..spatial {
            for col in 0..STRIDE {
                let a = (col % 16 + 1) * (sp % 31 + 1);
                let e = (9 * (320 * a + sp * STRIDE + col + 1)) as f32;
                expected[0].push((e * w * 0.125).to_bits());
                expected[1].push((e * w.abs() * if col % 2 == 0 { 0.25 } else { -0.25 }).to_bits());
            }
        }
    }
    Fixture {
        dims: [STRIDE, STRIDE, spatial, CHANNELS, 9, TAPS],
        m,
        d,
        weights,
        gather: (0..TAPS)
            .flat_map(|tap| (0..spatial).map(move |sp| tap / 9 * spatial + sp))
            .collect(),
        offsets: (0..=spatial * 9).collect(),
        runs: vec![
            Run {
                dest: 0,
                len: STRIDE - 1
            };
            spatial * 9
        ],
        expected,
    }
}

pub(super) fn limits() -> Limits {
    Limits {
        deadline: None,
        max_new_host_bytes: 16 << 20,
        max_device_bytes: 16 << 20,
        max_dispatch_steps: MAX_STEPS,
    }
}

struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn word(&mut self) -> Result<u32> {
        let bytes = self
            .0
            .get(..4)
            .ok_or_else(|| error("truncated packed CUDA fixture"))?;
        let word = u32::from_le_bytes(bytes.try_into().map_err(|_| error("fixture word"))?);
        self.0 = &self.0[4..];
        Ok(word)
    }
    fn vector(&mut self) -> Result<Vec<u32>> {
        let len = self.word()? as usize;
        if len > self.0.len() / 4 {
            return Err(error("packed CUDA fixture vector length"));
        }
        let mut result = allocate(len, 0, None)?;
        for word in &mut result {
            *word = self.word()?;
        }
        Ok(result)
    }
}

pub(super) fn fixtures(bytes: &[u8]) -> Result<Vec<Fixture>> {
    if !bytes.starts_with(b"NYPRKAT1") {
        return Err(error("packed CUDA fixture magic"));
    }
    let mut reader = Reader(&bytes[8..]);
    if reader.word()? != 9 {
        return Err(error("packed CUDA fixture count"));
    }
    let mut fixtures = Vec::with_capacity(9);
    for _ in 0..9 {
        let mut dims = [0; 6];
        for dim in &mut dims {
            *dim = reader.word()? as usize;
        }
        let m = reader.vector()?.into_iter().map(f32::from_bits).collect();
        let d = reader.vector()?.into_iter().map(f32::from_bits).collect();
        let weights = reader.vector()?.into_iter().map(f32::from_bits).collect();
        let gather = reader
            .vector()?
            .into_iter()
            .map(|x| {
                if x == u32::MAX {
                    usize::MAX
                } else {
                    x as usize
                }
            })
            .collect();
        let offsets = reader.vector()?.into_iter().map(|x| x as usize).collect();
        let runs = reader.vector()?;
        if runs.len() % 2 != 0 {
            return Err(error("packed CUDA fixture run pair"));
        }
        let runs = runs
            .chunks_exact(2)
            .map(|x| Run {
                dest: x[0] as usize,
                len: x[1] as usize,
            })
            .collect();
        let expected = [reader.vector()?, reader.vector()?];
        let fixture = Fixture {
            dims,
            m,
            d,
            weights,
            gather,
            offsets,
            runs,
            expected,
        };
        let layout = fixture
            .request()
            .layout(limits())?
            .ok_or_else(|| error("packed CUDA fixture refused by host contract"))?;
        if fixture
            .expected
            .iter()
            .any(|v| v.len() != layout.output_len)
        {
            return Err(error("packed CUDA fixture expected length"));
        }
        fixtures.push(fixture);
    }
    if !reader.0.is_empty() {
        return Err(error("packed CUDA fixture trailing bytes"));
    }
    Ok(fixtures)
}

fn mismatch(state: &State, message: String) -> NyError {
    state.faulted.store(true, Ordering::Release);
    error(message)
}

pub(super) fn check_chunk(
    state: &State,
    r: &Request<'_>,
    buffers: &[Buffer<'_>],
    output: &mut Output,
    audit: Audit<'_>,
    completed: usize,
) -> Result<()> {
    let expected = audit
        .expected
        .ok_or_else(|| error("missing packed CUDA audit oracle"))?;
    state.download(buffers[5].pointer(), &mut output.m, None)?;
    state.download(buffers[6].pointer(), &mut output.d, None)?;
    for (lane, actual) in [&output.m, &output.d].into_iter().enumerate() {
        if actual.len() != expected[lane].len() {
            return Err(error("packed CUDA oracle shape"));
        }
        for (i, value) in actual.iter().enumerate() {
            let row = i / r.output_stride;
            let column = i % r.output_stride;
            let oc = row / r.output_spatial;
            let sp = row % r.output_spatial;
            let owner = (sp * r.output_channels + oc) * r.output_stride + column;
            let wanted = if owner < completed {
                expected[lane][i]
            } else {
                POISON
            };
            if value.to_bits() != wanted {
                return Err(mismatch(state, format!("packed CUDA KAT lane={lane} index={i} completed={completed}: {:08x} != {wanted:08x}", value.to_bits())));
            }
        }
    }
    state.guards(buffers, None)
}

fn check_input(
    state: &State,
    buffer: &Buffer<'_>,
    expected: impl Iterator<Item = u32>,
) -> Result<()> {
    let mut expected = expected;
    let mut scratch = [0u32; 1024];
    for start in (0..buffer.words).step_by(scratch.len()) {
        let count = scratch.len().min(buffer.words - start);
        state.download(
            buffer.pointer() + (start * 4) as u64,
            &mut scratch[..count],
            None,
        )?;
        for &actual in &scratch[..count] {
            if Some(actual) != expected.next() {
                return Err(mismatch(state, "packed CUDA input changed".into()));
            }
        }
    }
    if expected.next().is_some() {
        return Err(error("packed CUDA audit input length"));
    }
    Ok(())
}

pub(super) fn check_inputs(
    state: &State,
    buffers: &[Buffer<'_>],
    r: &Request<'_>,
    gather: &[u32],
    inverse: &[u32],
) -> Result<()> {
    for (buffer, input) in buffers.iter().zip([r.m, r.d, r.weights]) {
        check_input(state, buffer, input.iter().map(|v| v.to_bits()))?;
    }
    check_input(state, &buffers[3], gather.iter().copied())?;
    check_input(state, &buffers[4], inverse.iter().copied())
}

pub(super) fn qualify(state: &State) -> Result<()> {
    let mut cases = fixtures(FIXTURES)?;
    cases.push(wide_fixture(2));
    for (index, fixture) in cases.iter().enumerate() {
        for chunked in [false, true] {
            let audit = Audit {
                expected: Some([&fixture.expected[0], &fixture.expected[1]]),
                chunked,
                ..Audit::production()
            };
            state
                .execute(&fixture.request(), limits(), audit)?
                .ok_or_else(|| error(format!("packed CUDA KAT fixture {index} declined")))?;
        }
    }
    // The arithmetic fixtures are deliberately small. Independently cover the
    // second host-copy chunk in both directions with exact, nonuniform bits.
    let words = COPY_WORDS + 1;
    let mut input = allocate(words, 0u32, None)?;
    for (i, value) in input.iter_mut().enumerate() {
        *value = (i as u32).wrapping_mul(0x9e37_79b9) ^ 0xa536_6b4d;
    }
    let mut output = allocate(words, 0u32, None)?;
    let mut buffer = state.buffer(words, None)?;
    state.upload(buffer.pointer(), &input, None)?;
    state.download(buffer.pointer(), &mut output, None)?;
    if input != output {
        return Err(mismatch(
            state,
            "packed CUDA multi-chunk copy mismatch".into(),
        ));
    }
    state.guards(std::slice::from_ref(&buffer), None)?;
    buffer.close()?;
    Ok(())
}

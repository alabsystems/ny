// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_propagate::certify::residency::{CancellationToken, MemoryAuthority};
use ny_propagate::certify::work::{WorkLimits, WorkRefusal, WorkReserve};
use sha2::Sha256;
use std::cell::RefCell;
use std::io::Write;
use std::time::{Duration, Instant};

thread_local! {
    static CANCEL_AFTER_BLOCK: RefCell<Option<(usize, CancellationToken)>> = const { RefCell::new(None) };
}

/// Deterministically interrupt a real load after a completed full block. The
/// thread-local hook neither changes production policy nor grants new work.
pub(super) fn after_decoded_block() {
    CANCEL_AFTER_BLOCK.with(|slot| {
        let mut state = slot.borrow_mut();
        if let Some((remaining, cancellation)) = state.as_mut() {
            *remaining -= 1;
            if *remaining == 0 {
                cancellation.cancel();
                *state = None;
            }
        }
    });
}

struct CancelHook;
impl CancelHook {
    fn arm(after: usize, token: &CancellationToken) -> Self {
        assert!(after > 0);
        CANCEL_AFTER_BLOCK.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some((after, token.clone()));
        });
        Self
    }
}
impl Drop for CancelHook {
    fn drop(&mut self) {
        CANCEL_AFTER_BLOCK.with(|slot| *slot.borrow_mut() = None);
    }
}

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::F32 => "F32",
        _ => panic!("fixture dtype"),
    }
}

fn width(dtype: Dtype) -> usize {
    if dtype == Dtype::F32 {
        4
    } else {
        2
    }
}

fn original_bits(dtype: Dtype, index: usize) -> u32 {
    // Both signed zeros, minimum/maximum subnormals, binade boundaries and
    // cancellation signs are repeated across every full and partial block.
    let values: &[u32] = match dtype {
        Dtype::F16 => &[
            0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x3555, 0xb555,
            0x7bff, 0xfbff,
        ],
        Dtype::BF16 => &[
            0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x3eab, 0xbeab,
            0x7f7f, 0xff7f,
        ],
        Dtype::F32 => &[
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x3eaa_aaab,
            0xbeaa_aaab,
            0x7f7f_ffff,
            0xff7f_ffff,
        ],
        _ => panic!("fixture dtype"),
    };
    values[index % values.len()]
}

/// Independent integer IEEE decoding: no half conversion or loader helper is
/// used to derive the expected F32 bit sequence.
fn decoded_bits(dtype: Dtype, original: u32) -> u32 {
    match dtype {
        Dtype::F32 => original,
        Dtype::BF16 => original << 16,
        Dtype::F16 => {
            let sign = (original & 0x8000) << 16;
            let exponent = (original >> 10) & 31;
            let fraction = original & 0x03ff;
            match exponent {
                0 if fraction == 0 => sign,
                0 => {
                    let shift = fraction.leading_zeros() - 21;
                    sign | ((113 - shift) << 23) | (((fraction << shift) & 0x03ff) << 13)
                }
                31 => sign | 0x7f80_0000 | (fraction << 13),
                _ => sign | ((exponent + 112) << 23) | (fraction << 13),
            }
        }
        _ => panic!("fixture dtype"),
    }
}

struct Fixture {
    file: tempfile::NamedTempFile,
    name: &'static str,
    count: usize,
    raw: Vec<u8>,
    header_bytes: usize,
}

fn fixture(dtype: Dtype, count: usize, poison: Option<(usize, u32)>) -> Fixture {
    let name = "original";
    let mut raw = Vec::new();
    for index in 0..count {
        let bits = match poison {
            Some((at, bits)) if index == at => bits,
            _ => original_bits(dtype, index),
        };
        raw.extend_from_slice(&bits.to_le_bytes()[..width(dtype)]);
    }
    let mut entries = serde_json::Map::new();
    entries.insert(
        name.into(),
        serde_json::json!({"dtype":dtype_name(dtype),"shape":[count],"data_offsets":[0,raw.len()]}),
    );
    let header = serde_json::to_vec(&entries).unwrap();
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&raw).unwrap();
    file.flush().unwrap();
    Fixture {
        file,
        name,
        count,
        raw,
        header_bytes: header.len(),
    }
}

/// Literal old scalar-stream format, independently rebuilding the domain,
/// tensor identity header and one four-byte SHA update per coefficient.
fn scalar_hash_start(name: &str, count: usize, dtype: &str) -> Sha256 {
    let mut hash = Sha256::new();
    hash.update(b"ny-original-decoder-tensor/v1");
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update((dtype.len() as u64).to_le_bytes());
    hash.update(dtype.as_bytes());
    hash.update(1u64.to_le_bytes());
    hash.update((count as u64).to_le_bytes());
    hash
}

fn source_floor(fixture: &Fixture) -> usize {
    8 + fixture.header_bytes
        + fixture.raw.len()
        + 16 * fixture.header_bytes
        + 4096
        + DECODED_HASH_BLOCK_BYTES
}

fn resident_floor(fixture: &Fixture) -> usize {
    fixture.count * 4 + 1024 + fixture.name.len() * 4 + 32
}

fn paid_options(
    fixture: &Fixture,
    local_scratch: usize,
    ledger_scratch: usize,
    cancellation: &CancellationToken,
) -> (DecoderOptions, WorkBudget, MemoryAuthority) {
    let resident = resident_floor(fixture);
    let memory = MemoryAuthority::new(source_floor(fixture) + resident);
    let deadline = Some(Instant::now() + Duration::from_secs(30));
    let budget = WorkBudget::new_in(
        &memory,
        cancellation,
        WorkLimits {
            deadline,
            scratch_bytes: ledger_scratch,
            persistent_bytes: resident,
            max_work: (8
                + fixture.header_bytes
                + fixture.raw.len()
                + DECODED_HASH_BLOCK_BYTES
                + fixture.count) as u64,
            max_proposals: 2,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let options = DecoderOptions {
        deadline,
        budget: Some(budget.clone()),
        max_scratch_bytes: local_scratch,
        max_resident_bytes: resident,
        ..DecoderOptions::default()
    };
    (options, budget, memory)
}

fn assert_released(budget: &WorkBudget, memory: &MemoryAuthority) {
    let receipt = budget.receipt().unwrap();
    assert_eq!(receipt.live_scratch_bytes, 0);
    assert_eq!(receipt.live_persistent_bytes, 0);
    let receipt = memory.receipt().unwrap();
    assert_eq!(receipt.live_pending_bytes, 0);
    assert_eq!(receipt.live_materialized_bytes, 0);
}

#[test]
fn decoder_hash_block_matches_independent_old_scalar_stream_all_encodings_and_boundaries() {
    for dtype in [Dtype::F16, Dtype::BF16, Dtype::F32] {
        for count in [0, 1, 4095, 4096, 4097, 8193] {
            let fixture = fixture(dtype, count, None);
            let loaded =
                load_decoder_safetensors(fixture.file.path(), &DecoderOptions::default()).unwrap();
            let identity = &loaded.identities[fixture.name];
            let mut source = scalar_hash_start(fixture.name, count, dtype_name(dtype));
            source.update(&fixture.raw);
            let mut decoded = scalar_hash_start(fixture.name, count, "F32-decoded");
            let mut expected = Vec::new();
            for raw in fixture.raw.chunks_exact(width(dtype)) {
                let mut word = [0u8; 4];
                word[..raw.len()].copy_from_slice(raw);
                let bits = decoded_bits(dtype, u32::from_le_bytes(word));
                decoded.update(bits.to_le_bytes());
                expected.push(bits);
            }
            let actual: Vec<_> = loaded
                .weights
                .get(fixture.name)
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect();
            assert_eq!(actual, expected, "{dtype:?}, count={count}");
            let source_digest: [u8; 32] = source.finalize().into();
            let decoded_digest: [u8; 32] = decoded.finalize().into();
            assert_eq!(identity.source_digest, source_digest);
            assert_eq!(identity.decoded_f32_digest, decoded_digest);
            assert_eq!(identity.source_dtype, dtype_name(dtype));
            assert_eq!(identity.shape, vec![count]);
            assert_eq!(loaded.coefficients, count as u64);
            assert_eq!(loaded.source_bytes, fixture.raw.len() as u64);
        }
    }
}

#[test]
fn decoder_hash_block_scratch_floor_is_prepaid_and_last_owner_releases_ledger() {
    let fixture = fixture(Dtype::F32, 4097, None);
    let floor = source_floor(&fixture);
    let cancellation = CancellationToken::default();
    let (options, budget, memory) = paid_options(&fixture, floor - 1, floor, &cancellation);
    assert!(matches!(
        load_decoder_safetensors(fixture.file.path(), &options),
        Err(Refusal::Resource("checkpoint parser scratch cap"))
    ));
    assert_eq!(budget.receipt().unwrap().attempted, 0);
    assert_eq!(memory.receipt().unwrap().attempted, 0);
    assert_released(&budget, &memory);

    let (options, budget, memory) = paid_options(&fixture, floor, floor - 1, &cancellation);
    assert!(matches!(
        load_decoder_safetensors(fixture.file.path(), &options),
        Err(Refusal::Budget(WorkRefusal::ScratchLimit))
    ));
    assert_eq!(budget.receipt().unwrap().admitted, 0);
    assert_eq!(memory.receipt().unwrap().attempted, 0);
    assert_released(&budget, &memory);

    let (options, budget, memory) = paid_options(&fixture, floor, floor, &cancellation);
    let loaded = load_decoder_safetensors(fixture.file.path(), &options).unwrap();
    let receipt = budget.receipt().unwrap();
    assert_eq!(receipt.admitted, 2);
    assert_eq!(receipt.completed, 2);
    assert_eq!(receipt.live_scratch_bytes, 0);
    assert_eq!(receipt.live_persistent_bytes, loaded.reserved_bytes());
    assert_eq!(receipt.peak_leased_bytes, floor + loaded.reserved_bytes());
    assert_eq!(
        receipt.work_reserved,
        (8 + fixture.header_bytes + fixture.raw.len() + DECODED_HASH_BLOCK_BYTES + fixture.count)
            as u64
    );
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        loaded.reserved_bytes()
    );
    drop(loaded);
    assert_released(&budget, &memory);

    // This bounded optimization does not raise from_store's old 4096-byte
    // scratch floor or change its separately labelled decoded-input identity.
    let mut store = WeightStore::new();
    store.insert(
        "explicit".into(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.0, f32::from_bits(1)]).unwrap(),
    );
    let loaded = DecoderWeights::from_store(
        store,
        &DecoderOptions {
            max_scratch_bytes: 4096,
            ..DecoderOptions::default()
        },
    )
    .unwrap();
    let mut expected = scalar_hash_start("explicit", 2, "F32-decoded-input");
    expected.update((-0.0f32).to_bits().to_le_bytes());
    expected.update(1u32.to_le_bytes());
    let expected: [u8; 32] = expected.finalize().into();
    assert_eq!(loaded.identities["explicit"].source_digest, expected);
    assert_eq!(loaded.identities["explicit"].decoded_f32_digest, expected);
}

#[test]
fn decoder_hash_block_cancellation_after_full_block_refuses_partial_load_and_releases() {
    let fixture = fixture(Dtype::BF16, 4097, None);
    let cancellation = CancellationToken::default();
    let floor = source_floor(&fixture);
    let (options, budget, memory) = paid_options(&fixture, floor, floor, &cancellation);
    let _hook = CancelHook::arm(1, &cancellation);
    assert!(matches!(
        load_decoder_safetensors(fixture.file.path(), &options),
        Err(Refusal::Budget(WorkRefusal::Cancelled))
    ));
    assert!(cancellation.is_cancelled());
    let receipt = budget.receipt().unwrap();
    assert_eq!(receipt.admitted, 2);
    assert_eq!(receipt.completed, 0);
    assert_eq!(
        receipt.work_reserved,
        (8 + fixture.header_bytes + fixture.raw.len() + DECODED_HASH_BLOCK_BYTES + fixture.count)
            as u64
    );
    assert_released(&budget, &memory);
}

#[test]
fn decoder_hash_block_nonfinite_first_or_partial_block_refuses_all_encodings_and_releases() {
    for (dtype, poisons) in [
        (Dtype::F16, [0x7c00, 0xfc00, 0x7e01]),
        (Dtype::BF16, [0x7f80, 0xff80, 0x7fc1]),
        (Dtype::F32, [0x7f80_0000, 0xff80_0000, 0x7fc0_0001]),
    ] {
        for poison in poisons {
            for index in [0, 4096] {
                let fixture = fixture(dtype, 4097, Some((index, poison)));
                let cancellation = CancellationToken::default();
                let floor = source_floor(&fixture);
                let (options, budget, memory) = paid_options(&fixture, floor, floor, &cancellation);
                assert!(matches!(
                    load_decoder_safetensors(fixture.file.path(), &options),
                    Err(Refusal::Tensor(message)) if message == "nonfinite original"
                ));
                assert_eq!(budget.receipt().unwrap().admitted, 2);
                assert_eq!(budget.receipt().unwrap().completed, 0);
                assert_released(&budget, &memory);
            }
        }
    }
}

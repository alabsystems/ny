use super::*;

fn request<'a>(
    m: &'a [f32],
    gather: &'a [usize],
    offsets: &'a [usize],
    runs: &'a [Run],
) -> Request<'a> {
    Request {
        m,
        d: m,
        weights: &[1.0, -2.0],
        gather,
        offsets,
        runs,
        source_stride: 4,
        output_stride: 6,
        output_spatial: 1,
        output_channels: 1,
        spatial_taps: 2,
        taps: 2,
    }
}

fn limits() -> Limits {
    Limits {
        deadline: None,
        max_new_host_bytes: usize::MAX,
        max_device_bytes: usize::MAX,
        max_dispatch_steps: 12,
    }
}

#[test]
fn fragmented_plan_accounts_both_lanes_and_complete_device_payload() {
    let r = request(
        &[0.0; 8],
        &[0, 1],
        &[0, 2, 3],
        &[
            Run { dest: 0, len: 2 },
            Run { dest: 4, len: 1 },
            Run { dest: 1, len: 3 },
        ],
    );
    let layout = r.layout(limits()).unwrap().unwrap();
    assert_eq!(layout.output_len, 6);
    assert_eq!(layout.inverse_len, 10);
    assert_eq!(layout.new_host_bytes, (12 + 2 + 10) * 4);
    assert_eq!(layout.device_bytes, (16 + 2 + 12 + 2 + 10) * 4);
    assert_eq!(layout.owners_per_dispatch, 3);
    let mut l = limits();
    l.max_new_host_bytes = layout.new_host_bytes - 1;
    assert!(r.layout(l).unwrap().is_none());
    l = limits();
    l.max_device_bytes = layout.device_bytes - 1;
    assert!(r.layout(l).unwrap().is_none());
    l = limits();
    l.max_dispatch_steps = 3;
    assert!(r.layout(l).unwrap().is_none());
}

#[test]
fn rejects_aliasing_runs_bad_gathers_and_nonfinite_sources() {
    let data = [0.0; 8];
    for runs in [
        [Run { dest: 0, len: 2 }, Run { dest: 1, len: 1 }],
        [Run { dest: 0, len: 4 }, Run { dest: 4, len: 1 }],
        [Run { dest: 0, len: 1 }, Run { dest: 5, len: 1 }],
        [Run { dest: 0, len: 0 }, Run { dest: 1, len: 1 }],
    ] {
        assert!(request(&data, &[0, 1], &[0, 2, 2], &runs)
            .layout(limits())
            .is_err());
    }
    assert!(request(&data, &[2, 1], &[0, 0, 0], &[])
        .layout(limits())
        .is_err());
    assert!(request(&data, &[0, 1], &[0, 1, 0], &[])
        .layout(limits())
        .is_err());
    let mut bad = data;
    bad[4] = f32::NAN;
    assert!(request(&bad, &[0, 1], &[0, 0, 0], &[])
        .layout(limits())
        .is_err());
}

#[test]
fn channel_sharing_padding_and_bias_only_are_checked() {
    let mut r = request(&[0.0; 8], &[0, 1], &[0, 0], &[]);
    r.spatial_taps = 1;
    assert!(r.layout(limits()).unwrap().is_some());
    r.gather = &[0, 0];
    assert!(r.layout(limits()).is_err());
    r.gather = &[usize::MAX, usize::MAX];
    assert!(r.layout(limits()).unwrap().is_some());
    r.m = &[1.0, -1.0];
    let subnormals = [f32::from_bits(1), f32::from_bits(0x80000001)];
    r.d = &subnormals;
    r.source_stride = 1;
    r.output_stride = 1;
    r.gather = &[0, 1];
    let layout = r.layout(limits()).unwrap().unwrap();
    assert_eq!(layout.inverse_len, 0);
    assert_eq!(layout.output_len, 1);
    assert_eq!(layout.device_bytes, (4 + 2 + 2 + 2) * 4);
}

#[test]
fn expired_authority_wins_before_shape_or_memory_admission() {
    let r = request(&[], &[], &[], &[]);
    let mut l = limits();
    l.deadline = Some(Instant::now());
    assert!(r.layout(l).unwrap_err().is_deadline_exceeded());
}

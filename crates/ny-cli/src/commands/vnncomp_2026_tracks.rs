// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Official VNN-COMP 2026 track membership.
//!
//! These category ids are shared by the late-submission workflow and the
//! winner-relative scorer. They match the [benchmark-voting result] announced
//! on 2026-06-07 and the evaluation-platform SPA.
//!
//! [benchmark-voting result]: https://github.com/VNN-COMP/vnncomp2026/issues/6#issuecomment-4643430287

/// Regular categories (at least 50% of the eight tool-author votes).
pub(crate) const REGULAR_TRACK_2026: [&str; 24] = [
    "acasxu_2023",
    "cersyve",
    "cgan2026",
    "challenging_certified_training_2026",
    "cifar100_2024",
    "collins_rul_cnn_2022",
    "cora_2024",
    "dist_shift_2023",
    "linearizenn_2024",
    "lsnc_relu",
    "malbeware",
    "metaroom_2023",
    "ml4acopf_2024",
    "nn4sys",
    "relusplitter_2026",
    "safenlp_2024",
    "sat_relu",
    "soundnessbench_2026",
    "tinyimagenet_2024",
    "tllverifybench_2023",
    "traffic_signs_recognition_2023",
    "vggnet16_2022",
    "vit_2023",
    "yolo_2023",
];

/// Extended categories (at least one vote, but not regular).
pub(crate) const EXTENDED_TRACK_2026: [&str; 6] = [
    "adaptive_cruise_control_non_linear_2026",
    "cctsdb_yolo_2023",
    "collins_aerospace_benchmark",
    "isomorphic_acasxu_2026",
    "monotonic_acasxu_2026",
    "smart_turn_multimodal_2026",
];

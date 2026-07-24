// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conv2d CROWN backward GPU pipeline creation (#3397).
//!
//! Split from crown_pipelines.rs for file size compliance.

use std::borrow::Cow;

use super::shaders::{CONV_COL2IM_SHADER, CONV_RESHAPE_SHADER};
use super::WgpuDevice;

impl WgpuDevice {
    /// Conv2d A-matrix reshape pipeline (#3397).
    /// 3 bindings: params (uniform), src (read), dst (write).
    pub(super) fn create_conv_reshape_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv_reshape_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CONV_RESHAPE_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv_reshape_bind_group_layout"),
            entries: &[
                // 0: Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: src (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: dst (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("conv_reshape_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv_reshape_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    /// Conv2d col2im gather pipeline (#3397).
    /// 3 bindings: params (uniform), gemm_out (read), dst (write).
    pub(super) fn create_conv_col2im_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv_col2im_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CONV_COL2IM_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv_col2im_bind_group_layout"),
            entries: &[
                // 0: Params uniform (64 bytes for ConvCol2imParams)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: gemm_out (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: dst (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("conv_col2im_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv_col2im_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }
}

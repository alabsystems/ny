// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! YAML config loader and resolver for CLI entrypoints.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::{BackendArg, MulBinaryRelaxationArg};

const DEFAULT_METHOD: &str = "alpha";
const DEFAULT_EPSILON: f32 = 0.01;
const DEFAULT_TIMEOUT: u64 = 60;
const DEFAULT_MAX_ITERATIONS: usize = 100;
const DEFAULT_TOLERANCE: f32 = 1e-4;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct VerifyConfigFile {
    pub(crate) general: Option<GeneralConfig>,
    pub(crate) model: Option<ModelConfig>,
    pub(crate) data: Option<DataConfig>,
    pub(crate) specification: Option<SpecificationConfig>,
    pub(crate) solver: Option<SolverConfig>,
    pub(crate) bab: Option<BabSection>,
}

/// The `bab:` config section: a full `BetaCrownConfig` plus the set of keys
/// the YAML actually declared.
///
/// `BetaCrownConfig` is `#[serde(default)]`, so after deserialization a
/// defaulted field is indistinguishable from one the user wrote. The verify
/// pipeline can honor only a subset of this section (see
/// [`apply_bab_section`]), so it needs the declared key set to reject the
/// settings it would otherwise drop without effect.
#[derive(Debug, Clone)]
pub(crate) struct BabSection {
    config: ny_propagate::BetaCrownConfig,
    declared_keys: Vec<String>,
}

impl BabSection {
    /// Top-level keys the YAML wrote in the `bab:` section, in file order.
    pub(crate) fn declared_keys(&self) -> &[String] {
        &self.declared_keys
    }
}

impl std::ops::Deref for BabSection {
    type Target = ny_propagate::BetaCrownConfig;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}

impl Serialize for BabSection {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.config.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BabSection {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        let value = serde_yaml::Value::deserialize(deserializer)?;
        let declared_keys = match &value {
            serde_yaml::Value::Mapping(mapping) => mapping
                .keys()
                .map(|key| {
                    key.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| D::Error::custom("bab section keys must be strings"))
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
            _ => return Err(D::Error::custom("bab section must be a mapping")),
        };
        let config = serde_yaml::from_value(value).map_err(D::Error::custom)?;
        Ok(Self {
            config,
            declared_keys,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct GeneralConfig {
    pub(crate) root_path: Option<PathBuf>,
    pub(crate) csv_name: Option<String>,
    pub(crate) device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct ModelConfig {
    pub(crate) path: Option<PathBuf>,
    pub(crate) onnx_path: Option<PathBuf>,
    pub(crate) input_shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct DataConfig {
    pub(crate) start: Option<usize>,
    pub(crate) end: Option<usize>,
    pub(crate) select_instance: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct SpecificationConfig {
    pub(crate) vnnlib_path: Option<PathBuf>,
    pub(crate) epsilon: Option<f32>,
    pub(crate) norm: Option<String>,
    pub(crate) rhs_offset: Option<f32>,
    pub(crate) peel_off_last_softmax_layer: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct SolverConfig {
    pub(crate) propagation: Option<PropagationOverrides>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct PropagationOverrides {
    pub(crate) method: Option<String>,
    pub(crate) tolerance: Option<f32>,
    pub(crate) max_iterations: Option<usize>,
    pub(crate) use_gpu: Option<bool>,
    pub(crate) mul_binary_relaxation: Option<String>,
    /// Use f64 (double precision) for all bound propagation.
    /// Set `true` for soundnessbench/sat_relu benchmarks.
    pub(crate) double_fp: Option<bool>,
}

#[derive(Debug, Clone)]
pub(crate) struct VerifyInstance {
    pub(crate) model: PathBuf,
    pub(crate) property: PathBuf,
    pub(crate) timeout: Option<u64>,
    pub(crate) index: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedVerifyConfig {
    pub(crate) config: VerifyConfigFile,
    pub(crate) config_path: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct VerifyConfigOverrides {
    pub(crate) model: Option<PathBuf>,
    pub(crate) property: Option<PathBuf>,
    pub(crate) epsilon: Option<f32>,
    pub(crate) method: Option<String>,
    pub(crate) mul_binary_relaxation: Option<MulBinaryRelaxationArg>,
    pub(crate) timeout: Option<u64>,
    pub(crate) backend: Option<BackendArg>,
    pub(crate) max_iterations: Option<usize>,
    pub(crate) tolerance: Option<f32>,
    pub(crate) peel_off_last_softmax_layer: Option<bool>,
    pub(crate) double_fp: Option<bool>,
}

#[derive(Debug, Clone)]
pub(crate) struct VerifySettings {
    pub(crate) model: Option<PathBuf>,
    pub(crate) property: Option<PathBuf>,
    pub(crate) epsilon: f32,
    pub(crate) method: String,
    pub(crate) mul_binary_relaxation: MulBinaryRelaxationArg,
    pub(crate) timeout: u64,
    pub(crate) backend: BackendArg,
    pub(crate) max_iterations: usize,
    pub(crate) tolerance: f32,
    pub(crate) peel_off_last_softmax_layer: bool,
    pub(crate) double_fp: bool,
    pub(crate) config_path: Option<PathBuf>,
}

impl VerifySettings {
    fn from_defaults() -> Self {
        Self {
            model: None,
            property: None,
            epsilon: DEFAULT_EPSILON,
            method: DEFAULT_METHOD.to_string(),
            mul_binary_relaxation: MulBinaryRelaxationArg::Mccormick,
            timeout: DEFAULT_TIMEOUT,
            backend: BackendArg::Cpu,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
            peel_off_last_softmax_layer: false,
            double_fp: false,
            config_path: None,
        }
    }

    fn apply_overrides(&mut self, overrides: VerifyConfigOverrides) {
        if let Some(model) = overrides.model {
            self.model = Some(model);
        }
        if let Some(property) = overrides.property {
            self.property = Some(property);
        }
        if let Some(epsilon) = overrides.epsilon {
            self.epsilon = epsilon;
        }
        if let Some(method) = overrides.method {
            self.method = method;
        }
        if let Some(mul_binary_relaxation) = overrides.mul_binary_relaxation {
            self.mul_binary_relaxation = mul_binary_relaxation;
        }
        if let Some(timeout) = overrides.timeout {
            self.timeout = timeout;
        }
        if let Some(backend) = overrides.backend {
            self.backend = backend;
        }
        if let Some(max_iterations) = overrides.max_iterations {
            self.max_iterations = max_iterations;
        }
        if let Some(tolerance) = overrides.tolerance {
            self.tolerance = tolerance;
        }
        if let Some(peel_off_last_softmax_layer) = overrides.peel_off_last_softmax_layer {
            self.peel_off_last_softmax_layer = peel_off_last_softmax_layer;
        }
        if let Some(double_fp) = overrides.double_fp {
            self.double_fp = double_fp;
        }
    }
}

pub(crate) fn resolve_verify_settings(
    config_path: Option<PathBuf>,
    root_path: Option<PathBuf>,
    cli_overrides: VerifyConfigOverrides,
) -> Result<VerifySettings> {
    let resolved = resolve_verify_config(config_path, root_path.clone())?;
    resolve_verify_settings_from_config(
        resolved.as_ref(),
        root_path.as_deref(),
        cli_overrides,
        VerifyConfigOverrides::default(),
    )
}

pub(crate) fn resolve_verify_settings_from_config(
    resolved: Option<&ResolvedVerifyConfig>,
    cli_root_path: Option<&Path>,
    cli_overrides: VerifyConfigOverrides,
    instance_overrides: VerifyConfigOverrides,
) -> Result<VerifySettings> {
    let mut settings = VerifySettings::from_defaults();
    if let Some(resolved) = resolved {
        let config_overrides = resolved
            .config
            .to_overrides(&resolved.config_path, cli_root_path)?;
        settings.apply_overrides(config_overrides);
        settings.config_path = Some(resolved.config_path.clone());
    }
    settings.apply_overrides(instance_overrides);
    settings.apply_overrides(cli_overrides);
    Ok(settings)
}

pub(crate) fn resolve_verify_config(
    config_path: Option<PathBuf>,
    root_path: Option<PathBuf>,
) -> Result<Option<ResolvedVerifyConfig>> {
    let resolved_path = resolve_config_path(config_path, root_path.as_deref())?;
    let Some(config_path) = resolved_path else {
        return Ok(None);
    };
    let config = load_verify_config(&config_path)?;
    Ok(Some(ResolvedVerifyConfig {
        config,
        config_path,
    }))
}

pub(crate) fn resolve_config_path(
    config_path: Option<PathBuf>,
    root_path: Option<&Path>,
) -> Result<Option<PathBuf>> {
    let resolved = if let Some(path) = config_path {
        Some(path)
    } else {
        root_path.map(|root| root.join("config.yaml"))
    };

    if let Some(path) = &resolved {
        match fs::metadata(path) {
            Ok(metadata) => {
                if !metadata.is_file() {
                    anyhow::bail!("Config path is not a file: {}", path.display());
                }
            }
            Err(err) if err.kind() == ErrorKind::NotFound => {
                anyhow::bail!("Config file not found: {}", path.display());
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("Failed to read config metadata: {}", path.display())
                });
            }
        }
    }

    Ok(resolved)
}

// Justification: CLI override builder — each parameter corresponds to an optional
// command-line flag that overrides the corresponding config file value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cli_overrides(
    model: Option<PathBuf>,
    property: Option<PathBuf>,
    epsilon: Option<f32>,
    method: Option<String>,
    mul_binary_relaxation: Option<MulBinaryRelaxationArg>,
    timeout: Option<u64>,
    backend: Option<BackendArg>,
    peel_off_last_softmax_layer: Option<bool>,
) -> VerifyConfigOverrides {
    let mut overrides = VerifyConfigOverrides::default();

    if let Some(model) = model {
        overrides.model = Some(model);
    }
    if let Some(property) = property {
        overrides.property = Some(property);
    }
    if let Some(epsilon) = epsilon {
        overrides.epsilon = Some(epsilon);
    }
    if let Some(method) = method {
        overrides.method = Some(method.to_lowercase());
    }
    if let Some(mul_binary_relaxation) = mul_binary_relaxation {
        overrides.mul_binary_relaxation = Some(mul_binary_relaxation);
    }
    if let Some(timeout) = timeout {
        overrides.timeout = Some(timeout);
    }
    if let Some(backend) = backend {
        overrides.backend = Some(backend);
    }
    if let Some(peel_off_last_softmax_layer) = peel_off_last_softmax_layer {
        overrides.peel_off_last_softmax_layer = Some(peel_off_last_softmax_layer);
    }

    overrides
}

pub(crate) fn instance_overrides(instance: &VerifyInstance) -> VerifyConfigOverrides {
    VerifyConfigOverrides {
        model: Some(instance.model.clone()),
        property: Some(instance.property.clone()),
        timeout: instance.timeout,
        ..Default::default()
    }
}

pub(crate) fn csv_instances(
    config: &VerifyConfigFile,
    config_path: &Path,
    cli_root_path: Option<&Path>,
) -> Result<Option<Vec<VerifyInstance>>> {
    let csv_name = match config
        .general
        .as_ref()
        .and_then(|general| general.csv_name.as_deref())
    {
        Some(name) => name,
        None => return Ok(None),
    };

    let base_path = effective_root_path(config, config_path, cli_root_path);
    let csv_path = resolve_path(base_path.as_deref(), PathBuf::from(csv_name));
    let contents = fs::read_to_string(&csv_path)
        .with_context(|| format!("Failed to read instances CSV: {}", csv_path.display()))?;

    let mut instances = Vec::new();
    let mut instance_index = 0usize;
    for (line_no, line) in contents.lines().enumerate() {
        let line_number = line_no + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').map(|part| part.trim()).collect();
        if parts.len() < 2 {
            anyhow::bail!(
                "Invalid CSV row at line {}: expected at least 2 columns",
                line_number
            );
        }
        let model_name = parts[0];
        let property_name = parts[1];
        if is_csv_header(model_name, property_name) {
            continue;
        }
        if model_name.is_empty() || property_name.is_empty() {
            anyhow::bail!(
                "Invalid CSV row at line {}: model/property cannot be empty",
                line_number
            );
        }
        let timeout =
            match parts.get(2) {
                Some(value) if !value.is_empty() => Some(value.parse().with_context(|| {
                    format!("Invalid timeout at line {}: {}", line_number, value)
                })?),
                _ => None,
            };

        let model = resolve_path(base_path.as_deref(), PathBuf::from(model_name));
        let property = resolve_path(base_path.as_deref(), PathBuf::from(property_name));
        instances.push(VerifyInstance {
            model,
            property,
            timeout,
            index: instance_index,
        });
        instance_index += 1;
    }

    Ok(Some(instances))
}

fn is_csv_header(model_name: &str, property_name: &str) -> bool {
    let model_lower = model_name.to_lowercase();
    let property_lower = property_name.to_lowercase();
    let model_matches = matches!(
        model_lower.as_str(),
        "network" | "model" | "onnx" | "model_path" | "onnx_path"
    );
    let property_matches = matches!(
        property_lower.as_str(),
        "property" | "vnnlib" | "spec" | "specification" | "vnnlib_path"
    );
    model_matches && property_matches
}

pub(crate) fn select_instances(
    instances: Vec<VerifyInstance>,
    data: Option<&DataConfig>,
) -> Vec<VerifyInstance> {
    let Some(data) = data else {
        return instances;
    };

    if let Some(index) = data.select_instance {
        return instances
            .into_iter()
            .filter(|instance| instance.index == index)
            .collect();
    }

    let start = data.start.unwrap_or(0);
    let mut end = data.end.unwrap_or(instances.len());
    if end > instances.len() {
        end = instances.len();
    }
    if start >= end {
        return Vec::new();
    }

    instances
        .into_iter()
        .filter(|instance| instance.index >= start && instance.index < end)
        .collect()
}

impl VerifyConfigFile {
    fn to_overrides(
        &self,
        config_path: &Path,
        cli_root_path: Option<&Path>,
    ) -> Result<VerifyConfigOverrides> {
        let mut overrides = VerifyConfigOverrides::default();

        let base_path = effective_root_path(self, config_path, cli_root_path);

        if let Some(model) = self
            .model
            .as_ref()
            .and_then(|m| m.path.clone().or(m.onnx_path.clone()))
        {
            overrides.model = Some(resolve_path(base_path.as_deref(), model));
        }

        if let Some(spec) = self.specification.as_ref() {
            if let Some(vnnlib_path) = spec.vnnlib_path.clone() {
                overrides.property = Some(resolve_path(base_path.as_deref(), vnnlib_path));
            }
            if let Some(epsilon) = spec.epsilon {
                overrides.epsilon = Some(epsilon);
            }
            if let Some(peel_off_last_softmax_layer) = spec.peel_off_last_softmax_layer {
                overrides.peel_off_last_softmax_layer = Some(peel_off_last_softmax_layer);
            }
        }

        if let Some(solver) = self.solver.as_ref() {
            if let Some(propagation) = solver.propagation.as_ref() {
                if let Some(method) = propagation.method.as_ref() {
                    overrides.method = Some(method.to_lowercase());
                }
                if let Some(relaxation) = propagation.mul_binary_relaxation.as_ref() {
                    overrides.mul_binary_relaxation =
                        MulBinaryRelaxationArg::from_config_str(&relaxation.to_lowercase());
                }
                if let Some(tolerance) = propagation.tolerance {
                    overrides.tolerance = Some(tolerance);
                }
                if let Some(max_iterations) = propagation.max_iterations {
                    overrides.max_iterations = Some(max_iterations);
                }
                if let Some(use_gpu) = propagation.use_gpu {
                    if use_gpu && overrides.backend.is_none() {
                        overrides.backend = Some(BackendArg::Wgpu);
                    }
                }
                if let Some(double_fp) = propagation.double_fp {
                    overrides.double_fp = Some(double_fp);
                }
            }
        }

        if let Some(general) = self.general.as_ref() {
            if let Some(device) = general.device.as_deref() {
                overrides.backend = Some(parse_backend(device)?);
            }
        }

        if let Some(bab) = self.bab.as_ref() {
            apply_bab_section(bab, &mut overrides)?;
        }

        Ok(overrides)
    }
}

/// Fold the `bab:` section into the verify overrides.
///
/// The verify pipeline builds its β-CROWN engine config from defaults plus the
/// handful of knobs carried by [`VerifyConfigOverrides`], so only the declared
/// keys with a matching override can take effect here. Every other declared
/// key is rejected: a `bab:` setting that parses but changes nothing would run
/// a different search than the one the user configured.
fn apply_bab_section(bab: &BabSection, overrides: &mut VerifyConfigOverrides) -> Result<()> {
    for key in bab.declared_keys() {
        match key.as_str() {
            "timeout" => {
                if bab.timeout.subsec_nanos() != 0 {
                    anyhow::bail!(
                        "bab.timeout must be a whole number of seconds, got {:?}",
                        bab.timeout
                    );
                }
                overrides.timeout = Some(bab.timeout.as_secs());
            }
            key => anyhow::bail!(
                "Unsupported bab setting '{}' in config: the verify pipeline honors only \
                 bab.timeout (iteration and tolerance knobs are set via \
                 solver.propagation.max_iterations / solver.propagation.tolerance)",
                key
            ),
        }
    }
    Ok(())
}

pub(crate) fn config_path_hint(config_path: &Path) -> String {
    format!("Loaded config: {}", config_path.display())
}

fn load_verify_config(path: &Path) -> Result<VerifyConfigFile> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;
    let config: VerifyConfigFile = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?;
    Ok(config)
}

fn effective_root_path(
    config: &VerifyConfigFile,
    config_path: &Path,
    cli_root_path: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(root) = cli_root_path {
        return Some(root.to_path_buf());
    }

    if let Some(root) = config.general.as_ref().and_then(|g| g.root_path.as_ref()) {
        if root.is_absolute() {
            return Some(root.clone());
        }
        if let Some(parent) = config_path.parent() {
            return Some(parent.join(root));
        }
        return Some(root.clone());
    }

    config_path.parent().map(Path::to_path_buf)
}

fn resolve_path(base: Option<&Path>, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    if let Some(base) = base {
        return base.join(path);
    }
    path
}

fn parse_backend(device: &str) -> Result<BackendArg> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(BackendArg::Cpu),
        "wgpu" => Ok(BackendArg::Wgpu),
        _ => anyhow::bail!(
            "Unsupported device '{}' in config (expected cpu, wgpu)",
            device
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_temp_config(config_path: &Path) {
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(config_path, "general:\n  device: cpu\n").unwrap();
    }

    #[test]
    fn cli_overrides_defaults_do_not_override() {
        let overrides = cli_overrides(None, None, None, None, None, None, None, None);
        assert!(overrides.model.is_none());
        assert!(overrides.property.is_none());
        assert!(overrides.epsilon.is_none());
        assert!(overrides.method.is_none());
        assert!(overrides.mul_binary_relaxation.is_none());
        assert!(overrides.timeout.is_none());
        assert!(overrides.backend.is_none());
    }

    #[test]
    fn config_path_infers_from_root() {
        let root = tempdir().unwrap();
        let config_path = root.path().join("config.yaml");
        write_temp_config(&config_path);
        let resolved = resolve_config_path(None, Some(root.path())).unwrap();
        assert_eq!(resolved, Some(config_path));
    }

    #[test]
    fn config_path_ignores_root_for_explicit_path() {
        let root = tempdir().unwrap();
        let config_root = tempdir().unwrap();
        let config_path = config_root.path().join("configs/verify.yaml");
        write_temp_config(&config_path);
        let resolved = resolve_config_path(Some(config_path.clone()), Some(root.path())).unwrap();
        assert_eq!(resolved, Some(config_path));
    }

    #[test]
    fn config_path_missing_returns_error() {
        let root = tempdir().unwrap();
        let config_path = root.path().join("missing.yaml");
        let err = resolve_config_path(Some(config_path.clone()), None).unwrap_err();
        assert!(err
            .to_string()
            .contains(&format!("Config file not found: {}", config_path.display())));
    }

    #[test]
    fn config_path_none_returns_none() {
        let resolved = resolve_config_path(None, None).unwrap();
        assert!(resolved.is_none());
    }

    #[test]
    fn config_path_directory_returns_error() {
        let root = tempdir().unwrap();
        let err = resolve_config_path(Some(root.path().to_path_buf()), None).unwrap_err();
        assert!(err.to_string().contains(&format!(
            "Config path is not a file: {}",
            root.path().display()
        )));
    }

    #[test]
    fn config_root_path_is_relative_to_config() {
        let dir = tempdir().unwrap();
        let config = VerifyConfigFile {
            general: Some(GeneralConfig {
                root_path: Some(PathBuf::from("data")),
                csv_name: None,
                device: None,
            }),
            ..VerifyConfigFile::default()
        };
        let config_path = dir.path().join("config.yaml");
        let resolved = effective_root_path(&config, &config_path, None);
        assert_eq!(resolved, Some(dir.path().join("data")));
    }

    #[test]
    fn csv_instances_resolve_paths_and_timeouts() {
        let dir = tempdir().unwrap();
        let csv_path = dir.path().join("instances.csv");
        fs::write(
            &csv_path,
            "network,property,timeout\nmodel.onnx,prop.vnnlib,45\nmodel2.onnx,prop2.vnnlib,\n",
        )
        .unwrap();
        let config_path = dir.path().join("config.yaml");
        fs::write(&config_path, "general:\n  csv_name: instances.csv\n").unwrap();

        let resolved = resolve_verify_config(Some(config_path), None)
            .unwrap()
            .expect("config should resolve");
        let instances = csv_instances(&resolved.config, &resolved.config_path, None)
            .unwrap()
            .expect("csv instances");

        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].model, dir.path().join("model.onnx"));
        assert_eq!(instances[0].property, dir.path().join("prop.vnnlib"));
        assert_eq!(instances[0].timeout, Some(45));
        assert_eq!(instances[1].timeout, None);
    }

    #[test]
    fn csv_instances_skips_vnncomp_header() {
        let dir = tempdir().unwrap();
        let csv_path = dir.path().join("instances.csv");
        fs::write(
            &csv_path,
            "onnx,vnnlib,timeout\nmodel.onnx,prop.vnnlib,30\n",
        )
        .unwrap();
        let config_path = dir.path().join("config.yaml");
        fs::write(&config_path, "general:\n  csv_name: instances.csv\n").unwrap();

        let resolved = resolve_verify_config(Some(config_path), None)
            .unwrap()
            .expect("config should resolve");
        let instances = csv_instances(&resolved.config, &resolved.config_path, None)
            .unwrap()
            .expect("csv instances");

        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].model, dir.path().join("model.onnx"));
    }

    #[test]
    fn csv_instances_rejects_invalid_timeout() {
        let dir = tempdir().unwrap();
        let csv_path = dir.path().join("instances.csv");
        fs::write(&csv_path, "model.onnx,prop.vnnlib,not_a_number\n").unwrap();
        let config_path = dir.path().join("config.yaml");
        fs::write(&config_path, "general:\n  csv_name: instances.csv\n").unwrap();

        let resolved = resolve_verify_config(Some(config_path), None)
            .unwrap()
            .expect("config should resolve");
        let err = csv_instances(&resolved.config, &resolved.config_path, None).unwrap_err();
        let message = format!("{err}");
        assert!(message.contains("Invalid timeout"));
    }

    #[test]
    fn select_instances_filters_by_range() {
        let instances = (0..5)
            .map(|index| VerifyInstance {
                model: PathBuf::from(format!("model_{index}")),
                property: PathBuf::from(format!("prop_{index}")),
                timeout: None,
                index,
            })
            .collect::<Vec<_>>();
        let data = DataConfig {
            start: Some(1),
            end: Some(3),
            select_instance: None,
        };
        let selected = select_instances(instances, Some(&data));
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].index, 1);
        assert_eq!(selected[1].index, 2);
    }

    #[test]
    fn select_instances_prefers_single_index() {
        let instances = (0..4)
            .map(|index| VerifyInstance {
                model: PathBuf::from(format!("model_{index}")),
                property: PathBuf::from(format!("prop_{index}")),
                timeout: None,
                index,
            })
            .collect::<Vec<_>>();
        let data = DataConfig {
            start: Some(0),
            end: Some(4),
            select_instance: Some(2),
        };
        let selected = select_instances(instances, Some(&data));
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].index, 2);
    }

    #[test]
    fn bab_timeout_flows_to_verify_settings() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.yaml");
        fs::write(
            &config_path,
            "bab:\n  timeout:\n    secs: 120\n    nanos: 0\n",
        )
        .unwrap();

        let resolved = resolve_verify_config(Some(config_path), None)
            .unwrap()
            .expect("config should resolve");
        let settings = resolve_verify_settings_from_config(
            Some(&resolved),
            None,
            VerifyConfigOverrides::default(),
            VerifyConfigOverrides::default(),
        )
        .unwrap();
        assert_eq!(settings.timeout, 120);
    }

    #[test]
    fn bab_timeout_rejects_subsecond_precision() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.yaml");
        fs::write(
            &config_path,
            "bab:\n  timeout:\n    secs: 120\n    nanos: 500\n",
        )
        .unwrap();

        let resolved = resolve_verify_config(Some(config_path), None)
            .unwrap()
            .expect("config should resolve");
        let err = resolve_verify_settings_from_config(
            Some(&resolved),
            None,
            VerifyConfigOverrides::default(),
            VerifyConfigOverrides::default(),
        )
        .unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("whole number of seconds"),
            "sub-second bab.timeout must be rejected, got: {message}"
        );
    }

    #[test]
    fn bab_section_rejects_settings_the_pipeline_cannot_honor() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.yaml");
        fs::write(&config_path, "bab:\n  max_queue_size: 500\n").unwrap();

        let resolved = resolve_verify_config(Some(config_path), None)
            .unwrap()
            .expect("config should resolve");
        let err = resolve_verify_settings_from_config(
            Some(&resolved),
            None,
            VerifyConfigOverrides::default(),
            VerifyConfigOverrides::default(),
        )
        .unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("max_queue_size"),
            "error must name the unsupported bab key, got: {message}"
        );
    }

    #[test]
    fn verify_config_file_deserializes_bab_conv_mode_3813() {
        let config: VerifyConfigFile = serde_yaml::from_str(
            r#"
bab:
  enable_cuts: true
  conv_mode: matrix
"#,
        )
        .expect("bab.conv_mode should deserialize through the CLI config surface");

        let bab = config
            .bab
            .expect("bab section should be present after config deserialization");
        assert_eq!(
            bab.conv_mode,
            ny_propagate::ConvMode::Matrix,
            "#3813: direct bab.conv_mode config must survive serde"
        );
        assert!(
            !bab.use_patches(),
            "#3813: matrix bab.conv_mode must force dense graph CROWN even with cuts enabled"
        );
    }
}

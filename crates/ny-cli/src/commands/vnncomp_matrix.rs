// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Local VNN-COMP script-protocol matrix runner.

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct Instance {
    year: u32,
    category: String,
    index: usize,
    model: PathBuf,
    property: PathBuf,
    timeout_secs: u64,
}

#[derive(Debug, Clone)]
struct ToolSpec {
    name: String,
    dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct MatrixRow {
    tool: String,
    year: u32,
    category: String,
    index: usize,
    model: String,
    property: String,
    timeout: u64,
    result: String,
    elapsed_s: f64,
    exit_code: i32,
    result_file: String,
    stdout_tail: String,
    stderr_tail: String,
}

pub(crate) fn handle_vnncomp_matrix_command(
    year: u32,
    tool_specs: Vec<String>,
    categories: Vec<String>,
    sample_per_category: usize,
    limit: usize,
    timeout_override: Option<u64>,
    skip_prepare: bool,
    output_dir: PathBuf,
    json_output: bool,
) -> Result<()> {
    let repo_root = find_repo_root(&std::env::current_dir()?)?;
    let tools = parse_tools(&repo_root, tool_specs)?;
    let category_filter = if categories.is_empty() {
        None
    } else {
        Some(categories)
    };
    let instances = select_instances(
        discover_instances(&repo_root, year, category_filter.as_deref())?,
        sample_per_category,
        limit,
    );
    if instances.is_empty() {
        bail!("no runnable VNN-COMP instances discovered");
    }

    let output_dir = if output_dir.is_absolute() {
        output_dir
    } else {
        repo_root.join(output_dir)
    };
    fs::create_dir_all(&output_dir)?;

    let mut rows = Vec::new();
    if !json_output {
        println!(
            "Running {} instances x {} tools",
            instances.len(),
            tools.len()
        );
    }
    for tool in &tools {
        for (position, instance) in instances.iter().enumerate() {
            if !json_output {
                println!(
                    "[{}] {}/{} {} idx={}",
                    tool.name,
                    position + 1,
                    instances.len(),
                    instance.category,
                    instance.index
                );
            }
            rows.push(run_one(
                &repo_root,
                tool,
                instance,
                &output_dir,
                timeout_override,
                skip_prepare,
            )?);
        }
    }

    let tag = unix_tag()?;
    let csv_path = output_dir.join(format!("vnncomp_matrix_{tag}.csv"));
    let json_path = output_dir.join(format!("vnncomp_matrix_{tag}.json"));
    write_csv(&csv_path, &rows)?;
    let counts = result_counts(&rows);
    fs::write(
        &json_path,
        serde_json::to_string_pretty(&json!({
            "summary": {
                "rows": rows.len(),
                "counts": counts,
                "csv": csv_path,
            },
            "rows": rows,
        }))?,
    )?;

    // Same summary either way: the JSON blob doubles as the human recap.
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "csv": csv_path,
            "json": json_path,
            "counts": counts,
        }))?
    );

    Ok(())
}

fn parse_tools(repo_root: &Path, specs: Vec<String>) -> Result<Vec<ToolSpec>> {
    let specs = if specs.is_empty() {
        vec!["ny=.".to_string()]
    } else {
        specs
    };
    specs
        .into_iter()
        .map(|spec| {
            let Some((name, dir)) = spec.split_once('=') else {
                bail!("invalid --tool {spec:?}; expected NAME=TOOL_DIR");
            };
            let dir = PathBuf::from(dir);
            let dir = if dir.is_absolute() {
                dir
            } else {
                repo_root.join(dir)
            };
            Ok(ToolSpec {
                name: name.to_string(),
                dir,
            })
        })
        .collect()
}

fn discover_instances(
    repo_root: &Path,
    year: u32,
    categories: Option<&[String]>,
) -> Result<Vec<Instance>> {
    let root = repo_root.join(format!("benchmarks/vnncomp{year}/benchmarks"));
    if !root.is_dir() {
        bail!("missing benchmark root: {}", root.display());
    }

    let mut csv_paths = Vec::new();
    collect_instances_csv(&root, &mut csv_paths)?;
    csv_paths.sort();

    let mut instances = Vec::new();
    for csv_path in csv_paths {
        let relative = csv_path.strip_prefix(&root).unwrap_or(&csv_path);
        let Some(category) = relative.components().next() else {
            continue;
        };
        let category = category.as_os_str().to_string_lossy().to_string();
        if categories.is_some_and(|items| !items.iter().any(|item| item == &category)) {
            continue;
        }
        let base = csv_path.parent().unwrap_or(&root);
        for row in parse_instances_csv(&csv_path)? {
            if row.len() < 2 {
                continue;
            }
            let model_name = first_network_path(&row[0]);
            if model_name.eq_ignore_ascii_case("network") || model_name.starts_with('#') {
                continue;
            }
            let property_name = row[1].trim();
            let Some(model) = resolve_file(base, &model_name) else {
                continue;
            };
            let Some(property) = resolve_file(base, property_name) else {
                continue;
            };
            let timeout_secs = row
                .get(2)
                .and_then(|value| value.trim().parse::<f64>().ok())
                .map(|value| value as u64)
                .filter(|&value| value > 0)
                .unwrap_or(300);
            instances.push(Instance {
                year,
                category: category.clone(),
                index: instances.len(),
                model,
                property,
                timeout_secs,
            });
        }
    }

    Ok(instances)
}

fn collect_instances_csv(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_instances_csv(&path, out)?;
        } else if path.file_name() == Some(OsStr::new("instances.csv")) {
            out.push(path);
        }
    }
    Ok(())
}

fn parse_instances_csv(path: &Path) -> Result<Vec<Vec<String>>> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("read CSV {}", path.display()))?;
    Ok(contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(split_csv_line)
        .collect())
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes && chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = !in_quotes;
                }
            }
            ',' if !in_quotes => {
                fields.push(field.trim().to_string());
                field.clear();
            }
            _ => field.push(ch),
        }
    }
    fields.push(field.trim().to_string());
    fields
}

fn first_network_path(field: &str) -> String {
    let trimmed = field.trim();
    if let Some(end_idx) = trimmed.find(".onnx").map(|idx| idx + ".onnx".len()) {
        let prefix = &trimmed[..end_idx];
        let start_idx = prefix
            .rfind(|ch: char| {
                ch == '\'' || ch == '"' || ch == '(' || ch == '[' || ch.is_whitespace()
            })
            .map(|idx| idx + 1)
            .unwrap_or(0);
        prefix[start_idx..].trim().to_string()
    } else {
        trimmed.to_string()
    }
}

fn resolve_file(base: &Path, name: &str) -> Option<PathBuf> {
    let path = Path::new(name);
    let mut candidates = vec![
        base.join(path),
        base.join(format!("{name}.gz")),
        base.join("onnx").join(path),
        base.join("onnx").join(format!("{name}.gz")),
        base.join("vnnlib").join(path),
        base.join("vnnlib").join(format!("{name}.gz")),
    ];
    if name.starts_with("onnx/original/") {
        if let Some(file_name) = path.file_name() {
            candidates.push(base.join("onnx").join(file_name));
            candidates.push(
                base.join("onnx")
                    .join(format!("{}.gz", file_name.to_string_lossy())),
            );
        }
    }
    candidates.into_iter().find(|candidate| candidate.exists())
}

fn select_instances(
    instances: Vec<Instance>,
    sample_per_category: usize,
    limit: usize,
) -> Vec<Instance> {
    let mut selected = if sample_per_category == 0 {
        instances
    } else {
        let mut grouped: BTreeMap<String, Vec<Instance>> = BTreeMap::new();
        for instance in instances {
            grouped
                .entry(instance.category.clone())
                .or_default()
                .push(instance);
        }
        let mut sampled = Vec::new();
        for (_category, items) in grouped {
            if sample_per_category >= items.len() {
                sampled.extend(items);
            } else {
                let step = items.len() as f64 / sample_per_category as f64;
                for sample_index in 0..sample_per_category {
                    sampled.push(items[(sample_index as f64 * step) as usize].clone());
                }
            }
        }
        sampled
    };
    if limit > 0 && limit < selected.len() {
        selected.truncate(limit);
    }
    selected
}

fn run_one(
    repo_root: &Path,
    tool: &ToolSpec,
    instance: &Instance,
    output_dir: &Path,
    timeout_override: Option<u64>,
    skip_prepare: bool,
) -> Result<MatrixRow> {
    let timeout_secs = timeout_override.unwrap_or(instance.timeout_secs);
    let result_file = output_dir.join(format!(
        "{}_{}_{}_{}.txt",
        tool.name, instance.year, instance.category, instance.index
    ));

    let prepare = tool.dir.join("prepare_instance.sh");
    if !skip_prepare && prepare.is_file() {
        let _ = run_command_with_timeout(
            Command::new(&prepare)
                .arg("v1")
                .arg(&instance.category)
                .arg(&instance.model)
                .arg(&instance.property)
                .current_dir(&tool.dir),
            Duration::from_secs(timeout_secs.min(60) + 10),
        );
    }

    let runner = tool.dir.join("run_instance.sh");
    if !runner.is_file() {
        bail!("missing run_instance.sh in {}", tool.dir.display());
    }

    // The result path is deterministic across runs, so a leftover file from a
    // previous sweep would be read back as THIS run's verdict if the tool dies
    // before writing. Remove it up front so the read below can only observe
    // output this invocation produced.
    if result_file.exists() {
        fs::remove_file(&result_file).with_context(|| {
            format!(
                "failed to remove stale result file {}",
                result_file.display()
            )
        })?;
    }

    let start = Instant::now();
    let output = run_command_with_timeout(
        Command::new(&runner)
            .arg("v1")
            .arg(&instance.category)
            .arg(&instance.model)
            .arg(&instance.property)
            .arg(&result_file)
            .arg(timeout_secs.to_string())
            .current_dir(&tool.dir),
        Duration::from_secs(timeout_secs + 20),
    )?;
    let elapsed_s = start.elapsed().as_secs_f64();

    let mut result = if output.timed_out {
        "timeout_ext".to_string()
    } else {
        "missing-result".to_string()
    };
    if result_file.is_file() {
        let contents = fs::read_to_string(&result_file).unwrap_or_default();
        result = contents
            .lines()
            .next()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .unwrap_or("empty-result")
            .to_string();
    }

    Ok(MatrixRow {
        tool: tool.name.clone(),
        year: instance.year,
        category: instance.category.clone(),
        index: instance.index,
        model: rel_string(repo_root, &instance.model),
        property: rel_string(repo_root, &instance.property),
        timeout: timeout_secs,
        result,
        elapsed_s,
        exit_code: output.exit_code,
        result_file: rel_string(repo_root, &result_file),
        stdout_tail: tail(&output.stdout, 500),
        stderr_tail: tail(&output.stderr, 500),
    })
}

#[derive(Debug)]
struct CapturedOutput {
    exit_code: i32,
    timed_out: bool,
    stdout: String,
    stderr: String,
}

fn run_command_with_timeout(command: &mut Command, timeout: Duration) -> Result<CapturedOutput> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(_status) = child.try_wait()? {
            let output = child.wait_with_output()?;
            return Ok(CapturedOutput {
                exit_code: output.status.code().unwrap_or(-1),
                timed_out: false,
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output()?;
            let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if !stderr.is_empty() {
                stderr.push('\n');
            }
            stderr.push_str(&format!("external timeout after {}s", timeout.as_secs()));
            return Ok(CapturedOutput {
                exit_code: 124,
                timed_out: true,
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr,
            });
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn write_csv(path: &Path, rows: &[MatrixRow]) -> Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "tool,year,category,index,model,property,timeout,result,elapsed_s,exit_code,result_file,stdout_tail,stderr_tail"
    )?;
    for row in rows {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{:.6},{},{},{},{}",
            csv_escape(&row.tool),
            row.year,
            csv_escape(&row.category),
            row.index,
            csv_escape(&row.model),
            csv_escape(&row.property),
            row.timeout,
            csv_escape(&row.result),
            row.elapsed_s,
            row.exit_code,
            csv_escape(&row.result_file),
            csv_escape(&row.stdout_tail),
            csv_escape(&row.stderr_tail),
        )?;
    }
    Ok(())
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn result_counts(rows: &[MatrixRow]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for row in rows {
        *counts.entry(row.result.clone()).or_insert(0) += 1;
    }
    counts
}

fn tail(value: &str, max_chars: usize) -> String {
    let chars: Vec<char> = value.trim().chars().collect();
    if chars.len() <= max_chars {
        chars.into_iter().collect()
    } else {
        chars[chars.len() - max_chars..].iter().collect()
    }
}

fn rel_string(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn unix_tag() -> Result<String> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow!("system clock is before UNIX epoch: {err}"))?;
    Ok(duration.as_secs().to_string())
}

fn find_repo_root(start: &Path) -> Result<PathBuf> {
    for ancestor in start.ancestors() {
        if ancestor.join("Cargo.toml").is_file() && ancestor.join("benchmarks").is_dir() {
            return Ok(ancestor.to_path_buf());
        }
    }
    Err(anyhow!(
        "could not find ny repo root from {}",
        start.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_split_preserves_quoted_tuple() {
        let fields = split_csv_line(
            "\"[('f', 'onnx/original/model.onnx'), ('g', 'onnx/perturbed/model.onnx')]\",./vnnlib/instance_0.vnnlib,100",
        );
        assert_eq!(fields.len(), 3);
        assert_eq!(first_network_path(&fields[0]), "onnx/original/model.onnx");
    }

    #[test]
    fn sampling_is_per_category() {
        let instances = vec![
            test_instance("a", 0),
            test_instance("a", 1),
            test_instance("b", 2),
            test_instance("b", 3),
        ];
        let selected = select_instances(instances, 1, 0);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].category, "a");
        assert_eq!(selected[1].category, "b");
    }

    fn test_instance(category: &str, index: usize) -> Instance {
        Instance {
            year: 2026,
            category: category.to_string(),
            index,
            model: PathBuf::from("model.onnx"),
            property: PathBuf::from("prop.vnnlib"),
            timeout_secs: 10,
        }
    }
}

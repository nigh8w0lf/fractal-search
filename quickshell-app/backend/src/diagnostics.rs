//! Model-independent accelerator checks, available even when the worker cannot start.
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_OUTPUT: usize = 256 * 1024;
const MAX_DIRECTORIES: usize = 256;
// ORT 1.22's CUDA 12 build uses cuDNN 9. The provider dependency check below
// also reports transitive requirements specific to the installed build.
const REQUIRED: &[(&str, &str)] = &[
    ("libcuda.so.1", "NVIDIA driver"),
    ("libcudart.so.12", "CUDA 12 runtime"),
    ("libcublas.so.12", "CUDA 12 cuBLAS"),
    ("libcublasLt.so.12", "CUDA 12 cuBLAS Lt"),
    ("libcufft.so.11", "CUDA 12 cuFFT"),
    ("libnvJitLink.so.12", "CUDA 12 nvJitLink"),
    ("libcudnn.so.9", "cuDNN 9"),
    (
        "libonnxruntime_providers_shared.so",
        "ONNX Runtime shared provider",
    ),
    (
        "libonnxruntime_providers_cuda.so",
        "ONNX Runtime CUDA provider",
    ),
];

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeDiagnostics {
    schema_version: u8,
    selected_provider: String,
    selected_device_id: Option<i32>,
    cuda_enabled_build: bool,
    platform: String,
    summary: String,
    checks_passed: bool,
    nvidia_driver: Probe,
    gpus: Vec<Gpu>,
    cuda_toolkit: Probe,
    linker_cache: Probe,
    provider_dependencies: Probe,
    libraries: Vec<Library>,
    missing_libraries: Vec<String>,
    found_but_not_visible_libraries: Vec<String>,
    discovered_library_directories: Vec<String>,
    library_search_locations: Vec<String>,
    ld_library_path: String,
    cuda_visible_devices: Option<String>,
    probe_errors: Vec<String>,
    install_steps: Vec<String>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Gpu {
    index: u32,
    driver_version: String,
    name: String,
}
#[derive(Debug, Serialize)]
struct Probe {
    state: String,
    detail: String,
}
impl Probe {
    fn new(state: &str, detail: impl Into<String>) -> Self {
        Self {
            state: state.into(),
            detail: detail.into(),
        }
    }
}
#[derive(Debug, Serialize)]
struct Library {
    name: String,
    component: String,
    state: String,
    paths: Vec<String>,
}

struct CommandOutput {
    stdout: String,
    stderr: String,
    success: bool,
    code: Option<i32>,
}

// Drain both pipes while waiting. Neither a stuck driver utility nor a full
// stderr pipe may hang diagnostics, including when launched from Settings.
fn run_probe(program: &str, args: &[&str], timeout: Duration) -> Result<CommandOutput> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("Cannot start {program}"))?;
    let mut stdout = child.stdout.take().context("Probe has no stdout pipe")?;
    let mut stderr = child.stderr.take().context("Probe has no stderr pipe")?;
    let read_result = (|| -> Result<CommandOutput> {
        for fd in [stdout.as_raw_fd(), stderr.as_raw_fd()] {
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            ensure!(
                flags >= 0
                    && unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } >= 0,
                "Cannot configure bounded output for {program}: {}",
                std::io::Error::last_os_error()
            );
        }
        let start = Instant::now();
        let mut out = Vec::new();
        let mut err = Vec::new();
        loop {
            drain(&mut stdout, &mut out)?;
            drain(&mut stderr, &mut err)?;
            if let Some(status) = child.try_wait()? {
                drain(&mut stdout, &mut out)?;
                drain(&mut stderr, &mut err)?;
                return Ok(CommandOutput {
                    stdout: String::from_utf8(out).context("Probe stdout is not UTF-8")?,
                    stderr: String::from_utf8(err).context("Probe stderr is not UTF-8")?,
                    success: status.success(),
                    code: status.code(),
                });
            }
            ensure!(
                start.elapsed() < timeout,
                "{program} timed out after {} seconds",
                timeout.as_secs_f32()
            );
            thread::sleep(Duration::from_millis(10));
        }
    })();
    if let Err(error) = read_result {
        // Even SIGKILL cannot immediately release a utility stuck in a driver
        // syscall. Bound cleanup too, so a broken driver cannot hang Settings.
        if matches!(child.try_wait(), Ok(Some(_))) {
            return Err(error);
        }
        if let Err(kill_error) = child.kill() {
            return Err(error.context(format!("Could not terminate {program}: {kill_error}")));
        }
        for _ in 0..10 {
            match child.try_wait() {
                Ok(Some(_)) => return Err(error),
                Ok(None) => thread::sleep(Duration::from_millis(10)),
                Err(wait_error) => {
                    return Err(error.context(format!("Could not reap {program}: {wait_error}")))
                }
            }
        }
        return Err(error.context(format!(
            "{program} has not exited after SIGKILL; it may be blocked in the driver"
        )));
    }
    read_result
}
fn drain(reader: &mut impl Read, output: &mut Vec<u8>) -> Result<()> {
    let mut buffer = [0; 4096];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => return Ok(()),
            Ok(count) => {
                ensure!(
                    output.len() + count <= MAX_OUTPUT,
                    "Diagnostic probe output exceeded {MAX_OUTPUT} bytes"
                );
                output.extend_from_slice(&buffer[..count]);
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
}
fn successful_probe(program: &str, args: &[&str]) -> Result<String> {
    let output = run_probe(program, args, PROBE_TIMEOUT)?;
    ensure!(
        output.success,
        "{program} exited {:?}: stdout: {}; stderr: {}",
        output.code,
        bounded(output.stdout.trim(), 2048),
        bounded(output.stderr.trim(), 2048)
    );
    Ok(output.stdout)
}
fn bounded(text: &str, maximum: usize) -> String {
    if text.len() <= maximum {
        return text.to_owned();
    }
    let mut end = maximum;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{} [truncated]", &text[..end])
}
fn parse_gpus(text: &str) -> Result<Vec<Gpu>> {
    let mut gpus = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        ensure!(gpus.len() < 64, "nvidia-smi reported more than 64 GPUs");
        let fields: Vec<_> = line.splitn(3, ',').map(str::trim).collect();
        ensure!(
            fields.len() == 3 && !fields[1].is_empty() && !fields[2].is_empty(),
            "Unexpected nvidia-smi GPU output: {}",
            bounded(line, 256)
        );
        gpus.push(Gpu {
            index: fields[0].parse().context("Invalid nvidia-smi GPU index")?,
            driver_version: bounded(fields[1], 64),
            name: bounded(fields[2], 256),
        });
    }
    ensure!(
        !gpus.is_empty(),
        "nvidia-smi succeeded but returned no GPUs"
    );
    Ok(gpus)
}
fn parse_cache(text: &str) -> Result<BTreeMap<String, Vec<PathBuf>>> {
    let mut libraries: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for line in text.lines() {
        if let Some((left, right)) = line.split_once("=>") {
            let name = left
                .split_whitespace()
                .next()
                .context("Missing ldconfig library name")?;
            let path = PathBuf::from(right.trim());
            ensure!(
                path.is_absolute(),
                "ldconfig returned a non-absolute library path"
            );
            libraries.entry(name.into()).or_default().push(path);
        }
    }
    ensure!(
        !libraries.is_empty(),
        "ldconfig succeeded but returned no library entries"
    );
    Ok(libraries)
}
fn add_directory(directories: &mut BTreeSet<PathBuf>, path: PathBuf, errors: &mut Vec<String>) {
    if directories.contains(&path) {
        return;
    }
    if directories.len() >= MAX_DIRECTORIES {
        if !errors.iter().any(|e| e.contains("directory limit")) {
            errors
                .push("Library search reached its directory limit; results are incomplete.".into());
        }
    } else if path.is_absolute() {
        directories.insert(path);
    } else {
        errors.push(format!("Ignoring relative library directory {}; use an absolute path in service configuration.", path.display()));
    }
}
fn child_directories(parent: &Path, prefix: &str, errors: &mut Vec<String>) -> Vec<PathBuf> {
    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(error) => {
            errors.push(format!("Cannot inspect {}: {error}", parent.display()));
            return Vec::new();
        }
    };
    let mut paths = Vec::new();
    for (index, entry) in entries.enumerate() {
        if index >= 8192 {
            errors.push(format!(
                "Directory enumeration limit reached in {}",
                parent.display()
            ));
            break;
        }
        match entry {
            Ok(entry) if entry.file_name().to_string_lossy().starts_with(prefix) => {
                paths.push(entry.path())
            }
            Ok(_) => (),
            Err(error) => errors.push(format!(
                "Cannot read entry in {}: {error}",
                parent.display()
            )),
        }
    }
    paths.sort();
    paths
}
fn file_exists(path: &Path, errors: &mut Vec<String>) -> bool {
    match fs::metadata(path) {
        Ok(metadata) => metadata.is_file(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            errors.push(format!("Cannot inspect {}: {error}", path.display()));
            false
        }
    }
}
fn classify_library(
    name: &str,
    component: &str,
    visible: &[PathBuf],
    discovered: &[PathBuf],
) -> Library {
    let paths: BTreeSet<_> = visible
        .iter()
        .chain(discovered)
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    Library {
        name: name.into(),
        component: component.into(),
        state: if !visible.is_empty() {
            "visible"
        } else if !discovered.is_empty() {
            "notVisible"
        } else {
            "notFound"
        }
        .into(),
        paths: paths.into_iter().take(8).collect(),
    }
}
fn missing_dependencies(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            let (name, target) = line.split_once("=>")?;
            (target.trim() == "not found").then(|| name.trim().to_owned())
        })
        .collect()
}

pub fn collect(provider: &str, device_id: Option<i32>) -> Result<RuntimeDiagnostics> {
    ensure!(
        matches!(provider, "cpu" | "cuda"),
        "Provider must be cpu or cuda"
    );
    ensure!(
        provider == "cuda" || device_id.is_none(),
        "--device-id is only valid with the CUDA provider"
    );
    ensure!(
        device_id.is_none_or(|value| value >= 0),
        "CUDA device ID must be non-negative"
    );
    let mut errors = Vec::new();
    let (nvidia_driver, gpus) = match successful_probe(
        "nvidia-smi",
        &[
            "--query-gpu=index,driver_version,name",
            "--format=csv,noheader",
        ],
    )
    .and_then(|text| parse_gpus(&text))
    {
        Ok(gpus) => (
            Probe::new(
                "available",
                "nvidia-smi can see the NVIDIA driver and GPU(s).",
            ),
            gpus,
        ),
        Err(error) => (
            Probe::new(
                "inconclusive",
                format!("{error:#}. A failed probe does not prove the driver is absent."),
            ),
            Vec::new(),
        ),
    };
    let cuda_toolkit = match successful_probe("nvcc", &["--version"]) {
        Ok(text) => Probe::new("available", bounded(text.trim(), 1024)),
        Err(error) => Probe::new(
            "optional",
            format!("{error:#}. nvcc is a development tool and is not required for inference."),
        ),
    };
    let ldconfig_program = if Path::new("/sbin/ldconfig").is_file() {
        "/sbin/ldconfig"
    } else {
        "ldconfig"
    };
    let (linker_cache, cache) =
        match successful_probe(ldconfig_program, &["-p"]).and_then(|text| parse_cache(&text)) {
            Ok(cache) => (
                Probe::new("available", "Read the dynamic linker cache."),
                cache,
            ),
            Err(error) => {
                errors.push(format!("{error:#}"));
                (
                    Probe::new("inconclusive", format!("{error:#}")),
                    BTreeMap::new(),
                )
            }
        };
    let executable = std::env::current_exe().context("Cannot locate diagnostics executable")?;
    let executable_directory = executable
        .parent()
        .context("Diagnostics executable has no directory")?;
    let mut runtime_dirs = BTreeSet::new();
    // ORT loads providers beside its own shared library; this may differ from
    // the executable directory in development and distro installations.
    match successful_probe(
        "ldd",
        &[executable
            .to_str()
            .context("Executable path is not UTF-8")?],
    ) {
        Ok(text) => {
            for line in text.lines() {
                if let Some((name, target)) = line.split_once("=>") {
                    if name.trim().starts_with("libonnxruntime.so") {
                        if let Some(path) = target
                            .split_whitespace()
                            .next()
                            .map(Path::new)
                            .filter(|p| p.is_absolute())
                        {
                            if let Some(parent) = path.parent() {
                                runtime_dirs.insert(parent.to_owned());
                            }
                        }
                    }
                }
            }
        }
        Err(error) => errors.push(format!("Cannot locate linked ONNX Runtime: {error:#}")),
    }
    let ld_library_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let mut visible_dirs = BTreeSet::new();
    if !ld_library_path.is_empty() {
        for dir in std::env::split_paths(&ld_library_path) {
            add_directory(&mut visible_dirs, dir, &mut errors);
        }
    }
    for path in [
        "/usr/lib",
        "/usr/lib64",
        "/lib",
        "/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
    ] {
        add_directory(&mut visible_dirs, PathBuf::from(path), &mut errors);
    }
    let mut search_dirs = visible_dirs.clone();
    add_directory(
        &mut search_dirs,
        executable_directory.to_owned(),
        &mut errors,
    );
    for dir in &runtime_dirs {
        add_directory(&mut search_dirs, dir.clone(), &mut errors);
    }
    for path in [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/opt/cuda/lib64",
        "/opt/cuda/targets/x86_64-linux/lib",
    ] {
        add_directory(&mut search_dirs, PathBuf::from(path), &mut errors);
    }
    for dir in child_directories(Path::new("/usr/local"), "cuda-12", &mut errors) {
        for suffix in ["lib64", "targets/x86_64-linux/lib"] {
            add_directory(&mut search_dirs, dir.join(suffix), &mut errors);
        }
    }
    // Inspect known active environments, never invoke Python or recurse through
    // arbitrary user files. Files outside these locations are explicitly untested.
    let mut python_roots = vec![PathBuf::from("/usr/lib"), PathBuf::from("/usr/local/lib")];
    if let Some(home) = dirs::home_dir() {
        python_roots.push(home.join(".local/lib"));
    }
    for variable in ["CONDA_PREFIX", "VIRTUAL_ENV"] {
        if let Some(prefix) = std::env::var_os(variable) {
            let root = PathBuf::from(prefix).join("lib");
            add_directory(&mut search_dirs, root.clone(), &mut errors);
            python_roots.push(root);
        }
    }
    for root in python_roots {
        for python in child_directories(&root, "python3", &mut errors) {
            for packages in ["site-packages", "dist-packages"] {
                for component in ["cuda_runtime", "cublas", "cufft", "cudnn", "nvjitlink"] {
                    add_directory(
                        &mut search_dirs,
                        python
                            .join(packages)
                            .join("nvidia")
                            .join(component)
                            .join("lib"),
                        &mut errors,
                    );
                }
            }
        }
    }
    let mut libraries = Vec::new();
    for (name, component) in REQUIRED {
        let mut visible: Vec<PathBuf> = cache
            .get(*name)
            .into_iter()
            .flatten()
            .filter(|path| file_exists(path, &mut errors))
            .cloned()
            .collect();
        for directory in &visible_dirs {
            let path = directory.join(name);
            if file_exists(&path, &mut errors) {
                visible.push(path);
            }
        }
        if name.starts_with("libonnxruntime_providers_") {
            for directory in &runtime_dirs {
                let path = directory.join(name);
                if file_exists(&path, &mut errors) {
                    visible.push(path);
                }
            }
        }
        let discovered: Vec<_> = search_dirs
            .iter()
            .map(|dir| dir.join(name))
            .filter(|path| file_exists(path, &mut errors))
            .collect();
        libraries.push(classify_library(name, component, &visible, &discovered));
    }
    let provider_library = libraries
        .iter()
        .find(|library| library.name == "libonnxruntime_providers_cuda.so");
    let mut transitive_missing = Vec::new();
    let provider_dependencies = if let Some(path) =
        provider_library.and_then(|library| library.paths.first())
    {
        match successful_probe("ldd", &[path]) {
            Ok(output) => {
                transitive_missing = missing_dependencies(&output);
                if transitive_missing.is_empty() {
                    Probe::new("available", "The installed CUDA provider's linked dependencies resolve. Actual CUDA session creation is tested when the model loads.")
                } else {
                    Probe::new(
                        "missing",
                        format!(
                            "CUDA provider dependencies cannot be resolved: {}",
                            transitive_missing.join(", ")
                        ),
                    )
                }
            }
            Err(error) => {
                errors.push(format!("{error:#}"));
                Probe::new("inconclusive", format!("{error:#}"))
            }
        }
    } else {
        Probe::new(
            "missing",
            "No CUDA provider library was found in the checked locations.",
        )
    };
    let missing_libraries: Vec<_> = libraries
        .iter()
        .filter(|lib| lib.state == "notFound")
        .map(|lib| lib.name.clone())
        .collect();
    let hidden: Vec<_> = libraries
        .iter()
        .filter(|lib| lib.state == "notVisible")
        .map(|lib| lib.name.clone())
        .collect();
    let discovered_library_directories: BTreeSet<_> = libraries
        .iter()
        .filter(|lib| lib.state == "notVisible")
        .flat_map(|lib| &lib.paths)
        .filter_map(|path| {
            Path::new(path)
                .parent()
                .map(|parent| parent.to_string_lossy().into_owned())
        })
        .collect();
    let checks_passed = cfg!(feature = "cuda")
        && nvidia_driver.state == "available"
        && errors.is_empty()
        && missing_libraries.is_empty()
        && hidden.is_empty()
        && provider_dependencies.state == "available";
    let summary = if checks_passed { "CUDA runtime checks passed. Model startup must still verify GPU execution." }
        else if !cfg!(feature = "cuda") { "This executable was built without CUDA support. Install a CUDA-enabled Fractal Search bundle to use the accelerator." }
        else if !missing_libraries.is_empty() { "Required CUDA components were not found in the checked locations. See missing libraries and setup steps below." }
        else if !hidden.is_empty() { "CUDA libraries were found outside the service's loader path. Configure their directories and restart the search service." }
        else { "CUDA verification is incomplete. Review the driver, dependency and probe results below." }.into();
    let mut steps = Vec::new();
    if !cfg!(feature = "cuda") {
        steps.push("Install/build the CUDA-enabled Quickshell bundle (--cuda); the CPU bundle cannot enable CUDA at runtime.".into());
    }
    if nvidia_driver.state != "available" {
        steps.push("Run nvidia-smi to verify the NVIDIA driver and GPU. If it fails, install or repair the driver for your GPU and distribution; a missing nvidia-smi command alone does not establish that the driver is absent.".into());
    }
    if !missing_libraries.is_empty() {
        steps.push(format!("Not found in the checked locations: {}. Install compatible NVIDIA CUDA 12 runtime components and cuDNN 9, or point the installer at their existing directories. ONNX Runtime provider libraries come from the CUDA-enabled bundle.", missing_libraries.join(", ")));
    }
    if !hidden.is_empty() || !transitive_missing.is_empty() {
        steps.push("Put the required shared libraries (or symlinks to them) in one directory and pass that absolute directory to the installer/launcher --cuda-lib-dir option. This configures LD_LIBRARY_PATH for both the systemd service and these diagnostics; setting it only in a terminal does not update a running service.".into());
    }
    steps.push("Select --provider cuda and the desired --device-id in the Quickshell installer/launcher. Restart the service after changing runtime libraries or configuration, then Refresh these checks. CUDA failures remain visible; the service does not switch silently to CPU.".into());
    steps.push("nvcc/the full CUDA toolkit is optional for inference. Use CUDA 12 and cuDNN 9 versions compatible with the bundled ONNX Runtime; CUDA 13 and cuDNN 8 libraries do not satisfy these SONAMEs. Requirements: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html".into());
    steps.push("These checks inspect the current diagnostic process environment and bounded, listed directories. They do not load models, initialize a GPU session, or search the entire disk. A configured CUDA_VISIBLE_DEVICES remaps logical device IDs.".into());
    if errors.len() > 32 {
        let omitted = errors.len() - 31;
        errors.truncate(31);
        errors.push(format!(
            "{omitted} additional probe errors omitted from this bounded report."
        ));
    }
    let report = RuntimeDiagnostics {
        schema_version: 1,
        selected_provider: provider.to_uppercase(),
        selected_device_id: if provider == "cuda" {
            Some(device_id.unwrap_or(0))
        } else {
            device_id
        },
        cuda_enabled_build: cfg!(feature = "cuda"),
        platform: std::env::consts::OS.into(),
        summary,
        checks_passed,
        nvidia_driver,
        gpus,
        cuda_toolkit,
        linker_cache,
        provider_dependencies,
        libraries,
        missing_libraries,
        found_but_not_visible_libraries: hidden,
        discovered_library_directories: discovered_library_directories.into_iter().collect(),
        library_search_locations: search_dirs
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        ld_library_path: bounded(&ld_library_path, 8192),
        cuda_visible_devices: std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        probe_errors: errors,
        install_steps: steps,
    };
    ensure!(
        serde_json::to_vec(&report)?.len() <= 60_000,
        "Accelerator diagnostic report exceeds 60,000 bytes; shorten the configured library paths."
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn driver_output_is_strict_and_preserves_gpu_names() {
        let gpus = parse_gpus(
            "0, 580.82.09, NVIDIA GeForce RTX 4090\n1, 580.82.09, NVIDIA GPU, second device\n",
        )
        .unwrap();
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[1].index, 1);
        assert_eq!(gpus[1].name, "NVIDIA GPU, second device");
        assert!(parse_gpus("").is_err());
        assert!(parse_gpus("driver failed").is_err());
    }
    #[test]
    fn linker_cache_and_provider_dependencies_are_parsed() {
        let cache = parse_cache("3 libs found in cache\n\tlibcudart.so.12 (libc6,x86-64) => /opt/cuda/lib64/libcudart.so.12\n").unwrap();
        assert_eq!(
            cache["libcudart.so.12"],
            vec![PathBuf::from("/opt/cuda/lib64/libcudart.so.12")]
        );
        assert!(parse_cache("permission denied").is_err());
        assert_eq!(
            missing_dependencies(
                "libcublas.so.12 => not found\nlibc.so.6 => /usr/lib/libc.so.6 (0xabc)\n"
            ),
            vec!["libcublas.so.12"]
        );
    }
    #[test]
    fn missing_is_distinct_from_installed_outside_loader_path() {
        let path = PathBuf::from("/opt/cuda/lib64/libcudart.so.12");
        assert_eq!(
            classify_library("libcudart.so.12", "runtime", &[], &[]).state,
            "notFound"
        );
        assert_eq!(
            classify_library(
                "libcudart.so.12",
                "runtime",
                &[],
                std::slice::from_ref(&path)
            )
            .state,
            "notVisible"
        );
        assert_eq!(
            classify_library("libcudart.so.12", "runtime", &[path], &[]).state,
            "visible"
        );
    }
    #[test]
    fn probes_time_out_and_bound_output() {
        let timeout = run_probe("sh", &["-c", "exec sleep 1"], Duration::from_millis(30));
        assert!(format!("{:#}", timeout.err().unwrap()).contains("timed out"));
        assert!(
            run_probe("sh", &["-c", "printf 'failure' >&2; exit 3"], PROBE_TIMEOUT)
                .is_ok_and(|output| !output.success && output.stderr == "failure")
        );
        assert!(run_probe("sh", &["-c", "exec yes x"], PROBE_TIMEOUT).is_err());
    }
    #[test]
    fn diagnostics_validate_configuration_without_initializing_models() {
        assert!(collect("auto", None).is_err());
        assert!(collect("cuda", Some(-1)).is_err());
        assert!(collect("cpu", Some(0)).is_err());
    }
}

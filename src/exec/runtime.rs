//! How a job's commands run on the target: in a container or natively.

use secrecy::SecretString;

use super::{Container, JobCommand, quote};
use crate::config::{DEFAULT_IMAGE, Engine, Runtime, Target};
use crate::train::JobStage;

/// Where a container sees the run directory.
pub const CONTAINER_ROOT: &str = "/workspace/run";
/// Where a container sees the Hugging Face cache.
const CONTAINER_CACHE: &str = "/workspace/hf-cache";
/// Where a container sees the tools cache of [`JobSpec::tools_dir`].
const CONTAINER_TOOLS: &str = "/workspace/cache";
/// The job's environment variable naming the tools cache as it sees it.
pub const TOOLS_ENV: &str = "OVERBRAINER_CACHE";

/// The runtime of a target, with its defaults applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobRuntime {
    /// `docker` runtime: an image run with `engine`.
    Container {
        /// Docker or Podman.
        engine: Engine,
        /// The image, `DEFAULT_IMAGE` unless the target sets one.
        image: String,
    },
    /// `native` runtime: the trainer's command line tool from `venv`, or from `PATH`.
    Native {
        /// Virtual environment holding `bin/<tool>`.
        venv: Option<String>,
        /// A POSIX shell file of `export` lines sourced before anything else, for a
        /// target whose SSH sessions lack the environment the trainer needs (a
        /// Runpod pod writes its image's `PATH` and CUDA libraries there). `None` on
        /// local and SSH targets.
        env_file: Option<String>,
    },
}

/// Everything a runtime needs to build the job of a run.
#[derive(Debug)]
pub struct JobSpec<'a> {
    /// ID of the run, which names the container.
    pub run_id: &'a str,
    /// Run directory on the target.
    pub run_dir: &'a str,
    /// Hugging Face cache on the target, mounted into containers.
    pub cache_dir: &'a str,
    /// Commands to run in order, as program and arguments.
    pub commands: &'a [Vec<String>],
    /// Environment of the commands, without secrets. In the native runtime, an
    /// entry named `PYTHONPATH` is prepended to the value the job inherits rather
    /// than replacing it; every other entry, and every entry in the container
    /// runtime, is set as given.
    pub env: &'a [(String, String)],
    /// A file, relative to the run directory, whose presence once the first
    /// command ended skips the others (see
    /// [`Trainer::stop_marker`](crate::train::Trainer::stop_marker)).
    pub stop_marker: Option<&'a str>,
    /// The stage of each command, written into a metrics file before it starts;
    /// `None` writes nothing.
    pub stages: Option<Stages<'a>>,
    /// Secret environment, passed through without its values on any command line.
    pub secrets: Vec<(String, SecretString)>,
    /// A directory on the target where the job caches the tools it downloads
    /// (llama.cpp for an export), shared by the runs; `None` for a job that
    /// downloads none. A container mounts it; either runtime names it, as the
    /// job sees it, in [`TOOLS_ENV`].
    pub tools_dir: Option<&'a str>,
    /// Checks run in order before `commands`, in the same environment but
    /// without the secrets. They write no stage event, and the stop marker
    /// still applies after the first of `commands`. The first that fails ends
    /// the job with its [`Preflight::code`].
    pub preflight: &'a [Preflight],
}

/// A check a job runs before its commands (see [`JobSpec::preflight`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preflight {
    /// The arguments of the trainer's Python: the job's own Python in a
    /// container, `<venv>/bin/python` in a native runtime with a virtual
    /// environment, else the `python` beside the first command's program on
    /// `PATH`, or `python3`.
    pub args: Vec<String>,
    /// The job's exit code when the check fails.
    pub code: u8,
}

/// The stage events of a job: one line in `file` before each command.
#[derive(Debug, Clone, Copy)]
pub struct Stages<'a> {
    /// The metrics file, relative to the run directory (the job's working
    /// directory in every runtime).
    pub file: &'a str,
    /// The stage of each command, in order; a command without one writes none.
    pub names: &'a [JobStage],
}

/// A shell command appending the stage event of `stage` to `file`, a path
/// relative to the job's working directory or absolute:
/// `{"event":"stage","name":"merge","time":1790000000}`. It never fails: a
/// line that cannot be written must not stop the job.
#[must_use]
pub fn stage_command(stage: JobStage, file: &str) -> String {
    format!(
        "{{ printf '{{\"event\":\"stage\",\"name\":\"%s\",\"time\":%s}}\\n' {} \"$(date +%s)\" >> {} || :; }}",
        stage.name(),
        shell_path(file)
    )
}

impl JobRuntime {
    /// The runtime of a local or SSH target; `None` for a Runpod target.
    #[must_use]
    pub fn from_target(target: &Target) -> Option<Self> {
        match target {
            Target::Local {
                runtime,
                engine,
                image,
                venv,
                ..
            }
            | Target::Ssh {
                runtime,
                engine,
                image,
                venv,
                ..
            } => Some(match runtime {
                Runtime::Docker => Self::Container {
                    engine: engine.unwrap_or(Engine::Docker),
                    image: image.clone().unwrap_or_else(|| DEFAULT_IMAGE.to_string()),
                },
                Runtime::Native => Self::Native {
                    venv: venv.clone(),
                    env_file: None,
                },
            }),
            Target::Runpod { .. } => None,
        }
    }

    /// The run directory as the job sees it, when it is `run_dir` on the target.
    #[must_use]
    pub fn root(&self, run_dir: &str) -> String {
        match self {
            Self::Container { .. } => CONTAINER_ROOT.to_string(),
            Self::Native { .. } => run_dir.to_string(),
        }
    }

    /// The job running `spec.commands`, stopping at the first failure.
    #[must_use]
    pub fn job(&self, spec: JobSpec<'_>) -> JobCommand {
        let (script, container) = match self {
            Self::Container { engine, image } => {
                let name = format!("overbrainer-{}", spec.run_id);
                let script = container_script(*engine, image, &name, &spec);
                (
                    script,
                    Some(Container {
                        engine: *engine,
                        name,
                    }),
                )
            },
            Self::Native { venv, env_file } => (
                native_script(venv.as_deref(), env_file.as_deref(), &spec),
                None,
            ),
        };
        JobCommand {
            dir: spec.run_dir.to_string(),
            script,
            secrets: spec.secrets,
            container,
        }
    }
}

/// The script running `spec` in one `engine` container of `image` named
/// `name`: the run directory, the Hugging Face cache and the tools cache
/// mounted, the environment and secret names passed, then [`chain`] under `sh`.
fn container_script(engine: Engine, image: &str, name: &str, spec: &JobSpec<'_>) -> String {
    let (gpus, run_label, cache_label) = match engine {
        Engine::Docker => (
            "--gpus all --ulimit memlock=-1 --ulimit stack=67108864",
            "",
            "",
        ),
        Engine::Podman => (
            "--device nvidia.com/gpu=all --security-opt=label=disable --ulimit host",
            ":Z",
            ":z",
        ),
    };
    let mut words = vec![
        engine.command().to_string(),
        "run --rm".to_string(),
        format!("--name {}", quote(name)),
        gpus.to_string(),
        "--ipc=host".to_string(),
        format!(
            "-v {}",
            quote(&format!("{}:{CONTAINER_ROOT}{run_label}", spec.run_dir))
        ),
        format!(
            "-v {}",
            quote(&format!(
                "{}:{CONTAINER_CACHE}{cache_label}",
                spec.cache_dir
            ))
        ),
    ];
    if let Some(tools) = spec.tools_dir {
        words.push(format!(
            "-v {}",
            quote(&format!("{tools}:{CONTAINER_TOOLS}{cache_label}"))
        ));
        words.push(format!("-e {TOOLS_ENV}={CONTAINER_TOOLS}"));
    }
    words.extend([
        format!("-w {CONTAINER_ROOT}"),
        format!("-e HF_HOME={CONTAINER_CACHE}"),
    ]);
    words.extend(
        spec.env
            .iter()
            .map(|(name, value)| format!("-e {}", quote(&format!("{name}={value}")))),
    );
    words.extend(
        spec.secrets
            .iter()
            .map(|(name, _)| format!("-e {}", quote(name))),
    );
    words.push(quote(image));
    words.push(format!(
        "sh -c {}",
        quote(&chain(spec, quote, &quote("python")))
    ));
    let dirs: Vec<String> = std::iter::once(spec.cache_dir)
        .chain(spec.tools_dir)
        .map(quote)
        .collect();
    format!("mkdir -p -- {} && {}", dirs.join(" "), words.join(" "))
}

/// The script running `spec` on the target itself: the env file sourced, the
/// environment exported, the preflight's Python found when there is no `venv`,
/// then [`chain`] with each program taken from `venv` or `PATH`.
fn native_script(venv: Option<&str>, env_file: Option<&str>, spec: &JobSpec<'_>) -> String {
    let resolve = |program: &str| match venv {
        Some(venv) => shell_path(&format!("{}/bin/{program}", venv.trim_end_matches('/'))),
        None => quote(program),
    };
    let mut parts = Vec::new();
    if let Some(env_file) = env_file {
        parts.push(format!(". {}", shell_path(env_file)));
    }
    let tools = spec
        .tools_dir
        .map(|tools| (TOOLS_ENV.to_string(), tools.to_string()));
    let env: Vec<&(String, String)> = spec.env.iter().chain(tools.as_ref()).collect();
    if !env.is_empty() {
        let exports: Vec<String> = env
            .iter()
            .map(|(name, value)| export_word(name, value))
            .collect();
        parts.push(format!("export {}", exports.join(" ")));
    }
    let python = if venv.is_some() {
        resolve("python")
    } else {
        if !spec.preflight.is_empty() {
            parts.push(python_beside(spec.commands));
        }
        "\"$py\"".to_string()
    };
    parts.push(chain(spec, resolve, &python));
    parts.join(" && ")
}

/// A shell command that sets `py` to the `python` in the directory of the first
/// command's program on `PATH`, where a venv or conda install keeps both, or to
/// `python3` when there is none. It always succeeds.
fn python_beside(commands: &[Vec<String>]) -> String {
    let program = commands
        .first()
        .and_then(|command| command.first())
        .map_or_else(|| quote("python3"), |program| quote(program));
    format!(
        "{{ {{ py=$(command -v {program} 2>/dev/null) && py=\"${{py%/*}}/python\" && [ -x \"$py\" ]; }} || py=python3; }}"
    )
}

/// One word passed to `export` for `name=value`. `PYTHONPATH` is prepended to the
/// value the job inherits, so a plugin directory the job needs does not blot out
/// whatever the target's own environment already sets there; every other name is
/// set as given.
fn export_word(name: &str, value: &str) -> String {
    if name == "PYTHONPATH" {
        format!(
            "PYTHONPATH={}\"${{PYTHONPATH:+:$PYTHONPATH}}\"",
            quote(value)
        )
    } else {
        quote(&format!("{name}={value}"))
    }
}

/// The preflight of `spec`, each check run by the shell word `python` without
/// the job's secrets, then its commands joined with `&&`, each program resolved
/// by `program` and each argument quoted, each after its stage event when
/// `spec` has stages. A failed check exits with its code. With a stop marker,
/// the commands after the first run only when that file does not exist once the
/// first ended; the job then exits 0 without them.
fn chain(spec: &JobSpec<'_>, program: impl Fn(&str) -> String, python: &str) -> String {
    let unset = spec
        .secrets
        .iter()
        .fold(String::new(), |mut unset, (name, _)| {
            unset.push_str(" -u ");
            unset.push_str(&quote(name));
            unset
        });
    let env = if unset.is_empty() {
        String::new()
    } else {
        format!("env{unset} ")
    };
    let checks = spec.preflight.iter().map(|check| {
        let args: Vec<String> = check.args.iter().map(|arg| quote(arg)).collect();
        format!(
            "{{ {env}{python} {} || exit {}; }}",
            args.join(" "),
            check.code
        )
    });
    let lines: Vec<String> = spec
        .commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| {
            let (first, args) = command.split_first()?;
            let mut words = vec![program(first)];
            words.extend(args.iter().map(|arg| quote(arg)));
            let line = words.join(" ");
            let stage = spec
                .stages
                .and_then(|stages| Some((stages.file, *stages.names.get(index)?)));
            Some(match stage {
                Some((file, stage)) => format!("{} && {line}", stage_command(stage, file)),
                None => line,
            })
        })
        .collect();
    let commands = match (spec.stop_marker, lines.split_first()) {
        (Some(marker), Some((first, rest))) if !rest.is_empty() => format!(
            "{first} && {{ [ -f {} ] || {{ {}; }}; }}",
            quote(marker),
            rest.join(" && ")
        ),
        _ => lines.join(" && "),
    };
    checks
        .chain(std::iter::once(commands))
        .collect::<Vec<_>>()
        .join(" && ")
}

/// A path as a shell word, with a leading `~/` expanded to the target's home.
#[must_use]
pub fn shell_path(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => format!("\"$HOME\"/{}", quote(rest)),
        None if path == "~" => "\"$HOME\"".to_string(),
        None => quote(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A job spec of `commands` with nothing else set.
    fn spec(commands: &[Vec<String>]) -> JobSpec<'_> {
        JobSpec {
            run_id: "r1",
            run_dir: "/w/r1",
            cache_dir: "/w/.hf-cache",
            commands,
            env: &[],
            stop_marker: None,
            stages: None,
            secrets: Vec::new(),
            tools_dir: None,
            preflight: &[],
        }
    }

    fn commands() -> Vec<Vec<String>> {
        vec![
            vec!["axolotl".into(), "train".into(), "axolotl.yaml".into()],
            vec!["axolotl".into(), "merge-lora".into(), "axolotl.yaml".into()],
        ]
    }

    #[test]
    fn docker_runs_one_named_container_with_gpus_and_passthrough_secrets() {
        let commands = commands();
        let env = [("AXOLOTL_DO_NOT_TRACK".to_string(), "1".to_string())];
        let runtime = JobRuntime::Container {
            engine: Engine::Docker,
            image: "img:1".into(),
        };
        let job = runtime.job(JobSpec {
            env: &env,
            secrets: vec![("HF_TOKEN".into(), SecretString::from("hf_x"))],
            ..spec(&commands)
        });
        assert_eq!(
            job.script,
            "mkdir -p -- '/w/.hf-cache' && docker run --rm --name 'overbrainer-r1' --gpus all --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v '/w/r1:/workspace/run' -v '/w/.hf-cache:/workspace/hf-cache' -w /workspace/run -e HF_HOME=/workspace/hf-cache -e 'AXOLOTL_DO_NOT_TRACK=1' -e 'HF_TOKEN' 'img:1' sh -c ''\\''axolotl'\\'' '\\''train'\\'' '\\''axolotl.yaml'\\'' && '\\''axolotl'\\'' '\\''merge-lora'\\'' '\\''axolotl.yaml'\\'''"
        );
        assert!(!job.script.contains("hf_x"));
        assert_eq!(
            job.container,
            Some(Container {
                engine: Engine::Docker,
                name: "overbrainer-r1".into()
            })
        );
        assert_eq!(job.dir, "/w/r1");
        assert_eq!(runtime.root("/w/r1"), "/workspace/run");
    }

    #[test]
    fn podman_uses_cdi_devices_and_relabelled_mounts() {
        let commands = commands();
        let runtime = JobRuntime::Container {
            engine: Engine::Podman,
            image: "img:1".into(),
        };
        let script = runtime.job(spec(&commands)).script;
        assert!(script.contains("podman run --rm --name 'overbrainer-r1' --device nvidia.com/gpu=all --security-opt=label=disable --ulimit host --ipc=host -v '/w/r1:/workspace/run:Z' -v '/w/.hf-cache:/workspace/hf-cache:z'"), "{script}");
    }

    #[test]
    fn native_exports_env_and_uses_the_venv() {
        let commands = commands();
        let env = [("PYTHONPATH".to_string(), "/w/r1/plugin".to_string())];
        let runtime = JobRuntime::Native {
            venv: Some("~/venvs/axo/".into()),
            env_file: None,
        };
        let job = runtime.job(JobSpec {
            env: &env,
            ..spec(&commands)
        });
        assert_eq!(
            job.script,
            "export PYTHONPATH='/w/r1/plugin'\"${PYTHONPATH:+:$PYTHONPATH}\" && \"$HOME\"/'venvs/axo/bin/axolotl' 'train' 'axolotl.yaml' && \"$HOME\"/'venvs/axo/bin/axolotl' 'merge-lora' 'axolotl.yaml'"
        );
        assert_eq!(job.container, None);
        assert_eq!(runtime.root("/w/r1"), "/w/r1");
        let on_path = JobRuntime::Native {
            venv: None,
            env_file: None,
        }
        .job(spec(&commands[..1]))
        .script;
        assert_eq!(on_path, "'axolotl' 'train' 'axolotl.yaml'");
    }

    #[test]
    fn native_pythonpath_is_prepended_to_the_inherited_value()
    -> Result<(), Box<dyn std::error::Error>> {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let commands = vec![vec!["true".to_string()]];
        let env = [("PYTHONPATH".to_string(), "/new".to_string())];
        let job = JobRuntime::Native {
            venv: None,
            env_file: None,
        }
        .job(JobSpec {
            env: &env,
            ..spec(&commands)
        });
        let probe = format!("{} && printf '%s' \"$PYTHONPATH\"", job.script);

        let with_inherited = run_probe(&probe, Some("/old"))?;
        assert_eq!(with_inherited, "/new:/old");

        let with_empty_inherited = run_probe(&probe, Some(""))?;
        assert_eq!(with_empty_inherited, "/new");

        let with_unset_inherited = run_probe(&probe, None)?;
        assert_eq!(with_unset_inherited, "/new");

        Ok(())
    }

    #[test]
    fn native_sources_the_env_file_first() {
        let commands = commands();
        let env = [("PYTHONPATH".to_string(), "/w/r1/plugin".to_string())];
        let runtime = JobRuntime::Native {
            venv: Some("/workspace/axolotl-venv".into()),
            env_file: Some("/etc/overbrainer/job.env".into()),
        };
        let job = runtime.job(JobSpec {
            env: &env,
            ..spec(&commands[..1])
        });
        assert_eq!(
            job.script,
            ". '/etc/overbrainer/job.env' && export PYTHONPATH='/w/r1/plugin'\"${PYTHONPATH:+:$PYTHONPATH}\" && '/workspace/axolotl-venv/bin/axolotl' 'train' 'axolotl.yaml'"
        );
    }

    #[test]
    fn the_env_file_reaches_the_commands() -> Result<(), Box<dyn std::error::Error>> {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let env_file = dir.path().join("job.env");
        std::fs::write(
            &env_file,
            "export HF_HOME='/w/.hf-cache'\nexport ODD='it'\\''s a value'\n",
        )?;
        let commands = vec![vec!["true".to_string()]];
        let job = JobRuntime::Native {
            venv: None,
            env_file: Some(env_file.to_string_lossy().into_owned()),
        }
        .job(spec(&commands));
        let probe = format!("{} && printf '%s|%s' \"$HF_HOME\" \"$ODD\"", job.script);
        assert_eq!(run_probe(&probe, None)?, "/w/.hf-cache|it's a value");
        Ok(())
    }

    #[test]
    fn the_tools_cache_is_mounted_and_named_only_when_the_job_needs_it() {
        let commands = commands();
        let docker = JobRuntime::Container {
            engine: Engine::Podman,
            image: "img:1".into(),
        };
        let tools = JobSpec {
            tools_dir: Some("/w/.cache"),
            ..spec(&commands)
        };
        let script = docker.job(tools).script;
        assert!(
            script.starts_with("mkdir -p -- '/w/.hf-cache' '/w/.cache' && podman run"),
            "{script}"
        );
        assert!(
            script.contains(
                "-v '/w/.cache:/workspace/cache:z' -e OVERBRAINER_CACHE=/workspace/cache -w /workspace/run"
            ),
            "{script}"
        );
        assert!(
            !docker
                .job(spec(&commands))
                .script
                .contains("OVERBRAINER_CACHE")
        );
        let native = JobRuntime::Native {
            venv: None,
            env_file: None,
        };
        let script = native
            .job(JobSpec {
                tools_dir: Some("/w/.cache"),
                ..spec(&commands[..1])
            })
            .script;
        assert_eq!(
            script,
            "export 'OVERBRAINER_CACHE=/w/.cache' && 'axolotl' 'train' 'axolotl.yaml'"
        );
    }

    #[test]
    fn home_relative_paths_expand_on_the_target() {
        assert_eq!(shell_path("~/a b"), "\"$HOME\"/'a b'");
        assert_eq!(shell_path("~"), "\"$HOME\"");
        assert_eq!(shell_path("/abs"), "'/abs'");
        assert_eq!(shell_path("rel/dir"), "'rel/dir'");
    }

    /// `sh` is required by [`native_pythonpath_is_prepended_to_the_inherited_value`];
    /// it skips (not fails) without it.
    #[test]
    fn a_stop_marker_skips_the_commands_after_the_first() -> Result<(), Box<dyn std::error::Error>>
    {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let step = |script: &str| vec!["sh".to_string(), "-c".to_string(), script.to_string()];
        let steps = vec![
            step("printf a >> trail; [ -z \"$STOP\" ] || : > snapshot.json"),
            step("printf b >> trail"),
            step("printf c >> trail"),
        ];
        let runtime = JobRuntime::Native {
            venv: None,
            env_file: None,
        };
        let job = runtime.job(JobSpec {
            stop_marker: Some("snapshot.json"),
            ..spec(&steps)
        });
        for (stop, trail) in [(false, "abc"), (true, "a")] {
            let dir = tempfile::tempdir()?;
            let mut command = std::process::Command::new("sh");
            command.arg("-c").arg(&job.script).current_dir(dir.path());
            if stop {
                command.env("STOP", "1");
            } else {
                command.env_remove("STOP");
            }
            assert!(command.status()?.success(), "{}", job.script);
            assert_eq!(std::fs::read_to_string(dir.path().join("trail"))?, trail);
        }
        let docker = JobRuntime::Container {
            engine: Engine::Docker,
            image: "img:1".into(),
        };
        let script = docker
            .job(JobSpec {
                stop_marker: Some("snapshot.json"),
                ..spec(&commands())
            })
            .script;
        assert!(
            script.ends_with(
                "sh -c ''\\''axolotl'\\'' '\\''train'\\'' '\\''axolotl.yaml'\\'' && { [ -f '\\''snapshot.json'\\'' ] || { '\\''axolotl'\\'' '\\''merge-lora'\\'' '\\''axolotl.yaml'\\''; }; }'"
            ),
            "{script}"
        );
        Ok(())
    }

    const STAGES: [JobStage; 2] = [JobStage::Train, JobStage::Merge];

    fn staged(commands: &[Vec<String>]) -> JobSpec<'_> {
        JobSpec {
            stages: Some(Stages {
                file: "metrics.jsonl",
                names: &STAGES,
            }),
            ..spec(commands)
        }
    }

    #[test]
    fn a_stage_event_comes_before_each_command() {
        let commands = commands();
        let native = JobRuntime::Native {
            venv: None,
            env_file: None,
        };
        assert_eq!(
            native.job(staged(&commands)).script,
            "{ printf '{\"event\":\"stage\",\"name\":\"%s\",\"time\":%s}\\n' train \"$(date +%s)\" >> 'metrics.jsonl' || :; } && 'axolotl' 'train' 'axolotl.yaml' && { printf '{\"event\":\"stage\",\"name\":\"%s\",\"time\":%s}\\n' merge \"$(date +%s)\" >> 'metrics.jsonl' || :; } && 'axolotl' 'merge-lora' 'axolotl.yaml'"
        );
        let docker = JobRuntime::Container {
            engine: Engine::Docker,
            image: "img:1".into(),
        };
        let script = docker
            .job(JobSpec {
                stop_marker: Some("snapshot.json"),
                ..staged(&commands)
            })
            .script;
        assert!(
            script.ends_with(
                "sh -c '{ printf '\\''{\"event\":\"stage\",\"name\":\"%s\",\"time\":%s}\\n'\\'' train \"$(date +%s)\" >> '\\''metrics.jsonl'\\'' || :; } && '\\''axolotl'\\'' '\\''train'\\'' '\\''axolotl.yaml'\\'' && { [ -f '\\''snapshot.json'\\'' ] || { { printf '\\''{\"event\":\"stage\",\"name\":\"%s\",\"time\":%s}\\n'\\'' merge \"$(date +%s)\" >> '\\''metrics.jsonl'\\'' || :; } && '\\''axolotl'\\'' '\\''merge-lora'\\'' '\\''axolotl.yaml'\\''; }; }'"
            ),
            "{script}"
        );
        // A command past the stages given writes none.
        let three = [commands.clone(), vec![vec!["true".to_string()]]].concat();
        let script = native.job(staged(&three)).script;
        assert!(script.ends_with("'axolotl.yaml' && 'true'"), "{script}");
    }

    #[test]
    fn stage_events_land_in_the_metrics_file_as_parseable_lines()
    -> Result<(), Box<dyn std::error::Error>> {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let step = |script: &str| vec!["sh".to_string(), "-c".to_string(), script.to_string()];
        let steps = vec![
            step("[ -z \"$STOP\" ] || : > snapshot.json"),
            step("printf b >> trail"),
        ];
        let job = JobRuntime::Native {
            venv: None,
            env_file: None,
        }
        .job(JobSpec {
            stop_marker: Some("snapshot.json"),
            ..staged(&steps)
        });
        for (stop, expected) in [
            (false, vec![JobStage::Train, JobStage::Merge]),
            (true, vec![JobStage::Train]),
        ] {
            let dir = tempfile::tempdir()?;
            let mut command = std::process::Command::new("sh");
            command.arg("-c").arg(&job.script).current_dir(dir.path());
            if stop {
                command.env("STOP", "1");
            } else {
                command.env_remove("STOP");
            }
            assert!(command.status()?.success(), "{}", job.script);
            let written = std::fs::read_to_string(dir.path().join("metrics.jsonl"))?;
            let stages: Vec<JobStage> = written
                .lines()
                .map(|line| match crate::train::parse_line(line)? {
                    crate::train::MetricLine::Stage { name, time } => {
                        assert!(time > 1_000_000_000.0, "{line}");
                        Ok(name)
                    },
                    other => Err(format!("not a stage line: {other:?}").into()),
                })
                .collect::<Result<_, Box<dyn std::error::Error>>>()?;
            assert_eq!(stages, expected);
        }
        Ok(())
    }

    /// A stage event whose file cannot be written leaves the job running.
    #[test]
    fn a_stage_event_that_cannot_be_written_does_not_stop_the_job()
    -> Result<(), Box<dyn std::error::Error>> {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let script = format!(
            "{} && printf ran > trail",
            stage_command(JobStage::Train, "missing/dir/metrics.jsonl")
        );
        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .current_dir(dir.path())
            .stderr(std::process::Stdio::null())
            .status()?;
        assert!(status.success(), "{script}");
        assert_eq!(std::fs::read_to_string(dir.path().join("trail"))?, "ran");
        Ok(())
    }

    /// The checks of the preflight.
    fn checks() -> Vec<Preflight> {
        vec![Preflight {
            args: vec![
                "/w/r1/gpu_check.py".into(),
                "/w/r1/gpu_check.json".into(),
                "1".into(),
            ],
            code: 90,
        }]
    }

    /// The preflight runs first with the venv's Python and without the
    /// secrets, writes no stage event, and leaves the stop marker after the
    /// first trainer command.
    #[test]
    fn the_preflight_runs_first_without_secrets_or_stage_events() {
        let commands = commands();
        let checks = checks();
        let job = JobRuntime::Native {
            venv: Some("/v".into()),
            env_file: None,
        }
        .job(JobSpec {
            stop_marker: Some("snapshot.json"),
            secrets: vec![("HF_TOKEN".into(), SecretString::from("hf_x"))],
            preflight: &checks,
            ..spec(&commands)
        });
        assert_eq!(
            job.script,
            "{ env -u 'HF_TOKEN' '/v/bin/python' '/w/r1/gpu_check.py' '/w/r1/gpu_check.json' '1' || exit 90; } && '/v/bin/axolotl' 'train' 'axolotl.yaml' && { [ -f 'snapshot.json' ] || { '/v/bin/axolotl' 'merge-lora' 'axolotl.yaml'; }; }"
        );
        assert!(!job.script.contains("hf_x"));
    }

    /// A container runs the preflight with the image's own Python.
    #[test]
    fn a_container_runs_the_preflight_with_its_python() {
        let commands = commands();
        let checks = checks();
        let script = JobRuntime::Container {
            engine: Engine::Docker,
            image: "img:1".into(),
        }
        .job(JobSpec {
            preflight: &checks,
            ..spec(&commands[..1])
        })
        .script;
        let inner = "{ 'python' '/w/r1/gpu_check.py' '/w/r1/gpu_check.json' '1' || exit 90; } && 'axolotl' 'train' 'axolotl.yaml'";
        assert!(
            script.ends_with(&format!("sh -c {}", quote(inner))),
            "{script}"
        );
    }

    /// Without a venv, the preflight's Python is the one beside the trainer's
    /// program on `PATH`; a failed check exits with its code before any
    /// command runs, a passed one lets them run.
    #[test]
    fn without_a_venv_the_python_beside_the_program_runs_the_check()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::PermissionsExt;

        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let bin = dir.path().join("bin");
        std::fs::create_dir_all(&bin)?;
        let stub = |name: &str, body: &str| -> std::io::Result<()> {
            let path = bin.join(name);
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n"))?;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
        };
        stub("trainer", "echo trained > trained")?;
        stub("python", "echo \"$1\" > checked; exit 4")?;
        let commands = vec![vec!["trainer".to_string()]];
        let checks = vec![Preflight {
            args: vec!["check.py".into()],
            code: 90,
        }];
        let job = JobRuntime::Native {
            venv: None,
            env_file: None,
        }
        .job(JobSpec {
            preflight: &checks,
            ..spec(&commands)
        });
        let path = format!(
            "{}:{}",
            bin.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let run = || {
            std::process::Command::new("sh")
                .arg("-c")
                .arg(&job.script)
                .current_dir(dir.path())
                .env("PATH", &path)
                .status()
        };
        assert_eq!(run()?.code(), Some(90));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("checked"))?,
            "check.py\n"
        );
        assert!(
            !dir.path().join("trained").exists(),
            "the command never ran"
        );

        stub("python", "exit 0")?;
        assert!(run()?.success());
        assert!(
            dir.path().join("trained").exists(),
            "a passed check lets it run"
        );
        Ok(())
    }

    fn sh_available() -> bool {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(":")
            .status()
            .is_ok_and(|status| status.success())
    }

    /// Runs `probe` with `sh`, in a cleared environment holding only `PATH` and,
    /// when `pythonpath` is set, `PYTHONPATH`, and returns its stdout.
    fn run_probe(
        probe: &str,
        pythonpath: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut command = std::process::Command::new("sh");
        command
            .arg("-c")
            .arg(probe)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default());
        if let Some(value) = pythonpath {
            command.env("PYTHONPATH", value);
        }
        let output = command.output()?;
        Ok(String::from_utf8(output.stdout)?)
    }
}

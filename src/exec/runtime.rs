//! How a job's commands run on the target: in a container or natively.

use secrecy::SecretString;

use super::{Container, JobCommand, quote};
use crate::config::{DEFAULT_IMAGE, Engine, Runtime, Target};

/// Where a container sees the run directory.
pub const CONTAINER_ROOT: &str = "/workspace/run";
/// Where a container sees the Hugging Face cache.
const CONTAINER_CACHE: &str = "/workspace/hf-cache";

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
    /// Secret environment, passed through without its values on any command line.
    pub secrets: Vec<(String, SecretString)>,
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
        format!("-w {CONTAINER_ROOT}"),
        format!("-e HF_HOME={CONTAINER_CACHE}"),
    ];
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
    words.push(format!("sh -c {}", quote(&chain(spec.commands, quote))));
    format!(
        "mkdir -p -- {} && {}",
        quote(spec.cache_dir),
        words.join(" ")
    )
}

fn native_script(venv: Option<&str>, env_file: Option<&str>, spec: &JobSpec<'_>) -> String {
    let resolve = |program: &str| match venv {
        Some(venv) => shell_path(&format!("{}/bin/{program}", venv.trim_end_matches('/'))),
        None => quote(program),
    };
    let mut parts = Vec::new();
    if let Some(env_file) = env_file {
        parts.push(format!(". {}", shell_path(env_file)));
    }
    if !spec.env.is_empty() {
        let exports: Vec<String> = spec
            .env
            .iter()
            .map(|(name, value)| export_word(name, value))
            .collect();
        parts.push(format!("export {}", exports.join(" ")));
    }
    parts.push(chain(spec.commands, resolve));
    parts.join(" && ")
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

/// `commands` joined with `&&`, each program resolved by `program` and each argument
/// quoted.
fn chain(commands: &[Vec<String>], program: impl Fn(&str) -> String) -> String {
    commands
        .iter()
        .filter_map(|command| {
            let (first, args) = command.split_first()?;
            let mut words = vec![program(first)];
            words.extend(args.iter().map(|arg| quote(arg)));
            Some(words.join(" "))
        })
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

    fn spec(commands: &[Vec<String>]) -> JobSpec<'_> {
        JobSpec {
            run_id: "r1",
            run_dir: "/w/r1",
            cache_dir: "/w/.hf-cache",
            commands,
            env: &[],
            secrets: Vec::new(),
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
    fn home_relative_paths_expand_on_the_target() {
        assert_eq!(shell_path("~/a b"), "\"$HOME\"/'a b'");
        assert_eq!(shell_path("~"), "\"$HOME\"");
        assert_eq!(shell_path("/abs"), "'/abs'");
        assert_eq!(shell_path("rel/dir"), "'rel/dir'");
    }

    /// `sh` is required by [`native_pythonpath_is_prepended_to_the_inherited_value`];
    /// it skips (not fails) without it.
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

//! `ssh_config` resolution for the built-in client: the subset of OpenSSH's
//! client configuration it applies, the directives it can ignore safely, and a
//! refusal for every other directive that applies to the destination.

use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use crate::exec::ssh::SshError;

/// The port when neither the destination nor the files set one.
const DEFAULT_PORT: u16 = 22;
/// The connect timeout when `ConnectTimeout` is not set.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// The keepalive interval when `ServerAliveInterval` is not set.
const DEFAULT_ALIVE_INTERVAL: Duration = Duration::from_secs(15);
/// The keepalive count when `ServerAliveCountMax` is not set.
const DEFAULT_ALIVE_COUNT: u32 = 3;
/// The key files tried when no `IdentityFile` applies (RSA keys are not supported).
const DEFAULT_IDENTITY_FILES: [&str; 2] = ["~/.ssh/id_ed25519", "~/.ssh/id_ecdsa"];
/// The user `known_hosts` files when no `UserKnownHostsFile` applies.
const DEFAULT_USER_KNOWN_HOSTS: [&str; 2] = ["~/.ssh/known_hosts", "~/.ssh/known_hosts2"];
/// The system `known_hosts` file when no `GlobalKnownHostsFile` applies.
const DEFAULT_GLOBAL_KNOWN_HOSTS: &str = "/etc/ssh/ssh_known_hosts";
/// How deep `Include`s may nest, as in OpenSSH.
const MAX_INCLUDE_DEPTH: usize = 16;
/// Algorithm-list directives ignored in system files only (S8): russh negotiates
/// its own modern algorithms, and crypto policies (Fedora, RHEL) set them there.
/// In user files they stay refused (lowercase).
const SYSTEM_ALGORITHMS: [&str; 9] = [
    "ciphers",
    "kexalgorithms",
    "macs",
    "hostkeyalgorithms",
    "pubkeyacceptedalgorithms",
    "casignaturealgorithms",
    "gssapikexalgorithms",
    "hostbasedacceptedalgorithms",
    "requiredrsasize",
];
/// Directives read but without effect on the destination or the authentication
/// (lowercase). Every `GSSAPI*` directive is ignored too.
const IGNORED: [&str; 18] = [
    "sendenv",
    "setenv",
    "forwardagent",
    "forwardx11",
    "forwardx11trusted",
    "compression",
    "loglevel",
    "hashknownhosts",
    "addkeystoagent",
    "usekeychain",
    "controlmaster",
    "controlpath",
    "controlpersist",
    "localforward",
    "remoteforward",
    "dynamicforward",
    "visualhostkey",
    "updatehostkeys",
];

/// Which agent the built-in client asks for keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentChoice {
    /// The agent at `SSH_AUTH_SOCK`, when set.
    Env,
    /// No agent (`IdentityAgent none`).
    None,
    /// The agent listening on this socket.
    Path(PathBuf),
}

/// What the built-in client needs to reach one host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostConfig {
    /// The host part of the destination, as given.
    pub alias: String,
    /// The name or address to connect to.
    pub host_name: String,
    /// The remote user.
    pub user: String,
    /// The remote port.
    pub port: u16,
    /// The key files to try, in order.
    pub identity_files: Vec<PathBuf>,
    /// Only the `identity_files`, never the agent's keys.
    pub identities_only: bool,
    /// The agent to ask for keys.
    pub identity_agent: AgentChoice,
    /// The `known_hosts` files checked, user files first.
    pub known_hosts: Vec<PathBuf>,
    /// The name looked up in `known_hosts` in place of the host name.
    pub host_key_alias: Option<String>,
    /// The jump hosts, in order; empty or `["none"]` means a direct connection.
    pub proxy_jump: Vec<String>,
    /// How long the connection may take.
    pub connect_timeout: Duration,
    /// The interval between keepalive messages.
    pub alive_interval: Duration,
    /// How many unanswered keepalive messages close the connection.
    pub alive_count: u32,
}

/// Files to read, in order (user then system), and the local user/home for defaults.
///
/// The first file is the user's (`~/.ssh/config`): an `Include` path in it, or in a
/// file it includes, is relative to `home/.ssh`. The others are system files: an
/// `Include` path in them is relative to the directory holding that system file
/// (`/etc/ssh` for `/etc/ssh/ssh_config`). A file that does not exist is skipped.
#[derive(Debug, Clone)]
pub struct ConfigSources {
    /// The configuration files, user file first.
    pub files: Vec<PathBuf>,
    /// The local user's home directory.
    pub home: PathBuf,
    /// The local user's name.
    pub local_user: String,
}

/// Resolves `destination` (`[user@]host` or `ssh://[user@]host[:port]`) through the
/// files of `sources`. A user or port in the destination wins over the files.
///
/// # Errors
///
/// Returns [`SshError::Unsupported`] for a directive the built-in client does not
/// support in a block that applies to the destination, [`SshError::Connect`] for a
/// malformed destination, `HostName`, user or `ProxyJump` hop, and
/// [`SshError::Other`] for an unreadable, unsafe or cyclic file or a malformed line.
pub fn resolve(destination: &str, sources: &ConfigSources) -> Result<HostConfig, SshError> {
    let destination = Destination::parse(destination)?;
    let mut reader = Reader {
        alias: &destination.host,
        pattern_host: destination.host.to_lowercase(),
        include_tokens: Tokens {
            host: &destination.host,
            port: destination.port.unwrap_or(DEFAULT_PORT),
            remote_user: destination.user.as_deref().unwrap_or(&sources.local_user),
            local_user: &sources.local_user,
            home: &sources.home,
        },
        found: Found::default(),
        open_files: Vec::new(),
    };
    for (index, file) in sources.files.iter().enumerate() {
        let origin = if index == 0 {
            Origin {
                base: sources.home.join(".ssh"),
                system: false,
            }
        } else {
            Origin {
                base: file.parent().map(Path::to_path_buf).unwrap_or_default(),
                system: true,
            }
        };
        reader.read(file, &origin, 0)?;
    }
    let found = reader.found;
    found.finish(&destination, sources)
}

/// Where a root file and the files it includes come from.
struct Origin {
    /// The directory relative `Include` paths start from.
    base: PathBuf,
    /// A system file (or one a system file includes), not the user's.
    system: bool,
}

/// Escapes control characters, quotes and backslashes so a name can be shown.
fn escaped(text: &str) -> String {
    text.escape_debug().to_string()
}

/// Why `value` cannot be a host or user name (`what` names which), if it cannot:
/// empty, starting with `-`, or holding whitespace, a control character or `/`.
fn name_problem(what: &str, value: &str) -> Option<String> {
    if value.is_empty() {
        Some(format!("the {what} is empty"))
    } else if value.starts_with('-') {
        Some(format!("the {what} starts with -"))
    } else if value.chars().any(char::is_control) {
        Some(format!("the {what} holds a control character"))
    } else if value.chars().any(char::is_whitespace) {
        Some(format!("the {what} holds whitespace"))
    } else if value.contains('/') {
        Some(format!("the {what} holds /"))
    } else {
        None
    }
}

/// Refuses a host or user name [`name_problem`] objects to, as a connection error
/// for `destination` (shown escaped).
fn check_name(destination: &str, what: &str, value: &str) -> Result<(), SshError> {
    match name_problem(what, value) {
        Some(reason) => Err(SshError::Connect {
            host: escaped(destination),
            reason,
        }),
        None => Ok(()),
    }
}

/// A destination split into its parts.
struct Destination {
    /// The user, when given.
    user: Option<String>,
    /// The host or alias.
    host: String,
    /// The port, when given (`ssh://` form only).
    port: Option<u16>,
}

impl Destination {
    /// Parses `[user@]host` or `ssh://[user@]host[:port]`.
    fn parse(text: &str) -> Result<Self, SshError> {
        match text.strip_prefix("ssh://") {
            Some(rest) => Self::split(text, rest.strip_suffix('/').unwrap_or(rest), true),
            None => Self::split(text, text, false),
        }
    }

    /// Splits `user_host` (`[user@]host`, with `[:port]` when `with_port`) and
    /// checks its names; errors show `text`.
    fn split(text: &str, user_host: &str, with_port: bool) -> Result<Self, SshError> {
        let invalid = |why: &str| SshError::Connect {
            host: escaped(text),
            reason: format!("invalid SSH destination: {why}"),
        };
        let (user, host_port) = match user_host.rsplit_once('@') {
            Some((user, rest)) if !user.is_empty() => (Some(user.to_owned()), rest),
            Some(_) => return Err(invalid("empty user")),
            None => (None, user_host),
        };
        let (host, port) = if !with_port {
            (host_port, None)
        } else if let Some(bracketed) = host_port.strip_prefix('[') {
            let (host, tail) = bracketed
                .split_once(']')
                .ok_or_else(|| invalid("unclosed ["))?;
            match tail.strip_prefix(':') {
                Some(port) => (host, Some(port)),
                None if tail.is_empty() => (host, None),
                None => return Err(invalid("text after ]")),
            }
        } else {
            match host_port.split_once(':') {
                Some((host, port)) => (host, Some(port)),
                None => (host_port, None),
            }
        };
        if let Some(user) = &user {
            check_name(text, "user", user)?;
        }
        check_name(text, "host", host)?;
        let port = port
            .map(|port| parse_port(port).ok_or_else(|| invalid("bad port")))
            .transpose()?;
        Ok(Self {
            user,
            host: host.to_owned(),
            port,
        })
    }
}

/// Parses a non-zero port number.
fn parse_port(text: &str) -> Option<u16> {
    text.parse::<u16>().ok().filter(|port| *port != 0)
}

/// The values obtained so far, raw (tokens not expanded yet).
#[derive(Debug, Default)]
struct Found {
    /// `HostName`.
    host_name: Option<String>,
    /// `User`.
    user: Option<String>,
    /// `Port`.
    port: Option<u16>,
    /// Every `IdentityFile`, in order.
    identity_files: Vec<String>,
    /// `IdentitiesOnly`.
    identities_only: Option<bool>,
    /// `IdentityAgent`.
    identity_agent: Option<String>,
    /// The files of the first `UserKnownHostsFile`.
    user_known_hosts: Option<Vec<String>>,
    /// The files of the first `GlobalKnownHostsFile`.
    global_known_hosts: Option<Vec<String>>,
    /// `HostKeyAlias`.
    host_key_alias: Option<String>,
    /// The hops of the first `ProxyJump`.
    proxy_jump: Option<Vec<String>>,
    /// `ConnectTimeout`, in seconds.
    connect_timeout: Option<u64>,
    /// `ServerAliveInterval`, in seconds.
    alive_interval: Option<u64>,
    /// `ServerAliveCountMax`.
    alive_count: Option<u32>,
}

impl Found {
    /// Applies the destination's user and port, the defaults and the token
    /// expansions.
    ///
    /// # Errors
    ///
    /// Returns [`SshError::Connect`] when the host name or user is not a valid name.
    fn finish(
        self,
        destination: &Destination,
        sources: &ConfigSources,
    ) -> Result<HostConfig, SshError> {
        let alias_tokens = Tokens {
            host: &destination.host,
            port: DEFAULT_PORT,
            remote_user: &sources.local_user,
            local_user: &sources.local_user,
            home: &sources.home,
        };
        let host_name = self.host_name.map_or_else(
            || destination.host.clone(),
            |name| alias_tokens.expand(&name),
        );
        check_name(&destination.host, "host name", &host_name)?;
        let user = destination
            .user
            .clone()
            .or(self.user)
            .unwrap_or_else(|| sources.local_user.clone());
        check_name(&destination.host, "user", &user)?;
        let port = destination.port.or(self.port).unwrap_or(DEFAULT_PORT);
        let tokens = Tokens {
            host: &host_name,
            port,
            remote_user: &user,
            local_user: &sources.local_user,
            home: &sources.home,
        };
        let paths = |raw: &[String]| -> Vec<PathBuf> {
            raw.iter()
                .filter(|path| !path.eq_ignore_ascii_case("none"))
                .map(|path| PathBuf::from(tokens.expand(path)))
                .collect()
        };
        let defaults =
            |list: &[&str]| -> Vec<String> { list.iter().map(|s| (*s).to_owned()).collect() };
        let identity_files = if self.identity_files.is_empty() {
            paths(&defaults(&DEFAULT_IDENTITY_FILES))
        } else {
            paths(&self.identity_files)
        };
        let mut known_hosts = paths(
            &self
                .user_known_hosts
                .unwrap_or_else(|| defaults(&DEFAULT_USER_KNOWN_HOSTS)),
        );
        known_hosts.extend(paths(
            &self
                .global_known_hosts
                .unwrap_or_else(|| defaults(&[DEFAULT_GLOBAL_KNOWN_HOSTS])),
        ));
        let identity_agent = match self.identity_agent.as_deref() {
            None | Some("SSH_AUTH_SOCK" | "$SSH_AUTH_SOCK") => AgentChoice::Env,
            Some(agent) if agent.eq_ignore_ascii_case("none") => AgentChoice::None,
            Some(agent) => AgentChoice::Path(PathBuf::from(tokens.expand(agent))),
        };
        Ok(HostConfig {
            alias: destination.host.clone(),
            host_name: host_name.clone(),
            user: user.clone(),
            port,
            identity_files,
            identities_only: self.identities_only.unwrap_or(false),
            identity_agent,
            known_hosts,
            host_key_alias: self.host_key_alias,
            proxy_jump: self.proxy_jump.unwrap_or_default(),
            connect_timeout: self
                .connect_timeout
                .map_or(DEFAULT_CONNECT_TIMEOUT, Duration::from_secs),
            alive_interval: self
                .alive_interval
                .map_or(DEFAULT_ALIVE_INTERVAL, Duration::from_secs),
            alive_count: self.alive_count.unwrap_or(DEFAULT_ALIVE_COUNT),
        })
    }
}

/// The values the `%` tokens and `~` stand for.
struct Tokens<'a> {
    /// `%h`: the remote host name.
    host: &'a str,
    /// `%p`: the remote port.
    port: u16,
    /// `%r`: the remote user.
    remote_user: &'a str,
    /// `%u`: the local user.
    local_user: &'a str,
    /// `%d` and `~`: the local home.
    home: &'a Path,
}

impl Tokens<'_> {
    /// Expands the `%h`, `%p`, `%r`, `%u`, `%d` and `%%` tokens of the raw value,
    /// then a leading `~` (alone or followed by `/`): a `%` in the home path is not
    /// read as a token. [`check_tokens`] has refused any other token.
    fn expand(&self, raw: &str) -> String {
        let home = self.home.to_string_lossy();
        let (mut out, rest) = match raw.strip_prefix('~') {
            Some(rest) if rest.is_empty() || rest.starts_with('/') => (home.to_string(), rest),
            _ => (String::new(), raw),
        };
        let mut chars = rest.chars();
        while let Some(c) = chars.next() {
            if c != '%' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('h') => out.push_str(self.host),
                Some('p') => out.push_str(&self.port.to_string()),
                Some('r') => out.push_str(self.remote_user),
                Some('u') => out.push_str(self.local_user),
                Some('d') => out.push_str(&home),
                Some(other) => out.push(other),
                None => out.push('%'),
            }
        }
        out
    }
}

/// Fails when `raw` holds a `%` token other than `%h`, `%p`, `%r`, `%u`, `%d` and
/// `%%`, or ends with a lone `%`.
fn check_tokens(raw: &str) -> Result<(), String> {
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            match chars.next() {
                Some('h' | 'p' | 'r' | 'u' | 'd' | '%') => {},
                Some(other) => return Err(format!("unknown token %{other}")),
                None => return Err("a lone % at the end".to_owned()),
            }
        }
    }
    Ok(())
}

/// Where a line comes from, for its messages and its `Include`s.
struct Here<'a> {
    /// The file holding the line.
    file: &'a Path,
    /// Where the root file of this line comes from.
    origin: &'a Origin,
    /// How many `Include`s led to this file.
    depth: usize,
    /// The line number, from 1.
    number: usize,
}

impl Here<'_> {
    /// An error about this line.
    fn error(&self, what: &str) -> SshError {
        SshError::Other(format!(
            "{} line {}: {what}",
            self.file.display(),
            self.number
        ))
    }
}

/// Reads configuration files for one destination.
struct Reader<'a> {
    /// The host part of the destination, as given, for the messages.
    alias: &'a str,
    /// The host matched against the `Host` patterns, lowercase.
    pattern_host: String,
    /// What the tokens of an `Include` path stand for.
    include_tokens: Tokens<'a>,
    /// The values obtained so far.
    found: Found,
    /// The files being read, outermost first, to refuse an `Include` cycle.
    open_files: Vec<PathBuf>,
}

impl Reader<'_> {
    /// Reads `file` (skipped when missing); its lines before any `Host` apply.
    ///
    /// A user file (or one it includes) writable by its group or by others is
    /// refused, as OpenSSH does; a file already being read is refused too.
    fn read(&mut self, file: &Path, origin: &Origin, depth: usize) -> Result<(), SshError> {
        let text = match fs::read_to_string(file) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(SshError::Other(format!(
                    "cannot read {}: {error}",
                    file.display()
                )));
            },
        };
        if !origin.system {
            check_permissions(file)?;
        }
        let identity = fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
        if self.open_files.contains(&identity) {
            return Err(SshError::Other(format!(
                "{} is included while it is being read",
                file.display()
            )));
        }
        self.open_files.push(identity);
        let mut active = true;
        for (index, raw) in text.lines().enumerate() {
            let here = Here {
                file,
                origin,
                depth,
                number: index + 1,
            };
            let Some(line) = Line::parse(raw).map_err(|why| here.error(&why))? else {
                continue;
            };
            match line.keyword.to_ascii_lowercase().as_str() {
                "host" if line.args.is_empty() => {
                    return Err(here.error("Host needs at least one pattern"));
                },
                "host" => active = self.host_matches(&line.args),
                "match" if matches_everything(&line.args) => active = true,
                "match" => return Err(self.unsupported(&line.keyword, file)),
                _ if active => self.apply(&line, &here)?,
                _ => {},
            }
        }
        self.open_files.pop();
        Ok(())
    }

    /// Whether a `Host` line's patterns select the destination: one pattern
    /// matches and no negated pattern does.
    fn host_matches(&self, patterns: &[String]) -> bool {
        let mut matched = false;
        for pattern in patterns {
            let pattern = pattern.to_lowercase();
            match pattern.strip_prefix('!') {
                Some(negated) if wildcard_match(negated, &self.pattern_host) => return false,
                Some(_) => {},
                None => matched |= wildcard_match(&pattern, &self.pattern_host),
            }
        }
        matched
    }

    /// The refusal of `directive` (as written) in `file`.
    fn unsupported(&self, directive: &str, file: &Path) -> SshError {
        SshError::Unsupported {
            directive: directive.to_owned(),
            host: self.alias.to_owned(),
            file: file.display().to_string(),
        }
    }

    /// Applies one line of a block that applies to the destination.
    fn apply(&mut self, line: &Line, here: &Here<'_>) -> Result<(), SshError> {
        let keyword = line.keyword.to_ascii_lowercase();
        let found = &mut self.found;
        match keyword.as_str() {
            "hostname" => {
                let value = line.one(here)?;
                check_tokens(value).map_err(|why| here.error(&why))?;
                set_first(&mut found.host_name, value.to_owned());
            },
            "user" => set_first(&mut found.user, line.one(here)?.to_owned()),
            "port" => {
                let port = parse_port(line.one(here)?).ok_or_else(|| here.error("bad Port"))?;
                set_first(&mut found.port, port);
            },
            "identityfile" => {
                let value = line.one(here)?;
                check_tokens(value).map_err(|why| here.error(&why))?;
                found.identity_files.push(value.to_owned());
            },
            "identitiesonly" => set_first(&mut found.identities_only, line.yes_no(here)?),
            "identityagent" => {
                let value = line.one(here)?;
                check_tokens(value).map_err(|why| here.error(&why))?;
                set_first(&mut found.identity_agent, value.to_owned());
            },
            "userknownhostsfile" => set_first(&mut found.user_known_hosts, line.files(here)?),
            "globalknownhostsfile" => set_first(&mut found.global_known_hosts, line.files(here)?),
            "hostkeyalias" => set_first(&mut found.host_key_alias, line.one(here)?.to_owned()),
            // Unknown hosts are always refused (D12): only the syntax is checked.
            "stricthostkeychecking" => {
                line.one(here)?;
            },
            "proxyjump" => {
                let hops = proxy_jump_hops(line.one(here)?, self.alias)?;
                set_first(&mut found.proxy_jump, hops);
            },
            "connecttimeout" => set_first(&mut found.connect_timeout, line.seconds(here)?),
            "serveraliveinterval" => set_first(&mut found.alive_interval, line.seconds(here)?),
            "serveralivecountmax" => {
                let count = line
                    .one(here)?
                    .parse()
                    .map_err(|_| here.error("bad count"))?;
                set_first(&mut found.alive_count, count);
            },
            "include" => self.include(line, here)?,
            other if here.origin.system && SYSTEM_ALGORITHMS.contains(&other) => {
                tracing::debug!(
                    directive = %line.keyword,
                    file = %here.file.display(),
                    "algorithm list in a system ssh_config ignored by the built-in SSH client"
                );
            },
            other if IGNORED.contains(&other) || other.starts_with("gssapi") => {},
            _ => return Err(self.unsupported(&line.keyword, here.file)),
        }
        Ok(())
    }

    /// Reads the files an `Include` names, in place, then restores the block.
    fn include(&mut self, line: &Line, here: &Here<'_>) -> Result<(), SshError> {
        if line.args.is_empty() {
            return Err(here.error("Include needs a value"));
        }
        if here.depth >= MAX_INCLUDE_DEPTH {
            return Err(here.error("Include nested too deep"));
        }
        for arg in &line.args {
            check_tokens(arg).map_err(|why| here.error(&why))?;
            let expanded = PathBuf::from(self.include_tokens.expand(arg));
            let pattern = if expanded.is_absolute() {
                expanded
            } else {
                here.origin.base.join(expanded)
            };
            for file in expand_glob(&pattern) {
                self.read(&file, here.origin, here.depth + 1)?;
            }
        }
        Ok(())
    }
}

/// Whether `Match` arguments are `all` or `final all` (S8), which apply to every
/// host.
fn matches_everything(args: &[String]) -> bool {
    let words: Vec<String> = args.iter().map(|arg| arg.to_ascii_lowercase()).collect();
    words == ["all"] || words == ["final", "all"]
}

/// The hops of a `ProxyJump` value: `none` (any case) alone means a direct
/// connection; otherwise each comma-separated hop (`[user@]host[:port]` or an
/// `ssh://` URL) is checked like a destination.
fn proxy_jump_hops(value: &str, alias: &str) -> Result<Vec<String>, SshError> {
    if value.eq_ignore_ascii_case("none") {
        return Ok(vec!["none".to_owned()]);
    }
    value
        .split(',')
        .map(|hop| {
            if hop.is_empty() {
                return Err(SshError::Connect {
                    host: escaped(alias),
                    reason: "ProxyJump holds an empty hop".to_owned(),
                });
            }
            let body = hop.strip_prefix("ssh://").unwrap_or(hop);
            Destination::split(hop, body, true)?;
            Ok(hop.to_owned())
        })
        .collect()
}

/// Refuses `file` when its group or others may write it (Unix only), as OpenSSH
/// does for the user's configuration.
fn check_permissions(file: &Path) -> Result<(), SshError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = fs::metadata(file)
            && metadata.permissions().mode() & 0o022 != 0
        {
            return Err(SshError::Other(format!(
                "Bad owner or permissions on {}",
                file.display()
            )));
        }
    }
    #[cfg(not(unix))]
    let _ = file;
    Ok(())
}

/// Keeps the first value obtained.
fn set_first<T>(slot: &mut Option<T>, value: T) {
    if slot.is_none() {
        *slot = Some(value);
    }
}

/// Matches `text` against a pattern where `*` is any run of characters and `?`
/// one character.
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        match pattern.get(p) {
            Some('*') => {
                star = Some((p, t));
                p += 1;
            },
            Some(c) if *c == '?' || *c == text[t] => {
                p += 1;
                t += 1;
            },
            _ => match star {
                Some((star_p, star_t)) => {
                    p = star_p + 1;
                    t = star_t + 1;
                    star = Some((star_p, star_t + 1));
                },
                None => return false,
            },
        }
    }
    pattern[p..].iter().all(|c| *c == '*')
}

/// The regular files matching `pattern`, whose components may hold `*` and `?`,
/// in sorted order. Names starting with `.` match only a pattern starting with
/// `.`, as in `glob(3)`.
fn expand_glob(pattern: &Path) -> Vec<PathBuf> {
    let mut candidates = vec![PathBuf::new()];
    for component in pattern.components() {
        let name = match component {
            Component::Normal(name) => name.to_string_lossy(),
            other => {
                for candidate in &mut candidates {
                    candidate.push(other);
                }
                continue;
            },
        };
        if !name.contains(['*', '?']) {
            for candidate in &mut candidates {
                candidate.push(name.as_ref());
            }
            continue;
        }
        let mut next = Vec::new();
        for dir in &candidates {
            let listed = if dir.as_os_str().is_empty() {
                Path::new(".")
            } else {
                dir
            };
            let Ok(entries) = fs::read_dir(listed) else {
                continue;
            };
            for entry in entries.flatten() {
                let file_name = entry.file_name();
                let file_name = file_name.to_string_lossy();
                let hidden = file_name.starts_with('.') && !name.starts_with('.');
                if !hidden && wildcard_match(&name, &file_name) {
                    next.push(dir.join(file_name.as_ref()));
                }
            }
        }
        candidates = next;
    }
    candidates.retain(|path| path.is_file());
    candidates.sort();
    candidates
}

/// One meaningful line: its keyword, as written, and its arguments.
struct Line {
    /// The keyword, as written.
    keyword: String,
    /// The arguments, unquoted.
    args: Vec<String>,
}

impl Line {
    /// Parses `Keyword args` or `Keyword=args`; `None` for a blank or comment line.
    fn parse(raw: &str) -> Result<Option<Self>, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return Ok(None);
        }
        let end = trimmed
            .find(|c: char| c.is_whitespace() || c == '=')
            .unwrap_or(trimmed.len());
        let (keyword, rest) = trimmed.split_at(end);
        let rest = rest.trim_start();
        let rest = rest.strip_prefix('=').unwrap_or(rest);
        let args = split_args(rest)?;
        Ok(Some(Self {
            keyword: keyword.to_owned(),
            args,
        }))
    }

    /// The single argument.
    fn one(&self, here: &Here<'_>) -> Result<&str, SshError> {
        match self.args.as_slice() {
            [value] => Ok(value),
            [] => Err(here.error(&format!("{} needs a value", self.keyword))),
            _ => Err(here.error(&format!("{} takes one value", self.keyword))),
        }
    }

    /// The single `yes` or `no` argument.
    fn yes_no(&self, here: &Here<'_>) -> Result<bool, SshError> {
        let value = self.one(here)?;
        if value.eq_ignore_ascii_case("yes") {
            Ok(true)
        } else if value.eq_ignore_ascii_case("no") {
            Ok(false)
        } else {
            Err(here.error(&format!("{} takes yes or no", self.keyword)))
        }
    }

    /// The file arguments, at least one, their tokens checked.
    fn files(&self, here: &Here<'_>) -> Result<Vec<String>, SshError> {
        if self.args.is_empty() {
            return Err(here.error(&format!("{} needs a value", self.keyword)));
        }
        for arg in &self.args {
            check_tokens(arg).map_err(|why| here.error(&why))?;
        }
        Ok(self.args.clone())
    }

    /// The single time argument in seconds (`30`, `1m5s`, `2h`).
    fn seconds(&self, here: &Here<'_>) -> Result<u64, SshError> {
        parse_time(self.one(here)?)
            .ok_or_else(|| here.error(&format!("{} takes a time", self.keyword)))
    }
}

/// Parses an OpenSSH time: numbers, each with an optional unit among `s`, `m`,
/// `h`, `d` and `w` (seconds when absent).
fn parse_time(text: &str) -> Option<u64> {
    let mut total: u64 = 0;
    let mut number: Option<u64> = None;
    for c in text.chars() {
        if let Some(digit) = c.to_digit(10) {
            number = Some(
                number
                    .unwrap_or(0)
                    .checked_mul(10)?
                    .checked_add(u64::from(digit))?,
            );
            continue;
        }
        let unit = match c.to_ascii_lowercase() {
            's' => 1,
            'm' => 60,
            'h' => 3_600,
            'd' => 86_400,
            'w' => 604_800,
            _ => return None,
        };
        total = total.checked_add(number.take()?.checked_mul(unit)?)?;
    }
    if let Some(seconds) = number {
        total = total.checked_add(seconds)?;
    }
    (!text.is_empty()).then_some(total)
}

/// Splits arguments on whitespace, with `"` and `'` quotes, backslash escapes of
/// quotes, backslashes and spaces, and a `#` starting a word ending the line.
fn split_args(text: &str) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let escaped = chars
                    .next_if(|n| matches!(n, '"' | '\'' | '\\') || (quote.is_none() && *n == ' '));
                current.push(escaped.unwrap_or('\\'));
                in_word = true;
            },
            c if Some(c) == quote => quote = None,
            c if quote.is_some() => current.push(c),
            '"' | '\'' => {
                quote = Some(c);
                in_word = true;
            },
            c if c.is_whitespace() => {
                if in_word {
                    args.push(std::mem::take(&mut current));
                    in_word = false;
                }
            },
            '#' if !in_word => break,
            c => {
                current.push(c);
                in_word = true;
            },
        }
    }
    if quote.is_some() {
        return Err("unterminated quote".to_owned());
    }
    if in_word {
        args.push(current);
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use tempfile::TempDir;

    use super::{AgentChoice, ConfigSources, parse_time, resolve, split_args, wildcard_match};
    use crate::exec::ssh::SshError;

    /// What a test returns: `?` fails it with the error.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// A temporary home holding `.ssh/`, with the user file at `.ssh/config`.
    struct Fixture {
        /// The temporary directory, removed on drop.
        dir: TempDir,
    }

    impl Fixture {
        /// Creates an empty home with its `.ssh` directory.
        fn new() -> std::io::Result<Self> {
            let dir = TempDir::new()?;
            fs::create_dir(dir.path().join(".ssh"))?;
            Ok(Self { dir })
        }

        /// The home directory.
        fn home(&self) -> PathBuf {
            self.dir.path().to_path_buf()
        }

        /// Writes `text` at `relative` under the home and returns its path.
        fn write(&self, relative: &str, text: &str) -> std::io::Result<PathBuf> {
            let path = self.dir.path().join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, text)?;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
            Ok(path)
        }

        /// Sources reading `files`, with local user `me`.
        fn sources(&self, files: &[&Path]) -> ConfigSources {
            ConfigSources {
                files: files.iter().map(|f| f.to_path_buf()).collect(),
                home: self.home(),
                local_user: "me".to_owned(),
            }
        }

        /// Sources reading only the user file `.ssh/config` holding `text`.
        fn user_config(&self, text: &str) -> std::io::Result<ConfigSources> {
            let path = self.write(".ssh/config", text)?;
            Ok(self.sources(&[&path]))
        }
    }

    /// The unsupported-directive message for `directive` in `file`, host `gpu`.
    fn refusal(file: &Path, directive: &str) -> String {
        format!(
            "{}: {directive} for host gpu is not supported by the built-in SSH client: use ssh_client = \"openssh\", or a host entry without it",
            file.display()
        )
    }

    /// An alias resolves to its `HostName`, `User`, `Port` and `IdentityFile`.
    #[test]
    fn resolves_an_alias() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config(
            "Host gpu\n  HostName 10.0.0.5\n  User alice\n  Port 2200\n  IdentityFile ~/.ssh/gpu_key\n",
        )?;
        let host = resolve("gpu", &sources)?;
        assert_eq!(host.alias, "gpu");
        assert_eq!(host.host_name, "10.0.0.5");
        assert_eq!(host.user, "alice");
        assert_eq!(host.port, 2200);
        assert_eq!(host.identity_files, vec![fx.home().join(".ssh/gpu_key")]);
        Ok(())
    }

    /// The first value obtained wins, across blocks and files.
    #[test]
    fn first_value_wins_across_blocks_and_files() -> TestResult {
        let fx = Fixture::new()?;
        let user = fx.write(
            ".ssh/config",
            "Host gpu\n User a\nHost *\n User b\n Port 2222\n",
        )?;
        let system = fx.write("etc/ssh_config", "Host *\n Port 2300\n HostName system\n")?;
        let host = resolve("gpu", &fx.sources(&[&user, &system]))?;
        assert_eq!(host.user, "a");
        assert_eq!(host.port, 2222);
        assert_eq!(host.host_name, "system");
        Ok(())
    }

    /// A negated pattern excludes the block even when another pattern matches.
    #[test]
    fn negated_pattern_excludes_the_block() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host * !gpu\n User other\n")?;
        let gpu = resolve("gpu", &sources).map(|h| h.user);
        let web = resolve("web", &sources).map(|h| h.user);
        assert_eq!(gpu.ok().as_deref(), Some("me"));
        assert_eq!(web.ok().as_deref(), Some("other"));
        Ok(())
    }

    /// A block holding only negated patterns matches nothing.
    #[test]
    fn only_negated_patterns_match_nothing() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host !gpu\n User other\n")?;
        assert_eq!(
            resolve("web", &sources).map(|h| h.user).ok().as_deref(),
            Some("me")
        );
        Ok(())
    }

    /// `?` matches one character and patterns ignore case.
    #[test]
    fn question_mark_and_case_insensitive_patterns() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host GPU?\n User one\nHost gpu*\n User many\n")?;
        assert_eq!(
            resolve("gpu1", &sources).map(|h| h.user).ok().as_deref(),
            Some("one")
        );
        assert_eq!(
            resolve("gpu12", &sources).map(|h| h.user).ok().as_deref(),
            Some("many")
        );
        assert_eq!(
            resolve("gp", &sources).map(|h| h.user).ok().as_deref(),
            Some("me")
        );
        Ok(())
    }

    /// `Include` globs are relative to `~/.ssh`, expanded in sorted order, and the
    /// included lines count where the `Include` stands.
    #[test]
    fn include_glob_relative_to_dot_ssh_in_sorted_order() -> TestResult {
        let fx = Fixture::new()?;
        fx.write(".ssh/conf.d/20-b.conf", "Host gpu\n HostName b\n User b\n")?;
        fx.write(".ssh/conf.d/10-a.conf", "Host gpu\n HostName a\n")?;
        fx.write(".ssh/conf.d/ignored.txt", "Host gpu\n Port 1\n")?;
        let sources = fx.user_config("Include conf.d/*.conf\nHost gpu\n User after\n")?;
        let host = resolve("gpu", &sources)?;
        assert_eq!(host.host_name, "a");
        assert_eq!(host.user, "b");
        assert_eq!(host.port, 22);
        Ok(())
    }

    /// An `Include` in a system file is relative to that file's directory.
    #[test]
    fn include_in_a_system_file_is_relative_to_its_directory() -> TestResult {
        let fx = Fixture::new()?;
        let user = fx.write(".ssh/config", "")?;
        let system = fx.write("etc/ssh/ssh_config", "Include ssh_config.d/*.conf\n")?;
        fx.write(
            "etc/ssh/ssh_config.d/50.conf",
            "Host gpu\n HostName from-system\n",
        )?;
        let host = resolve("gpu", &fx.sources(&[&user, &system])).map(|h| h.host_name);
        assert_eq!(host.ok().as_deref(), Some("from-system"));
        Ok(())
    }

    /// An `Include` with `~` and an absolute path is read as given.
    #[test]
    fn include_with_tilde() -> TestResult {
        let fx = Fixture::new()?;
        fx.write("other/extra", "Host gpu\n HostName extra\n")?;
        let sources = fx.user_config("Include ~/other/extra\n")?;
        let host = resolve("gpu", &sources).map(|h| h.host_name);
        assert_eq!(host.ok().as_deref(), Some("extra"));
        Ok(())
    }

    /// The block an `Include` stands in is still active after the included file,
    /// whatever `Host` lines that file holds.
    #[test]
    fn include_restores_the_enclosing_block() -> TestResult {
        let fx = Fixture::new()?;
        fx.write(".ssh/inc", "Host other\n User other\n")?;
        let sources = fx.user_config("Host gpu\n Include inc\n User gpu-user\n")?;
        assert_eq!(
            resolve("gpu", &sources).map(|h| h.user).ok().as_deref(),
            Some("gpu-user")
        );
        Ok(())
    }

    /// An `Include` in a block that does not apply is not read.
    #[test]
    fn include_in_a_non_matching_block_is_not_read() -> TestResult {
        let fx = Fixture::new()?;
        fx.write(".ssh/inc", "ProxyCommand nc %h %p\nMatch all\n")?;
        let sources = fx.user_config("Host other\n Include inc\n")?;
        assert!(resolve("gpu", &sources).is_ok(), "skipped include was read");
        Ok(())
    }

    /// Lines of an included file before any `Host` apply to the including block.
    #[test]
    fn included_top_lines_apply() -> TestResult {
        let fx = Fixture::new()?;
        fx.write(".ssh/inc", "Port 2201\n")?;
        let sources = fx.user_config("Host gpu\n Include inc\n")?;
        assert_eq!(resolve("gpu", &sources).map(|h| h.port).ok(), Some(2201));
        Ok(())
    }

    /// Tokens and `~` expand in `IdentityFile`, `UserKnownHostsFile` and
    /// `IdentityAgent`, with the final host name, port and users.
    #[test]
    fn expands_tokens_and_tilde() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config(
            "Host gpu\n HostName real\n User alice\n Port 2200\n IdentityFile ~/.ssh/%h_%r_%p_%u\n IdentityFile %d/k%%\n UserKnownHostsFile ~/kh/%h\n IdentityAgent ~/agent.sock\n",
        )?;
        let host = resolve("gpu", &sources)?;
        let home = fx.home();
        assert_eq!(
            host.identity_files,
            vec![home.join(".ssh/real_alice_2200_me"), home.join("k%")]
        );
        assert_eq!(
            host.known_hosts,
            vec![
                home.join("kh/real"),
                PathBuf::from("/etc/ssh/ssh_known_hosts")
            ]
        );
        assert_eq!(
            host.identity_agent,
            AgentChoice::Path(home.join("agent.sock"))
        );
        Ok(())
    }

    /// `%h` in `HostName` is the alias.
    #[test]
    fn host_name_expands_the_alias() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host *\n HostName %h.example.org\n")?;
        let host = resolve("gpu", &sources).map(|h| h.host_name);
        assert_eq!(host.ok().as_deref(), Some("gpu.example.org"));
        Ok(())
    }

    /// An unknown `%` token is an error naming the file.
    #[test]
    fn unknown_token_is_an_error() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("IdentityFile ~/.ssh/%z\n")?;
        let Err(SshError::Other(message)) = resolve("gpu", &sources) else {
            return Err("unknown token accepted".into());
        };
        assert!(message.contains("%z"), "message does not name the token");
        assert!(
            message.contains(".ssh/config"),
            "message does not name the file"
        );
        Ok(())
    }

    /// An ignored directive in a system `Host *` block has no effect.
    #[test]
    fn ignored_directive_passes() -> TestResult {
        let fx = Fixture::new()?;
        let user = fx.write(".ssh/config", "")?;
        let system = fx.write(
            "etc/ssh_config",
            "Host *\n SendEnv LANG LC_*\n GSSAPIAuthentication yes\n HashKnownHosts yes\n ForwardAgent no\n",
        )?;
        assert!(
            resolve("gpu", &fx.sources(&[&user, &system])).is_ok(),
            "ignored directive refused"
        );
        Ok(())
    }

    /// `ProxyCommand` in a matching block is refused with the exact message.
    #[test]
    fn proxy_command_in_a_matching_block_is_refused() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host gpu\n ProxyCommand nc %h %p\n")?;
        let path = fx.home().join(".ssh/config");
        let error = resolve("gpu", &sources).map_err(|e| e.to_string());
        assert_eq!(error.err(), Some(refusal(&path, "ProxyCommand")));
        Ok(())
    }

    /// A refusal keeps the directive's case as written and names the included file.
    #[test]
    fn refusal_keeps_case_and_names_the_included_file() -> TestResult {
        let fx = Fixture::new()?;
        let inc = fx.write(".ssh/inc", "proxycommand nc %h %p\n")?;
        let sources = fx.user_config("Include inc\n")?;
        let error = resolve("gpu", &sources).map_err(|e| e.to_string());
        assert_eq!(error.err(), Some(refusal(&inc, "proxycommand")));
        Ok(())
    }

    /// `Match` is refused wherever it is reached.
    #[test]
    fn match_is_refused() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host other\n User x\nMatch host gpu\n User y\n")?;
        let path = fx.home().join(".ssh/config");
        let error = resolve("gpu", &sources).map_err(|e| e.to_string());
        assert_eq!(error.err(), Some(refusal(&path, "Match")));
        Ok(())
    }

    /// A refused directive in a block that does not apply is not checked.
    #[test]
    fn refused_directive_in_a_non_matching_block_is_ignored() -> TestResult {
        let fx = Fixture::new()?;
        let sources =
            fx.user_config("Host other\n ProxyCommand nc %h %p\n Ciphers aes128-ctr\n")?;
        assert!(
            resolve("gpu", &sources).is_ok(),
            "non-matching block checked"
        );
        Ok(())
    }

    /// An unknown directive in a matching block is refused.
    #[test]
    fn unknown_directive_is_refused() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host *\n NoSuchThing yes\n")?;
        let path = fx.home().join(".ssh/config");
        let error = resolve("gpu", &sources).map_err(|e| e.to_string());
        assert_eq!(error.err(), Some(refusal(&path, "NoSuchThing")));
        Ok(())
    }

    /// `ProxyJump a,b` is a chain, and `ProxyJump none` a direct connection.
    #[test]
    fn proxy_jump_chain_and_none() -> TestResult {
        let fx = Fixture::new()?;
        let sources =
            fx.user_config("Host gpu\n ProxyJump a,user@b:2200\nHost web\n ProxyJump none\n")?;
        let gpu = resolve("gpu", &sources).map(|h| h.proxy_jump);
        let web = resolve("web", &sources).map(|h| h.proxy_jump);
        let direct = resolve("db", &sources).map(|h| h.proxy_jump);
        assert_eq!(
            gpu.ok(),
            Some(vec!["a".to_owned(), "user@b:2200".to_owned()])
        );
        assert_eq!(web.ok(), Some(vec!["none".to_owned()]));
        assert_eq!(direct.ok(), Some(Vec::new()));
        Ok(())
    }

    /// Without any file, the defaults apply.
    #[test]
    fn defaults() -> TestResult {
        let fx = Fixture::new()?;
        let missing = fx.home().join(".ssh/config");
        let host = resolve("gpu", &fx.sources(&[&missing]))?;
        let home = fx.home();
        assert_eq!(host.host_name, "gpu");
        assert_eq!(host.user, "me");
        assert_eq!(host.port, 22);
        assert_eq!(
            host.identity_files,
            vec![home.join(".ssh/id_ed25519"), home.join(".ssh/id_ecdsa")]
        );
        assert!(!host.identities_only, "identities_only on by default");
        assert_eq!(host.identity_agent, AgentChoice::Env);
        assert_eq!(
            host.known_hosts,
            vec![
                home.join(".ssh/known_hosts"),
                home.join(".ssh/known_hosts2"),
                PathBuf::from("/etc/ssh/ssh_known_hosts"),
            ]
        );
        assert_eq!(host.host_key_alias, None);
        assert!(host.proxy_jump.is_empty(), "proxy jump by default");
        assert_eq!(host.connect_timeout, Duration::from_secs(30));
        assert_eq!(host.alive_interval, Duration::from_secs(15));
        assert_eq!(host.alive_count, 3);
        Ok(())
    }

    /// A user and port in the destination win over the files.
    #[test]
    fn destination_user_and_port_win() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host gpu\n User config\n Port 2200\n")?;
        let plain = resolve("bob@gpu", &sources).map(|h| (h.alias, h.user, h.port));
        let url = resolve("ssh://carol@gpu:2300", &sources).map(|h| (h.alias, h.user, h.port));
        let url_no_port = resolve("ssh://gpu", &sources).map(|h| (h.user, h.port));
        assert_eq!(plain.ok(), Some(("gpu".to_owned(), "bob".to_owned(), 2200)));
        assert_eq!(url.ok(), Some(("gpu".to_owned(), "carol".to_owned(), 2300)));
        assert_eq!(url_no_port.ok(), Some(("config".to_owned(), 2200)));
        Ok(())
    }

    /// A malformed destination is an error.
    #[test]
    fn malformed_destination_is_an_error() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("")?;
        assert!(resolve("", &sources).is_err(), "empty destination accepted");
        assert!(
            resolve("ssh://gpu:port", &sources).is_err(),
            "bad port accepted"
        );
        assert!(resolve("me@", &sources).is_err(), "empty host accepted");
        Ok(())
    }

    /// Keywords ignore case; `Key=value`, quotes and trailing comments are read.
    #[test]
    fn line_syntax() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config(
            "# comment\n\nhost \"gpu\"\n  HOSTNAME=10.0.0.9\n  user = \"alice\" # trailing\n  IdentityFile \"~/.ssh/my key\"\n",
        )?;
        let host = resolve("gpu", &sources)?;
        assert_eq!(host.host_name, "10.0.0.9");
        assert_eq!(host.user, "alice");
        assert_eq!(host.identity_files, vec![fx.home().join(".ssh/my key")]);
        Ok(())
    }

    /// An unterminated quote is an error.
    #[test]
    fn unterminated_quote_is_an_error() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host gpu\n User \"alice\n")?;
        assert!(
            matches!(resolve("gpu", &sources), Err(SshError::Other(_))),
            "quote accepted"
        );
        Ok(())
    }

    /// `IdentityFile` accumulates; `UserKnownHostsFile` takes the files of its
    /// first line; `GlobalKnownHostsFile` replaces the system default.
    #[test]
    fn identity_files_accumulate_and_known_hosts_first_line_wins() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config(
            "Host gpu\n IdentityFile /k/one\n UserKnownHostsFile /kh/a /kh/b\nHost *\n IdentityFile /k/two\n UserKnownHostsFile /kh/c\n GlobalKnownHostsFile /g/one\n",
        )?;
        let host = resolve("gpu", &sources)?;
        assert_eq!(
            host.identity_files,
            vec![PathBuf::from("/k/one"), PathBuf::from("/k/two")]
        );
        assert_eq!(
            host.known_hosts,
            vec![
                PathBuf::from("/kh/a"),
                PathBuf::from("/kh/b"),
                PathBuf::from("/g/one")
            ]
        );
        Ok(())
    }

    /// `IdentitiesOnly`, `IdentityAgent`, `HostKeyAlias` and the timings apply.
    #[test]
    fn authentication_and_timing_directives() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config(
            "Host gpu\n IdentitiesOnly yes\n IdentityAgent none\n HostKeyAlias pinned\n StrictHostKeyChecking yes\n ConnectTimeout 1m5s\n ServerAliveInterval 20\n ServerAliveCountMax 5\nHost web\n IdentityAgent SSH_AUTH_SOCK\n",
        )?;
        let gpu = resolve("gpu", &sources)?;
        assert!(gpu.identities_only, "IdentitiesOnly not applied");
        assert_eq!(gpu.identity_agent, AgentChoice::None);
        assert_eq!(gpu.host_key_alias.as_deref(), Some("pinned"));
        assert_eq!(gpu.connect_timeout, Duration::from_secs(65));
        assert_eq!(gpu.alive_interval, Duration::from_secs(20));
        assert_eq!(gpu.alive_count, 5);
        let web = resolve("web", &sources).map(|h| h.identity_agent);
        assert_eq!(web.ok(), Some(AgentChoice::Env));
        Ok(())
    }

    /// A bad value in a matching block is an error naming the file and line.
    #[test]
    fn bad_value_is_an_error() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host gpu\n Port many\n")?;
        let Err(SshError::Other(message)) = resolve("gpu", &sources) else {
            return Err("bad port accepted".into());
        };
        assert!(
            message.contains(".ssh/config line 2"),
            "message does not name the line"
        );
        Ok(())
    }

    /// A chain of includes deeper than the limit is an error.
    #[test]
    fn include_depth_is_capped() -> TestResult {
        let fx = Fixture::new()?;
        for level in 0..20 {
            fx.write(
                &format!(".ssh/inc{level}"),
                &format!("Include inc{}\n", level + 1),
            )?;
        }
        let sources = fx.user_config("Include inc0\n")?;
        let Err(SshError::Other(message)) = resolve("gpu", &sources) else {
            return Err("deep include chain accepted".into());
        };
        assert!(message.contains("nested too deep"), "depth not reported");
        Ok(())
    }

    /// An `ssh://` destination may hold a bracketed IPv6 address and a port.
    #[test]
    fn ssh_url_with_ipv6() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("")?;
        let host = resolve("ssh://bob@[::1]:2200", &sources)?;
        assert_eq!(
            (host.host_name.as_str(), host.user.as_str(), host.port),
            ("::1", "bob", 2200)
        );
        Ok(())
    }

    /// OpenSSH times: bare seconds and unit suffixes.
    #[test]
    fn times() {
        assert_eq!(parse_time("30"), Some(30));
        assert_eq!(parse_time("1m5s"), Some(65));
        assert_eq!(parse_time("1h"), Some(3_600));
        assert_eq!(parse_time(""), None);
        assert_eq!(parse_time("5x"), None);
        assert_eq!(parse_time("m"), None);
    }

    /// Wildcards: `*` any run, `?` one character.
    #[test]
    fn wildcards() {
        assert!(wildcard_match("*", ""), "star on empty");
        assert!(wildcard_match("a*c", "abbbc"), "star in the middle");
        assert!(
            wildcard_match("*.example.org", "gpu.example.org"),
            "leading star"
        );
        assert!(
            !wildcard_match("a?c", "ac"),
            "question mark needs a character"
        );
        assert!(!wildcard_match("gpu", "gpu1"), "literal is whole");
    }

    /// Escapes and quotes in arguments.
    #[test]
    fn argument_escapes() {
        assert_eq!(
            split_args(r#"a\ b "c d" 'e"f' \"g"#).ok(),
            Some(vec![
                "a b".to_owned(),
                "c d".to_owned(),
                "e\"f".to_owned(),
                "\"g".to_owned()
            ])
        );
        assert_eq!(split_args(r#""""#).ok(), Some(vec![String::new()]));
    }

    /// A Fedora-like system configuration (`Match final all` and a crypto-policy
    /// include with algorithm lists) resolves.
    #[test]
    fn fedora_like_system_config_resolves() -> TestResult {
        let fx = Fixture::new()?;
        let user = fx.write(".ssh/config", "")?;
        let policy = fx.write(
            "etc/crypto-policies/back-ends/openssh.config",
            "Ciphers aes256-gcm@openssh.com\nMACs hmac-sha2-256\nGSSAPIKexAlgorithms gss-curve25519-sha256-\nKexAlgorithms curve25519-sha256\nPubkeyAcceptedAlgorithms ssh-ed25519\nHostKeyAlgorithms ssh-ed25519\nCASignatureAlgorithms ssh-ed25519\nHostbasedAcceptedAlgorithms ssh-ed25519\nRequiredRSASize 2048\n",
        )?;
        let system = fx.write(
            "etc/ssh/ssh_config",
            "Include ssh_config.d/*.conf\nHost *\n",
        )?;
        fx.write(
            "etc/ssh/ssh_config.d/50-redhat.conf",
            &format!(
                "Match final all\n Include {}\n GSSAPIAuthentication yes\n ForwardX11Trusted yes\n Match ALL\n User fedora\n",
                policy.display()
            ),
        )?;
        let host = resolve("gpu", &fx.sources(&[&user, &system]))?;
        assert_eq!(host.user, "fedora");
        Ok(())
    }

    /// Algorithm lists stay refused in the user file.
    #[test]
    fn algorithm_list_in_the_user_file_is_refused() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Match all\n Ciphers aes256-gcm@openssh.com\n")?;
        let path = fx.home().join(".ssh/config");
        let error = resolve("gpu", &sources).map_err(|e| e.to_string());
        assert_eq!(error.err(), Some(refusal(&path, "Ciphers")));
        Ok(())
    }

    /// Any other `Match` criterion stays refused, in system files too.
    #[test]
    fn match_with_a_criterion_is_refused_in_a_system_file() -> TestResult {
        let fx = Fixture::new()?;
        let user = fx.write(".ssh/config", "")?;
        let system = fx.write("etc/ssh/ssh_config", "Match host x\n")?;
        let error = resolve("gpu", &fx.sources(&[&user, &system])).map_err(|e| e.to_string());
        assert_eq!(error.err(), Some(refusal(&system, "Match")));
        Ok(())
    }

    /// Destinations that could be read as options or hold odd characters are
    /// refused as connection errors, with control characters escaped.
    #[test]
    fn bad_destinations_are_refused() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("")?;
        for (destination, reason) in [
            ("-oProxyCommand=x", "the host starts with -"),
            ("a b", "the host holds whitespace"),
            ("a\nb", "the host holds a control character"),
            ("../x", "the host holds /"),
            ("", "the host is empty"),
            ("-l@gpu", "the user starts with -"),
        ] {
            match resolve(destination, &sources) {
                Err(SshError::Connect { host, reason: got }) => {
                    assert_eq!(got, reason, "wrong reason");
                    assert!(!host.contains('\n'), "control character shown raw");
                },
                _ => return Err("bad destination accepted".into()),
            }
        }
        Ok(())
    }

    /// A `HostName` that expands to a bad name is refused.
    #[test]
    fn bad_host_name_is_refused() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host gpu\n HostName -oProxyCommand=x\n")?;
        let Err(SshError::Connect { reason, .. }) = resolve("gpu", &sources) else {
            return Err("bad HostName accepted".into());
        };
        assert_eq!(reason, "the host name starts with -");
        Ok(())
    }

    /// Bad and empty `ProxyJump` hops are refused; `none` ignores case.
    #[test]
    fn proxy_jump_hops_are_checked() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config(
            "Host empty\n ProxyJump a,,b\nHost dash\n ProxyJump a,-oProxyCommand=x\nHost upper\n ProxyJump NONE\nHost url\n ProxyJump ssh://u@[::1]:2200,b:22\n",
        )?;
        let Err(SshError::Connect { reason, .. }) = resolve("empty", &sources) else {
            return Err("empty hop accepted".into());
        };
        assert_eq!(reason, "ProxyJump holds an empty hop");
        let Err(SshError::Connect { reason, .. }) = resolve("dash", &sources) else {
            return Err("bad hop accepted".into());
        };
        assert_eq!(reason, "the host starts with -");
        assert_eq!(
            resolve("upper", &sources)?.proxy_jump,
            vec!["none".to_owned()]
        );
        assert_eq!(
            resolve("url", &sources)?.proxy_jump,
            vec!["ssh://u@[::1]:2200".to_owned(), "b:22".to_owned()]
        );
        Ok(())
    }

    /// A `%` in the home path is not read as a token.
    #[test]
    fn percent_in_the_home_is_not_a_token() -> TestResult {
        let fx = Fixture::new()?;
        let home = fx.home().join("we%hird");
        let config = fx.write("we%hird/.ssh/config", "IdentityFile ~/key_%h\n")?;
        let sources = ConfigSources {
            files: vec![config],
            home: home.clone(),
            local_user: "me".to_owned(),
        };
        assert_eq!(
            resolve("gpu", &sources)?.identity_files,
            vec![home.join("key_gpu")]
        );
        Ok(())
    }

    /// A file including itself is refused, naming it.
    #[test]
    fn self_include_is_refused() -> TestResult {
        let fx = Fixture::new()?;
        let inc = fx.write(".ssh/inc", "Include inc\n")?;
        let sources = fx.user_config("Include inc\n")?;
        let Err(SshError::Other(message)) = resolve("gpu", &sources) else {
            return Err("self include accepted".into());
        };
        assert_eq!(
            message,
            format!("{} is included while it is being read", inc.display())
        );
        Ok(())
    }

    /// A `Host` line without patterns is an error naming the file and line.
    #[test]
    fn host_without_patterns_is_an_error() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("User a\nHost\n")?;
        let Err(SshError::Other(message)) = resolve("gpu", &sources) else {
            return Err("empty Host accepted".into());
        };
        assert!(message.contains(".ssh/config line 2"), "line not named");
        Ok(())
    }

    /// A user file writable by others is refused; a system file is not checked.
    #[test]
    fn writable_user_file_is_refused() -> TestResult {
        let fx = Fixture::new()?;
        let sources = fx.user_config("Host gpu\n User a\n")?;
        let path = fx.home().join(".ssh/config");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o666))?;
        let Err(SshError::Other(message)) = resolve("gpu", &sources) else {
            return Err("writable user file accepted".into());
        };
        assert_eq!(
            message,
            format!("Bad owner or permissions on {}", path.display())
        );
        let system = fx.write("etc/ssh_config", "Host gpu\n User s\n")?;
        fs::set_permissions(&system, fs::Permissions::from_mode(0o666))?;
        let missing = fx.home().join(".ssh/none");
        assert_eq!(resolve("gpu", &fx.sources(&[&missing, &system]))?.user, "s");
        Ok(())
    }
}

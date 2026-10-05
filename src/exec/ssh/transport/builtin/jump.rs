//! `ProxyJump` for the built-in client (D13): the hosts a connection goes
//! through, in the order they are reached, each with its own configuration.

use super::config::{self, ConfigSources, HostConfig};
use crate::exec::ssh::SshError;

/// How deep the `ProxyJump` of a first hop may lead to further first hops
/// before the chain is refused as too long (or cyclic).
pub(super) const MAX_DEPTH: usize = 8;

/// One host on the way to the destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Hop {
    /// The host as written in the destination or in `ProxyJump`, for errors.
    pub name: String,
    /// Its own resolved configuration.
    pub host: HostConfig,
}

/// The hosts to reach `destination` through, first hop first and
/// `destination` last, each resolved through `sources`.
///
/// As in OpenSSH, `ProxyJump a,b` reaches `b` through `a` whatever `b`'s own
/// `ProxyJump` says, while `a` is reached as its own configuration says, its
/// `ProxyJump` included.
///
/// # Errors
///
/// Returns the [`SshError`] of [`config::resolve`] for any host on the way,
/// and [`SshError::Connect`] when first hops lead to further first hops more
/// than [`MAX_DEPTH`] times.
pub(super) fn chain(destination: &str, sources: &ConfigSources) -> Result<Vec<Hop>, SshError> {
    let mut hops = Vec::new();
    let route = Route {
        name: destination,
        destination,
        depth: 0,
    };
    route.push_onto(&mut hops, sources)?;
    Ok(hops)
}

/// A host to reach, with how deep in first hops it was found.
struct Route<'a> {
    /// The host as written, for errors.
    name: &'a str,
    /// The host as [`config::resolve`] reads it.
    destination: &'a str,
    /// How many first hops led to it.
    depth: usize,
}

impl Route<'_> {
    /// Pushes onto `hops` the hosts this one is reached through, then itself.
    fn push_onto(&self, hops: &mut Vec<Hop>, sources: &ConfigSources) -> Result<(), SshError> {
        let host = resolve_as(self.name, self.destination, sources)?;
        if let Some((first, rest)) = jumps(&host.proxy_jump).and_then(<[String]>::split_first) {
            if self.depth >= MAX_DEPTH {
                return Err(SshError::Connect {
                    host: self.name.to_owned(),
                    reason: format!("ProxyJump nests deeper than {MAX_DEPTH} hosts"),
                });
            }
            let first_destination = hop_destination(first);
            Route {
                name: first,
                destination: &first_destination,
                depth: self.depth + 1,
            }
            .push_onto(hops, sources)?;
            for hop in rest {
                hops.push(Hop {
                    name: hop.clone(),
                    host: resolve_as(hop, &hop_destination(hop), sources)?,
                });
            }
        }
        hops.push(Hop {
            name: self.name.to_owned(),
            host,
        });
        Ok(())
    }
}

/// [`config::resolve`] of `destination`, a connection error naming the host
/// `name`, as written, when the two differ (a hop read as an `ssh://` URL).
fn resolve_as(
    name: &str,
    destination: &str,
    sources: &ConfigSources,
) -> Result<HostConfig, SshError> {
    config::resolve(destination, sources).map_err(|error| match error {
        SshError::Connect { reason, .. } if name != destination => SshError::Connect {
            host: name.escape_debug().to_string(),
            reason,
        },
        other => other,
    })
}

/// A `ProxyJump` hop (`[user@]host[:port]` or an `ssh://` URL) as a
/// destination [`config::resolve`] reads with its port.
fn hop_destination(hop: &str) -> String {
    if hop.starts_with("ssh://") {
        hop.to_owned()
    } else {
        format!("ssh://{hop}")
    }
}

/// The jump hosts of `proxy_jump`, or `None` for a direct connection (empty
/// or `none`). As in OpenSSH, `none` counts only as the whole value, never as
/// one hop of a chain.
pub(super) fn jumps(proxy_jump: &[String]) -> Option<&[String]> {
    match proxy_jump {
        [] => None,
        [only] if only.eq_ignore_ascii_case("none") => None,
        hops => Some(hops),
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    use tempfile::TempDir;

    use super::*;

    /// The result of a test that may fail on I/O or SSH handling.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Sources reading only a user file holding `text`, in a fresh home that
    /// `dir` keeps alive.
    fn sources(dir: &TempDir, text: &str) -> std::io::Result<ConfigSources> {
        let ssh = dir.path().join(".ssh");
        fs::create_dir_all(&ssh)?;
        let file = ssh.join("config");
        fs::write(&file, text)?;
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600))?;
        Ok(ConfigSources {
            files: vec![file],
            home: dir.path().to_path_buf(),
            local_user: "me".to_owned(),
        })
    }

    /// The name, host name and port of each hop.
    fn route(hops: &[Hop]) -> Vec<(String, String, u16)> {
        hops.iter()
            .map(|hop| (hop.name.clone(), hop.host.host_name.clone(), hop.host.port))
            .collect()
    }

    /// Only an empty `ProxyJump` or `none` (any case) connects directly.
    #[test]
    fn proxy_jump_none_is_direct() {
        assert_eq!(jumps(&[]), None);
        assert_eq!(jumps(&["none".to_owned()]), None);
        assert_eq!(jumps(&["NONE".to_owned()]), None);
        let one = ["bastion".to_owned()];
        assert_eq!(jumps(&one), Some(&one[..]));
        let two = ["a".to_owned(), "b".to_owned()];
        assert_eq!(jumps(&two), Some(&two[..]));
    }

    /// A host without `ProxyJump`, or with `ProxyJump none`, is a chain of
    /// itself alone.
    #[test]
    fn a_direct_host_is_its_own_chain() -> TestResult {
        let dir = TempDir::new()?;
        let sources = sources(
            &dir,
            "Host web\n HostName 10.0.0.1\n ProxyJump none\nHost *\n ProxyJump far\n",
        )?;
        assert_eq!(
            route(&chain("web", &sources)?),
            vec![("web".to_owned(), "10.0.0.1".to_owned(), 22)]
        );
        let lone = sources_without_config(&dir);
        assert_eq!(
            route(&chain("db", &lone)?),
            vec![("db".to_owned(), "db".to_owned(), 22)]
        );
        Ok(())
    }

    /// Sources reading no file at all.
    fn sources_without_config(dir: &TempDir) -> ConfigSources {
        ConfigSources {
            files: Vec::new(),
            home: dir.path().to_path_buf(),
            local_user: "me".to_owned(),
        }
    }

    /// `ProxyJump a,user@b:2200` goes through `a` then `b`, each resolved
    /// through its own block, the destination last with its own settings.
    #[test]
    fn a_chain_is_reached_in_order_with_each_hop_config() -> TestResult {
        let dir = TempDir::new()?;
        let sources = sources(
            &dir,
            "Host gpu\n HostName 10.0.0.5\n Port 2222\n ProxyJump a,ops@b:2200\n\
             Host a\n HostName a.example\n User jumper\n Port 2201\n\
             Host b\n HostName b.example\n ProxyJump ignored\n",
        )?;
        let hops = chain("gpu", &sources)?;
        assert_eq!(
            route(&hops),
            vec![
                ("a".to_owned(), "a.example".to_owned(), 2201),
                ("ops@b:2200".to_owned(), "b.example".to_owned(), 2200),
                ("gpu".to_owned(), "10.0.0.5".to_owned(), 2222),
            ]
        );
        let users: Vec<&str> = hops.iter().map(|hop| hop.host.user.as_str()).collect();
        assert_eq!(users, ["jumper", "ops", "me"]);
        Ok(())
    }

    /// The first hop's own `ProxyJump` is followed: it is reached through its
    /// own jump host, ahead of it.
    #[test]
    fn the_first_hop_follows_its_own_proxy_jump() -> TestResult {
        let dir = TempDir::new()?;
        let sources = sources(
            &dir,
            "Host gpu\n ProxyJump inner\nHost inner\n ProxyJump ssh://edge:2022\n",
        )?;
        let names: Vec<String> = chain("gpu", &sources)?
            .into_iter()
            .map(|hop| hop.name)
            .collect();
        assert_eq!(names, ["ssh://edge:2022", "inner", "gpu"]);
        Ok(())
    }

    /// A hop that cannot be resolved is named as `ProxyJump` writes it, first
    /// hop and later hops alike.
    #[test]
    fn a_bad_hop_is_named_as_written() -> TestResult {
        let dir = TempDir::new()?;
        let sources = sources(
            &dir,
            "Host gpu\n ProxyJump ops@b:2200\nHost web\n ProxyJump a,b\nHost b\n HostName -x\n",
        )?;
        for destination in ["gpu", "web"] {
            let Err(SshError::Connect { host, reason }) = chain(destination, &sources) else {
                return Err("a bad hop was accepted".into());
            };
            let written = if destination == "gpu" {
                "ops@b:2200"
            } else {
                "b"
            };
            assert_eq!(
                (host.as_str(), reason.as_str()),
                (written, "the host name starts with -")
            );
        }
        Ok(())
    }

    /// A `ProxyJump` loop is refused once it nests past [`MAX_DEPTH`], and a
    /// chain exactly that deep is accepted.
    #[test]
    fn a_proxy_jump_loop_is_refused() -> TestResult {
        let dir = TempDir::new()?;
        let looping = sources(&dir, "Host a\n ProxyJump b\nHost b\n ProxyJump a\n")?;
        let Err(SshError::Connect { reason, .. }) = chain("a", &looping) else {
            return Err("a ProxyJump loop was accepted".into());
        };
        assert_eq!(
            reason,
            format!("ProxyJump nests deeper than {MAX_DEPTH} hosts")
        );
        let other = TempDir::new()?;
        let mut text = String::new();
        for depth in 0..MAX_DEPTH {
            writeln!(text, "Host h{depth}\n ProxyJump h{}", depth + 1)?;
        }
        let deep = sources(&other, &text)?;
        assert_eq!(chain("h0", &deep)?.len(), MAX_DEPTH + 1);
        Ok(())
    }
}

# Security policy

## Supported versions

Only the latest minor release receives security fixes.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

Report vulnerabilities privately through the [private reporting form](https://github.com/nayrosk/overbrainer/security/advisories/new). Do not open a public issue, pull request or discussion for them.

Include what you can of:

- the affected version and how overbrainer was installed
- the steps or configuration that reproduce the problem
- the impact you observed or expect

Leave real API keys, tokens and hostnames out of the report.

## What to expect

- An acknowledgement within 7 days.
- A first assessment within 14 days, saying whether the report is accepted.
- For an accepted report, a fix released as a patch version, then a published GitHub security advisory that credits you unless you ask otherwise.

## Scope

In scope:

- leaks of secrets (provider API keys, the Hugging Face token, the Runpod key, Vault tokens) to stdout, stderr, logs, error messages or run files
- command or shell injection through configuration, prompts, dataset content or remote hosts
- unsafe handling of SSH connections, Runpod pods or files written on remote machines
- vulnerabilities in the published crate, the release archives or the agent skill

Out of scope:

- vulnerabilities in model providers, Runpod or other third-party services
- behavior that requires an attacker to already control your machine or your `overbrainer.toml`
- dependencies with known advisories that overbrainer does not reach; `cargo deny` and Dependabot track those

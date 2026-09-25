//! The Claude Code plugin manifests in `.claude-plugin/`.

use std::path::Path;

fn manifest(name: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".claude-plugin")
        .join(name);
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

#[test]
fn plugin_version_matches_the_crate() -> Result<(), Box<dyn std::error::Error>> {
    let plugin = manifest("plugin.json")?;
    assert_eq!(plugin["name"], "overbrainer");
    assert_eq!(
        plugin["version"],
        env!("CARGO_PKG_VERSION"),
        "bump .claude-plugin/plugin.json"
    );
    Ok(())
}

#[test]
fn marketplace_lists_the_plugin_at_the_repo_root() -> Result<(), Box<dyn std::error::Error>> {
    let marketplace = manifest("marketplace.json")?;
    assert_eq!(marketplace["name"], "overbrainer");
    assert_eq!(marketplace["plugins"][0]["name"], "overbrainer");
    assert_eq!(marketplace["plugins"][0]["source"], "./");
    assert!(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("skills/overbrainer/SKILL.md")
            .is_file()
    );
    Ok(())
}

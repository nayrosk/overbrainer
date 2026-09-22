//! A YAML writer for the Axolotl config.
//!
//! The config is built as a JSON value and written here as block-style YAML that
//! `PyYAML` (YAML 1.1, which Axolotl uses) reads back unchanged: strings are always
//! double-quoted, characters `PyYAML` rejects are escaped, floats always carry a dot
//! and a signed exponent (`PyYAML` reads `1e-7` as a string), and keys that YAML 1.1
//! would read as booleans or null are quoted.

use std::fmt::Write as _;

use serde_json::{Map, Number, Value};

/// Writes `value` as a YAML document. A mapping becomes block-style YAML; any other
/// value is written as a single scalar or sequence.
#[must_use]
pub fn to_yaml(value: &Value) -> String {
    let mut out = String::new();
    match value {
        Value::Object(map) if !map.is_empty() => write_map(&mut out, map, 0, false),
        Value::Array(items) if !items.is_empty() => write_seq(&mut out, items, 0),
        scalar => {
            out.push_str(&inline(scalar));
            out.push('\n');
        },
    }
    out
}

fn write_map(out: &mut String, map: &Map<String, Value>, indent: usize, inline_first: bool) {
    for (index, (key, value)) in map.iter().enumerate() {
        if index > 0 || !inline_first {
            out.push_str(&" ".repeat(indent));
        }
        out.push_str(&key_text(key));
        out.push(':');
        match value {
            Value::Object(inner) if !inner.is_empty() => {
                out.push('\n');
                write_map(out, inner, indent + 2, false);
            },
            Value::Array(items) if !items.is_empty() => {
                out.push('\n');
                write_seq(out, items, indent + 2);
            },
            scalar => {
                out.push(' ');
                out.push_str(&inline(scalar));
                out.push('\n');
            },
        }
    }
}

fn write_seq(out: &mut String, items: &[Value], indent: usize) {
    for item in items {
        out.push_str(&" ".repeat(indent));
        match item {
            Value::Object(map) if !map.is_empty() => {
                out.push_str("- ");
                write_map(out, map, indent + 2, true);
            },
            Value::Array(inner) if !inner.is_empty() => {
                out.push_str("-\n");
                write_seq(out, inner, indent + 2);
            },
            scalar => {
                out.push_str("- ");
                out.push_str(&inline(scalar));
                out.push('\n');
            },
        }
    }
}

/// A scalar, or an empty mapping or sequence, on one line.
fn inline(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number_text(number),
        Value::String(text) => quote(text),
        Value::Array(_) => "[]".to_string(),
        Value::Object(_) => "{}".to_string(),
    }
}

/// Integers as they are; floats with a dot and, when there is one, a signed exponent.
fn number_text(number: &Number) -> String {
    if number.is_i64() || number.is_u64() {
        return number.to_string();
    }
    let Some(float) = number.as_f64() else {
        return number.to_string();
    };
    let text = format!("{float:?}");
    match text.split_once('e') {
        Some((mantissa, exponent)) => {
            let dot = if mantissa.contains('.') { "" } else { ".0" };
            let sign = if exponent.starts_with('-') { "" } else { "+" };
            format!("{mantissa}{dot}e{sign}{exponent}")
        },
        None => text,
    }
}

/// A plain key when YAML 1.1 reads it back as the same string, otherwise quoted.
fn key_text(key: &str) -> String {
    const SPECIAL: [&str; 11] = [
        "y", "n", "yes", "no", "on", "off", "true", "false", "null", "~", "",
    ];
    let plain = key
        .chars()
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.');
    if plain && !SPECIAL.contains(&key.to_ascii_lowercase().as_str()) {
        key.to_string()
    } else {
        quote(key)
    }
}

/// A double-quoted YAML string. Characters that `PyYAML` does not accept in a
/// stream, and the characters that need it in double quotes, are escaped.
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if printable(c) => out.push(c),
            c if u32::from(c) <= 0xFFFF => {
                write!(out, "\\u{:04X}", u32::from(c)).ok();
            },
            c => {
                write!(out, "\\U{:08X}", u32::from(c)).ok();
            },
        }
    }
    out.push('"');
    out
}

/// The printable set of YAML 1.1, as `PyYAML` checks it, minus tab and line breaks.
fn printable(c: char) -> bool {
    matches!(
        c,
        ' '..='~'
            | '\u{85}'
            | '\u{A0}'..='\u{D7FF}'
            | '\u{E000}'..='\u{FFFD}'
            | '\u{10000}'..='\u{10FFFF}'
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn mappings_and_sequences_are_block_style() {
        let value = json!({
            "base_model": "Qwen/Qwen3-4B",
            "datasets": [
                {"path": "/run/data/train.jsonl", "roles_to_train": ["assistant"], "type": "chat_template"}
            ],
            "empty": {},
            "lora_r": 16,
            "none": [],
            "special_tokens": {"pad_token": "<pad>"}
        });
        assert_eq!(
            to_yaml(&value),
            "base_model: \"Qwen/Qwen3-4B\"
datasets:
  - path: \"/run/data/train.jsonl\"
    roles_to_train:
      - \"assistant\"
    type: \"chat_template\"
empty: {}
lora_r: 16
none: []
special_tokens:
  pad_token: \"<pad>\"
"
        );
    }

    #[test]
    fn floats_always_carry_a_dot_and_a_signed_exponent() {
        let floats = [
            (2e-4, "0.0002"),
            (0.05, "0.05"),
            (1e-7, "1.0e-7"),
            (1.5e-7, "1.5e-7"),
            (3e20, "3.0e+20"),
            (1.0, "1.0"),
        ];
        for (float, text) in floats {
            assert_eq!(to_yaml(&json!(float)), format!("{text}\n"), "{float}");
        }
        assert_eq!(to_yaml(&json!(-3)), "-3\n");
    }

    #[test]
    fn strings_escape_what_yaml_rejects() {
        assert_eq!(
            to_yaml(&json!("a \"b\" \\ c\nd\u{7f}\u{1}é")),
            "\"a \\\"b\\\" \\\\ c\\nd\\u007F\\u0001é\"\n"
        );
    }

    #[test]
    fn keys_that_yaml_reads_as_booleans_are_quoted() {
        assert_eq!(
            to_yaml(&json!({"on": true, "off": null, "key with space": 1, "ok_key": false})),
            "\"key with space\": 1\n\"off\": null\nok_key: false\n\"on\": true\n"
        );
    }
}

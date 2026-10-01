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

/// The value of the top-level key `key` of the YAML document `text` when it
/// is a scalar on the key's own line: a key at the start of a line, so a
/// nested one of the same name is never taken; double-quoted (with escapes),
/// single-quoted (`''` for a quote) or plain, a plain one up to a ` #`
/// comment. `None` when the key is absent, or its value is empty or not
/// closed. Reads back what [`to_yaml`] writes, and a hand edit of it.
#[must_use]
pub fn top_level_scalar(text: &str, key: &str) -> Option<String> {
    let value = text
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix(':'))?
        .trim();
    if value.starts_with('"') {
        double_quoted(value)
    } else if let Some(rest) = value.strip_prefix('\'') {
        single_quoted(rest)
    } else {
        let plain = value.find(" #").map_or(value, |at| &value[..at]).trim();
        (!plain.is_empty() && !plain.starts_with('#')).then(|| plain.to_string())
    }
}

/// The double-quoted scalar `value` starts with, unescaped as JSON when it
/// can be, else as written between its quotes.
fn double_quoted(value: &str) -> Option<String> {
    let mut escaped = false;
    for (at, c) in value.char_indices().skip(1) {
        match c {
            _ if escaped => escaped = false,
            '\\' => escaped = true,
            '"' => {
                let quoted = &value[..=at];
                return serde_json::from_str(quoted)
                    .ok()
                    .or_else(|| Some(value[1..at].to_string()));
            },
            _ => {},
        }
    }
    None
}

/// The single-quoted scalar `rest` (after its opening quote) starts with.
fn single_quoted(rest: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = rest.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\'' {
            out.push(c);
        } else if chars.peek() == Some(&'\'') {
            chars.next();
            out.push('\'');
        } else {
            return Some(out);
        }
    }
    None
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
///
/// U+0085 (NEL) is deliberately left out even though YAML 1.1 counts it as printable:
/// `PyYAML` folds a bare NEL inside a double-quoted scalar as a line break (it reads
/// back as a space, or a newline when doubled), so it must always be escaped rather
/// than written literally.
fn printable(c: char) -> bool {
    matches!(
        c,
        ' '..='~'
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
    fn a_top_level_scalar_is_read_back_quoted_plain_or_commented() {
        let yaml = to_yaml(&json!({
            "base_model": "Qwen/Qwen3-0.6B",
            "sequence_len": 2048,
            "nested": {"base_model": "other"},
        }));
        assert_eq!(
            top_level_scalar(&yaml, "base_model").as_deref(),
            Some("Qwen/Qwen3-0.6B")
        );
        assert_eq!(
            top_level_scalar(&yaml, "sequence_len").as_deref(),
            Some("2048")
        );
        for (text, expected) in [
            ("base_model: Qwen/Qwen3-4B  # note\n", Some("Qwen/Qwen3-4B")),
            ("base_model: 'it''s/model' # note\n", Some("it's/model")),
            (
                "base_model: \"a \\\"b\\\" #c\" # note\n",
                Some("a \"b\" #c"),
            ),
            ("  base_model: nested\nbase_model: top\n", Some("top")),
            ("base_model_config: x\n", None),
            ("base_model: # nothing\n", None),
            ("base_model: 'open\n", None),
            ("base_model:\n", None),
        ] {
            assert_eq!(
                top_level_scalar(text, "base_model").as_deref(),
                expected,
                "{text}"
            );
        }
    }

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

    #[test]
    fn next_line_is_escaped_because_pyyaml_folds_it_into_a_space() {
        // PyYAML (YAML 1.1) treats U+0085 (NEL) as a line break inside a double-quoted
        // scalar and folds it away: `yaml.safe_load('"x\x85y"\n')` gives `'x y'`, and
        // two in a row give `'x\ny'`. Left unescaped, a NEL byte in the source text
        // would be silently corrupted on read-back, so it must always be escaped.
        assert_eq!(to_yaml(&json!("x\u{85}y")), "\"x\\u0085y\"\n");
    }

    #[test]
    fn line_and_paragraph_separators_are_left_unescaped() {
        // U+2028 and U+2029 are YAML 1.1 line-break characters too, but unlike NEL,
        // PyYAML does not fold them inside a double-quoted scalar: they round-trip
        // as themselves (verified with `yaml.safe_load`), so they stay in the
        // printable set and are written out literally rather than escaped.
        assert_eq!(to_yaml(&json!("x\u{2028}y")), "\"x\u{2028}y\"\n");
        assert_eq!(to_yaml(&json!("x\u{2029}y")), "\"x\u{2029}y\"\n");
    }
}

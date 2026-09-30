//! Redaction of log lines that come from outside overbrainer, such as a pod's
//! own logs: what is replaced by `***` before a line is written to disk, shown
//! or traced.
//!
//! A line loses every known secret (the literal values overbrainer holds), then
//! whatever looks like one: Runpod keys (`rpa_`, `rps_`), Hugging Face tokens
//! (`hf_`), PEM private key blocks, the value of `NAME=value` when NAME names a
//! key, token, secret or password (and of `NAME: value`, quoted or not, when
//! NAME ends with one of those words), runs of 200 or more base64 characters,
//! and lines that are nothing but 40 or more base64 characters (the body of a
//! PEM block whose header was not seen).

use secrecy::{ExposeSecret, SecretString};

/// What replaces a secret.
const MASK: &str = "***";

/// Known secrets shorter than this are not looked for: replacing a two-letter
/// value everywhere would ruin every line and hide nothing.
const MIN_KNOWN: usize = 4;

/// Token prefixes whose tokens are replaced, the prefix kept.
const TOKEN_PREFIXES: [&str; 3] = ["rpa_", "rps_", "hf_"];

/// Characters a token needs after its prefix to count as one.
const MIN_TOKEN: usize = 8;

/// Words that make the value of `NAME=value` a secret, when NAME holds one.
const SECRET_NAMES: [&str; 4] = ["KEY", "TOKEN", "SECRET", "PASSWORD"];

/// Base64 characters in a row from which the run is taken for key material.
const MIN_BASE64_RUN: usize = 200;

/// Base64 characters from which a line made of nothing else is taken for a
/// line of a PEM body (64 characters, the last one shorter).
const MIN_BASE64_LINE: usize = 40;

/// Start of a PEM header or footer.
const PEM_BEGIN: &str = "-----BEGIN ";
const PEM_END: &str = "-----END ";
/// What a PEM private key's header and footer end with.
const PEM_PRIVATE: &str = "PRIVATE KEY-----";

/// `line` with every secret replaced by `***`: each of `known` (values shorter
/// than 4 characters are ignored), then the patterns of the module. A PEM
/// private key block that starts on the line and does not end there is masked
/// to the end of the line; [`Redactor`] also masks the lines that follow.
#[must_use]
pub fn redact_line(line: &str, known: &[&str]) -> String {
    let (text, _) = redact_pem(&redact_known(line, known), false);
    patterns(&text)
}

/// Redacts a stream of lines: [`redact_line`], plus the lines inside a PEM
/// private key block spread over several lines.
pub struct Redactor {
    known: Vec<SecretString>,
    in_key: bool,
}

impl std::fmt::Debug for Redactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Redactor")
            .field("known", &self.known.len())
            .field("in_key", &self.in_key)
            .finish()
    }
}

impl Redactor {
    /// A redactor looking for `known` secrets.
    #[must_use]
    pub fn new(known: Vec<SecretString>) -> Self {
        Self {
            known,
            in_key: false,
        }
    }

    /// The next line of the stream, redacted.
    #[must_use]
    pub fn line(&mut self, line: &str) -> String {
        let known: Vec<&str> = self.known.iter().map(ExposeSecret::expose_secret).collect();
        let (text, in_key) = redact_pem(&redact_known(line, &known), self.in_key);
        self.in_key = in_key;
        patterns(&text)
    }
}

/// `line` without any of `known`.
fn redact_known(line: &str, known: &[&str]) -> String {
    known
        .iter()
        .filter(|secret| secret.len() >= MIN_KNOWN)
        .fold(line.to_string(), |text, secret| text.replace(secret, MASK))
}

/// The patterns that need no state, in order.
fn patterns(line: &str) -> String {
    if base64_line(line) {
        return MASK.to_string();
    }
    base64_runs(&assignments(&tokens(line)))
}

/// Whether `line` is nothing but 40 or more standard base64 characters,
/// letters and digits among them: a line of a PEM body.
fn base64_line(line: &str) -> bool {
    let text = line.trim();
    text.len() >= MIN_BASE64_LINE
        && text
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'='))
        && text.bytes().any(|byte| byte.is_ascii_alphabetic())
        && text.bytes().any(|byte| byte.is_ascii_digit())
}

/// Masks PEM private key material in `line`; `in_key` says whether a block
/// opened on an earlier line is still open. Returns the line and whether a
/// block is still open at its end.
fn redact_pem(line: &str, in_key: bool) -> (String, bool) {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    let mut open = in_key;
    loop {
        if open {
            let Some(end) = pem_footer_end(rest) else {
                if !rest.is_empty() {
                    out.push_str(MASK);
                }
                return (out, true);
            };
            out.push_str(MASK);
            rest = &rest[end..];
            open = false;
        } else {
            let Some(start) = pem_header_start(rest) else {
                out.push_str(rest);
                return (out, false);
            };
            out.push_str(&rest[..start]);
            rest = &rest[start..];
            open = true;
        }
    }
}

/// Where the first PEM private key header of `text` starts.
fn pem_header_start(text: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(found) = text[from..].find(PEM_BEGIN) {
        let start = from + found;
        let header = &text[start + PEM_BEGIN.len()..];
        if let Some(close) = header.find("-----")
            && header[..close + 5].ends_with(PEM_PRIVATE)
        {
            return Some(start);
        }
        from = start + PEM_BEGIN.len();
    }
    None
}

/// Where the first PEM private key footer of `text` ends.
fn pem_footer_end(text: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(found) = text[from..].find(PEM_END) {
        let start = from + found + PEM_END.len();
        let footer = &text[start..];
        if let Some(close) = footer.find("-----")
            && footer[..close + 5].ends_with(PEM_PRIVATE)
        {
            return Some(start + close + 5);
        }
        from = start;
    }
    None
}

/// Whether `byte` may be part of a word, for the boundaries of a token.
fn is_word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Masks every `rpa_`, `rps_` or `hf_` token, the prefix kept.
fn tokens(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut copied = 0;
    let mut at = 0;
    while at < bytes.len() {
        let boundary = at == 0 || !is_word(bytes[at - 1]);
        let prefix = TOKEN_PREFIXES
            .iter()
            .find(|prefix| boundary && bytes[at..].starts_with(prefix.as_bytes()));
        if let Some(prefix) = prefix {
            let body = at + prefix.len();
            let end = body
                + bytes[body..]
                    .iter()
                    .take_while(|byte| byte.is_ascii_alphanumeric())
                    .count();
            if end - body >= MIN_TOKEN {
                out.push_str(&line[copied..body]);
                out.push_str(MASK);
                copied = end;
                at = end;
                continue;
            }
        }
        at += 1;
    }
    out.push_str(&line[copied..]);
    out
}

/// Masks the value of every `NAME=value` whose NAME holds a secret word.
fn assignments(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut copied = 0;
    let mut at = 0;
    while at < bytes.len() {
        let Some(start) = value_start(bytes, at) else {
            at += 1;
            continue;
        };
        let Some((start, end)) = value_span(bytes, start) else {
            at += 1;
            continue;
        };
        out.push_str(&line[copied..start]);
        out.push_str(MASK);
        copied = end;
        at = end.max(at + 1);
    }
    out.push_str(&line[copied..]);
    out
}

/// Where the value starts when `at` is the separator of a secret: `=` after a
/// NAME holding a secret word, or `:` (then spaces) after a NAME ending with
/// one, the NAME possibly closed by a quote (`"HF_TOKEN": "..."`). The `:`
/// form asks more of its NAME since `tokenizer: ...` is common in logs.
fn value_start(bytes: &[u8], at: usize) -> Option<usize> {
    let separator = *bytes.get(at)?;
    if separator != b'=' && separator != b':' {
        return None;
    }
    let mut before = &bytes[..at];
    if separator == b':'
        && let Some((last, rest)) = before.split_last()
        && matches!(last, b'"' | b'\'')
    {
        before = rest;
    }
    let length = before
        .iter()
        .rev()
        .take_while(|byte| is_word(**byte))
        .count();
    let name = String::from_utf8_lossy(&before[before.len() - length..]).to_ascii_uppercase();
    if separator == b'=' {
        return SECRET_NAMES
            .iter()
            .any(|word| name.contains(word))
            .then_some(at + 1);
    }
    if !SECRET_NAMES.iter().any(|word| name.ends_with(word)) {
        return None;
    }
    let spaces = bytes[at + 1..]
        .iter()
        .take_while(|byte| **byte == b' ')
        .count();
    Some(at + 1 + spaces)
}

/// The span of the value starting at `start`: inside its quotes when quoted,
/// else up to a space, a quote, `,`, `;` or `&`. `None` for an empty value.
fn value_span(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let first = *bytes.get(start)?;
    if first == b'\'' || first == b'"' {
        let inner = start + 1;
        let length = bytes[inner..]
            .iter()
            .take_while(|byte| **byte != first)
            .count();
        return (length > 0).then_some((inner, inner + length));
    }
    let length = bytes[start..]
        .iter()
        .take_while(|byte| {
            !byte.is_ascii_whitespace() && !matches!(byte, b'\'' | b'"' | b',' | b';' | b'&')
        })
        .count();
    (length > 0).then_some((start, start + length))
}

/// Whether `byte` is a base64 character, standard or URL-safe.
fn is_base64(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'=' | b'-' | b'_')
}

/// Masks every run of 200 or more base64 characters holding both a letter and
/// a digit (a line of `=` or `-` is a separator, not key material).
fn base64_runs(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut copied = 0;
    let mut at = 0;
    while at < bytes.len() {
        if !is_base64(bytes[at]) {
            at += 1;
            continue;
        }
        let run = &bytes[at..];
        let length = run.iter().take_while(|byte| is_base64(**byte)).count();
        let run = &run[..length];
        if length >= MIN_BASE64_RUN
            && run.iter().any(u8::is_ascii_alphabetic)
            && run.iter().any(u8::is_ascii_digit)
        {
            out.push_str(&line[copied..at]);
            out.push_str(MASK);
            copied = at + length;
        }
        at += length;
    }
    out.push_str(&line[copied..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_line_is_unchanged() {
        let line = "step 10/200: loss=1.25 lr=2e-5, tokens/s 1234 · done";
        assert_eq!(redact_line(line, &[]), line);
    }

    #[test]
    fn known_secrets_are_masked_but_not_short_ones() {
        assert_eq!(
            redact_line("key abcd1234 and abcd1234", &["abcd1234", "ab", ""]),
            "key *** and ***"
        );
    }

    #[test]
    fn runpod_keys_and_hugging_face_tokens_are_masked() {
        assert_eq!(
            redact_line("using rpa_ABCDEF1234567890 and rps_XYZ98765432", &[]),
            "using rpa_*** and rps_***"
        );
        assert_eq!(
            redact_line("login hf_abcdefghijklmnopqrstuvwxyz0123", &[]),
            "login hf_***"
        );
        // Too short, or inside a word: not a token.
        assert_eq!(
            redact_line("hf_short my_hf_abcdefghijk", &[]),
            "hf_short my_hf_abcdefghijk"
        );
    }

    #[test]
    fn secret_assignments_lose_their_value() {
        assert_eq!(
            redact_line("RUNPOD_API_KEY=abc123 PATH=/usr/bin", &[]),
            "RUNPOD_API_KEY=*** PATH=/usr/bin"
        );
        assert_eq!(
            redact_line("export HF_TOKEN='a b c'; db_password=\"x\"", &[]),
            "export HF_TOKEN='***'; db_password=\"***\""
        );
        assert_eq!(redact_line("--api-key=zzz,next", &[]), "--api-key=***,next");
        assert_eq!(redact_line("MY_SECRET= empty", &[]), "MY_SECRET= empty");
    }

    #[test]
    fn quoted_and_colon_forms_lose_their_value() {
        assert_eq!(
            redact_line(r#"{"HF_TOKEN": "abc def", "n": 1}"#, &[]),
            r#"{"HF_TOKEN": "***", "n": 1}"#
        );
        assert_eq!(redact_line("{'api_key': 'xyz'}", &[]), "{'api_key': '***'}");
        assert_eq!(
            redact_line("password: hunter2 next", &[]),
            "password: *** next"
        );
        // Names that only contain a secret word, and URLs, stay.
        let kept = "tokenizer: loaded from https://huggingface.co/x";
        assert_eq!(redact_line(kept, &[]), kept);
    }

    #[test]
    fn a_replay_starting_inside_a_private_key_masks_its_body() {
        let body = "MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gun";
        let mut redactor = Redactor::new(Vec::new());
        assert_eq!(redactor.line(body), "***");
        assert_eq!(redactor.line("  dGVzdA=="), "  dGVzdA==");
        assert_eq!(redact_line(&body[..40], &[]), "***");
        // Plain words or a separator line of that length stay.
        let words = "a".repeat(50);
        assert_eq!(redact_line(&words, &[]), words);
    }

    #[test]
    fn a_private_key_on_one_line_is_masked() {
        let line = "key: -----BEGIN OPENSSH PRIVATE KEY-----\\nb3BlbnNzaA\\n-----END OPENSSH PRIVATE KEY----- end";
        assert_eq!(redact_line(line, &[]), "key: *** end");
        // A public key block stays.
        let public = "-----BEGIN PUBLIC KEY----- abc -----END PUBLIC KEY-----";
        assert_eq!(redact_line(public, &[]), public);
    }

    #[test]
    fn a_private_key_over_several_lines_is_masked_line_by_line() {
        let mut redactor = Redactor::new(Vec::new());
        let lines = [
            "before -----BEGIN RSA PRIVATE KEY-----",
            "MIIEowIBAAKCAQEA",
            "",
            "-----END RSA PRIVATE KEY----- after",
            "plain",
        ];
        let redacted: Vec<String> = lines.iter().map(|line| redactor.line(line)).collect();
        assert_eq!(redacted, ["before ***", "***", "", "*** after", "plain"]);
    }

    #[test]
    fn long_base64_runs_are_masked() {
        let run = "aB3+".repeat(50);
        assert_eq!(redact_line(&format!("env {run} end"), &[]), "env *** end");
        // Shorter, among other words: kept (alone on a line, it would be a
        // PEM body line).
        let short = format!("env {} end", &run[..199]);
        assert_eq!(redact_line(&short, &[]), short);
        let banner = "=".repeat(300);
        assert_eq!(redact_line(&banner, &[]), banner);
    }

    #[test]
    fn non_ascii_text_survives() {
        assert_eq!(
            redact_line("é TOKEN=ü€ fin hf_abcdefghijé", &[]),
            "é TOKEN=*** fin hf_***é"
        );
    }

    #[test]
    fn the_redactor_masks_known_secrets() {
        let mut redactor = Redactor::new(vec![SecretString::from("s3cr3tvalue")]);
        assert_eq!(redactor.line("x s3cr3tvalue y"), "x *** y");
    }
}

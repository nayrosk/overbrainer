use super::{ExecError, Executor};

/// Splits `bytes`, read from byte `offset` of a file, into complete lines. Returns
/// them with the offset just past the last one: an unterminated last line is left
/// for the next read.
#[must_use]
pub fn complete_lines(bytes: &[u8], offset: u64) -> (Vec<String>, u64) {
    let Some(end) = bytes.iter().rposition(|byte| *byte == b'\n') else {
        return (Vec::new(), offset);
    };
    let lines = bytes[..end]
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| String::from_utf8_lossy(line).into_owned())
        .collect();
    let consumed = u64::try_from(end + 1).unwrap_or(u64::MAX);
    (lines, offset.saturating_add(consumed))
}

/// A file on the target followed one read at a time, from a byte offset that can be
/// saved and given back to resume after a reconnection.
#[derive(Debug)]
pub struct LineStream<'a, E> {
    executor: &'a E,
    path: String,
    offset: u64,
}

impl<'a, E: Executor> LineStream<'a, E> {
    /// Follows `path` on `executor` from byte `offset`.
    #[must_use]
    pub fn new(executor: &'a E, path: &str, offset: u64) -> Self {
        Self {
            executor,
            path: path.to_string(),
            offset,
        }
    }

    /// Offset of the first byte not read yet.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The complete lines appended since the last read, possibly none.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecError`] when the file cannot be read.
    pub async fn read(&mut self) -> Result<Vec<String>, ExecError> {
        let bytes = self.executor.read_from(&self.path, self.offset).await?;
        let (lines, offset) = complete_lines(&bytes, self.offset);
        self.offset = offset;
        Ok(lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_complete_lines_are_returned() {
        assert_eq!(
            complete_lines(b"a\nbc\npartial", 10),
            (vec!["a".to_string(), "bc".to_string()], 15)
        );
        assert_eq!(complete_lines(b"partial", 3), (Vec::new(), 3));
        assert_eq!(complete_lines(b"\n\nx\n", 0), (vec!["x".to_string()], 4));
        assert_eq!(complete_lines(b"", 7), (Vec::new(), 7));
    }
}

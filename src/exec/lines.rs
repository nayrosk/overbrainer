use super::{ExecError, Executor};

/// Maximum bytes [`LineStream::read`] asks for in one poll, so a burst of output (or
/// a stalled connection) cannot pull an unbounded amount of data into memory at
/// once. A caller wanting more calls `read` again.
pub const MAX_TAIL_READ: u64 = 1_048_576;

/// Splits `bytes`, read from byte `offset` of a file, into complete lines, treating
/// `\n` and `\r` (as printed by a progress bar that redraws its line in place) alike
/// as terminators. Returns them with the offset just past the last terminator: an
/// unterminated last line is left for the next read.
#[must_use]
pub fn complete_lines(bytes: &[u8], offset: u64) -> (Vec<String>, u64) {
    let Some(end) = bytes
        .iter()
        .rposition(|byte| *byte == b'\n' || *byte == b'\r')
    else {
        return (Vec::new(), offset);
    };
    let lines = bytes[..end]
        .split(|byte| *byte == b'\n' || *byte == b'\r')
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
        self.read_with_limit(MAX_TAIL_READ).await
    }

    /// Implements [`Self::read`], reading at most `limit` bytes. Split out so tests
    /// can exercise the cap with a small `limit`, without allocating a multi
    /// megabyte buffer.
    ///
    /// When the read returns exactly `limit` bytes with no terminator anywhere in
    /// them, more data almost certainly remains past what was fetched: waiting for
    /// a terminator would never advance the offset, so this instead cuts the chunk
    /// into a line at its last complete UTF-8 character, keeping any trailing
    /// partial character for the next read.
    async fn read_with_limit(&mut self, limit: u64) -> Result<Vec<String>, ExecError> {
        let bytes = self
            .executor
            .read_from(&self.path, self.offset, limit)
            .await?;
        let (lines, offset) = complete_lines(&bytes, self.offset);
        if !lines.is_empty() || offset != self.offset {
            self.offset = offset;
            return Ok(lines);
        }
        let capped = u64::try_from(bytes.len()).unwrap_or(u64::MAX) >= limit;
        if capped && !bytes.is_empty() {
            let cut = match std::str::from_utf8(&bytes) {
                Ok(_) => bytes.len(),
                Err(error) => {
                    let valid = error.valid_up_to();
                    if valid > 0 { valid } else { bytes.len() }
                },
            };
            let line = String::from_utf8_lossy(&bytes[..cut]).into_owned();
            self.offset = self
                .offset
                .saturating_add(u64::try_from(cut).unwrap_or(u64::MAX));
            return Ok(vec![line]);
        }
        self.offset = offset;
        Ok(lines)
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::exec::{FileDigest, JobCommand, JobId, JobStatus};

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

    #[test]
    fn a_carriage_return_also_terminates_a_line() {
        assert_eq!(
            complete_lines(b"10%\r50%\r100%\npartial", 0),
            (
                vec!["10%".to_string(), "50%".to_string(), "100%".to_string()],
                13
            )
        );
    }

    /// Records the `limit` it was asked to read with, and serves bytes from a fixed
    /// buffer. Every other method is unused by these tests.
    struct RecordingExecutor {
        content: Vec<u8>,
        last_limit: AtomicU64,
    }

    impl Executor for RecordingExecutor {
        fn workdir(&self) -> &'static str {
            "/w"
        }

        fn upload(
            &self,
            _local: &Path,
            _remote: &str,
        ) -> impl Future<Output = Result<(), ExecError>> + Send {
            std::future::ready(Ok(()))
        }

        fn spawn(
            &self,
            _job: &JobCommand,
        ) -> impl Future<Output = Result<JobId, ExecError>> + Send {
            std::future::ready(Err(ExecError::Protocol("unused".to_string())))
        }

        fn read_from(
            &self,
            _path: &str,
            offset: u64,
            limit: u64,
        ) -> impl Future<Output = Result<Vec<u8>, ExecError>> + Send {
            self.last_limit.store(limit, Ordering::SeqCst);
            let start = usize::try_from(offset).unwrap_or(self.content.len());
            let mut bytes = self.content.get(start..).unwrap_or_default().to_vec();
            bytes.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
            std::future::ready(Ok(bytes))
        }

        fn status(
            &self,
            _job: &JobId,
        ) -> impl Future<Output = Result<JobStatus, ExecError>> + Send {
            std::future::ready(Err(ExecError::Protocol("unused".to_string())))
        }

        fn cancel(&self, _job: &JobId) -> impl Future<Output = Result<(), ExecError>> + Send {
            std::future::ready(Ok(()))
        }

        fn download(
            &self,
            _remote: &str,
            _local: &Path,
            _entries: &[String],
            _exclude: &[String],
        ) -> impl Future<Output = Result<(), ExecError>> + Send {
            std::future::ready(Ok(()))
        }

        fn manifest(
            &self,
            _remote: &str,
            _entries: &[String],
            _exclude: &[String],
        ) -> impl Future<Output = Result<Vec<FileDigest>, ExecError>> + Send {
            std::future::ready(Ok(Vec::new()))
        }
    }

    #[tokio::test]
    async fn read_caps_its_request_at_max_tail_read() -> Result<(), Box<dyn std::error::Error>> {
        let executor = RecordingExecutor {
            content: b"a\n".to_vec(),
            last_limit: AtomicU64::new(0),
        };
        let mut stream = LineStream::new(&executor, "job.log", 0);
        stream.read().await?;
        assert_eq!(executor.last_limit.load(Ordering::SeqCst), MAX_TAIL_READ);
        Ok(())
    }

    #[tokio::test]
    async fn read_never_livelocks_on_a_line_that_fills_the_whole_cap()
    -> Result<(), Box<dyn std::error::Error>> {
        let executor = RecordingExecutor {
            content: b"0123456789abcdef".to_vec(),
            last_limit: AtomicU64::new(0),
        };
        let mut stream = LineStream::new(&executor, "job.log", 0);

        let first = stream.read_with_limit(8).await?;
        assert_eq!(first, vec!["01234567".to_string()]);
        assert_eq!(stream.offset(), 8);

        let second = stream.read_with_limit(8).await?;
        assert_eq!(second, vec!["89abcdef".to_string()]);
        assert_eq!(stream.offset(), 16);
        Ok(())
    }

    #[tokio::test]
    async fn read_cuts_a_capped_chunk_at_a_utf8_boundary() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut content = b"ab".to_vec();
        content.extend_from_slice("\u{20ac}".as_bytes());
        content.extend_from_slice(b"cd");
        let executor = RecordingExecutor {
            content,
            last_limit: AtomicU64::new(0),
        };
        let mut stream = LineStream::new(&executor, "job.log", 0);

        // Caps at 4 bytes: "ab" (2 bytes) plus the first two of the three bytes
        // that encode the euro sign, cutting mid character.
        let first = stream.read_with_limit(4).await?;
        assert_eq!(first, vec!["ab".to_string()]);
        assert_eq!(stream.offset(), 2);
        Ok(())
    }
}

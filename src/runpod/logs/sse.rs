//! A reader of `text/event-stream` bytes, as Runpod's log stream sends them:
//! only `data:` and `id:` are read, comments and other fields are skipped,
//! `\r\n` is taken for `\n`. A line and an event are capped ([`MAX_LINE`],
//! [`MAX_FRAME`]), since the stream never ends.

/// Bytes kept of one line of the stream; the rest of the line is dropped.
pub(super) const MAX_LINE: usize = 1 << 20;

/// Bytes kept of one event's data; the rest is dropped.
pub(super) const MAX_FRAME: usize = 4 << 20;

/// One event of the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Frame {
    /// Its `id:`, the cursor to resume after it.
    pub(super) id: Option<String>,
    /// Its `data:` lines, joined; `None` for an event without data.
    pub(super) data: Option<String>,
    /// Whether some of its data was dropped.
    pub(super) truncated: bool,
}

/// The reader's state between chunks.
#[derive(Debug, Default)]
pub(super) struct SseReader {
    pending: Vec<u8>,
    /// Whether the line being read was cut at [`MAX_LINE`].
    cut: bool,
    data: String,
    has_data: bool,
    id: Option<String>,
    truncated: bool,
}

impl SseReader {
    /// Reads `chunk`, returning the events it completes.
    pub(super) fn feed(&mut self, chunk: &[u8]) -> Vec<Frame> {
        let mut frames = Vec::new();
        let mut rest = chunk;
        while let Some(at) = rest.iter().position(|byte| *byte == b'\n') {
            self.append(&rest[..at]);
            self.line(&mut frames);
            rest = &rest[at + 1..];
        }
        self.append(rest);
        frames
    }

    /// Ends the stream: a last line without newline and a last event without
    /// blank line still count.
    pub(super) fn finish(&mut self) -> Vec<Frame> {
        let mut frames = Vec::new();
        if !self.pending.is_empty() {
            self.line(&mut frames);
        }
        self.flush(&mut frames);
        frames
    }

    fn append(&mut self, piece: &[u8]) {
        if self.cut {
            return;
        }
        let room = MAX_LINE.saturating_sub(self.pending.len());
        if piece.len() > room {
            self.pending.extend_from_slice(&piece[..room]);
            self.cut = true;
        } else {
            self.pending.extend_from_slice(piece);
        }
    }

    fn line(&mut self, frames: &mut Vec<Frame>) {
        let bytes = std::mem::take(&mut self.pending);
        let cut = std::mem::take(&mut self.cut);
        let text = String::from_utf8_lossy(&bytes);
        let text = text.strip_suffix('\r').unwrap_or(&text);
        if text.is_empty() {
            self.flush(frames);
        } else if let Some(value) = field(text, "data") {
            if self.data.len() + value.len() + 1 > MAX_FRAME {
                self.truncated = true;
            } else {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
                self.truncated |= cut;
            }
        } else if let Some(value) = field(text, "id") {
            self.id = Some(value.to_string());
        }
    }

    fn flush(&mut self, frames: &mut Vec<Frame>) {
        let data = std::mem::take(&mut self.data);
        let has_data = std::mem::take(&mut self.has_data);
        let id = self.id.take();
        let truncated = std::mem::take(&mut self.truncated);
        if id.is_none() && !has_data {
            return;
        }
        frames.push(Frame {
            id,
            data: has_data.then_some(data),
            truncated,
        });
    }
}

/// The value of the field `name` on `line`, one space after the colon dropped.
fn field<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let value = line.strip_prefix(name)?.strip_prefix(':')?;
    Some(value.strip_prefix(' ').unwrap_or(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(chunks: &[&str]) -> Vec<Frame> {
        let mut reader = SseReader::default();
        let mut frames: Vec<Frame> = chunks
            .iter()
            .flat_map(|chunk| reader.feed(chunk.as_bytes()))
            .collect();
        frames.extend(reader.finish());
        frames
    }

    fn frame(id: Option<&str>, data: Option<&str>) -> Frame {
        Frame {
            id: id.map(str::to_string),
            data: data.map(str::to_string),
            truncated: false,
        }
    }

    #[test]
    fn events_are_read_across_chunks() {
        assert_eq!(
            frames(&[
                "id: a/1\nda",
                "ta: {\"x\":1}\n\nid: a/2\r\ndata: two\r\n",
                "\r\n"
            ]),
            [
                frame(Some("a/1"), Some("{\"x\":1}")),
                frame(Some("a/2"), Some("two"))
            ]
        );
    }

    #[test]
    fn comments_and_other_fields_are_skipped_and_keepalives_move_the_cursor() {
        assert_eq!(
            frames(&[": ping\n\nevent: log\nretry: 5\nid: k/9\n\n"]),
            [frame(Some("k/9"), None)]
        );
    }

    #[test]
    fn several_data_lines_join_and_a_last_event_without_blank_line_counts() {
        assert_eq!(
            frames(&["data: a\ndata:b\ndata:  c"]),
            [frame(None, Some("a\nb\n c"))]
        );
    }

    #[test]
    fn an_oversized_data_line_is_cut_and_marked() {
        let long = "x".repeat(MAX_LINE + 10);
        let events = frames(&["data: ", &long, "\n\ndata: next\n\n"]);
        assert_eq!(events.len(), 2);
        assert!(events[0].truncated);
        assert_eq!(events[0].data.as_ref().map(String::len), Some(MAX_LINE - 6));
        assert_eq!(events[1].data.as_deref(), Some("next"));
        assert!(!events[1].truncated);
    }

    #[test]
    fn an_oversized_comment_marks_nothing() {
        let long = "x".repeat(MAX_LINE + 10);
        let events = frames(&[": ", &long, "\ndata: one\n\n"]);
        assert_eq!(events, [frame(None, Some("one"))]);
    }
}

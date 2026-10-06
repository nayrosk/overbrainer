//! The Dataset view: the tree on the left, the detail or stats pane on the right.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Row, Table, Wrap};
use tui_tree_widget::Tree;

use crate::dataset::{AnswerText, Example, FinishReason, Id, ReasoningKind};
use crate::pipeline::SplitClass;
use crate::tui::app::App;
use crate::tui::dataset::{Model, Node, Stats, exclusion, sizes};
use crate::tui::format::WORKING;
use crate::tui::theme::Theme;

/// Columns of the tree's open or closed marker before a topic's label.
const TOGGLE: u16 = 2;
/// The first keys to know on a new project, in the order they are dropped
/// from the end when they do not all fit.
const FIRST_KEYS: [&str; 3] = ["r generates data", "t trains", "? all keys"];

/// A rounded pane with a column of padding, its border drawn in `border`.
pub(in crate::tui) fn pane<'a>(border: ratatui::style::Style) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(border)
        .padding(Padding::horizontal(1))
}

/// Draws the Dataset view in `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &mut App) {
    let theme = app.theme;
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)]).areas(area);
    let onboarding = app.training.runs.is_empty();
    let loading = app.motion.shows_load(app.load_at).then(|| {
        // The static glyph already ends in an ellipsis: none after the path.
        match app.motion.spinner() {
            WORKING => format!("{WORKING} reading data/"),
            spinner => format!("{spinner} reading data/…"),
        }
    });
    let view = &mut app.dataset;
    let Some(model) = &mut view.model else {
        let text = match (view.error.clone(), loading) {
            (Some(error), _) => failed(&error, &theme),
            (None, Some(loading)) => vec![Line::from(Span::styled(loading, theme.dim))],
            (None, None) => Vec::new(),
        };
        let block = pane(theme.border_focus).title(Span::styled(" data ", theme.title));
        frame.render_widget(
            Paragraph::new(text).wrap(Wrap { trim: false }).block(block),
            area,
        );
        return;
    };
    let title = match model.matches {
        Some(1) => " data: 1 match ".to_string(),
        Some(matches) => format!(" data: {matches} matches "),
        None => " data ".to_string(),
    };
    let bottom = match (&view.input, view.filter.is_empty()) {
        (Some(_), _) => Line::from(Span::styled(" /", theme.dim)),
        (None, true) => Line::from(Span::styled(" / filter ", theme.dim)),
        (None, false) => Line::from(Span::styled(
            format!(" filter: {} (Esc clears) ", view.filter),
            theme.dim,
        )),
    };
    let block = pane(theme.border_focus)
        .title(Span::styled(title, theme.title))
        .title_bottom(bottom);
    let inner = block.inner(left);
    model.fit(usize::from(inner.width.saturating_sub(TOGGLE)));
    let data = &model.data;
    let nothing = data.subtopics.is_empty() && data.questions.is_empty() && data.answers.is_empty();
    match &model.items {
        Ok(items) => match Tree::new(items) {
            Ok(tree) if items.is_empty() || (nothing && model.matches.is_none()) => {
                // The topics stay listed, their counts at zero, the selection
                // on one of them; what fills them comes under them.
                let lines = if model.matches.is_some() {
                    vec![Line::from(Span::styled(
                        "Nothing matches the filter: Esc clears it.",
                        theme.dim,
                    ))]
                } else {
                    empty(onboarding, inner.width, &theme)
                };
                let rows = u16::try_from(items.len()).unwrap_or(u16::MAX);
                let [listed, _, text] = Layout::vertical([
                    Constraint::Length(rows),
                    Constraint::Length(u16::from(rows > 0)),
                    Constraint::Fill(1),
                ])
                .areas(inner);
                frame.render_widget(block, left);
                let tree = tree.highlight_style(theme.selected);
                frame.render_stateful_widget(tree, listed, &mut view.tree);
                frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), text);
            },
            Ok(tree) => {
                let tree = tree.block(block).highlight_style(theme.selected);
                frame.render_stateful_widget(tree, left, &mut view.tree);
            },
            Err(error) => error_pane(frame, left, block, &error.to_string(), &theme),
        },
        Err(error) => error_pane(frame, left, block, error, &theme),
    }
    if let Some(input) = &view.input {
        // On the bottom border, after the corner and the " /" title.
        let row = Rect {
            x: left.x.saturating_add(3),
            y: left.bottom().saturating_sub(1),
            width: left.width.saturating_sub(4),
            height: left.height.min(1),
        };
        input.render(frame, row, theme.dim);
    }
    if view.stats {
        let topic = match view.tree.selected().first() {
            Some(Node::Topic(topic) | Node::MissingSubtopic(topic)) => Some(topic.clone()),
            _ => None,
        };
        render_stats(frame, right, app, topic.as_deref());
    } else {
        render_detail(frame, right, app);
    }
}

/// What an empty dataset says: which key fills it; with no run either, the
/// first keys to know, as many whole as fit `width` on one line.
fn empty(onboarding: bool, width: u16, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(
        "No data yet: press r to generate subtopics, questions and answers.",
    )];
    if onboarding {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            first_keys(usize::from(width)),
            theme.dim,
        )));
    }
    lines
}

/// The first keys to know, joined by ` · `: as many as fit `width` whole,
/// in order.
fn first_keys(width: usize) -> String {
    let mut line = String::new();
    for hint in FIRST_KEYS {
        let joined = if line.is_empty() {
            hint.to_string()
        } else {
            format!("{line} · {hint}")
        };
        if joined.chars().count() > width {
            break;
        }
        line = joined;
    }
    line
}

/// What a failed read says: `✗`, the error, and the key that reads again.
pub(in crate::tui) fn failed(error: &str, theme: &Theme) -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(format!("✗ {error}"), theme.error)),
        Line::from(Span::styled("R reloads", theme.dim)),
    ]
}

fn error_pane(frame: &mut Frame, area: Rect, block: Block, error: &str, theme: &Theme) {
    let text = Paragraph::new(failed(error, theme))
        .wrap(Wrap { trim: false })
        .block(block);
    frame.render_widget(text, area);
}

fn render_detail(frame: &mut Frame, area: Rect, app: &mut App) {
    let theme = app.theme;
    let view = &mut app.dataset;
    let (lines, parts) = match (&view.model, view.tree.selected().last()) {
        (Some(model), Some(node)) => detail(model, node, &app.project.topics, &theme),
        _ => (Vec::new(), Vec::new()),
    };
    let title = match view.tree.selected().last() {
        Some(Node::Topic(_)) => " topic ",
        Some(Node::Subtopic(_) | Node::MissingSubtopic(_)) => " subtopic ",
        Some(Node::Question(_)) => " question ",
        None => " detail ",
    };
    let inner_width = area.width.saturating_sub(4);
    // Where each part starts: only its prefix is copied, to count its rows.
    let offsets: Vec<usize> = parts
        .into_iter()
        .map(|part| {
            lines.get(..part).map_or(0, |prefix| {
                Paragraph::new(prefix.to_vec())
                    .wrap(Wrap { trim: false })
                    .line_count(inner_width)
            })
        })
        .collect();
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let height = usize::from(area.height.saturating_sub(2));
    let total = paragraph.line_count(inner_width);
    let last = u16::try_from(total.saturating_sub(height)).unwrap_or(u16::MAX);
    view.scroll = view.scroll.min(last);
    view.sections.clear();
    for offset in offsets {
        let offset = u16::try_from(offset).unwrap_or(u16::MAX).min(last);
        if view.sections.last() != Some(&offset) {
            view.sections.push(offset);
        }
    }
    let position = format!(
        " {}/{} ",
        (usize::from(view.scroll) + height).min(total),
        total
    );
    let block = pane(theme.border)
        .title(Span::styled(title, theme.title))
        .title_bottom(Line::from(Span::styled(position, theme.dim)).right_aligned());
    frame.render_widget(paragraph.scroll((view.scroll, 0)).block(block), area);
}

/// `id` shortened to its first and last four characters.
pub(in crate::tui) fn short(id: &Id) -> String {
    let text = id.as_str();
    match (text.get(..4), text.get(text.len().saturating_sub(4)..)) {
        (Some(head), Some(tail)) if text.len() > 8 => format!("{head}…{tail}"),
        _ => text.to_string(),
    }
}

/// The lines of the detail of `node`, and the index of the first line of each
/// part of a question (the question, its reasoning, its answer).
fn detail(
    model: &Model,
    node: &Node,
    topics: &[crate::tui::dataset::TopicInfo],
    theme: &Theme,
) -> (Vec<Line<'static>>, Vec<usize>) {
    let lines = match node {
        Node::Question(id) => return question_detail(model, id, theme),
        Node::Topic(topic) => topic_detail(model, topic, topics, theme),
        Node::Subtopic(id) => {
            let Some(subtopic) = model.data.subtopics.iter().find(|s| &s.id == id) else {
                return (Vec::new(), Vec::new());
            };
            let questions: Vec<&Id> = model
                .data
                .questions
                .iter()
                .filter(|q| &q.subtopic_id == id)
                .map(|q| &q.id)
                .collect();
            let answers = questions
                .iter()
                .filter(|q| model.answer(q).is_some())
                .count();
            vec![
                Line::from(Span::styled(subtopic.name.clone(), theme.title)),
                Line::from(Span::styled(
                    format!("id {}  topic {}", subtopic.id, subtopic.topic),
                    theme.dim,
                )),
                Line::from(format!("{} questions, {answers} answers", questions.len())),
            ]
        },
        Node::MissingSubtopic(topic) => vec![
            Line::from(Span::styled("(missing subtopic)", theme.title)),
            Line::from(format!(
                "Questions of {topic} whose subtopic is not in data/subtopics.jsonl, left \
                 by an interrupted change or a hand edit. d deletes them with their answers."
            )),
        ],
    };
    (lines, Vec::new())
}

fn topic_detail(
    model: &Model,
    topic: &str,
    topics: &[crate::tui::dataset::TopicInfo],
    theme: &Theme,
) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(topic.to_string(), theme.title))];
    match topics.iter().find(|info| info.name == topic) {
        Some(info) => {
            if let Some(description) = &info.description {
                lines.push(Line::from(description.clone()));
            }
            lines.push(Line::from(Span::styled(
                format!(
                    "configured: {} subtopics, {} questions each",
                    info.subtopics, info.questions_per_subtopic
                ),
                theme.dim,
            )));
        },
        None => lines.push(Line::from(Span::styled(
            "not in overbrainer.toml: split counts its answers as orphaned",
            theme.warn,
        ))),
    }
    if let Some(stats) = model.stats.get(&Some(topic.to_string())) {
        lines.push(Line::from(format!(
            "subtopics {}, questions {}, answers {}, unanswered {}",
            stats.subtopics, stats.questions, stats.answers, stats.unanswered
        )));
    }
    lines
}

/// The question, its metadata, then its answer when it has one; with the index
/// of the first line of each part.
fn question_detail(model: &Model, id: &Id, theme: &Theme) -> (Vec<Line<'static>>, Vec<usize>) {
    let Some(question) = model.question(id) else {
        return (Vec::new(), Vec::new());
    };
    let status = match model.class(id) {
        None => "unanswered".to_string(),
        Some(SplitClass::Usable) => "answered, usable".to_string(),
        Some(SplitClass::Excluded(reason)) => format!("answered, excluded: {}", exclusion(reason)),
        Some(SplitClass::Orphaned) => "answered, orphaned".to_string(),
    };
    let mut lines: Vec<Line<'static>> = question
        .text
        .lines()
        .map(|l| Line::from(l.to_string()))
        .collect();
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("id {}  {status}", short(id)),
        theme.dim,
    )));
    let mut parts = vec![0];
    if let Some(example) = model.answer(id) {
        lines.push(Line::from(""));
        let offset = lines.len();
        let (answer, headers) = answer_detail(model, example, theme);
        if headers.is_empty() {
            // No reasoning or answer header: the metadata itself is the section,
            // so `]` can still reach it.
            parts.push(offset);
        } else {
            parts.extend(headers.into_iter().map(|header| header + offset));
        }
        lines.extend(answer);
    }
    (lines, parts)
}

/// The metadata and text of an answer, with the index of each header line.
fn answer_detail(
    model: &Model,
    example: &Example,
    theme: &Theme,
) -> (Vec<Line<'static>>, Vec<usize>) {
    let meta = &example.meta;
    let excluded = match model.class(&example.id) {
        Some(SplitClass::Excluded(reason)) => format!("  excluded: {}", exclusion(reason)),
        Some(SplitClass::Orphaned) => "  orphaned".to_string(),
        Some(SplitClass::Usable) | None => String::new(),
    };
    let mut lines = vec![
        Line::from(Span::styled(
            format!(
                "model {}  in {}  out {}",
                meta.model, meta.input_tokens, meta.output_tokens
            ),
            theme.dim,
        )),
        Line::from(Span::styled(
            format!(
                "finish {}  reasoning {}{excluded}",
                finish(meta.finish_reason),
                reasoning(meta.reasoning_kind)
            ),
            theme.dim,
        )),
    ];
    let Some(text) = AnswerText::of(example) else {
        return (lines, Vec::new());
    };
    let mut headers = Vec::new();
    if let Some(reasoning) = &text.reasoning {
        headers.push(lines.len());
        lines.push(Line::from(Span::styled("── reasoning ──", theme.title)));
        lines.extend(reasoning.lines().map(|l| Line::from(l.to_string())));
    }
    headers.push(lines.len());
    lines.push(Line::from(Span::styled("── answer ──", theme.title)));
    lines.extend(text.content.lines().map(|l| Line::from(l.to_string())));
    (lines, headers)
}

fn finish(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Refusal => "refusal",
        FinishReason::Other => "other",
    }
}

fn reasoning(kind: ReasoningKind) -> &'static str {
    match kind {
        ReasoningKind::Raw => "raw",
        ReasoningKind::Summary => "summary",
        ReasoningKind::Redacted => "redacted",
        ReasoningKind::None => "none",
    }
}

fn render_stats(frame: &mut Frame, area: Rect, app: &App, topic: Option<&str>) {
    let theme = &app.theme;
    let Some(model) = &app.dataset.model else {
        return;
    };
    let empty = Stats::default();
    let stats = model
        .stats
        .get(&topic.map(str::to_string))
        .unwrap_or(&empty);
    let all = model.stats.get(&None).unwrap_or(&empty);
    let of = |have: usize, target: Option<u64>| {
        target.map_or_else(|| have.to_string(), |target| format!("{have} / {target}"))
    };
    let excluded: usize = stats.excluded.values().sum();
    let reasons: Vec<String> = stats
        .excluded
        .iter()
        .map(|(reason, count)| format!("{} {count}", exclusion(*reason)))
        .collect();
    let excluded = if reasons.is_empty() {
        excluded.to_string()
    } else {
        format!("{excluded} ({})", reasons.join(", "))
    };
    let lengths = |l: crate::tui::dataset::Lengths| format!("mean {}, max {}", l.mean(), l.max);
    let (train, eval, label) = sizes(
        all.usable,
        app.project.eval_ratio,
        app.dataset.split.as_ref(),
    );
    let rows = [
        ("subtopics", of(stats.subtopics, stats.subtopics_target)),
        ("questions", of(stats.questions, stats.questions_target)),
        ("answers", stats.answers.to_string()),
        ("unanswered", stats.unanswered.to_string()),
        ("usable", stats.usable.to_string()),
        ("excluded", excluded),
        ("orphaned", stats.orphaned.to_string()),
        ("question chars", lengths(stats.question_chars)),
        ("answer chars", lengths(stats.answer_chars)),
        ("reasoning chars", lengths(stats.reasoning_chars)),
        (
            "tokens",
            format!("in {}, out {}", stats.input_tokens, stats.output_tokens),
        ),
        ("rejected", stats.rejected.to_string()),
        ("", String::new()),
        (
            "all topics",
            format!("{} questions, {} answers", all.questions, all.answers),
        ),
        ("train / eval", format!("{train} / {eval} ({label})")),
    ];
    let rows: Vec<Row> = rows
        .into_iter()
        .map(|(label, value)| Row::new(vec![Span::styled(label, theme.dim), Span::raw(value)]))
        .collect();
    let title = format!(" stats: {} ", topic.unwrap_or("all topics"));
    let block = pane(theme.border).title(Span::styled(title, theme.title));
    let table = Table::new(rows, [Constraint::Length(16), Constraint::Fill(1)]).block(block);
    frame.render_widget(table, area);
}

#[cfg(test)]
mod tests {
    use ratatui::style::Style;

    use super::*;
    use crate::dataset::Role;
    use crate::tui::snapshots::{dataset, topics};
    use crate::tui::theme::{ColorLevel, Theme};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn an_answer_without_an_assistant_message_still_gets_a_section() -> TestResult {
        let mut data = dataset();
        let id = data.answers[0].id.clone();
        data.answers[0]
            .messages
            .retain(|message| message.role != Role::Assistant);
        let model = Model::new(data, &topics(), ("", None), Style::new());
        let theme = Theme::new(ColorLevel::TrueColor);
        let (lines, parts) = question_detail(&model, &id, &theme);
        assert_eq!(parts.len(), 2, "{parts:?}");
        let metadata_line = lines
            .get(parts[1])
            .ok_or("no line at the section's index")?;
        let text: String = metadata_line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.starts_with("model "), "{text}");
        Ok(())
    }
}

//! Draws the init wizard: a header with the step, the screen in a box, the
//! keys in the footer, and the quit question over it. A secret is only ever
//! drawn masked.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Wrap};

use super::{Field, STEPS, Screen, TOPIC_FIELDS, Wizard, protocol_name};
use crate::cli::init::{ENV_EXAMPLE_FILE, ENV_FILE, GITIGNORE};
use crate::config::CONFIG_FILE;
use crate::tui::app::{Action, Confirm};
use crate::tui::theme::Theme;
use crate::tui::widgets::{centered, dialog, too_small};

/// The widest the screen's box gets.
const MAX_WIDTH: u16 = 84;
/// Columns of a label, before its value.
const LABEL: usize = 12;

/// Draws `wizard` on `frame` with `theme`.
pub(super) fn render(frame: &mut Frame, wizard: &Wizard, theme: &Theme) {
    let area = frame.area();
    frame.buffer_mut().set_style(area, theme.base);
    if too_small::too_small(area) {
        too_small::render(frame, area);
        return;
    }
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(1),
    ])
    .areas(area);
    let screen = wizard.screen();
    let step = screen
        .step()
        .map_or_else(String::new, |step| format!("step {step} of {STEPS} "));
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ⠿ overbrainer", theme.accent),
            Span::styled("   new project", theme.title),
        ])),
        header,
    );
    frame.render_widget(
        Paragraph::new(Line::styled(step, theme.dim).right_aligned()),
        header,
    );
    let width = body.width.saturating_sub(4).min(MAX_WIDTH);
    let popup = centered(body, width, body.height);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme.border_focus)
        .title(Span::styled(format!(" {} ", screen.title()), theme.title))
        .padding(Padding::new(2, 2, 1, 0));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    let lines = match screen {
        Screen::Topics => topics(wizard, inner.width, theme),
        Screen::Summary => summary(wizard, theme),
        Screen::Start => start(wizard, theme),
        _ => fields(wizard, inner.width, theme),
    };
    frame.render_widget(
        Paragraph::new(lines)
            .style(theme.text_hi)
            .wrap(Wrap { trim: false }),
        inner,
    );
    frame.render_widget(
        Paragraph::new(Line::styled(keys(wizard), theme.dim)),
        footer,
    );
    if wizard.quitting() {
        quit_dialog(frame, area, theme);
    }
}

/// The keys the screen shown takes.
fn keys(wizard: &Wizard) -> String {
    let keys = if wizard.topic_form().is_some() {
        " ↑↓ field · Enter keep · Esc drop"
    } else {
        match wizard.screen() {
            Screen::Topics => {
                " a add · e edit · d delete · ↑↓ select · Enter next · Esc back · Ctrl-C quit"
            },
            Screen::Summary => " Enter write the files · Esc back · Ctrl-C quit",
            Screen::Start => " y yes · n no · ←→ choose · Enter answer",
            _ => " Enter next · Esc back · ↑↓ field · ←→ choose · Ctrl-C quit",
        }
    };
    keys.to_string()
}

/// The fields of the screen: label and value, then under the focused one
/// its hint, or why Next was refused.
fn fields(wizard: &Wizard, width: u16, theme: &Theme) -> Vec<Line<'static>> {
    let focused = wizard.focused();
    let value_width = width.saturating_sub(u16::try_from(LABEL).unwrap_or(0));
    let mut lines = Vec::new();
    if wizard.screen() == Screen::ApiKey {
        lines.push(Line::from(format!(
            "The provider's key goes to {ENV_FILE} (mode 600) and is never shown again."
        )));
        lines.push(Line::default());
    }
    for field in wizard.fields() {
        let focus = focused == Some(field);
        let label = Span::styled(
            format!("{:<LABEL$}", field.label()),
            if focus { theme.key } else { theme.dim },
        );
        let value = if field.is_choice() {
            let text = format!("‹ {} ›", wizard.display(field));
            Line::from(vec![
                label,
                Span::styled(text, if focus { theme.selected } else { theme.text_hi }),
            ])
        } else {
            match wizard.shown_input(field) {
                Some(input) if focus => {
                    let mut line = input.line(value_width, theme.text_hi);
                    line.spans.insert(0, label);
                    line
                },
                Some(input) => Line::from(vec![label, Span::raw(input.text().to_string())]),
                None => Line::from(label),
            }
        };
        lines.push(value);
        if focus {
            lines.push(under(wizard, field, theme));
        }
        if field == Field::Provider {
            lines.extend(preset_lines(wizard, theme));
        }
        lines.push(Line::default());
    }
    lines
}

/// What shows under the focused `field`: why Next was refused, else its hint.
fn under(wizard: &Wizard, field: Field, theme: &Theme) -> Line<'static> {
    let pad = " ".repeat(LABEL);
    match wizard.error() {
        Some(error) => Line::styled(format!("{pad}{error}"), theme.error),
        None => Line::styled(format!("{pad}{}", field.hint()), theme.dim),
    }
}

/// What a preset fills in, shown under the provider.
fn preset_lines(wizard: &Wizard, theme: &Theme) -> Vec<Line<'static>> {
    if wizard.fields().len() > 1 {
        return Vec::new();
    }
    let pad = " ".repeat(LABEL);
    vec![Line::styled(
        format!(
            "{pad}{}, {}",
            protocol_name(wizard.provider_protocol()),
            wizard.base_url()
        ),
        theme.text_hi,
    )]
}

/// The topics, or the form of the one being added or edited.
fn topics(wizard: &Wizard, width: u16, theme: &Theme) -> Vec<Line<'static>> {
    if let Some(form) = wizard.topic_form() {
        let value_width = width.saturating_sub(24);
        let mut lines = vec![
            Line::styled(
                if form.editing.is_some() {
                    "Edit the topic"
                } else {
                    "New topic"
                },
                theme.title,
            ),
            Line::default(),
        ];
        for (at, (label, input)) in TOPIC_FIELDS.iter().zip(&form.inputs).enumerate() {
            let focus = at == form.focus;
            let label = Span::styled(
                format!("{label:<24}"),
                if focus { theme.key } else { theme.dim },
            );
            let mut line = if focus {
                input.line(value_width, theme.text_hi)
            } else {
                Line::from(input.text().to_string())
            };
            line.spans.insert(0, label);
            lines.push(line);
            if focus && let Some(error) = &form.error {
                lines.push(Line::styled(format!("{:24}{error}", ""), theme.error));
            }
        }
        return lines;
    }
    let mut lines = vec![
        Line::from("The subjects the dataset covers: each gets subtopics, then questions."),
        Line::default(),
    ];
    if wizard.topics().is_empty() {
        lines.push(Line::styled("No topic yet: press a to add one.", theme.dim));
    }
    for (at, topic) in wizard.topics().iter().enumerate() {
        let selected = at == wizard.selected();
        let marker = if selected { "▶ " } else { "  " };
        let text = format!(
            "{marker}{}  {} subtopics x {} questions  {}",
            topic.name, topic.subtopics, topic.questions_per_subtopic, topic.description
        );
        lines.push(Line::styled(
            text,
            if selected {
                theme.selected
            } else {
                theme.text_hi
            },
        ));
    }
    if let Some(error) = wizard.error() {
        lines.push(Line::default());
        lines.push(Line::styled(error.to_string(), theme.error));
    }
    lines
}

/// What will be written.
fn summary(wizard: &Wizard, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = wizard
        .summary()
        .into_iter()
        .map(|(label, value)| {
            Line::from(vec![
                Span::styled(format!("{label:<LABEL$}"), theme.dim),
                Span::raw(value),
            ])
        })
        .collect();
    lines.push(Line::default());
    lines.push(Line::from(format!(
        "Enter writes {CONFIG_FILE}, {ENV_FILE} (mode 600), {ENV_EXAMPLE_FILE}, prompts/ and \
         {GITIGNORE} entries. No file is overwritten."
    )));
    if let Some(error) = wizard.error() {
        lines.push(Line::default());
        lines.push(Line::styled(error.to_string(), theme.error));
    }
    lines
}

/// The last question: start auto mode now?
fn start(wizard: &Wizard, theme: &Theme) -> Vec<Line<'static>> {
    let answer = |text: &'static str, chosen: bool| {
        Span::styled(
            format!(" {text} "),
            if chosen { theme.selected } else { theme.dim },
        )
    };
    vec![
        Line::from(format!(
            "✓ wrote {CONFIG_FILE}, {ENV_FILE}, {ENV_EXAMPLE_FILE}, prompts/ and {GITIGNORE}."
        )),
        Line::default(),
        Line::from(
            "Start auto now? It runs subtopics, questions, answers and split, then trains, \
             after one confirmation. No opens the Project view.",
        ),
        Line::default(),
        Line::from(vec![
            answer("y yes", wizard.auto()),
            Span::raw("  "),
            answer("n no", !wizard.auto()),
        ]),
    ]
}

/// Ctrl-C asks before anything is lost.
fn quit_dialog(frame: &mut Frame, area: Rect, theme: &Theme) {
    let confirm = Confirm {
        title: " Quit the wizard? ".to_string(),
        text: vec!["Nothing is written: the answers are lost.".to_string()],
        yes: "quit",
        no: "stay",
        action: Action::Quit,
    };
    dialog::render(frame, area, &confirm, theme, true);
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::super::{PRESETS, Secret, TopicDraft};
    use super::*;
    use crate::tui::theme::ColorLevel;
    use crate::tui::widgets::form::Input;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const SECRET: &str = "sk-placeholder-secret";

    /// Where the wizard's snapshots are kept.
    const SNAPSHOTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/wizard");

    fn draw(
        wizard: &Wizard,
        width: u16,
        height: u16,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let theme = Theme::new(ColorLevel::TrueColor);
        let mut terminal = Terminal::new(TestBackend::new(width, height))?;
        terminal.draw(|frame| render(frame, wizard, &theme))?;
        Ok(terminal.backend().to_string())
    }

    /// Checks `wizard` against `<name>_80x24` and `<name>_120x40`, and that
    /// no secret is drawn.
    fn snapshot(name: &str, wizard: &Wizard) -> TestResult {
        for (width, height) in [(80, 24), (120, 40)] {
            let screen = draw(wizard, width, height)?;
            assert!(!screen.contains(SECRET), "{name}: a secret is drawn");
            let mut settings = insta::Settings::clone_current();
            settings.set_snapshot_path(SNAPSHOTS);
            settings.set_prepend_module_to_snapshot(false);
            settings.set_omit_expression(true);
            let name = format!("{name}_{width}x{height}");
            settings.bind(|| insta::assert_snapshot!(name, screen));
        }
        Ok(())
    }

    /// A wizard with every screen's values filled, on `screen`.
    fn filled(screen: Screen) -> Wizard {
        let mut wizard = Wizard::new("rust_expert");
        wizard.provider = 1;
        wizard.api_key = Secret(Input::new(SECRET));
        wizard.generator = Input::new("qwen/qwen3-235b-a22b");
        wizard.parent = Input::new("deepseek/deepseek-r1");
        wizard.topics = vec![
            TopicDraft {
                name: "ownership".into(),
                description: "Moves, borrows and lifetimes".into(),
                subtopics: 10,
                questions_per_subtopic: 30,
            },
            TopicDraft {
                name: "traits".into(),
                description: String::new(),
                subtopics: 5,
                questions_per_subtopic: 20,
            },
        ];
        wizard.training = super::super::TrainingKind::Runpod;
        wizard.runpod_key = Secret(Input::new(SECRET));
        wizard.screen = screen;
        wizard
    }

    #[test]
    fn every_screen_is_drawn_and_no_secret_is() -> TestResult {
        for (name, screen) in [
            ("wizard_name", Screen::Name),
            ("wizard_provider", Screen::Provider),
            ("wizard_api_key", Screen::ApiKey),
            ("wizard_roles", Screen::Roles),
            ("wizard_topics", Screen::Topics),
            ("wizard_training", Screen::Training),
            ("wizard_model", Screen::Model),
            ("wizard_summary", Screen::Summary),
            ("wizard_start", Screen::Start),
        ] {
            snapshot(name, &filled(screen))?;
        }
        Ok(())
    }

    #[test]
    fn a_custom_provider_a_refusal_the_topic_form_and_quitting_are_drawn() -> TestResult {
        let mut custom = filled(Screen::Provider);
        custom.provider = PRESETS.len();
        custom.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        snapshot("wizard_provider_custom_refused", &custom)?;

        let mut form = filled(Screen::Topics);
        form.on_key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::NONE));
        snapshot("wizard_topic_form", &form)?;

        let mut quitting = filled(Screen::ApiKey);
        quitting.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        snapshot("wizard_quit", &quitting)
    }
}

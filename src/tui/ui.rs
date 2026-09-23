//! Draws the whole frame from the app's state: header, view, status line and
//! overlays. Reads nothing but the app.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Tabs};

use super::app::{App, Overlay, View};
use super::views;
use super::widgets::{help, status, too_small};

/// Draws `app` on `frame`.
pub(super) fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
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
    render_header(frame, header, app);
    match app.view {
        View::Logs => views::logs::render(frame, body, app),
        View::Dataset | View::Pipeline | View::Training => {
            let block = Block::bordered()
                .title(Span::styled(
                    format!(" {} ", app.view.title()),
                    app.theme.title,
                ))
                .border_style(app.theme.dim);
            frame.render_widget(block, body);
        },
    }
    status::render(frame, footer, app);
    if app.overlay == Some(Overlay::Help) {
        help::render(frame, area, app);
    }
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let titles = View::ALL
        .iter()
        .map(|view| format!("{} {}", view.index() + 1, view.title()));
    let tabs = Tabs::new(titles)
        .select(app.view.index())
        .highlight_style(app.theme.selected)
        .divider(" ");
    frame.render_widget(tabs, area);
    let name = Line::from(Span::styled(
        format!("{} ", app.project.name),
        app.theme.title,
    ))
    .right_aligned();
    frame.render_widget(Paragraph::new(name), area);
}

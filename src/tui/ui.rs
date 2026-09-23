//! Draws the whole frame from the app's state: header, view, status line and
//! overlays. Reads nothing but the app.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Tabs};

use super::app::{App, Overlay, View};
use super::views;
use super::widgets::{dialog, help, menu, status, too_small};

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
        View::Dataset => views::dataset::render(frame, body, app),
        View::Logs => views::logs::render(frame, body, app),
        View::Pipeline => views::pipeline::render(frame, body, app),
        View::Training => views::training::render(frame, body, app),
    }
    status::render(frame, footer, app);
    match &app.overlay {
        Some(Overlay::Help) => help::render(frame, area, app),
        Some(Overlay::Confirm(confirm)) => dialog::render(frame, area, confirm, &app.theme),
        Some(Overlay::Menu(selected)) => menu::render(frame, area, *selected, &app.theme),
        None => {},
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

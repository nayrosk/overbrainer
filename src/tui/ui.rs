//! Draws the whole frame from the app's state: the painted background, the
//! header, the view, the footer and the overlays, then the color effects of
//! motion over them. Reads nothing but the app: never the clock.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Tabs};

use super::app::{App, Overlay, View};
use super::motion::Areas;
use super::views;
use super::widgets::{dialog, help, menu, status, too_small};

/// Columns of the header's brand, before the tabs.
const BRAND_WIDTH: u16 = 17;

/// Draws `app` on `frame`.
pub(super) fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    app.observe_motion();
    frame.buffer_mut().set_style(area, app.theme.base);
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
    let pulse = match app.view {
        View::Dataset => {
            views::dataset::render(frame, body, app);
            None
        },
        View::Logs => {
            views::logs::render(frame, body, app);
            None
        },
        View::Pipeline => {
            views::pipeline::render(frame, body, app);
            None
        },
        View::Training => views::training::render(frame, body, app),
    };
    let toast = status::render(frame, footer, app);
    let overlay = match &app.overlay {
        Some(Overlay::Help) => Some(help::render(frame, area, app)),
        Some(Overlay::Confirm(confirm)) => {
            let destructive = dialog::destructive(&confirm.action, app.pipeline_task.is_some());
            Some(dialog::render(
                frame,
                area,
                confirm,
                &app.theme,
                destructive,
            ))
        },
        Some(Overlay::Menu(selected)) => Some(menu::render(frame, area, *selected, &app.theme)),
        None => None,
    };
    let areas = Areas {
        body,
        overlay,
        toast,
        pulse: pulse.filter(|_| app.pulse_shown()),
    };
    app.motion.apply(frame.buffer_mut(), &areas);
}

/// The brand, the tabs and the project name.
fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let [brand, tabs_area] =
        Layout::horizontal([Constraint::Length(BRAND_WIDTH), Constraint::Fill(1)]).areas(area);
    frame.render_widget(
        Paragraph::new(Span::styled(" ⠿ overbrainer", theme.accent)),
        brand,
    );
    let titles = View::ALL
        .iter()
        .map(|view| format!("{} {}", view.index() + 1, view.title()));
    let tabs = Tabs::new(titles)
        .select(app.view.index())
        .style(theme.dim)
        .highlight_style(theme.tab)
        .divider("  ")
        .padding("", "");
    frame.render_widget(tabs, tabs_area);
    let name =
        Line::from(Span::styled(format!("{} ", app.project.name), theme.title)).right_aligned();
    frame.render_widget(Paragraph::new(name), area);
}

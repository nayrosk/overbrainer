//! Draws the whole frame from the app's state: the painted background, the
//! header, the view, the footer and the overlays, then the color effects of
//! motion over them. Reads nothing but the app: never the clock.

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Tabs};

use super::app::{App, Overlay, View};
use super::format::cut;
use super::motion::Areas;
use super::views;
use super::widgets::{dialog, help, menu, status, too_small};

/// Columns of the header's brand, before the tabs.
const BRAND_WIDTH: u16 = 17;
/// What parts two tabs.
const DIVIDER: &str = "  ";

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
    if app.overlay.is_some() {
        // The view under an overlay goes dim, its symbols kept: muted text in
        // color, the dim attribute in monochrome.
        frame.buffer_mut().set_style(body, app.theme.dim);
    }
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
    if let Some(popup) = overlay {
        clear_margin(frame.buffer_mut(), popup, body);
    }
    let areas = Areas {
        body,
        overlay,
        toast,
        pulse: pulse.filter(|_| app.pulse_shown()),
    };
    app.motion.apply(frame.buffer_mut(), &areas);
}

/// Blanks the cells of `body` in a one-cell margin around `popup`, so no
/// glyph of the view touches the overlay's border; their style is kept.
fn clear_margin(buffer: &mut Buffer, popup: Rect, body: Rect) {
    let around = Rect {
        x: popup.x.saturating_sub(1),
        y: popup.y.saturating_sub(1),
        width: popup.width.saturating_add(2),
        height: popup.height.saturating_add(2),
    }
    .intersection(body);
    for position in around.positions() {
        if !popup.contains(position)
            && let Some(cell) = buffer.cell_mut(position)
        {
            cell.set_symbol(" ");
        }
    }
}

/// The brand, the tabs and the project name, each in its own columns. The
/// tabs come first: the name is cut to what is left, and the brand is dropped
/// when the tabs would not fit beside it.
fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let titles: Vec<String> = View::ALL
        .iter()
        .map(|view| format!("{} {}", view.index() + 1, view.title()))
        .collect();
    let tabs_width = titles
        .iter()
        .map(|title| title.chars().count())
        .sum::<usize>()
        + DIVIDER.chars().count() * titles.len().saturating_sub(1);
    let tabs_width = u16::try_from(tabs_width).unwrap_or(u16::MAX);
    let brand_width = if area.width >= BRAND_WIDTH.saturating_add(tabs_width) {
        BRAND_WIDTH
    } else {
        0
    };
    let [brand, tabs_area, name_area] = Layout::horizontal([
        Constraint::Length(brand_width),
        Constraint::Length(tabs_width),
        Constraint::Fill(1),
    ])
    .areas(area);
    frame.render_widget(
        Paragraph::new(Span::styled(" ⠿ overbrainer", theme.accent)),
        brand,
    );
    let tabs = Tabs::new(titles)
        .select(app.view.index())
        .style(theme.dim)
        .highlight_style(theme.tab)
        .divider(DIVIDER)
        .padding("", "");
    frame.render_widget(tabs, tabs_area);
    // Two columns free after the tabs, as between two tabs, and one after the
    // name.
    let room = usize::from(name_area.width.saturating_sub(3));
    // A lone `…` says nothing: no name then.
    let name = cut(&app.project.name, room);
    if room > 1 {
        let name = Line::from(Span::styled(format!("{name} "), theme.title)).right_aligned();
        frame.render_widget(Paragraph::new(name), name_area);
    }
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use crossterm::event::KeyCode;
    use ratatui::style::Modifier;

    use super::*;
    use crate::tui::snapshots::{app, draw, key, text, training_app};
    use crate::tui::theme::Theme;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const TABS: &str = "1 Dataset  2 Pipeline  3 Training  4 Logs";

    /// The header of `app` alone, drawn `width` columns wide.
    fn header(app: &App, width: u16) -> Result<String, Box<dyn std::error::Error>> {
        let mut terminal = Terminal::new(TestBackend::new(width, 1))?;
        terminal.draw(|frame| render_header(frame, frame.area(), app))?;
        Ok(text(&terminal).concat())
    }

    #[test]
    fn a_long_project_name_is_cut_and_never_covers_the_tabs() -> TestResult {
        let mut app = app();
        app.project.name = "a_project_name_forty_characters_long_xyz".into();
        assert_eq!(app.project.name.chars().count(), 40);
        let rows = text(&draw(&mut app, 80, 24)?);
        let row = rows.first().ok_or("no header")?;
        assert_eq!(
            row,
            &format!(" ⠿ overbrainer   {TABS}  a_project_name_for… ")
        );
        Ok(())
    }

    #[test]
    fn the_brand_goes_before_the_tabs_when_the_width_runs_out() -> TestResult {
        let app = app();
        let row = header(&app, 45)?;
        assert!(row.starts_with(TABS), "{row}");
        assert!(!row.contains("overbrainer"), "{row}");
        assert_eq!(row, format!("{TABS}    "), "no lone ellipsis");
        assert_eq!(header(&app, 47)?, format!("{TABS}  ru… "));
        assert_eq!(header(&app, 41)?, TABS);
        let row = header(&app, 38)?;
        let kept: String = TABS.chars().take(38).collect();
        assert_eq!(row, kept, "the tabs keep the columns there are");
        Ok(())
    }

    /// Where the dialog titled `title` is on `rows`: its left, top, right
    /// and bottom columns and rows, borders included.
    fn dialog(rows: &[String], title: &str) -> Option<(usize, usize, usize, usize)> {
        let top = rows.iter().position(|row| row.contains(title))?;
        let chars: Vec<char> = rows.get(top)?.chars().collect();
        let left = chars.iter().position(|c| *c == '╭')?;
        let right = chars.iter().position(|c| *c == '╮')?;
        let bottom = (top..rows.len()).find(|y| {
            rows.get(*y)
                .and_then(|row| row.chars().nth(left))
                .is_some_and(|c| c == '╰')
        })?;
        Some((left, top, right, bottom))
    }

    #[test]
    fn no_glyph_of_the_view_touches_an_overlay() -> TestResult {
        let mut app = training_app()?;
        app.on_input(&key(KeyCode::Char('c')));
        let rows = text(&draw(&mut app, 80, 24)?);
        let (left, top, right, bottom) = dialog(&rows, "╭ Cancel a run? ").ok_or("no dialog")?;
        let (before, above) = (left.saturating_sub(1), top.saturating_sub(1));
        let at = |x: usize, y: usize| rows.get(y).and_then(|row| row.chars().nth(x));
        for y in above..=bottom + 1 {
            assert_eq!(at(before, y), Some(' '), "left of row {y}: {rows:#?}");
            assert_eq!(at(right + 1, y), Some(' '), "right of row {y}: {rows:#?}");
        }
        for x in before..=right + 1 {
            assert_eq!(at(x, above), Some(' '), "above column {x}: {rows:#?}");
            assert_eq!(at(x, bottom + 1), Some(' '), "under column {x}: {rows:#?}");
        }
        let past = before.saturating_sub(1);
        assert_ne!(at(past, top), Some(' '), "the view shows past the margin");
        Ok(())
    }

    #[test]
    fn the_view_under_an_overlay_goes_dim_with_its_symbols() -> TestResult {
        let mut app = training_app()?;
        let before = text(&draw(&mut app, 80, 24)?);
        app.on_input(&key(KeyCode::Char('c')));
        let terminal = draw(&mut app, 80, 24)?;
        let buffer = terminal.backend().buffer();
        let run = buffer.cell((2, 3)).ok_or("no cell")?;
        assert_eq!(Some(run.fg), app.theme.dim.fg, "muted in color");
        let rows = text(&terminal);
        assert_eq!(rows.get(3), before.get(3), "symbols kept");
        let footer = buffer.cell((1, 23)).ok_or("no cell")?;
        assert_ne!(Some(footer.fg), app.theme.dim.fg, "the footer stays lit");
        app.theme = Theme::mono();
        let terminal = draw(&mut app, 80, 24)?;
        let run = terminal.backend().buffer().cell((2, 3)).ok_or("no cell")?;
        assert!(run.modifier.contains(Modifier::DIM), "dim in monochrome");
        Ok(())
    }
}

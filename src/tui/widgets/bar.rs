//! Progress bars in eighth blocks: full blocks, then one partial block, so a
//! bar reads without color.

/// The partial blocks, from one eighth to seven eighths of a cell.
const EIGHTHS: [char; 7] = ['▏', '▎', '▍', '▌', '▋', '▊', '▉'];

/// `ratio` of `width` cells, to the nearest eighth of a cell, padded with
/// spaces to `width`. `ratio` is clamped between 0 and 1.
pub(in crate::tui) fn bar(ratio: f64, width: u16) -> String {
    let eighths = u32::from(width) * 8;
    let filled = round_to(ratio.clamp(0.0, 1.0) * f64::from(eighths), eighths);
    let full = usize::try_from(filled / 8).unwrap_or(0);
    let mut text = "█".repeat(full);
    let partial = usize::try_from(filled % 8).unwrap_or(0);
    if let Some(block) = partial.checked_sub(1).and_then(|index| EIGHTHS.get(index)) {
        text.push(*block);
    }
    let used = usize::try_from(filled.div_ceil(8)).unwrap_or(0);
    text.push_str(&" ".repeat(usize::from(width).saturating_sub(used)));
    text
}

/// `value` rounded to the nearest whole number, between 0 and `max`, without
/// a float cast.
pub(in crate::tui) fn round_to(value: f64, max: u32) -> u32 {
    let target = value.round();
    let (mut low, mut high) = (0_u32, max);
    while low < high {
        let middle = low + (high - low) / 2;
        if f64::from(middle) < target {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bar_fills_to_the_nearest_eighth_of_a_cell() {
        assert_eq!(bar(0.0, 4), "    ");
        assert_eq!(bar(1.0, 4), "████");
        assert_eq!(bar(0.3, 10), "███       ");
        assert_eq!(bar(0.5125, 10), "█████▏    ");
        assert_eq!(bar(0.99, 10), "█████████▉");
        assert_eq!(bar(2.0, 3), "███", "clamped");
        assert_eq!(bar(-1.0, 3), "   ", "clamped");
    }

    #[test]
    fn rounding_stays_between_zero_and_max() {
        assert_eq!(round_to(2.5, 10), 3);
        assert_eq!(round_to(2.4, 10), 2);
        assert_eq!(round_to(99.0, 10), 10);
        assert_eq!(round_to(-3.0, 10), 0);
    }
}

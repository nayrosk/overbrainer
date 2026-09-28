//! Views of the Runpod catalog: GPU types filtered and ordered cheapest first,
//! data centers with the stock of chosen GPU types, and the resolution of a
//! target's `auto` choices. Pure functions over what
//! [`RunpodClient`](super::RunpodClient) lists.

use std::cmp::Ordering;

use crate::config::ListOrAuto;

use super::RunpodTarget;
use super::types::{Availability, DataCenter, GpuType, NetworkVolume, Stock, Template};

/// What [`select_gpus`] keeps. The default keeps every Secure Cloud GPU type.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuFilter {
    /// Least VRAM per GPU, in GB.
    pub min_vram_gb: Option<u32>,
    /// Highest Secure Cloud price of one GPU, in USD per hour; a GPU without a
    /// price is dropped when set.
    pub max_price: Option<f64>,
    /// Only GPU types offered in this data center; with `in_stock`, stock is
    /// judged there instead of overall.
    pub data_center: Option<String>,
    /// Only GPU types with some stock.
    pub in_stock: bool,
    /// Only GPU types whose Secure Cloud pod maximum covers this many GPUs.
    pub gpu_count: Option<u32>,
}

impl GpuFilter {
    /// Whether `gpu` passes every constraint.
    #[must_use]
    pub fn keeps(&self, gpu: &GpuType) -> bool {
        gpu.on_secure_cloud()
            && self.min_vram_gb.is_none_or(|min| gpu.memory >= min)
            && self
                .max_price
                .is_none_or(|max| gpu.secure_price().is_some_and(|price| price <= max))
            && self
                .gpu_count
                .is_none_or(|count| gpu.max_count.secure >= count)
            && self.stock_ok(gpu)
    }

    fn stock_ok(&self, gpu: &GpuType) -> bool {
        let stock = match &self.data_center {
            Some(center) => {
                if !gpu.data_centers.iter().any(|entry| &entry.id == center) {
                    return false;
                }
                gpu.stock_in(center)
            },
            None => gpu.availability,
        };
        !self.in_stock || stock.is_in_stock()
    }
}

/// Cheapest Secure Cloud price first (no price last), ties by more VRAM, then
/// by ID.
#[must_use]
pub fn by_price(a: &GpuType, b: &GpuType) -> Ordering {
    let price = match (a.secure_price(), b.secure_price()) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    price
        .then_with(|| b.memory.cmp(&a.memory))
        .then_with(|| a.id.cmp(&b.id))
}

/// The GPU types `filter` keeps, cheapest first (see [`by_price`]).
#[must_use]
pub fn select_gpus(gpus: &[GpuType], filter: &GpuFilter) -> Vec<GpuType> {
    let mut kept: Vec<GpuType> = gpus
        .iter()
        .filter(|gpu| filter.keeps(gpu))
        .cloned()
        .collect();
    kept.sort_by(by_price);
    kept
}

/// The rows of `overbrainer pod gpus`, header first, for `gpus` already
/// filtered and sorted (see [`select_gpus`]); nothing when `gpus` is empty.
/// `data_center`, when given, is shown as the stock column instead of the
/// overall band (see [`GpuType::stock_in`]).
#[must_use]
pub fn gpu_table(gpus: &[GpuType], data_center: Option<&str>) -> Vec<String> {
    if gpus.is_empty() {
        return Vec::new();
    }
    let rows: Vec<Vec<String>> = gpus
        .iter()
        .map(|gpu| {
            let price = gpu
                .secure_price()
                .map_or_else(|| "-".to_string(), |price| format!("{price:.2}"));
            let stock = data_center.map_or(gpu.availability, |center| gpu.stock_in(center));
            vec![
                gpu.id.clone(),
                gpu.memory.to_string(),
                price,
                gpu.max_count.secure.to_string(),
                stock.name().to_string(),
            ]
        })
        .collect();
    columns(
        &["ID", "VRAM GB", "$/H", "MAX COUNT", "STOCK"],
        &[
            Align::Left,
            Align::Right,
            Align::Right,
            Align::Right,
            Align::Left,
        ],
        &rows,
    )
}

/// `text`, from the Runpod API, without what could move the cursor or change
/// the terminal's state: ANSI escape sequences (CSI, string controls such as
/// OSC or DCS through their terminator, and two-character ones) are dropped, a control character that is whitespace (newline, tab,
/// CR...) becomes a space, and every other one (C0, DEL, C1) is dropped.
#[must_use]
pub fn printable(text: &str) -> String {
    let mut kept = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => match chars.next() {
                Some('[') => skip_csi(&mut chars),
                Some(']' | 'P' | 'X' | '^' | '_') => skip_string(&mut chars),
                _ => {},
            },
            '\u{9b}' => skip_csi(&mut chars),
            '\u{90}' | '\u{98}' | '\u{9d}' | '\u{9e}' | '\u{9f}' => skip_string(&mut chars),
            c if c.is_control() && c.is_whitespace() => kept.push(' '),
            c if c.is_control() => {},
            c => kept.push(c),
        }
    }
    kept
}

/// Skips a CSI sequence's parameters and intermediates, then its final byte.
fn skip_csi(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while chars.next_if(|c| matches!(c, ' '..='?')).is_some() {}
    chars.next_if(|c| matches!(c, '@'..='~'));
}

/// Skips a string control (OSC, DCS, SOS, PM or APC) up to its terminator:
/// BEL, ST or `ESC \`; an unterminated one takes the rest of the text.
fn skip_string(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while let Some(c) = chars.next() {
        match c {
            '\u{7}' | '\u{9c}' => return,
            '\u{1b}' => {
                chars.next_if_eq(&'\\');
                return;
            },
            _ => {},
        }
    }
}

/// Whether a [`columns`] column is aligned to the left or the right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Align {
    Left,
    Right,
}

/// `header` and `rows` laid out in columns two spaces apart, each as wide as
/// its longest value, `align` giving each column's side; the last column is
/// never padded. Each cell is [`printable`].
fn columns(header: &[&str], align: &[Align], rows: &[Vec<String>]) -> Vec<String> {
    let rows: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.iter().map(|cell| printable(cell)).collect())
        .collect();
    let mut widths: Vec<usize> = header.iter().map(|title| title.chars().count()).collect();
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.chars().count());
        }
    }
    let line = |cells: Vec<&str>| {
        let last = cells.len().saturating_sub(1);
        let padded: Vec<String> = cells
            .iter()
            .zip(&widths)
            .zip(align)
            .enumerate()
            .map(|(index, ((cell, width), side))| match side {
                _ if index == last => (*cell).to_string(),
                Align::Left => format!("{cell:<width$}"),
                Align::Right => format!("{cell:>width$}"),
            })
            .collect();
        padded.join("  ")
    };
    let mut lines = vec![line(header.to_vec())];
    lines.extend(
        rows.iter()
            .map(|row| line(row.iter().map(String::as_str).collect())),
    );
    lines
}

/// A data center with the stock of chosen GPU types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataCenterStock {
    /// Its ID.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Continental region.
    pub region: String,
    /// One entry per chosen GPU type, in the chosen order; `NONE` when the data
    /// center does not offer it.
    pub stock: Vec<Stock>,
}

impl DataCenterStock {
    /// Whether at least one chosen GPU type is in stock here.
    #[must_use]
    pub fn in_stock(&self) -> bool {
        self.stock
            .iter()
            .any(|entry| entry.availability.is_in_stock())
    }
}

/// Every data center with the stock of each `chosen` GPU type, ordered by ID.
#[must_use]
pub fn data_center_stock<S: AsRef<str>>(
    data_centers: &[DataCenter],
    chosen: &[S],
) -> Vec<DataCenterStock> {
    let mut rows: Vec<DataCenterStock> = data_centers
        .iter()
        .map(|center| DataCenterStock {
            id: center.id.clone(),
            name: center.name.clone(),
            region: center.region.clone(),
            stock: chosen
                .iter()
                .map(|gpu| {
                    let gpu = gpu.as_ref();
                    center
                        .gpu_availability
                        .iter()
                        .find(|entry| entry.id == gpu)
                        .cloned()
                        .unwrap_or_else(|| Stock {
                            id: gpu.to_string(),
                            name: String::new(),
                            availability: Availability::None,
                        })
                })
                .collect(),
        })
        .collect();
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    rows
}

/// The rows of `overbrainer pod datacenters`, header first, sorted by ID;
/// nothing when `data_centers` is empty. The stock column counts the GPU
/// types `data_centers` reports in stock there, not a chosen set.
#[must_use]
pub fn data_center_table(data_centers: &[DataCenter]) -> Vec<String> {
    if data_centers.is_empty() {
        return Vec::new();
    }
    let mut rows: Vec<&DataCenter> = data_centers.iter().collect();
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    let rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|center| {
            let in_stock = center
                .gpu_availability
                .iter()
                .filter(|entry| entry.availability.is_in_stock())
                .count();
            vec![
                center.id.clone(),
                center.name.clone(),
                center.region.clone(),
                in_stock.to_string(),
            ]
        })
        .collect();
    columns(
        &["ID", "NAME", "REGION", "GPU TYPES IN STOCK"],
        &[Align::Left; 4],
        &rows,
    )
}

/// The data centers with at least one `chosen` GPU type in stock, ordered by
/// the cheapest such GPU type they have (see [`by_price`]), ties by ID.
///
/// Built only from each chosen GPU's own `data_centers` (as
/// [`RunpodClient::list_gpu_types`](super::RunpodClient::list_gpu_types)
/// scopes it to a GPU count), never from the unscoped `catalog/datacenters`
/// stock: a data center is picked only when the GPU itself reports stock
/// there for that count.
#[must_use]
pub fn stocked_data_centers<S: AsRef<str>>(gpus: &[GpuType], chosen: &[S]) -> Vec<DataCenterStock> {
    let chosen_gpus: Vec<&GpuType> = chosen
        .iter()
        .filter_map(|id| gpus.iter().find(|gpu| gpu.id == id.as_ref()))
        .collect();
    let mut ids: Vec<(String, String)> = Vec::new();
    for gpu in &chosen_gpus {
        for entry in &gpu.data_centers {
            if entry.availability.is_in_stock() && !ids.iter().any(|(id, _)| *id == entry.id) {
                ids.push((entry.id.clone(), entry.name.clone()));
            }
        }
    }
    let cheapest = |row: &DataCenterStock| -> Option<&GpuType> {
        row.stock
            .iter()
            .filter(|entry| entry.availability.is_in_stock())
            .filter_map(|entry| gpus.iter().find(|gpu| gpu.id == entry.id))
            .min_by(|a, b| by_price(a, b))
    };
    let mut rows: Vec<DataCenterStock> = ids
        .into_iter()
        .map(|(id, name)| DataCenterStock {
            stock: chosen
                .iter()
                .map(|gpu_id| gpu_stock_at(&chosen_gpus, gpu_id.as_ref(), &id))
                .collect(),
            id,
            name,
            region: String::new(),
        })
        .collect();
    rows.sort_by(|a, b| {
        let order = match (cheapest(a), cheapest(b)) {
            (Some(x), Some(y)) => by_price(x, y),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        };
        order.then_with(|| a.id.cmp(&b.id))
    });
    rows
}

/// `gpu_id`'s own stock at the data center `center_id`, `NONE` when that GPU
/// (among `chosen_gpus`) does not report it. `id` names the GPU, not the data
/// center, to match [`DataCenterStock::stock`]'s "one entry per chosen GPU".
fn gpu_stock_at(chosen_gpus: &[&GpuType], gpu_id: &str, center_id: &str) -> Stock {
    let availability = chosen_gpus
        .iter()
        .find(|gpu| gpu.id == gpu_id)
        .and_then(|gpu| gpu.data_centers.iter().find(|entry| entry.id == center_id))
        .map_or(Availability::None, |entry| entry.availability);
    Stock {
        id: gpu_id.to_string(),
        name: String::new(),
        availability,
    }
}

/// `target` with its `auto` choices replaced by lists, from the catalog's
/// `gpus`, listed for the target's `gpu_count`; listed choices are kept as
/// they are.
///
/// `auto` GPU types are those in stock whose Secure Cloud pod maximum covers
/// `gpu_count`, within `min_vram_gb` and `max_price_per_hour`, in stock in one
/// of the listed data centers when there are any, cheapest first (see
/// [`by_price`]). `auto` data centers are those with a chosen GPU type in
/// stock for that count, ordered by the cheapest one they have (see
/// [`stocked_data_centers`]).
///
/// # Errors
///
/// Returns a message saying what was asked when nothing in stock matches.
pub fn resolve(target: &RunpodTarget, gpus: &[GpuType]) -> Result<RunpodTarget, String> {
    let mut resolved = target.clone();
    if target.gpu_types.is_auto() {
        let filter = GpuFilter {
            min_vram_gb: target.min_vram_gb,
            max_price: target.max_price_per_hour,
            data_center: None,
            in_stock: true,
            gpu_count: Some(target.gpu_count),
        };
        let listed = target.data_center_ids.list();
        let chosen: Vec<String> = select_gpus(gpus, &filter)
            .into_iter()
            .filter(|gpu| {
                listed.is_empty()
                    || listed
                        .iter()
                        .any(|center| gpu.stock_in(center).is_in_stock())
            })
            .map(|gpu| gpu.id)
            .collect();
        if chosen.is_empty() {
            return Err(format!(
                "no GPU type in stock on Runpod's Secure Cloud for gpu_types = \"auto\" ({})",
                asked(target)
            ));
        }
        resolved.gpu_types = ListOrAuto::List(chosen);
    }
    if target.data_center_ids.is_auto() {
        let chosen = resolved.gpu_types.list();
        let centers: Vec<String> = stocked_data_centers(gpus, chosen)
            .into_iter()
            .map(|row| row.id)
            .collect();
        if centers.is_empty() {
            return Err(format!(
                "no data center has {} in stock for data_center_ids = \"auto\" (gpu_count = {})",
                or_list(chosen),
                target.gpu_count
            ));
        }
        resolved.data_center_ids = ListOrAuto::List(centers);
    }
    Ok(resolved)
}

/// What an `auto` choice of GPU types asked for, as the target's fields.
fn asked(target: &RunpodTarget) -> String {
    let mut asked = vec![format!("gpu_count = {}", target.gpu_count)];
    if let Some(gb) = target.min_vram_gb {
        asked.push(format!("min_vram_gb = {gb}"));
    }
    if let Some(price) = target.max_price_per_hour {
        asked.push(format!("max_price_per_hour = {price}"));
    }
    if let ListOrAuto::List(centers) = &target.data_center_ids
        && !centers.is_empty()
    {
        asked.push(format!("data_center_ids = {}", centers.join(", ")));
    }
    asked.join(", ")
}

/// `a`, `a or b`, `a, b or c`.
fn or_list(items: &[String]) -> String {
    match items.split_last() {
        None => String::new(),
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} or {last}", rest.join(", ")),
    }
}

/// The rows of `overbrainer pod volumes`, header first, sorted by name;
/// nothing when `volumes` is empty.
#[must_use]
pub fn volume_table(volumes: &[NetworkVolume]) -> Vec<String> {
    if volumes.is_empty() {
        return Vec::new();
    }
    let mut rows: Vec<&NetworkVolume> = volumes.iter().collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    let rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|volume| {
            vec![
                volume.id.clone(),
                volume.name.clone(),
                volume.size.to_string(),
                volume.data_center.clone(),
            ]
        })
        .collect();
    columns(
        &["ID", "NAME", "SIZE GB", "DATA CENTER"],
        &[Align::Left, Align::Left, Align::Right, Align::Left],
        &rows,
    )
}

/// The rows of `overbrainer pod templates`, header first, sorted by name;
/// nothing when `templates` is empty.
#[must_use]
pub fn template_table(templates: &[Template]) -> Vec<String> {
    if templates.is_empty() {
        return Vec::new();
    }
    let mut rows: Vec<&Template> = templates.iter().collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    let rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|template| {
            vec![
                template.id.clone(),
                template.name.clone(),
                template.image.clone(),
            ]
        })
        .collect();
    columns(&["ID", "NAME", "IMAGE"], &[Align::Left; 3], &rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ListOrAuto;
    use crate::runpod::types::{GpuMaxCount, GpuPrice};

    fn stock(id: &str, availability: Availability) -> Stock {
        Stock {
            id: id.to_string(),
            name: String::new(),
            availability,
        }
    }

    fn gpu(id: &str, memory: u32, price: Option<f64>) -> GpuType {
        GpuType {
            id: id.to_string(),
            memory,
            price: GpuPrice {
                secure: price,
                community: None,
            },
            max_count: GpuMaxCount {
                secure: 8,
                community: 0,
            },
            availability: Availability::High,
            ..GpuType::default()
        }
    }

    fn target(gpu_types: ListOrAuto, data_center_ids: ListOrAuto) -> RunpodTarget {
        RunpodTarget {
            gpu_types,
            min_vram_gb: None,
            max_price_per_hour: None,
            gpu_count: 1,
            image: "img".to_string(),
            venv: "/venv".to_string(),
            container_disk_gb: 50,
            max_hours: 1.0,
            boot_grace: std::time::Duration::from_secs(1800),
            retrieve_grace: std::time::Duration::from_secs(3600),
            data_center_ids,
            network_volume_id: None,
        }
    }

    fn list(items: &[&str]) -> ListOrAuto {
        ListOrAuto::List(items.iter().map(|item| (*item).to_string()).collect())
    }

    fn ids(gpus: &[GpuType]) -> Vec<&str> {
        gpus.iter().map(|gpu| gpu.id.as_str()).collect()
    }

    fn center(id: &str, gpus: Vec<Stock>) -> DataCenter {
        DataCenter {
            id: id.to_string(),
            name: format!("{id} name"),
            region: "EUROPE".to_string(),
            gpu_availability: gpus,
        }
    }

    #[test]
    fn gpus_are_cheapest_first_ties_by_more_vram_then_id() {
        let gpus = [
            gpu("none", 80, None),
            gpu("b", 24, Some(0.5)),
            gpu("big", 48, Some(0.5)),
            gpu("a", 24, Some(0.5)),
            gpu("cheap", 16, Some(0.2)),
        ];
        let sorted = select_gpus(&gpus, &GpuFilter::default());
        assert_eq!(ids(&sorted), vec!["cheap", "big", "a", "b", "none"]);
    }

    #[test]
    fn a_zero_price_is_unknown_neither_first_nor_under_a_limit() {
        let gpus = [gpu("free", 24, Some(0.0)), gpu("paid", 24, Some(0.5))];
        let sorted = select_gpus(&gpus, &GpuFilter::default());
        assert_eq!(ids(&sorted), vec!["paid", "free"]);
        let capped = GpuFilter {
            max_price: Some(1.0),
            ..GpuFilter::default()
        };
        assert_eq!(ids(&select_gpus(&gpus, &capped)), vec!["paid"]);
    }

    #[test]
    fn vram_and_price_limits_filter() {
        let gpus = [
            gpu("small", 16, Some(0.2)),
            gpu("mid", 48, Some(0.8)),
            gpu("dear", 80, Some(2.5)),
            gpu("unpriced", 80, None),
        ];
        let filter = GpuFilter {
            min_vram_gb: Some(40),
            max_price: Some(1.0),
            ..GpuFilter::default()
        };
        assert_eq!(ids(&select_gpus(&gpus, &filter)), vec!["mid"]);
        let vram_only = GpuFilter {
            min_vram_gb: Some(48),
            ..GpuFilter::default()
        };
        assert_eq!(
            ids(&select_gpus(&gpus, &vram_only)),
            vec!["mid", "dear", "unpriced"]
        );
    }

    #[test]
    fn stock_and_gpu_count_filter() {
        let mut out = gpu("out", 24, Some(0.1));
        out.availability = Availability::None;
        let mut unknown = gpu("unknown", 24, Some(0.2));
        unknown.availability = Availability::Unknown;
        let mut single = gpu("single", 24, Some(0.3));
        single.max_count.secure = 1;
        let gpus = [out, unknown, single, gpu("ok", 24, Some(0.4))];
        let in_stock = GpuFilter {
            in_stock: true,
            ..GpuFilter::default()
        };
        assert_eq!(ids(&select_gpus(&gpus, &in_stock)), vec!["single", "ok"]);
        let two = GpuFilter {
            in_stock: true,
            gpu_count: Some(2),
            ..GpuFilter::default()
        };
        assert_eq!(ids(&select_gpus(&gpus, &two)), vec!["ok"]);
    }

    #[test]
    fn community_only_gpus_are_dropped() {
        let mut community = gpu("community", 24, Some(0.1));
        community.secure = Some(false);
        let gpus = [community, gpu("secure", 24, Some(0.2))];
        assert_eq!(
            ids(&select_gpus(&gpus, &GpuFilter::default())),
            vec!["secure"]
        );
    }

    #[test]
    fn a_data_center_filter_judges_stock_there() {
        let mut here = gpu("here", 24, Some(0.3));
        here.data_centers = vec![stock("EU-RO-1", Availability::Low)];
        let mut empty_here = gpu("empty-here", 24, Some(0.2));
        empty_here.data_centers = vec![stock("EU-RO-1", Availability::None)];
        let mut elsewhere = gpu("elsewhere", 24, Some(0.1));
        elsewhere.data_centers = vec![stock("US-KS-2", Availability::High)];
        let gpus = [here, empty_here, elsewhere];
        let offered = GpuFilter {
            data_center: Some("EU-RO-1".to_string()),
            ..GpuFilter::default()
        };
        assert_eq!(
            ids(&select_gpus(&gpus, &offered)),
            vec!["empty-here", "here"]
        );
        let stocked = GpuFilter {
            in_stock: true,
            ..offered
        };
        assert_eq!(ids(&select_gpus(&gpus, &stocked)), vec!["here"]);
    }

    #[test]
    fn data_centers_show_the_chosen_gpus_stock_in_order() {
        let centers = [
            center("US-KS-2", vec![stock("A", Availability::High)]),
            center(
                "EU-RO-1",
                vec![
                    stock("B", Availability::Medium),
                    stock("A", Availability::None),
                ],
            ),
        ];
        let rows = data_center_stock(&centers, &["B", "A", "C"]);
        let row_ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
        assert_eq!(row_ids, vec!["EU-RO-1", "US-KS-2"]);
        let bands: Vec<Availability> = rows[0].stock.iter().map(|s| s.availability).collect();
        assert_eq!(
            bands,
            vec![Availability::Medium, Availability::None, Availability::None]
        );
        assert_eq!(rows[1].stock[0].id, "B");
        assert!(rows[0].in_stock() && rows[1].in_stock());
        assert!(!data_center_stock(&centers, &["C"])[0].in_stock());
        assert_eq!(rows[0].name, "EU-RO-1 name");
        assert_eq!(rows[0].region, "EUROPE");
    }

    #[test]
    fn stocked_data_centers_are_ordered_by_their_cheapest_chosen_gpu() {
        let mut cheap = gpu("cheap", 24, Some(0.2));
        cheap.data_centers = vec![
            stock("B-EMPTY", Availability::None),
            stock("C-CHEAP", Availability::Low),
            stock("D-CHEAP", Availability::High),
        ];
        let mut dear = gpu("dear", 80, Some(1.5));
        dear.data_centers = vec![
            stock("A-DEAR", Availability::High),
            stock("B-EMPTY", Availability::None),
            stock("C-CHEAP", Availability::High),
        ];
        // "E-OTHER" is not in either chosen GPU's own list, unlike an unscoped
        // catalog/datacenters listing might show: it must never appear.
        let gpus = [cheap, dear];
        let rows = stocked_data_centers(&gpus, &["dear", "cheap"]);
        let row_ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
        assert_eq!(row_ids, vec!["C-CHEAP", "D-CHEAP", "A-DEAR"]);
    }

    #[test]
    fn listed_choices_are_kept_as_they_are() {
        let listed = target(list(&["B", "A"]), list(&["EU-RO-1"]));
        assert_eq!(resolve(&listed, &[]), Ok(listed.clone()));
        let any = target(list(&["A"]), ListOrAuto::default());
        assert_eq!(resolve(&any, &[]), Ok(any.clone()));
    }

    #[test]
    fn auto_gpus_are_those_in_stock_cheapest_first_ties_by_more_vram() {
        let mut out = gpu("out", 80, Some(0.1));
        out.availability = Availability::None;
        let mut single = gpu("single", 80, Some(0.2));
        single.max_count.secure = 1;
        let mut community = gpu("community", 80, Some(0.2));
        community.secure = Some(false);
        let gpus = [
            out,
            single,
            community,
            gpu("dear", 80, Some(1.0)),
            gpu("a", 24, Some(0.5)),
            gpu("big", 48, Some(0.5)),
        ];
        let mut auto = target(ListOrAuto::Auto, ListOrAuto::default());
        auto.gpu_count = 2;
        let resolved = resolve(&auto, &gpus);
        assert_eq!(
            resolved.map(|target| target.gpu_types),
            Ok(list(&["big", "a", "dear"]))
        );
    }

    #[test]
    fn auto_gpus_meet_the_vram_and_price_limits() {
        let gpus = [
            gpu("small", 16, Some(0.2)),
            gpu("mid", 48, Some(0.8)),
            gpu("dear", 80, Some(2.5)),
            gpu("unpriced", 80, None),
        ];
        let mut auto = target(ListOrAuto::Auto, ListOrAuto::default());
        auto.min_vram_gb = Some(40);
        assert_eq!(
            resolve(&auto, &gpus).map(|target| target.gpu_types),
            Ok(list(&["mid", "dear", "unpriced"]))
        );
        auto.max_price_per_hour = Some(1.0);
        assert_eq!(
            resolve(&auto, &gpus).map(|target| target.gpu_types),
            Ok(list(&["mid"]))
        );
    }

    #[test]
    fn auto_gpus_are_in_stock_in_a_listed_data_center() {
        let mut here = gpu("here", 24, Some(0.3));
        here.data_centers = vec![stock("EU-RO-1", Availability::Low)];
        let mut empty_here = gpu("empty-here", 24, Some(0.2));
        empty_here.data_centers = vec![stock("EU-RO-1", Availability::None)];
        let mut elsewhere = gpu("elsewhere", 24, Some(0.1));
        elsewhere.data_centers = vec![stock("US-KS-2", Availability::High)];
        let gpus = [here, empty_here, elsewhere];
        let auto = target(ListOrAuto::Auto, list(&["EU-RO-1"]));
        let resolved = resolve(&auto, &gpus);
        assert_eq!(
            resolved.clone().map(|target| target.gpu_types),
            Ok(list(&["here"]))
        );
        assert_eq!(
            resolved.map(|target| target.data_center_ids),
            Ok(list(&["EU-RO-1"]))
        );
    }

    #[test]
    fn auto_data_centers_have_a_chosen_gpu_in_stock_cheapest_first() {
        let mut cheap = gpu("cheap", 24, Some(0.2));
        cheap.data_centers = vec![
            stock("B-EMPTY", Availability::None),
            stock("C-CHEAP", Availability::Low),
        ];
        let mut dear = gpu("dear", 80, Some(1.5));
        dear.data_centers = vec![stock("A-DEAR", Availability::High)];
        let gpus = [cheap, dear];
        let listed = target(list(&["dear", "cheap"]), ListOrAuto::Auto);
        let resolved = resolve(&listed, &gpus);
        assert_eq!(
            resolved.clone().map(|target| target.data_center_ids),
            Ok(list(&["C-CHEAP", "A-DEAR"]))
        );
        assert_eq!(
            resolved.map(|target| target.gpu_types),
            Ok(list(&["dear", "cheap"]))
        );
        let both = target(ListOrAuto::Auto, ListOrAuto::Auto);
        let resolved = resolve(&both, &gpus);
        assert_eq!(
            resolved.clone().map(|target| target.gpu_types),
            Ok(list(&["cheap", "dear"]))
        );
        assert_eq!(
            resolved.map(|target| target.data_center_ids),
            Ok(list(&["C-CHEAP", "A-DEAR"]))
        );
    }

    #[test]
    fn nothing_in_stock_says_what_was_asked() {
        let mut out = gpu("out", 80, Some(0.1));
        out.availability = Availability::None;
        let gpus = [out, gpu("small", 16, Some(0.2))];
        let mut auto = target(ListOrAuto::Auto, list(&["EU-RO-1", "US-KS-2"]));
        auto.gpu_count = 2;
        auto.min_vram_gb = Some(48);
        auto.max_price_per_hour = Some(0.5);
        assert_eq!(
            resolve(&auto, &gpus),
            Err(
                "no GPU type in stock on Runpod's Secure Cloud for gpu_types = \"auto\" \
                 (gpu_count = 2, min_vram_gb = 48, max_price_per_hour = 0.5, \
                 data_center_ids = EU-RO-1, US-KS-2)"
                    .to_string()
            )
        );
        let plain = target(ListOrAuto::Auto, ListOrAuto::default());
        assert_eq!(
            resolve(&plain, &[]),
            Err(
                "no GPU type in stock on Runpod's Secure Cloud for gpu_types = \"auto\" \
                 (gpu_count = 1)"
                    .to_string()
            )
        );
        let mut a = gpu("A", 24, Some(0.3));
        a.data_centers = vec![stock("EU-RO-1", Availability::None)];
        let listed = target(list(&["A", "B"]), ListOrAuto::Auto);
        assert_eq!(
            resolve(&listed, &[a]),
            Err(
                "no data center has A or B in stock for data_center_ids = \"auto\" \
                 (gpu_count = 1)"
                    .to_string()
            )
        );
    }

    #[test]
    fn gpu_table_is_empty_without_rows() {
        assert!(gpu_table(&[], None).is_empty());
    }

    #[test]
    fn gpu_table_lists_price_vram_count_and_overall_stock() {
        let mut a40 = gpu("NVIDIA A40", 48, Some(0.4));
        a40.availability = Availability::High;
        let mut cheap = gpu("cheap", 16, None);
        cheap.availability = Availability::Low;
        let lines = gpu_table(&[a40, cheap], None);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(
            lines[0].starts_with("ID") && lines[0].ends_with("STOCK"),
            "{}",
            lines[0]
        );
        assert!(
            lines[1].starts_with("NVIDIA A40")
                && lines[1].contains("48")
                && lines[1].contains("0.40")
                && lines[1].ends_with("HIGH"),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].starts_with("cheap") && lines[2].ends_with("LOW"),
            "{}",
            lines[2]
        );
    }

    #[test]
    fn gpu_table_shows_stock_at_a_data_center_when_given() {
        let mut a40 = gpu("NVIDIA A40", 48, Some(0.4));
        a40.availability = Availability::High;
        a40.data_centers = vec![stock("EU-RO-1", Availability::Low)];
        let lines = gpu_table(&[a40], Some("EU-RO-1"));
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[1].ends_with("LOW"), "{}", lines[1]);
    }

    #[test]
    fn data_center_table_is_empty_without_rows() {
        assert!(data_center_table(&[]).is_empty());
    }

    #[test]
    fn data_center_table_counts_gpu_types_in_stock_sorted_by_id() {
        let centers = [
            center(
                "B-ID",
                vec![
                    stock("x", Availability::High),
                    stock("y", Availability::None),
                ],
            ),
            center("A-ID", vec![stock("x", Availability::Low)]),
        ];
        let lines = data_center_table(&centers);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(
            lines[0].starts_with("ID") && lines[0].ends_with("GPU TYPES IN STOCK"),
            "{}",
            lines[0]
        );
        assert!(
            lines[1].starts_with("A-ID") && lines[1].ends_with('1'),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].starts_with("B-ID") && lines[2].ends_with('1'),
            "{}",
            lines[2]
        );
    }

    #[test]
    fn volume_table_is_empty_without_rows() {
        assert!(volume_table(&[]).is_empty());
    }

    #[test]
    fn volume_table_is_sorted_by_name() {
        let volumes = [
            NetworkVolume {
                id: "v2".to_string(),
                name: "zeta".to_string(),
                size: 50,
                data_center: "EU-RO-1".to_string(),
            },
            NetworkVolume {
                id: "v1".to_string(),
                name: "alpha".to_string(),
                size: 100,
                data_center: "US-KS-2".to_string(),
            },
        ];
        let lines = volume_table(&volumes);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(
            lines[0].starts_with("ID") && lines[0].ends_with("DATA CENTER"),
            "{}",
            lines[0]
        );
        assert!(
            lines[1].contains("v1") && lines[1].contains("alpha") && lines[1].ends_with("US-KS-2"),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].contains("v2") && lines[2].contains("zeta") && lines[2].ends_with("EU-RO-1"),
            "{}",
            lines[2]
        );
    }

    #[test]
    fn template_table_is_empty_without_rows() {
        assert!(template_table(&[]).is_empty());
    }

    #[test]
    fn template_table_is_sorted_by_name() {
        let templates = [
            Template {
                id: "t2".to_string(),
                name: "zeta".to_string(),
                image: "img/z:1".to_string(),
                serverless: false,
            },
            Template {
                id: "t1".to_string(),
                name: "alpha".to_string(),
                image: "img/a:1".to_string(),
                serverless: false,
            },
        ];
        let lines = template_table(&templates);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(
            lines[0].starts_with("ID") && lines[0].ends_with("IMAGE"),
            "{}",
            lines[0]
        );
        assert!(
            lines[1].contains("t1") && lines[1].ends_with("img/a:1"),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].contains("t2") && lines[2].ends_with("img/z:1"),
            "{}",
            lines[2]
        );
    }

    /// A live GPU ID, far longer than the old fixed ID column of 28.
    const LONG: &str = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition";

    /// Where `word` starts in `line`, for checking that a row lines up with
    /// its header.
    fn at(line: &str, word: &str) -> Option<usize> {
        line.find(word)
    }

    #[test]
    fn gpu_table_widens_its_id_column_to_the_longest_id() {
        let lines = gpu_table(&[gpu(LONG, 32, Some(0.9)), gpu("a", 8, Some(0.1))], None);
        assert!(lines[1].starts_with(&format!("{LONG}  ")), "{}", lines[1]);
        for line in &lines[1..] {
            assert_eq!(at(line, "HIGH"), at(&lines[0], "STOCK"), "{lines:#?}");
        }
        assert_eq!(
            at(&lines[2], "0.10").map(|at| at + 4),
            at(&lines[0], "$/H").map(|at| at + 3),
            "{lines:#?}"
        );
    }

    #[test]
    fn data_center_table_widens_its_columns_to_the_longest_values() {
        let mut long = center("A", vec![stock("x", Availability::High)]);
        long.name = LONG.to_string();
        let lines = data_center_table(&[long, center("B", Vec::new())]);
        for line in &lines[1..] {
            assert_eq!(
                line.rfind("  ").map(|at| at + 2),
                at(&lines[0], "GPU TYPES"),
                "{lines:#?}"
            );
        }
        assert_eq!(
            at(&lines[2], "EUROPE"),
            at(&lines[0], "REGION"),
            "{lines:#?}"
        );
    }

    #[test]
    fn volume_and_template_tables_widen_their_name_columns() {
        let volume = |id: &str, name: &str| NetworkVolume {
            id: id.to_string(),
            name: name.to_string(),
            size: 50,
            data_center: "EU-RO-1".to_string(),
        };
        let lines = volume_table(&[volume("v1", LONG), volume("v2", "z")]);
        for line in &lines[1..] {
            assert_eq!(
                at(line, "EU-RO-1"),
                at(&lines[0], "DATA CENTER"),
                "{lines:#?}"
            );
        }
        let template = |id: &str, name: &str| Template {
            id: id.to_string(),
            name: name.to_string(),
            image: "img/a:1".to_string(),
            serverless: false,
        };
        let lines = template_table(&[template("t1", LONG), template("t2", "z")]);
        for line in &lines[1..] {
            assert_eq!(at(line, "img/"), at(&lines[0], "IMAGE"), "{lines:#?}");
        }
    }

    #[test]
    fn printable_strips_control_characters_and_escape_sequences() {
        assert_eq!(printable("A40\u{1b}[2J\nname"), "A40 name");
        assert_eq!(printable("a\nb\tc\r\nd  e"), "a b c  d  e", "no collapsing");
        assert_eq!(
            printable("a\u{1b}]0;title\u{7}b\u{1b}]8;;x\u{1b}\\c"),
            "abc"
        );
        assert_eq!(printable("a\u{1b}Mb\u{7f}c\u{9b}31md\te"), "abcd e");
        assert_eq!(
            printable("RTX 4090 é"),
            "RTX 4090 é",
            "printable text stays"
        );
    }

    #[test]
    fn string_controls_are_skipped_through_their_terminator() {
        for introducer in ['P', 'X', '^', '_'] {
            let st = format!("a\u{1b}{introducer}bad\u{1b}\\b");
            assert_eq!(printable(&st), "ab", "ESC {introducer}");
            let bel = format!("a\u{1b}{introducer}bad\u{7}b");
            assert_eq!(printable(&bel), "ab", "ESC {introducer} ended by BEL");
            let open = format!("a\u{1b}{introducer}bad and more");
            assert_eq!(printable(&open), "a", "ESC {introducer} unterminated");
        }
        for introducer in ['\u{90}', '\u{98}', '\u{9e}', '\u{9f}'] {
            let st = format!("a{introducer}bad\u{9c}b");
            assert_eq!(printable(&st), "ab", "{introducer:?}");
            let open = format!("a{introducer}bad and more");
            assert_eq!(printable(&open), "a", "{introducer:?} unterminated");
        }
    }

    #[test]
    fn every_table_prints_api_strings_without_control_characters() {
        let odd = "odd\u{1b}[2J\nname";
        let mut named = center("EU-RO-1", Vec::new());
        named.name = odd.to_string();
        let volume = NetworkVolume {
            id: "v1".to_string(),
            name: odd.to_string(),
            size: 50,
            data_center: "EU-RO-1".to_string(),
        };
        let template = Template {
            id: "t1".to_string(),
            name: odd.to_string(),
            image: "img\u{1b}[31m:1".to_string(),
            serverless: false,
        };
        let tables = [
            gpu_table(&[gpu(odd, 48, Some(0.4))], None),
            data_center_table(&[named]),
            volume_table(&[volume]),
            template_table(&[template]),
        ];
        for lines in tables {
            let text = lines.join("\n");
            assert!(text.contains("odd name"), "{text:?}");
            assert_eq!(lines.len(), 2, "{lines:?}");
            assert!(
                !text.chars().any(|c| c.is_control() && c != '\n'),
                "{text:?}"
            );
        }
    }

    #[test]
    fn a_gpu_id_from_the_api_reaches_a_resolve_error_clean() -> Result<(), serde_json::Error> {
        let gpus: Vec<GpuType> = serde_json::from_value(serde_json::json!([
            {"id": "odd\u{1b}[2J\ngpu", "memory": 48, "price": {"secure": 0.3},
             "maxCount": {"secure": 8}, "availability": "HIGH",
             "dataCenters": [{"id": "EU\u{1b}[31m-RO-1", "availability": "NONE"}]}
        ]))?;
        assert_eq!(gpus[0].id, "odd gpu", "cleaned when parsed");
        assert_eq!(gpus[0].data_centers[0].id, "EU-RO-1");
        let error = resolve(&target(list(&["odd gpu"]), ListOrAuto::Auto), &gpus).err();
        assert_eq!(
            error.as_deref(),
            Some(
                "no data center has odd gpu in stock for data_center_ids = \"auto\" (gpu_count = 1)"
            )
        );
        Ok(())
    }
}

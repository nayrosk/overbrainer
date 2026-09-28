//! Views of the Runpod catalog: GPU types filtered and ordered cheapest first,
//! data centers with the stock of chosen GPU types, and the resolution of a
//! target's `auto` choices. Pure functions over what
//! [`RunpodClient`](super::RunpodClient) lists.

use std::cmp::Ordering;

use crate::config::ListOrAuto;

use super::RunpodTarget;
use super::types::{Availability, DataCenter, GpuType, Stock};

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
}

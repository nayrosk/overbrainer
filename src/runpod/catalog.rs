//! Views of the Runpod catalog: GPU types filtered and ordered cheapest first,
//! and data centers with the stock of chosen GPU types. Pure functions over what
//! [`RunpodClient`](super::RunpodClient) lists.

use std::cmp::Ordering;

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
/// the cheapest such GPU type they have (see [`by_price`], `gpus` giving the
/// prices), ties by ID.
#[must_use]
pub fn stocked_data_centers<S: AsRef<str>>(
    data_centers: &[DataCenter],
    gpus: &[GpuType],
    chosen: &[S],
) -> Vec<DataCenterStock> {
    let cheapest = |row: &DataCenterStock| -> Option<&GpuType> {
        row.stock
            .iter()
            .filter(|entry| entry.availability.is_in_stock())
            .filter_map(|entry| gpus.iter().find(|gpu| gpu.id == entry.id))
            .min_by(|a, b| by_price(a, b))
    };
    let mut rows: Vec<DataCenterStock> = data_center_stock(data_centers, chosen)
        .into_iter()
        .filter(DataCenterStock::in_stock)
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

#[cfg(test)]
mod tests {
    use super::*;
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
        let gpus = [gpu("cheap", 24, Some(0.2)), gpu("dear", 80, Some(1.5))];
        let centers = [
            center("A-DEAR", vec![stock("dear", Availability::High)]),
            center(
                "B-EMPTY",
                vec![
                    stock("cheap", Availability::None),
                    stock("dear", Availability::None),
                ],
            ),
            center(
                "C-CHEAP",
                vec![
                    stock("cheap", Availability::Low),
                    stock("dear", Availability::High),
                ],
            ),
            center("D-CHEAP", vec![stock("cheap", Availability::High)]),
            center("E-OTHER", vec![stock("other", Availability::High)]),
        ];
        let rows = stocked_data_centers(&centers, &gpus, &["dear", "cheap"]);
        let row_ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
        assert_eq!(row_ids, vec!["C-CHEAP", "D-CHEAP", "A-DEAR"]);
    }
}

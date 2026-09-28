//! `RunpodClient`'s catalog and account listings against a local HTTP stub:
//! the v2 paths and query parameters, tolerant parsing of the reference
//! bodies, template pagination, retries, and errors without the key.

use std::time::Duration;

use overbrainer::retry::RetryPolicy;
use overbrainer::runpod::{ApiError, Availability, RunpodClient};
use secrecy::SecretString;
use serde_json::json;
use wiremock::matchers::{header, method, path, query_param, query_param_is_missing};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const KEY: &str = "rp_catalog_key_7c2e91";

fn client(server: &MockServer) -> Result<RunpodClient, ApiError> {
    Ok(
        RunpodClient::new(&format!("{}/v2", server.uri()), &SecretString::from(KEY))?.with_policy(
            RetryPolicy {
                max_retries: 2,
                base: Duration::from_millis(1),
                cap: Duration::from_millis(2),
            },
        ),
    )
}

/// `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE`, shaped
/// like the v2 reference example.
fn gpus_body() -> serde_json::Value {
    json!({"gpus": [
        {
            "id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "pool": "ADA_24",
            "manufacturer": "NVIDIA", "memory": 24, "secure": true, "community": true,
            "price": {"secure": 0.44, "community": 0.31, "serverless": 1.1},
            "maxCount": {"secure": 8, "community": 4},
            "availability": "HIGH",
            "dataCenters": [{"id": "US-KS-2", "name": "US Kansas 2", "availability": "HIGH"}],
            "cudaVersions": [{"version": "12.8", "available": true}]
        },
        {
            "id": "AMD Instinct MI300X OAM", "name": "MI300X", "pool": null,
            "manufacturer": "AMD", "memory": 192, "secure": true, "community": false,
            "price": {"secure": 2.49, "community": 0.0},
            "maxCount": {"secure": 8, "community": 0},
            "availability": "SOMETHING_NEW"
        }
    ]})
}

#[tokio::test]
async fn gpu_types_are_listed_with_their_pod_availability() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/catalog/gpus"))
        .and(query_param("include", "AVAILABILITY"))
        .and(query_param("product", "POD"))
        .and(query_param("cloud", "SECURE"))
        .and(query_param("count", "2"))
        .and(header("authorization", format!("Bearer {KEY}").as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(gpus_body()))
        .expect(1)
        .mount(&server)
        .await;
    let gpus = client(&server)?.list_gpu_types(2).await?;
    assert_eq!(gpus.len(), 2);
    let rtx = &gpus[0];
    assert_eq!(rtx.id, "NVIDIA GeForce RTX 4090");
    assert_eq!(rtx.name, "RTX 4090");
    assert_eq!(rtx.manufacturer, "NVIDIA");
    assert_eq!(rtx.memory, 24);
    assert_eq!(rtx.secure_price(), Some(0.44));
    assert_eq!(rtx.price.community, Some(0.31));
    assert_eq!(rtx.max_count.secure, 8);
    assert_eq!(rtx.availability, Availability::High);
    assert_eq!(rtx.data_centers.len(), 1);
    assert_eq!(rtx.data_centers[0].id, "US-KS-2");
    assert_eq!(rtx.data_centers[0].availability, Availability::High);
    assert_eq!(rtx.cuda_versions[0].version, "12.8");
    assert!(rtx.cuda_versions[0].available);
    let amd = &gpus[1];
    assert_eq!(amd.availability, Availability::Unknown);
    assert!(amd.data_centers.is_empty() && amd.cuda_versions.is_empty());
    Ok(())
}

#[tokio::test]
async fn a_gpu_count_below_one_asks_for_one() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/catalog/gpus"))
        .and(query_param("count", "1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"gpus": []})))
        .expect(1)
        .mount(&server)
        .await;
    assert!(client(&server)?.list_gpu_types(0).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn nulls_and_missing_fields_are_tolerated() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/catalog/gpus"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"gpus": [
            {"id": "X", "name": null, "memory": null, "price": null, "maxCount": null,
             "availability": null, "dataCenters": null, "somethingNew": [1, 2]}
        ]})))
        .mount(&server)
        .await;
    let gpus = client(&server)?.list_gpu_types(1).await?;
    assert_eq!(gpus.len(), 1);
    assert_eq!(gpus[0].name, "");
    assert_eq!(gpus[0].memory, 0);
    assert_eq!(gpus[0].secure_price(), None);
    assert_eq!(gpus[0].max_count.secure, 0);
    assert_eq!(gpus[0].availability, Availability::Unknown);
    Ok(())
}

#[tokio::test]
async fn data_centers_are_listed_with_their_gpu_availability() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/catalog/datacenters"))
        .and(query_param("include", "GPU_AVAILABILITY"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"dataCenters": [
            {
                "id": "US-KS-2", "name": "US Kansas 2", "region": "NORTH_AMERICA",
                "globalNetwork": true, "networkVolumeTypes": ["STANDARD", "HIGH_PERFORMANCE"],
                "compliance": ["SOC_2_TYPE_2"],
                "gpuAvailability": [
                    {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "availability": "HIGH"},
                    {"id": "NVIDIA A40", "name": "A40", "availability": "NONE"}
                ],
                "cpuAvailability": [{"id": "cpu3c-2-4", "name": "Compute-Optimized", "availability": "MEDIUM"}]
            },
            {"id": "EU-RO-1", "name": "EU Romania 1", "region": "EUROPE", "globalNetwork": false,
             "networkVolumeTypes": [], "compliance": []}
        ]})))
        .expect(1)
        .mount(&server)
        .await;
    let centers = client(&server)?.list_data_centers().await?;
    assert_eq!(centers.len(), 2);
    assert_eq!(centers[0].id, "US-KS-2");
    assert_eq!(centers[0].name, "US Kansas 2");
    assert_eq!(centers[0].region, "NORTH_AMERICA");
    assert_eq!(centers[0].gpu_availability.len(), 2);
    assert_eq!(centers[0].gpu_availability[1].id, "NVIDIA A40");
    assert_eq!(
        centers[0].gpu_availability[1].availability,
        Availability::None
    );
    assert!(centers[1].gpu_availability.is_empty());
    Ok(())
}

#[tokio::test]
async fn network_volumes_are_listed() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/network-volumes"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"networkVolumes": [
                {"id": "2q9m7x4c", "name": "training-dataset", "size": 100,
                 "dataCenter": "US-KS-2", "type": "HIGH_PERFORMANCE"},
                {"id": "agv6w2qcg7", "name": "older", "size": 50, "dataCenterId": "EU-RO-1"}
            ]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let volumes = client(&server)?.list_network_volumes().await?;
    assert_eq!(volumes.len(), 2);
    assert_eq!(volumes[0].id, "2q9m7x4c");
    assert_eq!(volumes[0].name, "training-dataset");
    assert_eq!(volumes[0].size, 100);
    assert_eq!(volumes[0].data_center, "US-KS-2");
    assert_eq!(volumes[1].data_center, "EU-RO-1");
    Ok(())
}

fn template(id: &str, serverless: bool) -> serde_json::Value {
    json!({
        "id": id, "name": format!("tpl {id}"), "image": format!("img/{id}:1"),
        "args": "", "disk": 50, "mounts": {}, "ports": ["22/tcp"],
        "env": {"HF_TOKEN": "hf_secret_value"}, "registry": null,
        "serverless": serverless, "public": false, "category": "NVIDIA",
        "startSsh": true, "startJupyter": false, "allowedCudaVersions": []
    })
}

#[tokio::test]
async fn templates_follow_every_page_and_keep_pod_templates_only() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/templates"))
        .and(query_param("cursor", "c2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "templates": [template("t3", false)],
            "pagination": {"hasNextPage": false, "nextCursor": null}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/templates"))
        .and(query_param("limit", "1000"))
        .and(query_param_is_missing("cursor"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "templates": [template("t1", false), template("t2", true)],
            "pagination": {"hasNextPage": true, "nextCursor": "c2"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let templates = client(&server)?.list_templates().await?;
    let ids: Vec<&str> = templates.iter().map(|t| t.id.as_str()).collect();
    assert_eq!(ids, vec!["t1", "t3"]);
    assert_eq!(templates[0].name, "tpl t1");
    assert_eq!(templates[0].image, "img/t1:1");
    assert!(!templates[0].serverless);
    assert!(
        !format!("{templates:?}").contains("hf_secret_value"),
        "a template's env was read"
    );
    Ok(())
}

#[tokio::test]
async fn a_repeated_template_cursor_is_an_error() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/templates"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "templates": [],
            "pagination": {"hasNextPage": true, "nextCursor": "same"}
        })))
        .expect(2)
        .mount(&server)
        .await;
    let error = client(&server)?
        .list_templates()
        .await
        .err()
        .ok_or("no error")?;
    assert!(matches!(error, ApiError::InvalidResponse(_)), "{error:?}");
    Ok(())
}

/// Every listing, answered 503 then 429 then success at `list_path`.
async fn assert_retried(list_path: &str, body: serde_json::Value) -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(list_path))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(list_path))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(list_path))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    let client = client(&server)?;
    match list_path {
        "/v2/catalog/gpus" => drop(client.list_gpu_types(1).await?),
        "/v2/catalog/datacenters" => drop(client.list_data_centers().await?),
        "/v2/network-volumes" => drop(client.list_network_volumes().await?),
        _ => drop(client.list_templates().await?),
    }
    let requests = server.received_requests().await.unwrap_or_default().len();
    assert_eq!(requests, 3, "{list_path}");
    Ok(())
}

#[tokio::test]
async fn every_listing_is_retried_on_server_errors_and_rate_limits() -> TestResult {
    assert_retried("/v2/catalog/gpus", json!({"gpus": []})).await?;
    assert_retried("/v2/catalog/datacenters", json!({"dataCenters": []})).await?;
    assert_retried("/v2/network-volumes", json!({"networkVolumes": []})).await?;
    assert_retried(
        "/v2/templates",
        json!({"templates": [], "pagination": {"hasNextPage": false, "nextCursor": null}}),
    )
    .await
}

#[tokio::test]
async fn no_listing_error_shows_the_key() -> TestResult {
    let server = MockServer::start().await;
    let echo = json!({"title": "Bad Request", "detail": format!("key {KEY} rejected")});
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(400).set_body_json(echo))
        .mount(&server)
        .await;
    let client = client(&server)?;
    let errors = [
        client.list_gpu_types(1).await.err(),
        client.list_data_centers().await.err(),
        client.list_network_volumes().await.err(),
        client.list_templates().await.err(),
    ];
    for error in errors {
        let error = error.ok_or("a listing succeeded")?;
        assert_eq!(error.status(), Some(400));
        let text = format!("{error} {error:?}");
        // Never quote the text: it is what is checked for the key.
        assert!(!text.contains(KEY), "an error holds the API key");
        assert!(text.contains("rejected"), "the error lost Runpod's detail");
    }
    Ok(())
}

#[tokio::test]
async fn a_listing_decode_error_never_quotes_the_body() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/templates"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"templates": "BODY-SECRET-TEXT"})),
        )
        .mount(&server)
        .await;
    let error = client(&server)?
        .list_templates()
        .await
        .err()
        .ok_or("no error")?;
    assert!(matches!(error, ApiError::InvalidResponse(_)), "{error:?}");
    assert!(
        !error.to_string().contains("BODY-SECRET-TEXT"),
        "the body was quoted"
    );
    Ok(())
}

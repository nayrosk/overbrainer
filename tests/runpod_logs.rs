//! The pod log stream against local stubs: replay, silence, resume with
//! `Last-Event-ID`, permanent refusals, redaction, and the log kept in the run
//! directory.

use std::io;
use std::time::{Duration, Instant};

use std::sync::atomic::AtomicBool;

use overbrainer::events::EventBus;
use overbrainer::runpod::{
    ApiError, DeleteReason, DeletedBy, LogError, LogQuery, LogSource, POD_LOG, POD_LOG_CURSOR,
    PodCtx, PodId, PodLogLine, PodRecord, RunpodClient, TAIL_MAX, Timing, drain, follow_logs,
    remove, snapshot, with_pod_logs,
};
use overbrainer::runs::Runs;
use secrecy::SecretString;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const KEY: &str = "rp_test_key_5f1d_logs";
const LOGS: &str = "/v2/pods/p1/logs";

fn client(uri: &str) -> Result<RunpodClient, ApiError> {
    RunpodClient::new(&format!("{uri}/v2"), &SecretString::from(KEY))
}

fn event(id: &str, source: &str, line: &str) -> String {
    let data = json!({"ts": "2026-06-01T12:02:03Z", "source": source, "line": line});
    format!("id: {id}\ndata: {data}\n\n")
}

fn stream(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

/// A sink keeping every line and the last cursor it was given.
#[derive(Default)]
struct Kept {
    lines: Vec<PodLogLine>,
    cursor: Option<String>,
}

impl Kept {
    fn sink(&mut self) -> impl FnMut(&[PodLogLine], Option<&str>) -> io::Result<()> + '_ {
        |lines: &[PodLogLine], cursor: Option<&str>| {
            self.lines.extend_from_slice(lines);
            if let Some(cursor) = cursor {
                self.cursor = Some(cursor.to_string());
            }
            Ok(())
        }
    }

    fn texts(&self) -> Vec<&str> {
        self.lines.iter().map(|line| line.line.as_str()).collect()
    }
}

#[tokio::test]
async fn a_snapshot_reads_the_replay_redacted_and_returns_its_cursor() -> TestResult {
    let server = MockServer::start().await;
    let body = [
        event("c/1", "system", "pulling image"),
        ": keepalive\n\n".to_string(),
        event(
            "c/2",
            "container",
            &format!("key {KEY} HF_TOKEN=hf_abcdefghijklmnop"),
        ),
        "id: c/3\n\n".to_string(),
    ]
    .concat();
    Mock::given(method("GET"))
        .and(path(LOGS))
        .and(query_param("tail", "10"))
        .and(query_param("source", "container"))
        .and(header("accept", "text/event-stream"))
        .and(header("authorization", format!("Bearer {KEY}").as_str()))
        .respond_with(stream(body))
        .expect(1)
        .mount(&server)
        .await;
    let query = LogQuery {
        source: Some(LogSource::Container),
        tail: Some(10),
        cursor: None,
    };
    let mut kept = Kept::default();
    let cursor = snapshot(
        &client(&server.uri())?,
        &PodId::new("p1")?,
        query,
        Duration::from_secs(10),
        &mut kept.sink(),
    )
    .await?;
    assert_eq!(cursor.as_deref(), Some("c/3"));
    assert_eq!(kept.texts(), ["pulling image", "key *** HF_TOKEN=***"]);
    assert_eq!(kept.lines[0].short_source(), "sys");
    assert_eq!(kept.cursor.as_deref(), Some("c/3"));
    Ok(())
}

/// A stream that stays open after its replay, as Runpod's does: the snapshot
/// ends on the silence.
#[tokio::test]
async fn a_snapshot_ends_on_silence_while_the_stream_stays_open() -> TestResult {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };
        let mut request = vec![0_u8; 4096];
        let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut request).await;
        let head =
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
        let body = [
            event("s/1", "container", "one"),
            event("s/2", "container", "two"),
        ]
        .concat();
        let _ = socket.write_all(head.as_bytes()).await;
        let _ = socket.write_all(body.as_bytes()).await;
        let _ = socket.flush().await;
        // Held open, silent, well past the snapshot.
        tokio::time::sleep(Duration::from_secs(30)).await;
    });
    let started = Instant::now();
    let mut kept = Kept::default();
    let cursor = snapshot(
        &client(&format!("http://{address}"))?,
        &PodId::new("p1")?,
        LogQuery::default(),
        Duration::from_secs(20),
        &mut kept.sink(),
    )
    .await?;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(kept.texts(), ["one", "two"]);
    assert_eq!(cursor.as_deref(), Some("s/2"));
    server.abort();
    Ok(())
}

#[tokio::test]
async fn follow_resumes_with_the_last_event_id_until_the_pod_is_gone() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(LOGS))
        .and(query_param("tail", "5000"))
        .respond_with(stream(
            [
                event("c/1", "container", "one"),
                event("c/2", "container", "two"),
            ]
            .concat(),
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(LOGS))
        .and(header("last-event-id", "c/2"))
        .respond_with(stream(event("c/3", "container", "three")))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(LOGS))
        .and(header("last-event-id", "c/3"))
        .respond_with(
            ResponseTemplate::new(404).set_body_json(json!({"status": 404, "title": "Not Found"})),
        )
        .mount(&server)
        .await;
    let query = LogQuery {
        source: None,
        tail: Some(TAIL_MAX),
        cursor: None,
    };
    let mut kept = Kept::default();
    let ended = tokio::time::timeout(
        Duration::from_secs(20),
        follow_logs(
            &client(&server.uri())?,
            &PodId::new("p1")?,
            query,
            &mut kept.sink(),
        ),
    )
    .await?;
    assert!(
        matches!(&ended, LogError::Api(error) if error.status() == Some(404)),
        "{ended:?}"
    );
    assert_eq!(kept.texts(), ["one", "two", "three"]);
    let requests = server.received_requests().await.unwrap_or_default();
    assert_eq!(requests.len(), 3);
    Ok(())
}

#[tokio::test]
async fn follow_stops_at_once_on_a_refusal() -> TestResult {
    for status in [401_u16, 403, 404] {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(LOGS))
            .respond_with(
                ResponseTemplate::new(status)
                    .set_body_json(json!({"status": status, "detail": format!("no {KEY}")})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let mut kept = Kept::default();
        let ended = tokio::time::timeout(
            Duration::from_secs(5),
            follow_logs(
                &client(&server.uri())?,
                &PodId::new("p1")?,
                LogQuery::default(),
                &mut kept.sink(),
            ),
        )
        .await?;
        assert!(
            matches!(&ended, LogError::Api(error) if error.status() == Some(status)),
            "{ended:?}"
        );
        assert!(!ended.to_string().contains(KEY), "{ended}");
    }
    Ok(())
}

#[tokio::test]
async fn an_answer_that_is_not_an_event_stream_is_refused() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(LOGS))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"id": "p1"})))
        .mount(&server)
        .await;
    let mut kept = Kept::default();
    let result = snapshot(
        &client(&server.uri())?,
        &PodId::new("p1")?,
        LogQuery::default(),
        Duration::from_secs(5),
        &mut kept.sink(),
    )
    .await;
    assert!(
        matches!(result, Err(LogError::Api(ApiError::InvalidResponse(_)))),
        "{result:?}"
    );
    assert!(kept.lines.is_empty());
    Ok(())
}

#[tokio::test]
async fn a_drain_keeps_the_lines_in_the_run_directory_and_resumes_from_there() -> TestResult {
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(LOGS))
        .and(query_param("tail", "5000"))
        .respond_with(stream(
            [
                event("c/1", "system", "start"),
                event(
                    "c/2",
                    "container",
                    "export RUNPOD_API_KEY=rpa_ABCDEFGH12345678",
                ),
            ]
            .concat(),
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(LOGS))
        .and(header("last-event-id", "c/2"))
        .respond_with(stream(event("c/3", "container", "later")))
        .mount(&server)
        .await;
    let client = client(&server.uri())?;
    let pod = PodId::new("p1")?;
    drain(&client, &runs, "r1", &pod).await;
    drain(&client, &runs, "r1", &pod).await;
    let dir = project.path().join("runs/r1");
    let kept = std::fs::read_to_string(dir.join(POD_LOG))?;
    let lines: Vec<PodLogLine> = kept
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let texts: Vec<&str> = lines.iter().map(|line| line.line.as_str()).collect();
    assert_eq!(texts, ["start", "export RUNPOD_API_KEY=***", "later"]);
    assert!(!kept.contains("rpa_ABCDEFGH"));
    assert_eq!(std::fs::read_to_string(dir.join(POD_LOG_CURSOR))?, "c/3");
    Ok(())
}

/// A stub pod `p1` whose API answers Runpod's 404 and accepts deletes.
async fn gone_pod(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(
            ResponseTemplate::new(404).set_body_json(json!({"status": 404, "title": "Not Found"})),
        )
        .mount(server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(204))
        .mount(server)
        .await;
}

fn timing() -> Timing {
    Timing {
        poll: Duration::from_millis(5),
        ready_timeout: Duration::from_millis(200),
        preflight_timeout: Duration::from_millis(200),
        reconcile_waits: [Duration::from_millis(5), Duration::from_millis(5)],
        delete_timeout: Duration::from_millis(300),
        gone_interval: Duration::from_millis(5),
    }
}

#[tokio::test]
async fn the_logs_are_kept_while_the_watch_runs() -> TestResult {
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(LOGS))
        .and(query_param("tail", "5000"))
        .respond_with(stream(event("c/1", "container", "epoch 1")))
        .mount(&server)
        .await;
    let client = client(&server.uri())?;
    let (bus, timing, interrupted) = (EventBus::new(), timing(), AtomicBool::new(false));
    let ctx = PodCtx {
        client: &client,
        runs: &runs,
        bus: &bus,
        timing: &timing,
        interrupted: &interrupted,
    };
    let mut record = PodRecord::new("r1", false, 1, "ssh-ed25519 AAAAhost");
    record.pod_id = Some(PodId::new("p1")?);
    let work = async {
        tokio::time::sleep(Duration::from_millis(300)).await;
        7
    };
    assert_eq!(with_pod_logs(&ctx, &record, work).await, 7);
    let kept = std::fs::read_to_string(project.path().join("runs/r1").join(POD_LOG))?;
    assert_eq!(kept.lines().count(), 1, "{kept}");
    assert!(kept.contains("epoch 1"), "{kept}");
    Ok(())
}

#[tokio::test]
async fn a_pod_is_deleted_only_after_a_last_read_of_its_logs() -> TestResult {
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(LOGS))
        .respond_with(stream(event("c/1", "container", "last words")))
        .mount(&server)
        .await;
    gone_pod(&server).await;
    let client = client(&server.uri())?;
    let (bus, timing, interrupted) = (EventBus::new(), timing(), AtomicBool::new(false));
    let ctx = PodCtx {
        client: &client,
        runs: &runs,
        bus: &bus,
        timing: &timing,
        interrupted: &interrupted,
    };
    std::fs::create_dir_all(project.path().join("runs/r1"))?;
    let mut record = PodRecord::new("r1", false, 1, "ssh-ed25519 AAAAhost");
    record.pod_id = Some(PodId::new("p1")?);
    remove(
        &ctx,
        &mut record,
        DeleteReason::Requested,
        DeletedBy::Client,
    )
    .await?;
    let requests = server.received_requests().await.unwrap_or_default();
    let order: Vec<String> = requests
        .iter()
        .filter(|request| request.url.path() == LOGS || request.method.as_str() == "DELETE")
        .map(|request| request.method.to_string())
        .collect();
    assert_eq!(order, ["GET", "DELETE"]);
    let kept = std::fs::read_to_string(project.path().join("runs/r1").join(POD_LOG))?;
    assert!(kept.contains("last words"), "{kept}");
    Ok(())
}

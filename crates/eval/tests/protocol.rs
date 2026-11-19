use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::Path,
    process::{Command, Stdio},
    time::Duration,
};

const DATA: &str = include_str!("../data/gsm8k-test.jsonl");
const SOURCE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const DATA_SHA: &str = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14";

fn job(dir: &Path, input: Value) -> std::process::Output {
    configured_job(dir, input, None)
}

fn configured_job(dir: &Path, input: Value, config: Option<&Path>) -> std::process::Output {
    job_with_threads(dir, input, config, "4")
}

fn job_with_threads(
    dir: &Path,
    input: Value,
    config: Option<&Path>,
    threads: &str,
) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_brewery-eval"));
    command.env_remove("BREWERY_ALLOW_GPU_LOAD");
    command.env("BREWERY_CPU_THREADS", threads);
    if let Some(config) = config {
        command.env("BREWERY_BASELINE_CONFIG", config);
        command.env("BREWERY_BASELINE_TOKEN_FIXTURE", "not-a-real-secret");
    } else {
        command.env_remove("BREWERY_BASELINE_CONFIG");
    }
    let mut child = command
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.to_string().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn fixture(rows: usize, wrong: usize) -> Value {
    use sha2::{Digest, Sha256};
    let predictions: Vec<_> = DATA.lines().take(rows).enumerate().map(|(i, line)| {
        let row: Value = serde_json::from_str(line).unwrap();
        let q = row["question"].as_str().unwrap();
        let a = row["answer"].as_str().unwrap().rsplit_once("####").unwrap().1.trim();
        json!({"id":i+1,"question_sha256":format!("{:x}",Sha256::digest(q)),"text":
            if i < wrong { "#### -999999".into() } else {format!("Synthetic derivation.\n#### {a}")}})
    }).collect();
    json!({"schema":"brewery-gsm8k-predictions-v1","source_sha256": SOURCE,
        "dataset_sha256":DATA_SHA,"protocol_id":"gsm8k-zero-shot-raw-v1",
        "synthetic":true,"predictions":predictions})
}

#[test]
fn standalone_score_can_be_reused_as_a_diagnostic_reference_for_another_source() {
    let dir = tempfile::tempdir().unwrap();
    let reference = dir.path().join("reference.json");
    let candidate = dir.path().join("candidate.json");
    let mut predictions = fixture(1319, 0);
    predictions["source_sha256"] = json!(DATA_SHA);
    predictions["provenance"] = json!({"precision":"int8"});
    std::fs::write(&reference, predictions.to_string()).unwrap();
    std::fs::write(&candidate, fixture(1319, 1).to_string()).unwrap();
    let standalone = job(
        dir.path(),
        json!({"kind":"evaluate","inputs":[{"kind":"gsm8k-predictions","path":reference}],
            "options":{"synthetic_fixture":true}}),
    );
    assert!(standalone.status.success(), "{standalone:?}");
    let standalone: Value = serde_json::from_slice(&standalone.stdout).unwrap();
    assert_eq!(standalone["score"]["correct"], 1319);
    assert!(standalone["reference_scope"].is_null());
    assert!(standalone["comparison"].is_null());
    let diagnostic = job(
        dir.path(),
        json!({"kind":"evaluate","inputs":[{"kind":"gsm8k-predictions","path":candidate},
            {"kind":"gsm8k-evaluation","path":standalone["artifact"]["path"]}],
            "options":{"synthetic_fixture":true}}),
    );
    assert!(diagnostic.status.success(), "{diagnostic:?}");
    let diagnostic: Value = serde_json::from_slice(&diagnostic.stdout).unwrap();
    assert_eq!(diagnostic["reference_scope"], "diagnostic");
    assert_eq!(diagnostic["identity"]["source_sha256"], SOURCE);
    assert_eq!(diagnostic["baseline_identity"]["source_sha256"], DATA_SHA);
    assert_eq!(
        diagnostic["baseline_identity"]["provenance"]["precision"],
        "int8"
    );
    assert_eq!(diagnostic["comparison"]["changed_answer_ids"], json!([1]));
    let mut bad_dataset = predictions.clone();
    bad_dataset["dataset_sha256"] = json!(SOURCE);
    let mut bad_protocol = predictions.clone();
    bad_protocol["protocol_id"] = json!("other-protocol");
    let mut bad_ids = predictions;
    bad_ids["predictions"][1]["id"] = json!(1);
    for (bad_reference, reason) in [
        (bad_dataset, "incompatible prediction"),
        (bad_protocol, "incompatible prediction"),
        (bad_ids, "GSM8K row 2"),
    ] {
        std::fs::write(&reference, bad_reference.to_string()).unwrap();
        let result = job(
            dir.path(),
            json!({"kind":"evaluate","inputs":[{"kind":"gsm8k-predictions","path":candidate},
                {"kind":"gsm8k-evaluation","path":reference}],
                "options":{"synthetic_fixture":true}}),
        );
        assert!(!result.status.success() && result.stdout.is_empty());
        assert!(String::from_utf8_lossy(&result.stderr).contains(reason));
    }
}

#[test]
fn full_synthetic_split_compares_per_question_and_checks_threshold_without_rounding() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.json");
    let b = dir.path().join("b.json");
    std::fs::write(&a, fixture(1319, 0).to_string()).unwrap();
    std::fs::write(&b, fixture(1319, 27).to_string()).unwrap();
    let baseline = job(
        dir.path(),
        json!({"kind":"baseline","inputs":[{"kind":"gsm8k-predictions","host":"local","path":a}],"options":{"synthetic_fixture":true}}),
    );
    assert!(
        baseline.status.success(),
        "{}",
        String::from_utf8_lossy(&baseline.stderr)
    );
    let out: Value = serde_json::from_slice(&baseline.stdout).unwrap();
    let baseline_path = out["artifact"]["path"].clone();
    assert_eq!(out["score"]["correct"], 1319);
    assert_eq!(out["identity"]["synthetic"], true);
    let request = json!({"kind":"evaluate","inputs":[
        {"kind":"gsm8k-predictions","host":"local","path":b},
        {"kind":"gsm8k-baseline","host":"local","path":out["artifact"]["path"]}],"options":{"synthetic_fixture":true}});
    let invalid_budget = job_with_threads(dir.path(), request.clone(), None, "0");
    assert!(!invalid_budget.status.success());
    assert!(String::from_utf8_lossy(&invalid_budget.stderr).contains("BREWERY_CPU_THREADS"));
    let comparison = job(dir.path(), request.clone());
    assert!(
        comparison.status.success(),
        "{}",
        String::from_utf8_lossy(&comparison.stderr)
    );
    let out: Value = serde_json::from_slice(&comparison.stdout).unwrap();
    let serial = job_with_threads(dir.path(), request, None, "1");
    assert!(serial.status.success(), "{serial:?}");
    let serial: Value = serde_json::from_slice(&serial.stdout).unwrap();
    assert_eq!(serial["score"], out["score"]);
    assert_eq!(serial["baseline_score"], out["baseline_score"]);
    assert_eq!(serial["comparison"], out["comparison"]);
    assert_eq!(serial["artifact"]["sha256"], out["artifact"]["sha256"]);
    assert_eq!(out["score"]["correct"], 1292);
    assert_eq!(out["comparison"]["changed_answers"], 27);
    assert_eq!(
        out["comparison"]["changed_answer_ids"],
        json!((1..=27).collect::<Vec<_>>())
    );
    assert_eq!(out["comparison"]["delta_correct"], -27);
    assert_eq!(out["comparison"]["within_two_percentage_points"], false);
    std::fs::write(&b, fixture(1319, 26).to_string()).unwrap();
    let valid = job(
        dir.path(),
        json!({"kind":"evaluate","inputs":[
        {"kind":"gsm8k-predictions","path":b},{"kind":"gsm8k-baseline","path":baseline_path}],
        "options":{"synthetic_fixture":true}}),
    );
    assert!(valid.status.success(), "{valid:?}");
    let valid: Value = serde_json::from_slice(&valid.stdout).unwrap();
    assert_eq!(valid["comparison"]["within_two_percentage_points"], true);
    std::fs::write(&b, fixture(1319, 27).to_string()).unwrap();
    let worse = job(
        dir.path(),
        json!({"kind":"baseline","inputs":[
        {"kind":"gsm8k-predictions","path":b}],"options":{"synthetic_fixture":true}}),
    );
    assert!(worse.status.success(), "{worse:?}");
    let worse: Value = serde_json::from_slice(&worse.stdout).unwrap();
    let gain = job(
        dir.path(),
        json!({"kind":"evaluate","inputs":[
        {"kind":"gsm8k-predictions","path":a},{"kind":"gsm8k-baseline","path":worse["artifact"]["path"]}],
        "options":{"synthetic_fixture":true}}),
    );
    assert!(gain.status.success(), "{gain:?}");
    let gain: Value = serde_json::from_slice(&gain.stdout).unwrap();
    assert_eq!(gain["comparison"]["delta_correct"], 27);
    assert_eq!(gain["comparison"]["within_two_percentage_points"], true);
    let mut corrupt = fixture(1319, 26);
    corrupt["predictions"][1318]["question_sha256"] = json!(SOURCE);
    std::fs::write(&b, corrupt.to_string()).unwrap();
    let bad = job(
        dir.path(),
        json!({"kind":"baseline","inputs":[{"kind":"gsm8k-predictions","host":"local","path":b}],"options":{"synthetic_fixture":true}}),
    );
    assert!(!bad.status.success() && bad.stdout.is_empty());
}

#[test]
fn refuses_gpu_load_without_operator_opt_in_and_refuses_job_selected_urls() {
    let dir = tempfile::tempdir().unwrap();
    let refused = job(
        dir.path(),
        json!({"kind":"generate","prompt":"hello","inputs":[
        {"kind":"hgn","path":dir.path()}]}),
    );
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("not enabled"));
    let url = job(
        dir.path(),
        json!({"kind":"baseline","options":{
        "baseline_source":"http://127.0.0.1:54321/v1/completions"}}),
    );
    assert!(!url.status.success());
    assert!(String::from_utf8_lossy(&url.stderr).contains("invalid baseline source ID"));
}

#[test]
fn local_endpoint_fixture_checks_every_protocol_request_and_fails_on_identity_drift() {
    for (synthetic, drift, attested) in [
        (true, false, false),
        (true, true, false),
        (false, false, false),
        (false, false, true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/completions", listener.local_addr().unwrap());
        let config = dir.path().join("endpoint.json");
        let runtime_id = format!("sha256:{SOURCE}");
        let precision = if attested { "bf16" } else { "int8" };
        std::fs::write(&config, json!({"test-fixture":{"url":url,"model":"synthetic-test-double",
            "source_sha256":SOURCE,"runtime_id":runtime_id,"bearer_env":"BREWERY_BASELINE_TOKEN_FIXTURE",
            "synthetic":synthetic,"precision":precision,"bf16_attested":attested}}).to_string()).unwrap();
        let options = json!({"baseline_source":"test-fixture","synthetic_fixture":synthetic});
        let request = if synthetic {
            let candidate = dir.path().join("candidate.json");
            let mut predictions = fixture(1319, 1);
            predictions["source_sha256"] = json!(DATA_SHA);
            std::fs::write(&candidate, predictions.to_string()).unwrap();
            json!({"kind":"evaluate","inputs":[{"kind":"gsm8k-predictions","path":candidate}],
                "options":options})
        } else {
            json!({"kind":"baseline","options":options})
        };
        let progress_path = dir.path().join("progress.json");
        let server = std::thread::spawn(move || {
            for (i, line) in DATA.lines().enumerate() {
                let (mut conn, _) = listener.accept().unwrap();
                conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut message = BufReader::new(&mut conn);
                let mut first = String::new();
                message.read_line(&mut first).unwrap();
                assert_eq!(first, "POST /v1/completions HTTP/1.1\r\n");
                let mut headers = String::new();
                loop {
                    let mut line = String::new();
                    message.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    assert!(!line.is_empty());
                    headers.push_str(&line);
                }
                let headers = headers.to_ascii_lowercase();
                assert!(headers.contains("authorization: bearer not-a-real-secret"));
                let len: usize = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap();
                let mut payload = vec![0; len];
                message.read_exact(&mut payload).unwrap();
                let job: Value = serde_json::from_slice(&payload).unwrap();
                let question: Value = serde_json::from_str(line).unwrap();
                assert_eq!(job["model"], "synthetic-test-double");
                assert_eq!(job["max_tokens"], 512);
                assert_eq!(job["temperature"], 0);
                assert_eq!(job["stream"], false);
                assert_eq!(job["n"], 1);
                assert_eq!(
                    job["prompt"],
                    format!(
                        "Solve the following math word problem. Show your reasoning and put the final numeric answer on the last line as #### <number>.\n\nQuestion: {}\nAnswer:",
                        question["question"].as_str().unwrap()
                    )
                );
                if i == 0 || (drift && i == 2) {
                    let snapshot = std::fs::read(&progress_path).unwrap();
                    assert!(!String::from_utf8_lossy(&snapshot).contains("not-a-real-secret"));
                    let snapshot: Value = serde_json::from_slice(&snapshot).unwrap();
                    assert_eq!(snapshot["phase"], "baseline_questions");
                    assert_eq!(snapshot["unit"], "questions");
                    assert_eq!(snapshot["total"], 1319);
                    assert_eq!(snapshot["completed"], i);
                }
                let answer = question["answer"]
                    .as_str()
                    .unwrap()
                    .rsplit_once("####")
                    .unwrap()
                    .1
                    .trim();
                let reply = json!({"model":"synthetic-test-double","choices":[{"text":format!("Fixture output only.\n#### {answer}"),"finish_reason":"stop"}]}).to_string();
                let source = if drift && i == 2 { DATA_SHA } else { SOURCE };
                write!(conn,"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {}\r\nX-Brewery-Source-Sha256: {source}\r\nX-Brewery-Runtime-Id: {runtime_id}\r\n\r\n{reply}",reply.len()).unwrap();
                if drift && i == 0 {
                    // Cross the throttle window before finishing row two.
                    std::thread::sleep(Duration::from_millis(1100));
                }
                if drift && i == 2 {
                    return 3;
                }
            }
            1319
        });
        let result = configured_job(dir.path(), request, Some(&config));
        if drift {
            assert!(!result.status.success());
            assert!(result.stdout.is_empty());
            assert!(String::from_utf8_lossy(&result.stderr).contains("row 3"));
            assert_eq!(server.join().unwrap(), 3);
        } else {
            assert!(result.status.success(), "{result:?}");
            assert!(!String::from_utf8_lossy(&result.stdout).contains("not-a-real-secret"));
            let valid: Value = serde_json::from_slice(&result.stdout).unwrap();
            let (identity, score, method) = if synthetic {
                assert_eq!(valid["reference_scope"], "diagnostic");
                assert_eq!(valid["identity"]["source_sha256"], DATA_SHA);
                assert_eq!(valid["comparison"]["changed_answer_ids"], json!([1]));
                (
                    &valid["baseline_identity"],
                    &valid["baseline_score"],
                    "synthetic-endpoint-fixture",
                )
            } else {
                let method = if attested {
                    "operator-bf16-endpoint"
                } else {
                    "operator-reference-endpoint"
                };
                (&valid["identity"], &valid["score"], method)
            };
            assert_eq!(score["total"], 1319);
            assert_eq!(score["correct"], 1319);
            assert_eq!(identity["synthetic"], synthetic);
            assert_eq!(identity["source_sha256"], SOURCE);
            assert_eq!(identity["provenance"]["baseline_source"], "test-fixture");
            assert_eq!(identity["provenance"]["precision"], precision);
            assert_eq!(identity["provenance"]["method"], method);
            assert_eq!(identity["provenance"]["bf16_attested"] == true, attested);
            assert_eq!(server.join().unwrap(), 1319);
        }
    }
}

use super::*;

const TOKEN: &str = "hf_fixture_NotARealCredential_17";
const REPLACEMENT: &str = "hf_fixture_NotARealCredential_23";
const LIST: &str = "/api/settings/hf-tokens";
const ALPHA: &str = "/api/settings/alpha/hf-token";

fn fixture() -> (Fixture, PathBuf) {
    let app = Fixture::new();
    let dir = app.path("private-creds");
    fs::create_dir(&dir).unwrap();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
    let token_file = dir.join("token");
    let worker_config = app.path("downloader.json");
    fs::write(
        &worker_config,
        json!({
            "root":app.path("downloader"),
            "engine":app.path("engine"),
            "hf_download":{"hfr": app.path("hfr"),"max_bytes":1000,"token_file":token_file}
        })
        .to_string(),
    )
    .unwrap();
    fs::write(
        app.path("dispatch.json"),
        json!({"hosts":{
            "alpha":{"local":true,"worker":app.path("worker"),"config":worker_config,
                "capabilities":{"hf_download":true},"hf_token_file":token_file},
            "beta":{"ssh":"beta","capabilities":{"hf_download":true},"hf_token_file":token_file}
        }})
        .to_string(),
    )
    .unwrap();
    (app, token_file)
}

#[test]
fn hf_token_is_write_only_and_rotates_across_restart() {
    let (mut app, token_file) = fixture();
    app.launch();
    assert_eq!(
        app.json("GET", LIST, None, 200),
        json!({"hosts":[{"id":"alpha","configured":false}]})
    );
    assert_eq!(
        app.json("PUT", ALPHA, Some(json!({"token":TOKEN})), 200),
        json!({"id":"alpha","configured":true})
    );
    assert_eq!(fs::read_to_string(&token_file).unwrap(), TOKEN);
    assert_eq!(
        fs::metadata(&token_file).unwrap().permissions().mode() & 0o777,
        0o600
    );
    app.stop();
    app.launch();
    let expected = json!({"hosts":[{"id":"alpha","configured":true}]});
    assert_eq!(app.json("GET", LIST, None, 200), expected);
    assert_eq!(
        app.json("PUT", ALPHA, Some(json!({"token":REPLACEMENT})), 200),
        json!({"id":"alpha","configured":true})
    );
    assert_eq!(fs::read_to_string(&token_file).unwrap(), REPLACEMENT);
    for path in [LIST, "/api/settings", "/api/state"] {
        let (status, text) = app.request("GET", path, None, "");
        assert_eq!(status, 200);
        for private in [TOKEN, REPLACEMENT, token_file.to_str().unwrap()] {
            assert!(!text.contains(private));
        }
    }
    let mut conn = app.open_request("GET", LIST, None, "");
    let mut response = String::new();
    conn.read_to_string(&mut response).unwrap();
    assert!(
        response
            .to_ascii_lowercase()
            .contains("\r\ncache-control: no-store\r\n")
    );
    assert_eq!(app.request("GET", ALPHA, None, "").0, 405);
    for _ in 0..2 {
        assert_eq!(
            app.json("DELETE", ALPHA, Some(json!({})), 200),
            json!({"id":"alpha","configured":false})
        );
        assert!(!token_file.exists());
    }
    for entry in fs::read_dir(app.path("state")).unwrap() {
        let bytes = fs::read(entry.unwrap().path()).unwrap();
        for secret in [TOKEN, REPLACEMENT] {
            assert!(
                !bytes
                    .windows(secret.len())
                    .any(|bytes| bytes == secret.as_bytes())
            );
        }
    }
}

#[test]
fn hf_token_rejects_nonlocal_cross_origin_busy_and_bad_input() {
    let (mut app, token_file) = fixture();
    fs::write(app.path("hold"), "").unwrap();
    app.launch();
    let token = Some(json!({"token":TOKEN}));
    for (route, status) in [
        ("/api/settings/beta/hf-token", 409),
        ("/api/settings/unknown/hf-token", 404),
    ] {
        assert_eq!(app.request("PUT", route, token.clone(), "").0, status);
    }
    for (method, body) in [("PUT", token.clone()), ("DELETE", Some(json!({})))] {
        for foreign in [
            "Origin: https://foreign.example\r\n",
            "Sec-Fetch-Site: cross-site\r\n",
        ] {
            assert_eq!(app.request(method, ALPHA, body.clone(), foreign).0, 403);
        }
        assert_eq!(app.request(method, ALPHA, None, "").0, 415);
    }
    let bad = [
        (json!({"token":["hf_invalid"]}), 422),
        (json!([TOKEN]), 422),
        (json!({"token":TOKEN,"path":"/wrong"}), 422),
        (json!({"token":format!("{TOKEN}\n")}), 400),
        (json!({"token":"x".repeat(2048)}), 413),
    ];
    for (body, status) in bad {
        let (seen, response) = app.request("PUT", ALPHA, Some(body), "");
        assert_eq!(seen, status);
        assert!(!response.contains(TOKEN));
    }
    assert_eq!(app.request("DELETE", ALPHA, Some(json!([])), "").0, 422);
    let job =
        app.enqueue(json!({"kind":"hf_download","host":"alpha","options":{"repo_id":"repo"}}));
    app.await_job(&job, |job| job["progress"] == "worker progress");
    assert_eq!(app.request("PUT", ALPHA, token, "").0, 409);
    assert_eq!(app.request("DELETE", ALPHA, Some(json!({})), "").0, 409);
    assert!(!token_file.exists());
}

#[test]
fn hf_token_refuses_unmatched_config_and_unsafe_files() {
    let (mut app, token_file) = fixture();
    app.launch();
    let token = Some(json!({"token":TOKEN}));
    let config_path = app.path("downloader.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
    config["hf_download"]["token_file"] = json!(app.path("other-token"));
    fs::write(&config_path, config.to_string()).unwrap();
    assert_eq!(app.request("PUT", ALPHA, token.clone(), "").0, 409);
    config["hf_download"]["token_file"] = json!(token_file);
    fs::write(&config_path, config.to_string()).unwrap();

    let unrelated = app.path("unrelated");
    fs::write(&unrelated, "unchanged").unwrap();
    fs::set_permissions(&unrelated, fs::Permissions::from_mode(0o600)).unwrap();
    std::os::unix::fs::symlink(&unrelated, &token_file).unwrap();
    assert_eq!(app.request("PUT", ALPHA, token.clone(), "").0, 503);
    assert_eq!(app.request("DELETE", ALPHA, Some(json!({})), "").0, 503);
    fs::remove_file(&token_file).unwrap();
    fs::hard_link(&unrelated, &token_file).unwrap();
    assert_eq!(app.request("PUT", ALPHA, token.clone(), "").0, 503);
    assert_eq!(fs::read_to_string(&unrelated).unwrap(), "unchanged");
    fs::remove_file(&token_file).unwrap();
    fs::set_permissions(
        token_file.parent().unwrap(),
        fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    assert_eq!(app.json("GET", LIST, None, 200), json!({"hosts":[]}));
    assert_eq!(app.request("PUT", ALPHA, token, "").0, 503);
    assert!(!token_file.exists());
}

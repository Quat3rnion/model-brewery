use super::*;

const ROUTE: &str = "/api/admin/ssh-hosts";
const DISCOVERY: &str = "/api/admin/ssh-host-key";
const FINGERPRINT: &str = "SHA256:kmYlmuZBwV+t4tJiZRRDQWzVv848g2LMqV6K5JdW6X8";
const HELPER: &str = r#"#!/usr/bin/python3
import json, pathlib, sys
here = pathlib.Path(__file__).parent
request = json.load(sys.stdin)
op = request["op"]
with (here / "calls").open("a") as f:
    print(op, file=f)
key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f"
pin = 'b' * 64
root = f"/home/brewery/.local/share/model-brewery/workers/{request['id']}/{pin}"
reply = {"state":{"test":"verified","plan":"planned","install":"installed"}[op],"host_key":key}
if op != "test":
    reply.update(bundle_id=pin,worker=root+'/brewery-worker',config=root+'/worker.json')
if op == "plan":
    reply["confirmation"] = 'a' * 64
if op == "install":
    assert request["confirmation"] == 'a' * 64
    if (here / "substitute").exists():
        reply["worker"] = reply["worker"].replace('/home/brewery/', '/home/other/')
if (here / "bad-key").exists():
    reply["host_key"] = "ssh-ed25519 invalid"
print(json.dumps(reply))
"#;

struct Sshd(Child);

impl Drop for Sshd {
    fn drop(&mut self) {
        let _ = self.0.kill();
        self.0.wait().unwrap();
    }
}

#[test]
#[ignore = "requires local OpenSSH sshd, ssh-keygen, and ssh-keyscan"]
fn discovers_the_loopback_sshd_public_key_without_enrolling_it() {
    let mut app = Fixture::new();
    let key = app.path("host_key");
    assert!(
        Command::new("/usr/bin/ssh-keygen")
            .args(["-q", "-t", "ed25519", "-N", "", "-f"])
            .arg(&key)
            .status()
            .unwrap()
            .success()
    );
    let public_key_file = app.path("host_key.pub");
    let public_key = fs::read_to_string(&public_key_file).unwrap();
    let public_key = public_key
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ");
    let fingerprint = Command::new("/usr/bin/ssh-keygen")
        .args(["-l", "-E", "sha256", "-f"])
        .arg(&public_key_file)
        .output()
        .unwrap();
    assert!(fingerprint.status.success());
    let fingerprint_text = String::from_utf8(fingerprint.stdout).unwrap();
    let fingerprint = fingerprint_text.split_whitespace().nth(1).unwrap();

    let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = reservation.local_addr().unwrap().port();
    let config = app.path("sshd_config");
    fs::write(
        &config,
        format!(
            "ListenAddress 127.0.0.1\nPort {port}\nHostKey {}\nPidFile {}\n\
             PasswordAuthentication no\nKbdInteractiveAuthentication no\n\
             PubkeyAuthentication no\nUsePAM no\n",
            key.display(),
            app.path("sshd.pid").display(),
        ),
    )
    .unwrap();
    let log = app.path("sshd.log");
    drop(reservation);
    let mut sshd = Sshd(
        Command::new("/usr/bin/sshd")
            .args(["-D", "-e", "-f"])
            .arg(&config)
            .stdin(Stdio::null())
            .stderr(fs::File::create(&log).unwrap())
            .spawn()
            .unwrap(),
    );
    wait_for(|| {
        let text = fs::read_to_string(&log).unwrap();
        assert!(sshd.0.try_wait().unwrap().is_none(), "{text}");
        text.contains("Server listening on").then_some(())
    });
    app.launch();
    assert_eq!(
        app.json(
            "POST",
            DISCOVERY,
            Some(json!({"host":"127.0.0.1","port":port})),
            200,
        ),
        json!({
            "host":"127.0.0.1","port":port,"algorithm":"ssh-ed25519",
            "fingerprint":fingerprint,"public_key":public_key
        })
    );
    assert_eq!(app.json("GET", ROUTE, None, 200)["hosts"], json!([]));
}

#[test]
fn definition_edits_are_local_durable_and_revision_guarded() {
    let mut app = Fixture::new();
    app.launch();
    assert_eq!(app.json("GET", ROUTE, None, 200)["hosts"], json!([]));
    let setup = json!({
        "id":"reader","name":"Reader","host":"reader.invalid",
        "user":"brewery","port":2222,"fingerprint":FINGERPRINT
    });
    let row = app.json("POST", ROUTE, Some(setup.clone()), 201);
    assert_eq!(row["status"], "defined");
    let resource = format!("{ROUTE}/reader");
    app.json("POST", ROUTE, Some(setup), 409);
    app.stop();
    app.launch();
    assert_eq!(app.json("GET", ROUTE, None, 200)["hosts"][0], row);
    app.json("DELETE", &resource, Some(json!({"revision":999})), 409);
    app.json(
        "DELETE",
        &resource,
        Some(json!({"revision":row["revision"]})),
        204,
    );
    assert_eq!(app.json("GET", ROUTE, None, 200)["hosts"], json!([]));
}

#[test]
fn provisioning_requires_verified_key_confirmed_plan_and_busy_checks() {
    let mut app = Fixture::new();
    let helper = app.path("provision");
    fs::write(&helper, HELPER).unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
    let env = [("BREWERY_SSH_PROVISION", helper)];
    app.launch_with(&env);
    let target = json!({
        "id":"reader","name":"Reader","host":"reader.invalid","user":"brewery","port":2222,
        "fingerprint":"SHA256:ZkAslGjFiUHdGf/WUL8rQvkib4PTvQatUV0OUQSncCA"
    });
    app.json("POST", ROUTE, Some(target.clone()), 201);
    assert_eq!(app.json("GET", ROUTE, None, 200)["available"], true);
    assert!(!app.path("calls").exists());
    let action = format!("{ROUTE}/reader/action");
    app.json(
        "POST",
        &action,
        Some(json!({"revision":1,"action":"plan"})),
        409,
    );
    assert!(!app.path("calls").exists());
    fs::write(app.path("bad-key"), "").unwrap();
    app.json(
        "POST",
        &action,
        Some(json!({"revision":1,"action":"test"})),
        502,
    );
    fs::remove_file(app.path("bad-key")).unwrap();
    let verified = app.json(
        "POST",
        &action,
        Some(json!({"revision":1,"action":"test"})),
        200,
    );
    assert_eq!(verified["status"], "verified");
    let planned = app.json(
        "POST",
        &action,
        Some(json!({"revision":2,"action":"plan"})),
        200,
    );
    assert_eq!(planned["status"], "planned");
    app.json(
        "POST",
        &action,
        Some(json!({"revision":3,"action":"install","confirmation":"wrong"})),
        409,
    );
    let confirm = json!({"revision":3,"action":"install","confirmation":planned["confirmation"]});
    fs::write(app.path("substitute"), "").unwrap();
    app.json("POST", &action, Some(confirm.clone()), 502);
    fs::remove_file(app.path("substitute")).unwrap();
    let installed = app.json("POST", &action, Some(confirm), 200);
    assert_eq!(installed["status"], "installed");
    // The installed worker must be manageable from Settings without restarting
    // the controller or enrolling it for evaluation.
    let settings = app.json("GET", "/api/settings", None, 200);
    let worker = settings["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|worker| worker["id"] == "reader")
        .expect("Settings must include an installed SSH worker");
    assert_eq!(worker["evaluation"]["status"], "profile_only");
    assert_eq!(worker["management"]["available"], true);
    assert_eq!(
        worker["management"]["href"],
        "#/settings/infrastructure?host=reader"
    );
    let state_url = worker["management"]["state_url"].as_str().unwrap();
    let action_url = worker["management"]["action_url"].as_str().unwrap();
    let management = app.json("GET", state_url, None, 200);
    let policy = management["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|worker| worker["id"] == "reader")
        .unwrap();
    let resources = json!({"action":"set_host_resources","worker":"reader",
        "revision":policy["revision"],"resources":{"cpu_threads":2}});
    app.json("POST", action_url, Some(resources.clone()), 200);
    app.json("POST", action_url, Some(resources), 409);
    let payload = settings::target_profile(worker, 9282);
    let mut invalid_eval = payload.clone();
    invalid_eval["evaluation_enabled"] = json!(true);
    app.json("PUT", "/api/settings/reader", Some(invalid_eval), 400);
    let saved = app.json("PUT", "/api/settings/reader", Some(payload), 200);
    app.stop();
    app.launch_with(&env);
    let settings = app.json("GET", "/api/settings", None, 200);
    assert!(settings["hosts"].as_array().unwrap().contains(&saved));
    let state = app.json("GET", "/api/state", None, 200);
    assert!(
        state["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|worker| worker["id"] == "reader"
                && worker["capabilities"]["evaluate"] == false
                && worker["benchmark_targets"] == json!([]))
    );
    let mut changed = target.clone();
    changed.as_object_mut().unwrap().remove("id");
    changed["host"] = json!("other.invalid");
    changed["revision"] = installed["revision"].clone();
    let path = format!("{ROUTE}/reader");
    app.json("PUT", &path, Some(changed), 409);
    let model = app.register("reader", "hf", "/shared/model");
    fs::write(app.path("hold"), "").unwrap();
    let job = app.enqueue(json!({"kind":"inspect","input_ids":[model]}));
    app.await_job(&job, |job| job["state"] == "running");
    let revision = json!({"revision":installed["revision"]});
    app.json("DELETE", &path, Some(revision.clone()), 409);
    app.cancel(&job);
    app.await_job(&job, |job| job["state"] == "cancelled");
    let calls = fs::read_to_string(app.path("calls")).unwrap();
    app.json("DELETE", &path, Some(revision), 204);
    assert_eq!(fs::read_to_string(app.path("calls")).unwrap(), calls);
    app.json("POST", ROUTE, Some(target), 409);
}

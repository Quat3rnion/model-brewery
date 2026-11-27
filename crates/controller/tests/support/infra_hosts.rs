use super::*;

const ROUTE: &str = "/api/admin/ssh-hosts";
const DISCOVERY: &str = "/api/admin/ssh-host-key";
const IDENTITIES: &str = "/api/admin/ssh-identities";
const FINGERPRINT: &str = "SHA256:kmYlmuZBwV+t4tJiZRRDQWzVv848g2LMqV6K5JdW6X8";
const HELPER: &str = r#"#!/usr/bin/python3
import json, pathlib, sys
here = pathlib.Path(__file__).parent
request = json.load(sys.stdin)
op = request["op"]
import os
with (here / "identities").open("a") as f:
    print(os.environ["BREWERY_PROVISION_IDENTITY"], file=f)
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
    let state_paths = || {
        fs::read_dir(app.path("state/ssh-hosts"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<std::collections::BTreeSet<_>>()
    };
    let before = state_paths();
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
    assert_eq!(state_paths(), before);
}

#[test]
fn host_key_discovery_only_accepts_one_host_and_port() {
    let mut app = Fixture::new();
    app.launch();
    let invalid_hosts = [
        "",
        "-f",
        " 127.0.0.1",
        "127.0.0.1 127.0.0.2",
        "127.0.0.1,127.0.0.2",
        "127.0.0.0/24",
        "127.0.0.1:22",
        "ssh://127.0.0.1",
        "user@127.0.0.1",
        "127.0.0.1\n127.0.0.2",
    ];
    for host in invalid_hosts {
        app.json("POST", DISCOVERY, Some(json!({"host":host,"port":22})), 400);
    }
    for (body, status) in [
        (json!({"host":"127.0.0.1","port":0}), 400),
        (json!({"host":"127.0.0.1","port":65536}), 422),
        (json!({"host":"127.0.0.1"}), 422),
        (json!(["127.0.0.1", 22]), 422),
        (json!({"host":["127.0.0.1"],"port":22}), 422),
        (json!({"host":"127.0.0.1","port":"22-23"}), 422),
        (json!({"host":"127.0.0.1","port":22,"user":"brewery"}), 422),
        (
            json!({"host":"127.0.0.1","port":22,"algorithm":"ssh-rsa"}),
            422,
        ),
        (json!({"host":"x".repeat(9000),"port":22}), 413),
    ] {
        app.json("POST", DISCOVERY, Some(body), status);
    }
    assert_eq!(
        app.request(
            "POST",
            DISCOVERY,
            Some(json!({"host":"127.0.0.1","port":22})),
            "Origin: http://other.test\r\n",
        )
        .0,
        403
    );
    assert_eq!(
        app.request("POST", DISCOVERY, None, "Content-Type: text/plain\r\n")
            .0,
        415
    );
    let mut malformed = app.open_request(
        "POST",
        DISCOVERY,
        None,
        "Content-Type: application/json\r\nContent-Length: 1\r\n",
    );
    malformed.write_all(b"!").unwrap();
    let mut response = String::new();
    malformed.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.0 400 "), "{response}");
}

fn generate_identity(path: &std::path::Path) {
    assert!(
        Command::new("/usr/bin/ssh-keygen")
            .args(["-q", "-t", "ed25519", "-N", "", "-f"])
            .arg(path)
            .status()
            .unwrap()
            .success()
    );
}

#[test]
fn identity_catalog_returns_only_eligible_local_paths() {
    let mut app = Fixture::new();
    let home = app.path("home");
    let ssh = home.join(".ssh");
    fs::create_dir_all(&ssh).unwrap();
    let good = ssh.join("approved");
    let outside = app.path("custom");
    generate_identity(&good);
    generate_identity(&outside);
    generate_identity(&ssh.join("loose"));
    fs::set_permissions(ssh.join("loose"), fs::Permissions::from_mode(0o644)).unwrap();
    generate_identity(&ssh.join("linked"));
    fs::hard_link(ssh.join("linked"), app.path("other-link")).unwrap();
    std::os::unix::fs::symlink(&outside, ssh.join("symlink")).unwrap();
    fs::write(ssh.join("config"), "Host *\n").unwrap();
    app.launch_with(&[("HOME", home), ("BREWERY_PROVISION_IDENTITY", outside)]);
    assert_eq!(
        app.json("GET", IDENTITIES, None, 200),
        json!({"identities":[{"path":good,"label":"approved"}],"default_configured":true})
    );
    assert_eq!(
        app.request("GET", IDENTITIES, None, "Origin: http://other.test\r\n")
            .0,
        403
    );
}

#[test]
fn definition_identity_changes_invalidate_verification_and_missing_keys_block_login() {
    let mut app = Fixture::new();
    let helper = app.path("provision");
    fs::write(&helper, HELPER).unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
    let home = app.path("home");
    fs::create_dir(&home).unwrap();
    let client = app.path("custom-client");
    let next = app.path("next-client");
    generate_identity(&client);
    generate_identity(&next);
    let symlink = app.path("symlink");
    std::os::unix::fs::symlink(&client, &symlink).unwrap();
    let symlink_dir = app.path("alias");
    std::os::unix::fs::symlink(app.root.path(), &symlink_dir).unwrap();
    let env = [
        ("HOME", home),
        ("BREWERY_SSH_PROVISION", helper),
        ("BREWERY_PROVISION_IDENTITY", app.path("absent-default")),
    ];
    app.launch_with(&env);
    assert_eq!(
        app.json("GET", IDENTITIES, None, 200),
        json!({"identities":[],"default_configured":false})
    );
    let setup = json!({
        "id":"reader","name":"Reader","host":"127.0.0.1","user":"brewery","port":2222,
        "fingerprint":"SHA256:ZkAslGjFiUHdGf/WUL8rQvkib4PTvQatUV0OUQSncCA"
    });
    for bad in [
        json!("relative"),
        json!("~/.ssh/id_ed25519"),
        json!(format!("{}/../custom-client", client.display())),
        json!(app.path("%h")),
        json!(app.path("${HOME}")),
        json!(app.path("injection\nIdentityAgent")),
        json!(app.path("custom-client.pub")),
        json!(symlink),
        json!(symlink_dir.join("custom-client")),
    ] {
        let mut invalid = setup.clone();
        invalid["identity_file"] = bad;
        app.json("POST", ROUTE, Some(invalid), 400);
    }
    let row = app.json("POST", ROUTE, Some(setup.clone()), 201);
    assert!(row["identity_file"].is_null());
    let resource = format!("{ROUTE}/reader");
    let action = format!("{resource}/action");
    let err = app.json(
        "POST",
        &action,
        Some(json!({"revision":1,"action":"test"})),
        409,
    );
    assert!(err["error"].as_str().unwrap().contains("SSH identity"));
    assert!(!app.path("calls").exists());
    let mut edit = setup;
    edit.as_object_mut().unwrap().remove("id");
    edit["revision"] = json!(1);
    edit["identity_file"] = json!(client);
    let row = app.json("PUT", &resource, Some(edit.clone()), 200);
    assert_eq!(row["identity_file"], json!(client));
    app.stop();
    app.launch_with(&env);
    assert_eq!(app.json("GET", ROUTE, None, 200)["hosts"][0], row);
    let test = json!({"revision":2,"action":"test"});
    fs::set_permissions(&client, fs::Permissions::from_mode(0o644)).unwrap();
    app.json("POST", &action, Some(test.clone()), 409);
    assert!(!app.path("calls").exists());
    fs::set_permissions(&client, fs::Permissions::from_mode(0o600)).unwrap();
    let verified = app.json("POST", &action, Some(test), 200);
    assert_eq!(verified["status"], "verified");
    assert_eq!(
        fs::read_to_string(app.path("identities")).unwrap().trim(),
        client.to_str().unwrap()
    );
    edit["revision"] = verified["revision"].clone();
    edit.as_object_mut().unwrap().remove("identity_file");
    let preserved = app.json("PUT", &resource, Some(edit.clone()), 200);
    assert_eq!(preserved["identity_file"], json!(client));
    assert_eq!(preserved["status"], "verified");
    edit["revision"] = preserved["revision"].clone();
    edit["identity_file"] = json!(next);
    let changed = app.json("PUT", &resource, Some(edit.clone()), 200);
    assert_eq!(changed["identity_file"], json!(next));
    assert_eq!(changed["status"], "defined");
    edit["revision"] = changed["revision"].clone();
    edit["identity_file"] = Value::Null;
    let cleared = app.json("PUT", &resource, Some(edit), 200);
    assert!(cleared["identity_file"].is_null());
}

#[test]
#[ignore = "requires local OpenSSH ssh-keyscan"]
fn host_key_discovery_does_not_queue_and_releases_its_slot_after_timeout() {
    let mut app = Fixture::new();
    app.launch();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let target = json!({"host":"127.0.0.1","port":listener.local_addr().unwrap().port()});
    let mut probe = app.open_request("POST", DISCOVERY, Some(target.clone()), "");
    let (silent, _) = wait_for(|| match listener.accept() {
        Ok(connection) => Some(connection),
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => None,
        Err(e) => panic!("loopback server failed: {e}"),
    });
    app.json("POST", DISCOVERY, Some(target.clone()), 429);
    let mut response = String::new();
    probe.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.0 502 "), "{response}");
    drop(silent);
    drop(listener);
    app.json("POST", DISCOVERY, Some(target), 502);
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
fn qsa_probes_only_explicitly_selected_enrolled_hosts() {
    let mut app = Fixture::new();
    let helper = app.path("provision");
    fs::write(&helper, HELPER).unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
    let identity = app.path("default-client");
    generate_identity(&identity);
    let env = [
        ("BREWERY_SSH_PROVISION", helper),
        ("BREWERY_PROVISION_IDENTITY", identity),
    ];
    app.launch_with(&env);
    for id in ["qsa-picked", "qsa-skipped"] {
        app.json(
            "POST",
            ROUTE,
            Some(json!({
                "id":id,"name":id,"host":format!("{id}.invalid"),"user":"brewery","port":2222,
                "fingerprint":"SHA256:ZkAslGjFiUHdGf/WUL8rQvkib4PTvQatUV0OUQSncCA"
            })),
            201,
        );
        let action = format!("{ROUTE}/{id}/action");
        app.json(
            "POST",
            &action,
            Some(json!({"revision":1,"action":"test"})),
            200,
        );
        let plan = app.json(
            "POST",
            &action,
            Some(json!({"revision":2,"action":"plan"})),
            200,
        );
        app.json(
            "POST",
            &action,
            Some(json!({
                "revision":3,"action":"install","confirmation":plan["confirmation"]
            })),
            200,
        );
    }
    app.stop();
    fs::write(app.path("qsa-runtime-ready"), "").unwrap();
    let calls = app.path("qsa-probe-calls");
    app.launch_with(&env);
    assert!(
        !calls.exists(),
        "QSA startup contacted an unselected enrolled worker"
    );
    app.stop();

    let file = app.path("dispatch.json");
    let mut config: Value = serde_json::from_str(&fs::read_to_string(&file).unwrap()).unwrap();
    config["qsa_ssh_hosts"] = json!(["qsa-picked"]);
    fs::write(&file, config.to_string()).unwrap();
    app.launch_with(&env);
    assert_eq!(fs::read_to_string(&calls).unwrap(), "qsa-picked\n");
    let state = app.json("GET", "/api/state", None, 200);
    for (id, ready) in [("qsa-picked", true), ("qsa-skipped", false)] {
        let worker = state["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|host| host["id"] == id)
            .unwrap();
        assert_eq!(worker["capabilities"]["qsa_train"], ready);
    }
    assert_eq!(fs::read_to_string(&calls).unwrap(), "qsa-picked\n");
}

#[test]
fn provisioning_requires_verified_key_confirmed_plan_and_busy_checks() {
    let mut app = Fixture::new();
    let helper = app.path("provision");
    fs::write(&helper, HELPER).unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
    let identity = app.path("default-client");
    generate_identity(&identity);
    let env = [
        ("BREWERY_SSH_PROVISION", helper),
        ("BREWERY_PROVISION_IDENTITY", identity.clone()),
    ];
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
    let config = fs::read_to_string(app.path("state/ssh-hosts/reader.ssh")).unwrap();
    assert!(config.contains(&format!("IdentityFile \"{}\"\n", identity.display())));
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
    changed["revision"] = installed["revision"].clone();
    let path = format!("{ROUTE}/reader");
    changed["identity_file"] = json!(app.path("other-client"));
    app.json("PUT", &path, Some(changed.clone()), 409);
    changed.as_object_mut().unwrap().remove("identity_file");
    changed["host"] = json!("other.invalid");
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

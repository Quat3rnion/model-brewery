use super::*;

const ROUTE: &str = "/api/admin/ssh-hosts";
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
    assert_eq!(worker["evaluation"]["status"], "not_enrolled");
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
    let invalid_eval = json!({"revision":0,"evaluation_enabled":true,
        "benchmark_targets":[],"baseline_endpoints":[],"enabled_suites":[]});
    let rejected = app.json("PUT", "/api/settings/reader", Some(invalid_eval), 409);
    assert!(rejected["error"].as_str().unwrap().contains("not enrolled"));
    app.stop();
    app.launch_with(&env);
    let state = app.json("GET", "/api/state", None, 200);
    assert!(
        state["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|worker| worker["id"] == "reader")
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

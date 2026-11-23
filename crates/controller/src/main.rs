mod admin;
mod backend_calibration;
mod backend_hf;
mod conversion;
mod dispatch;
mod domain;
mod feed;
mod gguf;
mod hf_download;
mod hf_tokens;
mod idempotency;
mod infra_hosts;
mod local_edit;
mod profiles;
mod remote_edit;
mod scratch;
mod settings;
mod store;
mod suggestions;
mod web;
mod weights;

use dispatch::Dispatcher;
use domain::Host;
use serde_json::Value;
use std::{
    env, fs,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use store::Store;
use tokio::net::TcpListener;

type Error = Box<dyn std::error::Error + Send + Sync>;

struct App {
    store: Mutex<Store>,
    hosts: Vec<Host>,
    settings: settings::Managed,
    hf_tokens: hf_tokens::Store,
    backend_download: bool,
    backend_calibration: Option<PathBuf>,
    backend_edit_formats: std::collections::BTreeSet<String>,
    pub admin: Option<admin::Admin>,
    local_edit: Arc<local_edit::Lanes>,
    ssh_hosts: Arc<infra_hosts::Registry>,
    remote_edit: Arc<remote_edit::Installer>,
    suggestions: suggestions::PathSuggestions,
    feed_subscribers: Arc<tokio::sync::Semaphore>,
    scratch_slots: Arc<tokio::sync::Semaphore>,
}

impl App {
    fn defined_hosts(&self) -> Result<Vec<Host>, Error> {
        let mut hosts = self.hosts.clone();
        hosts.extend(self.ssh_hosts.defined_hosts()?);
        Ok(hosts)
    }

    fn effective_hosts(&self) -> Result<Vec<Host>, Error> {
        let hosts = self.settings.effective(&self.defined_hosts()?);
        match &self.admin {
            Some(admin) => admin.effective_hosts(&hosts, self),
            None => Ok(hosts),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let listen: SocketAddr = env::var("BREWERY_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:4141".into())
        .parse()?;
    if !listen.ip().is_loopback() {
        return Err("BREWERY_LISTEN must be a numeric loopback socket address".into());
    }
    let home = || {
        env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or("HOME is not set; specify BREWERY_DATA")
    };
    let data = if let Some(path) = env::var_os("BREWERY_DATA") {
        PathBuf::from(path)
    } else if let Some(path) = env::var_os("XDG_DATA_HOME") {
        PathBuf::from(path).join("model-brewery")
    } else {
        home()?.join(".local/share/model-brewery")
    };
    let config = env::var_os("BREWERY_DISPATCH_CONFIG").map(PathBuf::from);
    let config_value = config
        .as_deref()
        .map(|path| -> Result<Value, Error> { Ok(serde_json::from_reader(fs::File::open(path)?)?) })
        .transpose()?;
    let mut hosts = config_value
        .as_ref()
        .map(load_hosts)
        .transpose()?
        .unwrap_or_default();
    let dist = env_path("BREWERY_WEB_DIST", "web/dist");
    let store = Store::open(&data)?;
    let local_edit = Arc::new(
        local_edit::Lanes::prepare(&data, &dist, config_value.as_ref())
            .await
            .unwrap_or_else(|_| {
                eprintln!("Local CPU editor storage is unavailable");
                local_edit::Lanes::default()
            }),
    );
    let ssh_hosts = Arc::new(infra_hosts::Registry::open(&data, config_value.as_ref())?);
    let remote_edit = Arc::new(remote_edit::Installer::open(&data)?);
    let mut dispatcher = Dispatcher {
        command: env_path("BREWERY_DISPATCH", "brewery-dispatch"),
        backend: None,
        local_edit: local_edit.clone(),
        ssh_hosts: ssh_hosts.clone(),
        remote_edit: remote_edit.clone(),
        launch_timeout: Duration::from_secs(env_num("BREWERY_DISPATCH_TIMEOUT_SECS", 21600)?),
        poll: Duration::from_millis(env_num("BREWERY_POLL_MS", 1000)?.max(25)),
    };
    let admin = admin::Admin::from_env(&data, &hosts, config.as_deref(), &dispatcher.command)?;
    let mut hf_tokens = hf_tokens::Store::from_config(config_value.as_ref(), &dist);
    match backend_hf::Lane::prepare(&data, &dist).await {
        Ok(Some(lane)) => {
            hosts.push(lane.host()?);
            hf_tokens.add_local(backend_hf::ID, lane.local_config());
            dispatcher.backend = Some(lane);
        }
        Ok(None) => {}
        Err(_) => eprintln!("Backend download storage is unavailable"),
    }
    let app = Arc::new(App {
        store: Mutex::new(store),
        hosts,
        settings: settings::Managed::from_config(config_value.as_ref()),
        hf_tokens,
        backend_download: dispatcher.backend.is_some(),
        backend_calibration: dispatcher
            .backend
            .as_ref()
            .and_then(|lane| lane.calibration.clone()),
        backend_edit_formats: dispatcher
            .backend
            .as_ref()
            .map(|lane| lane.edit_formats.clone())
            .unwrap_or_default(),
        admin,
        local_edit,
        ssh_hosts,
        remote_edit,
        suggestions: suggestions::PathSuggestions::new(dispatcher.command.clone()),
        feed_subscribers: Arc::new(tokio::sync::Semaphore::new(feed::SUBSCRIBERS)),
        scratch_slots: Arc::new(tokio::sync::Semaphore::new(1)),
    });
    let listener = TcpListener::bind(listen).await?;
    let addr = listener.local_addr()?;
    let router = web::router(app.clone(), addr, dist);
    println!("brewery listening on http://{addr}");
    // A persistence/queue failure must be visible, not an HTTP server that silently stops dispatching.
    tokio::select! {
        result = axum::serve(listener, router) => result?,
        result = dispatch::run(app, dispatcher) => result?,
    }
    Ok(())
}

fn env_path(name: &str, default: &str) -> PathBuf {
    env::var_os(name).map_or_else(|| PathBuf::from(default), PathBuf::from)
}

fn env_num(name: &str, default: u64) -> Result<u64, Error> {
    match env::var(name) {
        Ok(value) => Ok(value.parse::<u64>()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(err.into()),
    }
}

fn load_hosts(config: &Value) -> Result<Vec<Host>, Error> {
    let hosts = config
        .get("hosts")
        .and_then(Value::as_object)
        .ok_or("dispatch config requires a hosts object")?;
    hosts
        .iter()
        .map(|(id, entry)| {
            if id == backend_hf::ID {
                return Err("backend-hf is reserved for controller downloads".into());
            }
            Ok(Host::from_config(id, entry)?)
        })
        .collect()
}

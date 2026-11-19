//! hfr follows redirects and pagination itself. Keep it credential-free:
//! for authenticated jobs it only sees a scoped local view of the checked Hub.
use super::{Inventory, POLICY};
use anyhow::{anyhow, ensure, Context, Result};
use brewery_worker::hf_download::valid_token;
use serde::{Deserialize, Serialize};
use std::{
    ffi::CString,
    fs::File,
    io::{self, Read},
    os::unix::{
        ffi::OsStrExt,
        fs::MetadataExt,
        io::{AsRawFd, FromRawFd},
    },
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use url::Url;

pub(super) fn read_token(path: &Path) -> Result<Option<String>> {
    // Open the validated parent and then its basename relative to that FD;
    // neither a symlinked ancestor nor an atomic Settings replacement can
    // redirect the file read. No error contains the path or contents.
    let parent = CString::new(
        path.parent()
            .context("invalid Hub credential path")?
            .as_os_str()
            .as_bytes(),
    )?;
    let basename = CString::new(
        path.file_name()
            .context("invalid Hub credential path")?
            .as_bytes(),
    )?;
    // Linux openat2: RESOLVE_NO_SYMLINKS = 0x04.
    let how = [
        u64::from((libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_RDONLY) as u32),
        0,
        0x04,
    ];
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            libc::AT_FDCWD,
            parent.as_ptr(),
            &how,
            size_of_val(&how),
        )
    };
    ensure!(fd >= 0, "Hub credential directory unavailable");
    let dir = unsafe { File::from_raw_fd(fd as i32) };
    let owner = unsafe { libc::geteuid() };
    let meta = dir.metadata()?;
    ensure!(
        meta.is_dir() && meta.uid() == owner && meta.mode() & 0o7777 == 0o700,
        "Hub credential directory must be owner-only"
    );
    let fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            basename.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return if io::Error::last_os_error().kind() == io::ErrorKind::NotFound {
            Ok(None)
        } else {
            Err(anyhow!("Hub credential file unavailable"))
        };
    }
    let file = unsafe { File::from_raw_fd(fd) };
    let meta = file.metadata()?;
    ensure!(
        meta.is_file() && meta.uid() == owner && meta.nlink() == 1 && meta.mode() & 0o7777 == 0o600,
        "Hub credential file must be owner-only"
    );
    ensure!(
        unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) } == 0,
        "cannot protect Hub credential process"
    );
    let mut token = String::new();
    file.take(257)
        .read_to_string(&mut token)
        .map_err(|_| anyhow!("invalid Hub credential file"))?;
    ensure!(valid_token(&token), "invalid Hub credential file");
    Ok(Some(token))
}

pub(super) struct Broker {
    endpoint: String,
    stop: Arc<AtomicBool>,
}

impl Drop for Broker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Broker {
    pub(super) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub(super) fn start(inv: &Inventory, token: Option<String>, resumed: bool) -> Result<Self> {
        let server =
            Arc::new(Server::http("127.0.0.1:0").map_err(|_| anyhow!("Hub relay bind failed"))?);
        // This capability is not the Hub credential. Pass the endpoint only
        // in the child environment, not argv or a public status.
        let mut key = [0u8; 32];
        let count = unsafe { libc::getrandom(key.as_mut_ptr().cast(), key.len(), 0) };
        ensure!(
            count == key.len() as isize,
            "Hub relay random source unavailable"
        );
        use std::fmt::Write;
        let mut prefix = String::from("/relay-");
        for byte in key {
            let _ = write!(prefix, "{byte:02x}");
        }
        let endpoint = format!("http://{}{prefix}", server.server_addr());
        let relay = Arc::new(Relay::new(inv, prefix, token)?);
        if resumed {
            // Hub access is repository-scoped. Check it again instead of
            // inheriting a prior job's authorization from complete stage files.
            if relay.read_token_route.is_some() {
                relay.cas_ticket()?;
            } else if relay.bearer.is_some() {
                let (path, &size) = relay
                    .files
                    .iter()
                    .find(|(_, size)| **size != 0)
                    .context("empty download")?;
                relay.range(path, size, "bytes=0-0")?;
            }
        }
        let stop = Arc::new(AtomicBool::new(false));
        for _ in 0..4 {
            let (server, relay, stop) = (server.clone(), relay.clone(), stop.clone());
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    match server.recv_timeout(Duration::from_millis(100)) {
                        Ok(Some(request)) => relay.handle(request),
                        Ok(None) => (),
                        Err(_) => break,
                    }
                }
            });
        }
        Ok(Self { endpoint, stop })
    }
}

struct Relay {
    prefix: String,
    info: Vec<u8>,
    tree: Vec<u8>,
    revision: String,
    listing: String,
    read_token_route: Option<String>,
    fixture_xet_cas_port: Option<u16>,
    hub: Url,
    files: std::collections::BTreeMap<String, u64>,
    bearer: Option<ureq::http::HeaderValue>,
    client: ureq::Agent,
}

// Deliberately no Debug. Only these fields from Hub enter the child response.
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct CasTicket {
    cas_url: String,
    exp: u64,
    access_token: String,
}

fn encode(path: &str) -> String {
    use std::fmt::Write;
    path.bytes().fold(String::new(), |mut encoded, byte| {
        if byte.is_ascii_alphanumeric() || b"/-_.~".contains(&byte) {
            encoded.push(byte as char);
        } else {
            let _ = write!(encoded, "%{byte:02X}");
        }
        encoded
    })
}

impl Relay {
    fn new(inv: &Inventory, prefix: String, token: Option<String>) -> Result<Self> {
        let bearer = token
            .map(|token| -> Result<_> {
                let mut value = ureq::http::HeaderValue::from_str(&format!("Bearer {token}"))
                    .map_err(|_| anyhow!("invalid Hub credential"))?;
                value.set_sensitive(true);
                Ok(value)
            })
            .transpose()?;
        let resolve = format!("/{}/resolve/{}/", inv.repo, inv.commit);
        let files = inv
            .files
            .iter()
            .filter(|(_, f)| f.xet.is_none())
            .map(|(name, f)| (format!("{resolve}{}", encode(name)), f.bytes))
            .collect();
        let rows: Vec<_> = inv
            .files
            .iter()
            .map(|(name, f)| {
                serde_json::json!({
                    "type":"file", "path":name, "size":f.bytes,
                    "lfs": f.lfs.as_ref().map(|oid| serde_json::json!({"oid":oid})),
                    "xetHash": f.xet,
                })
            })
            .collect();
        Ok(Self {
            prefix,
            revision: format!(
                "/api/models/{}/revision/{}?expand=sha",
                inv.repo, inv.commit
            ),
            listing: format!(
                "/api/models/{}/tree/{}?recursive=true&expand=false",
                inv.repo, inv.commit
            ),
            read_token_route: inv
                .has_xet()
                .then(|| format!("/api/models/{}/xet-read-token/{}", inv.repo, inv.commit)),
            fixture_xet_cas_port: inv.fixture_xet_cas_port,
            info: serde_json::to_vec(&serde_json::json!({"sha": inv.commit}))?,
            tree: serde_json::to_vec(&rows)?,
            files,
            hub: Url::parse(&inv.origin)?,
            bearer,
            client: ureq::Agent::config_builder()
                .proxy(None)
                .http_status_as_error(false)
                .max_redirects(0)
                .timeout_global(Some(Duration::from_secs(120)))
                .build()
                .into(),
        })
    }

    fn handle(&self, request: Request) {
        let Some(path) = request.url().strip_prefix(&self.prefix) else {
            let _ = request.respond(Response::empty(404));
            return;
        };
        if *request.method() == Method::Get && (path == self.revision || path == self.listing) {
            let body = if path == self.revision {
                &self.info
            } else {
                &self.tree
            };
            let _ = request.respond(Response::from_data(body.clone()));
        } else if *request.method() == Method::Get && self.read_token_route.as_deref() == Some(path)
        {
            let ticket = self
                .cas_ticket()
                .and_then(|ticket| Ok(serde_json::to_vec(&ticket)?));
            if let Ok(ticket) = ticket {
                let _ = request.respond(Response::from_data(ticket));
            } else {
                let _ = request.respond(Response::empty(403));
            }
        } else if let Some(&size) = self.files.get(path) {
            match request.method() {
                Method::Head => {
                    let _ = request.respond(Response::empty(200));
                }
                Method::Get => {
                    let mut ranges = request.headers().iter().filter(|h| h.field.equiv("Range"));
                    let up = ranges
                        .next()
                        .filter(|_| ranges.next().is_none())
                        .and_then(|header| self.range(path, size, header.value.as_str()).ok());
                    match up {
                        Some(response) => {
                            let _ = request.respond(response);
                        }
                        None => {
                            let _ = request.respond(Response::empty(403));
                        }
                    }
                }
                _ => {
                    let _ = request.respond(Response::empty(405));
                }
            }
        } else {
            let _ = request.respond(Response::empty(404));
        }
    }

    fn cas_ticket(&self) -> Result<CasTicket> {
        let path = self
            .read_token_route
            .as_deref()
            .context("no Xet read scope")?;
        let url = self.hub.join(path)?;
        let mut get = self.client.get(url.as_str());
        if let Some(bearer) = &self.bearer {
            get = get.header("Authorization", bearer.clone());
        }
        let mut response = get
            .call()
            .map_err(|_| anyhow!("Hub Xet read authorization failed"))?;
        ensure!(
            response.status() == 200,
            "Hub Xet read authorization failed"
        );
        let mut ticket: CasTicket =
            brewery_worker::decode_limited(response.body_mut().as_reader(), 8192)
                .map_err(|_| anyhow!("invalid Hub Xet read ticket"))?;
        let cas = Url::parse(&ticket.cas_url).map_err(|_| anyhow!("invalid Hub Xet CAS origin"))?;
        let allowed = match self.fixture_xet_cas_port {
            Some(port) => {
                cas.scheme() == "http"
                    && cas.host_str() == Some("127.0.0.1")
                    && cas.port_or_known_default() == Some(port)
            }
            None => {
                cas.scheme() == "https"
                    && cas.port().is_none()
                    && cas.host_str() == Some("cas-server.xethub.hf.co")
            }
        };
        ensure!(
            allowed && url_is_origin(&cas),
            "untrusted Hub Xet CAS origin"
        );
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        ensure!(
            (now + 1..=now + 3600).contains(&ticket.exp)
                && (1..=4096).contains(&ticket.access_token.len())
                && ticket
                    .access_token
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-~+/=".contains(&b))
                && !ticket.access_token.starts_with("hf_")
                && self
                    .bearer
                    .as_ref()
                    .is_none_or(|bearer| bearer.to_str().is_ok_and(|v| !ticket
                        .access_token
                        .contains(v.trim_start_matches("Bearer ")))),
            "invalid Hub Xet read ticket"
        );
        // The pinned client receives only the brief CAS-scoped ticket. It
        // does not have the Hub bearer and cannot request a write token.
        ticket.cas_url = cas.origin().ascii_serialization();
        Ok(ticket)
    }

    fn range(&self, path: &str, size: u64, range: &str) -> Result<Response<impl Read + Send>> {
        let (start, end) = range
            .strip_prefix("bytes=")
            .and_then(|s| s.split_once('-'))
            .context("invalid range")?;
        let (start, end) = (start.parse::<u64>()?, end.parse::<u64>()?);
        ensure!(
            start <= end && end < size && end - start < POLICY.range_chunk_mib * 1024 * 1024,
            "invalid range bounds"
        );
        let len = end - start + 1;
        let content_range = format!("bytes {start}-{end}/{size}");
        let mut url = self.hub.join(path)?;
        let mut remote = false;
        for _ in 0..8 {
            let mut get = self
                .client
                .get(url.as_str())
                .header("Range", range)
                .header("Accept-Encoding", "identity");
            if !remote {
                if let Some(bearer) = &self.bearer {
                    get = get.header("Authorization", bearer.clone());
                }
            }
            let response = get
                .call()
                .map_err(|_| anyhow!("Hub transfer request failed"))?;
            if response.status().is_redirection() {
                let location = response
                    .headers()
                    .get("location")
                    .context("Hub redirect lacks location")?
                    .to_str()?;
                ensure!(
                    location.len() <= 8192
                        && self.bearer.as_ref().is_none_or(|bearer| bearer
                            .to_str()
                            .is_ok_and(|v| !location.contains(v.trim_start_matches("Bearer ")))),
                    "unsafe Hub redirect"
                );
                let next = url.join(location)?;
                ensure!(
                    next.username().is_empty()
                        && next.password().is_none()
                        && next.fragment().is_none(),
                    "unsafe Hub redirect"
                );
                if !remote && next.origin() == self.hub.origin() {
                    // Same Hub origin: only this file's immutable resolve or
                    // the Hub's same-repository immutable cache route.
                    let cache_path = format!(
                        "/api/resolve-cache/models{}",
                        path.replacen("/resolve/", "/", 1)
                    );
                    ensure!(
                        next.path() == path || next.path() == cache_path,
                        "unscoped Hub file redirect"
                    );
                } else {
                    ensure!(
                        self.hub.scheme() == "https"
                            && next.scheme() == "https"
                            && next.port().is_none()
                            && matches!(
                                next.host_str(),
                                Some(
                                    "cdn-lfs.huggingface.co"
                                        | "cdn-lfs.hf.co"
                                        | "cdn-lfs-us-1.hf.co"
                                        | "cdn-lfs-eu-1.hf.co"
                                        | "cas-bridge.xethub.hf.co"
                                )
                            ),
                        "untrusted Hub transfer origin"
                    );
                    remote = true;
                }
                url = next;
                continue;
            }
            let headers = response.headers();
            let valid_range = response.status() == 206
                && headers.get("content-range").and_then(|h| h.to_str().ok())
                    == Some(&content_range);
            ensure!(
                valid_range || (response.status() == 200 && start == 0 && len == size),
                "unexpected Hub payload status"
            );
            ensure!(
                headers.get("content-length").is_none_or(|value| value
                    .to_str()
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    == Some(len)),
                "unexpected Hub payload length"
            );
            let headers = vec![Header::from_bytes("Content-Range", content_range)
                .map_err(|()| anyhow!("invalid content range"))?];
            return Ok(Response::new(
                StatusCode(206),
                headers,
                response.into_body().into_reader().take(len),
                Some(len as usize),
                None,
            ));
        }
        Err(anyhow!("too many Hub redirects"))
    }
}

fn url_is_origin(url: &Url) -> bool {
    url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && url.path() == "/"
}

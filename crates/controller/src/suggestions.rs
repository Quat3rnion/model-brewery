use crate::dispatch::read_limited;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::PathBuf, process::Stdio, time::Duration};
use tokio::{io::AsyncWriteExt as _, process::Command, sync::Semaphore, time::timeout};

const MAX_PATH: usize = 4096;
const MAX_ITEMS: usize = 32;
const MAX_PATH_BYTES: usize = 16 * 1024;
const MAX_STDOUT: usize = 64 * 1024;
const SUGGEST_TIMEOUT: Duration = Duration::from_secs(3);

pub struct PathSuggestions {
    command: PathBuf,
    permits: Semaphore,
}

pub enum SuggestError {
    Busy,
    Timeout,
    Unavailable,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub host: String,
    prefix: String,
}

impl Query {
    pub fn valid_prefix(&self) -> bool {
        self.prefix.is_empty() || normalized_absolute(&self.prefix)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Suggestions {
    items: Vec<Item>,
    truncated: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Item {
    path: String,
    kind: Kind,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Kind {
    File,
    Directory,
}

impl PathSuggestions {
    pub fn new(command: PathBuf) -> Self {
        Self {
            command,
            permits: Semaphore::new(2),
        }
    }

    pub async fn get(&self, query: Query) -> Result<Suggestions, SuggestError> {
        let _permit = self.permits.try_acquire().map_err(|_| SuggestError::Busy)?;
        let payload = serde_json::to_vec(&serde_json::json!({ "prefix": query.prefix }))
            .map_err(|_| SuggestError::Unavailable)?;
        // Keep the child in the request future. Axum drops that future on
        // disconnect; timeout and invalid output also drop the child.
        let mut child = Command::new(&self.command)
            .args(["suggest-paths", &query.host])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| SuggestError::Unavailable)?;
        let mut group = ProcessGroup(child.id().expect("unreaped child") as libc::pid_t);
        let mut stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let (bytes, status) = timeout(SUGGEST_TIMEOUT, async {
            let write = async move {
                stdin.write_all(&payload).await?;
                drop(stdin);
                Ok(())
            };
            // Drain output before reaping the group leader. If a hung
            // transport keeps stdout open, the group can still be killed
            // without risking a reused process group ID.
            let (_, bytes) = tokio::try_join!(write, read_limited(stdout, MAX_STDOUT))?;
            let status = child.wait().await?;
            group.0 = 0;
            Ok::<_, std::io::Error>((bytes, status))
        })
        .await
        .map_err(|_| SuggestError::Timeout)?
        .map_err(|_| SuggestError::Unavailable)?;
        if !status.success() {
            return Err(SuggestError::Unavailable);
        }
        let reply: Suggestions =
            serde_json::from_slice(&bytes).map_err(|_| SuggestError::Unavailable)?;
        let mut seen = BTreeSet::new();
        if reply.items.len() > MAX_ITEMS
            || reply
                .items
                .iter()
                .map(|item| item.path.len())
                .sum::<usize>()
                > MAX_PATH_BYTES
            || reply.items.iter().any(|item| {
                !normalized_absolute(&item.path)
                    || item.path != "/" && item.path.ends_with('/')
                    || !item.path.starts_with(&query.prefix)
                    || !seen.insert(item.path.as_str())
            })
        {
            return Err(SuggestError::Unavailable);
        }
        Ok(reply)
    }
}

struct ProcessGroup(libc::pid_t);

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        if self.0 > 0 {
            // SAFETY: this is the process group created for this request,
            // and its leader has not been reaped.
            unsafe { libc::kill(-self.0, libc::SIGKILL) };
        }
    }
}

fn normalized_absolute(path: &str) -> bool {
    path.len() <= MAX_PATH
        && path.starts_with('/')
        && !path.chars().any(char::is_control)
        && !path.contains("//")
        && path.split('/').all(|part| part != "." && part != "..")
}

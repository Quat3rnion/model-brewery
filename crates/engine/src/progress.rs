//! Shared engine/eval wire format. Snapshots are best effort and never evidence.
use serde::{ser::SerializeSeq, Serialize, Serializer};
use std::{
    collections::BTreeMap,
    io::{self, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const INTERVAL: Duration = Duration::from_secs(1);
const MAX_THREAD_ROWS: usize = 16;

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Serialize)]
struct Snapshot {
    phase: &'static str,
    unit: &'static str,
    completed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    current_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    files_completed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    files_total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_count: Option<u64>,
    #[serde(
        skip_serializing_if = "BTreeMap::is_empty",
        serialize_with = "serialize_threads"
    )]
    threads: BTreeMap<usize, String>,
    updated_at_ms: u64,
}

fn serialize_threads<S: Serializer>(
    threads: &BTreeMap<usize, String>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct Assignment<'a> {
        slot: usize,
        task: &'a str,
    }
    let mut rows = serializer.serialize_seq(Some(threads.len().min(MAX_THREAD_ROWS)))?;
    for (&slot, task) in threads.iter().take(MAX_THREAD_ROWS) {
        rows.serialize_element(&Assignment { slot, task })?;
    }
    rows.end()
}

fn safe_label(name: &str, max_bytes: usize) -> bool {
    !name.is_empty()
        && name.len() <= max_bytes
        && !matches!(name, "." | "..")
        && !name.contains(['/', '\\'])
        && !name.chars().any(|ch| {
            ch.is_control() || matches!(ch, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        })
}

pub(crate) struct Progress {
    root: Option<PathBuf>,
}

impl Progress {
    pub(crate) fn new(root: &Path, inputs: impl IntoIterator<Item = impl AsRef<Path>>) -> Self {
        // Public CLI invocations need not use an isolated work directory. Do not
        // change an input tree (or overwrite an input named progress.json).
        let destination = root.join("progress.json");
        let overlaps = inputs.into_iter().any(|p| {
            p.as_ref()
                .canonicalize()
                .is_ok_and(|path| destination.starts_with(path))
        });
        Self {
            root: (!overlaps).then(|| root.to_owned()),
        }
    }

    pub(crate) fn start(
        &self,
        phase: &'static str,
        unit: &'static str,
        total: Option<u64>,
    ) -> Phase<'_> {
        self.start_for(phase, unit, total, None, None)
    }

    pub(crate) fn start_at(
        &self,
        phase: &'static str,
        unit: &'static str,
        total: Option<u64>,
        completed: u64,
    ) -> Phase<'_> {
        self.begin(phase, unit, total, None, None, completed)
    }

    pub(crate) fn start_for(
        &self,
        phase: &'static str,
        unit: &'static str,
        total: Option<u64>,
        files_total: Option<u64>,
        input: Option<(u64, u64)>,
    ) -> Phase<'_> {
        self.begin(phase, unit, total, files_total, input, 0)
    }

    fn begin(
        &self,
        phase: &'static str,
        unit: &'static str,
        total: Option<u64>,
        files_total: Option<u64>,
        input: Option<(u64, u64)>,
        completed: u64,
    ) -> Phase<'_> {
        let mut step = Phase {
            root: self.root.as_deref(),
            last: Instant::now(),
            reported_threads: false,
            snapshot: Snapshot {
                phase,
                unit,
                total,
                completed,
                current_file: None,
                files_completed: files_total.map(|_| 0),
                files_total,
                input_index: input.map(|(index, _)| index),
                input_count: input.map(|(_, count)| count),
                threads: BTreeMap::new(),
                updated_at_ms: 0,
            },
        };
        step.emit();
        step
    }
}

pub(crate) struct Phase<'a> {
    root: Option<&'a Path>,
    last: Instant,
    reported_threads: bool,
    snapshot: Snapshot,
}

impl Phase<'_> {
    pub(crate) fn file(&mut self, path: &Path) {
        self.snapshot.current_file = path
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| safe_label(name, 255))
            .map(str::to_owned);
        self.tick();
    }

    pub(crate) fn assign(&mut self, slot: usize, task: &str) {
        if (1..=64).contains(&slot) && safe_label(task, 96) {
            self.snapshot.threads.insert(slot, task.to_owned());
            if !self.reported_threads {
                self.reported_threads = true;
                self.emit();
            } else {
                self.tick();
            }
        }
    }

    pub(crate) fn unassign(&mut self, slot: usize) {
        self.snapshot.threads.remove(&slot);
    }

    pub(crate) fn file_done(&mut self) {
        if let Some(completed) = self.snapshot.files_completed.as_mut() {
            *completed += 1;
        }
        self.tick();
    }

    pub(crate) fn advance(&mut self, delta: u64) {
        self.snapshot.completed = self.snapshot.completed.saturating_add(delta);
        self.tick();
    }

    pub(crate) fn set_completed(&mut self, completed: u64) {
        self.snapshot.completed = completed;
        self.tick();
    }

    pub(crate) fn checkpoint(&mut self) {
        self.emit();
    }

    pub(crate) fn finish(mut self) {
        self.snapshot.threads.clear();
        self.emit();
    }

    fn tick(&mut self) {
        if self.last.elapsed() >= INTERVAL {
            self.emit();
        }
    }

    fn emit(&mut self) {
        self.last = Instant::now();
        if let Some(root) = self.root {
            self.snapshot.updated_at_ms = now_ms();
            if write_snapshot(root, &mut self.snapshot).is_err() {
                // No retry loop or log spam when disk/permissions fail.
                self.root = None;
            }
        }
    }
}

fn write_snapshot(root: &Path, snapshot: &mut Snapshot) -> io::Result<()> {
    let mut document = serde_json::to_vec(snapshot)?;
    if document.len() >= 4096 && snapshot.current_file.is_some() {
        // A basename full of JSON escape characters can be longer on the
        // wire. Prefer the active assignments over the legacy file hint.
        let name = snapshot.current_file.take();
        let compact = serde_json::to_vec(snapshot);
        snapshot.current_file = name;
        document = compact?;
    }
    if document.len() >= 4096 {
        return Err(io::Error::other("progress snapshot exceeds size limit"));
    }
    let mut file = tempfile::Builder::new()
        .prefix(".progress-")
        .tempfile_in(root)?;
    file.write_all(&document)?;
    file.write_all(b"\n")?;
    file.persist(root.join("progress.json"))?;
    Ok(())
}

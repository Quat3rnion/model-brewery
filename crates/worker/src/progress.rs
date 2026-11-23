use super::*;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const INTERVAL: Duration = Duration::from_millis(300);

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn read(path: &Path, since: u64) -> Option<ProgressDetail> {
    let fd = cache::no_symlink_open(path, libc::O_PATH).ok()?;
    let meta = fd.metadata().ok()?;
    if !meta.is_file() || meta.len() > PROGRESS_LIMIT || meta.nlink() != 1 {
        return None;
    }
    let file = File::open(format!("/proc/self/fd/{}", fd.as_raw_fd())).ok()?;
    let detail: ProgressDetail = decode_limited(file, PROGRESS_LIMIT).ok()?;
    (detail.valid()
        && detail.updated_at_ms >= since
        && detail.updated_at_ms <= now_ms().saturating_add(5_000))
    .then_some(detail)
}

pub fn hy4_encode(base: &Path, since: u64) -> Option<ProgressDetail> {
    let detail = read(&base.join(hy4::PROGRESS_FILE), since)?;
    let (unit, limit) = match detail.phase.as_str() {
        "hy4_encode_gate" => (ProgressUnit::Items, 8192),
        "hy4_copy_spans" | "hy4_verify_readback" => (ProgressUnit::Bytes, 224_299_813_888),
        _ => return None,
    };
    (detail.unit == unit
        && detail.total.is_some_and(|n| (1..=limit).contains(&n))
        && detail.current_file.is_none()
        && detail.files_total.is_none()
        && detail.files_completed.is_none()
        && detail.input_index.is_none()
        && detail.input_count.is_none()
        && detail.threads.is_empty())
    .then_some(detail)
}

pub fn engine(base: &Path, since: u64) -> Option<ProgressDetail> {
    let detail = read(&base.join("work/progress.json"), since)?;
    matches!(
        detail.phase.as_str(),
        "verify_input"
            | "identify_output"
            | "publish_output"
            | "compare_left"
            | "trace_prompts"
            | "damage_gradients"
            | "damage_swaps"
            | "sensitivity_gradients"
            | "circuit_attribution"
            | "circuit_patches"
            | scratch_eval::CHECKS
            | "locate_edit"
            | "compare_right"
            | "benchmark"
            | "admission"
            | "inspect"
            | "plan"
            | "convert"
            | "verify"
            | "generate"
            | "baseline_questions"
            | "evaluate_questions"
            | "baseline_scoring"
            | "evaluate_scoring"
            | "load_dataset"
            | "prepare_rotations_values"
            | "convert_passes_values"
            | "verify_payloads_values"
            | "verify_pinned_input"
            | "hf_download_resolve"
            | "hf_download_transfer"
            | "hf_download_verify"
    )
    .then_some(detail)
}

pub fn publish(path: &Path, detail: &ProgressDetail) -> Result<()> {
    // Transient data: atomic visibility matters, not crash durability. Use
    // exclusive, per-process temp names so competing RPCs cannot share a file.
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let tmp = path.with_file_name(format!(
        ".progress-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)?;
    let result = (|| -> Result<()> {
        serde_json::to_writer(&mut f, detail)?;
        drop(f);
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

pub struct Reporter<F> {
    emit: F,
    last: Option<Instant>,
    slots: PoolSlots,
    pub detail: ProgressDetail,
}

pub type Notify<'a> = Reporter<&'a mut dyn FnMut(&ProgressDetail) -> Result<()>>;

impl<F: FnMut(&ProgressDetail) -> Result<()>> Reporter<F> {
    pub fn new(emit: F, input: Option<(usize, usize)>) -> Self {
        Self {
            emit,
            last: None,
            slots: PoolSlots::default(),
            detail: ProgressDetail {
                phase: String::new(),
                unit: ProgressUnit::Items,
                completed: 0,
                total: None,
                current_file: None,
                files_completed: None,
                files_total: None,
                input_index: input.map(|(i, _)| i as u64 + 1),
                input_count: input.map(|(_, n)| n as u64),
                updated_at_ms: 0,
                threads: Vec::new(),
            },
        }
    }

    pub fn phase(&mut self, phase: &str, unit: ProgressUnit, total: Option<u64>) -> Result<()> {
        self.reset(phase, unit, total);
        self.tick(true)
    }

    pub fn reset(&mut self, phase: &str, unit: ProgressUnit, total: Option<u64>) {
        self.detail.phase = phase.into();
        self.detail.unit = unit;
        self.detail.completed = 0;
        self.detail.total = total;
        self.detail.current_file = None;
        self.detail.files_completed = None;
        self.detail.files_total = None;
        self.slots = PoolSlots::default();
    }

    pub fn file(&mut self, name: &str) {
        self.detail.current_file = valid_basename(name).then(|| name.to_owned());
    }

    pub fn tick(&mut self, force: bool) -> Result<()> {
        if force || self.last.is_none_or(|last| last.elapsed() >= INTERVAL) {
            self.detail.updated_at_ms = now_ms();
            let mut detail = self.detail.clone();
            detail.threads = self.slots.rows();
            // Leave room for either streaming RPC envelope plus its newline.
            (self.emit)(&bound_progress(detail, 192))?;
            self.last = Some(Instant::now());
        }
        Ok(())
    }

    pub fn slot(&mut self, slot: usize, event: brewery_hash_pool::ProgressEvent, basename: &str) {
        self.slots.update(slot, event, basename);
    }

    pub fn pooled<T>(&mut self, result: Result<T>) -> Result<T> {
        let cleared = self.tick(true);
        result.and_then(|value| cleared.map(|()| value))
    }
}

pub fn apply(reply: &mut Reply, detail: ProgressDetail) {
    let label = match detail.phase.as_str() {
        "inventory_input" => "Inventorying input files",
        "inventory_output" => "Inventorying engine output",
        "hash_input" => "Hashing input",
        "pinned_manifest" => "Checking pinned manifest",
        "plan_transfer" => "Checking target cache and disk space",
        "copy_input" => "Copying bundle",
        "verify_cache" => "Verifying cached bundle",
        "verify_staged" => "Verifying staged bundle checksums",
        "verify_output" => "Verifying engine output",
        "trace_prompts" => "Capturing and checking prompts",
        "damage_gradients" => "Measuring quantization damage",
        "damage_swaps" => "Checking single-tensor swaps",
        "sensitivity_gradients" => "Measuring and checking gradients",
        "circuit_attribution" => "Measuring activation attribution",
        "circuit_patches" => "Checking the top activation patches",
        scratch_eval::CHECKS => "Checking scratch model behavior",
        "locate_edit" => "Computing locate-then-edit factors",
        "hf_download_transfer" => "Downloading complete files",
        "hf_download_verify" => "Checking download checksums",
        "hy4_encode_gate" => "Encoding Hy4 gate",
        "hy4_copy_spans" => "Writing Hy4 artifact",
        "hy4_verify_readback" => "Verifying Hy4 readback and checksums",
        _ => {
            reply.progress_detail = Some(detail);
            return;
        }
    };
    let units = match detail.unit {
        ProgressUnit::Bytes => "bytes",
        ProgressUnit::Files => "files",
        ProgressUnit::Items if detail.phase == "hy4_encode_gate" => "rows",
        _ => "items",
    };
    reply.progress = match detail.total {
        Some(n) if detail.phase == "copy_input" => {
            format!("{label}: {} / {n} bytes staged", detail.completed)
        }
        Some(n) => format!("{label}: {} / {n} {units}", detail.completed),
        None => format!("{label}: {} {units}", detail.completed),
    };
    reply.progress_detail = Some(detail);
}

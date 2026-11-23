//! Admission settings shared by the controller and the worker. These are
//! launch limits and free-space checks, not ongoing memory or disk quotas.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path, process::Command};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_threads: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_free_ram_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_reserve_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Range {
    pub min: u64,
    pub max: u64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<Range>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_threads: Option<Range>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_free_ram_bytes: Option<Range>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_reserve_bytes: Option<Range>,
}

impl Bounds {
    pub const MAX: Self = Self {
        concurrency: Some(Range { min: 1, max: 16 }),
        cpu_threads: Some(Range { min: 1, max: 64 }),
        min_free_ram_bytes: Some(Range {
            min: 0,
            max: 16 << 40,
        }),
        disk_reserve_bytes: Some(Range {
            min: 0,
            max: 1 << 50,
        }),
    };

    pub fn standard(pool: bool) -> Self {
        Self {
            concurrency: pool.then_some(Range { min: 1, max: 1 }),
            cpu_threads: Some(Range { min: 1, max: 4 }),
            ..Self::MAX
        }
    }

    pub fn check_ranges(&self) -> Result<()> {
        for (range, absolute) in [
            (self.concurrency, Self::MAX.concurrency),
            (self.cpu_threads, Self::MAX.cpu_threads),
            (self.min_free_ram_bytes, Self::MAX.min_free_ram_bytes),
            (self.disk_reserve_bytes, Self::MAX.disk_reserve_bytes),
        ] {
            if let Some(range) = range {
                let absolute = absolute.expect("global bound");
                ensure!(
                    absolute.min <= range.min
                        && range.min <= range.max
                        && range.max <= absolute.max,
                    "invalid resource bounds"
                );
            }
        }
        Ok(())
    }

    pub fn validate(&self, limits: &Limits) -> Result<()> {
        self.check_ranges()?;
        for (name, value, bound) in [
            ("concurrency", limits.concurrency, self.concurrency),
            ("cpu_threads", limits.cpu_threads, self.cpu_threads),
            (
                "min_free_ram_bytes",
                limits.min_free_ram_bytes,
                self.min_free_ram_bytes,
            ),
            (
                "disk_reserve_bytes",
                limits.disk_reserve_bytes,
                self.disk_reserve_bytes,
            ),
        ] {
            if let Some(value) = value {
                let Some(range) = bound else {
                    anyhow::bail!("unsupported resource control: {name}");
                };
                ensure!(
                    (range.min..=range.max).contains(&value),
                    "{name} must be between {} and {}",
                    range.min,
                    range.max
                );
            }
        }
        Ok(())
    }
}

impl Limits {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    pub fn for_job(self, kind: &str) -> Self {
        if engine_job(kind) {
            Self {
                concurrency: self.concurrency.filter(|_| kind == "convert"),
                ..self
            }
        } else if matches!(kind, "transfer" | "transfer_preview") {
            Self {
                disk_reserve_bytes: self.disk_reserve_bytes,
                ..Self::default()
            }
        } else {
            Self::default()
        }
    }
}

pub fn engine_job(kind: &str) -> bool {
    matches!(
        kind,
        "inspect"
            | "plan"
            | "convert"
            | "verify"
            | "generate"
            | "compare"
            | crate::trace::JOB
            | "evaluate"
            | "baseline"
            | "benchmark"
            | "weight_inspect"
            | "weight_propose"
            | "weight_tune"
            | "weight_apply"
            | crate::exl3_layer::JOB
    )
}

pub fn ram_check(minimum: u64) -> Result<()> {
    if minimum != 0 {
        let meminfo = fs::read_to_string("/proc/meminfo")?;
        let available = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemAvailable:"))
            .and_then(|line| line.split_ascii_whitespace().next())
            .context("cannot determine worker MemAvailable")?
            .parse::<u64>()?
            .saturating_mul(1024);
        ensure!(available >= minimum, "Worker minimum free RAM not met");
    }
    Ok(())
}

pub fn disk_check(path: &Path, minimum: u64) -> Result<()> {
    if minimum != 0 {
        ensure!(
            crate::available(path)? >= minimum,
            "Worker disk reserve not met"
        );
    }
    Ok(())
}

// Ordinary worker descendants inherit the selected subset of the existing
// Linux CPU mask. Never widen the host/cgroup's affinity.
pub fn cap_cpu(command: &mut Command, requested: u64, operator: Option<&str>) -> Result<u64> {
    use std::os::unix::process::CommandExt;
    let configured = operator.map(str::parse::<u64>).transpose()?.unwrap_or(4);
    ensure!((1..=64).contains(&configured), "invalid worker CPU ceiling");
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of_val(&mask);
    if unsafe { libc::sched_getaffinity(0, size, &mut mask) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let available = std::thread::available_parallelism().map_or(1, usize::from) as u64;
    let limit = requested.min(configured).min(available);
    let mut selected = 0;
    for cpu in 0..libc::CPU_SETSIZE as usize {
        if unsafe { libc::CPU_ISSET(cpu, &mask) } {
            if selected < limit {
                selected += 1;
            } else {
                unsafe { libc::CPU_CLR(cpu, &mut mask) };
            }
        }
    }
    ensure!(selected > 0, "no CPU available for worker launch");
    for key in brewery_hash_pool::CPU_ENV {
        command.env(key, selected.to_string());
    }
    unsafe {
        command.pre_exec(move || {
            if libc::sched_setaffinity(0, size, &mask) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
    }
    Ok(selected)
}

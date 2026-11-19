//! Run independent file hashes concurrently. Only the caller publishes progress.
use anyhow::{Context, Result, anyhow, ensure};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

pub const CPU_ENV: &[&str] = &[
    "BREWERY_CPU_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
];

fn configured(name: &str, max: usize) -> Result<Option<usize>> {
    let n = match std::env::var(name) {
        Ok(s) => s
            .parse::<usize>()
            .with_context(|| format!("invalid {name}"))?,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    ensure!((1..=max).contains(&n), "{name} must be 1..{max}");
    Ok(Some(n))
}

fn available() -> usize {
    thread::available_parallelism().map_or(1, usize::from)
}

pub fn cpu_threads() -> Result<usize> {
    Ok(available().min(configured("BREWERY_CPU_THREADS", 64)?.unwrap_or(4)))
}

pub fn map_cpu<I: Sync, O: Send>(
    items: &[I],
    work: impl Fn(usize, &I, &dyn Fn(u64) -> Result<()>) -> Result<O> + Sync,
    progress: impl FnMut(usize, u64, bool) -> Result<()>,
) -> Result<Vec<O>> {
    map_cpu_observed(items, work, legacy(progress))
}

pub fn map<I: Sync, O: Send>(
    items: &[I],
    work: impl Fn(usize, &I, &dyn Fn(u64) -> Result<()>) -> Result<O> + Sync,
    progress: impl FnMut(usize, u64, bool) -> Result<()>,
) -> Result<Vec<O>> {
    map_observed(items, work, legacy(progress))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressEvent {
    Started,
    Advanced(u64),
    Finished,
    Stopped,
}

pub fn map_cpu_observed<I: Sync, O: Send>(
    items: &[I],
    work: impl Fn(usize, &I, &dyn Fn(u64) -> Result<()>) -> Result<O> + Sync,
    progress: impl FnMut(usize, usize, ProgressEvent) -> Result<()>,
) -> Result<Vec<O>> {
    map_limited(cpu_threads()?, items, work, progress)
}

/// Reports actual 1-based pool slots on the caller thread. An observed start
/// is followed by Finished on success or Stopped before an error returns.
pub fn map_observed<I: Sync, O: Send>(
    items: &[I],
    work: impl Fn(usize, &I, &dyn Fn(u64) -> Result<()>) -> Result<O> + Sync,
    progress: impl FnMut(usize, usize, ProgressEvent) -> Result<()>,
) -> Result<Vec<O>> {
    let limit = configured("BREWERY_CPU_THREADS", 64)?;
    let hashes = configured("BREWERY_HASH_THREADS", 16)?
        .unwrap_or(limit.unwrap_or(4))
        .min(16);
    map_limited(
        hashes.min(limit.unwrap_or(16)).min(available()),
        items,
        work,
        progress,
    )
}

fn legacy(
    mut progress: impl FnMut(usize, u64, bool) -> Result<()>,
) -> impl FnMut(usize, usize, ProgressEvent) -> Result<()> {
    move |_, index, event| match event {
        ProgressEvent::Advanced(n) => progress(index, n, false),
        ProgressEvent::Finished => progress(index, 0, true),
        _ => Ok(()),
    }
}

fn map_limited<I: Sync, O: Send>(
    limit: usize,
    items: &[I],
    work: impl Fn(usize, &I, &dyn Fn(u64) -> Result<()>) -> Result<O> + Sync,
    mut progress: impl FnMut(usize, usize, ProgressEvent) -> Result<()>,
) -> Result<Vec<O>> {
    let threads = items.len().min(limit);
    if threads <= 1 {
        return items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let result = (|| {
                    progress(1, i, ProgressEvent::Started)?;
                    let meter = std::cell::RefCell::new(&mut progress);
                    let value = work(i, item, &|n| {
                        (*meter.borrow_mut())(1, i, ProgressEvent::Advanced(n))
                    })?;
                    progress(1, i, ProgressEvent::Finished)?;
                    Ok(value)
                })();
                if result.is_err() {
                    let _ = progress(1, i, ProgressEvent::Stopped);
                }
                result
            })
            .collect();
    }
    enum Event<T> {
        Progress(usize, usize, ProgressEvent),
        Done(usize, usize, Result<T>),
    }
    let next = AtomicUsize::new(0);
    let mut active = vec![None; threads + 1];
    let result: Result<Vec<O>> = thread::scope(|scope| {
        let (send, recv) = mpsc::sync_channel(threads * 2);
        for slot in 1..=threads {
            let (send, next, work) = (send.clone(), &next, &work);
            thread::Builder::new()
                .name("brewery-sha256".into())
                .spawn_scoped(scope, move || {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        if send
                            .send(Event::Progress(slot, i, ProgressEvent::Started))
                            .is_err()
                        {
                            break;
                        }
                        let value = work(i, item, &|n| {
                            send.send(Event::Progress(slot, i, ProgressEvent::Advanced(n)))
                                .map_err(|_| anyhow!("hash cancelled"))
                        });
                        let failed = value.is_err();
                        if send.send(Event::Done(slot, i, value)).is_err() || failed {
                            break;
                        }
                    }
                })?;
        }
        drop(send);
        let mut results: Vec<_> = (0..items.len()).map(|_| None).collect();
        // Dropping recv on error wakes blocked senders; scope joins all readers.
        for event in recv {
            match event {
                Event::Progress(slot, i, event) => {
                    if event == ProgressEvent::Started {
                        active[slot] = Some(i);
                    }
                    progress(slot, i, event)?;
                }
                Event::Done(slot, i, result) => {
                    results[i] = Some(result?);
                    progress(slot, i, ProgressEvent::Finished)?;
                    active[slot] = None;
                }
            }
        }
        results
            .into_iter()
            .map(|r| r.context("hash worker did not finish"))
            .collect()
    });
    if result.is_err() {
        for (slot, index) in active.into_iter().enumerate() {
            if let Some(index) = index {
                let _ = progress(slot, index, ProgressEvent::Stopped);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_slots_stay_with_real_threads_and_clear_on_error() {
        let barrier = std::sync::Barrier::new(2);
        let mut active = std::collections::BTreeMap::new();
        let mut ids = std::collections::BTreeMap::new();
        let workers = std::sync::Mutex::new([None; 6]);
        let error = map_limited(
            2,
            &[(); 6],
            |i, _, tick| {
                workers.lock().unwrap()[i] = Some(thread::current().id());
                if i < 2 {
                    barrier.wait();
                }
                tick(1)?;
                ensure!(i != 3, "fixture failure");
                Ok(())
            },
            |slot, i, event| {
                assert!((1..=2).contains(&slot));
                match event {
                    ProgressEvent::Started => assert!(active.insert(slot, i).is_none()),
                    ProgressEvent::Advanced(_) => {
                        assert_eq!(active[&slot], i);
                        let id = workers.lock().unwrap()[i].unwrap();
                        assert_eq!(*ids.entry(slot).or_insert(id), id);
                    }
                    ProgressEvent::Finished | ProgressEvent::Stopped => {
                        assert_eq!(active.remove(&slot), Some(i));
                    }
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "fixture failure");
        assert!(active.is_empty());
        let workers = workers.lock().unwrap();
        assert_ne!(workers[0], workers[1]);
    }

    #[test]
    fn cancellation_releases_readers_and_preserves_reason() {
        let err = map(
            &[(); 5],
            |_, _, tick| {
                for _ in 0..32 {
                    tick(1)?;
                }
                Ok(())
            },
            |_, _, _| Err(anyhow!("operator cancelled")),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "operator cancelled");
    }

    #[test]
    fn read_error_prevents_returning_partial_manifest() {
        let err = map(
            &[0, 1, 2],
            |i, _, tick| {
                tick(1)?;
                ensure!(i != 1, "source changed");
                Ok(i)
            },
            |_, _, _| Ok(()),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "source changed");
    }
}

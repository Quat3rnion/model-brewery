// Measured throughput from timestamped samples of one progress counter.
//
// A rate is the difference between two real samples of the same counter,
// divided by the difference of their producer timestamps. Producer clock
// skew does not matter because only differences of the same clock are used.
// The rate itself is not extrapolated. The only projection is the phase
// time left (phaseEta), which is shown only for a known total and a recent,
// gap-free, positive measured rate; see the conditions there. When the samples are not sufficient, the view says so and
// the counter itself stays visible.

import type { ProgressDetail } from './progress';

/** Samples older than this, relative to the newest sample, leave the window. */
export const RATE_WINDOW_MS = 15_000;
/** Two samples closer than this in producer time give no rate yet. */
export const RATE_MIN_SPAN_MS = 1_000;
/** Without a new sample for this long (browser clock), the last rate is hidden. */
export const RATE_IDLE_MS = 20_000;
/**
 * A series not fed for this long (browser clock) restarts at the next sample:
 * the tab was suspended or the connection was lost, so the samples around the
 * gap are not comparable. Normal hidden-tab polling is every 10 to 15 s.
 */
export const RATE_SUSPEND_MS = 30_000;
/** Phase time left needs this many samples over at least this span. */
export const ETA_MIN_SAMPLES = 3;
export const ETA_MIN_SPAN_MS = 5_000;
/** Phase time left is not shown when two neighbouring samples are further apart. */
export const ETA_MAX_GAP_MS = 10_000;
const MAX_SAMPLES = 64;

/** Where a sample came from. Different sources never share a series. */
export type RateSource = 'controller' | 'live' | 'legacy';

interface Sample {
  /** Producer time, Unix milliseconds. */
  t: number;
  /** Counter value. */
  v: number;
}

interface Series {
  key: string;
  samples: Sample[];
  filesCompleted: number | null;
  /** Browser time when the newest sample arrived. */
  receivedAt: number;
  /** Browser time when the series was last fed, also with a duplicate sample. */
  seenAt: number;
}

export type RateView =
  | {
      kind: 'rate';
      perSecond: number;
      spanMs: number;
      samples: number;
      estimated: boolean;
      /** Largest producer-time gap between neighbouring samples in the window. */
      maxGapMs: number;
      /** Value and producer time of the newest sample. */
      latest: number;
      latestAt: number;
    }
  | { kind: 'warming'; samples: number }
  | { kind: 'idle'; idleMs: number };

/**
 * The identity of a counter. A change in any of these fields starts a new
 * series: a new job, source, phase, unit, total, input or file set is a
 * different counter. The current file is not part of it: the workers report
 * aggregate counters over all files, so a shard change is not a reset. A
 * per-file counter resets because its value goes down (see observe).
 */
export function counterKey(jobId: string, source: RateSource, d: ProgressDetail): string {
  return JSON.stringify([jobId, source, d.phase, d.unit, d.total, d.input_index, d.input_count, d.files_total, d.estimated]);
}

export class RateTracker {
  #series = new Map<string, Series>();

  /**
   * Record one sample. `t` is the producer timestamp of the measurement.
   * Returns true when the stored series changed. Duplicates (same or older
   * timestamp) are ignored, so several views or polls can feed the same
   * sample safely.
   */
  observe(jobId: string, source: RateSource, d: ProgressDetail, t: number | null, receivedAt: number): boolean {
    if (t === null || !Number.isFinite(t) || !Number.isFinite(d.completed)) return false;
    const id = `${jobId}\u0000${source}`;
    const key = counterKey(jobId, source, d);
    const s = this.#series.get(id);
    const fresh = (): true => {
      this.#series.set(id, { key, samples: [{ t, v: d.completed }], filesCompleted: d.files_completed, receivedAt, seenAt: receivedAt });
      return true;
    };
    if (!s || s.key !== key) return fresh();
    const suspended = receivedAt - s.seenAt > RATE_SUSPEND_MS;
    s.seenAt = Math.max(s.seenAt, receivedAt);
    const last = s.samples[s.samples.length - 1];
    if (t <= last.t) return false; // duplicate or out of order
    if (suspended) return fresh(); // do not average across a suspended tab
    const filesBack = d.files_completed !== null && s.filesCompleted !== null && d.files_completed < s.filesCompleted;
    if (d.completed < last.v || filesBack || (s.filesCompleted !== null && d.files_completed === null)) return fresh(); // counter reset
    s.samples.push({ t, v: d.completed });
    s.filesCompleted = d.files_completed;
    s.receivedAt = receivedAt;
    // Keep the newest two samples always; drop the rest once outside the window.
    while (s.samples.length > 2 && (s.samples.length > MAX_SAMPLES || s.samples[0].t < t - RATE_WINDOW_MS)) s.samples.shift();
    return true;
  }

  /** The rate for the counter that `d` shows, or null when no series matches it. */
  view(jobId: string, source: RateSource, d: ProgressDetail, now: number): RateView | null {
    const s = this.#series.get(`${jobId}\u0000${source}`);
    if (!s || s.key !== counterKey(jobId, source, d)) return null;
    const idle = now - s.receivedAt;
    if (idle > RATE_IDLE_MS) return { kind: 'idle', idleMs: idle };
    const first = s.samples[0];
    const last = s.samples[s.samples.length - 1];
    const span = last.t - first.t;
    if (s.samples.length < 2 || span < RATE_MIN_SPAN_MS) return { kind: 'warming', samples: s.samples.length };
    let maxGapMs = 0;
    for (let i = 1; i < s.samples.length; i++) maxGapMs = Math.max(maxGapMs, s.samples[i].t - s.samples[i - 1].t);
    return {
      kind: 'rate',
      perSecond: ((last.v - first.v) * 1000) / span,
      spanMs: span,
      samples: s.samples.length,
      estimated: d.estimated,
      maxGapMs,
      latest: last.v,
      latestAt: last.t,
    };
  }

  /** Drop every series of jobs not in `keep` (terminal, cancelled, removed). Returns true when something was dropped. */
  retain(keep: ReadonlySet<string>): boolean {
    let changed = false;
    for (const id of [...this.#series.keys()]) {
      if (!keep.has(id.slice(0, id.indexOf('\u0000')))) {
        this.#series.delete(id);
        changed = true;
      }
    }
    return changed;
  }

  /** Drop one series, for example when its source no longer applies. */
  forget(jobId: string, source: RateSource): boolean {
    return this.#series.delete(`${jobId}\u0000${source}`);
  }
}

function sig(n: number): string {
  return n >= 100 ? Math.round(n).toLocaleString('en-US') : n >= 10 ? n.toFixed(1) : n.toFixed(2);
}

/** Decimal bytes per second: kB/s, MB/s, GB/s (1 MB = 1,000,000 bytes). */
export function formatByteRate(perSecond: number): string {
  const units = ['B/s', 'kB/s', 'MB/s', 'GB/s', 'TB/s'];
  let v = perSecond;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i += 1;
  }
  return i === 0 ? `${Math.round(v)} B/s` : `${sig(v)} ${units[i]}`;
}

/** Counts per second; below 1 per second, per minute so slow work stays readable. */
export function formatCountRate(perSecond: number, unit: string): string {
  if (perSecond === 0) return `0 ${unit}/s`;
  return perSecond >= 1 ? `${sig(perSecond)} ${unit}/s` : `${sig(perSecond * 60)} ${unit}/min`;
}

function seconds(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}

/** One line for a rate view. `unit` is the display unit of the counter. */
export function rateText(view: RateView | null, unit: string): string {
  if (!view) return 'Rate: measuring';
  switch (view.kind) {
    case 'warming':
      return view.samples < 2 ? 'Rate: measuring, needs a second sample' : 'Rate: measuring';
    case 'idle':
      return `Rate: no new measurement for ${seconds(view.idleMs)}`;
    case 'rate': {
      const r = unit === 'bytes' ? formatByteRate(view.perSecond) : formatCountRate(view.perSecond, unit);
      return `Rate: ${r}${view.estimated ? ' (estimated counter)' : ''}, measured over ${seconds(view.spanMs)}`;
    }
  }
}

/**
 * Time left in the current phase, in milliseconds, or null. It is the known
 * remaining count divided by the measured rate, as of the newest sample.
 * Null when: no total, an estimated counter, nothing left, a rate that is
 * not positive, fewer than ETA_MIN_SAMPLES samples or less than
 * ETA_MIN_SPAN_MS of them, a producer gap over ETA_MAX_GAP_MS in the window,
 * or a shown detail that is not the newest tracked sample. Job state and
 * freshness are checked by the caller (progress.ts phaseEtaFor).
 */
export function phaseEta(view: RateView | null, d: ProgressDetail): number | null {
  if (!view || view.kind !== 'rate' || d.total === null || d.estimated || view.estimated) return null;
  if (!(view.perSecond > 0) || view.samples < ETA_MIN_SAMPLES || view.spanMs < ETA_MIN_SPAN_MS || view.maxGapMs > ETA_MAX_GAP_MS) return null;
  if (view.latest !== d.completed || (d.updated_at_ms !== null && view.latestAt !== d.updated_at_ms)) return null;
  const left = d.total - d.completed;
  if (!(left > 0)) return null;
  const ms = (left / view.perSecond) * 1000;
  return Number.isFinite(ms) ? ms : null;
}

/** Rounded, readable duration. Precision drops as the duration grows. */
export function formatDuration(ms: number): string {
  const min = ms / 60_000;
  if (min < 1) return 'less than 1 min';
  if (min < 10) return `about ${Math.round(min)} min`;
  if (min < 57.5) return `about ${Math.round(min / 5) * 5} min`;
  const h = min / 60;
  if (h < 10) {
    const m5 = Math.round(min / 5) * 5;
    const hh = Math.floor(m5 / 60);
    const mm = m5 % 60;
    return mm ? `about ${hh} h ${mm} min` : `about ${hh} h`;
  }
  if (h < 48) return `about ${Math.round(h)} h`;
  const days = h / 24;
  return days <= 7 ? `about ${Math.round(days)} days` : 'more than 7 days';
}

/** Phase time-left text. It names the phase so it is not read as the whole job. */
export function etaText(ms: number | null, compact: boolean): string | null {
  if (ms === null) return null;
  return compact ? `${formatDuration(ms)} left in this phase` : `Phase time left: ${formatDuration(ms)} (this phase only, at the measured rate)`;
}

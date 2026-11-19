// Worker path completion for the register form.
// POST /api/suggestions/paths {host, prefix} -> {items: [{path, kind}], truncated}
// Suggestions only help typing. A manual absolute path stays valid when the
// worker is offline or the controller has no suggestion endpoint.

import { ApiError, errorText } from './api';

export interface PathItem {
  path: string;
  kind: 'file' | 'directory';
}

export interface PathState {
  status: 'idle' | 'loading' | 'ready' | 'error' | 'unavailable';
  host: string;
  prefix: string;
  items: PathItem[];
  truncated: boolean;
  message: string | null;
}

export const MAX_ITEMS = 200;
export const DEBOUNCE_MS = 180;

export const IDLE: PathState = { status: 'idle', host: '', prefix: '', items: [], truncated: false, message: null };

export function parseSuggestions(data: unknown): { items: PathItem[]; truncated: boolean } | null {
  if (!data || typeof data !== 'object' || !Array.isArray((data as { items?: unknown }).items)) return null;
  const raw = (data as { items: unknown[] }).items;
  const seen = new Set<string>();
  const items: PathItem[] = [];
  let capped = false;
  for (const x of raw) {
    if (items.length >= MAX_ITEMS) {
      capped = true;
      break;
    }
    if (!x || typeof x !== 'object') continue;
    const { path, kind } = x as { path?: unknown; kind?: unknown };
    if (typeof path !== 'string' || !path.startsWith('/') || path.length > 4096 || seen.has(path)) continue;
    if (kind !== 'file' && kind !== 'directory') continue;
    seen.add(path);
    items.push({ path, kind });
  }
  return { items, truncated: (data as { truncated?: unknown }).truncated === true || capped };
}

/** Text to put in the input when an item is chosen. Directories end with '/' so the next level can be listed. */
export function completionText(item: PathItem): string {
  return item.kind === 'directory' && !item.path.endsWith('/') ? `${item.path}/` : item.path;
}

/** Prefix to ask for, or null when the text cannot be completed (relative path). Empty text asks for roots. */
export function prefixFor(text: string): string | null {
  if (text === '') return '';
  return text.startsWith('/') ? text : null;
}

export type PathFetcher = (host: string, prefix: string, signal: AbortSignal) => Promise<unknown>;

/**
 * Debounced, abortable suggestion requests. Only the response for the
 * current host and prefix is applied; an older response is dropped.
 */
export class PathSuggester {
  state: PathState = IDLE;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #ctrl: AbortController | null = null;
  #seq = 0;

  constructor(
    private fetcher: PathFetcher,
    private onChange: (s: PathState) => void,
    private delay = DEBOUNCE_MS,
  ) {}

  #set(s: PathState) {
    this.state = s;
    this.onChange(s);
  }

  request(host: string, text: string): void {
    this.cancel();
    const prefix = prefixFor(text);
    if (!host || prefix === null) {
      this.#set({ ...IDLE, host, prefix: text, message: host ? 'Suggestions need an absolute path that starts with /.' : null });
      return;
    }
    const seq = ++this.#seq;
    this.#set({ status: 'loading', host, prefix, items: this.state.host === host ? this.state.items : [], truncated: false, message: null });
    this.#timer = setTimeout(() => void this.#run(seq, host, prefix), this.delay);
  }

  async #run(seq: number, host: string, prefix: string) {
    this.#timer = null;
    const ctrl = new AbortController();
    this.#ctrl = ctrl;
    try {
      const data = await this.fetcher(host, prefix, ctrl.signal);
      if (seq !== this.#seq) return;
      const parsed = parseSuggestions(data);
      if (!parsed) this.#set({ ...IDLE, status: 'error', host, prefix, message: 'The controller sent suggestions in an unknown shape.' });
      else this.#set({ status: 'ready', host, prefix, items: parsed.items, truncated: parsed.truncated, message: null });
    } catch (e) {
      if (seq !== this.#seq || ctrl.signal.aborted) return;
      const unavailable = e instanceof ApiError && (e.status === 404 || e.status === 405);
      this.#set({
        ...IDLE,
        status: unavailable ? 'unavailable' : 'error',
        host,
        prefix,
        message: unavailable ? 'Path suggestions are not available from this controller.' : `No suggestions: ${errorText(e)}`,
      });
    } finally {
      if (this.#ctrl === ctrl) this.#ctrl = null;
    }
  }

  /** Stop the pending and running request. Late responses are ignored. */
  cancel(): void {
    this.#seq += 1;
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    this.#ctrl?.abort();
    this.#ctrl = null;
  }
}

export type View = 'library' | 'queue' | 'transfers' | 'compare' | 'convert' | 'evaluate' | 'benchmarks' | 'calibration' | 'settings' | 'profiles' | 'weights' | 'job' | 'not-found';

export const VIEWS: { id: Exclude<View, 'job' | 'not-found'>; label: string }[] = [
  { id: 'library', label: 'Library' },
  { id: 'queue', label: 'Jobs' },
  { id: 'transfers', label: 'Transfers' },
  { id: 'compare', label: 'Compare' },
  { id: 'convert', label: 'Convert' },
  { id: 'evaluate', label: 'Evaluate' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'settings', label: 'Settings' },
];

/** Query value from a hidden route such as #/weights?source=<id>. */
export function hashParam(hash: string, name: string): string | null {
  const q = hash.indexOf('?');
  return q < 0 ? null : new URLSearchParams(hash.slice(q + 1)).get(name);
}

export function parse(hash: string): { view: View; jobId: string | null } {
  const path = hash.replace(/^#\/?/, '').split('?')[0];
  const job = /^(?:jobs|job)\/([^/]+)$/.exec(path);
  if (job) {
    try {
      return { view: 'job', jobId: decodeURIComponent(job[1]) };
    } catch {
      return { view: 'not-found', jobId: null };
    }
  }
  if (path === 'admin' || path === 'settings/infrastructure' || path === 'settings/evaluation') {
    return { view: 'settings', jobId: null };
  }
  if (path === 'profiles' || path === 'weights') return { view: path, jobId: null };
  const found = VIEWS.find((v) => v.id === path);
  return { view: found?.id ?? (path ? 'not-found' : 'library'), jobId: null };
}

class Route {
  hash = $state('');
  view = $state<View>('library');
  jobId = $state<string | null>(null);
  #guards = new Set<() => boolean>();

  guardHashChange(allow: () => boolean) {
    this.#guards.add(allow);
    return () => { this.#guards.delete(allow); };
  }

  constructor() {
    if (typeof window === 'undefined') return;
    const entryIndex = (): number | undefined => history.state?.breweryRouteIndex;
    let index = entryIndex() ?? 0;
    const setEntry = (entry: number, url?: string) => history.replaceState({ ...history.state, breweryRouteIndex: entry }, '', url);
    let restoring = false;
    const apply = () => {
      setEntry(index, window.location.hash === '#/admin' ? '#/settings/infrastructure' : undefined);
      this.hash = window.location.hash;
      const r = parse(this.hash);
      this.view = r.view;
      this.jobId = r.jobId;
    };
    apply();
    window.addEventListener('hashchange', (event) => {
      if (restoring && this.hash === window.location.hash) {
        restoring = false;
        event.stopImmediatePropagation();
        return;
      }
      if (this.hash === window.location.hash) return;
      const target = entryIndex() ?? index + 1;
      if ([...this.#guards].some((allow) => !allow())) {
        event.stopImmediatePropagation();
        // Remember the new entry and return to the accepted one. Back and
        // Forward still work if the user changes their mind.
        setEntry(target);
        restoring = true;
        history.go(index - target);
        return;
      }
      index = target;
      apply();
    }, { capture: true });
  }
}

export const route = new Route();

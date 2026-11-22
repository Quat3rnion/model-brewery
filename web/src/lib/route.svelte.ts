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

/** An async result may update its record, but not take over a later visit or action. */
export class RouteActivity {
  #visit = 0;
  #action = 0;

  visit() { this.#visit++; }
  interact() { this.#action++; }

  capture(matchesLocation = () => true) {
    const visit = this.#visit;
    const action = this.#action;
    const current = () => this.#visit === visit && matchesLocation();
    return { current, canPresent: () => current() && this.#action === action };
  }
}

class Route {
  hash = $state('');
  entry = $state('');
  view = $state<View>('library');
  jobId = $state<string | null>(null);
  #guards = new Set<() => boolean>();
  #activity = new RouteActivity();

  capture() {
    const { entry } = this;
    // The URL can change before the browser delivers hashchange/popstate.
    return this.#activity.capture(() => typeof window === 'undefined' ||
      (location.hash === this.hash && history.state?.breweryRouteEntry === entry));
  }

  guardHashChange(allow: () => boolean) {
    this.#guards.add(allow);
    return () => { this.#guards.delete(allow); };
  }

  /** Update an accepted query selection without adding a history entry. */
  replaceQuery(hash: string) {
    if (hash !== this.hash) this.#activity.interact();
    history.replaceState(history.state, '', hash);
    this.hash = hash;
  }

  constructor() {
    if (typeof window === 'undefined') return;
    const entryIndex = (): number | undefined => history.state?.breweryRouteIndex;
    let index = entryIndex() ?? 0;
    const entryId = (): string | undefined => history.state?.breweryRouteEntry;
    const setEntry = (entry: number, url?: string, id = entryId() ?? crypto.getRandomValues(new Uint32Array(4)).join('-')) => {
      history.replaceState({ ...history.state, breweryRouteIndex: entry, breweryRouteEntry: id }, '', url);
      return id;
    };
    let restoring = false;
    const apply = () => {
      this.#activity.visit();
      this.entry = setEntry(index, window.location.hash === '#/admin' ? '#/settings/infrastructure' : undefined);
      this.hash = window.location.hash;
      const r = parse(this.hash);
      this.view = r.view;
      this.jobId = r.jobId;
    };
    apply();
    for (const type of ['pointerdown', 'keydown', 'click', 'input', 'wheel']) {
      window.addEventListener(type, () => this.#activity.interact(), { capture: true, passive: true });
    }
    window.addEventListener('popstate', () => {
      // Back or Forward can move between entries with identical hashes.
      // Different hashes are handled once, by hashchange below.
      if (this.hash !== window.location.hash) return;
      if (entryId() === this.entry) {
        // The hash need not change when returning from a blocked popstate.
        if (restoring) restoring = false;
        return;
      }
      if (accept()) apply();
    });
    const accept = () => {
      if (restoring) return false;
      const target = entryIndex() ?? index + 1;
      if ([...this.#guards].some((allow) => !allow())) {
        const delta = index - target;
        if (!delta) {
          setEntry(index, this.hash || `${location.pathname}${location.search}`, this.entry);
          return false;
        }
        // Remember the new entry and return to the accepted one. Back and
        // Forward still work if the user changes their mind.
        setEntry(target);
        restoring = true;
        history.go(delta);
        return false;
      }
      index = target;
      return true;
    };
    window.addEventListener('hashchange', (event) => {
      if (restoring && this.hash === window.location.hash) {
        restoring = false;
        event.stopImmediatePropagation();
        return;
      }
      if (this.hash === window.location.hash) return;
      if (accept()) apply();
      else event.stopImmediatePropagation();
    }, { capture: true });
  }
}

export const route = new Route();

export type View = 'library' | 'queue' | 'transfers' | 'compare' | 'convert' | 'evaluate' | 'benchmarks' | 'calibration' | 'settings' | 'profiles' | 'weights' | 'job';

export const VIEWS: { id: Exclude<View, 'job'>; label: string }[] = [
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
  const parts = hash.replace(/^#\/?/, '').split('/');
  if (parts[0] === 'jobs' && parts[1]) return { view: 'job', jobId: decodeURIComponent(parts[1]) };
  if (parts[0] === 'admin') return { view: 'settings', jobId: null };
  if (parts[0].split('?')[0] === 'profiles') return { view: 'profiles', jobId: null };
  if (parts[0].split('?')[0] === 'weights') return { view: 'weights', jobId: null };
  if (parts[0].split('?')[0] === 'library') return { view: 'library', jobId: null };
  const found = VIEWS.find((v) => v.id === parts[0]);
  return { view: found?.id ?? 'library', jobId: null };
}

class Route {
  view = $state<View>('library');
  jobId = $state<string | null>(null);

  constructor() {
    if (typeof window === 'undefined') return;
    const apply = () => {
      if (window.location.hash === '#/admin') history.replaceState(null, '', '#/settings/infrastructure');
      const r = parse(window.location.hash);
      this.view = r.view;
      this.jobId = r.jobId;
    };
    apply();
    window.addEventListener('hashchange', apply);
  }
}

export const route = new Route();

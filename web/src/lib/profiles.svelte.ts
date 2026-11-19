// Catalog and saved revisions, loaded on demand by Profiles and Convert.
// Read-only GETs; the only write is an explicit Save in the editor.
import { api, ApiError, errorText } from './api';
import { parseCatalog, parseGgufProfiles, parseProfiles, type Catalog, type SavedGgufProfile, type SavedProfile } from './profiles';
import { parseGgufCatalog, type GgufCatalog } from './gguf';

class ProfileStore {
  catalog = $state<Catalog | null>(null);
  saved = $state<SavedProfile[]>([]);
  /** GGUF catalog and revisions; null catalog with a reason when the controller has none. */
  gguf = $state<GgufCatalog | null>(null);
  ggufSaved = $state<SavedGgufProfile[]>([]);
  ggufStatus = $state<'idle' | 'ok' | 'unavailable' | 'error'>('idle');
  status = $state<'idle' | 'loading' | 'ok' | 'unavailable' | 'error'>('idle');
  error = $state<string | null>(null);
  /** Revision chosen with "Use in Convert"; Convert picks it up once. */
  pendingUse = $state<string | null>(null);

  async load(force = false) {
    if (this.status === 'loading' || (this.status === 'ok' && !force)) return;
    this.status = 'loading';
    try {
      const [c, p, g] = await Promise.all([
        api.profileCatalog(),
        api.profiles(),
        api.ggufCatalog().then(
          (x) => ({ ok: true as const, x }),
          (e) => ({ ok: false as const, e }),
        ),
      ]);
      const cat = parseCatalog(c);
      if (!cat) throw new Error('The controller sent a catalog this page cannot read.');
      this.catalog = cat;
      this.saved = parseProfiles(p, cat);
      this.ggufSaved = parseGgufProfiles(p);
      if (g.ok) {
        this.gguf = parseGgufCatalog(g.x);
        this.ggufStatus = this.gguf ? 'ok' : 'error';
      } else this.ggufStatus = g.e instanceof ApiError && (g.e.status === 404 || g.e.status === 405) ? 'unavailable' : 'error';
      this.status = 'ok';
      this.error = null;
    } catch (e) {
      this.status = e instanceof ApiError && (e.status === 404 || e.status === 405) ? 'unavailable' : 'error';
      this.error = errorText(e);
    }
  }

  upsertGguf(p: SavedGgufProfile) {
    this.ggufSaved = [p, ...this.ggufSaved.filter((x) => x.revision !== p.revision)];
  }

  upsert(p: SavedProfile) {
    this.saved = [p, ...this.saved.filter((x) => x.revision !== p.revision)];
  }
}

export const profileStore = new ProfileStore();

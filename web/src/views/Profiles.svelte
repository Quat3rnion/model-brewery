<script lang="ts">
  import { api, errorText } from '../lib/api';
  import { profileStore } from '../lib/profiles.svelte';
  import { formatBytes, formatTime, shortHash } from '../lib/format';
  import { signedBytes } from '../lib/calibration';
  import {
    BASE_LABEL,
    GROUP_LABEL,
    applyGroup,
    applyPolicy,
    buildProfileBody,
    layerSummary,
    parseProfile,
    predictSize,
    shortRev,
    type Base,
    type Choice,
    type Overrides,
    type SavedProfile,
  } from '../lib/profiles';
  import LayerMap from '../components/profiles/LayerMap.svelte';
  import TensorList from '../components/profiles/TensorList.svelte';
  import Badge from '../components/Badge.svelte';
  import GgufEditor from '../components/profiles/GgufEditor.svelte';
  import { focusAfterRemoval } from '../lib/focus';
  import { hashParam, route } from '../lib/route.svelte';
  import { unsavedChanges } from '../lib/unsaved-changes.svelte';

  $effect(() => {
    void profileStore.load();
  });
  const cat = $derived(profileStore.catalog);
  const format = $derived(hashParam(route.hash, 'format') === 'gguf' ? 'gguf' : 'hgn');
  function setFormat(event: MouseEvent, next: 'hgn' | 'gguf') {
    event.preventDefault();
    if (next !== format) location.hash = next === 'gguf' ? '#/profiles?format=gguf' : '#/profiles';
  }

  // Editor state. `from` is what this draft started from: a built-in or a saved revision.
  let editing = $state(false);
  let base = $state<Base>('v2');
  let name = $state('');
  let overrides = $state<Overrides>({});
  let from = $state<{ kind: 'built-in'; base: Base } | { kind: 'saved'; p: SavedProfile }>({ kind: 'built-in', base: 'v2' });
  let selectedLayers = $state<number[]>([]);
  let role = $state('');
  let choice = $state<Choice | 'base'>('ht43');
  let saving = $state(false);
  let saveError = $state<string | null>(null);
  let savedRev = $state<SavedProfile | null>(null);
  let opener = $state<HTMLButtonElement>();

  const startOverrides = $derived(from.kind === 'saved' ? from.p.overrides : {});
  const dirty = $derived(editing && format === 'hgn' && (JSON.stringify(overrides) !== JSON.stringify(startOverrides)
    || name.trim() !== (from.kind === 'saved' ? from.p.name : '')));
  const guard = unsavedChanges(() => dirty);
  $effect(() => { if (format !== 'hgn') editing = false; });
  const stats = $derived(cat ? layerSummary(cat, base, overrides) : new Map());
  const size = $derived(cat ? predictSize(cat, base, overrides) : null);
  const built = $derived(cat ? buildProfileBody(cat, base, name, overrides) : { body: null, problems: ['No catalog.'] });
  const trunkRoles = $derived(cat ? [...new Set(cat.tensors.filter((t) => t.group === 'trunk').map((t) => t.role).filter(Boolean))].sort() : []);

  function startFrom(b: Base, button: HTMLButtonElement) {
    if (saving || !guard.confirm()) return;
    opener = button;
    from = { kind: 'built-in', base: b };
    base = b;
    name = '';
    overrides = {};
    selectedLayers = [];
    editing = true;
    savedRev = null;
    saveError = null;
  }
  function edit(p: SavedProfile, button: HTMLButtonElement) {
    if (saving || !guard.confirm()) return;
    opener = button;
    from = { kind: 'saved', p };
    base = p.base;
    name = p.name;
    overrides = { ...p.overrides };
    selectedLayers = [];
    editing = true;
    savedRev = null;
    saveError = null;
  }
  function reset() {
    overrides = { ...startOverrides };
    name = from.kind === 'saved' ? from.p.name : '';
    savedRev = null;
  }
  function close() {
    if (saving || !guard.confirm()) return;
    void focusAfterRemoval(() => { editing = false; }, () => opener);
  }
  function applyBrush() {
    if (!cat || !selectedLayers.length) return;
    overrides = applyPolicy(cat, base, overrides, { layers: selectedLayers, role: role || null, choice });
  }
  async function save() {
    if (!built.body || saving) return;
    saving = true;
    saveError = null;
    try {
      const p = parseProfile(await api.saveProfile(built.body), cat);
      if (!p) throw new Error('The controller answered without a revision.');
      profileStore.upsert(p);
      savedRev = p;
      from = { kind: 'saved', p };
    } catch (e) {
      saveError = errorText(e);
    } finally {
      saving = false;
    }
  }
  function useInConvert(p: SavedProfile) {
    if (saving || !guard.confirm()) return;
    editing = false;
    profileStore.pendingUse = p.revision;
    location.hash = '#/convert';
  }
  const changedCount = (p: SavedProfile) => Object.keys(p.overrides).length;
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Conversion profiles</h1>
    <div class="fmt" role="radiogroup" aria-label="Output format">
      <label class:on={format === 'hgn'}><input type="radio" name="profile-format" checked={format === 'hgn'} onclick={(event) => setFormat(event, 'hgn')} /> HGN (v2 / HT43)</label>
      <label class:on={format === 'gguf'}><input type="radio" name="profile-format" checked={format === 'gguf'} onclick={(event) => setFormat(event, 'gguf')} /> GGUF (llama.cpp)</label>
    </div>
    <p class="muted small">
      {#if format === 'gguf'}
        GGUF profiles pick a llama.cpp preset and optional exact per-tensor types. They are separate from HGN profiles.
      {:else}
        Choose, tensor by tensor, between the two built-in encodings. HGN v2 and HT43 stay read-only; a custom profile starts from one of them and changes only tensors where the other profile has a different, legal encoding.
      {/if}
      Each save is a new immutable revision. <a href="#/convert">Back to Convert</a>
    </p>
  </div>

  {#if format === 'gguf'}
    {#if profileStore.status === 'loading' || profileStore.status === 'idle'}<p class="muted">Loading…</p>
    {:else if profileStore.gguf}<GgufEditor catalog={profileStore.gguf} />
    {:else if profileStore.ggufStatus === 'unavailable'}<p class="note warn">This controller does not provide GGUF output profiles.</p>
    {:else}<div class="note bad" role="alert"><p>The GGUF catalog could not be loaded.</p><button type="button" class="btn small" onclick={() => profileStore.load(true)}>Retry</button></div>{/if}
  {:else if profileStore.status === 'loading' && !cat}
    <p class="muted">Loading catalog…</p>
  {:else if profileStore.status === 'unavailable'}
    <p class="note warn">This controller does not provide conversion profiles.</p>
  {:else if profileStore.status === 'error'}
    <div class="note bad" role="alert"><p>The catalog could not be loaded: {profileStore.error}</p><button type="button" class="btn small" onclick={() => profileStore.load(true)}>Retry</button></div>
  {:else if cat}
    <section class="panel" aria-labelledby="pl-h">
      <header>
        <h2 id="pl-h">Profiles</h2>
        <span class="xs muted">{cat.architecture} · source <span class="mono" title={cat.sourceSha256}>{shortHash(cat.sourceSha256)}</span> · catalog <span class="mono" title={cat.catalogSha256}>{shortHash(cat.catalogSha256)}</span></span>
      </header>
      <div class="body stack">
        <div class="cards">
          {#each ['v2', 'ht43'] as const as b (b)}
            <article class="card builtin">
              <div class="ct"><strong>{BASE_LABEL[b]}</strong><Badge tone="muted" label="Built-in · read-only" /></div>
              <p class="xs muted num">{formatBytes(cat.baseBytes[b])}</p>
              <button type="button" class="btn small" onclick={(event) => startFrom(b, event.currentTarget)}>Start from {BASE_LABEL[b]}</button>
            </article>
          {/each}
        </div>
        {#if profileStore.saved.length}
          <div class="table-wrap">
            <table class="data stackable saved">
              <caption class="sr-only">Saved profile revisions, newest first</caption>
              <thead><tr><th scope="col">Name</th><th scope="col">Revision</th><th scope="col">Base</th><th scope="col" class="r">Changed tensors</th><th scope="col" class="r">Size change</th><th scope="col">Saved</th><th scope="col"><span class="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {#each profileStore.saved as p (p.revision)}
                  <tr class:cur={savedRev?.revision === p.revision}>
                    <th scope="row">{p.name}{#if !p.compatible}<div><Badge tone="warn" label="Other catalog" /></div>{/if}</th>
                    <td class="mono xs" data-label="Revision" title={p.revision}>{shortRev(p.revision)}</td>
                    <td data-label="Base">{BASE_LABEL[p.base]}</td>
                    <td class="num r" data-label="Changed tensors">{changedCount(p)}</td>
                    <td class="num r" data-label="Size change">{signedBytes(p.size?.delta ?? null)}</td>
                    <td class="xs" data-label="Saved">{p.createdAt ? formatTime(p.createdAt) : '—'}</td>
                    <td class="acts">
                      {#if p.compatible}
                        <button type="button" class="btn small" onclick={(event) => edit(p, event.currentTarget)} aria-label="Edit {p.name} as a new revision">Edit</button>
                        <button type="button" class="btn small primary" onclick={() => useInConvert(p)} aria-label="Use in Convert: {p.name} revision {shortRev(p.revision)}">Use in Convert</button>
                      {:else}<span class="xs muted">Made for another catalog</span>{/if}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {:else}
          <p class="xs muted">No saved profiles yet. Start from a built-in profile.</p>
        {/if}
      </div>
    </section>

    {#if editing}
      <section class="panel" aria-labelledby="ed-h">
        <header>
          <h2 id="ed-h">{from.kind === 'saved' ? `Editing ${from.p.name}` : `New profile from ${BASE_LABEL[base]}`}</h2>
          <button type="button" class="btn small ghost" onclick={close} disabled={saving}>Close editor</button>
        </header>
        <div class="body stack">
          <div class="summary" aria-live="polite">
            <div><span class="xs muted">Base</span><strong>{BASE_LABEL[base]}</strong></div>
            <div><span class="xs muted">Changed tensors</span><strong class="num">{size?.changed ?? 0}</strong></div>
            <div><span class="xs muted">Predicted size</span><strong class="num">{formatBytes(size?.predicted ?? null)}</strong></div>
            <div><span class="xs muted">Change</span><strong class="num">{signedBytes(size?.delta ?? null)}</strong></div>
            <button type="button" class="btn small" onclick={reset} disabled={!dirty}>Reset changes</button>
          </div>
          <p class="xs muted">A size prediction from the catalog, including companion tensors. It is not a quality result.</p>

          <div class="block">
            <h3 class="h">Trunk layers <span class="muted">({cat.trunkLayers.length})</span></h3>
            <LayerMap layers={cat.trunkLayers} {stats} bind:selected={selectedLayers} />
            <div class="brush" role="group" aria-labelledby="br-h">
              <span id="br-h" class="small"><strong>{selectedLayers.length ? `${selectedLayers.length} layer${selectedLayers.length === 1 ? '' : 's'} selected` : 'Select layers to change them together'}</strong>{#if selectedLayers.length}<span class="xs muted"> ({selectedLayers.length > 6 ? `${selectedLayers[0]}–${selectedLayers[selectedLayers.length - 1]}` : selectedLayers.join(', ')})</span>{/if}</span>
              <div class="field">
                <label for="br-role">Role</label>
                <select id="br-role" class="input" bind:value={role}>
                  <option value="">All roles</option>
                  {#each trunkRoles as r (r)}<option value={r}>{r}</option>{/each}
                </select>
              </div>
              <div class="field">
                <label for="br-choice">Set to</label>
                <select id="br-choice" class="input" bind:value={choice}>
                  <option value="ht43">HT43</option>
                  <option value="v2">HGN v2</option>
                  <option value="base">Base ({BASE_LABEL[base]})</option>
                </select>
              </div>
              <button type="button" class="btn" disabled={!selectedLayers.length} onclick={applyBrush}>Apply to selected layers</button>
              {#if selectedLayers.length}<button type="button" class="btn ghost small" onclick={() => (selectedLayers = [])}>Clear selection</button>{/if}
            </div>
            <p class="xs muted">Only tensors with a different, legal encoding change. Fixed tensors keep the base encoding.</p>
          </div>

          {#if cat.fixedGroups.length}
            <div class="block">
              <h3 class="h">Other groups</h3>
              <div class="cards">
                {#each cat.fixedGroups as g (g.group)}
                  <article class="card">
                    <div class="ct"><strong>{GROUP_LABEL[g.group] ?? g.group}</strong><span class="xs muted num">{g.count} tensor{g.count === 1 ? '' : 's'}</span></div>
                    {#if g.editable}
                      <div class="row">
                        <button type="button" class="btn small" onclick={() => (overrides = applyGroup(cat, base, overrides, g.group, 'ht43'))}>All HT43</button>
                        <button type="button" class="btn small" onclick={() => (overrides = applyGroup(cat, base, overrides, g.group, 'v2'))}>All v2</button>
                      </div>
                      <p class="xs muted">{g.editable} of {g.count} editable.</p>
                    {:else}
                      <p class="xs muted">Fixed: follows the base profile. See the tensor list for each reason.</p>
                    {/if}
                  </article>
                {/each}
              </div>
            </div>
          {/if}

          <div class="block">
            <h3 class="h">Tensors</h3>
            <TensorList catalog={cat} {base} bind:overrides {selectedLayers} />
          </div>

          <div class="block">
            <h3 class="h">Save revision</h3>
            <div class="field">
              <label for="pf-name">Profile name</label>
              <input id="pf-name" class="input" bind:value={name} maxlength="80" autocomplete="off" aria-describedby="pf-name-h" />
              <span class="hint" id="pf-name-h">Saving with the same name adds a new revision. Earlier revisions stay unchanged.</span>
            </div>
            {#if built.problems.length}<p class="why xs"><strong>Needed:</strong> {built.problems.join(' ')}</p>{/if}
            <div aria-live="polite">
              {#if saveError}<p class="note bad" role="alert">{saveError} Your changes are kept.</p>{/if}
              {#if savedRev}
                <div class="note info">
                  <p>Saved <strong>{savedRev.name}</strong>, revision <span class="mono" title={savedRev.revision}>{shortRev(savedRev.revision)}</span>. No job was started.</p>
                  <button type="button" class="btn small primary" onclick={() => useInConvert(savedRev!)}>Use in Convert</button>
                </div>
              {/if}
            </div>
          </div>
          <div class="save-bar">
            {#if !name.trim()}<a href="#pf-name" onclick={(event) => { event.preventDefault(); document.getElementById('pf-name')?.focus(); }}>Enter a profile name</a>{/if}
            <button type="button" class="btn primary" disabled={!built.body || saving || (!dirty && from.kind === 'saved')} onclick={save}>{saving ? 'Saving…' : 'Save revision'}</button>
          </div>
        </div>
      </section>
    {/if}
  {/if}
</div>

<style>
  .fmt {
    display: inline-flex;
    align-self: flex-start;
    padding: 2px;
    margin: 4px 0;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    background: var(--surface);
  }
  .fmt label {
    position: relative;
    display: inline-flex;
    align-items: center;
    min-height: 32px;
    padding: 0 12px;
    border-radius: var(--r-sm);
    font-size: var(--fs-sm);
    color: var(--text-2);
    cursor: pointer;
  }
  .fmt label.on {
    background: var(--accent);
    color: var(--accent-text);
    font-weight: 600;
  }
  .fmt input {
    position: absolute;
    inset: 0;
    opacity: 0;
    margin: 0;
    cursor: pointer;
  }
  .fmt label:has(input:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 1px;
  }
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: var(--s-3);
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: flex-start;
    padding: var(--s-3);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
    background: var(--surface-2);
  }
  .ct {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 10px;
    align-items: center;
    width: 100%;
  }
  .saved th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
    font-weight: 600;
  }
  .saved tr.cur {
    background: var(--accent-soft);
  }
  .acts {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: flex-end;
  }
  .r {
    text-align: right;
  }
  .summary {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-5);
    align-items: flex-end;
    padding: var(--s-3);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
    background: var(--surface-2);
    position: sticky;
    top: 55px;
    z-index: 2;
  }
  .summary > div {
    display: flex;
    flex-direction: column;
  }
  .summary > button {
    margin-left: auto;
  }
  .block {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .h {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
  }
  .brush {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-3);
    align-items: flex-end;
    padding: var(--s-3);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
  }
  .brush > span {
    flex: 1 1 100%;
  }
  .save-bar {
    position: sticky;
    bottom: 0;
    z-index: 2;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: flex-end;
    gap: var(--s-2);
    padding: var(--s-2);
    background: var(--surface-2);
    border-radius: var(--r-md);
  }
  .why {
    margin: 0;
    color: var(--text-2);
  }
  @media (max-width: 760px) {
    .summary {
      position: static;
    }
    .summary > button {
      margin-left: 0;
    }
  }
  @media (max-width: 640px) {
    .r {
      text-align: left;
    }
    .acts {
      justify-content: flex-start;
    }
  }
</style>

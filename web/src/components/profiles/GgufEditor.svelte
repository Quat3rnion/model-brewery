<script lang="ts">
  import { unsavedChanges } from '../../lib/unsaved-changes.svelte';
  import { untrack } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';
  import { api, errorText } from '../../lib/api';
  import { brewery } from '../../lib/store.svelte';
  import { profileStore } from '../../lib/profiles.svelte';
  import { WEIGHT_SOURCES, hostName, isSafeSource, formatBytes, formatTime } from '../../lib/format';
  import { parseGgufProfiles, shortRev, type SavedGgufProfile } from '../../lib/profiles';
  import { buildGgufProfileBody, imatrixRequired, layerOf, rangeOverrides, tensorBytes, forkSelected, profileMatches, FORK_NOTICE, type GgufCatalog } from '../../lib/gguf';
  import { ggufEvidenceIds, inventoryBlocker, resolveGgufSource, validSha, type ExactInventory } from '../../lib/ggufSource';
  import { inventoryJob, walkInventory } from '../../lib/weightmap';
  import type { JobDetail, JobSummary } from '../../lib/types';
  import Badge from '../Badge.svelte';
  import JobState from '../JobState.svelte';
  import ModelSelect from '../ModelSelect.svelte';

  let { catalog }: { catalog: GgufCatalog } = $props();
  const PAGE = 60;
  const uid = $props.id();

  let name = $state('');
  // The catalog is static for this page; the first preset is only the starting value.
  let preset = $state(untrack(() => catalog.presets[0]?.id ?? ''));
  let overrides = $state<Record<string, string>>({});
  let start = $state<SavedGgufProfile | null>(null);

  const models = $derived((brewery.state?.models ?? []).filter(m => WEIGHT_SOURCES.includes(m.kind)));
  const jobs = $derived(brewery.state?.jobs ?? []);
  let sourceId = $state('');
  const source = $derived(models.find(m => m.id === sourceId));
  const sourceKey = $derived(source ? JSON.stringify([source.id, source.kind, source.host, source.path, source.sha256, source.bundle_sha256]) : '');
  const evidenceKey = $derived(source ? JSON.stringify(ggufEvidenceIds(source, jobs)) : '');
  const lookupKey = $derived(sourceKey ? JSON.stringify([sourceKey, evidenceKey]) : '');
  let resolved = $state<{ key: string; exact: ExactInventory | null } | null>(null);
  const evidence = new Map<string, Promise<JobDetail>>();
  function loadDetail(id: string): Promise<JobDetail> {
    let pending = evidence.get(id);
    if (!pending) {
      pending = brewery.loadJob(id);
      evidence.set(id, pending);
    }
    return pending;
  }
  $effect(() => {
    const key = lookupKey;
    const ids = evidenceKey;
    const model = untrack(() => source);
    if (!key || !model) return;
    let active = true;
    const current = () => active && lookupKey === key;
    void resolveGgufSource(model, JSON.parse(ids), loadDetail, current).then(exact => {
      if (current()) resolved = { key, exact };
    });
    return () => { active = false; };
  });
  const lookup = $derived(resolved?.key === lookupKey ? resolved : null);
  const host = $derived(brewery.state?.hosts.find(h => h.id === source?.host));
  const blocked = $derived(source ? inventoryBlocker(source, host) : null);
  const packed = $derived(!!source && !isSafeSource(source.kind) && host?.capabilities?.weight_packed === true);
  const submitted = new SvelteMap<string, JobSummary | string | null>();
  const detailErrors = new SvelteMap<string, string>();
  const inventoryJobs = $derived([...jobs, ...[...submitted.values()].filter((j): j is JobSummary =>
    !!j && typeof j !== 'string' && !jobs.some(row => row.id === j.id))]);
  const inv = $derived(source && lookup && !lookup.exact ? walkInventory(source, inventoryJobs, brewery.details, {
    format: source.kind, packed, matchHost: true,
    sha: source.kind === 'hf' ? validSha(source.bundle_sha256) ?? validSha(source.sha256) : validSha(source.sha256),
  }) : null);
  const need = $derived(source && inv?.need ? inventoryJob(source.id, source.host, inv.need.offset, inv.need.sha, packed) : null);
  const needKey = $derived(need ? JSON.stringify([sourceKey, need.options]) : '');
  const readPending = $derived(!!needKey && submitted.get(needKey) === null);
  let lastRead = $state<{ key: string; job: JobSummary } | null>(null);
  const readJob = $derived(inv?.job ?? (lastRead?.key === sourceKey
    ? jobs.find(j => j.id === lastRead?.job.id) ?? lastRead.job : null));
  const invError = $derived(inv?.error || (inv?.job && detailErrors.get(inv.job.id)) ||
    (typeof submitted.get(needKey) === 'string' ? submitted.get(needKey) as string : null));
  $effect(() => {
    const job = inv?.job;
    if (job?.state !== 'succeeded' || brewery.details[job.id]) return;
    void loadDetail(job.id).catch(error => detailErrors.set(job.id, errorText(error)));
  });
  function readTensors() {
    const key = needKey;
    const request = need;
    if (!key || !request || blocked || readPending) return;
    const modelKey = sourceKey;
    submitted.set(key, null);
    void api.submitJob(request).then(job => {
      submitted.set(key, job);
      lastRead = { key: modelKey, job };
      void brewery.refresh();
    }).catch(error => submitted.set(key, errorText(error)));
  }
  const native = $derived(inv?.pages.flatMap(p => p.tensors) ?? []);
  const nativeNames = $derived(new Set(source?.kind === 'gguf' ? [] : native.map(t => t.name)));
  const tensors = $derived(lookup?.exact?.tensors ??
    (source?.kind === 'gguf' && native.length ? native.map(t => {
      const layer = layerOf(t.name);
      return { name: t.name, layer, role: layer === null ? t.name : t.name.split('.').slice(2).join('.'), shape: t.shape, currentType: t.dtype };
    }) : null));
  const exactNames = $derived(new Set(tensors?.map(t => t.name)));
  const layers = $derived(tensors ? [...new Set(tensors.filter((t) => t.layer !== null).map((t) => t.layer as number))].sort((a, b) => a - b) : []);
  const originalFormats = $derived.by(() => {
    const groups = new Map<number | null, Set<string>>();
    for (const t of tensors ?? native) {
      const formats = groups.get(t.layer) ?? new Set<string>();
      const type = 'dtype' in t ? t.dtype : t.currentType;
      for (const f of (type ?? 'Not reported').split(' + ')) formats.add(f);
      groups.set(t.layer, formats);
    }
    return [...groups].sort(([a], [b]) => (a ?? Infinity) - (b ?? Infinity))
      .map(([layer, formats]) => ({layer, types: [...formats].sort().join(', ')}));
  });
  const roles = $derived(tensors ? [...new Set(tensors.filter((t) => t.layer !== null).map((t) => t.role))].sort() : []);

  // Range tool
  let rFrom = $state(0);
  let rTo = $state(0);
  let rRole = $state('');
  let rType = $state('');
  function applyRange(clear = false) {
    if (!tensors) return;
    overrides = rangeOverrides(tensors, overrides, { from: rFrom, to: rTo, role: rRole || null, type: clear ? null : rType || null });
  }

  // Explicit name entry (when no inspect result lists names)
  let manualName = $state('');
  let manualType = $state('');
  const manualProblem = $derived(nativeNames.has(manualName.trim())
    ? 'This is a native source name, not a GGUF override.'
    : manualName && !/^[A-Za-z0-9_.\-]{1,200}$/.test(manualName.trim()) ? 'Use the exact GGUF tensor name, for example blk.12.ffn_down.weight.' : null);
  const discovering = $derived(!!source && (!lookup || readPending || (!!inv?.job && !blocked && !invError)));
  function addManual() {
    const n = manualName.trim();
    if (!n || manualProblem || !manualType || discovering) return;
    overrides = Object.fromEntries(Object.entries({ ...overrides, [n]: manualType }).sort(([a], [b]) => a.localeCompare(b)));
    manualName = '';
  }

  // List
  let q = $state('');
  let page = $state(0);
  const listed = $derived.by(() => {
    const exact = (tensors ?? []).map(t => ({ ...t, kind: 'exact' as const }));
    const readonly = native.filter(t => nativeNames.has(t.name)).map(t => ({ name: t.name, shape: t.shape, currentType: t.dtype, kind: 'native' as const }));
    const extra = Object.keys(overrides).filter(n => !exactNames.has(n) && !nativeNames.has(n))
      .map(name => ({ name, shape: null, currentType: null, kind: 'typed' as const }));
    const all = [...exact, ...readonly, ...extra];
    const k = q.toLowerCase().split(/\s+/).filter(Boolean);
    return k.length ? all.filter((t) => k.every((x) => `${t.name} ${t.currentType ?? ''} ${overrides[t.name] ?? ''}`.toLowerCase().includes(x))) : all;
  });
  $effect(() => {
    void q;
    void sourceKey;
    page = 0;
  });
  const pages = $derived(Math.max(1, Math.ceil(listed.length / PAGE)));
  const visiblePage = $derived(Math.min(page, pages - 1));

  const needsImatrix = $derived(imatrixRequired(catalog, preset, overrides));
  const built = $derived(buildGgufProfileBody(catalog, name, preset, overrides));
  const nativeOverrides = $derived(Object.keys(overrides).some(n => nativeNames.has(n)));
  const startJson = $derived(start ? JSON.stringify([start.name, start.preset, start.overrides]) : JSON.stringify(['', catalog.presets[0]?.id ?? '', {}]));
  const dirty = $derived(JSON.stringify([name.trim(), preset, overrides]) !== startJson);
  const guard = unsavedChanges(() => dirty);

  function edit(p: SavedGgufProfile) {
    if (saving || !guard.confirm()) return;
    start = p;
    name = p.name;
    preset = p.preset;
    overrides = { ...p.overrides };
    saved = null;
  }
  function reset() {
    name = start?.name ?? '';
    preset = start?.preset ?? catalog.presets[0]?.id ?? '';
    overrides = { ...(start?.overrides ?? {}) };
  }
  function setOne(n: string, t: string) {
    const next = { ...overrides };
    if (t) next[n] = t;
    else delete next[n];
    overrides = Object.fromEntries(Object.entries(next).sort(([a], [b]) => a.localeCompare(b)));
  }

  let saving = $state(false);
  let saveErr = $state<string | null>(null);
  let saved = $state<SavedGgufProfile | null>(null);
  async function save() {
    if (!built.body || saving || discovering || nativeOverrides) return;
    saving = true;
    saveErr = null;
    try {
      const p = parseGgufProfiles([await api.saveProfile(built.body)])[0];
      if (!p) throw new Error('The controller answered without a GGUF revision.');
      profileStore.upsertGguf(p);
      saved = p;
      start = p;
    } catch (e) {
      saveErr = errorText(e);
    } finally {
      saving = false;
    }
  }
  function useInConvert(p: SavedGgufProfile) {
    if (saving || !profileMatches(catalog,p) || !guard.confirm()) return;
    reset();
    profileStore.pendingUse = `gguf:${p.revision}`;
    location.hash = '#/convert?format=gguf';
  }
  const typeNote = (id: string) => (catalog.typeById.get(id)?.fork ? ' · ROCmFPX' : '') + (catalog.typeById.get(id)?.requiresImatrix ? ' (needs imatrix)' : '');
</script>

<div class="stack">
  <p class="note info">
    GGUF output uses llama.cpp tensor types. A preset sets every tensor; overrides change single tensors by their exact GGUF name. Only types the worker's llama.cpp build supports are listed. Sizes are shown only where the block size and the tensor shape are known.
  </p>

  {#if profileStore.ggufSaved.length}
    <div class="table-wrap">
      <table class="data stackable">
        <caption class="sr-only">Saved GGUF profile revisions</caption>
        <thead><tr><th scope="col">Name</th><th scope="col">Revision</th><th scope="col">Preset</th><th scope="col" class="r">Overrides</th><th scope="col">Saved</th><th scope="col"><span class="sr-only">Actions</span></th></tr></thead>
        <tbody>
          {#each profileStore.ggufSaved as p (p.revision)}
            <tr class:cur={saved?.revision === p.revision}>
              <th scope="row">{p.name}</th>
              <td class="mono xs" data-label="Revision" title={p.revision}>{shortRev(p.revision)}</td>
              <td class="mono" data-label="Preset">{p.preset}</td>
              <td class="num r" data-label="Overrides">{Object.keys(p.overrides).length}</td>
              <td class="xs" data-label="Saved">{p.createdAt ? formatTime(p.createdAt) : '—'}</td>
              <td class="acts">
                <button type="button" class="btn small" onclick={() => edit(p)} aria-label="Edit {p.name} as a new revision">Edit</button>
                <button type="button" class="btn small primary" disabled={!profileMatches(catalog,p)} onclick={() => useInConvert(p)} aria-label="Use in Convert: {p.name} revision {shortRev(p.revision)}">Use in Convert</button>
                {#if !profileMatches(catalog,p)}<span class="xs muted">Incompatible encoder revision</span>{/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  <div class="grid-2">
    <div class="field">
      <label for="{uid}-preset">Base preset</label>
      <select id="{uid}-preset" class="input" bind:value={preset} aria-describedby="{uid}-preset-h">
        {#each catalog.presets as p (p.id)}<option value={p.id}>{p.label ?? p.id}{p.fork ? ' · ROCmFPX' : ''}{p.requiresImatrix ? ' (needs imatrix)' : ''}</option>{/each}
      </select>
      <span class="hint" id="{uid}-preset-h">A llama.cpp quantization preset. It is not an HGN profile.</span>
      {#if forkSelected(catalog,preset,overrides)}<p class="note warn small">{FORK_NOTICE} Conversion requires a worker with ROCmFPX output support.</p>{/if}
    </div>
    <div class="stack">
      <ModelSelect bind:value={sourceId} kinds={WEIGHT_SOURCES} label="Source model" hint="Choose a Library model. You can also add overrides by exact name without a source." />
      {#if source && need && !blocked}
        <div class="row">
          <button type="button" class="btn" onclick={readTensors} disabled={readPending}>
            {inv?.pages.length ? 'Read next tensors' : 'Read tensor list'}
          </button>
          <span class="hint">This queues a <code>weight_inspect</code> job on {hostName(brewery.state?.hosts ?? [], source.host)}.</span>
        </div>
      {/if}
      {#if readJob}
        <div class="row" role="status">
          <a href="#/jobs/{encodeURIComponent(readJob.id)}">Tensor list read job</a>
          <JobState job={readJob} />
        </div>
      {/if}
      <span class="hint" role="status">
        {#if readPending}Submitting the tensor list read.
        {:else if source && !lookup}Checking for saved tensor names.
        {:else if discovering}Waiting for the tensor list.
        {:else if need && inv?.pages.length}{native.length} of {inv.pages[0].total} tensors.
        {:else if lookup?.exact?.planned}{tensors?.length} planned main GGUF tensor names in {layers.length} layers. This plan does not confirm conversion or list separate outputs.
        {:else if tensors}{tensors.length} exact GGUF tensor names in {layers.length} layers.
        {:else if native.length}{native.length} native source tensors. No verified GGUF tensor mapping is available.
        {:else if blocked}{blocked}
        {:else if source && !need && !invError}No tensor names were found for this source.
        {/if}
        {#if invError}{' '}{invError}{/if}
      </span>
    </div>
  </div>

  <div class="sum" aria-live="polite">
    <span><span class="xs muted">Preset</span><strong class="mono">{preset || '—'}</strong></span>
    <span><span class="xs muted">Overrides</span><strong class="num">{Object.keys(overrides).length}</strong></span>
    <span><span class="xs muted">imatrix</span><strong>{needsImatrix ? 'Required at conversion' : 'Not needed'}</strong></span>
    <button type="button" class="btn small" onclick={reset} disabled={!dirty}>Reset changes</button>
  </div>

  {#if tensors && layers.length && !discovering}
    <fieldset class="tool">
      <legend>Layer range</legend>
      <div class="field"><label for="{uid}-from">From layer</label><input id="{uid}-from" class="input num" type="number" min={layers[0]} max={layers[layers.length - 1]} bind:value={rFrom} /></div>
      <div class="field"><label for="{uid}-to">To layer</label><input id="{uid}-to" class="input num" type="number" min={layers[0]} max={layers[layers.length - 1]} bind:value={rTo} /></div>
      <div class="field">
        <label for="{uid}-role">Tensor</label>
        <select id="{uid}-role" class="input" bind:value={rRole}><option value="">All tensors in the layers</option>{#each roles as r (r)}<option value={r}>{r}</option>{/each}</select>
      </div>
      <div class="field">
        <label for="{uid}-type">Type</label>
        <select id="{uid}-type" class="input" bind:value={rType}><option value="" disabled>Choose a type</option>{#each catalog.types as t (t.id)}<option value={t.id}>{t.id}{typeNote(t.id)}</option>{/each}</select>
      </div>
      <button type="button" class="btn" disabled={!rType} onclick={() => applyRange()}>Set type</button>
      <button type="button" class="btn ghost" onclick={() => applyRange(true)}>Clear overrides</button>
    </fieldset>
  {/if}

  <fieldset class="tool">
    <legend>Add an override by exact name</legend>
    <div class="field grow">
      <label for="{uid}-mn">GGUF tensor name</label>
      <input id="{uid}-mn" class="input mono" bind:value={manualName} list={tensors ? `${uid}-names` : undefined} autocomplete="off" spellcheck="false" aria-invalid={manualProblem ? 'true' : undefined} aria-describedby="{uid}-mn-h" placeholder="blk.12.ffn_down.weight" />
      {#if tensors}<datalist id="{uid}-names">{#each tensors as t (t.name)}<option value={t.name}></option>{/each}</datalist>{/if}
      <span class="hint" id="{uid}-mn-h">{manualProblem ?? (tensors ? 'Known GGUF names are suggested.' : 'The worker checks that this name is part of the GGUF output.')}</span>
    </div>
    <div class="field">
      <label for="{uid}-mt">Type</label>
      <select id="{uid}-mt" class="input" bind:value={manualType}><option value="" disabled>Choose a type</option>{#each catalog.types as t (t.id)}<option value={t.id}>{t.id}{typeNote(t.id)}</option>{/each}</select>
    </div>
    <button type="button" class="btn" disabled={discovering || !manualName.trim() || !!manualProblem || !manualType} onclick={addManual}>Add override</button>
  </fieldset>

  {#if tensors?.length || native.length || Object.keys(overrides).length}
    {#if originalFormats.length}
      <details>
        <summary>Original formats by layer</summary>
        <dl>{#each originalFormats as group (group.layer)}
          <dt>{group.layer === null ? 'Other tensors' : `Layer ${group.layer}`}</dt><dd>{group.types}</dd>
        {/each}</dl>
      </details>
    {/if}
    <div class="field">
      <label for="{uid}-q">Search tensors and overrides</label>
      <input id="{uid}-q" class="input" type="search" bind:value={q} autocomplete="off" spellcheck="false" />
    </div>
    {#if !listed.length}
      <p class="xs muted">No tensors match this search.</p>
    {:else}
    <div class="table-wrap">
      <table class="data stackable">
        <caption class="sr-only">Source tensors and GGUF overrides</caption>
        <thead><tr><th scope="col">Tensor</th><th scope="col">Shape</th><th scope="col">Original format</th><th scope="col">Output type</th><th scope="col" class="r">Bytes</th></tr></thead>
        <tbody>
          {#each listed.slice(visiblePage * PAGE, visiblePage * PAGE + PAGE) as t (t.name)}
            {@const ov = Object.hasOwn(overrides, t.name) ? overrides[t.name] : ''}
            {@const bytes = ov && t.kind !== 'native' ? tensorBytes(catalog, ov, t.shape) : null}
            <tr class:changed={!!ov && t.kind !== 'native'}>
              <th scope="row" class="mono xs break">{t.name}{#if t.kind === 'typed'}<div><Badge tone="muted" label="Typed name, checked by the worker" /></div>{/if}</th>
              <td class="mono xs" data-label="Shape">{t.shape ? t.shape.join('×') : '—'}</td>
              <td class="mono xs" data-label="Original format">{t.currentType ?? 'Not reported'}</td>
              <td data-label="Output type">
                {#if t.kind === 'native'}
                  <span class="xs muted">No GGUF mapping</span>
                  {#if ov}<button type="button" class="btn small" onclick={() => setOne(t.name, '')}>Remove draft override</button>{/if}
                {:else}
                <label class="sr-only" for="{uid}-o-{t.name}">Output type for {t.name}</label>
                <select id="{uid}-o-{t.name}" class="input sel" value={ov} disabled={discovering} onchange={(e) => setOne(t.name, e.currentTarget.value)}>
                  <option value="">Preset ({preset})</option>
                  {#each catalog.types as ty (ty.id)}<option value={ty.id}>{ty.id}{typeNote(ty.id)}</option>{/each}
                </select>
                {/if}
              </td>
              <td class="num r xs" data-label="Bytes">{ov ? (bytes === null ? '—' : formatBytes(bytes)) : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    {#if pages > 1}
      <nav class="pager" aria-label="Tensor pages">
        <button type="button" class="btn small" disabled={visiblePage === 0} onclick={() => (page = visiblePage - 1)}>Previous</button>
        <span class="xs muted">Page {visiblePage + 1} of {pages}</span>
        <button type="button" class="btn small" disabled={visiblePage >= pages - 1} onclick={() => (page = visiblePage + 1)}>Next</button>
      </nav>
    {/if}
    <p class="xs muted">Bytes: payload of an overridden tensor from the catalog block size, when the shape is known. Preset tensor sizes are not estimated here.</p>
    {/if}
  {/if}

  <div class="row end">
    <div class="field grow">
      <label for="{uid}-name">Profile name</label>
      <input id="{uid}-name" class="input" bind:value={name} maxlength="80" autocomplete="off" aria-describedby="{uid}-name-h" />
      <span class="hint" id="{uid}-name-h">Saving with the same name adds a new revision; the imatrix is chosen at conversion time.</span>
    </div>
  </div>
  {#if built.problems.length}<p class="why xs"><strong>Needed:</strong> {built.problems.join(' ')}</p>{/if}
  {#if nativeOverrides}<p class="why xs">Remove draft overrides for source names without a GGUF mapping to save this profile.</p>{/if}
  <div aria-live="polite">
    {#if saveErr}<p class="note bad" role="alert">{saveErr} Your changes are kept.</p>{/if}
    {#if saved}<div class="note info"><p>Saved <strong>{saved.name}</strong>, revision <span class="mono">{shortRev(saved.revision)}</span>.</p><button type="button" class="btn small primary" onclick={() => useInConvert(saved!)}>Use in Convert</button></div>{/if}
  </div>
  <div class="save-bar">
    {#if !name.trim()}<a href="#{uid}-name" onclick={(event) => { event.preventDefault(); document.getElementById(`${uid}-name`)?.focus(); }}>Enter a profile name</a>{/if}
    <button type="button" class="btn primary" disabled={!built.body || saving || discovering || nativeOverrides || (!dirty && !!start)} onclick={save}>{saving ? 'Saving…' : 'Save revision'}</button>
  </div>
</div>

<style>
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
  .sum {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-5);
    align-items: flex-end;
    padding: var(--s-3);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
    background: var(--surface-2);
  }
  .sum > span {
    display: flex;
    flex-direction: column;
  }
  .sum > button {
    margin-left: auto;
  }
  .tool {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-3);
    align-items: flex-end;
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
    padding: var(--s-2) var(--s-3) var(--s-3);
    margin: 0;
    min-width: 0;
  }
  .tool legend {
    padding: 0 4px;
    font-weight: 600;
    font-size: var(--fs-sm);
  }
  .tool .input.num {
    width: 7em;
  }
  .grow {
    flex: 1 1 240px;
  }
  .end {
    align-items: flex-end;
  }
  .sel {
    min-height: 32px;
    padding: 2px 8px;
    font-size: var(--fs-sm);
  }
  tr.changed,
  tr.cur {
    background: var(--accent-soft);
  }
  th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
    font-weight: 500;
  }
  .r {
    text-align: right;
  }
  .acts {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: flex-end;
  }
  .pager {
    display: flex;
    gap: var(--s-3);
    align-items: center;
  }
  .why {
    margin: 0;
    color: var(--text-2);
  }
  @media (max-width: 640px) {
    .r {
      text-align: left;
    }
    .sum > button {
      margin-left: 0;
    }
  }
</style>

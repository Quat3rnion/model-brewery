<script lang="ts">
  import { untrack } from 'svelte';
  // GGUF profile: a llama.cpp preset plus exact per-tensor type overrides.
  // Tensor names come from an inspect/plan result of the chosen source, or are
  // typed explicitly; the worker validates every name and type.
  import { api, errorText } from '../../lib/api';
  import { brewery } from '../../lib/store.svelte';
  import { profileStore } from '../../lib/profiles.svelte';
  import { formatBytes, formatTime, isTerminal, modelLabel, newestFirst } from '../../lib/format';
  import { parseGgufProfiles, shortRev, type SavedGgufProfile } from '../../lib/profiles';
  import { buildGgufProfileBody, ggufTensorsFromResult, imatrixRequired, rangeOverrides, tensorBytes, type GgufCatalog, type GgufTensor } from '../../lib/gguf';
  import Badge from '../Badge.svelte';

  let { catalog }: { catalog: GgufCatalog } = $props();
  const PAGE = 60;
  const uid = $props.id();

  let name = $state('');
  // The catalog is static for this page; the first preset is only the starting value.
  let preset = $state(untrack(() => catalog.presets[0]?.id ?? ''));
  let overrides = $state<Record<string, string>>({});
  let start = $state<SavedGgufProfile | null>(null);

  // Tensor names: from the newest finished inspect/plan of a chosen source that reports gguf_tensors.
  const models = $derived(brewery.state?.models ?? []);
  const jobs = $derived(brewery.state?.jobs ?? []);
  const sourceJobs = $derived(newestFirst(jobs.filter((j) => (j.kind === 'inspect' || j.kind === 'plan') && j.state === 'succeeded')));
  let jobId = $state('');
  let loadErr = $state<string | null>(null);
  $effect(() => {
    if (jobId && !brewery.details[jobId]) void brewery.loadJob(jobId).catch((e) => (loadErr = errorText(e)));
  });
  const tensors = $derived<GgufTensor[] | null>(jobId ? ggufTensorsFromResult(brewery.details[jobId]?.result) : null);
  const layers = $derived(tensors ? [...new Set(tensors.filter((t) => t.layer !== null).map((t) => t.layer as number))].sort((a, b) => a - b) : []);
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
  const manualProblem = $derived(manualName && !/^[A-Za-z0-9_.\-]{1,200}$/.test(manualName.trim()) ? 'Use the exact GGUF tensor name, for example blk.12.ffn_down.weight.' : null);
  function addManual() {
    const n = manualName.trim();
    if (!n || manualProblem || !manualType) return;
    overrides = Object.fromEntries(Object.entries({ ...overrides, [n]: manualType }).sort(([a], [b]) => a.localeCompare(b)));
    manualName = '';
  }

  // List
  let q = $state('');
  let page = $state(0);
  const listed = $derived.by(() => {
    const names = tensors ? tensors.map((t) => t.name) : [];
    const extra = Object.keys(overrides).filter((n) => !names.includes(n));
    const all = [...(tensors ?? []), ...extra.map((n) => ({ name: n, layer: null, role: n, shape: null, currentType: null }) as GgufTensor)];
    const k = q.toLowerCase().split(/\s+/).filter(Boolean);
    return k.length ? all.filter((t) => k.every((x) => `${t.name} ${t.currentType ?? ''} ${overrides[t.name] ?? ''}`.toLowerCase().includes(x))) : all;
  });
  $effect(() => {
    void q;
    page = 0;
  });
  const pages = $derived(Math.max(1, Math.ceil(listed.length / PAGE)));

  const needsImatrix = $derived(imatrixRequired(catalog, preset, overrides));
  const built = $derived(buildGgufProfileBody(catalog, name, preset, overrides));
  const startJson = $derived(start ? JSON.stringify([start.name, start.preset, start.overrides]) : JSON.stringify(['', catalog.presets[0]?.id ?? '', {}]));
  const dirty = $derived(JSON.stringify([name.trim(), preset, overrides]) !== startJson);

  function edit(p: SavedGgufProfile) {
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
    if (!built.body || saving) return;
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
    profileStore.pendingUse = `gguf:${p.revision}`;
    location.hash = '#/convert';
  }
  const typeNote = (id: string) => (catalog.typeById.get(id)?.requiresImatrix ? ' (needs imatrix)' : '');
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
                <button type="button" class="btn small primary" onclick={() => useInConvert(p)} aria-label="Use GGUF profile {p.name} in Convert">Use in Convert</button>
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
        {#each catalog.presets as p (p.id)}<option value={p.id}>{p.label ?? p.id}{p.requiresImatrix ? ' (needs imatrix)' : ''}</option>{/each}
      </select>
      <span class="hint" id="{uid}-preset-h">A llama.cpp quantization preset. It is not an HGN profile.</span>
    </div>
    <div class="field">
      <label for="{uid}-src">Tensor names from</label>
      <select id="{uid}-src" class="input" bind:value={jobId} aria-describedby="{uid}-src-h">
        <option value="">No inspect result (type names yourself)</option>
        {#each sourceJobs as j (j.id)}<option value={j.id}>{j.kind} · {j.input_ids.map((id) => modelLabel(models.find((m) => m.id === id))).join(', ')} · {j.id.slice(0, 8)}</option>{/each}
      </select>
      <span class="hint" id="{uid}-src-h">
        {#if jobId && !brewery.details[jobId]}Loading…{:else if jobId && !tensors}This result lists no GGUF tensor names. Run inspect again on a worker that reports them, or type names below.{:else if tensors}{tensors.length} tensors in {layers.length} layers, names as written to the GGUF.{:else}Names in GGUF form (blk.N.…). The worker checks each name.{/if}
        {#if loadErr}{' '}{loadErr}{/if}
      </span>
    </div>
  </div>

  <div class="sum" aria-live="polite">
    <span><span class="xs muted">Preset</span><strong class="mono">{preset || '—'}</strong></span>
    <span><span class="xs muted">Overrides</span><strong class="num">{Object.keys(overrides).length}</strong></span>
    <span><span class="xs muted">imatrix</span><strong>{needsImatrix ? 'Required at conversion' : 'Not needed'}</strong></span>
    <button type="button" class="btn small" onclick={reset} disabled={!dirty}>Reset changes</button>
  </div>

  {#if tensors && layers.length}
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
      <span class="hint" id="{uid}-mn-h">{manualProblem ?? (tensors ? 'Names from the inspect result are suggested.' : 'Not checked here; the worker rejects names that do not exist in the source.')}</span>
    </div>
    <div class="field">
      <label for="{uid}-mt">Type</label>
      <select id="{uid}-mt" class="input" bind:value={manualType}><option value="" disabled>Choose a type</option>{#each catalog.types as t (t.id)}<option value={t.id}>{t.id}{typeNote(t.id)}</option>{/each}</select>
    </div>
    <button type="button" class="btn" disabled={!manualName.trim() || !!manualProblem || !manualType} onclick={addManual}>Add override</button>
  </fieldset>

  {#if listed.length}
    <div class="field">
      <label for="{uid}-q">Search tensors and overrides</label>
      <input id="{uid}-q" class="input" type="search" bind:value={q} autocomplete="off" spellcheck="false" />
    </div>
    <div class="table-wrap">
      <table class="data stackable">
        <caption class="sr-only">GGUF tensors and their type overrides</caption>
        <thead><tr><th scope="col">Tensor</th><th scope="col">Shape</th><th scope="col">Source type</th><th scope="col">Output type</th><th scope="col" class="r">Bytes</th></tr></thead>
        <tbody>
          {#each listed.slice(page * PAGE, page * PAGE + PAGE) as t (t.name)}
            {@const ov = overrides[t.name] ?? ''}
            {@const bytes = ov ? tensorBytes(catalog, ov, t.shape) : null}
            <tr class:changed={!!ov}>
              <th scope="row" class="mono xs break">{t.name}{#if !tensors?.some((x) => x.name === t.name)}<div><Badge tone="muted" label="Typed name, checked by the worker" /></div>{/if}</th>
              <td class="mono xs" data-label="Shape">{t.shape ? t.shape.join('×') : '—'}</td>
              <td class="mono xs" data-label="Source type">{t.currentType ?? '—'}</td>
              <td data-label="Output type">
                <label class="sr-only" for="{uid}-o-{t.name}">Output type for {t.name}</label>
                <select id="{uid}-o-{t.name}" class="input sel" value={ov} onchange={(e) => setOne(t.name, e.currentTarget.value)}>
                  <option value="">Preset ({preset})</option>
                  {#each catalog.types as ty (ty.id)}<option value={ty.id}>{ty.id}{typeNote(ty.id)}</option>{/each}
                </select>
              </td>
              <td class="num r xs" data-label="Bytes">{ov ? (bytes === null ? '—' : formatBytes(bytes)) : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    {#if pages > 1}
      <nav class="pager" aria-label="Tensor pages">
        <button type="button" class="btn small" disabled={page === 0} onclick={() => (page -= 1)}>Previous</button>
        <span class="xs muted">Page {page + 1} of {pages}</span>
        <button type="button" class="btn small" disabled={page >= pages - 1} onclick={() => (page += 1)}>Next</button>
      </nav>
    {/if}
    <p class="xs muted">Bytes: payload of an overridden tensor from the catalog block size, when the shape is known. Preset tensor sizes are not estimated here.</p>
  {/if}

  <div class="row end">
    <div class="field grow">
      <label for="{uid}-name">Profile name</label>
      <input id="{uid}-name" class="input" bind:value={name} maxlength="80" autocomplete="off" aria-describedby="{uid}-name-h" />
      <span class="hint" id="{uid}-name-h">Saving with the same name adds a new revision; the imatrix is chosen at conversion time.</span>
    </div>
    <button type="button" class="btn primary" disabled={!built.body || saving || (!dirty && !!start)} onclick={save}>{saving ? 'Saving…' : 'Save GGUF revision'}</button>
  </div>
  {#if built.problems.length}<p class="why xs"><strong>Needed:</strong> {built.problems.join(' ')}</p>{/if}
  <div aria-live="polite">
    {#if saveErr}<p class="note bad" role="alert">{saveErr} Your changes are kept.</p>{/if}
    {#if saved}<div class="note info"><p>Saved <strong>{saved.name}</strong>, revision <span class="mono">{shortRev(saved.revision)}</span>. No job was started.</p><button type="button" class="btn small primary" onclick={() => useInConvert(saved!)}>Use in Convert</button></div>{/if}
  </div>
</div>

<style>
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

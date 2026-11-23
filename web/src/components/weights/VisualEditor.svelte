<script lang="ts">
  // Visual weight editor: tensor map -> tile of real stored values -> cells.
  // Reads go through weight_inspect jobs: one inventory page per 128 tensors
  // and one job per 2-D tile, never one per cell. Earlier reads with equal
  // options are reused. Every read and proposal carries the source fingerprint
  // from the inventory, so a changed file is refused.
  import { untrack } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';
  import TensorMap from './TensorMap.svelte';
  import HeatGrid from './HeatGrid.svelte';
  import Submit from '../Submit.svelte';
  import SubmitStatus from '../SubmitStatus.svelte';
  import JobState from '../JobState.svelte';
  import { brewery } from '../../lib/store.svelte';
  import { acceptedJobSummary, guardReadLink, Submitter } from '../../lib/submit.svelte';
  import type { PlainAttempt } from '../../lib/submission-key';
  import { captureResultFocus } from '../../lib/focus';
  import { route } from '../../lib/route.svelte';
  import { theme } from '../../lib/theme.svelte';
  import { isSafeSource, hostName, isActive, shortHash } from '../../lib/format';
  import { MAX_COORDINATES, MAX_EDITS, coord, exact, percent, relChange, signed, tuneJob, parseValue, parseMaxUpdate } from '../../lib/weights';
  import { editKey, type DraftEdit } from '../../lib/weight-draft';
  import {
    isDtype, isPacked, type Neighbors,
    findRead,
    inventoryJob,
    keyOf,
    legend,
    niceRanges,
    plan as planOf,
    walkInventory,
    readTile,
    roundTo,
    short,
    step,
    tileJob,
    tileRequest,
    type TensorInfo,
    type Tile,
  } from '../../lib/weightmap';
  import type { JobSummary, Model } from '../../lib/types';

  let {
    source,
    canVisual,
    canChange,
    canTune,
    calibrationId,
    heldOutId,
    tuneReason,
    mode = $bindable('manual'),
    onproposal,
    pending,
    readAttempts,
  }: {
    source: Model;
    canVisual: boolean;
    canChange: boolean;
    canTune: boolean;
    calibrationId: string;
    heldOutId: string;
    tuneReason: string | null;
    mode?: 'manual' | 'auto';
    onproposal: (jobId: string) => void;
    pending: SvelteMap<string, DraftEdit>;
    readAttempts: PlainAttempt;
  } = $props();

  const stateJobs = $derived(brewery.state?.jobs ?? []);
  let submitted = $state<JobSummary[]>([]);
  const jobs = $derived([...stateJobs, ...submitted.filter(j => !stateJobs.some(row => row.id === j.id))]);
  const models = $derived(brewery.state?.models ?? []);

  // Ignore old reports only when the user asks the worker to read again.
  let ignored = $state<Set<string>>(new Set());
  const packedHost = $derived(!isSafeSource(source.kind) && brewery.state?.hosts.find(h => h.id === source.host)?.capabilities?.weight_packed === true);
  const usable = $derived(jobs.filter((j) => !ignored.has(j.id)));
  const sourceHost = $derived(hostName(brewery.state?.hosts ?? [], source.host));
  const reads = new Submitter(job => job.options?.mode === 'tile' ? 'weights.visual.tile' : 'weights.visual.inventory',
    untrack(() => readAttempts));
  let readKey = $state('');
  let acceptedReadMode = $state<'inventory' | 'tile'>();
  let inventoryLink: HTMLAnchorElement | undefined = $state();
  let tileLink: HTMLAnchorElement | undefined = $state();

  function want(job: JobSummary | null): void {
    if (job?.state === 'succeeded' && !brewery.details[job.id]) void brewery.loadJob(job.id).catch(() => {});
  }
  async function read(job: ReturnType<typeof inventoryJob>, event: MouseEvent) {
    readKey = JSON.stringify(job.options);
    const focus = captureResultFocus(event);
    const created = await reads.submit(job, event, false);
    if (!created) return;
    acceptedReadMode = job.options?.mode === 'tile' ? 'tile' : 'inventory';
    function record(summary: JobSummary) {
      if (summary.id !== created?.id || !findRead([summary], source.id, job.options!, source.host)) return;
      submitted = [summary, ...submitted.filter(item => item.id !== summary.id)];
    }
    record(acceptedJobSummary(created, job));
    void focus?.(() => job.options?.mode === 'tile' ? tileLink : inventoryLink);
    if (!('state' in created)) void brewery.loadJob(created.id).then(record).catch(() => {});
  }
  function ignoreReads(match: (job: JobSummary) => boolean) {
    const next = new Set(ignored);
    for (const j of jobs) if (j.kind === 'weight_inspect' && j.input_ids[0] === source.id && !isActive(j.state) && match(j)) next.add(j.id);
    ignored = next;
  }

  // ---- Inventory, page by page. ----
  const inv = $derived(walkInventory(source, usable, brewery.details, { packed: packedHost, format: isSafeSource(source.kind) ? source.kind : undefined }));
  const sha = $derived(inv.pages[0]?.source_sha256 ?? null);
  const total = $derived(inv.pages[0]?.total ?? 0);
  const tensors = $derived(inv.pages.flatMap((p) => p.tensors.map(t => isPacked(t.dtype) && (!packedHost || (source.kind === 'hgn') !== (t.dtype === 'Q8G64')) ? {...t, reason: t.reason ?? 'Packed editing is not enabled for this source and worker.'} : t)));
  const inventoryRequest = $derived(inv.need ? inventoryJob(source.id, source.host, inv.need.offset, inv.need.sha, packedHost) : null);
  const inventoryError = $derived(inv.error ?? (inventoryRequest && readKey === JSON.stringify(inventoryRequest.options) ? reads.error : null));
  const lastInventoryRead = $derived(submitted.find(j => j.options?.mode === 'inventory' && !ignored.has(j.id)));
  const inventoryJobFound = $derived(inv.job ?? jobs.find(j => j.id === lastInventoryRead?.id));
  $effect(() => {
    void brewery.lastUpdated;
    if (canVisual) want(inv.job ?? null);
  });
  function readInventoryPage(event: MouseEvent) {
    if (!canVisual || (inv.job && !inv.error)) return;
    if (!inventoryRequest) {
      ignoreReads(j => j.options?.mode === 'inventory');
      void read(inventoryJob(source.id, source.host, 0, undefined, packedHost), event);
    } else void read(inventoryRequest, event);
  }

  // ---- Current tensor and tile. ----
  let tensorName = $state<string | null>(null);
  const tensor = $derived.by(() => {
    const t = tensors.find((t) => t.name === tensorName);
    return t && !t.reason && (isDtype(t.dtype) || isPacked(t.dtype)) ? { ...t, dtype: t.dtype } : null;
  });
  const float = $derived(tensor && isDtype(tensor.dtype) ? tensor.dtype : null);
  const p = $derived(tensor ? planOf(tensor.shape) : null);
  let lead = $state<number[]>([]);
  let tr = $state(0);
  let tc = $state(0);
  let selected = $state<Set<string>>(new Set());
  let focusKey = $state<string | null>(null);

  function openTensor(t: TensorInfo) {
    if (t.name === tensorName) return;
    tensorName = t.name;
    lead = t.shape.length > 2 ? t.shape.slice(0, -2).map(() => 0) : [];
    tr = 0;
    tc = 0;
    selected = new Set();
    focusKey = null;
    queueMicrotask(() => document.getElementById('vw-tile-h')?.scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }));
  }
  // The parent passes a new object on every poll; react to the ID only.
  const sourceId = $derived(source.id);
  let lastSource: string | null = null;
  $effect(() => {
    // A new source starts over.
    if (sourceId === lastSource) return;
    lastSource = sourceId;
    tensorName = null;
    selected = new Set();
    focusKey = null;
  });

  const req = $derived(p ? tileRequest(p, { lead, tr, tc }) : null);
  const tileRead = $derived(tensor && req && sha ? tileJob(source.id, source.host, sha, tensor.name, req, !float) : null);
  const tileOptions = $derived(tileRead?.options);
  const tileJobFound = $derived(tileOptions ? findRead(usable, source.id, tileOptions) : null);
  const tile = $derived.by((): Tile | string | null => {
    if (!tensor || !req || !sha || !tileJobFound) return null;
    if (tileJobFound.state !== 'succeeded') return isActive(tileJobFound.state) ? null : (tileJobFound.error ?? 'The tile could not be read.');
    const d = brewery.details[tileJobFound.id];
    if (d && !float && (d.result as {packed?:unknown} | null)?.packed !== true) return 'The worker did not confirm packed inspection.';
    return d ? readTile(d.result, { tensor: tensor.name, origin: req.origin, shape: req.shape, sha, dtype: tensor.dtype, full: tensor.shape }) : null;
  });
  const tileOk = $derived(typeof tile === 'object' ? tile : null);
  const tileError = $derived(typeof tile === 'string' ? tile : tileOptions && readKey === JSON.stringify(tileOptions) ? reads.error : null);
  const otherTileActive = $derived(jobs.some((j) => j.kind === 'weight_inspect' && j.input_ids[0] === source.id && (j.options as Record<string, unknown> | undefined)?.mode === 'tile' && isActive(j.state)));
  $effect(() => {
    void brewery.lastUpdated;
    if (canVisual) want(tileJobFound);
  });
  function readCurrentTile(event: MouseEvent) {
    const request = tileRead;
    const options = request?.options;
    if (!canVisual || !request || !options || otherTileActive || (tileJobFound && !tileError)) return;
    if (tileJobFound) ignoreReads(j => !!findRead([j], source.id, options));
    void read(request, event);
  }

  // Stored values seen in loaded tiles, for edits across tiles.
  const levels = new SvelteMap<string, Neighbors>();
  const known = new SvelteMap<string, number>();
  $effect(() => {
    const t = tileOk;
    if (!t) return;
    for (let i = 0; i < t.values.length; i++)
      for (let j = 0; j < t.values[i].length; j++) {
        const idx = t.shape.length === 1 ? [t.origin[0] + j] : [...t.origin.slice(0, -2), t.origin.at(-2)! + i, t.origin.at(-1)! + j];
        if (t.steps) levels.set(`${t.tensor}|${keyOf(idx)}`, t.steps[i][j]);
        known.set(`${t.tensor}|${keyOf(idx)}`, t.values[i][j]);
      }
  });

  // Tile status for the overview: only tiles that were really read carry a value.
  const tiles = $derived.by(() => {
    if (!p || !tensor || !sha || p.gridRows * p.gridCols > 1024) return null;
    const out: { tr: number; tc: number; state: 'none' | 'active' | 'read' | 'failed'; absMax: number | null }[] = [];
    for (let r = 0; r < p.gridRows; r++)
      for (let c = 0; c < p.gridCols; c++) {
        const rq = tileRequest(p, { lead, tr: r, tc: c });
        const j = findRead(usable, source.id, tileJob(source.id, source.host, sha, tensor.name, rq, !float).options!);
        const d = j?.state === 'succeeded' ? brewery.details[j.id] : undefined;
        const t = d ? readTile(d.result, { tensor: tensor.name, origin: rq.origin, shape: rq.shape, sha, dtype: tensor.dtype, full: tensor.shape }) : null;
        out.push({ tr: r, tc: c, state: !j ? 'none' : j.state === 'succeeded' ? 'read' : isActive(j.state) ? 'active' : 'failed', absMax: t && typeof t === 'object' ? t.absMax : null });
      }
    return out;
  });
  const tilesMax = $derived(Math.max(0, ...(tiles ?? []).map((t) => t.absMax ?? 0)));
  const range = (start: number, size: number, n: number) => `${start * size}–${Math.min(n, (start + 1) * size) - 1}`;

  // ---- Pending manual edits (all tensors), keyed tensor|index. ----
  let lastSha: string | null = null;
  $effect(() => {
    if (sha && lastSha !== sha) {
      if (lastSha) {
        for (const [key, edit] of pending) if (edit.observed) pending.delete(key);
        known.clear(); levels.clear(); selected = new Set(); focusKey = null;
      }
      lastSha = sha;
    }
  });
  const pendingHere = $derived(new Map([...pending.values()].filter((e) => e.tensor === tensorName && e.observed).map((e) => [keyOf(e.index), e.value])));
  let notice = $state<string | null>(null);

  function selectedCells(): { key: string; index: number[]; before: number }[] {
    if (!tensor) return [];
    const out = [];
    for (const k of selected) {
      const before = known.get(`${tensor.name}|${k}`);
      if (before !== undefined) out.push({ key: k, index: k.split(',').map(Number), before });
    }
    return out;
  }
  function setAfter(index: number[], before: number, after: number | null): boolean {
    if (!tensor || !sha || after === null) return false;
    const id = editKey({ tensor: tensor.name, index });
    const current = pending.get(id);
    if (current && !current.observed) return false;
    if (after === before) pending.delete(id);
    else if (!current && pending.size >= MAX_EDITS) return false;
    else pending.set(id, { tensor: tensor.name, index, op: 'set', value: after, observed: { before, sha, packed: isPacked(tensor.dtype) } });
    return true;
  }
  function apply(fn: (before: number, current: number) => number | null) {
    if (!canChange || mode !== 'manual' || !tensor) return;
    let refused = 0;
    for (const c of selectedCells()) {
      const current = pending.get(`${tensor.name}|${c.key}`);
      const cur = current?.observed ? current.value : c.before;
      if (!setAfter(c.index, c.before, fn(c.before, cur))) refused++;
    }
    notice = refused ? `${refused} ${refused === 1 ? 'cell was' : 'cells were'} not changed. Check the ${tensor.dtype} range, the edit limit, and whether the draft already has a text edit for that cell.` : null;
  }
  function selectTile() {
    const t = tileOk;
    if (!t) return;
    const next = new Set(selected);
    for (let i = 0; i < t.values.length; i++)
      for (let j = 0; j < t.values[i].length; j++) {
        const idx = t.shape.length === 1 ? [t.origin[0] + j] : [...t.origin.slice(0, -2), t.origin.at(-2)! + i, t.origin.at(-1)! + j];
        if (selectable(idx)) next.add(keyOf(idx));
      }
    selected = next;
  }
  function neighbor(direction: keyof Neighbors) {
    if (!canChange || !tensor || !tileOk?.steps || mode !== 'manual') return;
    let skipped = 0;
    for (const c of selectedCells()) {
      const value = levels.get(`${tensor.name}|${c.key}`)?.[direction];
      if (value == null || !setAfter(c.index, c.before, value)) skipped++;
    }
    notice = skipped ? `${skipped} selected cells did not change. Check the ${direction} level, the edit limit, and existing text edits.` : null;
  }
  function nudge(n: number) {
    const dtype = float;
    if (dtype) apply((_, cur) => step(dtype, cur, n));
    else if (Math.abs(n) === 1) neighbor(n > 0 ? 'upper' : 'lower');
  }
  const reset = () => apply((before) => before);

  const sel = $derived(selectedCells());
  const focusBefore = $derived(tensor && focusKey ? known.get(`${tensor.name}|${focusKey}`) : undefined);
  const focusAfter = $derived(focusKey ? pendingHere.get(focusKey) : undefined);

  // Slider: shift from the stored value, applied to every selected cell.
  const ranges = $derived(niceRanges(tileOk?.absMax ?? 0));
  let rangeIdx = $state(1);
  const R = $derived(ranges[Math.min(rangeIdx, ranges.length - 1)]);
  let slide = $state(0);
  $effect(() => {
    // Follow the focused cell when the selection changes.
    void selected;
    slide = focusBefore !== undefined && focusAfter !== undefined ? focusAfter - focusBefore : 0;
  });
  function onSlide(v: number) {
    slide = v;
    apply((before) => roundTo(float!, before + v));
  }
  let exactText = $state('');
  const exactParsed = $derived(parseValue(exactText));

  // ---- Submit. ----
  const MAX_STEPS = 60;
  let maxIdx = $state(30);
  const maxFromSlider = (i: number) => Number((10 ** (-6 + (i / MAX_STEPS) * 6)).toPrecision(2));
  let maxText = $state('');
  const nearCap = $derived.by(() => {
    if (!tensor || float) return 0;
    let cap = 0;
    for (const c of sel) {
      const neighbors = levels.get(`${tensor.name}|${c.key}`);
      if (neighbors) for (const v of [neighbors.lower, neighbors.upper]) if (v !== null) cap = Math.max(cap, Math.abs(v - c.before));
    }
    return cap;
  });
  const maxUpdate = $derived(maxText.trim() ? parseMaxUpdate(maxText) : ({ ok: true, value: maxFromSlider(maxIdx) } as const));
  const tuneSub = new Submitter('weights.visual.tune');
  const sameHost = (id: string) => models.find((m) => m.id === id)?.host === source.host;
  const tuneBlockers = $derived.by(() => {
    const out: string[] = [];
    if (!canTune) out.push(tuneReason ?? 'Automatic tuning is not enabled on this worker.');
    if (!tensor) out.push('Open a tensor.');
    else if (!float && !tileOk?.steps) out.push('Read verified packed levels first.');
    else if (tensor.shape.length !== 2) out.push('Tuning needs a rank-2 tensor.');
    if (!selected.size) out.push('Select weights in the tile.');
    else if (!float && nearCap === 0) out.push('The selected cells have no other decoded level.');
    else if (selected.size > MAX_COORDINATES) out.push(`Select at most ${MAX_COORDINATES} weights (${selected.size} selected).`);
    if (!maxUpdate.ok) out.push(maxUpdate.error);
    if (!calibrationId) out.push('Select a calibration split.');
    if (!heldOutId) out.push('Select a held-out split.');
    else if (heldOutId === calibrationId) out.push('Use a different entry for the held-out split.');
    if ((calibrationId && !sameHost(calibrationId)) || (heldOutId && !sameHost(heldOutId))) out.push(`Both splits must be on ${hostName(brewery.state?.hosts ?? [], source.host)}.`);
    return out;
  });
  const tuneRequest = $derived.by(() => {
    if (!tensor || !sha || !maxUpdate.ok) return null;
    const indices = [...selected].map((k) => k.split(',').map(Number) as [number, number]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const job = tuneJob(source.id, source.host, calibrationId, heldOutId, tensor.name, indices, maxUpdate.value);
    job.options = { ...job.options, source_sha256: sha, ...(!float ? {packed:true} : {}) };
    return job;
  });
  async function tune(event: MouseEvent) {
    if (!tuneRequest || tuneBlockers.length) return;
    const visit = route.capture();
    const created = await tuneSub.submit(tuneRequest, event);
    if (created && !tuneSub.replayed && visit.canPresent()) onproposal(created.id);
  }

  const selectable = $derived((index: number[]) => canChange && (mode === 'manual' || index.length === 2));
  const scheme = $derived(theme.resolved === 'light' ? 'light' : 'dark');
</script>

{#if !canVisual}
  <p class="note warn small">The visual editor needs <code>weight_inspect</code> and <code>weight_visual</code> on {hostName(brewery.state?.hosts ?? [], source.host)}. Check these capabilities in <a href="#/settings/infrastructure?host={encodeURIComponent(source.host)}">worker settings</a>.{#if source.kind === 'gguf'} You can also use text entry below.{/if}</p>
{:else}
  <div class="ve stack">
    <!-- 1. Tensors -->
    <section class="step" aria-labelledby="vw-map-h">
      <div class="row between">
        <h3 class="h" id="vw-map-h">1. Tensor</h3>
        <div class="row xs muted">
          {#if sha}<span>Source <span class="mono" title={sha}>{shortHash(sha)}</span> · {tensors.length} of {total} tensors</span>{/if}
        </div>
      </div>
      {#if inventoryJobFound}
        <div class="row" role="status">
          <a bind:this={inventoryLink} href="#/jobs/{encodeURIComponent(inventoryJobFound.id)}" onclick={guardReadLink}>Tensor list read job</a>
          <JobState job={inventoryJobFound} />
        </div>
      {/if}
      {#if !inv.job || inv.error}
        <div class="row">
          <button type="button" class="btn small" onclick={readInventoryPage} disabled={reads.busy}>
            {!inventoryRequest ? 'Read tensor list again' : inv.pages.length ? 'Read next tensors' : 'Read tensor list'}
          </button>
          <span class="hint">This queues a <code>weight_inspect</code> job on {sourceHost}.</span>
        </div>
      {/if}
      {#if source.sha256 && sha && source.sha256 !== sha}
        <p class="note warn small">The Library hash ({shortHash(source.sha256)}) differs from the file the worker read ({shortHash(sha)}). Edits use the file the worker read.</p>
      {/if}
      {#if inventoryError}<p class="note bad" role="alert">{inventoryError}</p>{/if}
      {#if reads.replayed && acceptedReadMode === 'inventory'}
        <SubmitStatus s={reads} />
      {/if}
      {#if tensors.length}
        <TensorMap {tensors} selected={tensorName} onopen={openTensor} />
      {:else if inv.job && !inv.error}
        <p class="muted small" role="status">Waiting for the tensor list.</p>
      {:else if !inventoryRequest && !inventoryError}
        <p class="muted small">No tensors were found.</p>
      {/if}
    </section>

    <!-- 2. Tile -->
    {#if tensor && p}
      <section class="step" aria-labelledby="vw-tile-h">
        <div class="row between">
          <h3 class="h" id="vw-tile-h" tabindex="-1">2. Values <span class="mono break tn">{tensor.name}</span></h3>
          <span class="xs muted">{tensor.shape.join(' × ')} · {tensor.dtype} · {tensor.elements.toLocaleString()} values</span>
        </div>

        <div class="nav">
          {#each p.lead as size, d (d)}
            <div class="field">
              <label for="vw-lead-{d}">{p.rank === 3 && d === 0 ? 'Slice (dim 0)' : `Dim ${d}`}</label>
              {#if size <= 1024}
                <select id="vw-lead-{d}" class="input" value={lead[d]} onchange={(e) => { lead[d] = Number(e.currentTarget.value); }}>
                  {#each Array.from({ length: size }, (_, i) => i) as i (i)}<option value={i}>{i}</option>{/each}
                </select>
              {:else}
                <input id="vw-lead-{d}" class="input" type="number" min="0" max={size - 1} value={lead[d]} onchange={(e) => { lead[d] = Math.max(0, Math.min(size - 1, Math.trunc(Number(e.currentTarget.value)) || 0)); }} />
              {/if}
            </div>
          {/each}
          {#if p.gridRows > 1}
            <div class="field">
              <label for="vw-tr">Rows</label>
              <select id="vw-tr" class="input" bind:value={tr}>
                {#each Array.from({ length: p.gridRows }, (_, i) => i) as i (i)}<option value={i}>{range(i, p.tileRows, p.rows)}</option>{/each}
              </select>
            </div>
          {/if}
          {#if p.gridCols > 1}
            <div class="field">
              <label for="vw-tc">{p.rank === 1 ? 'Elements' : 'Columns'}</label>
              <select id="vw-tc" class="input" bind:value={tc}>
                {#each Array.from({ length: p.gridCols }, (_, i) => i) as i (i)}<option value={i}>{range(i, p.tileCols, p.cols)}</option>{/each}
              </select>
            </div>
          {/if}
          {#if tiles && tiles.length > 1}
            <div class="overview" aria-hidden="true" style:grid-template-columns="repeat({p.gridCols}, 10px)" title="Tiles of this slice. Filled tiles were read; the shade is their largest absolute value.">
              {#each tiles as t (`${t.tr}-${t.tc}`)}
                <button
                  type="button"
                  tabindex="-1"
                  aria-label="Tile {t.tr + 1}, {t.tc + 1}"
                  class="ot {t.state}"
                  class:cur={t.tr === tr && t.tc === tc}
                  style:--fill={t.absMax !== null && tilesMax > 0 ? `${Math.round(25 + 75 * (t.absMax / tilesMax))}%` : '0%'}
                  onclick={() => { tr = t.tr; tc = t.tc; }}
                ></button>
              {/each}
            </div>
          {/if}
        </div>

        {#if tileJobFound}
          <div class="row" role="status">
            <a bind:this={tileLink} href="#/jobs/{encodeURIComponent(tileJobFound.id)}" onclick={guardReadLink}>Tile read job</a>
            <JobState job={tileJobFound} />
          </div>
        {/if}
        {#if tileError}<p class="note bad" role="alert">{tileError}</p>{/if}
        {#if reads.replayed && acceptedReadMode === 'tile'}
          <SubmitStatus s={reads} />
        {/if}
        {#if !tileJobFound || tileError}
          <div class="row">
            <button type="button" class="btn small" onclick={readCurrentTile} disabled={reads.busy || otherTileActive}>{tileError ? 'Read tile again' : 'Read tile'}</button>
            <span class="hint">This queues a <code>weight_inspect</code> job on {sourceHost}.</span>
          </div>
          {#if otherTileActive}<p class="hint" role="status">Wait for the previous tile read to finish.</p>{/if}
        {/if}
        {#if tileOk}
          <div class="tile">
            <HeatGrid
              tile={tileOk}
              pending={pendingHere}
              bind:selected
              bind:focusKey
              {selectable}
              onnudge={nudge}
              onreset={reset}
              label="Stored values of {tensor.name}{canChange ? (mode === 'manual' ? ', select cells to edit' : ', select weights to tune') : ', read-only'}"
            />
            <div class="scale xs" aria-label="Colour scale from {short(-tileOk.absMax)} to {short(tileOk.absMax)}, this tile only">
              <span class="num">{short(-tileOk.absMax)}</span>
              <span class="bar" style:background={legend(scheme)}></span>
              <span class="num">{short(tileOk.absMax)}</span>
              <span class="muted">this tile only · <span class="mark" aria-hidden="true"></span> edited</span>
            </div>
          </div>
        {/if}
      </section>

      <!-- 3. Edit or tune -->
      {#if canChange}<section class="step" aria-labelledby="vw-edit-h">
        <div class="row between">
          <h3 class="h" id="vw-edit-h">3. {mode === 'manual' ? 'Change values' : 'Tune selected weights'}</h3>
          <div class="row">
            <button type="button" class="btn ghost small" disabled={!tileOk} onclick={selectTile}>Select whole tile</button>
            <button type="button" class="btn ghost small" disabled={!selected.size} onclick={() => (selected = new Set())}>Clear selection</button>
          </div>
        </div>
        <p class="xs muted" aria-live="polite">
          {selected.size} {selected.size === 1 ? 'cell' : 'cells'} selected{#if focusKey && focusBefore !== undefined}{' · '}focus <span class="mono">{coord(focusKey.split(',').map(Number))}</span> stored <span class="mono">{exact(focusBefore)}</span>{#if focusAfter !== undefined}{' → '}<span class="mono">{exact(focusAfter)}</span>{/if}{/if}.
          Click or drag to select; Shift and Ctrl add; on a touch screen, tap to add or remove. In the grid, + and − nudge, Delete resets.
        </p>

        {#if mode === 'manual'}
          {#if !float}
            <p class="small">Each button requests the next decoded value above or below the original. The block scale stays fixed.</p>
            <div class="row" role="group" aria-label="Packed code levels">
              {#each ['lower','upper'] as direction}
                {@const dir = direction as keyof Neighbors}
                {@const target = focusKey && levels.get(`${tensor.name}|${focusKey}`)?.[dir]}
                <button type="button" class="btn small" disabled={!tileOk?.steps || !sel.some(c => levels.get(`${tensor.name}|${c.key}`)?.[dir] != null)} onclick={() => neighbor(dir)}>{dir === 'lower' ? 'Lower' : 'Higher'} level{#if sel.length === 1 && typeof target === 'number'}: {exact(target)}{/if}</button>
              {/each}
              <button type="button" class="btn ghost small" disabled={!sel.length} onclick={reset}>Reset</button>
            </div>
          {:else}
          <div class="ctl">
            <div class="group" role="group" aria-label="Nudge selected cells by stored steps">
              <span class="glabel xs muted">Nudge</span>
              <div class="row nudges">
                <button type="button" class="btn small" disabled={!sel.length} onclick={() => nudge(-10)} aria-label="Lower by 10 stored steps">−10</button>
                <button type="button" class="btn small" disabled={!sel.length} onclick={() => nudge(-1)} aria-label="Lower by 1 stored step">−1</button>
                <button type="button" class="btn small" disabled={!sel.length} onclick={() => nudge(1)} aria-label="Raise by 1 stored step">+1</button>
                <button type="button" class="btn small" disabled={!sel.length} onclick={() => nudge(10)} aria-label="Raise by 10 stored steps">+10</button>
                <button type="button" class="btn ghost small" disabled={!sel.length} onclick={reset}>Reset</button>
              </div>
              <span class="xs muted">One step is the next value that {tensor.dtype} can store.</span>
            </div>
            <div class="group slider">
              <label class="glabel xs muted" for="vw-slide">Shift from stored value</label>
              <input
                id="vw-slide"
                type="range"
                min={-R}
                max={R}
                step={R / 100}
                value={slide}
                disabled={!sel.length}
                aria-valuetext="{signed(Number(slide.toPrecision(3)))}"
                oninput={(e) => onSlide(Number(e.currentTarget.value))}
              />
              <div class="row between xs">
                <span class="num mono">{signed(Number(slide.toPrecision(3)))}{#if focusBefore} <span class="muted">({percent(relChange(focusBefore, focusBefore + slide))})</span>{/if}</span>
                <span class="seg" role="radiogroup" aria-label="Slider range">
                  {#each ranges as r, i (i)}
                    <label class:on={rangeIdx === i}><input type="radio" name="vw-range" value={i} bind:group={rangeIdx} />±{short(r)}</label>
                  {/each}
                </span>
              </div>
            </div>
          </div>
          <details class="disclosure">
            <summary>Exact value</summary>
            <div class="row pad">
              <label class="sr-only" for="vw-exact">Exact value or change</label>
              <input id="vw-exact" class="input mono" bind:value={exactText} inputmode="decimal" autocomplete="off" spellcheck="false" placeholder="0.125" aria-invalid={!!exactText.trim() && !exactParsed.ok} />
              <button type="button" class="btn small" disabled={!sel.length || !exactParsed.ok} onclick={() => exactParsed.ok && apply(() => roundTo(float!, exactParsed.value))}>Set to value</button>
              <button type="button" class="btn small" disabled={!sel.length || !exactParsed.ok} onclick={() => exactParsed.ok && apply((before) => roundTo(float!, before + exactParsed.value))}>Add to stored value</button>
              {#if exactText.trim() && !exactParsed.ok}<span class="xs ferr">{exactParsed.error}</span>{/if}
            </div>
          </details>
          {/if}
          {#if notice}<p class="note warn small" role="status">{notice}</p>{/if}
        {:else}
          {#if tensor.shape.length !== 2}<p class="note warn small">Automatic tuning works on rank-2 tensors only. Open a matrix such as <span class="mono">ffn_down.weight</span>.</p>{/if}
          {#if !float}<div class="row"><button type="button" class="btn small" disabled={!nearCap} onclick={() => maxText = String(nearCap)}>Use adjacent-level limit</button><span class="xs muted">{nearCap ? `Largest adjacent change in this selection: ${exact(nearCap)}.` : 'Select cells with another decoded level.'}</span></div>{/if}
          <div class="grid-2">
            <div class="group slider">
              <label class="glabel xs muted" for="vw-max">Largest change per weight</label>
              <input id="vw-max" type="range" min="0" max={MAX_STEPS} step="1" bind:value={maxIdx} disabled={!!maxText.trim()} aria-valuetext={String(maxFromSlider(maxIdx))} />
              <span class="xs num mono">{maxUpdate.ok ? exact(maxUpdate.value) : '—'}</span>
            </div>
            <div class="field">
              <label for="vw-max-x">Exact limit (optional)</label>
              <input id="vw-max-x" class="input mono" bind:value={maxText} inputmode="decimal" autocomplete="off" spellcheck="false" placeholder={String(maxFromSlider(maxIdx))} aria-invalid={!maxUpdate.ok} />
            </div>
          </div>
        {/if}
      </section>
      {/if}
    {/if}

    {#if canChange && mode === 'auto' && tensor}
      <section class="step">
        <SubmitStatus s={tuneSub} />
        <Submit blockers={tuneBlockers} busy={tuneSub.busy} onclick={tune}>Tune {selected.size || ''} selected {selected.size === 1 ? 'weight' : 'weights'}</Submit>
      </section>
    {/if}
  </div>
{/if}

<style>
  .ve {
    gap: var(--s-5);
  }
  .step {
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
    min-width: 0;
  }
  .step + .step {
    padding-top: var(--s-4);
    border-top: 1px solid var(--line-soft);
  }
  .h {
    margin: 0;
    font-size: var(--fs-md);
    font-weight: 600;
  }
  .h:focus {
    outline: none;
  }
  /* Keep headings clear of the sticky top bar when scrolled into view. */
  .step {
    scroll-margin-top: 72px;
  }
  .h {
    scroll-margin-top: 72px;
  }
  .tn {
    font-weight: 500;
    font-size: var(--fs-sm);
    margin-left: 6px;
  }
  .between {
    justify-content: space-between;
  }
  .nav {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: var(--s-3);
  }
  .nav select,
  .nav input {
    min-width: 7rem;
  }
  .overview {
    display: grid;
    gap: 2px;
    padding: 4px;
    border: 1px solid var(--line-soft);
    border-radius: var(--r-sm);
    align-self: center;
  }
  .ot {
    width: 10px;
    height: 10px;
    padding: 0;
    border: 1px solid var(--line);
    border-radius: 2px;
    cursor: pointer;
    background: repeating-linear-gradient(45deg, transparent, transparent 2px, var(--line-soft) 2px, var(--line-soft) 4px);
  }
  .ot.read {
    background: color-mix(in srgb, var(--accent) var(--fill), var(--surface));
  }
  .ot.active {
    border-color: var(--info);
  }
  .ot.failed {
    border-color: var(--bad);
  }
  .ot.cur {
    outline: 2px solid var(--text);
    outline-offset: 1px;
  }
  .tile {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    min-width: 0;
  }
  .scale {
    display: flex;
    align-items: center;
    gap: var(--s-2);
    flex-wrap: wrap;
  }
  .bar {
    width: 140px;
    height: 8px;
    border-radius: var(--r-pill);
  }
  .mark {
    display: inline-block;
    border-style: solid;
    border-width: 0 7px 7px 0;
    border-color: transparent var(--text) transparent transparent;
    vertical-align: middle;
  }
  .ctl {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: var(--s-4) var(--s-5);
    align-items: start;
  }
  .group {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }
  .glabel {
    font-weight: 600;
  }
  .nudges .btn {
    min-width: 2.75rem;
  }
  .slider input[type='range'] {
    width: 100%;
    accent-color: var(--accent);
  }
  .seg {
    display: inline-flex;
    padding: 2px;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
  }
  .seg label {
    position: relative;
    padding: 2px 8px;
    border-radius: var(--r-sm);
    cursor: pointer;
    color: var(--text-2);
  }
  .seg label.on {
    background: var(--accent);
    color: var(--accent-text);
    font-weight: 600;
  }
  .seg input {
    position: absolute;
    inset: 0;
    opacity: 0;
    margin: 0;
    cursor: pointer;
  }
  .seg label:has(input:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 1px;
  }
  .pad {
    padding-top: var(--s-3);
  }
  .pad .input {
    width: 12rem;
  }
  .ferr {
    color: var(--bad);
  }
  @media (max-width: 640px) {
    .ctl {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>

<script lang="ts">
  import { onMount } from 'svelte';
  import ModelSelect from '../components/ModelSelect.svelte';
  import Submit from '../components/Submit.svelte';
  import Badge from '../components/Badge.svelte';
  import JobState from '../components/JobState.svelte';
  import JobProgress from '../components/JobProgress.svelte';
  import CancelButton from '../components/CancelButton.svelte';
  import WeightReport from '../components/weights/WeightReport.svelte';
  import VisualEditor from '../components/weights/VisualEditor.svelte';
  import CalibrationSplit from '../components/weights/CalibrationSplit.svelte';
  import { brewery } from '../lib/store.svelte';
  import { errorText } from '../lib/api';
  import { Submitter } from '../lib/submit.svelte';
  import { hashParam } from '../lib/route.svelte';
  import { WEIGHT_SOURCES, isSafeSource, capability, formatTime, hostName, isActive, isTerminal, kindTitle, modelLabel, newestFirst, shortHash } from '../lib/format';
  import {
    MAX_COORDINATES,
    MAX_EDITS,
    PROPOSAL_KINDS,
    WEIGHT_KINDS,
    appliesOf,
    applyJob,
    checkEdits,
    exact,
    inspectJob,
    jobsFor,
    parseCoordinates,
    parseEditLines,
    parseIndex,
    parseMaxUpdate,
    parseTensor,
    proposeJob,
    readApplied,
    readInspect,
    readProposal,
    readTuneReport,
    tuneJob,
    sentence,
    type EditRow,
    type Op,
  } from '../lib/weights';
  import type { JobDetail } from '../lib/types';

  const models = $derived(brewery.state?.models ?? []);
  const hosts = $derived(brewery.state?.hosts ?? []);
  const jobs = $derived(brewery.state?.jobs ?? []);

  // ---- Source (shared by both modes). ----
  let sourceId = $state(hashParam(window.location.hash, 'source') ?? '');
  onMount(() => {
    const read = () => {
      const s = hashParam(window.location.hash, 'source');
      if (s) sourceId = s;
    };
    window.addEventListener('hashchange', read);
    return () => window.removeEventListener('hashchange', read);
  });
  $effect(() => {
    // Keep the address shareable without a new history entry.
    const want = sourceId ? `#/weights?source=${encodeURIComponent(sourceId)}` : '#/weights';
    if (window.location.hash !== want) history.replaceState(null, '', want);
  });

  const source = $derived(models.find((m) => m.id === sourceId && WEIGHT_SOURCES.includes(m.kind)));
  const sourceHost = $derived(source ? hosts.find((h) => h.id === source.host) : undefined);
  const hgn = $derived(source?.kind === 'hgn');
  const safe = $derived(isSafeSource(source?.kind));
  const native = $derived(capability(sourceHost, safe ? 'weight_safetensors' : 'weight_hgn'));
  const formatAllowed = $derived(!(hgn || safe) || native.allowed);
  const sourceFormat = $derived(source?.kind === 'hf' ? 'HF checkpoint' : safe ? 'SafeTensors file' : hgn ? 'HGN' : 'GGUF');
  const caps = $derived(Object.fromEntries(WEIGHT_KINDS.map((k) => [k, capability(sourceHost, k)])) as Record<(typeof WEIGHT_KINDS)[number], ReturnType<typeof capability>>);
  const capLabels = { weight_inspect: 'Inspect', weight_propose: 'Manual proposal', weight_tune: 'Automatic tuning', weight_apply: 'Apply' } as const;
  const visual = $derived(capability(sourceHost, 'weight_visual'));
  const canVisual = $derived(!!source && source.available !== false && caps.weight_inspect.allowed && visual.allowed && formatAllowed);
  const sourceBlockers = $derived.by(() => {
    if (!sourceId) return ['Select a registered source.'];
    if (!source) return ['This Library entry is not a supported weight source.'];
    if (!formatAllowed) return [sentence(native.reason ?? 'Editing this format is disabled on this worker.')];
    if (source.available === false) return [sentence(source.reason ?? 'The source file is not available.')];
    return [];
  });
  function gate(kind: (typeof WEIGHT_KINDS)[number]): string[] {
    if (sourceBlockers.length) return sourceBlockers;
    const c = caps[kind];
    return c.allowed ? [] : [sentence(c.reason ?? `${capLabels[kind]} is not enabled on this worker.`)];
  }

  // ---- Mode. ----
  let mode = $state<'manual' | 'auto'>('manual');

  // ---- Manual edits. ----
  let seq = 0;
  const nextKey = () => `e${seq++}`;
  const blankRow = (tensor = ''): EditRow => ({ key: nextKey(), tensor, index: '', op: 'set', value: '' });
  let rows = $state<EditRow[]>([blankRow()]);
  let touched = $state(false);
  const manual = $derived(checkEdits(rows));
  function addRow() {
    if (rows.length >= MAX_EDITS) return;
    rows.push(blankRow(rows.at(-1)?.tensor ?? ''));
    const key = rows.at(-1)!.key;
    queueMicrotask(() => document.getElementById(`w-${key}-index`)?.focus());
  }
  function removeRow(i: number) {
    rows.splice(i, 1);
    if (rows.length === 0) rows.push(blankRow());
  }
  let bulkText = $state('');
  let bulkError = $state<string | null>(null);
  function addBulk() {
    const parsed = parseEditLines(bulkText, nextKey);
    if (!parsed.ok) {
      bulkError = parsed.error;
      return;
    }
    const kept = rows.filter((r) => r.tensor.trim() || r.index.trim() || r.value.trim());
    if (kept.length + parsed.value.length > MAX_EDITS) {
      bulkError = `That makes ${kept.length + parsed.value.length} edits. Use at most ${MAX_EDITS}.`;
      return;
    }
    rows = [...kept, ...parsed.value];
    bulkText = '';
    bulkError = null;
    touched = true;
  }
  const proposeSub = new Submitter();
  const proposeBlockers = $derived([...gate('weight_propose'), ...(manual.blocker ? [manual.blocker] : [])]);
  async function propose() {
    touched = true;
    if (!source || !manual.edits) return;
    const job = await proposeSub.submit(proposeJob(source.id, source.host, manual.edits));
    if (job) reviewId = job.id;
  }

  // ---- Inspect helper. ----
  let inspectTensor = $state('');
  let inspectIndex = $state('');
  const inspectParsed = $derived({ t: parseTensor(inspectTensor), i: parseIndex(inspectIndex) });
  const inspectSub = new Submitter();
  const inspectBlockers = $derived([
    ...gate('weight_inspect'),
    ...(inspectParsed.t.ok ? [] : [inspectParsed.t.error]),
    ...(inspectParsed.i.ok ? [] : [inspectParsed.i.error]),
  ]);
  async function inspect() {
    if (!source || !inspectParsed.t.ok || !inspectParsed.i.ok) return;
    await inspectSub.submit(inspectJob(source.id, source.host, inspectParsed.t.value, inspectParsed.i.value));
  }
  const lastInspect = $derived(newestFirst(jobsFor(jobs, sourceId, new Set(['weight_inspect'])))[0]);
  let inspectDetail = $state<JobDetail | null>(null);
  $effect(() => {
    const j = lastInspect;
    inspectDetail = null;
    if (!j || j.state !== 'succeeded') return;
    const id = j.id;
    brewery.loadJob(id).then((d) => {
      if (lastInspect?.id === id) inspectDetail = d;
    }, () => {});
  });
  const inspected = $derived(inspectDetail ? readInspect(inspectDetail.result) : null);
  function useInspected() {
    if (!inspected) return;
    const row = { key: nextKey(), tensor: inspected.tensor, index: inspected.index.join(', '), op: 'set' as Op, value: exact(inspected.value) };
    const empty = rows.length === 1 && !rows[0].tensor.trim() && !rows[0].index.trim() && !rows[0].value.trim();
    if (empty) rows = [row];
    else if (rows.length < MAX_EDITS) rows.push(row);
    queueMicrotask(() => document.getElementById(`w-${row.key}-value`)?.focus());
  }

  // ---- Automatic tuning. ----
  let tuneTensor = $state('');
  let tuneCoords = $state('');
  let tuneMax = $state('');
  let calibrationId = $state('');
  let heldOutId = $state('');
  $effect(() => { void sourceId; calibrationId = ''; heldOutId = ''; });
  let tuneTouched = $state(false);
  const tuneParsed = $derived({ t: parseTensor(tuneTensor), c: parseCoordinates(tuneCoords), m: parseMaxUpdate(tuneMax) });
  const coordCount = $derived(tuneCoords.split('\n').filter((l) => l.trim()).length);
  const tuneSub = new Submitter();
  const sameHost = (id: string) => models.find((m) => m.id === id)?.host === source?.host;
  const tuneBlockers = $derived.by(() => {
    const out = [...gate('weight_tune')];
    if (!tuneParsed.t.ok) out.push(tuneParsed.t.error);
    if (!tuneParsed.c.ok) out.push(tuneParsed.c.error);
    if (!tuneParsed.m.ok) out.push(tuneParsed.m.error);
    if (!calibrationId) out.push('Select a calibration split.');
    if (!heldOutId) out.push('Select a held-out split.');
    else if (heldOutId === calibrationId) out.push('Use a different entry for the held-out split.');
    if (source && ((calibrationId && !sameHost(calibrationId)) || (heldOutId && !sameHost(heldOutId)))) out.push(`Both splits must be on ${hostName(hosts, source.host)}, the host of the source.`);
    return out;
  });
  async function tune() {
    tuneTouched = true;
    if (!source || !tuneParsed.t.ok || !tuneParsed.c.ok || !tuneParsed.m.ok || tuneBlockers.length) return;
    const job = await tuneSub.submit(tuneJob(source.id, source.host, calibrationId, heldOutId, tuneParsed.t.value, tuneParsed.c.value, tuneParsed.m.value));
    if (job) reviewId = job.id;
  }

  // ---- Review and apply. ----
  const proposals = $derived(newestFirst(jobsFor(jobs, sourceId, PROPOSAL_KINDS)).slice(0, 20));
  let reviewId = $state<string | null>(null);
  const reviewed = $derived(proposals.find((j) => j.id === reviewId) ?? null);
  let detail = $state<JobDetail | null>(null);
  let detailError = $state<string | null>(null);
  $effect(() => {
    const j = reviewed;
    detail = null;
    detailError = null;
    if (!j || !isTerminal(j.state)) return;
    const id = j.id;
    brewery.loadJob(id).then(
      (d) => {
        if (reviewId === id) detail = d;
      },
      (e) => {
        if (reviewId === id) detailError = errorText(e);
      },
    );
  });
  const proposal = $derived(detail ? readProposal(detail.result) : null);
  const tuneRegress = $derived(detail ? (readTuneReport(detail.result)?.held_out.regression ?? false) : false);
  const applies = $derived(reviewed ? newestFirst(appliesOf(jobs, reviewed.id)) : []);
  const activeApply = $derived(applies.find((j) => isActive(j.state)));
  const doneApply = $derived(applies.find((j) => j.state === 'succeeded'));
  let confirmed = $state(false);
  $effect(() => {
    // A new proposal needs a new review.
    void reviewId;
    confirmed = false;
  });
  const applySub = new Submitter();
  const applyBlockers = $derived.by(() => {
    const out = [...gate('weight_apply')];
    if (!reviewed) return [...out, 'Select a proposal.'];
    if (reviewed.state !== 'succeeded') out.push('The proposal job has not succeeded.');
    else if (!proposal) out.push('The proposal cannot be read.');
    if (activeApply) out.push('An apply job for this proposal is already running.');
    if (doneApply) out.push('This proposal is already applied.');
    if (!confirmed) out.push('Confirm that you reviewed the values.');
    return out;
  });
  async function applyNow() {
    if (!reviewed || !source || applyBlockers.length) return;
    // The controller requires the exact source of the proposal job.
    await applySub.submit(applyJob(reviewed.input_ids[0], source.host, reviewed.id));
  }
  let appliedDetail = $state<JobDetail | null>(null);
  $effect(() => {
    const j = doneApply;
    appliedDetail = null;
    if (!j) return;
    const id = j.id;
    brewery.loadJob(id).then((d) => {
      if (doneApply?.id === id) appliedDetail = d;
    }, () => {});
  });
  const applied = $derived(appliedDetail ? readApplied(appliedDetail.result) : null);
  const appliedModel = $derived(applied?.library_model_id ? models.find((m) => m.id === applied.library_model_id) : undefined);

  const err = (show: boolean, msg: string | undefined) => (show && msg ? msg : undefined);
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Weight edits</h1>
    <p class="muted small">
      Select weights on the source host, then adjust them by hand or tune them on a calibration split. Review the proposal before applying it as a new revision in the original format. The source stays unchanged.
    </p>
  </div>

  <p class="note info small">
    Each proposal can contain at most {MAX_EDITS} manual edits or {MAX_COORDINATES} tuned weights in one rank-2 tensor per proposal. Measured errors describe one layer output. They do not show model quality, and an edited file is not release-qualified.
  </p>

  <section class="panel" aria-labelledby="w-src-h">
    <header><h2 id="w-src-h">Source</h2>{#if source}<a class="xs" href="#/library">Library</a>{/if}</header>
    <div class="body stack">
      <ModelSelect bind:value={sourceId} kinds={WEIGHT_SOURCES} label="Source model" id="w-source" hint="Choose a GGUF, native HGN, standalone SafeTensors file or indexed HF checkpoint. Jobs run on the host of that entry." />
      {#if sourceId && !source}
        <p class="note warn">This entry is missing or has an unsupported source format.</p>
      {:else if source}
        <dl class="kv small">
          <dt>Host</dt><dd>{hostName(hosts, source.host)}</dd>
          <dt>Path</dt><dd class="mono xs break">{source.path}</dd>
          <dt>Format</dt><dd>{sourceFormat}{#if source.kind === 'hf'} · indexed SafeTensors directory. Apply preserves the index and shard layout.{/if}</dd>
          <dt>{source.kind === 'hf' ? 'Bundle SHA-256' : 'SHA-256'}</dt><dd>{#if source.sha256}<span class="mono" title={source.sha256}>{shortHash(source.sha256)}</span>{:else}<span class="muted">No digest in the Library record. The worker binds edits to the source fingerprint.</span>{/if}</dd>
          {#if source.job_id}<dt>Made by</dt><dd><a class="mono xs" href="#/jobs/{encodeURIComponent(source.job_id)}">{source.job_id}</a></dd>{/if}
        </dl>
        <ul class="caps" aria-label="Weight editing on {hostName(hosts, source.host)}">
          {#each WEIGHT_KINDS as k (k)}
            <li>
              <Badge tone={caps[k].allowed && formatAllowed ? 'ok' : 'muted'} label="{capLabels[k]}: {caps[k].allowed && formatAllowed ? 'enabled' : 'not enabled'}" title={caps[k].reason ?? ''} />
            </li>
          {/each}
        </ul>
        {#if caps.weight_inspect.allowed && formatAllowed}
          <p class="xs muted">Visual reads: {visual.allowed ? 'enabled' : sentence(visual.reason ?? 'not enabled')}</p>
        {/if}
        <a class="small" href="#/settings/infrastructure?host={encodeURIComponent(source.host)}">Settings for {hostName(hosts, source.host)}</a>
        {#if !WEIGHT_KINDS.some((k) => caps[k].allowed)}
          <p class="xs muted">{sentence(caps.weight_propose.reason ?? 'Weight editing is not enabled on this host.')} An operator enables each job kind on the worker.</p>
        {/if}
      {/if}
    </div>
  </section>

  <section class="panel" aria-labelledby="w-make-h">
    <header>
      <h2 id="w-make-h">Choose and change weights</h2>
      <div class="fmt" role="radiogroup" aria-label="Editing mode">
        <label class:on={mode === 'manual'}><input type="radio" name="w-mode" bind:group={mode} value="manual" /> Manual</label>
        <label class:on={mode === 'auto'}><input type="radio" name="w-mode" bind:group={mode} value="auto" /> Automatic</label>
      </div>
    </header>

    <div class="body stack">
      {#if source}
        {#if mode === 'auto'}
          {#key source.id}
            <h3>Calibration and held-out data</h3>
            <p class="small muted">Choose two different splits on {hostName(hosts, source.host)}. Use one to fit the selected tensor and one to measure its error. Their activation rows must not overlap.</p>
            {#if source.host === 'backend-hf' && caps.weight_tune.allowed && !sourceHost?.scheduling_paused}
              <p class="xs muted">Selecting a JSON file uploads it to this worker’s Library. Each file must be at most 256 KiB and contain two numeric matrices: <code>activations</code> and <code>targets</code>. Use one activation and target row per example (up to 1,024 examples).</p>
            {:else if source.host !== 'backend-hf'}
              <p class="xs muted">To add a split, register its path as a <code>weight-calibration</code> entry for this host in the <a href="#/library">Library</a>.</p>
            {/if}
            <div class="grid-2">
              <CalibrationSplit host={sourceHost} label="Calibration split (fitting)" bind:value={calibrationId} other={heldOutId} />
              <CalibrationSplit host={sourceHost} label="Held-out split (measurement only)" bind:value={heldOutId} other={calibrationId} />
            </div>
          {/key}
        {/if}
        {#if !formatAllowed}
          <p class="note warn">{safe ? 'SafeTensors' : 'Native HGN'} editing is unavailable on {hostName(hosts, source.host)}. {sentence(native.reason ?? 'This source host is not enrolled.')} Check <code>{safe ? 'weight_safetensors' : 'weight_hgn'}</code> in <a href="#/settings/infrastructure?host={encodeURIComponent(source.host)}">worker settings</a>.</p>
        {:else}
        {#key source.id}
        <VisualEditor
          {source}
          {canVisual}
          canPropose={caps.weight_propose.allowed}
          canTune={caps.weight_tune.allowed}
          {calibrationId}
          {heldOutId}
          proposeReason={caps.weight_propose.reason}
          tuneReason={caps.weight_tune.reason}
          bind:mode
          onproposal={(id) => {
            reviewId = id;
            queueMicrotask(() => document.getElementById('w-rev-h')?.scrollIntoView({ block: 'start' }));
          }}
        />
        {/key}
        {/if}
      {:else}
        <p class="empty">Select a registered source above.</p>
      {/if}
      {#if source?.kind === 'gguf'}
      <details class="disclosure text-entry" open={!!source && !canVisual}>
        <summary>Text entry: type tensor names and coordinates</summary>
    {#if mode === 'manual'}
      <div class="stack pad">
        <p class="xs muted">Coordinates are zero-based in row-major order, one number per tensor dimension, for example <span class="mono">3, 17</span>. <b>Set</b> writes the value. <b>Delta</b> adds it to the stored value. The worker rounds each result to the stored type and refuses edits that do not change the stored value.</p>

        <fieldset class="sub">
          <legend>Look up a stored value (optional)</legend>
          <div class="lookup">
            <div class="field grow">
              <label for="w-ins-t">Tensor</label>
              <input id="w-ins-t" class="input mono" bind:value={inspectTensor} autocomplete="off" spellcheck="false" placeholder="blk.0.attn_q.weight" />
            </div>
            <div class="field">
              <label for="w-ins-i">Coordinate</label>
              <input id="w-ins-i" class="input mono coord" bind:value={inspectIndex} autocomplete="off" spellcheck="false" placeholder="0, 0" />
            </div>
            <Submit variant="secondary" blockers={inspectTensor || inspectIndex ? inspectBlockers : gate('weight_inspect')} busy={inspectSub.busy} onclick={inspect}>Inspect</Submit>
          </div>
          {#if inspectSub.error}<p class="note bad" role="alert">{inspectSub.error}</p>{/if}
          <div aria-live="polite">
            {#if lastInspect}
              <div class="ins">
                <div class="row">
                  <JobState job={lastInspect} />
                  <a class="xs" href="#/jobs/{encodeURIComponent(lastInspect.id)}">Latest inspect</a>
                  {#if lastInspect.created_at}<span class="xs muted">{formatTime(lastInspect.created_at)}</span>{/if}
                </div>
                {#if lastInspect.error}<p class="xs err break">{lastInspect.error}</p>{/if}
                {#if inspectDetail}
                  <WeightReport job={inspectDetail} />
                  {#if inspected}<button type="button" class="btn small" onclick={useInspected} disabled={rows.length >= MAX_EDITS}>Add as edit row</button>{/if}
                {/if}
              </div>
            {/if}
          </div>
        </fieldset>

        <div class="table-wrap">
          <table class="data stackable edits">
            <caption class="sr-only">Manual edits</caption>
            <thead><tr><th scope="col" class="n">#</th><th scope="col">Tensor</th><th scope="col">Coordinate</th><th scope="col">Operation</th><th scope="col">Value</th><th scope="col"><span class="sr-only">Remove</span></th></tr></thead>
            <tbody>
              {#each rows as r, i (r.key)}
                {@const e = manual.errors[i] ?? {}}
                {@const show = touched || !!(r.tensor && r.index && r.value)}
                <tr>
                  <td class="n num muted" data-label="Edit">{i + 1}</td>
                  <td data-label="Tensor">
                    <input id="w-{r.key}-tensor" class="input mono" bind:value={r.tensor} aria-label="Tensor, edit {i + 1}" aria-invalid={!!err(show, e.tensor)} aria-describedby={err(show, e.tensor) ? `w-${r.key}-tensor-e` : undefined} autocomplete="off" spellcheck="false" placeholder="blk.0.attn_q.weight" />
                    {#if err(show, e.tensor)}<span class="ferr xs" id="w-{r.key}-tensor-e">{e.tensor}</span>{/if}
                  </td>
                  <td data-label="Coordinate">
                    <input id="w-{r.key}-index" class="input mono coord" bind:value={r.index} aria-label="Coordinate, edit {i + 1}" aria-invalid={!!err(show, e.index)} aria-describedby={err(show, e.index) ? `w-${r.key}-index-e` : undefined} autocomplete="off" spellcheck="false" placeholder="0, 0" />
                    {#if err(show, e.index)}<span class="ferr xs" id="w-{r.key}-index-e">{e.index}</span>{/if}
                  </td>
                  <td data-label="Operation">
                    <select class="input" bind:value={r.op} aria-label="Operation, edit {i + 1}">
                      <option value="set">Set to</option>
                      <option value="delta">Add delta</option>
                    </select>
                  </td>
                  <td data-label="Value">
                    <input id="w-{r.key}-value" class="input mono val" bind:value={r.value} inputmode="decimal" aria-label="Value, edit {i + 1}" aria-invalid={!!err(show, e.value)} aria-describedby={err(show, e.value) ? `w-${r.key}-value-e` : undefined} autocomplete="off" spellcheck="false" placeholder={r.op === 'set' ? '0.5' : '-0.002'} />
                    {#if err(show, e.value)}<span class="ferr xs" id="w-{r.key}-value-e">{e.value}</span>{/if}
                  </td>
                  <td class="rm">
                    <button type="button" class="btn ghost small" onclick={() => removeRow(i)} aria-label="Remove edit {i + 1}" disabled={rows.length === 1 && !r.tensor && !r.index && !r.value}>Remove</button>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
        <div class="row">
          <button type="button" class="btn small" onclick={addRow} disabled={rows.length >= MAX_EDITS}>Add edit</button>
          <span class="xs muted num">{rows.length} of {MAX_EDITS}</span>
        </div>

        <details class="disclosure">
          <summary>Add many edits from text</summary>
          <div class="stack pad">
            <div class="field">
              <label for="w-bulk">One edit per line</label>
              <textarea id="w-bulk" class="input mono" rows="4" bind:value={bulkText} spellcheck="false" aria-describedby="w-bulk-hint" aria-invalid={!!bulkError} placeholder={'blk.0.attn_q.weight 3,17 delta -0.002\nblk.0.attn_q.weight 4,0 set 0.125'}></textarea>
              <span class="hint" id="w-bulk-hint">Write <span class="mono">tensor coordinate set|delta value</span>. Lines that start with # are skipped. Empty rows above are replaced.</span>
            </div>
            {#if bulkError}<p class="note bad" role="alert">{bulkError}</p>{/if}
            <div><button type="button" class="btn small" onclick={addBulk} disabled={!bulkText.trim()}>Add lines</button></div>
          </div>
        </details>

        {#if proposeSub.error}<p class="note bad" role="alert">{proposeSub.error}</p>{/if}
        <Submit blockers={proposeBlockers} busy={proposeSub.busy} onclick={propose}>Create proposal</Submit>
      </div>
    {:else}
      <div class="stack pad">
        <p class="xs muted">The worker fits the selected weights to the calibration split with bounded coordinate descent, then measures the same layer on the held-out split. The held-out split is never used for fitting. Each split is a <span class="mono">weight-calibration</span> Library entry with activations and targets for this tensor.</p>
        <div class="grid-2">
          <div class="field">
            <label for="w-tune-t">Tensor (rank 2, [output, input])</label>
            <input id="w-tune-t" class="input mono" bind:value={tuneTensor} autocomplete="off" spellcheck="false" placeholder="blk.0.ffn_down.weight" aria-invalid={tuneTouched && !tuneParsed.t.ok} />
          </div>
          <div class="field">
            <label for="w-tune-m">Largest change per weight</label>
            <input id="w-tune-m" class="input mono" bind:value={tuneMax} inputmode="decimal" autocomplete="off" spellcheck="false" placeholder="0.01" aria-describedby="w-tune-m-hint" aria-invalid={tuneTouched && !tuneParsed.m.ok} />
            <span class="hint" id="w-tune-m-hint">Absolute limit after rounding to the stored type.</span>
          </div>
        </div>
        <div class="field">
          <label for="w-tune-c">Selected weights</label>
          <textarea id="w-tune-c" class="input mono" rows="5" bind:value={tuneCoords} spellcheck="false" aria-describedby="w-tune-c-hint" aria-invalid={tuneTouched && !tuneParsed.c.ok} placeholder={'0, 3\n0, 4\n7, 12'}></textarea>
          <span class="hint" id="w-tune-c-hint">One <span class="mono">row, column</span> pair per line, zero-based. <span class="num">{coordCount}</span> of {MAX_COORDINATES}.{#if tuneCoords.trim() && !tuneParsed.c.ok} <span class="ferr">{tuneParsed.c.error}</span>{/if}</span>
        </div>
        <p class="xs muted">The worker refuses splits whose activation rows overlap, files over 8 MiB or 1,024 rows, and proposals that do not lower the calibration error.</p>
        {#if tuneSub.error}<p class="note bad" role="alert">{tuneSub.error}</p>{/if}
        <Submit blockers={tuneBlockers} busy={tuneSub.busy} onclick={tune}>Tune selected weights</Submit>
      </div>
    {/if}
      </details>
      {/if}
    </div>
  </section>

  <section class="panel" aria-labelledby="w-rev-h">
    <header><h2 id="w-rev-h">Review and apply</h2>{#if proposals.length}<span class="xs muted">{proposals.length} {proposals.length === 1 ? 'proposal' : 'proposals'} for this source</span>{/if}</header>
    {#if !source}
      <p class="empty">Select a source to see its proposals.</p>
    {:else if proposals.length === 0}
      <p class="empty">No proposals for this source yet. Create one above.</p>
    {:else}
      <div class="review">
        <fieldset class="plist">
          <legend class="sr-only">Proposal to review</legend>
          {#each proposals as p (p.id)}
            <label class="pitem" class:on={reviewId === p.id}>
              <input type="radio" name="w-review" value={p.id} bind:group={reviewId} />
              <span class="pmain">
                <span class="row"><JobState job={p} /><span class="pname">{p.kind === 'weight_tune' ? 'Automatic' : 'Manual'}</span></span>
                <span class="xs muted mono truncate">{p.id}</span>
                {#if p.created_at}<span class="xs muted">{formatTime(p.created_at)}</span>{/if}
                {#if appliesOf(jobs, p.id).some((a) => a.state === 'succeeded')}<span class="xs ok">Applied</span>{/if}
              </span>
            </label>
          {/each}
        </fieldset>

        <div class="detail stack" aria-live="polite">
          {#if !reviewed}
            <p class="muted small">Select a proposal to see its before and after values.</p>
          {:else}
            <div class="row between">
              <div class="row">
                <JobState job={reviewed} />
                <a href="#/jobs/{encodeURIComponent(reviewed.id)}" class="title">{kindTitle(reviewed.kind)}</a>
              </div>
              <CancelButton job={reviewed} />
            </div>
            {#if reviewed.input_ids[0] !== sourceId}<p class="note warn">This proposal belongs to another source.</p>{/if}
            {#if isActive(reviewed.state)}
              <JobProgress job={reviewed} label="Progress of proposal {reviewed.id}" />
              <p class="xs muted">The proposal shows here when the job finishes. You can leave this page; the controller keeps the job.</p>
            {:else if reviewed.state !== 'succeeded'}
              <p class="note bad" role="alert">{reviewed.error ?? 'The job did not succeed. Nothing can be applied.'}</p>
            {:else if detailError}
              <p class="note bad" role="alert">{detailError}</p>
            {:else if !detail}
              <p class="xs muted">Loading proposal…</p>
            {:else}
              <WeightReport job={detail} />
              <p class="xs muted">Effects on quantized blocks are not measured here. If you quantize the new file, neighbouring weights in the same block can change too.</p>

              <div class="apply stack">
                <h3 class="h">Apply</h3>
                {#if doneApply}
                  <p class="note info small">
                    Applied by <a class="mono" href="#/jobs/{encodeURIComponent(doneApply.id)}">{doneApply.id}</a>.
                    {#if applied?.library_model_id}
                      New Library entry: <a href="#/weights?source={encodeURIComponent(applied.library_model_id)}">{appliedModel ? modelLabel(appliedModel) : applied.library_model_id}</a>{#if applied.output_sha256}, SHA-256 <span class="mono" title={applied.output_sha256}>{shortHash(applied.output_sha256)}</span>{/if}.
                    {/if}
                  </p>
                {:else}
                  {#each applies.filter((a) => a.state !== 'succeeded').slice(0, 3) as a (a.id)}
                    <div class="row">
                      <JobState job={a} /><a class="xs mono" href="#/jobs/{encodeURIComponent(a.id)}">{a.id}</a>
                      {#if a.error}<span class="xs err break">{a.error}</span>{/if}
                    </div>
                    {#if a.state === 'running'}<JobProgress job={a} label="Progress of apply {a.id}" />{/if}
                  {/each}
                  {#if proposal}
                    <label class="confirm">
                      <input type="checkbox" bind:checked={confirmed} disabled={!!activeApply} />
                      <span>I reviewed all {proposal.edits.length} proposed {proposal.edits.length === 1 ? 'value' : 'values'} for source <span class="mono">{shortHash(proposal.source_sha256)}</span>.{#if tuneRegress}{' '}I accept that the held-out error increased.{/if}</span>
                    </label>
                  {/if}
                  {#if applySub.error}<p class="note bad" role="alert">{applySub.error}</p>{/if}
                  <Submit blockers={applyBlockers} busy={applySub.busy} onclick={applyNow}>Apply as new {sourceFormat}</Submit>
                  <p class="xs muted">Apply registers a verified revision as a separate Library entry and keeps the original.</p>
                {/if}
              </div>
            {/if}
          {/if}
        </div>
      </div>
    {/if}
  </section>
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .head p {
    margin: 0;
    max-width: 80ch;
  }
  .kv {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 4px var(--s-4);
    margin: 0;
  }
  .kv dt {
    color: var(--text-2);
  }
  .kv dd {
    margin: 0;
    min-width: 0;
  }
  .caps {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2);
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .fmt {
    display: inline-flex;
    padding: 2px;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    background: var(--surface);
  }
  .fmt label {
    position: relative;
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 0 10px;
    border-radius: var(--r-sm);
    font-size: var(--fs-sm);
    color: var(--text-2);
    cursor: pointer;
    white-space: nowrap;
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
  .sub {
    margin: 0;
    padding: var(--s-3);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
    min-width: 0;
  }
  .sub legend {
    padding: 0 4px;
    font-size: var(--fs-sm);
    font-weight: 600;
  }
  .lookup {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: var(--s-3);
  }
  .grow {
    flex: 1 1 240px;
  }
  .coord {
    width: 9rem;
  }
  .ins {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    align-items: flex-start;
  }
  .edits td {
    vertical-align: top;
  }
  .edits .n {
    width: 2.5rem;
    padding-top: 14px;
  }
  .edits .input {
    width: 100%;
    min-width: 0;
  }
  .edits td:nth-child(2) {
    min-width: 14rem;
  }
  .edits .coord {
    min-width: 7rem;
  }
  .edits .val {
    min-width: 8rem;
  }
  .edits .rm {
    width: 1%;
    white-space: nowrap;
  }
  .ferr {
    display: block;
    margin-top: 2px;
    color: var(--bad);
  }
  .pad {
    padding-top: var(--s-3);
  }
  .text-entry {
    margin-top: var(--s-2);
  }
  .review {
    display: grid;
    grid-template-columns: minmax(200px, 260px) minmax(0, 1fr);
  }
  .plist {
    margin: 0;
    padding: var(--s-2);
    border: none;
    border-right: 1px solid var(--line-soft);
    display: flex;
    flex-direction: column;
    gap: 2px;
    max-height: 560px;
    overflow-y: auto;
    min-width: 0;
  }
  .pitem {
    position: relative;
    display: flex;
    padding: var(--s-2) var(--s-3);
    border-radius: var(--r-sm);
    cursor: pointer;
    border: 1px solid transparent;
  }
  .pitem:hover {
    background: var(--surface-2);
  }
  .pitem.on {
    background: var(--accent-soft);
    border-color: var(--line);
  }
  .pitem input {
    position: absolute;
    inset: 0;
    opacity: 0;
    margin: 0;
    cursor: pointer;
  }
  .pitem:has(input:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: -2px;
  }
  .pmain {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  .pname {
    font-weight: 600;
    font-size: var(--fs-sm);
  }
  .ok {
    color: var(--ok);
    font-weight: 600;
  }
  .err {
    color: var(--bad);
  }
  .detail {
    padding: var(--s-4);
    min-width: 0;
  }
  .between {
    justify-content: space-between;
  }
  .title {
    font-weight: 600;
    color: var(--text);
  }
  .h {
    margin: 0;
    font-size: var(--fs-sm);
    font-weight: 600;
  }
  .apply {
    gap: var(--s-3);
    padding-top: var(--s-3);
    border-top: 1px solid var(--line-soft);
  }
  .confirm {
    display: flex;
    gap: var(--s-2);
    align-items: flex-start;
    font-size: var(--fs-sm);
    cursor: pointer;
  }
  .confirm input {
    margin-top: 3px;
    width: 16px;
    height: 16px;
    accent-color: var(--accent);
  }
  @media (max-width: 760px) {
    .review {
      grid-template-columns: minmax(0, 1fr);
    }
    .plist {
      border-right: none;
      border-bottom: 1px solid var(--line-soft);
      max-height: 240px;
    }
  }
  @media (max-width: 640px) {
    .edits .n {
      padding-top: 3px;
    }
    .edits td:nth-child(2) {
      min-width: 0;
    }
    .edits .rm {
      width: auto;
    }
  }
</style>

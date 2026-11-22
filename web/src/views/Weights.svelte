<script lang="ts">
  import { SvelteMap } from 'svelte/reactivity';
  import ModelSelect from '../components/ModelSelect.svelte';
  import Submit from '../components/Submit.svelte';
  import Badge from '../components/Badge.svelte';
  import JobState from '../components/JobState.svelte';
  import JobProgress from '../components/JobProgress.svelte';
  import CancelButton from '../components/CancelButton.svelte';
  import WeightReport from '../components/weights/WeightReport.svelte';
  import VisualEditor from '../components/weights/VisualEditor.svelte';
  import CalibrationSplit from '../components/weights/CalibrationSplit.svelte';
  import FieldError from '../components/FieldError.svelte';
  import Blockers from '../components/Blockers.svelte';
  import KeyValues from '../components/KeyValues.svelte';
  import { brewery } from '../lib/store.svelte';
  import { errorText } from '../lib/api';
  import { Submitter } from '../lib/submit.svelte';
  import { hashParam, route } from '../lib/route.svelte';
  import { watchRetry } from '../lib/rerun';
  import { fieldError, visibleBlockers, type Blocker } from '../lib/blockers';
  import { WEIGHT_SOURCES, isSafeSource, formatTime, hostName, isActive, isTerminal, kindTitle, modelLabel, newestFirst, shortHash } from '../lib/format';
  import { canInspectVisually, weightAccess, weightCapability, weightFormat } from '../lib/weight-access';
  import { addDraftEdits, editKey, type DraftEdit } from '../lib/weight-draft';
  import { focusAfterRemoval } from '../lib/focus';
  import { unsavedChanges } from '../lib/unsaved-changes.svelte';
  import {
    MAX_COORDINATES,
    MAX_EDITS,
    PROPOSAL_KINDS,
    WEIGHT_KINDS,
    appliesOf,
    applyJob,
    checkEdits,
    coord,
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
  import type { JobDetail, Model } from '../lib/types';

  const models = $derived(brewery.state?.models ?? []);
  const hosts = $derived(brewery.state?.hosts ?? []);
  const jobs = $derived(brewery.state?.jobs ?? []);

  // ---- Source (shared by both modes). ----
  const sourceId = $derived(hashParam(route.hash, 'source') ?? '');
  function selectSource(id: string) {
    if (id === sourceId || !draftGuard.confirm()) return;
    route.replaceQuery(id ? `#/weights?source=${encodeURIComponent(id)}` : '#/weights');
  }

  const source = $derived(models.find((m) => m.id === sourceId && WEIGHT_SOURCES.includes(m.kind)));
  const sourceHost = $derived(source ? hosts.find((h) => h.id === source.host) : undefined);
  const hgn = $derived(source?.kind === 'hgn');
  const safe = $derived(isSafeSource(source?.kind));
  const native = $derived(source ? weightFormat(source, sourceHost) : undefined);
  const formatAllowed = $derived(native?.allowed !== false);
  const sourceFormat = $derived(source?.kind === 'hf' ? 'HF checkpoint' : safe ? 'SafeTensors file' : hgn ? 'HGN' : 'GGUF');
  const caps = $derived(Object.fromEntries(WEIGHT_KINDS.map((k) => [k, weightCapability(sourceHost, k, source?.host)])) as Record<(typeof WEIGHT_KINDS)[number], ReturnType<typeof weightCapability>>);
  const capLabels = { weight_inspect: 'Inspect', weight_propose: 'Manual proposal', weight_tune: 'Automatic tuning', weight_apply: 'Apply' } as const;
  const visual = $derived(weightCapability(sourceHost, 'weight_visual', source?.host));
  const access = $derived(source ? weightAccess(source, sourceHost) : null);
  function unavailable(model: Model) {
    const host = hosts.find((host) => host.id === model.host);
    return canInspectVisually(model, host) ? null : weightAccess(model, host).reason;
  }
  const canVisual = $derived(!!source && canInspectVisually(source, sourceHost));
  const sourceBlockers = $derived.by(() => {
    if (!sourceId) return ['Select a registered source.'];
    if (!source) return ['This Library entry is missing or does not have a supported weight format.'];
    if (source.available === false) return [sentence(source.reason ?? 'The source file is not available.')];
    if (!formatAllowed) return [sentence(native?.reason ?? 'Editing this format is disabled on this host.')];
    return [];
  });
  function gate(kind: (typeof WEIGHT_KINDS)[number]): string[] {
    if (sourceBlockers.length) return sourceBlockers;
    const c = caps[kind];
    return c.allowed ? [] : [sentence(c.reason ?? `${capLabels[kind]} is not enabled on this worker.`)];
  }

  // ---- Mode. ----
  let mode = $state<'manual' | 'auto'>('manual');
  let autoEntry = $state<'visual' | 'text'>('visual');
  const canChange = $derived(!!access?.allowed && (mode === 'manual' ? caps.weight_propose.allowed : caps.weight_tune.allowed));
  const showText = $derived(canChange && source?.kind === 'gguf' && (mode === 'manual' || !canVisual || autoEntry === 'text'));
  const showVisual = $derived(!canChange || mode === 'manual' || source?.kind !== 'gguf' || autoEntry === 'visual');

  // ---- Manual edits. ----
  let seq = 0;
  const nextKey = () => `e${seq++}`;
  const blankRow = (tensor = ''): EditRow => ({ key: nextKey(), tensor, index: '', op: 'set', value: '' });
  let rows = $state<EditRow[]>([blankRow()]);
  const pending = new SvelteMap<string, DraftEdit>();
  const draft = $derived([...pending.values()]);
  let touched = $state<Record<string, boolean>>({});
  let textError = $state<string | null>(null);
  const manual = $derived(checkEdits(rows));
  const hasText = (r: EditRow) => !!(r.tensor.trim() || r.index.trim() || r.value.trim());
  const draftGuard = unsavedChanges(() => pending.size > 0 || rows.some(hasText) || !!bulkText.trim());
  const manualFields = $derived(manual.errors.flatMap((errors, i) =>
    Object.entries(errors).map(([key, text]) => ({ text, field: `${rows[i].key}-${key}` }))));
  const visibleManual = $derived(visibleBlockers(manualFields, touched));
  function addRow() {
    if (rows.length >= MAX_EDITS) return;
    rows.push(blankRow(rows.at(-1)?.tensor ?? ''));
    const key = rows.at(-1)!.key;
    queueMicrotask(() => document.getElementById(`w-${key}-index`)?.focus());
  }
  function removeRow(i: number) {
    void focusAfterRemoval(() => {
      rows.splice(i, 1);
      if (rows.length === 0) rows.push(blankRow());
    }, () => document.getElementById(`w-${(rows[i] ?? rows[i - 1]).key}-tensor`));
  }
  let bulkText = $state('');
  let bulkError = $state<string | null>(null);
  function addBulk() {
    const parsed = parseEditLines(bulkText, nextKey);
    if (!parsed.ok) {
      bulkError = parsed.error;
      return;
    }
    const kept = rows.filter(hasText);
    if (kept.length + parsed.value.length > MAX_EDITS) {
      bulkError = `That makes ${kept.length + parsed.value.length} edits. Use at most ${MAX_EDITS}.`;
      return;
    }
    rows = [...kept, ...parsed.value];
    bulkText = '';
    bulkError = null;
  }
  function stageText() {
    if (!manual.edits || gate('weight_propose').length) return;
    textError = addDraftEdits(pending, manual.edits);
    if (textError) return;
    rows = [blankRow()];
    touched = {};
    document.getElementById('w-draft-h')?.focus();
  }
  const proposeSub = new Submitter();
  const draftPin = $derived(draft.find((edit) => edit.observed)?.observed?.sha);
  const draftPacked = $derived(draft.some((edit) => edit.observed?.packed));
  const proposeBlockers = $derived([
    ...gate('weight_propose'),
    ...(draft.length ? [] : ['Add one or more edits.']),
    ...(rows.some(hasText) || bulkText.trim() ? ['Add or clear the text edits before you submit.'] : []),
    ...(draftPin ? gate('weight_inspect').concat(visual.allowed ? [] : [visual.reason!]) : []),
    ...(draft.some((edit) => edit.observed && edit.observed.sha !== draftPin) ? ['Read the source again before you submit.'] : []),
    ...(draftPacked && sourceHost?.capabilities?.weight_packed !== true ? ['Packed editing is not enabled on this host.'] : []),
  ]);
  async function propose() {
    if (!source || proposeBlockers.length) return;
    const id = source.id;
    const submitted = draft;
    const request = proposeJob(id, source.host, submitted);
    if (draftPin) request.options = { ...request.options, source_sha256: draftPin, ...(draftPacked ? { packed: true } : {}) };
    const job = await proposeSub.submit(request);
    if (job && sourceId === id) {
      for (const edit of submitted) {
        const key = editKey(edit);
        if (pending.get(key) === edit) pending.delete(key);
      }
      openReview(job.id);
    }
  }
  function openReview(id: string) {
    reviewId = id;
    queueMicrotask(() => document.getElementById('w-rev-h')?.focus());
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
    if (!source || inspectBlockers.length || !inspectParsed.t.ok || !inspectParsed.i.ok) return;
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
  const inspected = $derived(inspectDetail ? readInspect(inspectDetail.result, 'value') : null);
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
  let tuneTouched = $state<Record<string, boolean>>({});
  const tuneParsed = $derived({ t: parseTensor(tuneTensor), c: parseCoordinates(tuneCoords), m: parseMaxUpdate(tuneMax) });
  const coordCount = $derived(tuneCoords.split('\n').filter((l) => l.trim()).length);
  const tuneSub = new Submitter();
  const tuneFields = $derived<Blocker[]>([
    ...(tuneParsed.t.ok ? [] : [{ field: 'tensor', text: tuneParsed.t.error }]),
    ...(tuneParsed.c.ok ? [] : [{ field: 'coordinates', text: tuneParsed.c.error }]),
    ...(tuneParsed.m.ok ? [] : [{ field: 'max', text: tuneParsed.m.error }]),
  ]);
  const visibleTune = $derived(visibleBlockers(tuneFields, tuneTouched));
  const sameHost = (id: string) => models.find((m) => m.id === id)?.host === source?.host;
  const tuneBlockers = $derived.by(() => {
    const out = [...gate('weight_tune')];
    out.push(...tuneFields.map((error) => error.text));
    if (!calibrationId) out.push('Select a calibration split.');
    if (!heldOutId) out.push('Select a held-out split.');
    else if (heldOutId === calibrationId) out.push('Use a different entry for the held-out split.');
    if (source && ((calibrationId && !sameHost(calibrationId)) || (heldOutId && !sameHost(heldOutId)))) out.push(`Both splits must be on ${hostName(hosts, source.host)}, the host of the source.`);
    return out;
  });
  async function tune() {
    if (!source || !tuneParsed.t.ok || !tuneParsed.c.ok || !tuneParsed.m.ok || tuneBlockers.length) return;
    const job = await tuneSub.submit(tuneJob(source.id, source.host, calibrationId, heldOutId, tuneParsed.t.value, tuneParsed.c.value, tuneParsed.m.value));
    if (job) openReview(job.id);
  }

  // ---- Review and apply. ----
  let retryProposal = $state<JobDetail | null>(null);
  const proposals = $derived.by(() => {
    const list = newestFirst(jobsFor(jobs, sourceId, PROPOSAL_KINDS)).slice(0, 20);
    return retryProposal && !list.some((job) => job.id === retryProposal?.id) ? [...list, retryProposal] : list;
  });
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

  let previous = $state<JobDetail | null>(null);
  let retryError = $state(false);
  const retrySub = new Submitter();
  $effect(() => {
    previous = null;
    retryProposal = null;
    retryError = false;
    return watchRetry(route.hash, WEIGHT_KINDS, async (job, signal) => {
      if (sourceId !== job.input_ids[0]) throw new Error('Source mismatch.');
      if (job.kind !== 'weight_apply') {
        previous = job;
        return;
      }
      const id = job.options?.proposal_job_id;
      if (typeof id !== 'string') throw new Error('No prior proposal.');
      const proposal = await brewery.loadJob(id);
      if (signal.aborted) return;
      if (proposal.id !== id || proposal.input_ids[0] !== sourceId || !PROPOSAL_KINDS.has(proposal.kind)) throw new Error('Proposal mismatch.');
      retryProposal = proposal;
      reviewId = id;
      confirmed = false;
    }, () => { retryError = true; });
  });
  const previousBlockers = $derived.by(() => {
    if (!previous || !source) return ['Select an available source entry.'];
    const kind = WEIGHT_KINDS.find((kind) => kind === previous?.kind);
    if (!kind) return ['This request is not a weight job.'];
    const out = [...gate(kind)];
    if (previous.host !== source.host || previous.input_ids.some((id) => !models.some((m) =>
      m.id === id && m.host === source?.host && m.available !== false))) {
      out.push('The original inputs must be available on the original host.');
    }
    if ((previous.options?.mode || previous.options?.source_sha256 || (kind !== 'weight_inspect' && source.kind !== 'gguf')) && !canVisual) {
      out.push(visual.reason ?? 'Visual inspection is not available.');
    }
    if (previous.options?.packed === true && sourceHost?.capabilities?.weight_packed !== true) {
      out.push('Packed editing is not enabled on this host.');
    }
    if (kind !== 'weight_inspect' && jobs.some((job) => isActive(job.state) && job.kind === kind && job.input_ids[0] === source?.id)) {
      out.push('Wait until the active job on this source finishes.');
    }
    return out;
  });
  async function queuePrevious() {
    if (!previous || !source || previousBlockers.length || retrySub.busy) return;
    const { kind, input_ids, options } = previous;
    const queued = await retrySub.submit({ kind, host: source.host, input_ids, options });
    if (queued) location.hash = `#/jobs/${encodeURIComponent(queued.id)}`;
  }

  $effect.pre(() => {
    void sourceId;
    pending.clear();
    rows = [blankRow()];
    touched = {};
    bulkText = '';
    bulkError = null;
    textError = null;
    autoEntry = 'visual';
    mode = 'manual';
    inspectTensor = '';
    inspectIndex = '';
    tuneTensor = '';
    tuneCoords = '';
    tuneMax = '';
    tuneTouched = {};
    reviewId = null;
  });
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Weight edits</h1>
    <p class="muted small">
      Select weights on the source host, then adjust them by hand or tune them on a calibration split. Review the proposal before applying it as a new revision in the original format. The source stays unchanged.
    </p>
  </div>

  <p class="note info small">
    A manual proposal can contain up to {MAX_EDITS} edits across tensors. Automatic tuning can change up to {MAX_COORDINATES} weights in one rank-2 tensor. Tuning measures one layer, not model quality. Edited files are not release-qualified.
  </p>

  <section class="panel" aria-labelledby="w-src-h">
    <header><h2 id="w-src-h">Source</h2>{#if source}<a class="xs" href="#/library">Library</a>{/if}</header>
    <div class="body stack">
      <ModelSelect bind:value={() => sourceId, selectSource} {unavailable} kinds={WEIGHT_SOURCES} label="Source entry" id="w-source" hint="Choose a GGUF, native HGN, SafeTensors file or indexed Hugging Face checkpoint. Jobs run on the source host." />
      {#if sourceId && !source}
        <p class="note warn">{sourceBlockers[0]}</p>
      {:else if source}
        <dl class="kv small">
          <dt>Host</dt><dd>{hostName(hosts, source.host)}</dd>
          <dt>Path</dt><dd class="mono xs break">{source.path}</dd>
          <dt class="format">Format</dt><dd class="format">{sourceFormat}{#if source.kind === 'hf'}. Apply keeps the index and shard layout.{/if}</dd>
          <dt>{source.kind === 'hf' ? 'Bundle SHA-256' : 'SHA-256'}</dt><dd>{#if source.sha256}<span class="mono" title={source.sha256}>{shortHash(source.sha256)}</span>{:else}<span class="muted">No digest in the Library record. The worker binds edits to the source fingerprint.</span>{/if}</dd>
          {#if source.job_id}<dt>Output of job</dt><dd><a class="mono xs break" href="#/jobs/{encodeURIComponent(source.job_id)}">{source.job_id}</a></dd>{/if}
        </dl>
        <ul class="caps" aria-label="Weight editing on {hostName(hosts, source.host)}">
          {#each WEIGHT_KINDS as k (k)}
            {@const status = formatAllowed ? caps[k] : native!}
            <li>
              <Badge tone={status.allowed ? 'ok' : 'muted'} label="{capLabels[k]}: {status.allowed ? 'enabled' : 'not enabled'}" />
              {#if status.reason}<a class="xs break" href={status.fixAnchor}>{sentence(status.reason)}</a>{/if}
            </li>
          {/each}
        </ul>
        {#if caps.weight_inspect.allowed && formatAllowed}
          <p class="xs muted">Visual reads: {#if visual.allowed}enabled{:else}<a href={visual.fixAnchor}>{sentence(visual.reason!)}</a>{/if}</p>
        {/if}
        <a class="small" href="#/settings/infrastructure?host={encodeURIComponent(source.host)}&section=policy">Settings for {hostName(hosts, source.host)}</a>
      {/if}
    </div>
  </section>

  {#if retryError}<p class="note warn">Cannot open the previous request. Enter the inputs again.</p>{/if}
  {#if previous}
    <section class="panel" aria-labelledby="w-previous-h">
      <header><h2 id="w-previous-h">Previous weight request</h2></header>
      <div class="body stack">
        <p>{kindTitle(previous.kind)} on {hostName(hosts, previous.host)}</p>
        <p class="small">Inputs: {previous.input_ids.map((id) => models.find((m) => m.id === id)?.name ?? id).join(', ')}</p>
        <KeyValues data={previous.options} />
        <p class="hint">Review the prior inputs and options before you queue the job. To change them, use the editor below.</p>
        <Submit blockers={previousBlockers} busy={retrySub.busy} onclick={queuePrevious}>Queue previous request</Submit>
        {#if retrySub.error}<p class="note bad" role="alert">{retrySub.error}</p>{/if}
      </div>
    </section>
  {/if}

  <section class="panel" aria-labelledby="w-make-h">
    <header>
      <h2 id="w-make-h">Choose and change weights</h2>
      {#if access?.allowed}<div class="fmt" role="radiogroup" aria-label="Editing mode">
        <label class:on={mode === 'manual'}><input type="radio" name="w-mode" bind:group={mode} value="manual" disabled={!caps.weight_propose.allowed} /> Manual</label>
        <label class:on={mode === 'auto'}><input type="radio" name="w-mode" bind:group={mode} value="auto" disabled={!caps.weight_tune.allowed} /> Automatic</label>
      </div>{/if}
    </header>

    <div class="body stack">
      {#if source && (access?.allowed || canVisual)}
        {#if !canChange}<p class="note warn small">Read-only: {sentence((mode === 'manual' ? caps.weight_propose.reason : caps.weight_tune.reason) ?? 'Select an available editing mode.')}</p>{/if}
        {#if source.kind === 'gguf' && !canVisual && caps.weight_inspect.allowed && !(showText && mode === 'manual')}
          {@render valueLookup(false)}
        {/if}
        {#if mode === 'auto' && canChange}
          {#key source.id}
            <h3>Calibration and held-out data</h3>
            <p class="small muted">Choose two different splits on {hostName(hosts, source.host)}. Use one to fit the selected tensor and one to measure its error. Their activation rows must not overlap.</p>
            <p class="xs muted">
              Use JSON with two matrices of finite numbers: <code>activations</code> and <code>targets</code>.
              Use the same number of rows in both matrices (1 to 1,024 rows). The column counts must match the input and output widths of the tensor.
              {#if source.host === 'backend-hf'}Each upload, including its name, can be at most 256 KiB. Each matrix can have at most 16,384 columns. The upload adds a split to Library.
              {:else}Each file can be at most 8 MiB. Register the file as <code>weight-calibration</code> on this host in <a href="#/library">Library</a>.{/if}
            </p>
            <div class="grid-2">
              <CalibrationSplit host={sourceHost} label="Calibration data" uploadLabel="Upload calibration data (JSON)" bind:value={calibrationId} other={heldOutId} />
              <CalibrationSplit host={sourceHost} label="Held-out data" uploadLabel="Upload held-out data (JSON)" bind:value={heldOutId} other={calibrationId} />
            </div>
          {/key}
        {/if}
        {#if mode === 'auto' && canChange && source.kind === 'gguf' && canVisual}
          <div class="field">
            <label for="w-auto-entry">Automatic tuning input</label>
            <select id="w-auto-entry" class="input" bind:value={autoEntry}>
              <option value="visual">Select cells in the visual editor</option>
              <option value="text">Type the tensor and coordinates</option>
            </select>
          </div>
        {/if}
        {#if showVisual}
        {#key source.id}
        <VisualEditor
          {source}
          {canVisual}
          {canChange}
          canTune={caps.weight_tune.allowed}
          {calibrationId}
          {heldOutId}
          tuneReason={caps.weight_tune.reason}
          bind:mode
          {pending}
          onproposal={openReview}
        />
        {/key}
        {/if}
      {#if showText}
      <details class="disclosure text-entry" open={!canVisual || mode === 'auto'}>
        <summary>Text entry: type tensor names and coordinates</summary>
    {#if mode === 'manual'}
      <div class="stack pad">
        <p class="xs muted">Coordinates are zero-based in row-major order, one number per tensor dimension, for example <span class="mono">3, 17</span>. <b>Set</b> writes the value. <b>Delta</b> adds it to the stored value. The worker rounds each result to the stored type and refuses edits that do not change the stored value.</p>

        {@render valueLookup(true)}

        <div class="table-wrap">
          <table class="data stackable edits">
            <caption class="sr-only">Manual edits</caption>
            <thead><tr><th scope="col" class="n">#</th><th scope="col">Tensor</th><th scope="col">Coordinate</th><th scope="col">Operation</th><th scope="col">Value</th><th scope="col"><span class="sr-only">Remove</span></th></tr></thead>
            <tbody>
               {#each rows as r, i (r.key)}
                {@const tensorError = fieldError(visibleManual, `${r.key}-tensor`)}
                {@const indexError = fieldError(visibleManual, `${r.key}-index`)}
                {@const valueError = fieldError(visibleManual, `${r.key}-value`)}
                <tr>
                  <td class="n num muted" data-label="Edit">{i + 1}</td>
                  <td data-label="Tensor">
                    <input id="w-{r.key}-tensor" class="input mono" bind:value={r.tensor} aria-label="Tensor, edit {i + 1}" aria-invalid={!!tensorError} aria-describedby={tensorError ? `w-${r.key}-tensor-e` : undefined} onblur={() => { touched[`${r.key}-tensor`] = true; }} autocomplete="off" spellcheck="false" placeholder="blk.0.attn_q.weight" />
                    <FieldError id="w-{r.key}-tensor-e" error={tensorError} />
                  </td>
                  <td data-label="Coordinate">
                    <input id="w-{r.key}-index" class="input mono coord" bind:value={r.index} aria-label="Coordinate, edit {i + 1}" aria-invalid={!!indexError} aria-describedby={indexError ? `w-${r.key}-index-e` : undefined} onblur={() => { touched[`${r.key}-index`] = true; }} autocomplete="off" spellcheck="false" placeholder="0, 0" />
                    <FieldError id="w-{r.key}-index-e" error={indexError} />
                  </td>
                  <td data-label="Operation">
                    <select class="input" bind:value={r.op} aria-label="Operation, edit {i + 1}">
                      <option value="set">Set to</option>
                      <option value="delta">Add delta</option>
                    </select>
                  </td>
                  <td data-label="Value">
                    <input id="w-{r.key}-value" class="input mono val" bind:value={r.value} inputmode="decimal" aria-label="Value, edit {i + 1}" aria-invalid={!!valueError} aria-describedby={valueError ? `w-${r.key}-value-e` : undefined} onblur={() => { touched[`${r.key}-value`] = true; }} autocomplete="off" spellcheck="false" placeholder={r.op === 'set' ? '0.5' : '-0.002'} />
                    <FieldError id="w-{r.key}-value-e" error={valueError} />
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

        {#if textError}<p class="note bad" role="alert">{textError}</p>{/if}
        <Submit variant="secondary" blockers={[...gate('weight_propose'), ...(manual.blocker ? [manual.blocker] : [])]} busy={proposeSub.busy} onclick={stageText}>Add to proposal</Submit>
      </div>
    {:else}
      <div class="stack pad">
        <p class="xs muted">The worker rejects a proposal if it does not decrease the calibration error.</p>
        <div class="grid-2">
          <div class="field">
            <label for="w-tune-t">Tensor (rank 2, [output, input])</label>
            <input id="w-tune-t" class="input mono" bind:value={tuneTensor} autocomplete="off" spellcheck="false" placeholder="blk.0.ffn_down.weight" onblur={() => { tuneTouched.tensor = true; }} aria-invalid={!!fieldError(visibleTune, 'tensor')} aria-describedby={fieldError(visibleTune, 'tensor') ? 'w-tune-t-e' : undefined} />
            <FieldError id="w-tune-t-e" error={fieldError(visibleTune, 'tensor')} />
          </div>
          <div class="field">
            <label for="w-tune-m">Largest change per weight</label>
            <input id="w-tune-m" class="input mono" bind:value={tuneMax} inputmode="decimal" autocomplete="off" spellcheck="false" placeholder="0.01" onblur={() => { tuneTouched.max = true; }} aria-describedby={fieldError(visibleTune, 'max') ? 'w-tune-m-e w-tune-m-hint' : 'w-tune-m-hint'} aria-invalid={!!fieldError(visibleTune, 'max')} />
            <span class="hint" id="w-tune-m-hint">Absolute limit after rounding to the stored type.</span>
            <FieldError id="w-tune-m-e" error={fieldError(visibleTune, 'max')} />
          </div>
        </div>
        <div class="field">
          <label for="w-tune-c">Selected weights</label>
          <textarea id="w-tune-c" class="input mono" rows="5" bind:value={tuneCoords} spellcheck="false" onblur={() => { tuneTouched.coordinates = true; }} aria-describedby={fieldError(visibleTune, 'coordinates') ? 'w-tune-c-e w-tune-c-hint' : 'w-tune-c-hint'} aria-invalid={!!fieldError(visibleTune, 'coordinates')} placeholder={'0, 3\n0, 4\n7, 12'}></textarea>
          <span class="hint" id="w-tune-c-hint">One <span class="mono">row, column</span> pair per line. Start at 0. <span class="num">{coordCount}</span> of {MAX_COORDINATES}.</span>
          <FieldError id="w-tune-c-e" error={fieldError(visibleTune, 'coordinates')} />
        </div>
        {#if tuneSub.error}<p class="note bad" role="alert">{tuneSub.error}</p>{/if}
        <Submit blockers={tuneBlockers} busy={tuneSub.busy} onclick={tune}>Tune selected weights</Submit>
      </div>
    {/if}
      </details>
      {/if}
        {#if mode === 'manual' && canChange}
          <section class="stack draft" aria-labelledby="w-draft-h">
            <div class="row between">
              <h3 id="w-draft-h" tabindex="-1">Proposal <span class="muted num">({draft.length} of {MAX_EDITS} edits)</span></h3>
              {#if draft.length}
                <button type="button" class="btn ghost small" disabled={proposeSub.busy} onclick={() => {
                  void focusAfterRemoval(() => pending.clear(), () => document.getElementById('w-draft-h'));
                }}>Discard all</button>
              {/if}
            </div>
            {#if draft.length}
              <div class="table-wrap">
                <table class="data stackable">
                  <caption class="sr-only">Edits in this proposal</caption>
                  <thead><tr><th scope="col">Tensor</th><th scope="col">Coordinate</th><th scope="col">Stored</th><th scope="col">Edit</th><th scope="col"><span class="sr-only">Actions</span></th></tr></thead>
                  <tbody>
                    {#each draft as edit (editKey(edit))}
                      <tr>
                        <td class="mono xs break" data-label="Tensor">{edit.tensor}</td>
                        <td class="mono num" data-label="Coordinate">{coord(edit.index)}</td>
                        <td class="mono num" data-label="Stored">{edit.observed ? exact(edit.observed.before) : 'Not read'}</td>
                        <td class="mono num" data-label="Edit">{edit.op === 'set' ? 'Set to' : 'Add'} {exact(edit.value)}</td>
                        <td><button type="button" class="btn ghost small" disabled={proposeSub.busy} aria-label="Remove edit for {edit.tensor} at {coord(edit.index)}" onclick={() => {
                          void focusAfterRemoval(() => { pending.delete(editKey(edit)); }, () => document.getElementById('w-draft-h'));
                        }}>Remove</button></td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
              <p class="xs muted">The worker checks these values and returns a proposal for review. The source stays unchanged.</p>
            {:else}
              <p class="xs muted">No edits yet.{#if source.kind === 'gguf'} Add edits from the visual editor or text entry above.{/if}</p>
            {/if}
            {#if proposeSub.error}<p class="note bad" role="alert">{proposeSub.error}</p>{/if}
            <Submit blockers={proposeBlockers} busy={proposeSub.busy} onclick={propose}>Create proposal</Submit>
          </section>
        {/if}
      {:else if source && access?.reason}
        <p class="note warn">{access.reason}</p>
        {#if source.available !== false}
          <Blockers blockers={[{ text: 'Check weight editing on the source host', fixAnchor: access.fixAnchor }]} />
        {/if}
      {:else}
        <p class="empty">Select a source above.</p>
      {/if}
    </div>
  </section>

  <section class="panel" aria-labelledby="w-rev-h">
    <header><h2 id="w-rev-h" tabindex="-1">Review and apply</h2>{#if proposals.length}<span class="xs muted">{proposals.length} {proposals.length === 1 ? 'proposal' : 'proposals'} for this source</span>{/if}</header>
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

{#snippet valueLookup(canAddEdit: boolean)}
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
      <Submit variant="secondary" blockers={inspectBlockers} busy={inspectSub.busy} onclick={inspect}>Inspect</Submit>
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
            {#if canAddEdit && inspected}<button type="button" class="btn small" onclick={useInspected} disabled={rows.length >= MAX_EDITS}>Add as edit row</button>{/if}
          {/if}
        </div>
      {/if}
    </div>
  </fieldset>
{/snippet}

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
  .kv .format {
    padding-block: var(--s-1);
  }
  .caps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(220px, 100%), 1fr));
    gap: var(--s-2);
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .caps li {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--s-1);
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
  .draft {
    border-top: 1px solid var(--line-soft);
    padding-top: var(--s-3);
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

<script lang="ts">
  // Spectral weight health of one scratch revision (X2). The operator queues a
  // weight_spectra job with POST /api/weight-spectra (shared Submission helper,
  // with an idempotency key). By default the controller compares with the
  // parent. The panel then reads the job (GET only) until it is terminal.
  import { untrack } from 'svelte';
  import { api } from '../../lib/api';
  import { brewery } from '../../lib/store.svelte';
  import { Submission } from '../../lib/submit.svelte';
  import { isTerminal, modelLabel, newestFirst, shortHash } from '../../lib/format';
  import { fmt, type ScratchRevision } from '../../lib/weightspace';
  import {
    METRICS,
    changed,
    layerSeries,
    parseSpectra,
    spectraBlocker,
    spectraRequest,
    tensorGroups,
    type SpectraForm,
    type SpectraTensor,
  } from '../../lib/spectra';
  import type { JobDetail } from '../../lib/types';
  import Advanced from '../Advanced.svelte';
  import Badge from '../Badge.svelte';
  import JobLink from '../JobLink.svelte';
  import JobState from '../JobState.svelte';
  import LayerChart from './LayerChart.svelte';
  import SpectrumChart from './SpectrumChart.svelte';

  let { rev, revisions = [] }: { rev: ScratchRevision; revisions?: ScratchRevision[] } = $props();

  const POLL_MS = 1500;
  const models = $derived(brewery.state?.models ?? []);
  const nameOf = (id: string) => {
    const m = models.find((x) => x.id === id);
    if (m) return modelLabel(m);
    return revisions.some((r) => r.id === id) ? `scratch ${id.slice(0, 8)}` : id;
  };

  let form = $state<SpectraForm>({ compare: '', representation: 'safetensors', compareRepresentation: 'safetensors', layer: '', tensor: '' });
  const blocker = $derived(spectraBlocker(form));
  const submission = new Submission();
  let jobId = $state<string | null>(null);
  let job = $state<JobDetail | null>(null);
  let readError = $state<string | null>(null);

  // A new revision starts clean, and shows its newest earlier spectra job if any.
  // Key on the ID: the list poll replaces the revision object every few seconds.
  const revId = $derived(rev.id);
  $effect(() => {
    const id = revId;
    untrack(() => {
      form = { compare: '', representation: 'safetensors', compareRepresentation: 'safetensors', layer: '', tensor: '' };
      submission.error = null;
      job = null;
      readError = null;
      const prior = newestFirst((brewery.state?.jobs ?? []).filter((j) => j.kind === 'weight_spectra' && j.input_ids[0] === id))[0];
      jobId = prior?.id ?? null;
    });
  });

  $effect(() => {
    const id = jobId;
    if (!id) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      try {
        // untrack: loadJob reads and writes brewery.details, which would restart this effect.
        const d = await untrack(() => brewery.loadJob(id, true));
        if (stop) return;
        job = d;
        readError = null;
        if (isTerminal(d.state)) return;
      } catch (e) {
        if (stop) return;
        readError = e instanceof Error ? e.message : String(e);
      }
      timer = setTimeout(tick, document.visibilityState === 'hidden' ? 10000 : POLL_MS);
    };
    void tick();
    return () => {
      stop = true;
      if (timer) clearTimeout(timer);
    };
  });

  async function measure(event: MouseEvent) {
    const body = spectraRequest(rev.id, form);
    const r = await submission.send('edits.spectra', body, (b, key) => api.weightSpectra(b, key), event);
    if (r && typeof r.id === 'string') {
      job = null;
      jobId = r.id;
      void brewery.refresh();
    }
  }

  const result = $derived(job?.state === 'succeeded' ? parseSpectra(job.result) : null);
  const hasReference = $derived(!!result?.reference || !!result?.tensors.some((t) => t.reference));
  let group = $state('');
  const groups = $derived(result ? tensorGroups(result) : []);
  $effect(() => {
    if (group && !groups.includes(group)) group = '';
  });
  const series = $derived(result ? METRICS.map((m) => ({ ...m, points: layerSeries(result, m.key, group) })) : []);
  const layers = $derived(series[0]?.points.map((p) => p.layer) ?? []);

  const withSpectrum = $derived(result?.tensors.filter((t) => t.source?.singular_values.length || t.reference?.singular_values.length) ?? []);
  let spectrumName = $state('');
  const sigmaShift = (t: SpectraTensor) => Math.abs(t.delta?.sigma_max ?? 0);
  $effect(() => {
    const list = withSpectrum;
    if (list.some((t) => t.name === spectrumName)) return;
    const most = [...list].sort((a, b) => sigmaShift(b) - sigmaShift(a))[0];
    spectrumName = most?.name ?? '';
  });
  const spectrumTensor = $derived(withSpectrum.find((t) => t.name === spectrumName));

  const referenceText = $derived(form.compare ? nameOf(form.compare) : rev.provenance.compiled_program ? 'none (this program has no parent)' : `the parent (${nameOf(rev.provenance.parent_id)})`);
  const optionsChanged = $derived([
    form.compare && 'Reference',
    form.representation !== 'safetensors' && 'Source GGUF',
    form.compareRepresentation !== 'safetensors' && 'Reference GGUF',
    form.layer.trim() && `Layer ${form.layer.trim()}`,
    form.tensor.trim() && 'Tensor',
  ].filter((x): x is string => !!x));
  const signed = (v: number | null) => (v !== null && v > 0 ? `+${fmt(v)}` : fmt(v));
  const shapeText = (t: SpectraTensor) => (t.source?.shape.length ? `[${t.source.shape.join(', ')}]` : '—');
</script>

<section class="panel" id="weight-health" aria-labelledby="wh-h">
  <header>
    <h2 id="wh-h">Weight health</h2>
    <Badge tone="warn" label="Not release-qualified" />
  </header>
  <div class="body stack">
    <p class="small muted">
      Singular-value spectra per tensor: effective rank, stable rank and the heavy-tail exponent alpha. A change of 1% or more against the reference is marked.
    </p>
    <p class="small">Reference: {referenceText}</p>

    <Advanced title="Options" changed={optionsChanged}>
      <div class="grid-2">
        <div class="field">
          <label for="wh-ref">Reference</label>
          <select id="wh-ref" class="input" bind:value={form.compare}>
            <option value="">The parent (default)</option>
            {#each revisions.filter((r) => r.id !== rev.id) as r (r.id)}<option value={r.id}>Scratch {r.id.slice(0, 8)}</option>{/each}
            {#each models as m (m.id)}<option value={m.id}>{modelLabel(m)}</option>{/each}
          </select>
        </div>
        <div class="field">
          <label for="wh-layer">Layer <span class="muted">(optional)</span></label>
          <input id="wh-layer" class="input num" bind:value={form.layer} inputmode="numeric" autocomplete="off" aria-invalid={blocker ? 'true' : undefined} aria-describedby={blocker ? 'wh-why' : 'wh-layer-h'} />
          <span class="hint" id="wh-layer-h">Empty measures all layers.</span>
        </div>
        <div class="field">
          <label for="wh-rep">Source representation</label>
          <select id="wh-rep" class="input" bind:value={form.representation} aria-describedby="wh-rep-h">
            <option value="safetensors">SafeTensors</option>
            <option value="gguf" disabled={!rev.gguf}>GGUF, dequantized</option>
          </select>
          <span class="hint" id="wh-rep-h">{rev.gguf ? 'GGUF reads the private conversion of this revision.' : 'GGUF needs a private conversion of this revision.'}</span>
        </div>
        <div class="field">
          <label for="wh-crep">Reference representation</label>
          <select id="wh-crep" class="input" bind:value={form.compareRepresentation} aria-describedby="wh-crep-h">
            <option value="safetensors">SafeTensors</option>
            <option value="gguf">GGUF, dequantized</option>
          </select>
          <span class="hint" id="wh-crep-h">Only tensor types that the reader supports are dequantized.</span>
        </div>
        <div class="field span">
          <label for="wh-tensor">Tensor <span class="muted">(optional)</span></label>
          <input id="wh-tensor" class="input mono" bind:value={form.tensor} autocomplete="off" spellcheck="false" placeholder="model.layers.0.mlp.down_proj.weight" />
        </div>
      </div>
    </Advanced>

    <div class="row action">
      <button type="button" class="btn primary" disabled={!!blocker || submission.busy} aria-busy={submission.busy} aria-describedby={blocker ? 'wh-why' : undefined} onclick={measure}>
        {submission.busy ? 'Queuing…' : 'Measure spectra'}
      </button>
      {#if blocker}<span class="xs err" id="wh-why">{blocker}</span>{/if}
      {#if jobId}
        <span class="xs muted job">Job <JobLink id={jobId} label={jobId.length > 12 ? jobId.slice(0, 8) : jobId} />{#if job}{' '}<JobState {job} />{/if}</span>
      {/if}
    </div>
    {#if submission.error}<p class="note bad small" role="alert">{submission.error}</p>{/if}
    <div aria-live="polite" class="xs muted">
      {#if readError}<span class="err">Could not read the job: {readError}</span>
      {:else if jobId && (!job || !isTerminal(job.state))}Measuring. The page reads the job every {POLL_MS / 1000} seconds.
      {:else if job && job.state !== 'succeeded'}The job ended as {job.state}.{#if job.error}{` ${job.error}`}{/if}
      {:else if job && !result}The job result is not a weight_spectra result.{/if}
    </div>

    {#if result}
      <dl class="prov xs">
        <dt>Source</dt><dd class="break">{nameOf(result.source.id)} <span class="mono muted" title={result.source.sha256}>{shortHash(result.source.sha256)}</span> {result.source.format}</dd>
        {#if result.reference}
          <dt>Reference</dt><dd class="break">{nameOf(result.reference.id)} <span class="mono muted" title={result.reference.sha256}>{shortHash(result.reference.sha256)}</span> {result.reference.format}</dd>
        {/if}
        <dt>Tensors</dt><dd class="num">{result.tensors.length}</dd>
      </dl>

      {#if !layers.length}
        <p class="empty">No tensor in this result has a layer number.</p>
      {:else}
        <div class="toolbar">
          <div class="field">
            <label for="wh-group">Tensor type</label>
            <select id="wh-group" class="input" bind:value={group}>
              <option value="">All tensors (mean per layer)</option>
              {#each groups as g (g)}<option value={g}>{g}</option>{/each}
            </select>
          </div>
          <ul class="legend xs" aria-label="Legend">
            <li><svg width="22" height="8" aria-hidden="true"><line x1="0" x2="22" y1="4" y2="4" class="lg-src" /></svg>Source</li>
            {#if hasReference}<li><svg width="22" height="8" aria-hidden="true"><line x1="0" x2="22" y1="4" y2="4" class="lg-ref" /></svg>Reference</li>{/if}
            <li><span class="lg-band" aria-hidden="true"></span>Changed 1% or more</li>
          </ul>
        </div>
        <div class="charts">
          {#each series as s (s.key)}
            <LayerChart label={s.label} points={s.points} {hasReference} />
          {/each}
        </div>

        <!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users must be able to scroll this table.) -->
        <div class="table-wrap tall" tabindex="0" role="region" aria-label="Per-layer values">
          <table class="data num stackable">
            <caption class="sr-only">Per-layer values, {group || 'all tensors, mean per layer'}</caption>
            <thead>
              <tr>
                <th scope="col" rowspan="2">Layer</th>
                {#each METRICS as m (m.key)}<th scope="colgroup" colspan={hasReference ? 3 : 1}>{m.label}</th>{/each}
                {#if hasReference}<th scope="col" rowspan="2">Result</th>{/if}
              </tr>
              <tr>
                {#each METRICS as m (m.key)}
                  <th scope="col">Source</th>
                  {#if hasReference}<th scope="col">Reference</th><th scope="col">Change</th>{/if}
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each layers as layer, i (layer)}
                {@const any = series.some((s) => changed(s.points[i].delta, s.points[i].reference))}
                <tr>
                  <th scope="row"><span class="narrow">Layer</span> {layer}</th>
                  {#each series as s (s.key)}
                    {@const p = s.points[i]}
                    <td data-label="{s.label}, source">{fmt(p.source)}</td>
                    {#if hasReference}
                      <td data-label="{s.label}, reference">{fmt(p.reference)}</td>
                      <td data-label="{s.label}, change" class:hot={changed(p.delta, p.reference)}>{signed(p.delta)}</td>
                    {/if}
                  {/each}
                  {#if hasReference}
                    <td data-label="Result">{#if any}<Badge tone="warn" label="Changed" />{:else if series.some((s) => s.points[i].delta !== null)}<Badge tone="muted" label="Unchanged" />{:else}<Badge tone="muted" label="No comparison" />{/if}</td>
                  {/if}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}

      <div class="spectrum">
        <h3>Spectrum</h3>
        {#if !withSpectrum.length}
          <p class="empty">No tensor in this result has singular values.</p>
        {:else}
          <div class="field">
            <label for="wh-spec">Spectrum tensor</label>
            <select id="wh-spec" class="input mono" bind:value={spectrumName}>
              {#each withSpectrum as t (t.name)}<option value={t.name}>{t.name}</option>{/each}
            </select>
          </div>
          {#if spectrumTensor}
            <SpectrumChart name={spectrumTensor.name} source={spectrumTensor.source} reference={spectrumTensor.reference} />
          {/if}
        {/if}
      </div>

      <details class="disclosure">
        <summary><span class="t">All tensor values</span> <span class="s xs muted">{result.tensors.length} tensors</span></summary>
        <!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users must be able to scroll this table.) -->
        <div class="table-wrap tall" tabindex="0" role="region" aria-label="Values for each tensor">
          <table class="data num stackable">
            <caption class="sr-only">Values for each tensor</caption>
            <thead>
              <tr>
                <th scope="col">Tensor</th><th scope="col">Shape</th><th scope="col">Method</th>
                <th scope="col">Effective rank</th><th scope="col">Stable rank</th><th scope="col">Alpha</th>
                {#if hasReference}<th scope="col">σmax change</th>{/if}
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {#each result.tensors as t (t.name)}
                <tr>
                  <th scope="row" class="mono break">{t.name}</th>
                  <td data-label="Shape" class="mono">{t.source?.dtype ?? ''} {shapeText(t)}</td>
                  <td data-label="Method">{t.source?.method || '—'}</td>
                  {#each METRICS as m (m.key)}
                    <td data-label={m.label}>{fmt(t.source?.[m.key] ?? null)}{#if hasReference && t.delta?.[m.key]}<span class="xs" class:hot={changed(t.delta[m.key] ?? null, t.reference?.[m.key] ?? null)}> ({signed(t.delta[m.key] ?? null)})</span>{/if}</td>
                  {/each}
                  {#if hasReference}<td data-label="σmax change">{signed(t.delta?.sigma_max ?? null)}</td>{/if}
                  <td data-label="Status" class="mono">{t.status ?? 'ok'}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </details>
      <p class="xs muted">Alpha follows the WeightWatcher power-law fit (Martin and Mahoney, 2021). For large matrices, randomized SVD estimates the leading singular values. Effective rank and alpha use only those values.</p>
    {/if}
  </div>
</section>

<style>
  header {
    flex-wrap: wrap;
    gap: var(--s-2);
  }
  .err {
    color: var(--bad);
  }
  .span {
    grid-column: 1 / -1;
  }
  .action {
    flex-wrap: wrap;
    align-items: center;
    gap: var(--s-2) var(--s-3);
  }
  .job {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .prov {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 2px var(--s-3);
    margin: 0;
  }
  .prov dt {
    color: var(--text-2);
  }
  .prov dd {
    margin: 0;
    min-width: 0;
  }
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: var(--s-2) var(--s-4);
  }
  .toolbar .field {
    min-width: min(100%, 280px);
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-1) var(--s-3);
    margin: 0;
    padding: 0;
    list-style: none;
    color: var(--text-2);
  }
  .legend li {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .lg-src {
    stroke: var(--accent);
    stroke-width: 2;
  }
  .lg-ref {
    stroke: var(--text-2);
    stroke-width: 1.5;
    stroke-dasharray: 4 3;
  }
  .lg-band {
    width: 12px;
    height: 12px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-line);
    border-radius: 2px;
  }
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: var(--s-4);
  }
  .hot {
    color: var(--warn);
    font-weight: 600;
  }
  h3 {
    font-size: var(--fs-md);
    font-weight: 600;
    margin-bottom: var(--s-2);
  }
  .spectrum {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .spectrum .field {
    max-width: 520px;
  }
  .tall {
    max-height: 420px;
  }
  .narrow {
    display: none;
  }
  @media (max-width: 640px) {
    .narrow {
      display: inline;
    }
  }
  .disclosure .t {
    font-weight: 500;
    font-size: var(--fs-sm);
  }
</style>

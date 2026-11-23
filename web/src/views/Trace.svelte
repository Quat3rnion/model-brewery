<script lang="ts">
  import { untrack } from 'svelte';
  import { api, errorText } from '../lib/api';
  import { brewery } from '../lib/store.svelte';
  import { hashParam, route } from '../lib/route.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { capability, formatBytes, formatTime, isTerminal, modelLabel, newestFirst } from '../lib/format';
  import { obj } from '../lib/reports';
  import {
    MAX_PATCHES,
    MAX_PROMPTS,
    PATCH_ROLES,
    ROLE_LABELS,
    TRACE_ROLES,
    buildTraceJob,
    estimateTraceBytes,
    needsFullDump,
    parseIndexList,
    type PatchForm,
    type TraceRole,
  } from '../lib/weightspace';
  import ModelSelect from '../components/ModelSelect.svelte';
  import HostSelect from '../components/HostSelect.svelte';
  import Submit from '../components/Submit.svelte';
  import SubmitStatus from '../components/SubmitStatus.svelte';
  import Advanced from '../components/Advanced.svelte';
  import JobState from '../components/JobState.svelte';
  import JobProgress from '../components/JobProgress.svelte';
  import CancelButton from '../components/CancelButton.svelte';
  import JobLink from '../components/JobLink.svelte';
  import TraceResults from '../components/trace/TraceResults.svelte';
  import type { JobDetail } from '../lib/types';

  const GGUF = ['gguf'];
  const MIB = 1 << 20;
  const models = $derived(brewery.state?.models ?? []);
  const hosts = $derived(brewery.state?.hosts ?? []);
  const traces = $derived(newestFirst((brewery.state?.jobs ?? []).filter((j) => j.kind === 'trace')));

  // ---- Byte cap from Settings (GET only) ----
  let caps = $state({ default_byte_cap: 2 * MIB, max_byte_cap: 16 * MIB });
  $effect(() => {
    const ac = new AbortController();
    api
      .settings(ac.signal)
      .then((s) => {
        const t = obj(obj(s)?.trace);
        if (t && typeof t.default_byte_cap === 'number' && typeof t.max_byte_cap === 'number') caps = { default_byte_cap: t.default_byte_cap, max_byte_cap: t.max_byte_cap };
      })
      .catch(() => {});
    return () => ac.abort();
  });

  // ---- Saved interventions (W2). The list is optional: a failed read hides it. ----
  type Artifact = { id: string; name: string; model_id: string; latest_version: number };
  let artifacts = $state<Artifact[]>([]);
  $effect(() => {
    const ac = new AbortController();
    api
      .interventions(ac.signal)
      .then((list) => (artifacts = Array.isArray(list) ? (list as Artifact[]).filter((a) => a && typeof a.id === 'string') : []))
      .catch(() => (artifacts = []));
    return () => ac.abort();
  });

  // ---- Form ----
  let model = $state('');
  let host = $state('');
  let prompts = $state<string[]>(['']);
  let layers = $state('0-3');
  let roles = $state<TraceRole[]>(['l_out']);
  let positions = $state('0-7');
  let topK = $state(5);
  let capMiB = $state<number | null>(null);
  let fullDump = $state(false);
  let hidden = $state<number | null>(null);
  let heads = $state<number | null>(null);
  let artifactId = $state('');
  let artifactVersion = $state(1);
  let patches = $state<PatchForm[]>([]);
  let ablate = $state(false);
  let ablationLayer = $state<number | undefined>(0);
  let ablationExpert = $state<number | undefined>(undefined);
  let ablationPositions = $state('0');

  const modelArtifacts = $derived(artifacts.filter((a) => a.model_id === model));
  const artifact = $derived(modelArtifacts.find((a) => a.id === artifactId) ?? null);
  $effect(() => {
    if (artifactId && !artifact) artifactId = '';
  });
  $effect(() => {
    if (artifact && (artifactVersion < 1 || artifactVersion > artifact.latest_version)) artifactVersion = artifact.latest_version;
  });
  const patchSources = $derived(traces.filter((j) => j.state === 'succeeded' && j.input_ids[0] === model));

  const parsedLayers = $derived(parseIndexList(layers, 4096, true));
  const parsedPositions = $derived(parseIndexList(positions, 4096));
  const effectiveHost = $derived(host || models.find((m) => m.id === model)?.host || '');

  const built = $derived(
    buildTraceJob({
      model,
      host,
      prompts,
      layers,
      roles,
      positions,
      topK,
      byteCapMiB: capMiB,
      maxByteCap: caps.max_byte_cap,
      fullDump,
      artifact: artifact ? { id: artifact.id, version: artifactVersion } : null,
      patches,
      ablation: ablate ? { layer: ablationLayer, expert: ablationExpert, positions: ablationPositions } : undefined,
    }),
  );
  const blockers = $derived.by(() => {
    const out = [...built.blockers];
    if (effectiveHost) {
      const c = capability(hosts.find((h) => h.id === effectiveHost), 'trace');
      if (!c.allowed && c.reason) out.push(c.reason);
    }
    return out;
  });

  const nl = $derived(parsedLayers.ok ? parsedLayers.value.length : 0);
  const np = $derived(parsedPositions.ok ? parsedPositions.value.length : 0);
  const cap = $derived(capMiB !== null && Number.isFinite(capMiB) ? Math.round(capMiB * MIB) : caps.default_byte_cap);
  const estimate = $derived(
    estimateTraceBytes({
      prompts: prompts.length,
      layers: nl,
      positions: np,
      roles,
      topK: Number.isInteger(topK) ? topK : 0,
      hidden,
      heads,
      keys: parsedPositions.ok && np ? parsedPositions.value[np - 1] + 1 : 1,
    }),
  );
  const ratio = $derived(cap > 0 ? estimate.bytes / cap : 0);
  const large = $derived(needsFullDump(nl, np, roles.length));

  function toggleRole(r: TraceRole, on: boolean) {
    roles = on ? TRACE_ROLES.filter((x) => x === r || roles.includes(x)) : roles.filter((x) => x !== r);
  }
  function addPatch() {
    patches = [...patches, { source: patchSources[0]?.id ?? '', prompt: 0, layer: parsedLayers.ok && parsedLayers.value.length ? String(parsedLayers.value[0]) : '0', role: 'l_out', positions: '0' }];
  }

  const advChanged = $derived(
    [
      capMiB !== null ? `cap ${capMiB} MiB` : '',
      fullDump ? 'full dump' : '',
      artifact ? `intervention ${artifact.name} v${artifactVersion}` : '',
      patches.length ? `${patches.length} ${patches.length === 1 ? 'patch' : 'patches'}` : '',
    ].filter(Boolean),
  );

  const s = new Submitter('trace.queue');
  function submit(event: MouseEvent) {
    if (!built.job || blockers.length) return;
    const { canPresent } = route.capture();
    void s.submit(built.job, event, false, undefined, (job) => {
      if (canPresent()) location.hash = `#/trace?job=${encodeURIComponent(job.id)}`;
    });
  }

  // ---- Selected run (#/trace?job=<id>) ----
  let hash = $state(typeof window === 'undefined' ? '' : window.location.hash);
  $effect(() => {
    const on = () => (hash = window.location.hash);
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  });
  const selectedId = $derived(hashParam(hash, 'job') ?? traces[0]?.id ?? null);
  const selected = $derived(traces.find((j) => j.id === selectedId) ?? null);
  let detail = $state<JobDetail | null>(null);
  let detailError = $state<string | null>(null);
  $effect(() => {
    const id = selected?.id;
    const state = selected?.state;
    detail = null;
    detailError = null;
    if (!id || state !== 'succeeded') return;
    let stop = false;
    untrack(() => brewery.loadJob(id, !isTerminal(state)))
      .then((d) => {
        if (!stop) detail = d;
      })
      .catch((e) => {
        if (!stop) detailError = errorText(e);
      });
    return () => {
      stop = true;
    };
  });
  const uid = $props.id();
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Trace</h1>
    <p class="muted small">
      Run prompts through one GGUF model on the CPU and capture activations, logit lens rows and attention. Traces do not generate text and do not change the model. Results are evidence for analysis. They are not release-qualified.
    </p>
  </div>

  <section class="panel" aria-labelledby="tr-new-h">
    <header><h2 id="tr-new-h">New trace</h2></header>
    <div class="body stack">
      <div class="grid-2">
        <ModelSelect bind:value={model} kinds={GGUF} label="Model" hint="A single-file GGUF in the Library." />
        <HostSelect bind:value={host} kind="trace" label="Run on" optional optionalLabel="Host of the model" />
      </div>

      <fieldset class="group">
        <legend class="small">Prompts</legend>
        {#each prompts as _, i (i)}
          <div class="field">
            <div class="prompt-head">
              <label for="{uid}-p{i}">Prompt {i + 1}</label>
              {#if prompts.length > 1}
                <button type="button" class="btn ghost small" aria-label="Remove prompt {i + 1}" onclick={() => (prompts = prompts.filter((__, k) => k !== i))}>Remove</button>
              {/if}
            </div>
            <textarea id="{uid}-p{i}" class="input" rows="2" bind:value={prompts[i]}></textarea>
          </div>
        {/each}
        <div class="row">
          <button type="button" class="btn small" disabled={prompts.length >= MAX_PROMPTS} onclick={() => (prompts = [...prompts, ''])}>Add prompt</button>
          <span class="xs muted">1 to {MAX_PROMPTS} prompts. The worker reads them raw, with no chat template. Positions include special tokens such as BOS.</span>
        </div>
      </fieldset>

      <fieldset class="group">
        <legend class="small">Capture</legend>
        <div class="grid-3">
          <div class="field">
            <label for="{uid}-layers">Layers</label>
            <input id="{uid}-layers" class="input mono" bind:value={layers} autocomplete="off" spellcheck="false" aria-invalid={!parsedLayers.ok || undefined} aria-describedby="{uid}-layers-h" />
            <span class="hint" id="{uid}-layers-h">{parsedLayers.ok ? `${nl} layers. Example: 0-3, 8.` : parsedLayers.error}</span>
          </div>
          <div class="field">
            <label for="{uid}-pos">Token positions</label>
            <input id="{uid}-pos" class="input mono" bind:value={positions} autocomplete="off" spellcheck="false" aria-invalid={!parsedPositions.ok || undefined} aria-describedby="{uid}-pos-h" />
            <span class="hint" id="{uid}-pos-h">{parsedPositions.ok ? `${np} positions, zero-based.` : parsedPositions.error}</span>
          </div>
          <div class="field">
            <label for="{uid}-topk">Top-k logits</label>
            <input id="{uid}-topk" class="input num" type="number" min="0" max="100" step="1" bind:value={topK} inputmode="numeric" aria-describedby="{uid}-topk-h" />
            <span class="hint" id="{uid}-topk-h">0 to 100. Above 0 adds model logits and a logit lens for dense Llama and Qwen2.</span>
          </div>
        </div>
        <label><input type="checkbox" bind:checked={ablate} /> Zero one MoE expert</label>
        {#if ablate}
          <div class="grid-3">
            <label class="field">Ablation layer<input class="input" type="number" min="0" max="4095" bind:value={ablationLayer} /></label>
            <label class="field">Expert ID<input class="input" type="number" min="0" max="4095" bind:value={ablationExpert} /></label>
            <label class="field">Ablation positions<input class="input" bind:value={ablationPositions} /></label>
          </div>
          <p class="hint">Use the same model and prompt as the baseline. A position that does not select this expert does not change.</p>
        {/if}
        <div class="checks" role="group" aria-label="Roles">
          {#each TRACE_ROLES as r (r)}
            <label><input type="checkbox" checked={roles.includes(r)} onchange={(e) => toggleRole(r, e.currentTarget.checked)} /> {ROLE_LABELS[r]}</label>
          {/each}
        </div>

        <div class="estimate">
          <div class="grid-3">
            <div class="field">
              <label for="{uid}-hidden">Hidden size <span class="muted">(estimate only)</span></label>
              <input id="{uid}-hidden" class="input num" type="number" min="1" step="1" bind:value={hidden} inputmode="numeric" placeholder="896…" />
            </div>
            <div class="field">
              <label for="{uid}-heads">Attention heads <span class="muted">(estimate only)</span></label>
              <input id="{uid}-heads" class="input num" type="number" min="1" step="1" bind:value={heads} inputmode="numeric" placeholder="14…" />
            </div>
            <div class="field">
              <span>Size estimate</span>
              <div class="meter" role="meter" aria-label="Estimated export size of the byte cap" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.min(100, Math.round(ratio * 100))} aria-valuetext="{formatBytes(estimate.bytes)} of {formatBytes(cap)}">
                <div class="fill" class:over={ratio > 1} style:width="{Math.min(100, ratio * 100)}%"></div>
              </div>
              <span class="hint num">
                {estimate.complete ? '' : 'At least '}{formatBytes(estimate.bytes)} of {formatBytes(cap)} cap{capMiB === null ? ' (Settings default)' : ''}.
              </span>
            </div>
          </div>
          {#if ratio > 1}
            <p class="note warn small">The estimate is above the byte cap. The worker stops the trace if the export does not fit. Select fewer layers, positions or roles, or increase the cap.</p>
          {:else if !estimate.complete}
            <p class="xs muted">Enter the hidden size{roles.includes('kq_soft_max') ? ' and the head count' : ''} for a full estimate. The worker checks the real size against the cap.</p>
          {/if}
        </div>
      </fieldset>

      <Advanced changed={advChanged}>
        <div class="stack">
          <div class="grid-2">
            <div class="field">
              <label for="{uid}-cap">Byte cap in MiB <span class="muted">(optional)</span></label>
              <input id="{uid}-cap" class="input num" type="number" min="0.001" max={caps.max_byte_cap / MIB} step="any" bind:value={capMiB} inputmode="decimal" aria-describedby="{uid}-cap-h" />
              <span class="hint" id="{uid}-cap-h">Total for all prompts. Default {formatBytes(caps.default_byte_cap)}, at most {formatBytes(caps.max_byte_cap)} (Settings).</span>
            </div>
            <div class="checks">
              <label><input type="checkbox" bind:checked={fullDump} /> Full dump</label>
              <span class="xs muted">{large ? 'Required for this selection.' : 'Needed above 64 layers, 128 positions or 2048 selections. It does not increase the cap.'}</span>
            </div>
          </div>

          <fieldset class="group">
            <legend class="small">Interventions</legend>
            {#if !model}
              <p class="xs muted">Select a model to see its saved interventions and source traces.</p>
            {:else}
              <div class="grid-2">
                <div class="field">
                  <label for="{uid}-art">Saved intervention</label>
                  <select id="{uid}-art" class="input" bind:value={artifactId} disabled={!modelArtifacts.length}>
                    <option value="">{modelArtifacts.length ? 'None' : 'None for this model'}</option>
                    {#each modelArtifacts as a (a.id)}<option value={a.id}>{a.name}</option>{/each}
                  </select>
                </div>
                {#if artifact}
                  <div class="field">
                    <label for="{uid}-artv">Version</label>
                    <select id="{uid}-artv" class="input" bind:value={artifactVersion}>
                      {#each Array.from({ length: artifact.latest_version }, (_, i) => artifact.latest_version - i) as v (v)}<option value={v}>{v}{v === artifact.latest_version ? ' (latest)' : ''}</option>{/each}
                    </select>
                  </div>
                {/if}
              </div>

              {#each patches as p, i (i)}
                <div class="patch" role="group" aria-label="Activation patch {i + 1}">
                  <div class="field">
                    <label for="{uid}-ps{i}">Source trace</label>
                    <select id="{uid}-ps{i}" class="input" bind:value={p.source}>
                      <option value="">Select a trace</option>
                      {#each patchSources as j (j.id)}<option value={j.id}>{j.id.slice(0, 8)} · {formatTime(j.created_at)}</option>{/each}
                    </select>
                  </div>
                  <div class="field">
                    <label for="{uid}-pp{i}">Source prompt</label>
                    <input id="{uid}-pp{i}" class="input num" type="number" min="0" max="3" step="1" bind:value={p.prompt} inputmode="numeric" />
                  </div>
                  <div class="field">
                    <label for="{uid}-pl{i}">Layer</label>
                    <input id="{uid}-pl{i}" class="input mono" bind:value={p.layer} autocomplete="off" />
                  </div>
                  <div class="field">
                    <label for="{uid}-pr{i}">Role</label>
                    <select id="{uid}-pr{i}" class="input" bind:value={p.role}>
                      {#each PATCH_ROLES as r (r)}<option value={r}>{r}</option>{/each}
                    </select>
                  </div>
                  <div class="field">
                    <label for="{uid}-pq{i}">Positions</label>
                    <input id="{uid}-pq{i}" class="input mono" bind:value={p.positions} autocomplete="off" />
                  </div>
                  <button type="button" class="btn ghost small" aria-label="Remove patch {i + 1}" onclick={() => (patches = patches.filter((__, k) => k !== i))}>Remove</button>
                </div>
              {/each}
              <div class="row">
                <button type="button" class="btn small" disabled={patches.length >= MAX_PATCHES || !patchSources.length} onclick={addPatch}>Add activation patch</button>
                <span class="xs muted">{patchSources.length ? 'A patch replaces rows with the rows of a finished trace of the same model.' : 'Patches need a finished trace of this model.'}</span>
              </div>
            {/if}
          </fieldset>
        </div>
      </Advanced>

      <Submit {blockers} busy={s.busy} onclick={submit}>Queue trace</Submit>
      <SubmitStatus {s} />
    </div>
  </section>

  <section class="panel" aria-labelledby="tr-runs-h">
    <header><h2 id="tr-runs-h">Trace runs</h2><span class="xs muted">{traces.length} on the controller</span></header>
    {#if !traces.length}
      <p class="empty">No traces yet. Queue one above.</p>
    {:else}
      <div class="table-wrap runs">
        <table class="data stackable">
          <thead><tr><th scope="col">Run</th><th scope="col">Model</th><th scope="col">State</th><th scope="col">Queued</th></tr></thead>
          <tbody>
            {#each traces.slice(0, 20) as j (j.id)}
              <tr class:current={j.id === selected?.id}>
                <td data-label="Run"><a class="mono" href="#/trace?job={encodeURIComponent(j.id)}" aria-current={j.id === selected?.id ? 'true' : undefined}>{j.id.slice(0, 8)}</a></td>
                <td data-label="Model" class="break">{modelLabel(models.find((m) => m.id === j.input_ids[0])) || j.input_ids[0]}</td>
                <td data-label="State"><JobState job={j} /></td>
                <td data-label="Queued" class="num">{formatTime(j.created_at)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>

  {#if selected}
    <section class="panel" aria-labelledby="tr-sel-h">
      <header>
        <h2 id="tr-sel-h">Trace <span class="mono">{selected.id.slice(0, 8)}</span></h2>
        <div class="row"><JobState job={selected} /><JobLink id={selected.id} label="Job details" /></div>
      </header>
      <div class="body stack">
        {#if selected.state === 'queued'}
          <p class="small">Waiting in the queue. It starts when the jobs before it end.</p>
          <div><CancelButton job={selected} small={false} /></div>
        {:else if selected.state === 'running'}
          <JobProgress job={selected} label="Trace progress" />
          <div><CancelButton job={selected} small={false} /></div>
        {:else if selected.state === 'succeeded'}
          {#if detailError}<p class="note bad" role="alert">{detailError}</p>
          {:else if detail}<TraceResults jobId={selected.id} result={detail.result} hasSae={!!detail.options?.sae} />
          {:else}<p class="muted small">Loading…</p>{/if}
        {:else}
          <p class="note bad" role="alert">The trace did not finish ({selected.state}). {selected.error ?? ''}</p>
        {/if}
      </div>
    </section>
  {/if}
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .group {
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    padding: var(--s-2) var(--s-3) var(--s-3);
    margin: 0;
    min-width: 0;
  }
  .group legend {
    padding: 0 4px;
    color: var(--text-2);
    font-weight: 500;
  }
  .prompt-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s-2);
    font-size: var(--fs-sm);
    font-weight: 500;
    color: var(--text-2);
  }
  .checks {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--s-2) var(--s-4);
    font-size: var(--fs-sm);
  }
  .checks label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 32px;
    cursor: pointer;
  }
  .estimate {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    padding-top: var(--s-3);
    border-top: 1px solid var(--line-soft);
  }
  .meter {
    height: 10px;
    margin-top: 12px;
    border-radius: var(--r-pill);
    background: var(--surface-2);
    border: 1px solid var(--line-strong);
    overflow: hidden;
  }
  .meter .fill {
    height: 100%;
    background: var(--accent);
  }
  .meter .fill.over {
    background: var(--warn);
  }
  .patch {
    display: grid;
    grid-template-columns: minmax(10rem, 2fr) repeat(4, minmax(5rem, 1fr)) auto;
    gap: var(--s-2);
    align-items: end;
    padding: var(--s-2) 0;
    border-top: 1px solid var(--line-soft);
  }
  @media (max-width: 760px) {
    .patch {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }
  }
  .runs {
    max-height: 320px;
  }
  /* A bar, not a tint: status badges keep their contrast on the plain surface. */
  tr.current > td:first-child {
    box-shadow: inset 3px 0 0 var(--accent);
  }
</style>

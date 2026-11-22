<script lang="ts">
  import Combobox from '../components/Combobox.svelte';
  import { promptOptions } from '../lib/pickers';
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { capability, formatBytes } from '../lib/format';
  import { fieldError, visibleBlockers, type Blocker } from '../lib/blockers';
  import { globalJobEligibility, hostWorkflowEligibility } from '../lib/eligibility';
  import { unsavedChanges } from '../lib/unsaved-changes.svelte';
  import { focusAfterRemoval } from '../lib/focus';
  import { hashParam, route } from '../lib/route.svelte';
  import { readRetry } from '../lib/rerun';
  import WorkflowGate from '../components/WorkflowGate.svelte';
  import Blockers from '../components/Blockers.svelte';
  import FieldError from '../components/FieldError.svelte';
  import HostSelect from '../components/HostSelect.svelte';
  import ModelSelect from '../components/ModelSelect.svelte';
  import SubmitStatus from '../components/SubmitStatus.svelte';
  import RecentJobs from '../components/RecentJobs.svelte';
  import Advanced from '../components/Advanced.svelte';
  import BuildDiffReport from '../components/results/BuildDiffReport.svelte';
  import PairedQualityReport from '../components/results/PairedQualityReport.svelte';
  import { readBuildFileBounded, MAX_BUILD_FILE_BYTES, BUILD_DIFF_SCHEMA, PAIRED_QUALITY_SCHEMA, type BuildParsed } from '../lib/buildreports';

  const GGUF = ['gguf'];
  const prompts = $derived(brewery.state?.prompts ?? []);
  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);
  const setup = $derived(globalJobEligibility(brewery.state, 'compare'));

  let a = $state('');
  let b = $state('');
  let promptMode = $state<'saved' | 'text'>('saved');
  let promptId = $state('');
  let promptText = $state('');
  let host = $state('');

  let n = $state<number | undefined>(16);
  let threads = $state<number | undefined>();
  let think = $state(false);
  let raw = $state(false);
  let useGpu = $state(false);
  let gpuDevice = $state('');
  let gpuLayers = $state<number | undefined>();

  const selectedPrompt = $derived(prompts.find((p) => p.id === promptId));
  const effectiveHost = $derived(host || models.find((m) => m.id === a)?.host || '');
  const touched = $state<Record<string, boolean>>({});
  let attempted = $state(false);
  let lastSubmittedPrompt = $state('');
  let retryMessage = $state('');
  let retryLoading = $state(false);
  unsavedChanges(() => promptText !== '' && promptText !== lastSubmittedPrompt);

  $effect(() => {
    const hash = route.hash;
    retryMessage = '';
    if (!hashParam(hash, 'retry')) {
      retryLoading = false;
      return;
    }
    const controller = new AbortController();
    retryLoading = true;
    readRetry(hash, ['compare'], controller.signal).then((job) => {
      if (!job || controller.signal.aborted) return;
      const [first, second] = job.input_ids;
      a = models.some((m) => m.id === first && m.kind === 'gguf') ? first : '';
      b = models.some((m) => m.id === second && m.kind === 'gguf') ? second : '';
      host = job.requested_host ?? job.host ?? '';
      promptId = job.prompt_id ?? '';
      promptText = job.prompt ?? '';
      promptMode = promptId ? 'saved' : 'text';
      // A restored prompt is not a new edit.
      lastSubmittedPrompt = promptText;
      const o = job.options;
      n = typeof o?.n === 'number' ? o.n : 16;
      threads = typeof o?.threads === 'number' ? o.threads : undefined;
      think = o?.think === true;
      raw = o?.raw === true;
      gpuDevice = typeof o?.gpu_device === 'string' ? o.gpu_device : '';
      gpuLayers = typeof o?.gpu_layers === 'number' ? o.gpu_layers : undefined;
      useGpu = !!gpuDevice || gpuLayers !== undefined;
      retryMessage = 'Review the previous job options before you queue a comparison.';
    }).catch(() => {
      if (!controller.signal.aborted) retryMessage = 'Could not open the previous job options. Enter the options again.';
    }).finally(() => {
      if (!controller.signal.aborted) retryLoading = false;
    });
    return () => controller.abort();
  });

  const blockers = $derived.by((): Blocker[] => {
    const out: Blocker[] = [];
    if (!a) out.push({ field: 'model-a', text: 'Select model A.' });
    if (!b) out.push({ field: 'model-b', text: 'Select model B.' });
    else if (a === b) out.push({ field: 'model-b', text: 'Select a different model.' });
    if (promptMode === 'saved' && !selectedPrompt) out.push({ field: 'saved', text: 'Select a saved prompt.' });
    if (promptMode === 'text' && !promptText.trim()) out.push({ field: 'text', text: 'Enter a prompt.' });
    if (n === undefined || !Number.isInteger(n) || n < 0 || n > 128) out.push({ field: 'n', text: 'Enter a whole number from 0 to 128.' });
    if (threads != null && (!Number.isInteger(threads) || threads < 1 || threads > 64)) out.push({ field: 'threads', text: 'Enter a whole number from 1 to 64.' });
    if (useGpu) {
      if (gpuLayers != null && (!Number.isInteger(gpuLayers) || gpuLayers < 1)) out.push({ field: 'layers', text: 'Enter a positive whole number.' });
      if (gpuDevice && !/^[A-Za-z0-9_.:-]{1,64}$/.test(gpuDevice)) out.push({ field: 'device', text: 'Use a device name such as CUDA0 (letters, digits, _ . : -).' });
    }
    if (effectiveHost) {
      const worker = hosts.find((h) => h.id === effectiveHost);
      const name = worker?.name ?? effectiveHost;
      const entry = hostWorkflowEligibility(brewery.state?.eligibility, effectiveHost, 'compare');
      const legacy = entry ? null : capability(worker, 'compare');
      const reasons = entry?.reasons ?? (worker?.capability_reasons?.compare ? [worker.capability_reasons.compare] : []);
      if (entry?.eligible === false || legacy?.allowed === false) {
        if (reasons.length) {
          for (const r of reasons) out.push({ field: 'host', host: name, text: r.text, fixAnchor: r.fix_anchor });
        } else out.push({ field: 'host', host: name, text: legacy?.reason ?? 'This host cannot run comparisons.' });
      }
    }
    return out;
  });
  const visible = $derived(visibleBlockers(blockers, touched, attempted));
  const error = (field: string) => fieldError(visible, field);
  const touch = (field: string) => () => { touched[field] = true; };

  // ---- Offline build analysis reports (browser-local, never uploaded) ----
  let reports = $state<{ key: string; name: string; parsed: BuildParsed }[]>([]);
  let reading = $state(false);
  let rseq = 0;
  let fileInput: HTMLInputElement | undefined = $state();
  async function openReports(files: FileList | null) {
    if (!files?.length) return;
    reading = true;
    const next: typeof reports = [];
    for (const f of Array.from(files)) next.push({ key: `${rseq++}`, name: f.name, parsed: await readBuildFileBounded(f) });
    reports = [...reports, ...next];
    reading = false;
    if (fileInput) fileInput.value = '';
  }
  const reportTitle = { 'build-diff': 'Build difference', 'paired-quality': 'Paired GSM8K outcomes', error: 'Not opened' } as const;

  const advChanged = $derived(
    [think ? 'thinking mode' : '', raw ? 'raw prompt' : '', useGpu ? `GPU${gpuDevice ? ` ${gpuDevice}` : ''}${gpuLayers ? `, ${gpuLayers} layers` : ''}` : '', threads ? `${threads} threads` : ''].filter(Boolean),
  );

  const s = new Submitter();
  async function submit(event: SubmitEvent) {
    event.preventDefault();
    attempted = true;
    if (blockers.length || s.busy || retryLoading) return;
    const text = promptMode === 'text' ? promptText : undefined;
    const job = await s.submit({
      kind: 'compare',
      host: host || undefined,
      input_ids: [a, b],
      prompt_id: promptMode === 'saved' ? promptId : undefined,
      prompt: text,
      options: {
        n,
        threads: threads ?? undefined,
        think: think || undefined,
        raw: raw || undefined,
        gpu_device: useGpu ? gpuDevice || undefined : undefined,
        gpu_layers: useGpu ? (gpuLayers ?? undefined) : undefined,
      },
    });
    if (job && text !== undefined) lastSubmittedPrompt = text;
  }
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Compare</h1>
    <p class="muted small">
      Run one prompt through two GGUF models and trace each run. Different tokenizers are shown side by side. Trace values are compared directly only when inputs, positions and signal meanings match.
    </p>
  </div>

  <section class="panel" aria-labelledby="cmp-h">
    <header><h2 id="cmp-h">New comparison</h2></header>
    <WorkflowGate workflow="compare" decision={setup}>
    <form class="body stack" onsubmit={submit} novalidate>
      {#if retryLoading || retryMessage}
        <p class="small muted" role="status">{retryLoading ? 'Loading the previous job options.' : retryMessage}</p>
      {/if}
      <fieldset class="retry-fields" disabled={retryLoading}>
      <div class="grid-2">
        <ModelSelect bind:value={a} kinds={GGUF} label="Model A" exclude={b ? [b] : []} error={error('model-a')} onblur={touch('model-a')} />
        <ModelSelect bind:value={b} kinds={GGUF} label="Model B" exclude={a ? [a] : []} error={error('model-b')} onblur={touch('model-b')} />
      </div>

      <fieldset class="group">
        <legend class="small">Prompt</legend>
        <div class="seg" role="radiogroup" aria-label="Prompt source">
          <label><input type="radio" name="compare-prompt" bind:group={promptMode} value="saved" /> Saved prompt</label>
          <label><input type="radio" name="compare-prompt" bind:group={promptMode} value="text" /> Type a prompt</label>
        </div>
        {#if promptMode === 'saved'}
          <div class="field">
            <Combobox bind:value={promptId} id="cmp-prompt" label="Saved prompt" labelHidden options={promptOptions(prompts)} disabled={!prompts.length} placeholder={prompts.length ? 'Select or type a prompt name' : 'No saved prompts'} emptyText="No saved prompt matches." invalid={!!error('saved')} describedby={error('saved') ? 'cmp-prompt-error' : undefined} onblur={touch('saved')} />
            <FieldError id="cmp-prompt-error" error={error('saved')} />
            {#if !prompts.length}<span class="hint">Save prompts in Library, or type one here.</span>{/if}
          </div>
          {#if selectedPrompt}<pre class="block">{selectedPrompt.text}</pre>{/if}
        {:else}
          <div class="field">
            <label for="cmp-text" class="sr-only">Prompt text</label>
            <textarea id="cmp-text" class="input" rows="4" bind:value={promptText} aria-invalid={error('text') ? 'true' : undefined} aria-describedby={error('text') ? 'cmp-text-error' : undefined} onblur={touch('text')}></textarea>
            <FieldError id="cmp-text-error" error={error('text')} />
            <span class="hint">Sent exactly as typed. Save it in Library to reuse it.</span>
          </div>
        {/if}
      </fieldset>

      <div class="grid-2">
        <HostSelect bind:value={host} kind="compare" workflow="compare" label="Run on" optional optionalLabel="Host of model A" hint="Missing models are copied to this host first." error={error('host')} onblur={touch('host')} />
        <div class="field">
          <label for="cmp-n">Decode tokens</label>
          <input id="cmp-n" class="input num" type="number" min="0" max="128" step="1" bind:value={n} inputmode="numeric" aria-invalid={error('n') ? 'true' : undefined} aria-describedby={error('n') ? 'cmp-n-error' : 'cmp-n-hint'} onblur={touch('n')} />
          <FieldError id="cmp-n-error" error={error('n')} />
          <span class="hint" id="cmp-n-hint">0 to 128. Default 16.</span>
        </div>
      </div>

      <Advanced changed={advChanged}>
        <div class="checks">
          <label><input type="checkbox" bind:checked={think} /> Thinking mode</label>
          <label><input type="checkbox" bind:checked={raw} /> Raw prompt (no chat template)</label>
          <label><input type="checkbox" bind:checked={useGpu} /> Offload to GPU</label>
        </div>
        <div class="grid-3">
          <div class="field">
            <label for="cmp-threads">CPU threads <span class="muted">(optional)</span></label>
            <input id="cmp-threads" class="input num" type="number" min="1" max="64" step="1" bind:value={threads} inputmode="numeric" aria-invalid={error('threads') ? 'true' : undefined} aria-describedby={error('threads') ? 'cmp-threads-error' : undefined} onblur={touch('threads')} />
            <FieldError id="cmp-threads-error" error={error('threads')} />
          </div>
          {#if useGpu}
            <div class="field">
              <label for="cmp-dev">GPU device <span class="muted">(optional)</span></label>
              <input id="cmp-dev" class="input mono" bind:value={gpuDevice} placeholder="CUDA0…" autocomplete="off" spellcheck="false" aria-invalid={error('device') ? 'true' : undefined} aria-describedby={error('device') ? 'cmp-dev-error' : undefined} onblur={touch('device')} />
              <FieldError id="cmp-dev-error" error={error('device')} />
            </div>
            <div class="field">
              <label for="cmp-layers">GPU layers <span class="muted">(optional)</span></label>
              <input id="cmp-layers" class="input num" type="number" min="1" step="1" bind:value={gpuLayers} inputmode="numeric" aria-invalid={error('layers') ? 'true' : undefined} aria-describedby={error('layers') ? 'cmp-layers-error' : undefined} onblur={touch('layers')} />
              <FieldError id="cmp-layers-error" error={error('layers')} />
            </div>
          {/if}
        </div>
        <p class="xs muted">By default the runs use the CPU and the chat template.</p>
      </Advanced>

      <div class="row">
        <button type="submit" class="btn primary" disabled={s.busy || retryLoading} aria-busy={s.busy}>{s.busy ? 'Submitting…' : 'Queue comparison'}</button>
      </div>
      <Blockers blockers={visible} />
      <SubmitStatus {s} />
      </fieldset>
    </form>
    </WorkflowGate>
  </section>

  <RecentJobs kinds={['compare']} title="Recent comparisons" />

  <section class="panel" aria-labelledby="ba-h">
    <header><h2 id="ba-h">Build analysis reports</h2></header>
    <div class="body stack">
      <p class="small muted">
        Open exported <code>{BUILD_DIFF_SCHEMA}</code> or <code>{PAIRED_QUALITY_SCHEMA}</code> files to read two builds side by side. Each file is shown on its own: these reports carry no digest that links a build difference to a quality result, so the page does not match them.
      </p>
      <div class="field">
        <label for="ba-files">Report files (JSON)</label>
        <input id="ba-files" bind:this={fileInput} class="input" type="file" accept=".json,application/json" multiple disabled={reading} aria-describedby="ba-files-hint" onchange={(e) => openReports(e.currentTarget.files)} />
        <span class="hint" id="ba-files-hint">Read in this browser tab only. Nothing goes to the controller. At most {formatBytes(MAX_BUILD_FILE_BYTES)} per file.</span>
      </div>
      <p class="sr-only" aria-live="polite">{reading ? 'Reading files…' : reports.length ? `${reports.filter((x) => x.parsed.kind !== 'error').length} of ${reports.length} files opened.` : ''}</p>
    </div>
  </section>

  {#each reports as rep (rep.key)}
    <section class="panel" aria-labelledby="ba-{rep.key}-h">
      <header>
        <h2 id="ba-{rep.key}-h" class="rh"><span>{reportTitle[rep.parsed.kind]}</span> <span class="mono xs muted break">{rep.name}</span></h2>
        <button type="button" class="btn small ghost" aria-label="Close {rep.name}" onclick={() => focusAfterRemoval(() => { reports = reports.filter((x) => x.key !== rep.key); }, () => fileInput)}>Close</button>
      </header>
      <div class="body">
        {#if rep.parsed.kind === 'error'}<p class="note bad" role="alert">{rep.parsed.message}</p>
        {:else if rep.parsed.kind === 'build-diff'}<BuildDiffReport report={rep.parsed.report} />
        {:else}<PairedQualityReport report={rep.parsed.report} />{/if}
      </div>
    </section>
  {/each}
</div>

<style>
  .retry-fields {
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
    margin: 0;
    padding: 0;
    border: 0;
    min-width: 0;
  }
  .rh {
    display: flex;
    flex-wrap: wrap;
    gap: 0 8px;
    align-items: baseline;
    min-width: 0;
  }
  input[type='file'] {
    padding: 6px;
  }
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
  .seg,
  .checks {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-4);
    font-size: var(--fs-sm);
  }
  .seg label,
  .checks label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 32px;
    cursor: pointer;
  }
  input[type='radio'] {
    accent-color: var(--accent);
    width: 16px;
    height: 16px;
  }
</style>

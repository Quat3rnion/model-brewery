<script lang="ts">
  import { SvelteMap } from 'svelte/reactivity';
  import { api, errorText } from '../lib/api';
  import { STAGES, type ConversionBody } from '../lib/conversion';
  import { conversions, type ConversionStarter } from '../lib/conversions.svelte';
  import { capability, hostName, isActive, modelLabel } from '../lib/format';
  import { inventoryBlocker, validSha } from '../lib/ggufSource';
  import { obj } from '../lib/reports';
  import { brewery } from '../lib/store.svelte';
  import type { JobDetail, JobSummary, Model } from '../lib/types';
  import { inventoryJob, readInventory, walkInventory } from '../lib/weightmap';
  import JobState from './JobState.svelte';
  import Submit from './Submit.svelte';
  import { trellisProblem } from '../lib/trellis-layer';

  let { source, format, starter }: { source: Model; format: 'gguf' | 'hgn'; starter: ConversionStarter } = $props();

  const submitted = new SvelteMap<number, JobSummary | null>();
  const loaded = new SvelteMap<string, JobDetail | 'rejected' | null>();
  let error = $state('');
  const host = $derived(brewery.state?.hosts.find(h => h.id === source.host));
  const blocked = $derived(inventoryBlocker(source, host));
  const expectedSha = $derived(source.kind === 'hf'
    ? validSha(source.bundle_sha256) ?? validSha(source.sha256)
    : validSha(source.sha256));
  const currentJobs = $derived([...submitted.values()].flatMap(job =>
    job ? [brewery.state?.jobs.find(j => j.id === job.id) ?? job] : []));
  // Terminal inventory jobs may be reused only after their receipts have been
  // checked. The exporter verifies those bytes again before it writes.
  const jobs = $derived([
    ...currentJobs,
    ...(brewery.state?.jobs ?? []).filter(job => job.kind === 'weight_inspect' &&
      job.host === source.host && job.input_ids[0] === source.id &&
      (isActive(job.state) || (job.state === 'succeeded' && job.has_result !== false)) &&
      !currentJobs.some(current => current.id === job.id)),
  ].filter(job => loaded.get(job.id) !== 'rejected'));
  const details = $derived(Object.fromEntries([...loaded].filter(([, detail]) => detail && detail !== 'rejected')) as Record<string, JobDetail>);
  const inventory = $derived(walkInventory(source, jobs, details, { format: source.kind, sha: expectedSha, matchHost: true }));
  // Physical HF tree digest of the pinned release (qualification/release_matrix.md).
  const flashDigest = '0eb8fb9e3b0f2422297174c880aa2ad8304e77a34d37b15c592ded1c3f9b4cd1';
  const flash = $derived(source.kind === 'hf' &&
    (expectedSha === flashDigest || inventory.pages[0]?.source_sha256 === flashDigest));

  const readPending = $derived(!!inventory.need && submitted.get(inventory.need.offset) === null);
  const readJob = $derived(inventory.job ?? currentJobs.at(-1));

  function readLayers() {
    const next = inventory.need;
    if (flash || blocked || error || !next || submitted.has(next.offset)) return;
    const { offset, sha } = next;
    const request = inventoryJob(source.id, source.host, offset, sha);
    submitted.set(offset, null);
    void api.submitJob(request).then(job => {
      if (job.kind !== request.kind || job.host !== request.host || job.input_ids?.length !== 1 || job.input_ids[0] !== request.input_ids[0]) {
        error = 'The inventory job does not match the selected source.';
        return;
      }
      submitted.set(offset, job);
      void brewery.refresh();
    }).catch(e => { error = errorText(e); });
  }

  $effect(() => {
    const job = inventory.job;
    if (flash || error || job?.state !== 'succeeded') return;
    const jobId = job.id;
    const fresh = currentJobs.some(current => current.id === jobId);
    function reject(reason: string) {
      if (fresh) error = reason;
      else loaded.set(jobId, 'rejected');
    }
    if (inventory.error) {
      reject(inventory.error);
      return;
    }
    if (loaded.has(jobId)) return;
    const { id, host, kind } = source;
    loaded.set(jobId, null);
    void brewery.loadJob(jobId).then(detail => {
      const inputs = obj(detail.result)?.inputs;
      const input = obj(Array.isArray(inputs) && inputs.length === 1 ? inputs[0] : undefined);
      const page = readInventory(detail.result, kind);
      if (detail.id !== jobId || detail.state !== 'succeeded' || input?.id !== id || input?.kind !== kind || input?.source_host !== host ||
          !page || input.sha256 !== page.source_sha256 ||
          (kind === 'hf' && input.bundle_sha256 !== page.source_sha256)) {
        reject('The inventory report does not match the selected source or its digest.');
        return;
      }
      loaded.set(jobId, detail);
    }).catch(e => reject(errorText(e)));
  });

  const common = ['BF16', 'F16', 'F32'] as const;
  const trellis = ['HT3','HT4'] as const;
  type OutputType = typeof common[number] | 'Q8_0' | 'Q8G64' | typeof trellis[number];
  const formatTypes = $derived<readonly OutputType[]>([...common, format === 'gguf' ? 'Q8_0' : 'Q8G64']);
  let defaultType = $state<OutputType>('BF16');
  let overrides = $state<Record<number, OutputType | ''>>({});
  const complete = $derived(!flash && !error && !inventory.need && !inventory.job && inventory.pages.length > 0);
  const problem = $derived(flash ? 'No layer-recipe exporter admits this pinned Flash-Next checkpoint.'
    : error || (source.available === false || (!complete && !inventory.job) ? blocked : null) ||
      (inventory.job?.state === 'succeeded' ? null : inventory.error));
  const layers = $derived(complete ? [...new Set(inventory.pages.flatMap(({ tensors }) => tensors.flatMap(t => {
    const match = /^(?:model\.layers|transformer\.h)\.(0|[1-9]\d*)\./.exec(t.name);
    const layer = Number(match?.[1]);
    return t.shape.length === 2 && Number.isSafeInteger(layer) ? [layer] : [];
  })))].sort((a, b) => a - b) : []);
  const sourceSha = $derived(complete ? inventory.pages[0].source_sha256 : null);
  const trellisIssues = $derived.by(() => {
    if (format !== 'hgn') return {};
    const byName = new Map(inventory.pages.flatMap(p=>p.tensors.map(t=>[t.name,t] as const)));
    return Object.fromEntries(layers.map(n=>[n,trellisProblem(n,byName)]));
  });
  const incompatible = $derived([...new Set([
    ...(!formatTypes.includes(defaultType) ? [defaultType] : []),
    ...layers.map(n=>overrides[n]).filter(type=>type && !formatTypes.includes(type) && !(format==='hgn' && (type==='HT3' || type==='HT4'))),
  ])]);
  const badGeometry = $derived(layers.some(n=>trellisIssues[n] && (overrides[n]==='HT3' || overrides[n]==='HT4')));
  const layerTypes = $derived(Object.fromEntries(layers.filter(layer => overrides[layer]).map(layer => [layer, overrides[layer]])));
  const tooMany = $derived(Object.keys(layerTypes).length > 256);
  const recipe = $derived(!problem && sourceSha && layers.length && !incompatible.length && !badGeometry && !tooMany ? {
    schema: 'brewery.layer-quant.v1',
    source_sha256: sourceSha,
    default_type: defaultType,
    layers: layerTypes,
  } : null);

  const standalone = $derived(source.kind === 'safetensors');
  const assets = $derived((brewery.state?.models ?? []).filter(m => m.kind === 'assets' && m.host === source.host && m.available !== false));
  let assetId = $state('');
  const selectedAsset = $derived(assets.find(asset => asset.id === assetId));
  const requirements = $derived(format === 'hgn' ? ['hgn_layer_recipe', ...STAGES]
    : ['gguf_layer_recipe', 'gguf_output', standalone ? 'gguf_safetensors_output' : 'gguf_hf_output', ...STAGES]);
  const setup = $derived(flash ? '' : !host ? 'The controller did not report a worker for this source.'
    : [...new Set(requirements.map(kind => capability(host, kind)).flatMap(c => !c.allowed && c.reason ? [c.reason] : []))].join(' '));
  const runBlockers = $derived.by(() => {
    if (flash) return ['Select a checkpoint supported by the layer-recipe exporter.'];
    const out: string[] = [];
    if (setup) out.push(setup);
    if (conversions.status === 'unavailable') out.push('This controller has no managed conversion workflow.');
    if (!recipe && !problem) out.push(!complete ? 'Read the source layers first.' : !layers.length
      ? 'Choose a source with recognized layer matrices.' : tooMany ? 'Use at most 256 layer overrides.' : 'Change the incompatible types.');
    if (problem) out.push('Resolve the source inventory problem above.');
    if (format === 'hgn' && recipe && !Object.values(recipe.layers).some(type => type === 'HT3' || type === 'HT4')) {
      out.push('Select HT3 or HT4 for at least one layer to run a managed HGN recipe.');
    }
    if (standalone && !selectedAsset) out.push('Select matching config and tokenizer assets on the source host.');
    return out;
  });
  const body = $derived<ConversionBody | null>(recipe && !runBlockers.length ? {
    format,
    host: source.host,
    input_ids: selectedAsset && standalone ? [source.id, selectedAsset.id] : [source.id],
    options: { layer_recipe: recipe },
  } : null);
  let lastBody = $state('');
  const currentSubmission = $derived(body !== null && lastBody === JSON.stringify(body));

  async function run() {
    if (!body || starter.busy) return;
    lastBody = JSON.stringify(body);
    await starter.start(body);
  }

  function download() {
    if (!recipe) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(recipe, null, 2) + '\n'], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'layer-recipe.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
</script>

{#snippet typeOptions(value: OutputType | '', layer?: number)}
  {#each formatTypes as type (type)}<option value={type}>{type}</option>{/each}
  {#if layer !== undefined && format === 'hgn'}
    {#each trellis as type}<option value={type} disabled={!!trellisIssues[layer]}>{type}</option>{/each}
  {/if}
  {#if value && !formatTypes.includes(value) && !(layer !== undefined && format === 'hgn' && (value === 'HT3' || value === 'HT4'))}
    <option {value} disabled>{value} (not supported for {format.toUpperCase()})</option>
  {/if}
{/snippet}

{#if readJob && !flash}
  <div class="row" role="status">
    <a href="#/jobs/{encodeURIComponent(readJob.id)}">Layer read job</a>
    <JobState job={readJob} />
  </div>
{/if}
{#if problem}
  <p class="note bad" role="alert">{problem}</p>
{:else if inventory.need}
  <div class="row">
    <button type="button" class="btn" onclick={readLayers} disabled={readPending}>
      {inventory.pages.length ? 'Read next layers' : 'Read layers'}
    </button>
    <span class="hint">This queues a <code>weight_inspect</code> job on {hostName(brewery.state?.hosts ?? [], source.host)}.</span>
  </div>
  {#if readPending}<p class="hint" role="status">Submitting the layer read.</p>{/if}
{:else if !complete}
  <p class="muted small" role="status">Waiting for the source SHA-256 and layers.</p>
{:else if !layers.length}
  <p class="note warn small">The source inventory does not list any recognized layer matrices.</p>
{:else}
  <p class="small">Source SHA-256: <code class="mono" style="overflow-wrap:anywhere">{sourceSha}</code></p>
  <p class="hint">Choose a default type and optional layer overrides. The exporter checks the architecture, assets and tensor layout before it writes output.</p>
  <div class="grid-3">
    <label class="field">
      Default type
      <select class="input" bind:value={defaultType}>
        {@render typeOptions(defaultType)}
      </select>
    </label>
    {#each layers as layer (layer)}
      <div class="field">
        <label for="recipe-layer-{layer}">Layer {layer} type</label>
        <select id="recipe-layer-{layer}" class="input" aria-describedby={trellisIssues[layer] ? `ht-help-${layer}` : undefined} bind:value={overrides[layer]}>
          <option value="">Default ({defaultType})</option>
          {@render typeOptions(overrides[layer] ?? '', layer)}
        </select>
        {#if trellisIssues[layer]}<span class="hint" id="ht-help-{layer}">HT3/HT4 unavailable. {trellisIssues[layer]}</span>{/if}
      </div>
    {/each}
  </div>
  {#if incompatible.length}
    <p id="recipe-codec-warning" class="note warn small" role="alert">{incompatible.join(', ')} cannot be used for {format.toUpperCase()}. Change the incompatible types or switch formats.</p>
  {/if}
  {#if tooMany}<p class="note warn small" role="alert">Use at most 256 layer overrides.</p>{/if}
  <div><button type="button" class="btn" onclick={download} disabled={!recipe} aria-describedby={incompatible.length ? 'recipe-codec-warning' : undefined}>Download recipe</button></div>
  {#if format === 'gguf'}
    <p class="hint">The GGUF exporter supports GPT-2 and Qwen2 layouts. Layer overrides apply to decoder layer matrices. The default also applies to other eligible matrices.</p>
  {:else}
    <p class="hint">Offline HGN exporter: <code>python3 tools/layer_recipe_hgn.py SOURCE layer-recipe.json OUTPUT --helper PATH</code>. For a standalone file, also supply <code>--assets DIR</code>.</p>
    <p class="hint">Managed native HGN checks dense Llama and Qwen2 layouts on a configured local worker. HT3 and HT4 apply only to layer matrices. Verification checks the archive structure; it does not check runtime loading or model quality.</p>
    <p class="hint">Native HGN HT3/HT4 is not EXL3 or a reference QTIP checkpoint. GGUF KT encoding is not available through this recipe.</p>
  {/if}
{/if}

{#if standalone}
  <label class="field">
    Config and tokenizer assets
    <select class="input" bind:value={assetId} disabled={!assets.length}>
      <option value="">Select matching assets</option>
      {#each assets as asset (asset.id)}<option value={asset.id}>{modelLabel(asset)} · {asset.path}</option>{/each}
    </select>
  </label>
  {#if !assets.length}<p class="hint"><a href="#/library">Register the matching assets in Library</a> on {host?.name ?? source.host}.</p>{/if}
{/if}
<Submit blockers={runBlockers} busy={starter.busy} onclick={run}>Run layer recipe</Submit>
{#if setup}
  <p class="hint">Ask an operator to configure managed {format.toUpperCase()} layer recipes on this worker. <a href="#/settings/infrastructure?host={encodeURIComponent(source.host)}">Open host settings</a>.</p>
{/if}
<div aria-live="polite" class="xs">
  {#if currentSubmission && starter.error}
    <p class="note bad" role="alert">{starter.error} Run the same recipe again to retry the same request.</p>
  {:else if currentSubmission && !starter.busy && starter.last}
    <p class="muted">{starter.replayed ? 'This request already started conversion' : 'Started conversion'} <span class="mono">{starter.last.id}</span>. {#if starter.replayed}No second conversion was started. {/if}
      <button type="button" class="btn ghost small" onclick={() => {
        (document.getElementById(`run-${starter.last?.id}`) ?? document.getElementById('conversion-runs'))?.scrollIntoView({ block: 'start' });
      }}>View conversion progress</button>
    </p>
  {/if}
</div>

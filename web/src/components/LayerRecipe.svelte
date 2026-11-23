<script module lang="ts">
  const common = ['BF16', 'F16', 'F32'] as const;
  export const trellis = ['HT3', 'HT4'] as const;
  export type OutputType = typeof common[number] | 'Q8_0' | 'Q8G64' | typeof trellis[number];
  export const recipeTypes = (format: 'gguf' | 'hgn'): readonly OutputType[] =>
    [...common, format === 'gguf' ? 'Q8_0' : 'Q8G64'];
  export interface RecipeDraft {
    defaultType: OutputType;
    overrides: Record<number, OutputType | ''>;
    assetId: string;
  }
</script>

<script lang="ts">
  import { untrack } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';
  import { conversionRunHref, STAGES, type ConversionBody } from '../lib/conversion';
  import type { Blocker } from '../lib/blockers';
  import { conversions, type ConversionStarter } from '../lib/conversions.svelte';
  import { capability, hostName, isActive, modelLabel } from '../lib/format';
  import { roleLabel } from '../lib/components';
  import { inventoryBlocker, validSha } from '../lib/ggufSource';
  import { obj } from '../lib/reports';
  import { brewery } from '../lib/store.svelte';
  import type { JobDetail, JobSummary, Model } from '../lib/types';
  import { inventoryJob, readInventory, walkInventory } from '../lib/weightmap';
  import JobState from './JobState.svelte';
  import Blockers from './Blockers.svelte';
  import Submit from './Submit.svelte';
  import SubmitStatus from './SubmitStatus.svelte';
  import { acceptedJobSummary, guardReadLink, jobRequestKey, ReadVisit, Submitter } from '../lib/submit.svelte';
  import { captureResultFocus } from '../lib/focus';
  import { trellisProblem } from '../lib/trellis-layer';
  import { bodyKey, type PlainAttempt } from '../lib/submission-key';

  let { source, format, starter, readAttempts, acceptedReads, draft = $bindable() }: {
    source: Model; format: 'gguf' | 'hgn'; starter: ConversionStarter; readAttempts: PlainAttempt;
    acceptedReads: SvelteMap<string, JobSummary>; draft: RecipeDraft;
  } = $props();

  const launcher = 'convert.recipe.inventory';
  const loaded = new SvelteMap<string, JobDetail | null>();
  const rejected = new SvelteMap<string, string>();
  const attempted = new Map<string, Date | null>();
  const reads = new Submitter(launcher, untrack(() => readAttempts));
  const readVisit = new ReadVisit();
  let receiptError = $state('');
  const host = $derived(brewery.state?.hosts.find(h => h.id === source.host));
  const blocked = $derived(inventoryBlocker(source, host));
  const expectedSha = $derived(source.kind === 'hf'
    ? validSha(source.bundle_sha256) ?? validSha(source.sha256)
    : validSha(source.sha256));
  const currentJobs = $derived([...acceptedReads.values()]
    .filter(job => job.host === source.host && job.input_ids[0] === source.id)
    .map(job => brewery.state?.jobs.find(j => j.id === job.id) ?? brewery.details[job.id] ?? job)
    .filter(job => readVisit.shows(job)));
  // Terminal inventory jobs may be reused only after their receipts have been
  // checked. The exporter verifies those bytes again before it writes.
  const jobs = $derived([
    ...currentJobs,
    ...(brewery.state?.jobs ?? []).filter(job => job.kind === 'weight_inspect' &&
      job.host === source.host && job.input_ids[0] === source.id &&
      (isActive(job.state) || (job.state === 'succeeded' && job.has_result !== false)) &&
      !currentJobs.some(current => current.id === job.id)),
  ].filter(job => !rejected.has(job.id)));
  const details = $derived(Object.fromEntries([...loaded].filter(([, detail]) => detail)) as Record<string, JobDetail>);
  const inventory = $derived(walkInventory(source, jobs, details, { format: source.kind, sha: expectedSha, matchHost: true }));
  // Physical HF tree digest of the pinned release (qualification/release_matrix.md).
  const flashDigest = '0eb8fb9e3b0f2422297174c880aa2ad8304e77a34d37b15c592ded1c3f9b4cd1';
  const flash = $derived(source.kind === 'hf' &&
    (expectedSha === flashDigest || inventory.pages[0]?.source_sha256 === flashDigest));

  const readPending = $derived(!!inventory.need && reads.busy);
  const readJob = $derived(inventory.job ?? currentJobs.at(-1));
  let readLink: HTMLAnchorElement | undefined = $state();

  async function readLayers(event: MouseEvent) {
    const next = inventory.need;
    if (flash || blocked || !next) return;
    const { offset, sha } = next;
    const request = inventoryJob(source.id, source.host, offset, sha);
    const key = jobRequestKey(launcher, request);
    const focus = captureResultFocus(event);
    const accepted = await reads.submit(request, event, false, admitted => { if (admitted) receiptError = ''; });
    if (!accepted) return;
    const id = accepted.id;
    function record(job: JobSummary) {
      if (job.id !== id || job.kind !== request.kind || job.host !== request.host || job.input_ids?.length !== 1 || job.input_ids[0] !== request.input_ids[0]) {
        receiptError = 'The inventory job does not match the selected source.';
        return;
      }
      readVisit.remember(id);
      acceptedReads.set(key, job);
    }
    record(acceptedJobSummary(accepted, request));
    void focus?.(() => readLink);
    // A match can return only the ID. If the detail GET fails, wait for the
    // state poll to replace the placeholder instead of latching an error.
    if (!('state' in accepted)) void brewery.loadJob(id).then(detail => {
      if (acceptedReads.get(key)?.id === id) record(detail);
    }).catch(() => {});
  }

  $effect(() => {
    // Try a transient detail error again after the next successful state poll.
    const snapshot = brewery.lastUpdated;
    const job = inventory.job;
    if (flash || job?.state !== 'succeeded') return;
    const jobId = job.id;
    function reject(reason: string) {
      rejected.set(jobId, reason);
    }
    if (inventory.error) {
      reject(inventory.error);
      return;
    }
    if (loaded.has(jobId) || attempted.get(jobId) === snapshot) return;
    attempted.set(jobId, snapshot);
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
    }).catch(() => loaded.delete(jobId));
  });

  const formatTypes = $derived(recipeTypes(format));
  const complete = $derived(!flash && !inventory.need && !inventory.job && inventory.pages.length > 0);
  const problem = $derived(flash ? 'The layer-recipe exporters do not support this Flash-Next checkpoint.'
    : source.available === false || (!complete && !inventory.job) ? blocked : null);
  const readError = $derived(receiptError || (inventory.job?.state === 'succeeded' ? null : inventory.error) ||
    (readJob && rejected.get(readJob.id)));
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
  const trellisReasons = $derived([...new Set(Object.values(trellisIssues).filter(Boolean))]);
  const incompatible = $derived([...new Set([
    ...(!formatTypes.includes(draft.defaultType) ? [draft.defaultType] : []),
    ...layers.map(n=>draft.overrides[n]).filter(type=>type && !formatTypes.includes(type) && !(format==='hgn' && (type==='HT3' || type==='HT4'))),
  ])]);
  const badGeometry = $derived(layers.some(n=>trellisIssues[n] && (draft.overrides[n]==='HT3' || draft.overrides[n]==='HT4')));
  const layerTypes = $derived(Object.fromEntries(layers.filter(layer => draft.overrides[layer]).map(layer => [layer, draft.overrides[layer]])));
  const tooMany = $derived(Object.keys(layerTypes).length > 256);
  const recipe = $derived(!problem && sourceSha && layers.length && !incompatible.length && !badGeometry && !tooMany ? {
    schema: 'brewery.layer-quant.v1',
    source_sha256: sourceSha,
    default_type: draft.defaultType,
    layers: layerTypes,
  } : null);

  const standalone = $derived(source.kind === 'safetensors');
  const assets = $derived((brewery.state?.models ?? []).filter(m => m.kind === 'assets' && m.host === source.host && m.available !== false));
  const selectedAsset = $derived(assets.find(asset => asset.id === draft.assetId));
  const requirements = $derived(format === 'hgn' ? ['hgn_layer_recipe', ...STAGES]
    : ['gguf_layer_recipe', 'gguf_output', standalone ? 'gguf_safetensors_output' : 'gguf_hf_output', ...STAGES]);
  const setup = $derived.by((): Blocker[] => {
    const worker = hostName(brewery.state?.hosts ?? [], source.host);
    const fallback = `#/settings/infrastructure?host=${encodeURIComponent(source.host)}&section=policy`;
    if (!host) return [{ host: worker, text: 'The controller did not report this host', fixAnchor: fallback }];
    const reasons = host.scheduling_paused ? [{ text: 'Resume job scheduling', fix_anchor: fallback }] : [];
    for (const kind of requirements) {
      const cap = capability(host, kind);
      const reason = host.capability_reasons?.[kind] ?? { text: cap.reason ?? `Enable ${kind}`, fix_anchor: fallback };
      if (!cap.allowed && !reasons.some((r) => r.text === reason.text)) reasons.push(reason);
    }
    return reasons.map((r) => ({ host: worker, text: r.text, fixAnchor: r.fix_anchor }));
  });
  const runBlockers = $derived.by(() => {
    if (flash) return ['Select a checkpoint supported by the layer-recipe exporter.'];
    const out: string[] = [];
    if (setup.length) out.push('Use a host that supports this recipe output.');
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
  let attemptedBody = $state<string>();
  const currentAttempt = $derived(body !== null && attemptedBody === bodyKey(body));

  let runLink: HTMLAnchorElement | undefined = $state();
  function run(event: MouseEvent) {
    if (!body) return;
    attemptedBody = bodyKey(body);
    void starter.start(body, event, () => runLink);
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

<Blockers blockers={setup} />
<p class="hint">Supported architectures: {format === 'gguf' ? 'GPT-2 and Qwen2' : 'dense Llama and Qwen2'}. The exporter checks the architecture and assets before it writes output.</p>
{#if readJob && !flash}
  <div class="row" role="status">
    <a bind:this={readLink} href="#/jobs/{encodeURIComponent(readJob.id)}" onclick={(event) => guardReadLink(event, launcher)}>Layer read job</a>
    <JobState job={readJob} />
  </div>
{/if}
{#if reads.error || reads.replayed}<SubmitStatus s={reads} />{/if}
{#if readError}<p class="note bad" role="alert">{readError}</p>{/if}
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
  {#if trellisReasons.length}<p class="note warn small" id="recipe-ht-help">HT3 and HT4 are not available for some layers. {trellisReasons.join(' ')}</p>{/if}
  <p class="hint">Choose a default type and optional layer overrides.</p>
  <div class="grid-3">
    <label class="field">
      Default type
      <select class="input" bind:value={draft.defaultType}>
        {@render typeOptions(draft.defaultType)}
      </select>
    </label>
    {#each layers as layer (layer)}
      <div class="field">
        <label for="recipe-layer-{layer}">Layer {layer} type</label>
        <select id="recipe-layer-{layer}" class="input" aria-describedby={trellisIssues[layer] ? 'recipe-ht-help' : undefined} bind:value={draft.overrides[layer]}>
          <option value="">Default ({draft.defaultType})</option>
          {@render typeOptions(draft.overrides[layer] ?? '', layer)}
        </select>
      </div>
    {/each}
  </div>
  {#if incompatible.length}
    <p id="recipe-codec-warning" class="note warn small" role="alert">{incompatible.join(', ')} cannot be used for {format.toUpperCase()}. Change the incompatible types or switch formats.</p>
  {/if}
  {#if tooMany}<p class="note warn small" role="alert">Use at most 256 layer overrides.</p>{/if}
  <div><button type="button" class="btn" onclick={download} disabled={!recipe} aria-describedby={incompatible.length ? 'recipe-codec-warning' : undefined}>Download recipe</button></div>
  {#if format === 'gguf'}
    <p class="hint">Layer overrides apply to decoder layer matrices. The default also applies to other supported matrices.</p>
  {:else}
    <p class="hint">Offline HGN exporter: <code>python3 tools/layer_recipe_hgn.py SOURCE layer-recipe.json OUTPUT --helper PATH</code>. For a standalone file, also supply <code>--assets DIR</code>.</p>
    <p class="hint">HT3 and HT4 apply only to layer matrices. Verification checks the archive structure. It does not check runtime loading or model quality.</p>
    <p class="hint">Native HGN HT3/HT4 is not EXL3 or a reference QTIP checkpoint. GGUF KT encoding is not available through this recipe.</p>
  {/if}
{/if}

{#if standalone}
  <label class="field">
    {roleLabel('assets')}
    <select class="input" bind:value={draft.assetId} disabled={!assets.length}>
      <option value="">Select matching assets</option>
      {#each assets as asset (asset.id)}<option value={asset.id}>{modelLabel(asset)} · {asset.path}</option>{/each}
    </select>
  </label>
  {#if !assets.length}<p class="hint"><a href="#/library">Register the matching assets in Library</a> on {host?.name ?? source.host}.</p>{/if}
{/if}
<Submit blockers={runBlockers} busy={starter.busy} onclick={run}>Run layer recipe</Submit>
<div aria-live="polite" class="xs">
  {#if currentAttempt && starter.error}
    <p class="note bad" role="alert">{starter.error}</p>
  {:else if !starter.busy && starter.last}
    <p class="muted">{starter.replayed ? 'This request was already queued as job' : 'Started conversion'} <a bind:this={runLink} href={conversionRunHref(starter.last.id, 'recipe')}>{starter.last.id}</a>. {#if starter.replayed}No second conversion was started.{/if}</p>
  {/if}
</div>

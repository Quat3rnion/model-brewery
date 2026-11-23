<script module lang="ts">
  export interface GgufDraft {
    choice: string;
    assets: string;
    imatrix: string;
    imatrixLabel?: string;
    threads: number | null;
  }
</script>

<script lang="ts">
  // GGUF output conversion: source, preset or saved profile, and importance matrix.
  import { brewery } from '../lib/store.svelte';
  import { capability, hostName, modelLabel } from '../lib/format';
  import { profileStore } from '../lib/profiles.svelte';
  import { conversionRunHref, STAGES, type ConversionBody } from '../lib/conversion';
  import { hostWorkflowEligibility } from '../lib/eligibility';
  import type { Blocker } from '../lib/blockers';
  import { validSha } from '../lib/ggufSource';
  import { roleLabel } from '../lib/components';
  import { conversions, ConversionStarter } from '../lib/conversions.svelte';
  import { buildGgufConversion, imatrixRequired, IMATRIX_KIND, forkSelected, profileMatches, FORK_NOTICE, type GgufCatalog } from '../lib/gguf';
  import { shortRev } from '../lib/profiles';
  import ModelSelect from './ModelSelect.svelte';
  import ComponentPicker from './ComponentPicker.svelte';
  import Blockers from './Blockers.svelte';
  import Submit from './Submit.svelte';
  import SelectionReview from './SelectionReview.svelte';

  let { catalog, source = $bindable(''), draft = $bindable() }: {
    catalog: GgufCatalog; source?: string; draft: GgufDraft;
  } = $props();
  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);

  $effect(() => {
    const p = profileStore.pendingUse;
    if (p?.startsWith('gguf:')) {
      draft.choice = `rev:${p.slice(5)}`;
      profileStore.pendingUse = null;
    }
  });

  const src = $derived(models.find((m) => m.id === source));
  const standalone = $derived(src?.kind==='safetensors');
  const assetChoices = $derived(models.filter(m=>m.kind==='assets' && m.host===src?.host && m.available!==false));
  const sameHostMatrix = $derived(models.filter(m=>m.kind===IMATRIX_KIND && m.host===src?.host && m.available!==false));
  const missingMatrix = $derived(draft.imatrix && !sameHostMatrix.some(m => m.id === draft.imatrix)
    ? [{ id: draft.imatrix, label: draft.imatrixLabel || draft.imatrix }] : []);
  $effect(() => {
    if (!brewery.state) return;
    if (draft.assets && !assetChoices.some(m => m.id === draft.assets)) draft.assets = '';
  });
  function selectMatrix(id: string) {
    draft.imatrix = id;
    draft.imatrixLabel = id ? modelLabel(sameHostMatrix.find(m => m.id === id)) : '';
  }
  function selectSource(id: string) {
    if (id === source) return;
    source = id;
    draft.assets = '';
    selectMatrix('');
  }
  const rev = $derived(draft.choice.startsWith('rev:') ? profileStore.ggufSaved.find((p) => p.revision === draft.choice.slice(4)) : undefined);
  const preset = $derived(draft.choice.startsWith('preset:') ? draft.choice.slice(7) : '');
  const fork = $derived(forkSelected(catalog, rev?.preset ?? preset, rev?.overrides ?? {}));
  const needsImatrix = $derived(rev ? imatrixRequired(catalog, rev.preset, rev.overrides) : !!catalog.presets.find((p) => p.id === preset)?.requiresImatrix);
  const built = $derived(
    buildGgufConversion(catalog, { source, sourceKind:src?.kind, assets: draft.assets, preset,
      revision: rev ? { revision: rev.revision, needsImatrix } : null, imatrix: draft.imatrix, threads: draft.threads, host: src?.host ?? '' }),
  );
  const target = $derived(hosts.find((h) => h.id === src?.host));
  const hostBlockers = $derived.by((): Blocker[] => {
    if (!src) return [];
    const kind = src.kind;
    const host = hostName(hosts, src.host);
    if (!target) return [{ text: 'The controller did not report this host.', host }];
    if (kind !== 'hf' && kind !== 'gguf' && kind !== 'safetensors') return [];
    const decision = hostWorkflowEligibility(brewery.state?.eligibility, src.host, 'gguf_output', { kind, rocmfpx: fork });
    if (decision) return decision.reasons.map((r) => ({ host, text: r.text, fixAnchor: r.fix_anchor }));
    if (target.scheduling_paused) return [{ host, text: 'Resume job scheduling', fixAnchor: `#/settings/infrastructure?host=${encodeURIComponent(src.host)}&section=policy` }];
    const required = ['gguf_output', ...STAGES, ...(fork ? ['gguf_rocmfpx_output'] : []), ...(kind === 'gguf' ? [] : [`gguf_${kind}_output`])];
    return required.flatMap((key) => {
      const cap = capability(target, key);
      const reason = target.capability_reasons?.[key];
      return cap.allowed ? [] : [{ host, text: reason?.text ?? cap.reason ?? 'GGUF output is not available.', fixAnchor: reason?.fix_anchor }];
    });
  });
  const blockers = $derived.by(() => {
    const out = [...built.problems];
    if (rev && !profileMatches(catalog,rev)) out.push('This profile does not match the current encoder catalog.');
    if (source && (!src || src.available===false || !['hf','gguf','safetensors'].includes(src.kind))) out.push('Choose an available HF checkpoint, standalone SafeTensors file or GGUF input.');
    if (hostBlockers.length) out.push('Select a source on a host that can write GGUF.');
    if (standalone && draft.assets && !assetChoices.some(m=>m.id===draft.assets)) out.push('Choose available assets on the source host.');
    if (missingMatrix.length) out.push('Review the importance matrix selection.');
    if (draft.choice.startsWith('rev:') && !rev) out.push('The chosen GGUF profile revision is no longer listed.');
    if (conversions.status === 'unavailable') out.push('This controller does not provide managed conversions.');
    return out;
  });
  const starter = new ConversionStarter('convert.gguf');
  let runLink: HTMLAnchorElement | undefined = $state();
  let typeSelect: HTMLSelectElement | undefined = $state();
  function start(event: MouseEvent) {
    if (!built.body || blockers.length) return;
    void starter.start(built.body as unknown as ConversionBody, event, () => runLink);
  }
</script>

<div class="stack">
  <div class="grid-2">
    <ModelSelect bind:value={() => source, selectSource} kinds={['hf','gguf','safetensors']} label="Source checkpoint" hint="HF checkpoint, standalone SafeTensors file, or GGUF. Worker support depends on the input format." id="gg-source" />
    <div class="field">
      <label for="gg-choice">GGUF type</label>
      <select id="gg-choice" class="input" bind:this={typeSelect} bind:value={draft.choice} aria-describedby="gg-choice-h" aria-required="true">
        <option value="" disabled>Choose a preset or profile</option>
        <optgroup label="llama.cpp presets">
          {#each catalog.presets as p (p.id)}<option value="preset:{p.id}">{p.label ?? p.id}{p.fork ? ' · ROCmFPX' : ''}{p.requiresImatrix ? ' (needs imatrix)' : ''}</option>{/each}
        </optgroup>
        {#if profileStore.ggufSaved.length}
          <optgroup label="Saved GGUF profiles">
            {#each profileStore.ggufSaved as p (p.revision)}<option disabled={!profileMatches(catalog,p)} value="rev:{p.revision}">{p.name} · {shortRev(p.revision)} ({p.preset} + {Object.keys(p.overrides).length}){profileMatches(catalog,p) ? '' : ' · incompatible revision'}</option>{/each}
          </optgroup>
        {/if}
      </select>
      <span class="hint" id="gg-choice-h">
        {#if rev}Preset {rev.preset} with {Object.keys(rev.overrides).length} tensor overrides.{:else}A llama.cpp quantization type for every tensor.{/if}
        <a href="#/profiles?format=gguf">{profileStore.ggufSaved.length ? 'Edit GGUF profiles' : 'Create a GGUF profile'}</a>
      </span>
    </div>
  </div>
  {#if src && !validSha(src.sha256)}<p class="hint">Source checkpoint unverified.</p>{/if}
  {#if fork}<p class="note warn small"><abbr title="GGUF encoder fork with ROCm support">ROCmFPX</abbr>: {FORK_NOTICE}</p>{/if}
  <dl><dt>Run on</dt><dd>{src ? hostName(hosts, src.host) : 'Host of the source'}</dd></dl>
  {#if hostBlockers.length}
    <Blockers blockers={hostBlockers} />
    <p class="hint">Register or copy the source to a host that can write GGUF. <a href="#/library">Open Library</a> or <a href="#/transfers">Open Transfers</a>.</p>
  {/if}
  {#if standalone && src}
    <div class="field">
      <label for="gg-assets">{roleLabel('assets')} (required)</label>
      <select bind:value={draft.assets} id="gg-assets" class="input" disabled={!assetChoices.length} aria-describedby="gg-assets-hint">
        <option value="">Select matching assets</option>
        {#each assetChoices as m (m.id)}<option value={m.id}>{modelLabel(m)} · {m.path}</option>{/each}
      </select>
      <span class="hint" id="gg-assets-hint">{#if !assetChoices.length}No available assets on {hostName(hosts,src.host)}. <a href="#/library">Register the matching assets directory in Library</a>.{:else}Choose the registered assets directory for these weights on {hostName(hosts,src.host)}.{/if}</span>
    </div>
    <p class="xs muted">The weights alone do not identify a complete model. The exporter checks the weights, config, tokenizer and architecture; worker support does not guarantee that this model can be exported.</p>
  {:else if src}<ComponentPicker source={src} idPrefix="gg-comp" gguf />{/if}
  {#if src?.kind==='hf' || standalone}<p class="xs muted">The GGUF exporter does not support the Flash-Next layout. For a Flash-Next HF checkpoint, choose Managed HGN instead.</p>{/if}
  <div class="grid-2">
    {#if needsImatrix || draft.imatrix}
      <ModelSelect bind:value={() => draft.imatrix, selectMatrix} kinds={[IMATRIX_KIND]} exclude={models.filter(m=>!sameHostMatrix.some(x=>x.id===m.id)).map(m=>m.id)} label="Importance matrix (imatrix)" optional={!needsImatrix} hint="Register a matching importance matrix on the source host." id="gg-imatrix" />
  {/if}
    <div class="field">
      <label for="gg-threads">Threads <span class="muted">(optional)</span></label>
      <input id="gg-threads" class="input num" type="number" min="1" max="64" step="1" bind:value={draft.threads} inputmode="numeric" />
      <span class="hint">1 to 64.</span>
    </div>
  </div>
  <SelectionReview missing={missingMatrix} acknowledge={(shown) => {
    if (shown.some(item => item.id === draft.imatrix)) selectMatrix('');
  }} focusRemaining={() => typeSelect} />
  <p class="xs muted">The output is a single .gguf file on the worker, registered in the Library when the workflow finishes. The workflow runs inspect, plan, convert and verify like an HGN conversion.</p>
  <Submit {blockers} busy={starter.busy} onclick={start}>Start GGUF conversion</Submit>
  <div aria-live="polite" class="xs">
    {#if starter.error}<p class="note bad" role="alert">{starter.error}</p>
    {:else if starter.last}
      <p class="muted">{starter.replayed ? 'This request was already queued as job' : 'Started conversion'} <a bind:this={runLink} href={conversionRunHref(starter.last.id, 'gguf')}>{starter.last.id}</a>. {starter.replayed ? 'No second conversion was started.' : ''}</p>
    {/if}
  </div>
</div>

<style>
  dl { margin: 0; }
  dt { font-weight: 600; }
  dd { margin: 0; }
</style>

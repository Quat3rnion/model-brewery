<script lang="ts">
  // GGUF output conversion: source, preset or saved GGUF profile, imatrix when
  // required, host. One POST /api/conversions with format 'gguf'.
  import { brewery } from '../lib/store.svelte';
  import { capability, hostName, modelLabel } from '../lib/format';
  import { profileStore } from '../lib/profiles.svelte';
  import { conversions, ConversionStarter } from '../lib/conversions.svelte';
  import { buildGgufConversion, imatrixRequired, IMATRIX_KIND, forkSelected, profileMatches, FORK_NOTICE, type GgufCatalog } from '../lib/gguf';
  import { shortRev } from '../lib/profiles';
  import ModelSelect from './ModelSelect.svelte';
  import ComponentPicker from './ComponentPicker.svelte';
  import HostSelect from './HostSelect.svelte';
  import Submit from './Submit.svelte';
  import type { ConversionBody } from '../lib/conversion';

  let { catalog, source=$bindable('') }: { catalog: GgufCatalog; source?:string } = $props();
  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);

  let choice = $state(''); // 'preset:<id>' | 'rev:<hex>'
  let imatrix = $state('');
  let assets = $state('');
  let host = $state('');
  let threads = $state<number | null>(null);

  $effect(() => {
    const p = profileStore.pendingUse;
    if (p?.startsWith('gguf:')) {
      choice = `rev:${p.slice(5)}`;
      profileStore.pendingUse = null;
    }
  });

  const src = $derived(models.find((m) => m.id === source));
  const standalone = $derived(src?.kind==='safetensors');
  const assetChoices = $derived(models.filter(m=>m.kind==='assets' && m.host===src?.host && m.available!==false));
  const sameHostMatrix = $derived(models.filter(m=>m.kind===IMATRIX_KIND && m.host===src?.host && m.available!==false));
  $effect(()=>{void source; assets=''; imatrix='';});
  const rev = $derived(choice.startsWith('rev:') ? profileStore.ggufSaved.find((p) => p.revision === choice.slice(4)) : undefined);
  const preset = $derived(choice.startsWith('preset:') ? choice.slice(7) : '');
  const fork = $derived(forkSelected(catalog, rev?.preset ?? preset, rev?.overrides ?? {}));
  const needsImatrix = $derived(rev ? imatrixRequired(catalog, rev.preset, rev.overrides) : !!catalog.presets.find((p) => p.id === preset)?.requiresImatrix);
  const built = $derived(
    buildGgufConversion(catalog, { source, sourceKind:src?.kind, assets, preset, revision: rev ? { revision: rev.revision, needsImatrix } : null, imatrix: needsImatrix ? imatrix : '', threads, host }),
  );
  const target = $derived(hosts.find((h) => h.id === (host || src?.host)));
  const blockers = $derived.by(() => {
    const out = [...built.problems];
    if (rev && !profileMatches(catalog,rev)) out.push('This profile does not match the current encoder catalog.');
    if (fork && target?.capabilities?.gguf_rocmfpx_output!==true) out.push(target?.reasons?.gguf_rocmfpx_output ?? 'Choose a worker that reports ROCmFPX output support.');
    if (source && (!src || src.available===false || !['hf','gguf','safetensors'].includes(src.kind))) out.push('Choose an available HF checkpoint, standalone SafeTensors file or GGUF input.');
    if (src && !target) out.push('The controller did not report a worker for this source.');
    if (src && host && host!==src.host) out.push('Run on the host of the registered inputs.');
    if (standalone && assets && !assetChoices.some(m=>m.id===assets)) out.push('Choose available assets on the source host.');
    if (imatrix && needsImatrix && !sameHostMatrix.some(m=>m.id===imatrix)) out.push('Choose an available importance matrix on the source host.');
    if (choice.startsWith('rev:') && !rev) out.push('The chosen GGUF profile revision is no longer listed.');
    if (target) {
      const g = capability(target, 'gguf_output');
      if (!g.allowed) out.push(g.reason ?? `${target.name ?? target.id} cannot write GGUF.`);
      else if (standalone && !capability(target,'gguf_safetensors_output').allowed) out.push(target.reasons?.gguf_safetensors_output ?? `${target.name ?? target.id} does not report standalone SafeTensors to GGUF support.`);
      else if (src?.kind === 'hf') {
        const h = capability(target, 'gguf_hf_output');
        if (!h.allowed) out.push(h.reason ?? `${target.name ?? target.id} cannot convert a Hugging Face checkpoint to GGUF.`);
      }
    }
    return out;
  });
  const starter = new ConversionStarter();
  async function start() {
    if (!built.body || starter.busy || blockers.length) return;
    await starter.start(built.body as unknown as ConversionBody);
  }
</script>

<div class="stack">
  <div class="grid-2">
    <ModelSelect bind:value={source} kinds={['hf','gguf','safetensors']} label="Source checkpoint" hint="HF checkpoint, standalone SafeTensors file, or GGUF. Worker support depends on the input format." id="gg-source" />
    <div class="field">
      <label for="gg-choice">GGUF type</label>
      <select id="gg-choice" class="input" bind:value={choice} aria-describedby="gg-choice-h" aria-required="true">
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
  {#if fork}<p class="note warn small">{FORK_NOTICE}</p>{/if}
  {#if standalone && src}
    <div class="field">
      <label for="gg-assets">Config and tokenizer assets (required)</label>
      <select bind:value={assets} id="gg-assets" class="input" disabled={!assetChoices.length} aria-describedby="gg-assets-hint">
        <option value="">Select matching assets</option>
        {#each assetChoices as m (m.id)}<option value={m.id}>{modelLabel(m)} · {m.path}</option>{/each}
      </select>
      <span class="hint" id="gg-assets-hint">{#if !assetChoices.length}No available assets on {hostName(hosts,src.host)}. <a href="#/library">Register the matching assets directory in Library</a>.{:else}Choose the registered assets directory for these weights on {hostName(hosts,src.host)}.{/if}</span>
    </div>
    <p class="xs muted">The weights alone do not identify a complete model. The exporter checks the weights, config, tokenizer and architecture; worker support does not guarantee that this model can be exported.</p>
  {:else if src}<ComponentPicker source={src} idPrefix="gg-comp" gguf />{/if}
  {#if src?.kind==='hf' || standalone}<p class="xs muted">The bounded GGUF exporter does not support the Flash-Next layout. For a Flash-Next HF checkpoint, choose HGN bundle output instead.</p>{/if}
  <div class="grid-2">
    {#if needsImatrix}
      <ModelSelect bind:value={imatrix} kinds={[IMATRIX_KIND]} exclude={models.filter(m=>!sameHostMatrix.some(x=>x.id===m.id)).map(m=>m.id)} label="Importance matrix (imatrix)" hint="Register a matching importance matrix on the source host." id="gg-imatrix" />
    {/if}
    <HostSelect bind:value={host} kind="gguf_output" label="Run on" optional optionalLabel="Host of the source" id="gg-host" />
    <div class="field">
      <label for="gg-threads">Threads <span class="muted">(optional)</span></label>
      <input id="gg-threads" class="input num" type="number" min="1" max="64" step="1" bind:value={threads} inputmode="numeric" />
    </div>
  </div>
  <p class="xs muted">The output is a single .gguf file on the worker, registered in the Library when the workflow finishes. The workflow runs inspect, plan, convert and verify like an HGN conversion.</p>
  <Submit {blockers} busy={starter.busy} onclick={start}>Start GGUF conversion</Submit>
  <div aria-live="polite" class="xs">
    {#if starter.error}<p class="note bad" role="alert">{starter.error} Pressing Start again retries the same request.</p>
    {:else if starter.last && starter.replayed}<p class="muted">This exact request already started conversion <span class="mono">{starter.last.id}</span>; nothing new was started.</p>
    {:else if starter.last}<p class="muted">Started conversion <span class="mono">{starter.last.id}</span>. It is listed under Conversions.</p>{/if}
  </div>
  {#if conversions.status === 'unavailable'}<p class="note warn">This controller has no conversion workflow.</p>{/if}
</div>

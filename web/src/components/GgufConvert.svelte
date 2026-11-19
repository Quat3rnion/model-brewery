<script lang="ts">
  // GGUF output conversion: source, preset or saved GGUF profile, imatrix when
  // required, host. One POST /api/conversions with format 'gguf'.
  import { brewery } from '../lib/store.svelte';
  import { capability, CONVERT_SOURCE_KINDS } from '../lib/format';
  import { profileStore } from '../lib/profiles.svelte';
  import { conversions, ConversionStarter } from '../lib/conversions.svelte';
  import { buildGgufConversion, imatrixRequired, IMATRIX_KIND, type GgufCatalog } from '../lib/gguf';
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
  const rev = $derived(choice.startsWith('rev:') ? profileStore.ggufSaved.find((p) => p.revision === choice.slice(4)) : undefined);
  const preset = $derived(choice.startsWith('preset:') ? choice.slice(7) : '');
  const needsImatrix = $derived(rev ? imatrixRequired(catalog, rev.preset, rev.overrides) : !!catalog.presets.find((p) => p.id === preset)?.requiresImatrix);
  const built = $derived(
    buildGgufConversion(catalog, { source, preset, revision: rev ? { revision: rev.revision, needsImatrix } : null, imatrix: needsImatrix ? imatrix : imatrix || '', threads, host }),
  );
  const target = $derived(hosts.find((h) => h.id === (host || src?.host)));
  const blockers = $derived.by(() => {
    const out = [...built.problems];
    if (src && src.kind!=='hf' && src.kind!=='gguf') out.push('GGUF output requires a Hugging Face checkpoint or GGUF input.');
    if (choice.startsWith('rev:') && !rev) out.push('The chosen GGUF profile revision is no longer listed.');
    if (target) {
      const g = capability(target, 'gguf_output');
      if (!g.allowed) out.push(g.reason ?? `${target.name ?? target.id} cannot write GGUF.`);
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
    <ModelSelect bind:value={source} kinds={CONVERT_SOURCE_KINDS} label="Source checkpoint" hint="BF16 checkpoint or GGUF. HF sources need a worker with GGUF export." id="gg-source" />
    <div class="field">
      <label for="gg-choice">GGUF type</label>
      <select id="gg-choice" class="input" bind:value={choice} aria-describedby="gg-choice-h" aria-required="true">
        <option value="" disabled>Choose a preset or profile</option>
        <optgroup label="llama.cpp presets">
          {#each catalog.presets as p (p.id)}<option value="preset:{p.id}">{p.label ?? p.id}{p.requiresImatrix ? ' (needs imatrix)' : ''}</option>{/each}
        </optgroup>
        {#if profileStore.ggufSaved.length}
          <optgroup label="Saved GGUF profiles">
            {#each profileStore.ggufSaved as p (p.revision)}<option value="rev:{p.revision}">{p.name} · {shortRev(p.revision)} ({p.preset} + {Object.keys(p.overrides).length})</option>{/each}
          </optgroup>
        {/if}
      </select>
      <span class="hint" id="gg-choice-h">
        {#if rev}Preset {rev.preset} with {Object.keys(rev.overrides).length} tensor overrides.{:else}A llama.cpp quantization type for every tensor.{/if}
        <a href="#/profiles?format=gguf">{profileStore.ggufSaved.length ? 'Edit GGUF profiles' : 'Create a GGUF profile'}</a>
      </span>
    </div>
  </div>
  {#if src}<ComponentPicker source={src} idPrefix="gg-comp" gguf />{/if}
  <div class="grid-2">
    {#if needsImatrix}
      <ModelSelect bind:value={imatrix} kinds={[IMATRIX_KIND]} label="Importance matrix (imatrix)" hint="Required by the IQ types in this choice. Register the .imatrix file in Library; it must match the source model." id="gg-imatrix" />
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

<script lang="ts">
  import Advanced from '../components/Advanced.svelte';
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { CONVERT_SOURCE_KINDS, capability, isSafeSource } from '../lib/format';
  import { CONVERT_ENUMS, VERIFY_OUTPUT_KINDS, buildConvertJob, poolHosts, type ConvertForm, type ConvertStep } from '../lib/convert';
  import HostSelect from '../components/HostSelect.svelte';
  import ModelSelect from '../components/ModelSelect.svelte';
  import Submit from '../components/Submit.svelte';
  import SubmitStatus from '../components/SubmitStatus.svelte';
  import RecentJobs from '../components/RecentJobs.svelte';
  import ConversionRun from '../components/ConversionRun.svelte';
  import ComponentPicker from '../components/ComponentPicker.svelte';
  import { componentBlockers } from '../lib/components';
  import { profileStore } from '../lib/profiles.svelte';
  import { BASE_LABEL, shortRev } from '../lib/profiles';
  import { REV_PREFIX } from '../lib/conversion';
  import GgufConvert from '../components/GgufConvert.svelte';
  import LayerRecipe from '../components/LayerRecipe.svelte';
  import Hy4Plan from '../components/Hy4Plan.svelte';
  import { conversions, ConversionStarter } from '../lib/conversions.svelte';
  import { buildConversion, isActiveConversion } from '../lib/conversion';

  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);

  let step = $state<ConvertStep>('inspect');
  let form = $state<ConvertForm>({
    source: '',
    bundle: '',
    extras: { assets: '', ngram: '', vision: '', mtp: '' },
    host: '',
    anyHost: false,
    profile: '',
    ht_backend: '',
    i4r_backend: '',
    q4cp_backend: '',
    decode_backend: '',
    seed: null,
    threads: null,
    ht_beam: null,
  });

  const STEPS: { id: ConvertStep; label: string; text: string }[] = [
    { id: 'inspect', label: 'Inspect', text: 'Read the checkpoint and report what the converter would write. Writes nothing.' },
    { id: 'plan', label: 'Plan', text: 'Plan the conversion and check disk space on the host. Writes nothing.' },
    { id: 'convert', label: 'Convert', text: 'Write a new bundle to a fresh directory on the host. The source is not changed.' },
    { id: 'verify', label: 'Verify', text: 'Check a complete bundle against its original source. File verification only; it does not prove GPU use or quality.' },
  ];

  const built = $derived(buildConvertJob(step, form, models, hosts));
  const blockers = $derived.by(() => {
    const out = [...built.problems];
    if (!(step === 'convert' && form.anyHost)) {
      const target = form.host || models.find((m) => m.id === form.source)?.host;
      if (target) {
        const c = capability(hosts.find((h) => h.id === target), step);
        if (!c.allowed && c.reason) out.push(c.reason);
      }
    }
    return out;
  });
  const eligible = $derived(poolHosts(hosts));

  // ---- Full workflow (default): one durable controller record ----
  let wfHost = $state('');
  let wfAny = $state(false);
  const wfForm = $derived<ConvertForm>({ ...form, host: wfHost, anyHost: wfAny });
  const wfBuilt = $derived(buildConversion(wfForm, models));
  const wfSource = $derived(models.find((m) => m.id === form.source));
  let outFormat = $state<'hgn' | 'gguf'>('hgn');
  let useLayerRecipe = $state(false);
  const recipeStarter = new ConversionStarter();
  const recipeSource = $derived(wfSource && isSafeSource(wfSource.kind) ? wfSource : undefined);
  const recipeKey = $derived(recipeSource && JSON.stringify([
    recipeSource.id, recipeSource.kind, recipeSource.host, recipeSource.path, recipeSource.sha256, recipeSource.bundle_sha256,
  ]));
  $effect(() => {
    if (profileStore.pendingUse?.startsWith('gguf:')) outFormat = 'gguf';
  });
  // Saved custom profiles (read-only GET). A revision is chosen as 'rev:<hex>'.
  $effect(() => {
    void profileStore.load();
  });
  const savedProfiles = $derived(profileStore.saved.filter((p) => p.compatible));
  const chosenRev = $derived(form.profile.startsWith(REV_PREFIX) ? profileStore.saved.find((p) => p.revision === form.profile.slice(REV_PREFIX.length)) : undefined);
  $effect(() => {
    // "Use in Convert" from the Profiles page.
    const rev = profileStore.pendingUse;
    if (rev && !rev.startsWith('gguf:') && profileStore.saved.some((p) => p.revision === rev)) {
      form.profile = `${REV_PREFIX}${rev}`;
      profileStore.pendingUse = null;
    }
  });
  const wfBlockers = $derived.by(() => {
    const out = [...wfBuilt.problems];
    if (chosenRev) {
      const target = wfAny ? null : hosts.find((h) => h.id === (wfHost || wfSource?.host));
      if (target) {
        const c = capability(target, 'custom_profiles');
        if (!c.allowed) out.push(c.reason ?? `${target.name ?? target.id} does not accept custom profiles.`);
      } else if (wfAny && eligible.length && !eligible.some((h) => capability(h, 'custom_profiles').allowed)) out.push('No pool worker accepts custom profiles.');
      const cat = profileStore.catalog;
      if (cat && wfSource?.sha256 && cat.sourceSha256 && wfSource.sha256 !== cat.sourceSha256) out.push('This profile was made for a different source checkpoint.');
    }
    if (!wfAny) out.push(...componentBlockers(wfSource, form.extras, hosts.find((h) => h.id === (wfHost || wfSource?.host))));
    else if (eligible.length && eligible.every((h) => componentBlockers(wfSource, form.extras, h).length)) out.push('No pool worker accepts component replacements. Remove them or pick a host.');
    if (!wfAny) {
      const target = wfHost || models.find((m) => m.id === form.source)?.host;
      if (target) {
        const c = capability(hosts.find((h) => h.id === target), 'convert');
        if (!c.allowed && c.reason) out.push(c.reason);
      }
    }
    return out;
  });
  const starter = new ConversionStarter();
  async function startConversion() {
    // One POST per click. The body is built only here, never from an effect.
    if (useLayerRecipe || !wfBuilt.body || starter.busy) return;
    const c = await starter.start(wfBuilt.body);
    if (c) queueMicrotask(() => document.getElementById(`run-${c.id}`)?.scrollIntoView({ block: 'nearest' }));
  }
  $effect(() => conversions.watch());
  const runs = $derived(conversions.list);
  const activeRuns = $derived(runs.filter(isActiveConversion));
  let showOlder = $state(false);
  const finishedRuns = $derived(runs.filter((c) => !isActiveConversion(c)));
  const optsChangedWf = $derived(
    [
      ...(['ht_backend', 'i4r_backend', 'q4cp_backend'] as const).filter((k) => form[k]).map((k) => `${k.replace('_backend', '').toUpperCase()} ${form[k]}`),
      form.seed !== null ? `seed ${form.seed}` : '',
      form.threads !== null ? `${form.threads} threads` : '',
      form.ht_beam !== null ? `beam ${form.ht_beam}` : '',
      form.decode_backend ? `verify decode ${form.decode_backend}` : '',
    ].filter(Boolean),
  );

  const s = new Submitter();
  function submit() {
    if (!useLayerRecipe && built.job) s.submit(built.job);
  }
  const usesBackends = $derived(step === 'inspect' || step === 'plan' || step === 'convert');
  const usesPlanOpts = $derived(step === 'plan' || step === 'convert');
  const optsChanged = $derived(
    [
      ...(usesBackends ? (['ht_backend', 'i4r_backend', 'q4cp_backend'] as const).filter((k) => form[k]).map((k) => `${k.replace('_backend', '').toUpperCase()} ${form[k]}`) : []),
      ...(usesPlanOpts ? [form.seed !== null ? `seed ${form.seed}` : '', form.threads !== null ? `${form.threads} threads` : '', form.ht_beam !== null ? `beam ${form.ht_beam}` : ''] : []),
      step === 'verify' && form.decode_backend ? `decode ${form.decode_backend}` : '',
    ].filter(Boolean),
  );
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Convert</h1>
    <p class="muted small">Convert a checkpoint to HGN or GGUF. Managed conversions run inspect, plan, convert and verify. To set the output type per layer for SafeTensors, select Layer recipe.</p>
  </div>

  <section class="panel" aria-labelledby="wf-h">
    <header>
      <h2 id="wf-h">Convert a checkpoint</h2>
      <div class="fmt" role="radiogroup" aria-label="Output format">
        <label class:on={outFormat === 'hgn'}><input type="radio" name="convert-format" bind:group={outFormat} value="hgn" /> {useLayerRecipe ? 'HGN archive' : 'HGN bundle'}</label>
        <label class:on={outFormat === 'gguf'}><input type="radio" name="convert-format" bind:group={outFormat} value="gguf" /> GGUF file</label>
      </div>
    </header>
    <div class="body">
      <label class="inline"><input type="checkbox" bind:checked={useLayerRecipe} /> Layer recipe</label>
    </div>
    {#if useLayerRecipe}
      <div class="body stack">
        <ModelSelect bind:value={form.source} kinds={['hf', 'safetensors']} label="Source checkpoint" hint="Choose a SafeTensors file or an indexed HF directory for the recipe." id="recipe-source" />
        {#if recipeSource}
          {#key recipeKey}
            <LayerRecipe source={recipeSource} format={outFormat} starter={recipeStarter} />
          {/key}
        {:else}
          <Submit blockers={['Select a SafeTensors source.']} onclick={() => {}}>Run layer recipe</Submit>
        {/if}
      </div>
    {:else if outFormat === 'gguf'}
    <div class="body">
      {#if profileStore.gguf}<GgufConvert catalog={profileStore.gguf} bind:source={form.source} />
      {:else if profileStore.ggufStatus === 'unavailable'}<p class="note warn">This controller does not offer GGUF output.</p>
      {:else if profileStore.status === 'loading' || profileStore.status === 'idle'}<p class="muted">Loading GGUF types…</p>
      {:else}<p class="note bad" role="alert">The GGUF catalog could not be loaded.</p>{/if}
    </div>
    {:else}
    <div class="body stack">
      <div class="grid-2">
        <ModelSelect bind:value={form.source} kinds={CONVERT_SOURCE_KINDS} label="Source checkpoint" hint="Hugging Face BF16 directory, GGUF or HGN." id="wf-source" />
        <div class="field">
          <label for="wf-profile">Output profile</label>
          <select id="wf-profile" class="input" bind:value={form.profile} aria-describedby="wf-profile-hint" aria-required="true">
            <option value="" disabled>Choose a profile</option>
            <optgroup label="Built-in">
              {#each CONVERT_ENUMS.profile as p (p)}<option value={p}>{p === 'v2' ? 'HGN v2' : p === 'ht43' ? 'HT43' : p}</option>{/each}
            </optgroup>
            {#if savedProfiles.length}
              <optgroup label="Saved custom profiles">
                {#each savedProfiles as p (p.revision)}<option value="{REV_PREFIX}{p.revision}">{p.name} · {shortRev(p.revision)} (from {BASE_LABEL[p.base]})</option>{/each}
              </optgroup>
            {/if}
          </select>
          <span class="hint" id="wf-profile-hint">
            {#if chosenRev}Custom revision <span class="mono" title={chosenRev.revision}>{shortRev(chosenRev.revision)}</span>: {Object.keys(chosenRev.overrides).length} tensors changed from {BASE_LABEL[chosenRev.base]}.{:else}Format of the bundle. Required, so the result is never a silent default.{/if}
            <a href="#/profiles">{savedProfiles.length ? 'Edit profiles' : 'Create a custom profile'}</a>
          </span>
        </div>
      </div>
      <div class="grid-2">
        <div class="field">
          {#if wfAny}
            <span class="flabel">Run on</span>
            <p class="hint">
              The controller assigns the conversion to an idle pool worker and keeps all four stages on it.
              {eligible.length ? `Pool: ${eligible.map((h) => h.name ?? h.id).join(', ')}.` : 'No pool is configured, so it runs on the source host.'}
            </p>
          {:else}
            <HostSelect bind:value={wfHost} kind="convert" label="Run on" optional optionalLabel="Host of the source" id="wf-host" />
          {/if}
          <label class="inline"><input type="checkbox" bind:checked={wfAny} /> Any eligible pool worker</label>
        </div>
      </div>

      {#if wfSource}<ComponentPicker bind:value={form.extras} source={wfSource} idPrefix="wf-comp" />{/if}
      <Advanced changed={optsChangedWf}>
        <div class="grid-3">
          {#each ['ht_backend', 'i4r_backend', 'q4cp_backend'] as const as key (key)}
            <div class="field">
              <label for="wf-{key}">{key.replace('_backend', '').toUpperCase()} encoder backend</label>
              <select id="wf-{key}" class="input" bind:value={form[key]}>
                <option value="">Converter default</option>
                {#each CONVERT_ENUMS.backend as b (b)}<option value={b}>{b}</option>{/each}
              </select>
            </div>
          {/each}
          <div class="field">
            <label for="wf-seed">Seed <span class="muted">(optional)</span></label>
            <input id="wf-seed" class="input num" type="number" min="0" step="1" bind:value={form.seed} inputmode="numeric" />
          </div>
          <div class="field">
            <label for="wf-threads">Threads <span class="muted">(optional)</span></label>
            <input id="wf-threads" class="input num" type="number" min="1" max="64" step="1" bind:value={form.threads} inputmode="numeric" />
          </div>
          <div class="field">
            <label for="wf-beam">HT beam <span class="muted">(optional)</span></label>
            <input id="wf-beam" class="input num" type="number" min="1" max="256" step="1" bind:value={form.ht_beam} inputmode="numeric" />
          </div>
          <div class="field">
            <label for="wf-dec">Verify decode backend</label>
            <select id="wf-dec" class="input" bind:value={form.decode_backend}>
              <option value="">Converter default</option>
              {#each CONVERT_ENUMS.backend as b (b)}<option value={b}>{b}</option>{/each}
            </select>
          </div>
        </div>
        <p class="xs muted">The receipt records the backends that actually ran. A requested backend is not evidence of GPU use.</p>
      </Advanced>

      <Submit blockers={conversions.status === 'unavailable' ? ['This controller has no conversion workflow. Use the individual steps below.'] : wfBlockers} busy={starter.busy} onclick={startConversion}>Start conversion</Submit>
      <div aria-live="polite" class="xs">
        {#if starter.error}<p class="note bad" role="alert">{starter.error} Pressing Start again retries the same request; it will not start a second conversion.</p>
        {:else if starter.last && starter.replayed}<p class="muted">This exact request already started conversion <span class="mono">{starter.last.id}</span>; no second conversion was started. Change the source, profile or options to start another.</p>
        {:else if starter.last}<p class="muted">Started conversion <span class="mono">{starter.last.id}</span>. It continues on the controller if you leave this page.</p>{/if}
      </div>
    </div>
    {/if}
  </section>

  <Hy4Plan bind:sourceId={form.source} />

  <section class="panel" id="conversion-runs" aria-labelledby="runs-h">
    <header><h2 id="runs-h">Conversions <span class="muted num small">({runs.length})</span></h2></header>
    {#if !conversions.loaded}
      <p class="empty">Loading…</p>
    {:else if conversions.status === 'unavailable'}
      <p class="empty">This controller does not provide the conversion workflow.</p>
    {:else if conversions.status === 'error' && runs.length === 0}
      <p class="note bad" role="alert" style="margin:12px 16px">Conversions could not be loaded: {conversions.error}</p>
    {:else if runs.length === 0}
      <p class="empty">No conversions yet.</p>
    {:else}
      {#if conversions.status === 'error'}<p class="xs muted" style="margin:8px 16px 0">Showing the last loaded list ({conversions.error}).</p>{/if}
      <div>
        {#each activeRuns as c (c.id)}<div id="run-{c.id}"><ConversionRun {c} /></div>{/each}
        {#each showOlder ? finishedRuns : finishedRuns.slice(0, 5) as c (c.id)}<div id="run-{c.id}"><ConversionRun {c} /></div>{/each}
      </div>
      {#if finishedRuns.length > 5}
        <div class="more"><button type="button" class="btn small ghost" aria-expanded={showOlder} onclick={() => (showOlder = !showOlder)}>{showOlder ? 'Show fewer' : `Show ${finishedRuns.length - 5} older`}</button></div>
      {/if}
    {/if}
  </section>

  {#if !useLayerRecipe}
  <details class="disclosure manual">
    <summary><span class="t">Individual steps (diagnostics)</span><span class="s xs">Run inspect, plan, convert or verify as a single job</span></summary>
    <div class="stack manual-body">

  <div class="steps" role="tablist" aria-label="Converter step">
    {#each STEPS as st, i (st.id)}
      <button
        type="button"
        role="tab"
        id="tab-{st.id}"
        aria-selected={step === st.id}
        aria-controls="conv-panel"
        tabindex={step === st.id ? 0 : -1}
        class="step"
        onclick={() => (step = st.id)}
        onkeydown={(e) => {
          if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
          const next = STEPS[(i + (e.key === 'ArrowRight' ? 1 : STEPS.length - 1)) % STEPS.length];
          step = next.id;
          document.getElementById(`tab-${next.id}`)?.focus();
        }}
      >
        <span class="n num" aria-hidden="true">{i + 1}</span>{st.label}
      </button>
    {/each}
  </div>

  <div class="panel" id="conv-panel" role="tabpanel" aria-labelledby="tab-{step}" tabindex="-1">
    <div class="body stack">
      <p class="small">{STEPS.find((x) => x.id === step)?.text}</p>

      <div class="grid-2">
        <ModelSelect bind:value={form.source} kinds={CONVERT_SOURCE_KINDS} label="Source checkpoint" hint="Same source as above." />
        {#if step === 'verify'}
          <ModelSelect bind:value={form.bundle} kinds={VERIFY_OUTPUT_KINDS} label="Bundle to verify" hint="Choose the output registered as kind hgn." exclude={form.source ? [form.source] : []} />
        {:else}
          <div class="field">
            {#if step === 'convert' && form.anyHost}
              <span class="flabel">Run on</span>
              <p class="hint">
                The controller assigns this job to an idle pool worker when it starts, and never moves it. Each job converts one complete checkpoint on one host; queue more jobs to use more workers.
                {eligible.length ? `Pool: ${eligible.map((h) => h.name ?? h.id).join(', ')}.` : 'No pool is configured, so the job runs on the source host.'}
              </p>
            {:else}
              <HostSelect bind:value={form.host} kind={step} label="Run on" optional optionalLabel="Host of the source" />
            {/if}
            {#if step === 'convert'}
              <label class="inline"><input type="checkbox" bind:checked={form.anyHost} /> Any eligible pool worker</label>
            {/if}
          </div>
        {/if}
      </div>
      {#if step === 'verify'}
        <div class="grid-2">
          <HostSelect bind:value={form.host} kind="verify" label="Run on" optional optionalLabel="Host of the source" />
        </div>
      {/if}

      {#if usesBackends}
        <div class="grid-2">
          <div class="field">
            <label for="cv-profile">Profile</label>
            <select id="cv-profile" class="input" bind:value={form.profile} aria-describedby="cv-profile-hint">
              <option value="">Converter default</option>
              {#each CONVERT_ENUMS.profile as p (p)}<option value={p}>{p === 'v2' ? 'HGN v2' : p === 'ht43' ? 'HT43' : p}</option>{/each}
            </select>
            <span class="hint" id="cv-profile-hint">Output format of the bundle.</span>
          </div>
        </div>
      {/if}

      <p class="xs muted">Components and advanced options are shared with the form above.</p>

      {#if usesBackends || usesPlanOpts || step === 'verify'}
        <Advanced changed={optsChanged}>
          <div class="grid-3">
            {#if usesBackends}
              {#each ['ht_backend', 'i4r_backend', 'q4cp_backend'] as const as key (key)}
                <div class="field">
                  <label for="cv-{key}">{key.replace('_backend', '').toUpperCase()} encoder backend</label>
                  <select id="cv-{key}" class="input" bind:value={form[key]}>
                    <option value="">Converter default</option>
                    {#each CONVERT_ENUMS.backend as b (b)}<option value={b}>{b}</option>{/each}
                  </select>
                </div>
              {/each}
            {/if}
            {#if usesPlanOpts}
              <div class="field">
                <label for="cv-seed">Seed <span class="muted">(optional)</span></label>
                <input id="cv-seed" class="input num" type="number" min="0" step="1" bind:value={form.seed} inputmode="numeric" />
              </div>
              <div class="field">
                <label for="cv-threads">Threads <span class="muted">(optional)</span></label>
                <input id="cv-threads" class="input num" type="number" min="1" max="64" step="1" bind:value={form.threads} inputmode="numeric" />
              </div>
              <div class="field">
                <label for="cv-beam">HT beam <span class="muted">(optional)</span></label>
                <input id="cv-beam" class="input num" type="number" min="1" max="256" step="1" bind:value={form.ht_beam} inputmode="numeric" />
              </div>
            {/if}
            {#if step === 'verify'}
              <div class="field">
                <label for="cv-dec">Decode backend</label>
                <select id="cv-dec" class="input" bind:value={form.decode_backend}>
                  <option value="">Converter default</option>
                  {#each CONVERT_ENUMS.backend as b (b)}<option value={b}>{b}</option>{/each}
                </select>
              </div>
            {/if}
          </div>
          <p class="xs muted">The receipt records the backends that actually ran. A requested backend is not evidence of GPU use.</p>
        </Advanced>
      {/if}

      <Submit {blockers} busy={s.busy} onclick={submit}>Queue {STEPS.find((x) => x.id === step)?.label.toLowerCase()}</Submit>
      <SubmitStatus {s} />
    </div>
  </div>
    </div>
  </details>
  {/if}

  <RecentJobs kinds={['inspect', 'plan', 'convert', 'verify']} title="Recent converter jobs" limit={8} />
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  /* The four steps wrap on narrow screens instead of scrolling sideways. */
  .steps {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .step {
    flex: 1 1 auto;
    justify-content: center;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 40px;
    padding: 0 var(--s-4);
    background: transparent;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    color: var(--text-2);
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
  }
  .step:hover {
    color: var(--text);
  }
  .step[aria-selected='true'] {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--text);
  }
  .n {
    display: inline-grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 1px solid currentColor;
    font-size: var(--fs-xs);
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
  .manual > summary .t {
    font-weight: 500;
    font-size: var(--fs-sm);
  }
  .manual > summary .s {
    color: var(--text-2);
  }
  .manual-body {
    padding: var(--s-2) 0 var(--s-3);
  }
  .more {
    padding: var(--s-2) var(--s-4) var(--s-3);
  }
  .flabel {
    font-size: var(--fs-sm);
    font-weight: 500;
    color: var(--text-2);
  }
  .inline {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 36px;
    font-size: var(--fs-sm);
    cursor: pointer;
  }
</style>

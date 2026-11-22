<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { limits, resourceDisplay, resourceInput, resourceUnit, validateResources, type HostPolicy, type PolicyAction } from '../lib/host-policy';
  import { fieldError, visibleBlockers } from '../lib/blockers';
  import { focusAfterRemoval } from '../lib/focus';
  import { kindTitle } from '../lib/format';
  import Badge from './Badge.svelte';
  import FieldError from './FieldError.svelte';
  let {host, active, busy, save, onDirty}:
    {host:HostPolicy; active:boolean; busy:boolean; save:(action:PolicyAction,label:string)=>Promise<boolean>; onDirty:(dirty:boolean)=>void} = $props();
  let paused = $state(false), disabled = $state<string[]>([]), inputs = $state<Record<string,string>>({});
  let baseAdmission = $state<{paused:boolean; disabled:string[]}>();
  let baseResources = $state<Record<string,string>>();
  let touched = $state<Record<string,boolean>>({});
  let review = $state<'policy'|'resources'|null>(null);
  let confirmButton = $state<HTMLButtonElement>();
  let admissionButton = $state<HTMLButtonElement>();
  let resourceButton = $state<HTMLButtonElement>();
  const uid = $props.id();
  const headingId = $derived(active ? 'host-policy-h' : `${uid}-policy`);
  const savedResources = $derived(Object.fromEntries(Object.entries(host.resources).map(([k,v])=>[k,resourceInput(k,v)])));
  const resourceDiffers = (a: Record<string,string>, b: Record<string,string>) =>
    [...new Set([...Object.keys(a), ...Object.keys(b)])].some(k => (a[k] ?? '').trim() !== (b[k] ?? '').trim());
  const admissionDirty = $derived(!!baseAdmission && (baseAdmission.paused !== paused || baseAdmission.disabled.length !== disabled.length ||
    disabled.some(k => !baseAdmission?.disabled.includes(k))));
  const resourceDirty = $derived(!!baseResources && resourceDiffers(inputs, baseResources));
  function resetAdmission() {
    baseAdmission = { paused: host.paused, disabled: [...host.disabled_jobs] };
    paused = host.paused;
    disabled = [...host.disabled_jobs];
  }
  function resetResources() {
    baseResources = { ...savedResources };
    inputs = { ...savedResources };
    touched = {};
  }
  $effect(() => {
    // Refresh untouched fields. Dirty fields stay paired with their saved values.
    void host;
    untrack(() => {
      if (!admissionDirty) resetAdmission();
      if (!resourceDirty) resetResources();
    });
  });
  $effect(() => {
    onDirty(admissionDirty || resourceDirty);
    return () => onDirty(false);
  });
  const added = $derived(disabled.filter(k=>!host.disabled_jobs.includes(k)));
  const removed = $derived(host.disabled_jobs.filter(k=>!disabled.includes(k)));
  const changed = $derived(added.length > 0 || removed.length > 0 || paused !== host.paused);
  const validation = $derived(validateResources(host,inputs));
  const errors = $derived(visibleBlockers(validation.blockers, touched));
  const next = $derived(validation.values);
  const resourceChanges = $derived(next ? Object.keys(host.resource_bounds).filter(k=>host.resources[k] !== next?.[k]) : []);
  const admissionStale = $derived(admissionDirty && !!baseAdmission && (baseAdmission.paused !== host.paused ||
    baseAdmission.disabled.length !== host.disabled_jobs.length || baseAdmission.disabled.some(k => !host.disabled_jobs.includes(k)) ||
    disabled.some(k => !host.installed_jobs.includes(k))));
  const resourcesStale = $derived(resourceDirty && !!baseResources && resourceDiffers(baseResources, savedResources));
  const reviewStale = $derived(review === 'policy' ? admissionStale : resourcesStale || !next);
  const name = (key:string) => limits[key as keyof typeof limits];
  const display = (key:string, value:number|undefined) => value === undefined ? 'inherit' : resourceDisplay(key,value);
  function restrict(key:string) {
    disabled = disabled.includes(key) ? disabled.filter(k=>k!==key) : [...disabled,key];
  }
  async function openReview(section: 'policy'|'resources') {
    review = section;
    await tick();
    confirmButton?.focus();
  }
  function closeReview() {
    const next = review === 'policy' ? admissionButton : resourceButton;
    return focusAfterRemoval(() => review = null, () => next?.disabled ? document.getElementById(headingId) : next);
  }
  async function submit() {
    if (busy || !review || reviewStale) return;
    const common = {worker:host.id, revision:host.revision};
    const action:PolicyAction|null = review === 'policy' && changed
      ? {...common,action:'set_host_policy',paused, disabled_jobs:[...disabled].sort()}
      : review === 'resources' && next && resourceChanges.length ? {...common, action:'set_host_resources', resources:next} : null;
    if (!action) return;
    const hadFocus = document.activeElement === confirmButton;
    if (await save(action, `${review === 'policy' ? 'Change admission policy' : 'Change resource limits'} for ${host.id}`)) {
      if (review === 'policy') resetAdmission();
      else resetResources();
      await closeReview();
      if (hadFocus && document.activeElement === document.body) document.getElementById(headingId)?.focus();
    }
  }
</script>

<section class="stack" aria-labelledby={headingId}>
  <h4 id={headingId} tabindex="-1">Admission and resource limits</h4>
  <div class="row">
    <strong class="break">{host.name}</strong>
    <span class="mono muted xs break" title="The version of the saved host controls">{host.name !== host.id ? `${host.id} · ` : ''}revision {host.revision}</span>
    <Badge tone={host.paused ? 'warn' : 'muted'} label={host.draining ? 'Draining' : host.paused ? 'Paused' : 'Not paused'} />
  </div>
  <p class="small"><strong>Permitted job types:</strong> {host.allowed_jobs.map(kindTitle).join(', ') || 'None'}.
    {#if host.draining}Running jobs can finish, but this host accepts no new jobs.
    {:else if host.paused}New work is paused.{/if}</p>
  <fieldset class="stack" disabled={busy || review !== null}>
    <legend>Proposed admission policy</legend>
    <label><input type="checkbox" bind:checked={paused} /> Pause new work</label>
    <p class="xs muted">Running jobs can finish. Resuming does not add runtimes or change hardware checks.</p>
    <div>
      <strong class="small">Restrict job types</strong>
      <p class="xs muted">Check a type to block it. Clearing a restriction does not grant a missing capability.</p>
    </div>
    <div class="checks">
      {#each host.installed_jobs as key (key)}
        <label class="small"><input type="checkbox" checked={disabled.includes(key)} onchange={()=>restrict(key)} /> {kindTitle(key)}</label>
      {:else}<p class="small muted">No job types reported.</p>{/each}
    </div>
    {#if admissionStale}<p class="note warn">Saved admission settings changed. Discard your edits to load the new version.</p>{/if}
    <div class="row"><button bind:this={admissionButton} type="button" class="btn small" disabled={!changed || admissionStale} onclick={()=>openReview('policy')}>Review admission changes</button>
      {#if admissionDirty}<button type="button" class="btn small ghost" onclick={() => focusAfterRemoval(resetAdmission, () => document.getElementById(headingId))}>Discard admission edits</button>{/if}</div>
  </fieldset>
  <fieldset class="stack" disabled={busy || review !== null || !host.resourceEditable}>
    <legend>Resource limits</legend>
    {#if Object.keys(host.resource_bounds).length}
      <p class="xs muted">Leave a field empty to use the worker setting. RAM and disk limits apply when a job starts. Local limits can be stricter. “Not reported” means the worker did not report the limit it uses.</p>
      <div class="grid-2">
        {#each Object.entries(host.resource_bounds) as [k,bound] (k)}
          {@const error = fieldError(errors, k)}
          <div class="field">
            <label for="{uid}-{k}">{name(k)}{resourceUnit(k) ? ` (${resourceUnit(k)})` : ''}</label>
            <input class="input num" id="{uid}-{k}" inputmode={resourceUnit(k) ? 'decimal' : 'numeric'} value={inputs[k] ?? ''} placeholder="Inherit"
              oninput={e=>inputs[k]=e.currentTarget.value} onblur={()=>touched[k]=true} aria-invalid={error ? 'true' : undefined}
              aria-describedby="{error ? `${uid}-${k}-error ` : ''}{uid}-{k}-hint" />
            <FieldError id="{uid}-{k}-error" {error} />
            <span class="hint" id="{uid}-{k}-hint">Effective: {host.effective_resources[k] === undefined ? 'not reported' : resourceDisplay(k, host.effective_resources[k])} · range {resourceDisplay(k,bound.min)} to {resourceDisplay(k,bound.max)}</span>
          </div>
        {/each}
      </div>
      {#if !host.resourceEditable}<p class="note warn">This browser cannot safely replace the current resource settings.</p>
      {/if}
      {#if resourcesStale}<p class="note warn">Saved resource limits changed. Discard your edits to load the new version.</p>{/if}
      <div class="row"><button bind:this={resourceButton} type="button" class="btn small" disabled={!resourceChanges.length || resourcesStale} onclick={()=>openReview('resources')}>Review resource changes</button>
        {#if resourceDirty}<button type="button" class="btn small ghost" onclick={() => focusAfterRemoval(resetResources, () => document.getElementById(headingId))}>Discard resource edits</button>{/if}</div>
    {:else}<p class="small muted">No adjustable limits reported for this host.</p>{/if}
  </fieldset>
  {#if review}
    <div class="note stack" aria-live="polite">
      <strong>Review {review === 'policy' ? 'admission policy' : 'resource limits'} for {host.name}</strong>
      <ul>
        {#if review === 'policy'}
          {#if paused !== host.paused}<li>New work: {host.paused ? 'paused' : 'not paused'} → {paused ? 'paused' : 'not paused'}</li>{/if}
          {#each added as k}<li>Block {kindTitle(k)}</li>{/each}
          {#each removed as k}<li>Remove restriction on {kindTitle(k)}</li>{/each}
        {:else}
          {#each resourceChanges as k}<li>{name(k)}: {display(k,host.resources[k])} → {display(k,next?.[k])}</li>{/each}
        {/if}
      </ul>
      <div class="row">
        <button bind:this={confirmButton} type="button" class="btn primary" disabled={busy || reviewStale} onclick={submit}>Confirm {review === 'policy' ? 'admission policy' : 'resource limits'}</button>
        <button type="button" class="btn" disabled={busy} onclick={closeReview}>Back to editing</button>
      </div>
    </div>
  {/if}
</section>
<style>
  fieldset { padding: var(--s-3); margin: 0; border: 1px solid var(--line-soft); border-radius: var(--r-tile); min-width: 0; }
  legend { font-weight: 600; padding-inline: var(--s-1); }
  .checks { display: flex; flex-wrap: wrap; gap: var(--s-2) var(--s-4); }
  label input[type=checkbox] { accent-color: var(--accent); vertical-align: middle; }
</style>

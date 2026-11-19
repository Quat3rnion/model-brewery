<script lang="ts">
  import { limits, resourceValues, type HostPolicy, type PolicyAction } from '../lib/host-policy';
  import { kindTitle } from '../lib/format';
  import Badge from './Badge.svelte';
  let {host, busy, save}: {host:HostPolicy; busy:boolean; save:(action:PolicyAction,label:string)=>Promise<void>} = $props();
  // Parent keys the form by host and revision. Confirmation always refers to that snapshot.
  let paused = $state(false), disabled = $state<string[]>([]), inputs = $state<Record<string,string>>({});
  let review = $state<'policy'|'resources'|null>(null);
  const uid = $props.id();
  $effect(() => {
    paused = host.paused;
    disabled = [...host.disabled_jobs];
    inputs = Object.fromEntries(Object.entries(host.resources).map(([k,v])=>[k,String(v)]));
    review = null;
  });
  const added = $derived(disabled.filter(k=>!host.disabled_jobs.includes(k)));
  const removed = $derived(host.disabled_jobs.filter(k=>!disabled.includes(k)));
  const changed = $derived(added.length > 0 || removed.length > 0 || paused !== host.paused);
  const next = $derived(resourceValues(host,inputs));
  const resourceChanges = $derived(next ? Object.keys(host.resource_bounds).filter(k=>host.resources[k] !== next![k]) : []);
  const name = (key:string) => limits[key as keyof typeof limits];
  function restrict(key:string) {
    disabled = disabled.includes(key) ? disabled.filter(k=>k!==key) : [...disabled,key];
  }
  async function submit() {
    if (busy || !review) return;
    const common = {worker:host.id, revision:host.revision};
    const action:PolicyAction|null = review === 'policy' && changed
      ? {...common,action:'set_host_policy',paused, disabled_jobs:[...disabled].sort()}
      : review === 'resources' && next && resourceChanges.length ? {...common, action:'set_host_resources', resources:next} : null;
    if (action) await save(action,`Update ${host.id}`);
    review = null;
  }
</script>

<div class="stack">
  <div class="row">
    <h4 class="break">{host.name}</h4>
    <span class="mono muted xs break">{host.id} · revision {host.revision}</span>
    <Badge tone={host.paused ? 'warn' : 'muted'} label={host.draining ? 'Draining' : host.paused ? 'Paused' : 'Not paused'} />
  </div>
  <p class="small"><strong>Permitted job types:</strong> {host.allowed_jobs.map(kindTitle).join(', ') || 'None'}.
    {#if host.paused}New work is paused.{/if}</p>
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
    <div><button type="button" class="btn small" disabled={!changed} onclick={()=>review='policy'}>Review admission changes</button></div>
  </fieldset>
  <fieldset class="stack" disabled={busy || review !== null || !host.resourceEditable}>
    <legend>Resource limits</legend>
    {#if Object.keys(host.resource_bounds).length}
      <p class="xs muted">Leave a field empty to inherit the worker setting. RAM and disk limits are admission checks, not live quotas. Local launch limits may be stricter.</p>
      <div class="grid-2">
        {#each Object.entries(host.resource_bounds) as [k,bound] (k)}
          <div class="field">
            <label for="{uid}-{k}">{name(k)}</label>
            <input class="input num" id="{uid}-{k}" inputmode="numeric" value={inputs[k] ?? ''} placeholder="Inherit" oninput={e=>inputs[k]=e.currentTarget.value} aria-describedby="{uid}-{k}-hint" />
            <span class="hint" id="{uid}-{k}-hint">Effective: {host.effective_resources[k]?.toLocaleString() ?? 'not reported'} · range {bound.min.toLocaleString()}–{bound.max.toLocaleString()}</span>
          </div>
        {/each}
      </div>
      {#if !host.resourceEditable}<p class="note warn">This browser cannot safely replace the current resource settings.</p>
      {:else if !next}<p class="note warn">Enter a whole number in the reported range, or clear the field to inherit.</p>{/if}
      <div><button type="button" class="btn small" disabled={!next || !resourceChanges.length} onclick={()=>review='resources'}>Review resource changes</button></div>
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
          {#each resourceChanges as k}<li>{name(k)}: {host.resources[k] ?? 'inherit'} → {next?.[k] ?? 'inherit'}</li>{/each}
        {/if}
      </ul>
      <div class="row">
        <button type="button" class="btn primary" disabled={busy} onclick={submit}>Confirm {review === 'policy' ? 'admission policy' : 'resource limits'}</button>
        <button type="button" class="btn" disabled={busy} onclick={()=>review=null}>Back to editing</button>
      </div>
    </div>
  {/if}
</div>
<style>
  fieldset { padding: var(--s-3); margin: 0; border: 1px solid var(--line-soft); border-radius: var(--r-tile); min-width: 0; }
  legend { font-weight: 600; padding-inline: var(--s-1); }
  .checks { display: flex; flex-wrap: wrap; gap: var(--s-2) var(--s-4); }
  label input[type=checkbox] { accent-color: var(--accent); vertical-align: middle; }
</style>

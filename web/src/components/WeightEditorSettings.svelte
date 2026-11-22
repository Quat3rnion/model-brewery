<script lang="ts">
  import Badge from './Badge.svelte';
  import { brewery } from '../lib/store.svelte';
  import { WEIGHT_SOURCES, modelLabel } from '../lib/format';
  import type { HostPolicy, PolicyAction } from '../lib/host-policy';
  let {host,active,policies,busy,save,onDirty}:
    {host:HostPolicy; active:boolean; policies:HostPolicy[]; busy:boolean; save:(action:PolicyAction,label:string)=>Promise<boolean>; onDirty:(dirty:boolean)=>void} = $props();
  const uid = $props.id();
  const headingId = $derived(active ? 'weight-settings-h' : `${uid}-weights`);
  const w = $derived(host.weightEditor);
  let directory = $state<string|null>(null);
  let reference = $state<string|null>(null);
  let submittingSetup = $state(false);
  const remotePath = $derived(directory ?? w?.setup?.model_directory ?? '');
  const credential = $derived(reference ?? w?.setup?.credentials[0]?.id ?? '');
  const dirty = $derived(directory !== null || reference !== null);
  function reset() { directory = reference = null; }
  $effect(() => {
    onDirty(dirty);
    return () => onDirty(false);
  });
  const canSetup = $derived(!busy && !submittingSetup && !w?.setup?.blocker && remotePath.trim().startsWith('/') && w?.setup?.credentials.some(c=>c.id===credential));
  async function setupEditor() {
    if (!canSetup) return;
    submittingSetup=true;
    try {
      if (await save({action:'setup_weight_editor',worker:host.id,revision:host.revision,model_directory:remotePath.trim(),credential},`Set up the editor on ${host.id}`)) reset();
    } finally { submittingSetup=false; }
  }
  const sources = $derived((brewery.state?.models ?? []).filter(m => m.available !== false && WEIGHT_SOURCES.includes(m.kind)));
  const owners = $derived(policies.filter(p => p.id !== host.id && p.weightEditor?.available)
    .map(p => ({policy:p, models:sources.filter(m => m.host === p.id)})).filter(o => o.models.length));
  const features = [ ['weight_inspect','Inspect weights'],['weight_propose','Manual proposals'],['weight_tune','Automatic tuning'],['weight_apply','Apply reviewed revisions'],['weight_visual','Visual tensor selection'],['weight_hgn','Native HGN','Models in Brewery’s HGN format'],['weight_safetensors','SafeTensors and indexed HF'],['weight_packed','Packed-code editing','Edit quantized weights without first converting the whole model'] ];
  async function change() {
    if (!w?.available || busy) return;
    await save({action:'set_weight_editor',worker:host.id,revision:host.revision,enabled:!w.enabled},`${w.enabled ? 'Disable' : 'Enable'} weight editing on ${host.id}`);
  }
</script>
<section id={active ? 'weight-editor-settings' : undefined} class="stack" aria-labelledby={headingId}>
  <div class="row"><h4 id={headingId} tabindex="-1">Weight editing</h4><Badge tone={w?.enabled ? 'ok':'muted'} label={!w || (!w.available && !w.setup) ? 'Not reported' : w.enabled ? 'Enabled' : w.available ? 'Disabled' : 'Setup required'} /></div>
  {#if !w || (!w.available && !w.setup)}<p class="note warn">The controller did not report how to set up weight editing on this host.</p>
  {/if}
  {#if w?.reason}<p class="note" class:warn={!w.available}>{w.reason}</p>{/if}
  {#if w?.setup}
    <p class="small">Editor setup: <strong>{w.setup.status.replaceAll('_',' ')}</strong></p>
    {#if w.setup.reason}<p class="note small">{w.setup.reason}</p>{/if}
    {#if w.available && w.setup.model_directory}<p class="small">Remote model directory: <code>{w.setup.model_directory}</code></p>{/if}
  {/if}
  {#if w?.setup && !w.available}
    <div class="stack">
      {#if w.setup.blocker === 'runtime'}
        <p class="small">Ask the controller administrator to configure the editor runtime, then select <strong>Refresh infrastructure</strong>.</p>
      {:else if w.setup.credentials.length && !w.setup.blocker}
        <div class="field">
          <label for="{uid}-path">Editor model directory</label>
          <input id="{uid}-path" class="input" type="text" autocomplete="off" spellcheck="false" value={remotePath}
            disabled={busy} oninput={e=>directory=e.currentTarget.value} aria-describedby="{uid}-path-help" />
          <p class="xs muted" id="{uid}-path-help">Use a path on {host.name}. This directory is for editing, not read-only source access.</p>
        </div>
        <div class="field"><label for="{uid}-credential">SSH login for editor setup</label>
          <select id="{uid}-credential" class="input" value={credential} disabled={busy} onchange={e=>reference=e.currentTarget.value}>
            {#each w.setup.credentials as ref (ref.id)}<option value={ref.id}>{ref.label}</option>{/each}
          </select>
        </div>
        <p class="small">When you select <strong>Set up and enable editor</strong>, the controller will use this login to install the editor on {host.name} and verify it before enabling weight editing.</p>
        <div><button type="button" class="btn primary" disabled={!canSetup} onclick={setupEditor}>Set up and enable editor</button></div>
      {:else if !w.setup.blocker || w.setup.blocker === 'credential'}
        <p class="small">In SSH hosts, select <strong>Add SSH host</strong> or <strong>Edit definition</strong>. Use the worker’s address and SSH port. For a new definition, choose a different host ID that is not already in use. Choose the SSH login identity, save the definition, then select <strong>Test connection</strong>. After verification, select <strong>Refresh infrastructure</strong> to see the login here.</p>
        <div><button class="btn small" type="button" onclick={()=>document.getElementById('ssh-hosts-heading')?.focus()}>Go to SSH hosts</button></div>
      {:else}
        <p class="small muted">The controller does not allow editor setup now.</p>
      {/if}
      {#if submittingSetup}<p role="status" class="small">Setting up and checking the editor on {host.name}…</p>{/if}
    </div>
  {/if}
  {#if dirty}<div><button type="button" class="btn ghost small" disabled={busy} onclick={reset}>Discard editor setup edits</button></div>{/if}
  {#if !w?.available}
    {#if host.directories?.available && !host.directories.granted.length && !host.directories.configured.length}
      <p class="small">No model source directory is listed for this worker. <a href="#/settings/infrastructure?host={encodeURIComponent(host.id)}&section=directories">Approve a source directory for {host.name}</a>.</p>
    {/if}
    <p class="small"><a href="#/library">Open the Library</a> to find your model and its host. Changes to editor settings apply only to models on that host.</p>
    {#if owners.length}
      <ul class="owners small" aria-label="Editor settings for hosts with registered models">
        {#each owners as {policy, models} (policy.id)}
          <li><a href="#/settings/infrastructure?host={encodeURIComponent(policy.id)}">Editor settings for {policy.name}</a> · {modelLabel(models[0])}{models.length > 1 ? ` (+${models.length - 1} more)` : ''}</li>
        {/each}
      </ul>
    {/if}
  {/if}
  {#if w?.available}
    {#if host.paused}<p class="note warn">New jobs are paused. Resume admission in the host policy below before submitting an edit.</p>{/if}
    <div class="coverage">
      {#each features as [key,label,help]}
        <div class="row small"><span>{label}{#if help}<br /><span class="xs muted">{help}</span>{/if}</span>
          <Badge tone={w.enabled && w.capabilities[key] && !host.disabled_jobs.includes(key) && !host.paused ? 'ok':'muted'} label={host.disabled_jobs.includes(key) ? 'Restricted by policy' : !w.enabled ? 'Off' : !w.capabilities[key] ? 'Not available' : host.paused ? 'Paused' : 'Enabled'} />
        </div>
      {/each}
    </div>
    <p class="xs muted">GGUF uses the base editing route. File access, tensor encoding, source verification and resource limits are checked for each job. Tuning also needs separate calibration and held-out entries on this host.</p>
    {#if w.available}
      <div><button class="btn" disabled={busy} onclick={change}>{w.enabled ? 'Disable' : 'Enable'} weight editing</button></div>
    {/if}
  {/if}
</section>
<style>.owners {margin:0; padding-inline-start:var(--s-5);} .coverage { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,18rem),1fr)); gap:var(--s-2) var(--s-5); } .coverage .row {justify-content:space-between;} section {overflow-wrap:anywhere;min-width:0;padding-bottom:var(--s-4);border-bottom:1px solid var(--line-soft);}</style>

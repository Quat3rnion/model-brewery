<script lang="ts">
  import Badge from './Badge.svelte';
  import { brewery } from '../lib/store.svelte';
  import { WEIGHT_SOURCES, modelLabel } from '../lib/format';
  import type { HostPolicy, PolicyAction } from '../lib/host-policy';
  let {host,policies,busy,save}: {host:HostPolicy; policies:HostPolicy[]; busy:boolean; save:(action:PolicyAction,label:string)=>Promise<void>} = $props();
  const w = $derived(host.weightEditor);
  let directory = $state<string|null>(null);
  let reference = $state<string|null>(null);
  let submittingSetup = $state(false);
  const remotePath = $derived(directory ?? w?.setup?.model_directory ?? '');
  const credential = $derived(reference ?? w?.setup?.credentials[0]?.id ?? '');
  const canSetup = $derived(!busy && remotePath.trim().startsWith('/') && w?.setup?.credentials.some(c=>c.id===credential));
  async function setupEditor() {
    if (!canSetup) return;
    submittingSetup=true;
    try {
      await save({action:'setup_weight_editor',worker:host.id,revision:host.revision,model_directory:remotePath.trim(),credential},`Set up the editor on ${host.id}`);
    } finally { submittingSetup=false; }
  }
  const sources = $derived((brewery.state?.models ?? []).filter(m => m.available !== false && WEIGHT_SOURCES.includes(m.kind)));
  const owners = $derived(policies.filter(p => p.id !== host.id && p.weightEditor?.available)
    .map(p => ({policy:p, models:sources.filter(m => m.host === p.id)})).filter(o => o.models.length));
  const features = [ ['weight_inspect','Inspect weights'],['weight_propose','Manual proposals'],['weight_tune','Automatic tuning'],['weight_apply','Apply reviewed revisions'],['weight_visual','Visual tensor selection'],['weight_hgn','Native HGN'],['weight_safetensors','SafeTensors and indexed HF'],['weight_packed','Packed-code editing'] ];
  async function change() {
    if (!w?.available || busy) return;
    await save({action:'set_weight_editor',worker:host.id,revision:host.revision,enabled:!w.enabled},`${w.enabled ? 'Disable' : 'Enable'} weight editing on ${host.id}`);
  }
</script>
<section id="weight-editor-settings" class="stack" aria-labelledby="weight-settings-h">
  <div class="row"><h4 id="weight-settings-h" tabindex="-1">Weight editing</h4><Badge tone={w?.enabled ? 'ok':'muted'} label={!w ? 'Not reported' : w.enabled ? 'Enabled' : w.available ? 'Disabled' : 'Setup required'} /></div>
  <p class="small">Selected host: <strong>{host.name}</strong> · <span class="mono">{host.id}</span></p>
  {#if !w}<p class="note warn">The controller did not report an editor configuration for this host.</p>
  {:else if w.reason}<p class="note" class:warn={!w.available}>{w.reason}</p>{/if}
  {#if w?.setup}
    <p class="small">Editor setup: <strong>{w.setup.status.replaceAll('_',' ')}</strong></p>
    {#if w.setup.reason}<p class="note small">{w.setup.reason}</p>{/if}
    {#if w.available && w.setup.model_directory}<p class="small">Remote model directory: <code>{w.setup.model_directory}</code></p>{/if}
  {/if}
  {#if w && !w.available}
    <div class="stack">
      <div class="field">
        <label for="editor-path">Remote model directory</label>
        <input id="editor-path" class="input" type="text" autocomplete="off" spellcheck="false" value={remotePath}
          disabled={!w.setup || busy} oninput={e=>directory=e.currentTarget.value} aria-describedby="editor-path-help" />
        <p class="xs muted" id="editor-path-help">Use the absolute path on {host.name}, not a directory on this device.</p>
      </div>
      {#if w.setup?.credentials.length}
        <div class="field"><label for="editor-credential">Saved SSH credential</label>
          <select id="editor-credential" class="input" value={credential} disabled={busy} onchange={e=>reference=e.currentTarget.value}>
            {#each w.setup.credentials as ref (ref.id)}<option value={ref.id}>{ref.label}</option>{/each}
          </select>
        </div>
      {:else if !w.setup?.reason}
        <p class="small muted">{w.setup ? 'No saved SSH credential is available for setup.' : 'This controller did not report editor setup for this host.'}</p>
      {/if}
      <div><button type="button" class="btn primary" disabled={!canSetup} onclick={setupEditor}>Set up and enable editor</button></div>
      {#if submittingSetup}<p role="status" class="small">Setting up and checking the editor on {host.name}…</p>{/if}
    </div>
  {/if}
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
  {#if w}
    {#if host.paused}<p class="note warn">New jobs are paused. Resume admission in the host policy below before submitting an edit.</p>{/if}
    <div class="coverage">
      {#each features as [key,label]}
        <div class="row small"><span>{label}</span>
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

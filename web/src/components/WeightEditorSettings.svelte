<script lang="ts">
  import Badge from './Badge.svelte';
  import { brewery } from '../lib/store.svelte';
  import { WEIGHT_SOURCES, modelLabel } from '../lib/format';
  import type { HostPolicy, PolicyAction } from '../lib/host-policy';
  let {host,policies,busy,save}: {host:HostPolicy; policies:HostPolicy[]; busy:boolean; save:(action:PolicyAction,label:string)=>Promise<void>} = $props();
  const w = $derived(host.weightEditor);
  const sources = $derived((brewery.state?.models ?? []).filter(m => m.available !== false && WEIGHT_SOURCES.includes(m.kind)));
  const owners = $derived(policies.filter(p => p.id !== host.id && p.weightEditor?.available)
    .map(p => ({policy:p, models:sources.filter(m => m.host === p.id)})).filter(o => o.models.length));
  let confirming = $state(false);
  const features = [ ['weight_inspect','Inspect weights'],['weight_propose','Manual proposals'],['weight_tune','Automatic tuning'],['weight_apply','Apply reviewed revisions'],['weight_visual','Visual tensor selection'],['weight_hgn','Native HGN'],['weight_safetensors','SafeTensors and indexed HF'],['weight_packed','Packed-code editing'] ];
  async function change() {
    if (!w?.available || !confirming || busy) return;
    await save({action:'set_weight_editor',worker:host.id,revision:host.revision,enabled:!w.enabled},`${w.enabled ? 'Disable' : 'Enable'} weight editing on ${host.id}`);
    confirming=false;
  }
</script>
<section id="weight-editor-settings" class="stack" aria-labelledby="weight-settings-h">
  <div class="row"><h4 id="weight-settings-h" tabindex="-1">Weight editing</h4><Badge tone={w?.enabled ? 'ok':'muted'} label={!w ? 'Not reported' : w.enabled ? 'Enabled' : w.available ? 'Disabled' : 'Setup required'} /></div>
  <p class="small">Manage the editor on <strong>{host.name}</strong>. Changing this setting does not install software, start a job, or change a model.</p>
  {#if !w}<p class="note warn">The controller did not report an editor configuration for this host.</p>
  {:else if w.reason}<p class="note" class:warn={!w.available}>{w.reason}</p>{/if}
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
      {#if confirming}<div class="note stack"><strong>{w.enabled ? 'Disable' : 'Enable'} weight editing on {host.name}?</strong><span class="small">This changes the editor policy for this host only. Existing job restrictions remain in effect.</span>
        <div class="row"><button class="btn primary" disabled={busy} onclick={change}>Confirm {w.enabled ? 'disable' : 'enable'}</button><button class="btn" disabled={busy} onclick={()=>confirming=false}>Cancel</button></div></div>
      {:else}<div><button class="btn" disabled={busy} onclick={()=>confirming=true}>{w.enabled ? 'Disable' : 'Enable'} weight editing</button></div>{/if}
    {/if}
  {/if}
</section>
<style>.owners {margin:0; padding-inline-start:var(--s-5); overflow-wrap:anywhere;} .coverage { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,18rem),1fr)); gap:var(--s-2) var(--s-5); } .coverage .row {justify-content:space-between;} section {padding-bottom:var(--s-4);border-bottom:1px solid var(--line-soft);}</style>

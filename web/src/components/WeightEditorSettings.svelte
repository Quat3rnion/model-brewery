<script lang="ts">
  import Badge from './Badge.svelte';
  import type { HostPolicy, PolicyAction } from '../lib/host-policy';
  let {host,busy,save}: {host:HostPolicy; busy:boolean; save:(action:PolicyAction,label:string)=>Promise<void>} = $props();
  const w = $derived(host.weightEditor);
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
  {#if !w}<p class="note warn">The controller did not report an editor configuration for this host. Refresh infrastructure after the controller and worker have been configured.</p>
  {:else}
    {#if w.reason}<p class="note" class:warn={!w.available}>{w.reason}</p>{/if}
    {#if !w.available}<p class="small muted">Configure a supported weight-edit runtime on this host, then refresh. SSH installation alone does not establish editing support.</p>{/if}
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
<style>.coverage { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,18rem),1fr)); gap:var(--s-2) var(--s-5); } .coverage .row {justify-content:space-between;} section {padding-bottom:var(--s-4);border-bottom:1px solid var(--line-soft);}</style>

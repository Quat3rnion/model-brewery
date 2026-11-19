<script lang="ts">
  import { onMount } from 'svelte';
  import { listHosts, permitted, problem, removeHost, saveHost, sshAction, type Action, type Definition, type SshHost } from '../lib/ssh-hosts';
  const blank = ():Definition => ({id:'',name:'',host:'',user:'',port:22,fingerprint:''});
  let hosts = $state<SshHost[]>([]);
  let available = $state(false), ready = $state(false), busy = $state(false);
  let error = $state(''), note = $state('');
  let editing = $state(false), revision = $state<number>();
  let draft = $state<Definition>(blank());
  let removing = $state(''), approved = $state('');
  const key = (h:SshHost) => JSON.stringify([h.id,h.revision,h.confirmation]);
  async function run(work:()=>Promise<void>) {
    if (busy) return;
    busy = true; error = note = approved = removing = '';
    try { await work(); } catch(e) { error = e instanceof Error ? e.message : 'Request failed. Refresh the list.'; }
    finally { busy = false; }
  }
  function refresh() { return run(async () => {
    ready = false; editing = false;
    const state = await listHosts();
    hosts = state.hosts; available = state.available; ready = true;
  }); }
  onMount(()=>{ void refresh(); });
  function edit(h?:SshHost) {
    draft = h ? {...h} : blank(); revision = h?.revision; editing = true;
    error = note = removing = approved = '';
  }
  function update(h:SshHost) { hosts = [...hosts.filter(x=>x.id !== h.id),h].sort((a,b)=>a.id.localeCompare(b.id)); }
  function save() { return run(async () => { update(await saveHost(draft,revision)); editing = false; note = 'Host definition saved.'; }); }
  function action(h:SshHost,a:Action) {
    const confirmed = approved === key(h);
    return run(async () => { update(await sshAction(h,a,confirmed)); note = {test:'Connection verified.',plan:'Installation plan ready for review.',install:'Installation complete.'}[a]; });
  }
  function remove(h:SshHost) {
    if (removing !== key(h)) return;
    return run(async () => { await removeHost(h); hosts = hosts.filter(x=>x.id!==h.id); note = 'Host definition removed. Remote files were not changed.'; });
  }
</script>

<section class="panel" aria-labelledby="ssh-hosts-heading">
  <header><h2 id="ssh-hosts-heading">SSH hosts</h2></header>
  <div class="body stack">
    <p class="small muted">Define a remote host, verify its SSH identity, then review the installation plan. Saving a definition does not connect to the host.</p>
    <div class="row">
      <button type="button" class="btn small" disabled={busy} onclick={refresh}>Refresh SSH hosts</button>
      {#if ready && !editing}<button type="button" class="btn small" disabled={busy} onclick={()=>edit()}>Add SSH host</button>{/if}
    </div>
    {#if !ready && busy}<p role="status">Loading SSH hosts…</p>{/if}
    {#if ready}
      {#if !available}<p class="note warn">The SSH helper is unavailable. Connections and installations are disabled.</p>{/if}
      {#if editing}
        <form class="stack" onsubmit={(e)=>{e.preventDefault(); if (!problem(draft)) void save();}}>
          <fieldset disabled={busy} class="stack">
            <legend>{revision === undefined ? 'New SSH host' : 'Edit SSH host'}</legend>
            <div class="grid-2">
              <label class="field">Host ID<input class="input" required disabled={revision !== undefined} bind:value={draft.id} /></label>
              <label class="field">Display name<input class="input" required bind:value={draft.name} /></label>
              <label class="field">Host address<input class="input" required bind:value={draft.host} autocomplete="off" spellcheck="false" /></label>
              <label class="field">SSH port<input class="input" required type="number" min="1" max="65535" bind:value={draft.port} /></label>
              <label class="field">SSH account<input class="input" required bind:value={draft.user} autocomplete="off" spellcheck="false" /></label>
              <label class="field">Host key fingerprint<input class="input mono" required bind:value={draft.fingerprint} placeholder="SHA256:…" aria-describedby="ssh-pin-help" autocomplete="off" spellcheck="false" /></label>
            </div>
            <p id="ssh-pin-help" class="xs muted">Get the expected SHA256 fingerprint from the host administrator through a trusted channel. Do not enter a password or private key.</p>
            <div class="row">
              <button class="btn primary" disabled={!!problem(draft)} type="submit">Save definition</button>
              <button class="btn" type="button" onclick={()=>editing=false}>Cancel edit</button>
              {#if problem(draft)}<span class="xs muted">{problem(draft)}</span>{/if}
            </div>
          </fieldset>
        </form>
      {:else if !hosts.length}<p class="small muted">No SSH hosts defined.</p>{/if}
      {#each hosts as h (h.id)}
        <article class="stack">
          <h3 class="break">{h.name} <span class="small muted">({h.id})</span></h3>
          <div class="small break"><strong>{h.status}</strong> · {h.user}@{h.host} · port {h.port} · revision {h.revision}</div>
          <div class="xs mono break">{h.fingerprint}</div>
          {#if !editing}
            <div class="row">
              <button class="btn small" type="button" disabled={busy} onclick={()=>edit(h)}>Edit definition</button>
              <button class="btn small" type="button" disabled={busy || !available || !permitted(h,'test')} onclick={()=>action(h,'test')}>Test connection</button>
              <button class="btn small" type="button" disabled={busy || !available || !permitted(h,'plan')} onclick={()=>action(h,'plan')}>Plan installation</button>
              <button class="btn small ghost" type="button" disabled={busy} onclick={()=>{approved='';removing=key(h);}}>Remove definition</button>
            </div>
            {#if removing === key(h)}
              <p class="small">Remove this definition from the controller? No remote software or data will be deleted.</p>
              <div class="row"><button class="btn danger" type="button" onclick={()=>remove(h)}>Confirm removal</button><button class="btn" type="button" onclick={()=>removing=''}>Keep host</button></div>
            {:else if permitted(h,'install')}
              <div class="note info stack">
                <strong>Installation plan</strong>
                <p class="small">Install the verified Brewery worker, engine and worker configuration in the home directory of {h.user} on {h.host}. No system services or accounts are changed.</p>
                <div class="xs break">Plan ID: <span class="mono">{h.confirmation}</span></div>
                <label><input type="checkbox" disabled={!available || busy} checked={approved === key(h)} onchange={(e)=>approved=e.currentTarget.checked ? key(h) : ''} /> I confirm this plan for {h.name}.</label>
                <div><button class="btn primary" type="button" disabled={busy || !available || approved !== key(h)} onclick={()=>action(h,'install')}>Install on host</button></div>
              </div>
            {/if}
          {/if}
        </article>
      {/each}
    {/if}
    {#if note}<p class="small" role="status">{note}</p>{/if}
    {#if error}<p class="note bad" role="alert">{error}</p>{/if}
  </div>
</section>

<style>
  fieldset { min-width: 0; border: 1px solid var(--line-soft); padding: var(--s-3); border-radius: var(--r-tile); }
  article { min-width: 0; padding-top: var(--s-3); border-top: 1px solid var(--line-soft); }
</style>

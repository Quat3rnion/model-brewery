<script lang="ts">
  import { tick } from 'svelte';
  import { fieldError, visibleBlockers } from '../lib/blockers';
  import { focusAfterRemoval } from '../lib/focus';
  import FieldError from './FieldError.svelte';
  import type { HostPolicy } from '../lib/host-policy';
  import type { DirectoryChange } from '../lib/model-directories';
  let {host,active,busy,save,onDirty}:
    {host:HostPolicy; active:boolean; busy:boolean; save:(change:DirectoryChange)=>Promise<boolean>; onDirty:(dirty:boolean)=>void} = $props();
  const uid = $props.id();
  const headingId = $derived(active ? 'model-directories-h' : `${uid}-directories`);
  const dirs = $derived(host.directories);
  let action = $state<DirectoryChange['action']>('add');
  let previous = $state('');
  let path = $state('');
  let touched = $state(false);
  let input = $state<HTMLInputElement>();
  let saveButton = $state<HTMLButtonElement>();
  const problem = $derived(!path ? 'Enter a directory path.' :
    !path.startsWith('/') || path === '/' || path.trim() !== path || /[\u0000-\u001f\u007f\u202a-\u202e\u2066-\u2069]/u.test(path)
      ? 'Enter an absolute directory path, not the filesystem root.' :
    action !== 'remove' && (dirs?.configured.includes(path) || dirs?.granted.includes(path)) ? 'This directory is already listed.' : '');
  const error = $derived(fieldError(visibleBlockers(problem ? [{field: 'path', text: problem}] : [], { path: touched }), 'path'));
  const pending = $derived(!!path || action !== 'add');
  $effect(() => {
    onDirty(pending);
    return () => onDirty(false);
  });
  function reset() { action = 'add'; previous = path = ''; touched = false; }
  async function draft(next:DirectoryChange['action'], old='') {
    action=next; previous=old; path=old; touched=false;
    await tick();
    if (action==='remove') saveButton?.focus(); else { input?.focus(); input?.select(); }
  }
  async function submit(e: SubmitEvent) {
    e.preventDefault();
    if (busy || problem || !dirs?.available) return;
    if (await save({worker:host.id,revision:host.revision,action,path,...(action==='replace'?{previous_path:previous}:{})})) {
      await focusAfterRemoval(reset, () => document.getElementById(headingId));
      if (document.activeElement === document.body) document.getElementById(headingId)?.focus();
    }
  }
</script>

<section class="stack directories" aria-labelledby={headingId}>
  <h4 tabindex="-1" id={headingId}>Model source directories</h4>
  {#if !dirs}
    <p class="small muted">The controller did not report source directory settings for this worker.</p>
  {:else}
    <p class="small">Read-only source access for <strong>{host.name}</strong> (<code>{host.id}</code>). Saving a directory does not enable editing, register a model, or change files.</p>
    <div class="stack">
      <strong class="small">Configured directories</strong>
      {#each dirs.configured as p}<code class="path">{p}</code>{:else}<span class="small muted">None configured.</span>{/each}
      {#if dirs.configured.length}<span class="xs muted">These paths come from worker configuration and cannot be changed here.</span>{/if}
    </div>
    <div class="stack">
      <strong class="small">Directories approved in Settings</strong>
      {#each dirs.granted as p}
        <div class="row">
          <code class="path">{p}</code>
          {#if dirs.available}
            <button class="btn small" disabled={busy} onclick={()=>draft('replace',p)} aria-label="Change {p}">Change</button>
            <button class="btn small" disabled={busy} onclick={()=>draft('remove',p)} aria-label="Remove {p}">Remove</button>
          {/if}
        </div>
      {:else}<span class="small muted">None approved.</span>{/each}
    </div>
    {#if !dirs.available}
      <p class="note warn">{dirs.reason ?? 'This worker does not support directory changes in Settings.'}</p>
    {:else}
      <form class="stack" onsubmit={submit}>
        {#if action !== 'remove'}
          <div class="field">
            <label for="{uid}-path">{action === 'replace' ? 'Replacement directory' : 'Add a directory'}</label>
            <input bind:this={input} id="{uid}-path" class="input mono" bind:value={path} onblur={() => touched = true} disabled={busy} autocomplete="off" spellcheck="false" placeholder="/path/to/models" aria-invalid={error ? 'true' : undefined} aria-describedby={error ? `${uid}-error` : `${uid}-hint`} />
            <FieldError id="{uid}-error" {error} />
            {#if !error}<span class="hint" id="{uid}-hint">Enter the absolute path to a model directory on this worker.</span>{/if}
          </div>
        {/if}
        {#if path && !problem}
          <div class="note small">
            <strong>{action==='add'?'Approve read access':action==='replace'?'Replace read access':'Remove read access'} for <code>{host.id}</code></strong>
            {#if action!=='add'}<div>Remove: <code class="path">{previous}</code></div>{/if}
            {#if action!=='remove'}<div>Approve: <code class="path">{path}</code></div>{/if}
          </div>
        {/if}
        <div class="row">
          <button bind:this={saveButton} type="submit" class="btn primary" disabled={busy || !!problem}>Save directory change</button>
          {#if action!=='add' || path}<button type="button" class="btn" disabled={busy} onclick={()=>draft('add')}>Cancel</button>{/if}
        </div>
      </form>
    {/if}
  {/if}
</section>
<style>
  .directories {padding-bottom:var(--s-4);border-bottom:1px solid var(--line-soft);}
  .path {overflow-wrap:anywhere;white-space:pre-wrap;min-width:0;}
</style>

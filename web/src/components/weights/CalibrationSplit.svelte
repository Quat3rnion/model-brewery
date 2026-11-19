<script lang="ts">
  import { onDestroy } from 'svelte';
  import { brewery } from '../../lib/store.svelte';
  import { modelLabel } from '../../lib/format';
  import { uploadBatch } from '../../lib/weightCalibration';
  import type { Host } from '../../lib/types';

  let { label, host, other, value = $bindable('') }: { label: string; host: Host | undefined; other: string; value?: string } = $props();
  const options = $derived((brewery.state?.models ?? []).filter(m => host && m.host === host.id && m.available !== false && m.kind === 'weight-calibration' && m.id !== other));
  const uploadable = $derived(host?.id === 'backend-hf' && host.capabilities?.weight_tune === true && !host.scheduling_paused);
  let busy = $state(false);
  let error = $state('');
  let live = true;
  onDestroy(() => { live = false; });
  async function upload(input: HTMLInputElement) {
    const file = input.files?.[0];
    if (!file || busy || !uploadable) return;
    busy = true;
    error = '';
    try {
      const id = await uploadBatch(file);
      await brewery.refresh();
      if (live) value = id;
    } catch (e) { error = e instanceof Error ? e.message : 'Upload failed.'; }
    finally { input.value = ''; busy = false; }
  }
</script>

<div class="field">
  <label>{label}
    <select class="input" bind:value disabled={busy || !options.length}>
      <option value="">{options.length ? 'Choose a registered split' : 'No matching splits on this worker'}</option>
      {#each options as m (m.id)}<option value={m.id}>{modelLabel(m)}</option>{/each}
    </select>
  </label>
  {#if uploadable}
    <label class="small">Upload {label.toLowerCase()} JSON
      <input class="input" type="file" accept=".json,application/json" disabled={busy} onchange={e => upload(e.currentTarget)} />
    </label>
    {#if busy}<span role="status">Registering split…</span>{/if}
  {/if}
  {#if error}<span class="note bad" role="alert">{error}</span>{/if}
</div>

<style>
  label { display: grid; gap: var(--s-1); min-width: 0; }
  .input { max-width: 100%; min-width: 0; }
</style>

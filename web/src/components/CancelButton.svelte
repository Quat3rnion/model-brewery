<script lang="ts">
  import { api, errorText } from '../lib/api';
  import { brewery } from '../lib/store.svelte';
  import { isActive, isCancelling } from '../lib/format';
  import type { JobSummary } from '../lib/types';

  let { job, small = true }: { job: JobSummary; small?: boolean } = $props();
  let confirming = $state(false);
  let busy = $state(false);
  let error = $state<string | null>(null);
  const can = $derived(isActive(job.state) && !isCancelling(job));

  async function cancel() {
    busy = true;
    error = null;
    try {
      await api.cancelJob(job.id);
      confirming = false;
      await brewery.refresh();
    } catch (e) {
      error = errorText(e);
    } finally {
      busy = false;
    }
  }
</script>

{#if can}
  <span class="cancel">
    {#if confirming}
      <button type="button" class="btn danger {small ? 'small' : ''}" disabled={busy} onclick={cancel}>{busy ? 'Cancelling…' : 'Confirm cancel'}</button>
      <button type="button" class="btn ghost {small ? 'small' : ''}" disabled={busy} onclick={() => (confirming = false)}>Keep</button>
    {:else}
      <button type="button" class="btn danger {small ? 'small' : ''}" onclick={() => (confirming = true)} aria-label="Cancel job {job.id}">Cancel</button>
    {/if}
    {#if error}<span class="xs err" role="alert">{error}</span>{/if}
  </span>
{:else if isCancelling(job)}
  <span class="xs muted">Waiting for the job to stop.</span>
{/if}

<style>
  .cancel {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }
  .err {
    color: var(--bad);
  }
</style>

<script lang="ts">
  import type { Submitter } from '../lib/submit.svelte';
  import JobLink from './JobLink.svelte';
  let { s, replayOnly = false }: { s: Submitter; replayOnly?: boolean } = $props();
</script>

<div aria-live="polite" class="xs">
  {#if s.error}
    <p class="note bad" role="alert">{s.error}</p>
  {:else if s.job && (!replayOnly || s.replayed)}
    <p class="muted">{s.replayed ? 'This request already queued job' : 'Queued as job'} <JobLink id={s.job.id} />. {s.replayed ? 'No second job was queued.' : 'It runs when the queue reaches it.'}</p>
  {/if}
</div>

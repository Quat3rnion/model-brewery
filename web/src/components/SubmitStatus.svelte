<script lang="ts">
  import type { Submitter } from '../lib/submit.svelte';
  import JobLink from './JobLink.svelte';
  let { s }: { s: Submitter } = $props();
</script>

<div aria-live="polite" class="xs">
  {#if s.error}
    <p class="note bad" role="alert">{s.error}</p>
  {:else if s.job}
    <p class="muted">Queued as job <JobLink id={s.job.id} />. It runs when the queue reaches it.</p>
  {/if}
</div>

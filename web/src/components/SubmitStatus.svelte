<script lang="ts">
  import type { Submitter } from '../lib/submit.svelte';
  import JobLink from './JobLink.svelte';
  let { s, resultLinkLabel }: { s: Submitter; resultLinkLabel?: string } = $props();
</script>

<div aria-live="polite" class="xs">
  {#if s.error}
    <p class="note bad" role="alert">{s.error}</p>
  {:else if s.job}
    <p class="muted">{s.replayed ? 'This request was already queued as job' : 'Queued as job'} <JobLink id={s.job.id} label={resultLinkLabel} />. {s.replayed ? 'No second job was queued.' : 'It runs when the queue reaches it.'}</p>
  {/if}
</div>

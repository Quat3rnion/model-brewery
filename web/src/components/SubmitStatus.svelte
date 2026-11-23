<script lang="ts">
  import type { Submitter } from '../lib/submit.svelte';
  import JobLink from './JobLink.svelte';
  import NewRun from './NewRun.svelte';
  let { s, previous = false, prepare, resultLinkLabel }: {
    s: Submitter; previous?: boolean; prepare?: () => boolean; resultLinkLabel?: string;
  } = $props();
  const openPlain = $derived(prepare ?? s.newRun);
</script>

<div aria-live="polite" class="xs">
  {#if s.error}
    <p class="note bad" role="alert">{s.error}</p>
  {:else if s.job}
    {#if previous}<p class="muted">Result of the previous submission</p>{/if}
    <p class="muted">{s.replayed ? 'This request was already queued as job' : 'Queued as job'} <JobLink id={s.job.id} label={resultLinkLabel} />. {s.replayed ? 'No second job was queued.' : 'It runs when the queue reaches it.'}</p>
  {/if}
</div>
{#if !s.error && s.replayed && openPlain}
  <NewRun prepare={openPlain} disabled={s.busy} />
{/if}

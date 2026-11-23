<script lang="ts">
  import type { Submitter } from '../lib/submit.svelte';
  import JobLink from './JobLink.svelte';
  let { s, resultLinkLabel, allowRepeat = true }: { s: Submitter; resultLinkLabel?: string; allowRepeat?: boolean } = $props();
  let element: HTMLElement | undefined = $state();
  $effect(() => {
    const submitter = s;
    const status = element;
    submitter.statusElement = status;
    return () => {
      if (submitter.statusElement === status) submitter.statusElement = undefined;
    };
  });
</script>

<div aria-live="polite" class="xs" bind:this={element}>
  {#if s.error}
    <p class="note bad" role="alert" tabindex="-1">{s.error}</p>
  {:else if s.job}
    <p class="muted">{s.repeat ? 'This request is already queued as job' : s.replayed ? 'This request was already queued as job' : 'Queued as job'} <JobLink id={s.job.id} label={resultLinkLabel} />. {s.replayed ? 'No second job was queued.' : 'It runs when the queue reaches it.'}</p>
  {/if}
  {#if s.repeat && allowRepeat}
    <button type="button" class="btn small" disabled={s.busy} onclick={s.repeat}>Queue another</button>
  {/if}
</div>

<script lang="ts">
  import { stripEmpty, type Submitter } from '../lib/submit.svelte';
  import type { NewJob } from '../lib/types';
  import { bodyKey, resultOfPreviousSubmission } from '../lib/retry-keys';
  import JobLink from './JobLink.svelte';
  import NewRun from './NewRun.svelte';
  import ResultAttribution from './ResultAttribution.svelte';
  let { s, current, prepare, allowRepeat = true, resultLinkLabel }: {
    s: Submitter; current: NewJob | null | undefined; prepare?: () => boolean; allowRepeat?: boolean; resultLinkLabel?: string;
  } = $props();
  const openPlain = $derived(prepare ?? (s.newRun ? () => s.newRun?.('submitted') ?? false : undefined));
  const currentBody = $derived(current && stripEmpty(current));
  const accepted = $derived(bodyKey(s.acceptedBody));
</script>

<div aria-live="polite" class="xs">
  {#if s.error}
    <p class="note bad" role="alert">{s.error}</p>
  {:else if s.job}
    <ResultAttribution {accepted} current={currentBody} />
    <p class="muted">{s.replayed ? 'This request was already queued as job' : 'Queued as job'} <JobLink id={s.job.id} label={resultLinkLabel} />. {s.replayed ? 'No second job was queued.' : 'It runs when the queue reaches it.'}</p>
  {/if}
</div>
{#if !s.error && s.replayed && openPlain && allowRepeat}
  <NewRun prepare={openPlain} disabled={s.busy} changed={resultOfPreviousSubmission(accepted, currentBody)}
    current={!prepare && s.newRun ? () => s.newRun?.('current') ?? false : undefined} />
{/if}

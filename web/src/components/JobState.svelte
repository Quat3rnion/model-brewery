<script lang="ts">
  import Badge from './Badge.svelte';
  import { isCancelling, stateLabel, stateTone } from '../lib/format';
  import type { JobSummary } from '../lib/types';
  let { job }: { job: Pick<JobSummary, 'state' | 'cancel_requested'> } = $props();
  const cancelling = $derived(isCancelling(job));
</script>

<Badge tone={cancelling ? 'warn' : stateTone(job.state)} label={cancelling ? 'Cancelling' : stateLabel(job.state)} />

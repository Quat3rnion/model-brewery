<script lang="ts">
  import Badge from './Badge.svelte';
  import { stateLabel, stateTone } from '../lib/format';
  import type { JobSummary } from '../lib/types';
  let { job }: { job: Pick<JobSummary, 'state' | 'cancel_requested'> } = $props();
  const cancelling = $derived(job.cancel_requested && (job.state === 'running' || job.state === 'queued'));
</script>

<Badge tone={cancelling ? 'warn' : stateTone(job.state)} label={cancelling ? 'Cancelling' : stateLabel(job.state)} />

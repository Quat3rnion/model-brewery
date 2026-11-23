<script lang="ts">
  import { resultOfPreviousSubmission } from '../lib/retry-keys';
  let { accepted, current }: { accepted: string | undefined; current: unknown } = $props();
</script>

{#if resultOfPreviousSubmission(accepted, current)}
  <p class="muted">Result of the previous submission</p>
  <p class="hint">The current form has different inputs.</p>
{/if}

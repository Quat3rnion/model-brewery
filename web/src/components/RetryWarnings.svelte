<script lang="ts">
  import type { RetryOmission } from '../lib/retry-prefill.svelte';

  let { items, acknowledge, acknowledged = false, label = 'Acknowledge omitted inputs' }: {
    items: readonly RetryOmission[];
    acknowledge?: () => void;
    acknowledged?: boolean;
    label?: string;
  } = $props();
</script>

{#if items.length}
  <div class="note warn break">
    <div role="status">
      {#each items as item (item.id)}
        <p>{item.text}</p>
      {/each}
      {#if acknowledged}<p>Omissions acknowledged.</p>{/if}
    </div>
    {#if acknowledge && !acknowledged && !items.some(item => item.required)}
      <button type="button" class="btn" onclick={acknowledge}>{label}</button>
    {/if}
  </div>
{/if}

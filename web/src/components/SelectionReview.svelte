<script lang="ts">
  let { missing, acknowledge }: {
    missing: readonly { id: string; label: string }[];
    acknowledge: () => void;
  } = $props();
</script>

{#if missing.length}
  <div class="note warn stack">
    <div role="alert">
      <p>These selections are no longer available:</p>
      <ul>
        {#each missing as item (item.id)}<li>{item.label}</li>{/each}
      </ul>
      <p>Check the remaining selection before you continue.</p>
    </div>
    <div><button type="button" class="btn" onclick={acknowledge}>Keep remaining selection</button></div>
  </div>
{/if}

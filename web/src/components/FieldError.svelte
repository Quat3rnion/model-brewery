<script lang="ts">
  let { id, error }: { id: string; error?: string } = $props();
</script>

{#if error}<span class="xs" {id}>{error}</span>{/if}

<style>
  span { color: var(--warn); }
</style>

<script lang="ts">
  import { focusAfterRemoval } from '../lib/focus';

  type Omission = { id: string; label: string };
  let { missing, acknowledge, focusRemaining }: {
    missing: readonly Omission[];
    acknowledge: (shown: readonly Omission[]) => void;
    focusRemaining: () => HTMLElement | null | undefined;
  } = $props();
</script>

{#if missing.length}
  <!-- Replace the activation target when the IDs change. An in-progress press
       on the old target cannot acknowledge the newly rendered omissions. -->
  {#each [missing] as shown (JSON.stringify(shown.map(item => item.id).sort()))}
    <div class="note warn stack">
      <div role="alert">
        <p>These selections are no longer available:</p>
        <ul>
          {#each shown as item (item.id)}<li>{item.label}</li>{/each}
        </ul>
        <p>Check the remaining selection before you continue.</p>
      </div>
      <div><button type="button" class="btn" onclick={() => focusAfterRemoval(() => acknowledge(shown), focusRemaining)}>Keep remaining selection</button></div>
    </div>
  {/each}
{/if}

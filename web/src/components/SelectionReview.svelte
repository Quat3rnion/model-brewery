<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { focusAfterRemoval } from '../lib/focus';

  type Omission = { id: string; label: string };
  let { missing, acknowledge, focusRemaining }: {
    missing: readonly Omission[];
    acknowledge: (shown: readonly Omission[]) => void;
    focusRemaining: () => HTMLElement | null | undefined;
  } = $props();

  const ids = $derived(JSON.stringify(missing.map(item => item.id).sort()));
  let reviewButton: HTMLButtonElement | undefined = $state();
  let spaceTarget: HTMLButtonElement | undefined;
  $effect.pre(() => {
    void ids;
    const previous = untrack(() => reviewButton);
    if (previous !== document.activeElement) return;
    void tick().then(() => {
      if (!previous?.isConnected && document.activeElement === document.body) {
        (reviewButton ?? focusRemaining())?.focus();
      }
    });
  });

  function press(event: KeyboardEvent & { currentTarget: HTMLButtonElement }) {
    if (event.key !== ' ' && event.key !== 'Enter') return;
    if (event.repeat) event.preventDefault();
    else if (event.key === ' ') spaceTarget = event.currentTarget;
  }
  function release(event: KeyboardEvent & { currentTarget: HTMLButtonElement }) {
    if (event.key !== ' ') return;
    if (spaceTarget !== event.currentTarget) event.preventDefault();
    spaceTarget = undefined;
  }
</script>

{#if missing.length}
  <!-- Replace the activation target when the IDs change. An in-progress press
       on the old target cannot acknowledge the newly rendered omissions. -->
  {#each [missing] as shown (ids)}
    <div class="note warn stack">
      <div role="alert">
        <p>These selections are no longer available:</p>
        <ul>
          {#each shown as item (item.id)}<li>{item.label}</li>{/each}
        </ul>
        <p>Check the remaining selection before you continue.</p>
      </div>
      <div><button bind:this={reviewButton} type="button" class="btn" onkeydown={press} onkeyup={release}
        onclick={() => focusAfterRemoval(() => acknowledge(shown), focusRemaining)}>Keep remaining selection</button></div>
    </div>
  {/each}
{/if}

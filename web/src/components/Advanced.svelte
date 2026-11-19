<script lang="ts">
  // Disclosure for optional, rarely changed settings. The summary always
  // names the values that differ from the defaults, so a changed setting is
  // never hidden. It opens on its own when something is changed.
  import type { Snippet } from 'svelte';

  let { title = 'Advanced options', changed = [], children }: { title?: string; changed?: string[]; children: Snippet } = $props();
  let open = $state(false);
  let seeded = false;
  $effect(() => {
    if (!seeded && changed.length) {
      seeded = true;
      open = true;
    }
  });
</script>

<details class="disclosure" bind:open>
  <summary>
    <span class="t">{title}</span>
    <span class="s xs">{changed.length ? `Changed: ${changed.join(' · ')}` : 'Defaults'}</span>
  </summary>
  <div class="content">{@render children()}</div>
</details>

<style>
  .t {
    font-weight: 500;
    font-size: var(--fs-sm);
  }
  .s {
    color: var(--text-2);
    overflow-wrap: anywhere;
  }
  .content {
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
    padding: var(--s-3) 0 var(--s-1);
  }
</style>

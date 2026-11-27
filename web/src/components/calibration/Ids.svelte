<script lang="ts">
  import { shortHash } from '../../lib/format';
  let { title, items, compact = true }: { title: string; items: [string, string][]; compact?: boolean } = $props();
  const isHash = (v: string) => /^[0-9a-f]{40,}$/i.test(v);
</script>

<div class="ids">
  <h3 class="h">{title}</h3>
  {#if items.length === 0}
    <p class="xs muted">Not stated.</p>
  {:else}
    <dl class="xs">
      {#each items as [k, v]}
        <dt class="break">{k.replace(/_/g, ' ')}</dt>
        <dd class:mono={isHash(v)} class="break" title={compact && isHash(v) ? v : undefined}>{compact && isHash(v) ? shortHash(v) : v}</dd>
      {/each}
    </dl>
  {/if}
</div>

<style>
  .ids {
    min-width: 0;
  }
  .h {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
    margin-bottom: 4px;
  }
  dl {
    display: grid;
    grid-template-columns: minmax(0, max-content) minmax(50%, 1fr);
    gap: 2px var(--s-3);
    margin: 0;
  }
  dt {
    color: var(--text-2);
  }
  dd {
    margin: 0;
  }
</style>

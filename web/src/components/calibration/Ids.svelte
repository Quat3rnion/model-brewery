<script lang="ts">
  import { shortHash } from '../../lib/format';
  let { title, items }: { title: string; items: [string, string][] } = $props();
  const isHash = (v: string) => /^[0-9a-f]{40,}$/i.test(v);
</script>

<div class="ids">
  <h4 class="h">{title}</h4>
  {#if items.length === 0}
    <p class="xs muted">Not stated.</p>
  {:else}
    <dl class="xs">
      {#each items as [k, v] (k)}
        <dt>{k.replace(/_/g, ' ')}</dt>
        <dd class:mono={isHash(v)} class="break" title={isHash(v) ? v : undefined}>{isHash(v) ? shortHash(v) : v}</dd>
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
    grid-template-columns: max-content minmax(0, 1fr);
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

<script lang="ts">
  let { data, skip = [] }: { data: unknown; skip?: string[] } = $props();
  const entries = $derived(
    data && typeof data === 'object' && !Array.isArray(data)
      ? Object.entries(data as Record<string, unknown>).filter(([k]) => !skip.includes(k))
      : [],
  );
  function scalar(v: unknown): v is string | number | boolean | null {
    return v === null || ['string', 'number', 'boolean'].includes(typeof v);
  }
  function label(k: string) {
    return k.replace(/_/g, ' ');
  }
</script>

{#if entries.length}
  <dl class="kv">
    {#each entries as [k, v] (k)}
      <dt>{label(k)}</dt>
      <dd class="break">
        {#if scalar(v)}
          <span class={typeof v === 'string' && v.length > 24 && !v.includes(' ') ? 'mono' : ''}>{v === null ? '—' : String(v)}</span>
        {:else}
          <details><summary class="xs">{Array.isArray(v) ? `${v.length} items` : 'object'}</summary><pre class="block">{JSON.stringify(v, null, 2)}</pre></details>
        {/if}
      </dd>
    {/each}
  </dl>
{:else if data !== undefined && data !== null}
  <pre class="block">{JSON.stringify(data, null, 2)}</pre>
{/if}

<style>
  .kv {
    display: grid;
    grid-template-columns: minmax(96px, max-content) minmax(0, 1fr);
    gap: 4px var(--s-3);
    margin: 0;
    font-size: var(--fs-sm);
  }
  dt {
    color: var(--text-2);
  }
  dd {
    margin: 0;
    min-width: 0;
  }
</style>

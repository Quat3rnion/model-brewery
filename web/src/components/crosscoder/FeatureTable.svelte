<script lang="ts">
  // One filterable, paged feature table of a crosscoder report.
  import { CLASS_INFO, SORT_LABELS, filterFeatures, type CrosscoderFeature, type FeatureClass, type FeatureSort } from '../../lib/crosscoder';
  import { fmt } from '../../lib/weightspace';
  import { focusAfterRemoval } from '../../lib/focus';
  import Badge from '../Badge.svelte';

  const PAGE = 50;
  let {
    features,
    cls,
    caption,
    sort: initialSort = 'strength',
  }: { features: readonly CrosscoderFeature[]; cls: FeatureClass | 'all'; caption: string; sort?: FeatureSort } = $props();
  const uid = $props.id();

  let query = $state('');
  // svelte-ignore state_referenced_locally
  let sort = $state<FeatureSort>(initialSort);
  let limit = $state(PAGE);
  let wrap = $state<HTMLElement | null>(null);
  const rows = $derived(filterFeatures(features, cls, query, sort));
  const shown = $derived(rows.slice(0, limit));
  $effect(() => {
    void query;
    void sort;
    limit = PAGE;
  });
</script>

<div class="stack tight">
  <div class="controls">
    <div class="field">
      <label for="{uid}-q">Feature number</label>
      <input id="{uid}-q" class="input num" type="search" inputmode="numeric" autocomplete="off" bind:value={query} aria-describedby="{uid}-count" />
    </div>
    <div class="field">
      <label for="{uid}-sort">Sort</label>
      <select id="{uid}-sort" class="input" bind:value={sort}>
        {#each Object.entries(SORT_LABELS) as [value, label] (value)}<option {value}>{label}</option>{/each}
      </select>
    </div>
  </div>
  <p class="xs muted" id="{uid}-count" aria-live="polite">
    Showing {shown.length} of {rows.length}.
  </p>
  {#if !rows.length}
    <p class="empty">{query.trim() ? 'No feature matches the filter.' : 'No features in this class.'}</p>
  {:else}
    <!-- A focusable scroll area: keyboard users can scroll long tables. -->
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <div class="table-wrap tall" bind:this={wrap} tabindex="0" role="region" aria-label="{caption}, scrollable">
      <table class="data num stackable">
        <caption class="sr-only">{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Feature</th>
            {#if cls === 'all'}<th scope="col">Class</th>{/if}
            <th scope="col">Parent norm</th>
            <th scope="col">Child norm</th>
            <th scope="col">Child share <span class="xs muted">child / (parent + child)</span></th>
            <th scope="col">Decoder cosine</th>
          </tr>
        </thead>
        <tbody>
          {#each shown as f (f.feature)}
            <tr>
              <th scope="row">{f.feature}</th>
              {#if cls === 'all'}<td data-label="Class"><Badge tone={CLASS_INFO[f.classification].tone} label={CLASS_INFO[f.classification].label} /></td>{/if}
              <td data-label="Parent norm" title={String(f.parent_norm)}>{fmt(f.parent_norm)}</td>
              <td data-label="Child norm" title={String(f.child_norm)}>{fmt(f.child_norm)}</td>
              <td data-label="Child share" title={f.child_norm_ratio === null ? 'No value: both decoder norms are zero.' : String(f.child_norm_ratio)}>{fmt(f.child_norm_ratio)}</td>
              <td data-label="Decoder cosine" title={f.decoder_cosine === null ? 'No cosine: a decoder norm is zero.' : String(f.decoder_cosine)}>{fmt(f.decoder_cosine)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    {#if rows.length > shown.length}
      <div><button type="button" class="btn small" onclick={() => focusAfterRemoval(() => (limit += PAGE), () => wrap)}>Show {Math.min(PAGE, rows.length - shown.length)} more</button></div>
    {/if}
  {/if}
</div>

<style>
  .tight {
    gap: var(--s-2);
  }
  .controls {
    display: grid;
    grid-template-columns: minmax(0, 10rem) minmax(0, 14rem);
    gap: var(--s-3);
  }
  @media (max-width: 480px) {
    .controls {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .tall {
    max-height: 420px;
  }
</style>

<script lang="ts">
  import { MAX_EXPERTS_SHOWN, MAX_LAYER_ROWS, fmtExperts, fmtSignal, readSignals } from '../../lib/signals';
  import type { TokenRow } from '../../lib/trace';

  let { raw, rows, label }: { raw: unknown; rows: TokenRow[]; label: string } = $props();
  const uid = $props.id();
  const signals = $derived(readSignals(raw));
  let chosen = $state<number | null>(null);
  const position = $derived(chosen !== null && signals?.positions.includes(chosen) ? chosen : (signals?.positions[0] ?? null));
  const allRows = $derived(signals && position !== null ? signals.rowsAt(position) : []);
  const shown = $derived(allRows.slice(0, MAX_LAYER_ROWS));
  const missingHere = $derived(allRows.filter((r) => !r.cell).length);
  const hasMoe = $derived(allRows.some((r) => r.cell?.experts));
  const tokenOf = (p: number) => rows.find((r) => r.pos === p);
  const cov = $derived(signals?.coverage);
  const spanText = (s: [number, number] | null | undefined) => (s ? `${s[0]}–${s[1]}` : 'none');
</script>

<div class="stack sig">
  {#if raw === undefined || raw === null}
    <p class="muted small">This result has no layer signals. Older receipts show tokens only.</p>
  {:else if !signals}
    <p class="note warn">The result has a signals record in a format this UI does not read. See the raw result JSON.</p>
  {:else}
    {#if cov}
      <p class="xs muted num">
        Captured {cov.captured_cells ?? '?'} cells, shown {cov.included_cells ?? '?'}. Layers {spanText(cov.layer_span)} of {cov.total_layers ?? '?'};
        positions {spanText(cov.token_span)} of {cov.total_tokens ?? '?'}.
      </p>
    {/if}
    {#if cov?.truncated}
      <p class="note warn">The engine capped the signal record. Layers or positions outside the spans above were not included.</p>
    {/if}
    {#if signals.positions.length === 0}
      <p class="muted small">The signal record has no cells.</p>
    {:else}
      <div class="field">
        <label for="{uid}-pos">Position for {label}</label>
        <select id="{uid}-pos" class="input" value={position} onchange={(e) => (chosen = Number(e.currentTarget.value))}>
          {#each signals.positions as p (p)}
            {@const t = tokenOf(p)}
            <option value={p}>{p}{t ? ` · ${JSON.stringify(t.text)} (${t.phase || '?'})` : ''}</option>
          {/each}
        </select>
      </div>
      <!-- Scrollable region must take keyboard focus so keyboard users can scroll it. -->
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <div class="table-wrap rows" role="region" aria-label="Layer signals of {label}, scrollable" tabindex="0">
        <table class="data">
          <caption class="sr-only">Layer signals of {label} at position {position}</caption>
          <thead>
            <tr>
              <th scope="col">Layer</th>
              <th scope="col">Attention path</th>
              <th scope="col" class="r">Attention write RMS</th>
              <th scope="col" class="r">FFN write RMS</th>
              {#if hasMoe}<th scope="col">Experts (weight, native order)</th>{/if}
            </tr>
          </thead>
          <tbody>
            {#each shown as { layer, cell: c } (layer)}
              {#if !c}
                <tr class="gap">
                  <td class="num mono">{layer}</td>
                  <td colspan={hasMoe ? 4 : 3} class="xs muted">No native cell at this position</td>
                </tr>
              {:else}
              <tr>
                <td class="num mono">{c.layer}</td>
                <td class="xs">{c.kind}</td>
                <td class="num mono r">{#if c.attn === null}<span class="nc" title="Not captured">—</span>{:else}{fmtSignal(c.attn)}{/if}</td>
                <td class="num mono r">{#if c.ffn === null}<span class="nc" title="Not captured">—</span>{:else}{fmtSignal(c.ffn)}{/if}</td>
                {#if hasMoe}<td class="mono xs ex">{#if c.experts}{fmtExperts(c.experts, c.weights)}{:else}<span class="nc" title="Not captured">—</span>{/if}</td>{/if}
              </tr>
              {/if}
            {/each}
          </tbody>
        </table>
      </div>
      <p class="xs muted">
        "—" means not captured, not zero. The UI shows up to {MAX_LAYER_ROWS} layers and {MAX_EXPERTS_SHOWN} experts per cell.
        {#if missingHere}{missingHere} recorded layer{missingHere === 1 ? ' has' : 's have'} no native cell at this position.{/if}
        {#if allRows.length > shown.length}<strong>{allRows.length - shown.length} more layers are hidden at this position.</strong>{/if}
      </p>
    {/if}
    {#if signals.omissions}
      {@const o = signals.omissions}
      {@const u = (o.unavailable && typeof o.unavailable === 'object' ? o.unavailable : {}) as Record<string, unknown>}
      <details>
        <summary class="xs">Omissions declared by the engine</summary>
        <dl class="om xs num">
          {#if typeof o.missing_cells === 'number'}<dt>No native cell</dt><dd>{o.missing_cells}</dd>{/if}
          {#if typeof o.skipped_cells === 'number'}<dt>Outside the bounded view</dt><dd>{o.skipped_cells}</dd>{/if}
          {#each ['attn', 'ffn', 'experts', 'weights'] as k (k)}
            {#if typeof u[k] === 'number'}<dt>Shown cells without {k}</dt><dd>{u[k]}</dd>{/if}
          {/each}
          {#if Array.isArray(o.inline_excluded) && o.inline_excluded.length}<dt>Excluded from inline record</dt><dd class="break">{o.inline_excluded.join(', ')}</dd>{/if}
        </dl>
        <pre class="block">{JSON.stringify(o, null, 2)}</pre>
      </details>
    {/if}
  {/if}
</div>

<style>
  .rows {
    max-height: 360px;
    overflow: auto;
  }
  .rows table {
    min-width: 520px;
  }
  .ex {
    white-space: nowrap;
  }
  .r {
    text-align: right;
  }
  .om {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 2px var(--s-3);
    margin: 6px 0;
  }
  .om dt {
    color: var(--text-2);
  }
  .om dd {
    margin: 0;
  }
  tr.gap td {
    font-style: italic;
  }
  .nc {
    color: var(--muted);
  }
</style>

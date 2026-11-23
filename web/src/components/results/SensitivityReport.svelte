<script lang="ts">
  import { obj } from '../../lib/reports';

  let { result }: { result: Record<string, unknown> } = $props();
  const tensors = $derived((Array.isArray(result.tensors) ? result.tensors : []).flatMap((value) => {
    const tensor = obj(value);
    if (!tensor || typeof tensor.name !== 'string' || typeof tensor.fisher !== 'number'
      || !Number.isFinite(tensor.fisher) || tensor.fisher < 0) return [];
    const layer = typeof tensor.layer === 'number' ? tensor.layer : null;
    return [{
      layer, fisher: tensor.fisher,
      column: layer === null ? tensor.name : tensor.name.replace(/^blk\.\d+\./, ''),
    }];
  }));
  const columns = $derived([...new Set(tensors.map((t) => t.column))].sort());
  const layers = $derived([...new Set(tensors.map((t) => t.layer))].sort((a, b) => (a ?? Infinity) - (b ?? Infinity)));
  const totals = $derived(new Map((Array.isArray(result.layers) ? result.layers : []).flatMap((value) => {
    const layer = obj(value);
    return layer && typeof layer.layer === 'number' && typeof layer.fisher === 'number'
      ? [[layer.layer, layer.fisher] as const] : [];
  })));
  const peak = $derived(Math.max(0, ...tensors.map((t) => t.fisher)));
  const tint = (value: number) => peak > 0 ? `${Math.sqrt(value / peak) * 22}%` : '0%';
</script>

<section class="stack" aria-labelledby="sensitivity-h">
  <h3 id="sensitivity-h">Tensor sensitivity</h3>
  <p class="small">Not release-qualified</p>
  <p class="small muted">
    Empirical Fisher is the mean of squared gradients over the measured tokens and tensor elements.
    Stronger accent shading indicates a larger value within this job. Layer means are weighted by the
    number of measured elements.
  </p>
  <dl class="kv small">
    <dt>Backend</dt><dd>{String(result.backend).toUpperCase()}</dd>
    <dt>Observations</dt><dd class="num">{String(result.observation_count)}</dd>
    <dt>Revision SHA-256</dt><dd class="mono break">{String(result.revision_sha256)}</dd>
    {#if result.model_sha256 !== result.revision_sha256}
      <dt>GGUF SHA-256</dt><dd class="mono break">{String(result.model_sha256)}</dd>
    {/if}
    <dt>W4 corpus SHA-256</dt><dd class="mono break">{String(result.corpus_sha256)}</dd>
  </dl>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users must be able to scroll this table.) -->
  <div class="table-wrap" role="region" aria-label="Fisher heatmap scroll area" tabindex="0">
    <table class="data">
      <caption class="sr-only">Empirical Fisher by layer and tensor</caption>
      <thead>
        <tr>
          <th scope="col">Layer</th>
          {#each columns as column (column)}<th scope="col" class="mono">{column}</th>{/each}
          <th scope="col">Layer mean</th>
        </tr>
      </thead>
      <tbody>
        {#each layers as layer (layer)}
          <tr>
            <th scope="row">{layer ?? 'No layer'}</th>
            {#each columns as column (column)}
              {@const tensor = tensors.find((t) => t.layer === layer && t.column === column)}
              {#if tensor}
                <td class="num heat" style:--heat={tint(tensor.fisher)}>{tensor.fisher}</td>
              {:else}
                <td class="muted">Not measured</td>
              {/if}
            {/each}
            <td class="num">{layer === null ? 'Not applicable' : (totals.get(layer) ?? 'Not measured')}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</section>

<style>
  .kv {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 4px var(--s-3);
    margin: 0;
  }
  .kv dt { color: var(--text-2); }
  .kv dd { margin: 0; }
  th, td { white-space: nowrap; }
  .heat { background-color: color-mix(in srgb, var(--accent) var(--heat), var(--surface)); }
</style>

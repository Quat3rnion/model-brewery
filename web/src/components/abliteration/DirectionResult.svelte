<script lang="ts">
  import { behaviorNames, contrastSources, score, type DirectionArtifact } from '../../lib/abliteration';
  import Plot from './Plot.svelte';

  let { artifact }: { artifact: DirectionArtifact } = $props();
  const d = $derived(artifact.direction);
  const rows = $derived(d.fit.layers);
  const source = $derived(contrastSources[d.contrast_sha256]);
  const metrics = [
    { label: "Held-out Cohen's d", key: 'held_out_cohens_d' },
    { label: 'Held-out margin', key: 'held_out_margin' },
    { label: 'Direction norm', key: 'norm' },
    { label: 'Mean residual norm', key: 'mean_residual_norm' },
  ] as const;
  const show = (n: number | null | undefined) => n == null ? 'Not measured' : score(n);
</script>

<section class="panel" aria-labelledby="direction-title">
  <header><h2 id="direction-title">{behaviorNames[d.behavior] ?? d.behavior} direction</h2><span class="small">Version {artifact.version}</span></header>
  <div class="body stack">
    <p class="note"><strong>Selected layer: {d.fit.selected_layer ?? 'None'}.</strong>
      {d.fit.train_pairs} fit pairs and {d.fit.held_out_pairs} holdout pairs. These values measure separation, not a change in generated behavior.</p>
    <div class="grid-2">
      {#each metrics as m (m.key)}
        <Plot label={m.label} axis="Layer" curve points={rows.map(r => ({
          x: r.layer, y: r[m.key], marked: r.layer === d.fit.selected_layer,
          label: `Layer ${r.layer}: ${show(r[m.key])}`,
        }))} />
      {/each}
    </div>
    <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard users must be able to scroll the table) -->
    <div class="table-wrap" tabindex="0" role="region" aria-label="Direction layer values">
      <table class="data" aria-label="Direction layer scores">
        <thead><tr><th scope="col">Layer</th>{#each metrics as m (m.key)}<th scope="col">{m.label}</th>{/each}</tr></thead>
        <tbody>
          {#each rows as r (r.layer)}
            <tr><th scope="row">{r.layer}{r.layer === d.fit.selected_layer ? ' Selected' : ''}</th>
              {#each metrics as m (m.key)}<td class="num">{show(r[m.key])}</td>{/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <div>
      <h3>Contrast-set pins</h3>
      <p class="small">Derived set SHA-256: <span class="mono break">{d.contrast_sha256}</span></p>
      {#if source}
        <p class="small"><a href={source.source}>{source.name}</a> · <a href={source.licenceUrl}>{source.licence}</a></p>
        <p class="small">Source commit: <span class="mono break">{source.commit}</span></p>
        <p class="small">Source SHA-256: <span class="mono break">{source.sha256}</span></p>
      {:else}
        <p class="small muted">No source attribution for this pin.</p>
      {/if}
    </div>
    <div>
      <h3>Capture</h3>
      <p class="small">Model: <span class="mono break">{d.model_id}</span> · {d.fit.role} · {d.position === null ? 'Last user token' : `Token ${d.position}`}</p>
      <p class="small">Model SHA-256: <span class="mono break">{d.model_sha256}</span></p>
      <p class="small">Lanternfish commit: <span class="mono break">{d.lanternfish_commit}</span></p>
      {#if d.executable_sha256}<p class="small">Executable SHA-256: <span class="mono break">{d.executable_sha256}</span></p>{/if}
    </div>
  </div>
</section>

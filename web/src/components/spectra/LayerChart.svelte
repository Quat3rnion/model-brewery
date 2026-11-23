<script lang="ts">
  // One metric per layer: source (solid) and reference (dashed) lines, and a
  // delta strip under them. Layers that changed by 1% or more get a warn band
  // in the plot and a filled warn bar in the strip, so the shape and the
  // position also carry the signal. The values table repeats every number.
  import { changed, largestChange, linePath, ticks, type LayerPoint } from '../../lib/spectra';
  import { fmt } from '../../lib/weightspace';

  let { label, points, hasReference }: { label: string; points: LayerPoint[]; hasReference: boolean } = $props();

  const uid = $props.id();
  let width = $state(320);
  const H = 132, DH = 40, GAP = 10, L = 44, R = 10, T = 8, B = 18;
  const plotW = $derived(Math.max(60, width - L - R));

  const layers = $derived(points.map((p) => p.layer));
  const x0 = $derived(Math.min(...layers, 0));
  const x1 = $derived(Math.max(...layers, x0 + 1));
  const step = $derived(points.length > 1 ? plotW / (x1 - x0) : plotW);
  const x = (l: number) => L + (points.length > 1 ? ((l - x0) / (x1 - x0)) * plotW : plotW / 2);

  const values = $derived(points.flatMap((p) => [p.source, hasReference ? p.reference : null]).filter((v): v is number => v !== null));
  const pad = $derived.by(() => {
    const lo = Math.min(...values), hi = Math.max(...values);
    const span = hi - lo || Math.abs(hi) * 0.1 || 1;
    return [lo - span * 0.08, hi + span * 0.08] as const;
  });
  const yt = $derived(values.length ? ticks(pad[0], pad[1], 3) : []);
  const y = (v: number) => T + (1 - (v - pad[0]) / (pad[1] - pad[0])) * (H - T - B);

  const dmax = $derived(Math.max(1e-12, ...points.map((p) => Math.abs(p.delta ?? 0))));
  const dTop = H + GAP;
  const dMid = $derived(dTop + DH / 2);
  const dy = (d: number) => (d / dmax) * (DH / 2 - 2);
  const barW = $derived(Math.max(2, Math.min(14, step * 0.6)));

  const top = $derived(largestChange(points));
  const changedCount = $derived(points.filter((p) => changed(p.delta, p.reference)).length);
  const labelEvery = $derived(Math.max(1, Math.ceil(points.length / Math.max(1, Math.floor(plotW / 28)))));
  const pct = (p: LayerPoint) => (p.reference ? ` (${((p.delta! / Math.abs(p.reference)) * 100).toFixed(1)}%)` : '');
</script>

<figure class="chart">
  <figcaption class="cap"><span class="name">{label}</span></figcaption>
  <div class="frame" bind:clientWidth={width}>
    <svg role="img" aria-label="{label} by layer" aria-describedby="{uid}-s" width={width} height={dTop + DH + 14} viewBox="0 0 {width} {dTop + DH + 14}">
      {#each points as p (p.layer)}
        {#if changed(p.delta, p.reference)}
          <rect class="band" x={x(p.layer) - Math.max(3, step / 2)} y={T} width={Math.max(6, step)} height={H - T - B} />
        {/if}
      {/each}
      {#each yt as v (v)}
        <line class="grid" x1={L} x2={L + plotW} y1={y(v)} y2={y(v)} />
        <text class="tick" x={L - 6} y={y(v)} text-anchor="end" dominant-baseline="middle">{fmt(v, 3)}</text>
      {/each}
      {#each points as p, i (p.layer)}
        {#if i % labelEvery === 0 || i === points.length - 1}
          <text class="tick" x={x(p.layer)} y={dTop + DH + 11} text-anchor="middle">{p.layer}</text>
        {/if}
      {/each}
      {#if hasReference}
        <path class="ref" d={linePath(points.map((p) => [p.layer, p.reference]), x, y)} />
      {/if}
      <path class="src" d={linePath(points.map((p) => [p.layer, p.source]), x, y)} />
      {#each points as p (p.layer)}
        {#if p.source !== null}<circle class="dot" cx={x(p.layer)} cy={y(p.source)} r={points.length > 40 ? 1.5 : 2.5} />{/if}
      {/each}

      <text class="tick strip-label" x={L - 6} y={dMid} text-anchor="end" dominant-baseline="middle">Δ</text>
      <line class="zero" x1={L} x2={L + plotW} y1={dMid} y2={dMid} />
      {#each points as p (p.layer)}
        {#if p.delta !== null && p.delta !== 0}
          {@const h = dy(p.delta)}
          <rect class="bar" class:hot={changed(p.delta, p.reference)} x={x(p.layer) - barW / 2} y={h > 0 ? dMid - h : dMid} width={barW} height={Math.max(1, Math.abs(h))} />
        {/if}
      {/each}
      {#each points as p (p.layer)}
        <rect class="hit" x={x(p.layer) - Math.max(3, step / 2)} y={0} width={Math.max(6, step)} height={dTop + DH}>
          <title>Layer {p.layer}: source {fmt(p.source)}{hasReference ? `, reference ${fmt(p.reference)}, change ${fmt(p.delta)}` : ''}</title>
        </rect>
      {/each}
    </svg>
  </div>
  <p class="xs summary" id="{uid}-s">
    {#if !hasReference}
      No reference. The chart shows the source only.
    {:else if top}
      <strong>Largest change: layer {top.layer}</strong>, {fmt(top.delta)}{pct(top)}. {changedCount} of {points.length} layers changed by 1% or more.
    {:else}
      No layer changed by 1% or more.
    {/if}
  </p>
</figure>

<style>
  .chart {
    margin: 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-1);
  }
  .cap .name {
    font-size: var(--fs-sm);
    font-weight: 600;
  }
  .frame {
    min-width: 0;
  }
  svg {
    display: block;
    overflow: visible;
  }
  .band {
    fill: var(--warn-bg);
  }
  .grid {
    stroke: var(--line-soft);
  }
  .zero {
    stroke: var(--line-strong);
  }
  .tick {
    fill: var(--muted);
    font-size: 10px;
    font-family: var(--font-sans);
    font-variant-numeric: tabular-nums;
  }
  .strip-label {
    font-size: 11px;
  }
  .ref {
    fill: none;
    stroke: var(--text-2);
    stroke-width: 1.5;
    stroke-dasharray: 4 3;
  }
  .src {
    fill: none;
    stroke: var(--accent);
    stroke-width: 2;
    stroke-linejoin: round;
  }
  .dot {
    fill: var(--accent);
  }
  .bar {
    fill: var(--line-strong);
  }
  .bar.hot {
    fill: var(--warn);
  }
  .hit {
    fill: transparent;
  }
  .hit:hover {
    fill: var(--accent-soft);
  }
  .summary {
    color: var(--text-2);
    margin: 0;
  }
  .summary strong {
    color: var(--text);
    font-weight: 600;
  }
</style>

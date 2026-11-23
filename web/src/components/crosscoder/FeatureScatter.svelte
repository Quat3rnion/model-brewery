<script lang="ts">
  // Scatter of the child share of the decoder norm, child / (parent + child)
  // (x, 0 to 1), against the decoder cosine (y). Shape and colour both show the class. The parent view
  // gives the same values as tables.
  import { CLASSES, CLASS_INFO, type FeatureClass, type ScatterPoint } from '../../lib/crosscoder';
  import { fmt } from '../../lib/weightspace';

  let { points, label, describedby }: { points: readonly ScatterPoint[]; label: string; describedby?: string } = $props();

  let width = $state(640);
  const height = $derived(width < 480 ? 240 : 300);
  const m = { top: 12, right: 12, bottom: 40, left: 44 };
  const iw = $derived(Math.max(10, width - m.left - m.right));
  const ih = $derived(height - m.top - m.bottom);
  const sx = (x: number) => m.left + x * iw;
  const sy = (y: number) => m.top + ((1 - y) / 2) * ih;
  const xTicks = [0, 0.25, 0.5, 0.75, 1];
  const yTicks = [-1, -0.5, 0, 0.5, 1];
  const COLOR: Record<FeatureClass, string> = { parent_unique: 'var(--warn)', child_unique: 'var(--info)', changed: 'var(--bad)', shared: 'var(--ok)', dead: 'var(--muted)' };
  // Shared features are usually the most; draw them first so rarer classes stay on top.
  const ordered = $derived([...CLASSES].reverse().flatMap((c) => points.filter((p) => p.cls === c)));
  const present = $derived(CLASSES.filter((c) => points.some((p) => p.cls === c)));
</script>

{#snippet mark(cls: FeatureClass, x: number, y: number, r: number)}
  {@const shape = CLASS_INFO[cls].shape}
  {#if shape === 'circle'}<circle cx={x} cy={y} {r} />
  {:else if shape === 'square'}<rect x={x - r} y={y - r} width={2 * r} height={2 * r} />
  {:else if shape === 'triangle'}<polygon points="{x},{y - r * 1.2} {x + r * 1.1},{y + r * 0.8} {x - r * 1.1},{y + r * 0.8}" />
  {:else if shape === 'diamond'}<polygon points="{x},{y - r * 1.3} {x + r * 1.3},{y} {x},{y + r * 1.3} {x - r * 1.3},{y}" />
  {:else}<path d="M{x - r} {y - r}L{x + r} {y + r}M{x + r} {y - r}L{x - r} {y + r}" class="cross" />{/if}
{/snippet}

<figure class="plot">
  <div class="canvas" bind:clientWidth={width}>
    <svg {width} {height} role="img" aria-label={label} aria-describedby={describedby}>
      <g class="grid" aria-hidden="true">
        {#each yTicks as t (t)}
          <line x1={m.left} x2={m.left + iw} y1={sy(t)} y2={sy(t)} class:zero={t === 0} />
          <text x={m.left - 6} y={sy(t)} dy="0.32em" text-anchor="end">{t}</text>
        {/each}
        {#each xTicks as t (t)}
          <line x1={sx(t)} x2={sx(t)} y1={m.top} y2={m.top + ih} class:zero={t === 0.5} />
          <text x={sx(t)} y={m.top + ih + 14} text-anchor={t === 0 ? 'start' : t === 1 ? 'end' : 'middle'}>{t}</text>
        {/each}
        <text class="axis" x={m.left + iw / 2} y={height - 6} text-anchor="middle">Child share of decoder norm, child / (parent + child)</text>
        <text class="axis" transform="translate(11 {m.top + ih / 2}) rotate(-90)" text-anchor="middle">Decoder cosine</text>
      </g>
      <g aria-hidden="true">
        {#each ordered as p (p.feature)}
          <g class="pt" style:color={COLOR[p.cls]}>
            <title>Feature {p.feature}, {CLASS_INFO[p.cls].label}: child share {fmt(p.x)}, cosine {fmt(p.y)}</title>
            {@render mark(p.cls, sx(p.x), sy(p.y), p.cls === 'shared' ? 2.5 : 3.5)}
          </g>
        {/each}
      </g>
    </svg>
  </div>
  <figcaption>
    <ul class="legend xs" aria-label="Plot legend">
      {#each present as c (c)}
        <li>
          <svg width="12" height="12" aria-hidden="true" style:color={COLOR[c]}><g class="pt">{@render mark(c, 6, 6, 4)}</g></svg>
          {CLASS_INFO[c].label}
        </li>
      {/each}
    </ul>
  </figcaption>
</figure>

<style>
  .plot {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .canvas {
    width: 100%;
    min-width: 0;
  }
  svg {
    display: block;
    overflow: visible;
  }
  .grid line {
    stroke: var(--line);
    stroke-width: 1;
  }
  .grid line.zero {
    stroke: var(--line-strong);
  }
  .grid text {
    fill: var(--text-2);
    font-size: var(--fs-xs);
    font-variant-numeric: tabular-nums;
  }
  .grid text.axis {
    fill: var(--text-2);
  }
  .pt {
    fill: currentColor;
    fill-opacity: 0.75;
    stroke: currentColor;
    stroke-width: 1;
  }
  .pt :global(.cross) {
    fill: none;
    stroke-width: 1.5;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-1) var(--s-4);
    margin: 0;
    padding: 0;
    list-style: none;
    color: var(--text-2);
  }
  .legend li {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
</style>

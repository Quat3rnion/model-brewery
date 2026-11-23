<script lang="ts">
  import { score } from '../../lib/abliteration';
  type Point = { x: number; y: number | null | undefined; label: string; marked?: boolean };
  let { label, axis, points, controls = [], curve = false, front = [] }: {
    label: string; axis: string; points: Point[]; controls?: Point[]; curve?: boolean; front?: Point[];
  } = $props();
  const numeric = (p: Point): p is Point & { y: number } => typeof p.y === 'number' && Number.isFinite(p.y);
  const ys = $derived([...points, ...controls].filter(numeric).map(p => p.y));
  const xs = $derived([...points, ...controls].map(p => p.x));
  const xmin = $derived(xs.length ? Math.min(...xs) : 0);
  const xmax = $derived(xs.length ? Math.max(...xs) : 0);
  const ymin = $derived(ys.length ? Math.min(...ys) : 0);
  const ymax = $derived(ys.length ? Math.max(...ys) : 0);
  const x = (n: number) => xmin === xmax ? 174 : 28 + (n - xmin) / (xmax - xmin) * 292;
  const y = (n: number) => ymin === ymax ? 74 : 126 - (n - ymin) / (ymax - ymin) * 104;
  const path = (rows: Point[]) => rows.map((p, i) => numeric(p)
    ? `${i && numeric(rows[i - 1]) ? 'L' : 'M'}${x(p.x)},${y(p.y)}` : '').join(' ');
</script>

<figure class="plot">
  <figcaption>{label}</figcaption>
  {#if ys.length}
    <svg viewBox="0 0 348 148" role="img" aria-label={label}>
      <path class="axis" d="M28,14 V126 H326" />
      {#if ymin < 0 && ymax > 0}
        <line class="axis zero" x1="28" x2="326" y1={y(0)} y2={y(0)} />
      {/if}
      {#if curve}<path class="fit" d={path(points)} />{/if}
      {#if front.length > 1}<path class="fit front" d={path(front)} />{/if}
      {#each points.filter(numeric) as p}
        {#if p.marked}
          <circle class="ring" cx={x(p.x)} cy={y(p.y)} r="7"><title>{p.label}</title></circle>
        {/if}
        <circle class="point" cx={x(p.x)} cy={y(p.y)} r="4"><title>{p.label}</title></circle>
      {/each}
      {#each controls.filter(numeric) as p}
        <rect class="control" x={x(p.x) - 4} y={y(p.y) - 4} width="8" height="8"><title>{p.label}</title></rect>
      {/each}
    </svg>
    <p class="small muted num">{axis}: {score(xmin)}{xmin === xmax ? '' : ` to ${score(xmax)}`} · Vertical: {score(ymin)}{ymin === ymax ? '' : ` to ${score(ymax)}`}</p>
  {:else}
    <p class="small muted">No measured values.</p>
  {/if}
</figure>

<style>
  .plot { margin: 0; padding: var(--s-3); background: var(--surface-2); border-radius: var(--r-tile); min-width: 0; }
  figcaption { font-weight: 600; }
  svg { display: block; width: 100%; max-height: 14rem; }
  .axis { fill: none; stroke: var(--line-strong); }
  .zero, .front { stroke-dasharray: 4 3; }
  .fit, .ring { fill: none; stroke: var(--accent); stroke-width: 1.5; }
  .point { fill: var(--accent); }
  .control { fill: var(--surface-2); stroke: var(--text-2); stroke-width: 1.5; }
</style>

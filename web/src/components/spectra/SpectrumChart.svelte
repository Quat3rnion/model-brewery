<script lang="ts">
  // Singular values of one tensor on log-log axes: source solid, reference dashed.
  import { linePath, spectrumPoints, type SpectraSide } from '../../lib/spectra';
  import { fmt } from '../../lib/weightspace';

  let { name, source, reference }: { name: string; source?: SpectraSide; reference?: SpectraSide } = $props();

  const uid = $props.id();
  let width = $state(320);
  const H = 180, L = 52, R = 10, T = 8, B = 28;
  const plotW = $derived(Math.max(60, width - L - R));
  const s = $derived(spectrumPoints(source?.singular_values ?? []));
  const r = $derived(spectrumPoints(reference?.singular_values ?? []));
  const all = $derived([...s, ...r]);
  const n = $derived(Math.max(2, ...all.map(([i]) => i)));
  const lo = $derived(Math.log10(Math.min(...all.map(([, v]) => v))));
  const hi = $derived(Math.log10(Math.max(...all.map(([, v]) => v))));
  const span = $derived(hi - lo || 1);
  const x = (i: number) => L + (Math.log10(i) / Math.log10(n)) * plotW;
  const y = (v: number) => T + (1 - (Math.log10(v) - lo) / span) * (H - T - B);
  const decades = (a: number, b: number) => {
    const out: number[] = [];
    for (let e = Math.floor(a); e <= Math.ceil(b); e++) if (e >= a - 1e-9 && e <= b + 1e-9) out.push(e);
    return out.length ? out : [a, b];
  };
  const yt = $derived(decades(lo, hi));
  const xt = $derived(decades(0, Math.log10(n)).map((e) => Math.round(10 ** e)));
  const sideText = (t?: SpectraSide) => {
    const v = t?.singular_values ?? [];
    return v.length ? `σmax ${fmt(Math.max(...v))}, σmin ${fmt(Math.min(...v))}, ${v.length} values, ${t?.method || 'method not reported'}` : 'No singular values';
  };
</script>

<figure class="chart">
  {#if !all.length}
    <p class="empty">This tensor has no positive singular values to show on a log scale.</p>
  {:else}
    <div class="frame" bind:clientWidth={width}>
      <svg role="img" aria-label="Singular values of {name}" aria-describedby="{uid}-s" {width} height={H} viewBox="0 0 {width} {H}">
        {#each yt as e (e)}
          <line class="grid" x1={L} x2={L + plotW} y1={y(10 ** e)} y2={y(10 ** e)} />
          <text class="tick" x={L - 6} y={y(10 ** e)} text-anchor="end" dominant-baseline="middle">{fmt(10 ** e, 2)}</text>
        {/each}
        {#each xt as i (i)}
          <text class="tick" x={x(i)} y={H - B + 14} text-anchor="middle">{i}</text>
        {/each}
        <text class="tick" x={L + plotW} y={H - 2} text-anchor="end">Index (log)</text>
        {#if r.length}<path class="ref" d={linePath(r, x, y)} />{/if}
        {#if s.length}<path class="src" d={linePath(s, x, y)} />{/if}
      </svg>
    </div>
    <dl class="xs summary" id="{uid}-s">
      <dt>Source</dt><dd class="num">{sideText(source)}</dd>
      {#if reference}<dt>Reference</dt><dd class="num">{sideText(reference)}</dd>{/if}
    </dl>
  {/if}
</figure>

<style>
  .chart {
    margin: 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-1);
  }
  svg {
    display: block;
    overflow: visible;
  }
  .grid {
    stroke: var(--line-soft);
  }
  .tick {
    fill: var(--muted);
    font-size: 10px;
    font-family: var(--font-sans);
    font-variant-numeric: tabular-nums;
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
  .summary {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 2px var(--s-3);
    margin: 0;
    color: var(--text-2);
  }
  .summary dd {
    margin: 0;
    color: var(--text);
  }
</style>

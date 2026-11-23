<script lang="ts">
  // Read-only heat map of a matrix, such as layer x token activation norms or
  // query x key attention. One focusable grid with aria-activedescendant:
  // arrows move, Home/End go to the row ends, Enter selects the cell. A
  // table with the same values is the text alternative.
  import { theme } from '../../lib/theme.svelte';
  import { divergent, legend } from '../../lib/weightmap';
  import { fmt, type Matrix } from '../../lib/weightspace';

  let {
    matrix,
    label,
    rowName,
    colName,
    colLabel = (c: number) => String(c),
    selected = null,
    onselect,
  }: {
    matrix: Matrix;
    label: string;
    rowName: string;
    colName: string;
    colLabel?: (c: number) => string;
    selected?: { r: number; c: number } | null;
    onselect?: (cell: { r: number; c: number; row: number; col: number; value: number | null }) => void;
  } = $props();

  const uid = $props.id();
  const scheme = $derived(theme.resolved === 'light' ? 'light' : 'dark');
  let focus = $state<{ r: number; c: number } | null>(null);
  let gridEl: HTMLDivElement | undefined = $state();
  const nr = $derived(matrix.rows.length);
  const nc = $derived(matrix.cols.length);
  const cellId = (r: number, c: number) => `${uid}-${r}-${c}`;
  const value = (r: number, c: number) => matrix.values[r]?.[c] ?? null;
  const describe = (r: number, c: number) => {
    const v = value(r, c);
    return `${rowName} ${matrix.rows[r]}, ${colName} ${colLabel(matrix.cols[c])}: ${v === null ? 'not captured' : fmt(v)}`;
  };

  function move(r: number, c: number) {
    focus = { r: Math.max(0, Math.min(nr - 1, r)), c: Math.max(0, Math.min(nc - 1, c)) };
    document.getElementById(cellId(focus.r, focus.c))?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  function pick(r: number, c: number) {
    move(r, c);
    onselect?.({ r, c, row: matrix.rows[r], col: matrix.cols[c], value: value(r, c) });
  }
  function key(e: KeyboardEvent) {
    const cur = focus ?? { r: 0, c: 0 };
    const d: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    if (e.key in d) {
      e.preventDefault();
      move(cur.r + d[e.key][0], cur.c + d[e.key][1]);
    } else if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault();
      move(cur.r, e.key === 'Home' ? 0 : nc - 1);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      pick(cur.r, cur.c);
    }
  }
</script>

{#if nr && nc}
  <div class="heat">
    <div class="wrap">
      <div
        bind:this={gridEl}
        class="grid"
        role="grid"
        tabindex="0"
        aria-label="{label}. Use the arrow keys to move and Enter to select."
        aria-rowcount={nr}
        aria-colcount={nc}
        aria-activedescendant={focus ? cellId(focus.r, focus.c) : undefined}
        style:grid-template-columns="auto repeat({nc}, minmax(16px, 26px))"
        onkeydown={key}
        onfocus={() => {
          if (!focus) focus = selected ?? { r: 0, c: 0 };
        }}
      >
        <div class="corner" aria-hidden="true">{rowName} \ {colName}</div>
        {#each matrix.cols as col, c (c)}
          <div class="ch" aria-hidden="true">{#if nc <= 16 || c % 4 === 0}{col}{/if}</div>
        {/each}
        {#each matrix.rows as row, r (r)}
          <div class="rh" aria-hidden="true">{row}</div>
          <div role="row" class="row">
            {#each matrix.cols as _, c (c)}
              {@const v = value(r, c)}
              <div
                id={cellId(r, c)}
                role="gridcell"
                tabindex="-1"
                class="cell"
                class:none={v === null}
                class:foc={focus?.r === r && focus?.c === c}
                class:sel={selected?.r === r && selected?.c === c}
                aria-selected={selected?.r === r && selected?.c === c}
                aria-label={describe(r, c)}
                title={describe(r, c)}
                style:background={v === null ? undefined : divergent(v, matrix.max, scheme)}
                onmousedown={(e) => e.preventDefault()}
                onpointerdown={(e) => {
                  if (e.button !== 0) return;
                  gridEl?.focus({ preventScroll: true });
                  pick(r, c);
                }}
              ></div>
            {/each}
          </div>
        {/each}
      </div>
    </div>
    <div class="scale xs muted">
      <span>0</span><span class="bar" style:background={legend(scheme)} aria-hidden="true"></span><span class="num">{fmt(matrix.max)}</span>
      <span>Hatched: not captured.</span>
    </div>
    <details class="disclosure">
      <summary><span class="small">Values as a table</span></summary>
      <div class="table-wrap">
        <table class="data num">
          <caption class="sr-only">{label}</caption>
          <thead><tr><th scope="col">{rowName}</th>{#each matrix.cols as col (col)}<th scope="col">{colName} {colLabel(col)}</th>{/each}</tr></thead>
          <tbody>
            {#each matrix.rows as row, r (r)}
              <tr><th scope="row">{row}</th>{#each matrix.cols as _, c (c)}<td>{fmt(value(r, c))}</td>{/each}</tr>
            {/each}
          </tbody>
        </table>
      </div>
    </details>
  </div>
{:else}
  <p class="empty">No values to show.</p>
{/if}

<style>
  .heat {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    min-width: 0;
  }
  .wrap {
    overflow: auto;
    max-width: 100%;
    max-height: 480px;
    padding: 4px;
  }
  .grid {
    display: grid;
    width: max-content;
    gap: 1px;
    padding: 4px;
    border-radius: var(--r-tile);
    background: var(--line-soft);
  }
  .grid:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .row {
    display: contents;
  }
  .corner,
  .ch,
  .rh {
    background: var(--surface);
    color: var(--muted);
    font-size: 10px;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    white-space: nowrap;
  }
  .corner {
    padding: 2px 6px 4px 2px;
  }
  .ch {
    padding: 2px 0 4px;
  }
  .rh {
    padding: 0 6px 0 2px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
  }
  .cell {
    aspect-ratio: 1;
    cursor: pointer;
    position: relative;
  }
  .cell.none {
    background: repeating-linear-gradient(45deg, var(--surface), var(--surface) 3px, var(--line-soft) 3px, var(--line-soft) 6px);
  }
  .cell.sel {
    box-shadow: inset 0 0 0 2px var(--accent);
    z-index: 1;
  }
  .cell.foc {
    outline: 2px solid var(--focus);
    outline-offset: 1px;
    z-index: 2;
  }
  .grid:not(:focus-visible) .cell.foc:not(.sel) {
    outline: none;
  }
  .scale {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }
  .bar {
    width: 120px;
    height: 8px;
    border-radius: var(--r-sm);
    /* Only the neutral-to-positive half applies to values of 0 and above. */
    background-size: 200% 100% !important;
    background-position: right !important;
  }
  table.data th[scope='row'] {
    color: var(--text-2);
    font-weight: 500;
  }
</style>

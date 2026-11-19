<script lang="ts">
  // One tile of real stored values as a heat map. A single focusable grid with
  // aria-activedescendant. Click selects, Ctrl/Cmd+click toggles, Shift+click
  // or drag selects a rectangle; a tap on a touch screen toggles one cell. Keys: arrows move and select, Shift+arrows
  // extend, Ctrl+arrows move only, Space toggles, Ctrl/Cmd+A selects the tile,
  // Escape clears; + and - nudge by one stored step (PageUp/PageDown by ten),
  // Delete resets the selected cells.
  import { theme } from '../../lib/theme.svelte';
  import { cellIndex, divergent, keyOf, short, type Tile } from '../../lib/weightmap';

  let {
    tile,
    pending,
    selected = $bindable(new Set<string>()),
    focusKey = $bindable(null),
    selectable = () => true,
    onnudge,
    onreset,
    label,
  }: {
    tile: Tile;
    pending: ReadonlyMap<string, number>;
    selected?: Set<string>;
    focusKey?: string | null;
    selectable?: (index: number[]) => boolean;
    onnudge?: (steps: number) => void;
    onreset?: () => void;
    label: string;
  } = $props();

  const uid = $props.id();
  const WRAP = 32;
  const oneD = $derived(tile.shape.length === 1);
  const n = $derived(tile.values[0]?.length ?? 0);
  const rows = $derived(oneD ? Math.ceil(n / WRAP) : tile.values.length);
  const cols = $derived(oneD ? Math.min(WRAP, n) : n);

  interface C {
    r: number;
    c: number;
    key: string;
    index: number[];
    value: number;
  }
  const grid = $derived.by(() => {
    const out: (C | null)[][] = [];
    for (let r = 0; r < rows; r++) {
      const line: (C | null)[] = [];
      for (let c = 0; c < cols; c++) {
        const [i, j] = oneD ? [0, r * WRAP + c] : [r, c];
        if (j >= n) {
          line.push(null);
          continue;
        }
        const index = cellIndex(tile, i, j);
        line.push({ r, c, key: keyOf(index), index, value: tile.values[i][j] });
      }
      out.push(line);
    }
    return out;
  });
  const cells = $derived(grid.flat().filter((x): x is C => x !== null));
  const byKey = $derived(new Map(cells.map((x) => [x.key, x])));
  const scale = $derived(Math.max(tile.absMax, ...cells.map((x) => Math.abs(pending.get(x.key) ?? 0))));
  const rowLabel = (r: number) => (oneD ? tile.origin[0] + r * WRAP : tile.origin[tile.origin.length - 2] + r);
  const colLabel = (c: number) => (oneD ? c : tile.origin[tile.origin.length - 1] + c);
  const scheme = $derived(theme.resolved === 'light' ? 'light' : 'dark');

  const focused = $derived(byKey.get(focusKey ?? '') ?? null);
  let anchor = $state<C | null>(null);
  let dragging = false;
  let gridEl: HTMLDivElement | undefined = $state();

  const cellId = (x: C) => `${uid}-${x.r}-${x.c}`;
  function describe(x: C) {
    const after = pending.get(x.key);
    return `${x.index.join(', ')}: ${x.value}${after !== undefined ? `, edited to ${after}` : ''}`;
  }

  function rect(a: C, b: C): Set<string> {
    const out = new Set<string>();
    for (let r = Math.min(a.r, b.r); r <= Math.max(a.r, b.r); r++)
      for (let c = Math.min(a.c, b.c); c <= Math.max(a.c, b.c); c++) {
        const x = grid[r]?.[c];
        if (x && selectable(x.index)) out.add(x.key);
      }
    return out;
  }
  // Selection outside this tile stays; only this tile's part changes.
  function outside(): Set<string> {
    return new Set([...selected].filter((k) => !byKey.has(k)));
  }
  function setFocus(x: C) {
    focusKey = x.key;
    document.getElementById(cellId(x))?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  function only(x: C) {
    selected = selectable(x.index) ? new Set([x.key]) : new Set();
    anchor = x;
  }
  function toggle(x: C) {
    if (!selectable(x.index)) return;
    const s = new Set(selected);
    if (s.has(x.key)) s.delete(x.key);
    else s.add(x.key);
    selected = s;
    anchor = x;
  }
  function extend(x: C, keepOthers: boolean) {
    const base = keepOthers ? new Set(selected) : outside();
    for (const k of rect(anchor ?? x, x)) base.add(k);
    selected = base;
  }

  function down(e: PointerEvent, x: C) {
    if (e.button !== 0) return;
    gridEl?.focus({ preventScroll: true });
    setFocus(x);
    // Touch: a tap adds or removes one cell and the page can still scroll.
    if (e.pointerType === 'touch') toggle(x);
    else if (e.shiftKey && anchor) extend(x, e.ctrlKey || e.metaKey);
    else if (e.ctrlKey || e.metaKey) toggle(x);
    else {
      only(x);
      dragging = true;
    }
  }
  function moveDrag(e: PointerEvent) {
    if (!dragging) return;
    const el = (document.elementFromPoint(e.clientX, e.clientY) as HTMLElement | null)?.closest<HTMLElement>('[data-k]');
    const x = el ? byKey.get(el.dataset.k!) : undefined;
    if (x && x.key !== focusKey) {
      focusKey = x.key;
      extend(x, false);
    }
  }

  function key(e: KeyboardEvent) {
    const cur = focused ?? cells[0];
    if (!cur) return;
    const d: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    const mod = e.ctrlKey || e.metaKey;
    if (e.key in d) {
      e.preventDefault();
      const [dr, dc] = d[e.key];
      const next = grid[Math.max(0, Math.min(rows - 1, cur.r + dr))]?.[Math.max(0, Math.min(cols - 1, cur.c + dc))] ?? cur;
      setFocus(next);
      if (e.shiftKey) extend(next, false);
      else if (!mod) only(next);
    } else if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault();
      const next = e.key === 'Home' ? cells[0] : cells[cells.length - 1];
      setFocus(next);
      if (!mod) only(next);
    } else if (e.key === ' ') {
      e.preventDefault();
      toggle(cur);
    } else if (mod && e.key.toLowerCase() === 'a') {
      e.preventDefault();
      const s = outside();
      for (const x of cells) if (selectable(x.index)) s.add(x.key);
      selected = s;
    } else if (e.key === 'Escape' && selected.size) {
      e.preventDefault();
      selected = new Set();
    } else if (e.key === '+' || e.key === '=') {
      e.preventDefault();
      onnudge?.(1);
    } else if (e.key === '-' || e.key === '_') {
      e.preventDefault();
      onnudge?.(-1);
    } else if (e.key === 'PageUp' || e.key === 'PageDown') {
      e.preventDefault();
      onnudge?.(e.key === 'PageUp' ? 10 : -10);
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      onreset?.();
    }
  }
</script>

<svelte:window onpointerup={() => (dragging = false)} />

<div class="wrap">
  <div
    bind:this={gridEl}
    class="grid"
    role="grid"
    tabindex="0"
    aria-label={label}
    aria-multiselectable="true"
    aria-rowcount={rows}
    aria-colcount={cols}
    aria-activedescendant={focused ? cellId(focused) : undefined}
    style:grid-template-columns="auto repeat({cols}, minmax(14px, 22px))"
    onkeydown={key}
    onpointermove={moveDrag}
      onmousedown={(e) => e.preventDefault()}
    onfocus={() => {
      if (!focused && cells[0]) focusKey = cells[0].key;
    }}
  >
    <div class="corner" aria-hidden="true"></div>
    {#each Array.from({ length: cols }, (_, c) => c) as c (c)}
      <div class="ch" aria-hidden="true">{#if c % 4 === 0 || cols <= 8}{colLabel(c)}{/if}</div>
    {/each}
    {#each grid as line, r (r)}
      <div class="rh" aria-hidden="true">{#if r % 4 === 0 || rows <= 8}{rowLabel(r)}{/if}</div>
      <div role="row" class="row">
        {#each line as x, c (c)}
          {#if x}
            {@const after = pending.get(x.key)}
            {@const can = selectable(x.index)}
            <div
              id={cellId(x)}
              role="gridcell"
              tabindex="-1"
              data-k={x.key}
              class="cell"
              class:edited={after !== undefined}
              class:sel={selected.has(x.key)}
              class:foc={focusKey === x.key}
              class:off={!can}
              aria-selected={selected.has(x.key)}
              aria-disabled={!can || undefined}
              aria-label={describe(x)}
              title="{x.index.join(', ')} = {short(x.value)}{after !== undefined ? ` → ${short(after)}` : ''}"
              style:background={divergent(after ?? x.value, scale, scheme)}
              onpointerdown={(e) => down(e, x)}
            ></div>
          {:else}
            <div class="cell none" role="gridcell" aria-hidden="true"></div>
          {/if}
        {/each}
      </div>
    {/each}
  </div>
</div>

<style>
  .wrap {
    overflow: auto;
    max-width: 100%;
    padding: 4px;
  }
  .grid {
    display: grid;
    width: max-content;
    gap: 1px;
    padding: 4px;
    border-radius: var(--r-tile);
    background: var(--line-soft);
    touch-action: manipulation;
    user-select: none;
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
  .ch {
    padding: 2px 0 4px;
    text-align: left;
    overflow: visible;
  }
  .rh {
    padding: 0 6px 0 2px;
    text-align: right;
    display: flex;
    align-items: center;
    justify-content: flex-end;
  }
  .cell {
    position: relative;
    aspect-ratio: 1;
    cursor: crosshair;
  }
  .cell.none {
    background: repeating-linear-gradient(45deg, var(--surface), var(--surface) 3px, var(--line-soft) 3px, var(--line-soft) 6px);
    cursor: default;
  }
  .cell.off {
    cursor: not-allowed;
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
  .grid:not(:focus-visible) .cell.foc {
    outline-color: var(--text-2);
  }
  /* Edited cells keep a corner mark, so colour is not the only signal. */
  .cell.edited::after {
    content: '';
    position: absolute;
    right: 0;
    top: 0;
    border-style: solid;
    border-width: 0 7px 7px 0;
    border-color: transparent var(--text) transparent transparent;
  }
</style>

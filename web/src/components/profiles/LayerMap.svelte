<script lang="ts">
  // Trunk layer map: one cell per layer. The fill shows the share of
  // editable tensors that use HT43; a dot marks layers changed from the base.
  // Multi-select listbox: click toggles, Shift+click selects a range;
  // arrows move, Space toggles, Shift+arrows extend, Ctrl/Cmd+A selects all, Escape clears.
  import { rangeSelect, type LayerStat } from '../../lib/profiles';

  let { layers, stats, selected = $bindable([]) }: { layers: number[]; stats: Map<number, LayerStat>; selected?: number[] } = $props();
  const uid = $props.id();
  let focusIdx = $state(0);
  let anchor = $state<number | null>(null);
  let list: HTMLDivElement | undefined = $state();
  const sel = $derived(new Set(selected));

  function cols(): number {
    const el = list?.querySelector<HTMLElement>('[role=option]');
    if (!el || !list) return 12;
    return Math.max(1, Math.round(list.clientWidth / (el.offsetWidth + 4)));
  }
  function focusAt(i: number) {
    focusIdx = Math.max(0, Math.min(layers.length - 1, i));
    list?.querySelectorAll<HTMLElement>('[role=option]')[focusIdx]?.focus();
  }
  function toggle(l: number) {
    selected = sel.has(l) ? selected.filter((x) => x !== l) : [...selected, l].sort((a, b) => a - b);
    anchor = l;
  }
  function extendTo(l: number) {
    const from = anchor ?? l;
    selected = [...new Set([...selected, ...rangeSelect(layers, from, l)])].sort((a, b) => a - b);
  }
  function click(e: MouseEvent, i: number) {
    focusIdx = i;
    const l = layers[i];
    if (e.shiftKey && anchor !== null) extendTo(l);
    else toggle(l);
  }
  function key(e: KeyboardEvent) {
    const c = cols();
    const moves: Record<string, number> = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: c, ArrowUp: -c };
    if (e.key in moves) {
      e.preventDefault();
      const next = Math.max(0, Math.min(layers.length - 1, focusIdx + moves[e.key]));
      if (e.shiftKey) {
        if (anchor === null) anchor = layers[focusIdx];
        extendTo(layers[next]);
      }
      focusAt(next);
    } else if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault();
      focusAt(e.key === 'Home' ? 0 : layers.length - 1);
    } else if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      toggle(layers[focusIdx]);
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
      e.preventDefault();
      selected = [...layers];
    } else if (e.key === 'Escape' && selected.length) {
      e.preventDefault();
      selected = [];
      anchor = null;
    }
  }
  const pct = (s: LayerStat | undefined) => (s && s.editable ? Math.round((s.ht43 / s.editable) * 100) : 0);
</script>

<div class="wrap">
  <div
    bind:this={list}
    class="map"
    role="listbox"
    aria-multiselectable="true"
    aria-label="Trunk layers"
    aria-describedby="{uid}-help"
    tabindex="-1"
    onkeydown={key}
  >
    {#each layers as l, i (l)}
      {@const s = stats.get(l)}
      <!-- Keyboard selection is handled on the listbox (arrows, Space, Shift, Ctrl+A). -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <div
        role="option"
        class="cell"
        class:on={sel.has(l)}
        class:changed={!!s?.changed}
        aria-selected={sel.has(l)}
        aria-label="Layer {l}: {s?.ht43 ?? 0} of {s?.editable ?? 0} editable tensors use HT43{s?.changed ? `, ${s.changed} changed` : ''}{s && s.editable === 0 ? ', nothing editable' : ''}"
        tabindex={i === focusIdx ? 0 : -1}
        onclick={(e) => click(e, i)}
        onfocus={() => (focusIdx = i)}
        style:--fill="{pct(s)}%"
        title="Layer {l} · HT43 {s?.ht43 ?? 0}/{s?.editable ?? 0}{s?.changed ? ` · ${s.changed} changed` : ''}"
      >
        <span class="n num">{l}</span>
        {#if s?.changed}<span class="dot" aria-hidden="true"></span>{/if}
      </div>
    {/each}
  </div>
  <div class="legend xs muted" id="{uid}-help">
    <span><span class="sw v2" aria-hidden="true"></span>All v2</span>
    <span><span class="sw ht" aria-hidden="true"></span>All HT43</span>
    <span><span class="dotk" aria-hidden="true"></span>Changed from base</span>
    <span class="keys">Click or Space to select · Shift for a range · Ctrl+A all · Esc clears</span>
  </div>
</div>

<style>
  .wrap {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .map {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(44px, 1fr));
    gap: 4px;
  }
  .cell {
    position: relative;
    height: 44px;
    border-radius: var(--r-sm);
    border: 1px solid var(--line);
    background: linear-gradient(to top, color-mix(in srgb, var(--accent) 70%, transparent) var(--fill), var(--surface-2) var(--fill));
    display: grid;
    place-items: center;
    cursor: pointer;
    user-select: none;
    touch-action: manipulation;
    transition: border-color 120ms var(--ease), box-shadow 120ms var(--ease);
  }
  .cell:hover {
    border-color: var(--accent);
  }
  .cell.on {
    border: 2px solid var(--text);
    box-shadow: inset 0 0 0 1px var(--bg);
  }
  .cell:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .n {
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text);
    padding: 0 3px;
    border-radius: 3px;
    background: color-mix(in srgb, var(--surface) 75%, transparent);
  }
  .dot {
    position: absolute;
    top: 4px;
    right: 4px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--warn);
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 4px var(--s-4);
    align-items: center;
  }
  .legend > span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .sw {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    border: 1px solid var(--line);
  }
  .sw.v2 {
    background: var(--surface-2);
  }
  .sw.ht {
    background: color-mix(in srgb, var(--accent) 70%, transparent);
  }
  .dotk {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--warn);
  }
  .keys {
    margin-left: auto;
  }
  @media (max-width: 560px) {
    .map {
      grid-template-columns: repeat(auto-fill, minmax(38px, 1fr));
    }
    .cell {
      height: 40px;
    }
    .keys {
      margin-left: 0;
    }
  }
</style>

<script lang="ts">
  // Tensor map after Lanternfish TensorCloud: one cluster per layer, global
  // tensors first. Inside a cluster the tensors sit on a sunflower spiral in
  // file order; the dot area follows log2(element count) and the colour the
  // role from the name. One listbox with aria-activedescendant:
  // Left/Right step in file order, Up/Down jump to the previous or next layer,
  // Home/End, Enter or Space opens.
  import { theme } from '../../lib/theme.svelte';
  import { blockedReason, clusters, matches, ROLES, type Role, type TensorInfo } from '../../lib/weightmap';

  let { tensors, selected = $bindable(null), onopen }: { tensors: TensorInfo[]; selected?: string | null; onopen?: (t: TensorInfo) => void } = $props();

  const uid = $props.id();
  let query = $state('');
  let roles = $state<Set<Role>>(new Set());
  let active = $state<string | null>(null);
  let list: HTMLDivElement | undefined = $state();

  const shown = $derived(tensors.filter((t) => matches(t, query, roles)));
  const groups = $derived(clusters(shown));
  const flat = $derived(groups.flatMap((g) => g.members));
  const present = $derived(new Set(tensors.map((t) => t.role)));
  const maxLog = $derived(Math.max(1, ...tensors.map((t) => Math.log2(t.elements + 1))));

  // Lanternfish role colours; light theme uses darker tones for 3:1 on white.
  const COLORS: Record<'dark' | 'light', Record<Role, string>> = {
    dark: { attn: '#6e9bf2', expert: '#e3b55a', shared: '#f28fdc', router: '#62d8f5', ffn: '#a593f5', norm: '#8b93a3', embed: '#e0846f', output: '#c3d46a', other: '#b8bec9' },
    light: { attn: '#2f5fc4', expert: '#9a6a00', shared: '#b0307f', router: '#0e7f9b', ffn: '#6a4fd1', norm: '#6b7180', embed: '#b34a32', output: '#5f7300', other: '#6f7480' },
  };
  const colors = $derived(COLORS[theme.resolved === 'light' ? 'light' : 'dark']);

  const GOLDEN = Math.PI * (3 - Math.sqrt(5));
  const SPACING = 11;
  function radius(t: TensorInfo) {
    return 3 + 6 * (Math.log2(t.elements + 1) / maxLog);
  }
  function layout(members: TensorInfo[]) {
    const pts = members.map((t, i) => {
      const r = SPACING * Math.sqrt(i + 0.5);
      return { t, x: r * Math.cos(i * GOLDEN), y: r * Math.sin(i * GOLDEN), r: radius(t) };
    });
    const ext = Math.max(18, ...pts.map((p) => Math.hypot(p.x, p.y) + p.r)) + 3;
    return { pts, size: ext * 2, ext };
  }

  const optId = (t: TensorInfo) => `${uid}-t${t.order}`;
  const activeT = $derived(flat.find((t) => t.name === active) ?? null);

  function label(t: TensorInfo) {
    const why = blockedReason(t);
    return `${t.name}, ${t.shape.join(' × ')}, ${t.dtype}, ${ROLES.find((r) => r.id === t.role)?.label}${why ? `. ${why}` : ''}`;
  }
  function open(t: TensorInfo) {
    active = t.name;
    if (blockedReason(t)) return;
    selected = t.name;
    onopen?.(t);
  }
  function move(to: TensorInfo | undefined) {
    if (!to) return;
    active = to.name;
    document.getElementById(optId(to))?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  function key(e: KeyboardEvent) {
    if (!flat.length) return;
    const i = Math.max(0, flat.findIndex((t) => t.name === (active ?? selected)));
    const cur = flat[i];
    const gi = groups.findIndex((g) => g.members.includes(cur));
    const within = groups[gi]?.members.indexOf(cur) ?? 0;
    const jump = (d: number) => {
      const g = groups[gi + d];
      return g ? g.members[Math.min(within, g.members.length - 1)] : undefined;
    };
    const moves: Record<string, () => TensorInfo | undefined> = {
      ArrowRight: () => flat[i + 1],
      ArrowLeft: () => flat[i - 1],
      ArrowDown: () => jump(1),
      ArrowUp: () => jump(-1),
      Home: () => flat[0],
      End: () => flat.at(-1),
    };
    if (e.key in moves) {
      e.preventDefault();
      move(moves[e.key]());
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      open(cur);
    }
  }
  function toggleRole(r: Role) {
    const next = new Set(roles);
    if (next.has(r)) next.delete(r);
    else next.add(r);
    roles = next;
  }
</script>

<div class="map stack">
  <div class="tools">
    <div class="field q">
      <label for="{uid}-q">Filter tensors</label>
      <input id="{uid}-q" class="input" type="search" bind:value={query} autocomplete="off" spellcheck="false" placeholder="attn_q, L3, ffn" aria-describedby="{uid}-qh" />
      <span class="hint" id="{uid}-qh">All words must match. <span class="mono">L3</span> means layer 3.</span>
    </div>
    <div class="chips" role="group" aria-label="Roles">
      {#each ROLES.filter((r) => present.has(r.id)) as r (r.id)}
        <button type="button" class="chip" aria-pressed={roles.has(r.id)} onclick={() => toggleRole(r.id)}>
          <span class="sw" style:background={colors[r.id]} aria-hidden="true"></span>{r.label}
        </button>
      {/each}
    </div>
  </div>

  {#if flat.length === 0}
    <p class="empty">No tensor matches the filter.</p>
  {:else}
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <div
      class="clusters"
      role="listbox"
      tabindex="0"
      bind:this={list}
      aria-label="Tensors by layer, {flat.length} shown"
      aria-activedescendant={activeT ? optId(activeT) : undefined}
      onkeydown={key}
      onmousedown={(e) => e.preventDefault()}
      onfocus={() => {
        if (!active) active = selected ?? flat[0]?.name ?? null;
      }}
    >
      {#each groups as g (g.layer ?? 'global')}
        {@const l = layout(g.members)}
        <div class="cluster" role="group" aria-label={g.layer === null ? 'Global tensors' : `Layer ${g.layer}`}>
          <span class="cl xs muted" aria-hidden="true">{g.layer === null ? 'Global' : `L${g.layer}`}</span>
          <div class="plot" style:width="{l.size}px" style:height="{l.size}px">
            {#each l.pts as p (p.t.name)}
              {@const blocked = !!blockedReason(p.t)}
              <div
                id={optId(p.t)}
                role="option"
                tabindex="-1"
                class="dot"
                class:sel={selected === p.t.name}
                class:act={active === p.t.name}
                class:blocked
                aria-selected={selected === p.t.name}
                aria-disabled={blocked || undefined}
                aria-label={label(p.t)}
                title={label(p.t)}
                style:left="{l.ext + p.x - p.r}px"
                style:top="{l.ext + p.y - p.r}px"
                style:width="{p.r * 2}px"
                style:height="{p.r * 2}px"
                style:background={colors[p.t.role]}
                onclick={() => {
                  list?.focus({ preventScroll: true });
                  open(p.t);
                }}
                onkeydown={() => {}}
              ></div>
            {/each}
          </div>
        </div>
      {/each}
    </div>
    {#if flat.some((t) => blockedReason(t))}
      <details class="disclosure"><summary>Read-only tensors ({flat.filter((t) => blockedReason(t)).length})</summary>
        <ul>{#each flat.filter((t) => blockedReason(t)) as t (t.name)}<li class="small break"><span class="mono">{t.name}</span> ({t.dtype}): {blockedReason(t)}</li>{/each}</ul>
      </details>
    {/if}
    <p class="now xs" aria-live="polite">
      {#if activeT}
        <span class="mono break">{activeT.name}</span>
        <span class="muted">· {activeT.shape.join(' × ')} · {activeT.dtype} · {activeT.elements.toLocaleString()} values{#if blockedReason(activeT)} · {blockedReason(activeT)}{/if}</span>
      {:else}
        <span class="muted">Click a dot, or focus the map and use the arrow keys. Enter opens the tensor.</span>
      {/if}
    </p>
  {/if}
</div>

<style>
  .map {
    gap: var(--s-3);
  }
  .tools {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: var(--s-3);
  }
  .q {
    flex: 1 1 220px;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding-bottom: 20px;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-height: 28px;
    padding: 0 10px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-pill);
    background: var(--surface);
    color: var(--text-2);
    font: inherit;
    font-size: var(--fs-xs);
    cursor: pointer;
  }
  .chip:hover {
    color: var(--text);
  }
  .chip[aria-pressed='true'] {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--text);
    font-weight: 600;
  }
  .sw {
    width: 9px;
    height: 9px;
    border-radius: 50%;
  }
  .clusters {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-3);
    padding: var(--s-3);
    max-height: 340px;
    overflow: auto;
    background: var(--bg);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
  }
  .clusters:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .cluster {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 4px;
    border-radius: var(--r-tile);
    background: var(--surface);
    border: 1px solid var(--line-soft);
  }
  .cl {
    font-variant-numeric: tabular-nums;
  }
  .plot {
    position: relative;
  }
  .dot {
    position: absolute;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 0 0 1px var(--surface);
  }
  .dot:hover {
    box-shadow: 0 0 0 2px var(--text-2);
    z-index: 1;
  }
  .dot.blocked {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .dot.act {
    box-shadow: 0 0 0 2px var(--focus);
    z-index: 2;
  }
  .dot.sel {
    box-shadow: 0 0 0 2px var(--surface), 0 0 0 4px var(--accent);
    z-index: 3;
  }
  .now {
    margin: 0;
    min-height: 1.5em;
  }
</style>

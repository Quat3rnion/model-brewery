<script lang="ts">
  // Searchable tensor list, rendered in pages so a 2,000-tensor catalog never
  // puts more than PAGE rows in the DOM at once.
  import { formatBytes } from '../../lib/format';
  import { signedBytes } from '../../lib/calibration';
  import { BASE_LABEL, GROUP_LABEL, effective, encodingOf, fixedReason, normalizeOverrides, searchTensors, type Base, type Catalog, type Overrides } from '../../lib/profiles';

  const PAGE = 60;
  let { catalog, base, overrides = $bindable(), selectedLayers, readonly = false }: { catalog: Catalog; base: Base; overrides: Overrides; selectedLayers: number[]; readonly?: boolean } = $props();
  const uid = $props.id();
  let q = $state('');
  let group = $state('');
  let changedOnly = $state(false);
  let editableOnly = $state(false);
  let onlySelected = $state(false);
  let page = $state(0);

  const rows = $derived(searchTensors(catalog, base, overrides, { q, group: group || undefined, changedOnly, editableOnly, layers: onlySelected ? selectedLayers : undefined }));
  const pages = $derived(Math.max(1, Math.ceil(rows.length / PAGE)));
  $effect(() => {
    // Back to the first page when the filter changes.
    void q, group, changedOnly, editableOnly, onlySelected, selectedLayers.length;
    page = 0;
  });
  const shown = $derived(rows.slice(page * PAGE, page * PAGE + PAGE));

  function set(name: string, v: Base) {
    overrides = normalizeOverrides(catalog, base, { ...overrides, [name]: v });
  }
</script>

<div class="tl">
  <div class="filters">
    <div class="field grow">
      <label for="{uid}-q">Search tensors</label>
      <input id="{uid}-q" class="input" type="search" bind:value={q} placeholder="Name, role, encoding or file…" autocomplete="off" spellcheck="false" aria-describedby="{uid}-count" />
    </div>
    <div class="field">
      <label for="{uid}-g">Group</label>
      <select id="{uid}-g" class="input" bind:value={group}>
        <option value="">All groups</option>
        {#each Object.keys(GROUP_LABEL) as g (g)}<option value={g}>{GROUP_LABEL[g]}</option>{/each}
      </select>
    </div>
    <div class="checks">
      <label><input type="checkbox" bind:checked={changedOnly} /> Changed only</label>
      <label><input type="checkbox" bind:checked={editableOnly} /> Editable only</label>
      <label><input type="checkbox" bind:checked={onlySelected} disabled={!selectedLayers.length} /> Selected layers{selectedLayers.length ? ` (${selectedLayers.length})` : ''}</label>
    </div>
  </div>
  <p class="xs muted" id="{uid}-count" aria-live="polite">{rows.length.toLocaleString()} of {catalog.tensors.length.toLocaleString()} tensors{rows.length > PAGE ? ` · page ${page + 1} of ${pages}` : ''}</p>

  {#if rows.length === 0}
    <p class="empty">No tensor matches.</p>
  {:else}
    <div class="table-wrap">
      <table class="data stackable">
        <caption class="sr-only">Tensors with their v2 and HT43 encodings and the current choice</caption>
        <thead>
          <tr>
            <th scope="col">Tensor</th>
            <th scope="col">Layer · role</th>
            <th scope="col">v2</th>
            <th scope="col">HT43</th>
            <th scope="col" class="r">HT43 − v2</th>
            <th scope="col">Use</th>
          </tr>
        </thead>
        <tbody>
          {#each shown as t (t.name)}
            {@const why = fixedReason(t)}
            {@const cur = effective(base, overrides, t)}
            <tr class:changed={cur !== base}>
              <th scope="row" class="mono xs break">{t.name}{#if t.file}<div class="muted">{t.file}</div>{/if}</th>
              <td class="xs" data-label="Layer · role">{t.group === 'trunk' ? `L${t.layer}` : GROUP_LABEL[t.group] ?? t.group}{t.role ? ` · ${t.role}` : ''}</td>
              <td class="mono xs" data-label="v2" class:cur={cur === 'v2'}>{t.v2}</td>
              <td class="mono xs" data-label="HT43" class:cur={cur === 'ht43'}>{t.ht43}</td>
              <td class="num r xs" data-label="HT43 − v2">{t.delta === null ? '—' : signedBytes(t.delta)}</td>
              <td data-label="Use">
                {#if why}
                  <span class="xs muted">Fixed: {why}</span>
                {:else}
                  <div class="seg" role="radiogroup" aria-label="Encoding for {t.name}">
                    {#each ['v2', 'ht43'] as const as v (v)}
                      <label class:on={cur === v}>
                        <input type="radio" name="{uid}-{t.name}" value={v} checked={cur === v} disabled={readonly} onchange={() => set(t.name, v)} />
                        {BASE_LABEL[v]}{#if v !== base && cur === v}<span class="sr-only"> (changed)</span>{/if}
                      </label>
                    {/each}
                  </div>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    {#if pages > 1}
      <nav class="pager" aria-label="Tensor pages">
        <button type="button" class="btn small" disabled={page === 0} onclick={() => (page -= 1)}>Previous</button>
        <span class="xs muted num">{page * PAGE + 1}–{Math.min(rows.length, (page + 1) * PAGE)} of {rows.length}</span>
        <button type="button" class="btn small" disabled={page >= pages - 1} onclick={() => (page += 1)}>Next</button>
      </nav>
    {/if}
  {/if}
  <p class="xs muted">Sizes include companion tensors. Base sizes: v2 {formatBytes(catalog.baseBytes.v2)}, HT43 {formatBytes(catalog.baseBytes.ht43)}.</p>
</div>

<style>
  .tl {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .filters {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-3);
    align-items: flex-end;
  }
  .grow {
    flex: 1 1 260px;
  }
  .checks {
    display: flex;
    flex-wrap: wrap;
    gap: 4px var(--s-3);
    font-size: var(--fs-sm);
  }
  .checks label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-height: 36px;
    cursor: pointer;
  }
  tr.changed {
    background: var(--accent-soft);
  }
  td.cur {
    font-weight: 700;
    color: var(--text);
  }
  th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
    font-weight: 500;
  }
  .r {
    text-align: right;
  }
  .seg {
    display: inline-flex;
    padding: 2px;
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    background: var(--surface);
  }
  .seg label {
    position: relative;
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 0 9px;
    border-radius: var(--r-sm);
    font-size: var(--fs-xs);
    color: var(--text-2);
    cursor: pointer;
    white-space: nowrap;
  }
  .seg label.on {
    background: var(--accent);
    color: var(--accent-text);
    font-weight: 600;
  }
  .seg input {
    position: absolute;
    inset: 0;
    opacity: 0;
    margin: 0;
    cursor: pointer;
  }
  .seg label:has(input:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 1px;
  }
  .pager {
    display: flex;
    align-items: center;
    gap: var(--s-3);
  }
  @media (max-width: 640px) {
    .r {
      text-align: left;
    }
  }
</style>

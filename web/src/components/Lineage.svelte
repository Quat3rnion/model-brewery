<script lang="ts">
  import { onDestroy } from 'svelte';
  import { api, errorText } from '../lib/api';
  import { modelLabel } from '../lib/format';
  import type { Model } from '../lib/types';

  let { models }: { models: Model[] } = $props();
  let target = $state('');
  let selected = $state(['', '', '']);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let report = $state<Awaited<ReturnType<typeof api.lineage>> | null>(null);
  let pending: AbortController | undefined;
  onDestroy(() => pending?.abort());

  const supported = $derived(models.filter((m) => m.available !== false && ['safetensors', 'hf', 'gguf'].includes(m.kind)));
  const model = $derived(supported.find((m) => m.id === target));
  const candidates = $derived(model ? supported.filter((m) => m.id !== model?.id &&
    m.host === model.host && (m.kind === 'gguf') === (model.kind === 'gguf')) : []);
  const tooMany = $derived(candidates.length > 3 && !selected.some(Boolean));
  const labels: Record<string, string> = {
    rank_one_edit: 'Rank-one edit', low_rank_edit: 'Low-rank edit', sparse_edit: 'Sparse edit',
    fine_tune: 'Possible fine-tune', pruning: 'Pruning', merge: 'Merge',
  };
  const statuses: Record<string, string> = {
    related: 'Related weights', no_match: 'No match', identical_weights: 'Identical weights',
    incompatible_inventory: 'Different tensor names, shapes, or types', unreadable_checkpoint: 'Cannot read weights',
  };
  const percent = (n: number) => n > 0 && n < 0.0001 ? '<0.01%' : `${(n * 100).toFixed(n >= 0.1 ? 0 : 2)}%`;
  function label(id: string) {
    const entry = models.find((m) => m.id === id);
    return entry ? modelLabel(entry) : id;
  }
  function clear() {
    report = null;
    error = null;
  }
  async function analyze(event: SubmitEvent) {
    event.preventDefault();
    if (busy || !model || tooMany) return;
    clear();
    busy = true;
    pending = new AbortController();
    try {
      report = await api.lineage(model.id, selected.filter((id) => candidates.some((m) => m.id === id)), pending.signal);
    } catch (err) {
      if (!pending.signal.aborted) error = errorText(err);
    } finally {
      busy = false;
    }
  }
</script>

<section class="panel" aria-labelledby="lineage-h">
  <header><h2 id="lineage-h">Model lineage</h2></header>
  <div class="body stack">
    <p class="muted small">Compare local weights to find a likely base. This check does not change Library records.</p>
    <form class="stack" onsubmit={analyze}>
      <div class="choices">
        <label class="field">
          <span>Target model</span>
          <select class="input" bind:value={target} disabled={busy} onchange={() => { selected = ['', '', '']; clear(); }}>
            <option value="">Select a Library model</option>
            {#each supported as entry (entry.id)}
              <option value={entry.id}>{modelLabel(entry)}</option>
            {/each}
          </select>
        </label>
        {#each selected as _, i (i)}
          <label class="field">
            <span>Candidate {i + 1}</span>
            <select class="input" bind:value={selected[i]} disabled={busy || !candidates.length} onchange={clear}>
              <option value="">None selected</option>
              {#each candidates as entry (entry.id)}
                <option value={entry.id} disabled={selected.some((id, j) => id === entry.id && j !== i)}>{modelLabel(entry)}</option>
              {/each}
            </select>
          </label>
        {/each}
      </div>
      <p class="muted small">
        {#if tooMany}Select up to three candidates on the same host.
        {:else}Leave all candidates blank to check up to three eligible entries. For a merge, select the base and both possible parents.{/if}
      </p>
      <div class="row">
        <button type="submit" class="btn primary" disabled={!model || busy || tooMany}>
          {busy ? 'Checking lineage…' : 'Check lineage'}
        </button>
      </div>
    </form>
    <div aria-live="polite">
      {#if error}<p class="note bad" role="alert">{error}</p>{/if}
      {#if report}
        {#if report.most_likely}
          <p><strong>Likely base: {label(report.most_likely.base_id)}</strong></p>
          <p>{labels[report.most_likely.kind] ?? report.most_likely.kind} · {percent(report.most_likely.confidence)} confidence score</p>
        {:else}
          <p><strong>No unique base found.</strong></p>
        {/if}
        <p class="muted small">Not release-qualified</p>
      {/if}
    </div>
    {#if report && report.candidates.length}
      <details>
        <summary>Candidate evidence</summary>
        <ul class="stack">
          {#each report.candidates as candidate (candidate.id)}
            <li>
              <strong>{label(candidate.id)}</strong> — {statuses[candidate.status] ?? candidate.status}
              {#if candidate.evidence}
                {@const evidence = candidate.evidence}
                <p class="small">{percent(evidence.changed_fraction)} of weights changed in {evidence.changed_tensors} tensors. Relative delta: {percent(evidence.relative_delta)}.</p>
                <p class="small">{percent(evidence.changed_to_zero_fraction)} of changes set a weight to zero. Rank-one delta energy: {percent(evidence.rank_one_energy_fraction)}. Low-rank delta energy: {percent(evidence.low_rank_energy_fraction)}.</p>
                {#if evidence.merge}
                  <p class="small">
                    Merge parents: {evidence.merge.parent_ids.map((id, i) => `${label(id)} (${evidence.merge?.weights[i].toFixed(3)})`).join(', ')}.
                    Relative fit error: {percent(evidence.merge.relative_residual)}. Parent delta cosine: {evidence.merge.parent_delta_cosine.toFixed(3)}.
                  </p>
                {/if}
                {#if evidence.tensors.length}
                  <ul>
                    {#each evidence.tensors as tensor (tensor.name)}
                      <li class="small">
                        <code>{tensor.name}</code>: {percent(tensor.changed_fraction)} changed.
                        Bounds: {tensor.bounds.start.join(', ')} to {tensor.bounds.end_exclusive.join(', ')} (exclusive).
                        {#if tensor.delta_spectrum}Stable rank: {tensor.delta_spectrum.stable_rank.toFixed(2)}.{/if}
                      </li>
                    {/each}
                  </ul>
                {/if}
              {/if}
            </li>
          {/each}
        </ul>
      </details>
    {/if}
  </div>
</section>

<style>
  .choices { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 14rem), 1fr)); gap: var(--sp-3, 0.75rem); }
  .field, select { min-width: 0; width: 100%; }
  summary { cursor: pointer; }
  code { overflow-wrap: anywhere; }
</style>

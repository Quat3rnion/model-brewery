<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import { api, errorText } from '../../lib/api';
  import type { TraceSummary } from '../../lib/weightspace';

  let { jobId, summary }: { jobId: string; summary: TraceSummary } = $props();
  type Cluster = { token_id: number; previous_token_id?: number | null; selected: number; observations: number };
  type Expert = { expert: number; selected: number; load: number; mean_selected_weight: number | null; token_clusters: Cluster[]; context_clusters: Cluster[] };
  type Report = {
    layer: number; positions: number; experts_per_token: number; mean_entropy_nats: number;
    experts: Expert[]; coactivation: { a: number; b: number; count: number }[];
    ablation: { expert: number; rows: { position: number; selected: boolean; residual_l2: number; top_token_changed: boolean | null }[] } | null;
  };

  const initial = untrack(() => ({ layer: summary.spec.layers[0], start: summary.spec.positions[0] ?? 0 }));
  let layer = $state(initial.layer);
  let start = $state<number | undefined>(initial.start);
  let end = $state<number | undefined>(initial.start + 1);
  let ablation = $state('');
  let report = $state<Report | null>(null);
  let expert = $state<number | undefined>();
  let error = $state('');
  let busy = $state(false);
  let pending: AbortController | undefined;
  const selected = $derived(report?.experts.find((row) => row.expert === expert));
  const pairs = $derived(report?.coactivation.filter((pair) => pair.a === expert || pair.b === expert) ?? []);
  onDestroy(() => pending?.abort());

  async function load(event: SubmitEvent) {
    event.preventDefault();
    if (start === undefined || end === undefined || start >= end || end - start > 128) {
      error = 'Select 1 to 128 contiguous captured positions.';
      return;
    }
    pending?.abort();
    const ac = new AbortController();
    pending = ac;
    busy = true;
    error = '';
    report = null;
    const query: Record<string, string | number> = { prompt: summary.prompt, layer, start, end };
    if (ablation.trim()) query.ablation_job_id = ablation.trim();
    try {
      const data = await api.trace(jobId, query, 'routing-atlas', ac.signal) as Report;
      if (ac.signal.aborted) return;
      report = data;
      expert = data.experts.reduce((a, b) => a.selected >= b.selected ? a : b).expert;
    } catch (e) {
      if (!ac.signal.aborted) error = errorText(e);
    } finally {
      if (!ac.signal.aborted) busy = false;
    }
  }
</script>

<section class="box stack">
  <h3>MoE routing atlas</h3>
  <p class="small muted">Read the expert selections for one layer. These results are not release-qualified.</p>
  <form class="atlas-form" onsubmit={load}>
    <label class="field">Layer<select class="input" bind:value={layer}>
      {#each summary.spec.layers as n (n)}<option value={n}>{n}</option>{/each}
    </select></label>
    <label class="field">Start<input class="input" type="number" min="0" max="4095" bind:value={start} required /></label>
    <label class="field">End (exclusive)<input class="input" type="number" min="1" max="4096" bind:value={end} required /></label>
    <label class="field">Ablation job ID (optional)<input class="input" bind:value={ablation} /></label>
    <button class="btn" type="submit" disabled={busy}>{busy ? 'Reading...' : 'Read atlas'}</button>
  </form>
  {#if error}<p class="small" role="alert">{error}</p>{/if}
  {#if report}
    <p class="small">Layer {report.layer}: {report.positions} positions, {report.experts_per_token} experts per token, mean router entropy {report.mean_entropy_nats.toFixed(3)} nats.</p>
    <label class="field">Expert<select class="input" bind:value={expert}>
      {#each report.experts as row (row.expert)}
        <option value={row.expert}>Expert {row.expert} ({row.selected} selections)</option>
      {/each}
    </select></label>
    {#if selected}
      <p class="small">Load: {(100 * selected.load).toFixed(1)}% of positions. Mean selected weight: {selected.mean_selected_weight?.toFixed(4) ?? 'not selected'}.</p>
      <dl class="groups small">
        <div><dt>Token clusters</dt><dd>{selected.token_clusters.map(c => `#${c.token_id}: ${c.selected}/${c.observations}`).join(', ') || 'None'}</dd></div>
        <div><dt>Previous token and current token</dt><dd>{selected.context_clusters.map(c => `${c.previous_token_id == null ? 'Start' : `#${c.previous_token_id}`} then #${c.token_id}: ${c.selected}/${c.observations}`).join(', ') || 'None'}</dd></div>
        <div><dt>Co-activation</dt><dd>{pairs.map(p => `Expert ${p.a === expert ? p.b : p.a}: ${p.count}`).join(', ') || 'None'}</dd></div>
      </dl>
    {/if}
    {#if report.ablation}
      <h4>Expert {report.ablation.expert} ablation</h4>
      <ul class="small">
        {#each report.ablation.rows as row (row.position)}
          <li>Position {row.position}: {row.selected ? 'selected' : 'not selected'}, residual L2 {row.residual_l2.toPrecision(4)}{row.top_token_changed === null ? '' : row.top_token_changed ? ', top token changed' : ', top token unchanged'}</li>
        {/each}
      </ul>
    {/if}
  {/if}
</section>

<style>
  .atlas-form, .groups { display: flex; flex-wrap: wrap; gap: .75rem; align-items: end; }
  .atlas-form label { flex: 1 1 8rem; }
  .groups div { flex: 1 1 10rem; min-width: 0; }
  dd { margin: 0; overflow-wrap: anywhere; }
</style>

<script lang="ts">
  // X6 crosscoder model diff: one immutable report, read by its ID with GET only.
  // Route: #/edits/crosscoder?id=<64 hex>&rev=<scratch revision, optional>.
  import { api, errorText } from '../lib/api';
  import { brewery } from '../lib/store.svelte';
  import { hashParam, route } from '../lib/route.svelte';
  import { formatBytes, modelLabel, shortHash } from '../lib/format';
  import { fmt } from '../lib/weightspace';
  import {
    CLASSES,
    CLASS_INFO,
    classCounts,
    isCrosscoderId,
    pairSplit,
    parseCrosscoder,
    scatterPoints,
    weightsUrl,
    type CrosscoderReport,
    type FeatureClass,
  } from '../lib/crosscoder';
  import Badge from '../components/Badge.svelte';
  import FeatureScatter from '../components/crosscoder/FeatureScatter.svelte';
  import FeatureTable from '../components/crosscoder/FeatureTable.svelte';

  const id = $derived(hashParam(route.hash, 'id') ?? '');
  const rev = $derived(hashParam(route.hash, 'rev'));
  const valid = $derived(isCrosscoderId(id));

  let report = $state<CrosscoderReport | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(false);
  $effect(() => {
    const target = id;
    report = null;
    error = null;
    if (!isCrosscoderId(target)) return;
    const ac = new AbortController();
    loading = true;
    api
      .crosscoder(target, ac.signal)
      .then((raw) => {
        const r = parseCrosscoder(raw);
        if (r.ok && r.value.id !== target) error = 'The controller returned a report with another ID.';
        else if (r.ok) report = r.value;
        else error = r.error;
      })
      .catch((e) => {
        if (!ac.signal.aborted) error = errorText(e);
      })
      .finally(() => {
        if (!ac.signal.aborted) loading = false;
      });
    return () => ac.abort();
  });

  const counts = $derived(report ? classCounts(report.features) : null);
  const scatter = $derived(report ? scatterPoints(report.features) : { points: [], omitted: 0 });
  const pairs = $derived(pairSplit(report?.pairs));
  const backRev = $derived(rev ?? report?.child.id ?? null);
  const models = $derived(brewery.state?.models ?? []);
  const name = (modelId: string) => {
    const m = models.find((x) => x.id === modelId);
    return m ? modelLabel(m) : modelId;
  };
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const METRICS: { key: 'mse' | 'explained_variance' | 'l0' | 'dead_features'; label: string; hint: string; show: (v: number) => string }[] = [
    { key: 'mse', label: 'Mean squared error', hint: 'Lower is better.', show: (v) => fmt(v) },
    { key: 'explained_variance', label: 'Explained variance', hint: 'Higher is better.', show: pct },
    { key: 'l0', label: 'Active features per token (L0)', hint: 'Mean count.', show: (v) => fmt(v) },
    { key: 'dead_features', label: 'Dead features', hint: 'Never active on this side.', show: (v) => String(v) },
  ];
  const TABLES: { cls: FeatureClass; caption: string; text: string; sort: 'strength' | 'cosine' }[] = [
    { cls: 'parent_unique', caption: 'Features unique to the parent', text: 'Near-zero child decoder norm. The child lost these directions.', sort: 'strength' },
    { cls: 'child_unique', caption: 'Features unique to the child', text: 'Near-zero parent decoder norm. The child gained these directions.', sort: 'strength' },
    { cls: 'changed', caption: 'Shared features that changed', text: 'Both decoder norms are not near zero, but the decoder directions differ. Lowest cosine is first.', sort: 'cosine' },
  ];
</script>

<div class="stack">
  <div class="head">
    {#if backRev}<a class="small" href="#/edits?rev={encodeURIComponent(backRev)}">← Back to the revision</a>{:else}<a class="small" href="#/edits">← Edit sessions</a>{/if}
    <h1 tabindex="-1">Crosscoder diff</h1>
    <p class="muted small">A crosscoder learns one sparse feature dictionary over the parent and child activations at one layer. Each feature has a decoder for each side. A feature with a near-zero decoder norm on one side is unique to the other side. A shared feature with a different decoder direction changed.</p>
  </div>

  {#if !valid}
    <p class="note bad small" role="alert">The report ID is not valid. Open a report from a revision in <a href="#/edits">Edits</a> with its 64-character hexadecimal ID.</p>
  {:else if error}
    <p class="note bad small" role="alert">Could not read crosscoder report <span class="mono">{id.slice(0, 12)}</span>: {error}</p>
  {:else if loading || !report}
    <p class="muted small" role="status">Reading the crosscoder report…</p>
  {:else}
    {@const r = report}
    <section class="panel" aria-labelledby="xc-src-h">
      <header>
        <h2 id="xc-src-h">Report <span class="mono">{r.id.slice(0, 12)}</span></h2>
        <Badge tone="warn" label="Not release-qualified" />
      </header>
      <div class="body stack">
        {#if rev && rev !== r.child.id}
          <p class="note warn small">This report is not for revision {rev}. Its child is <span class="mono break">{r.child.id}</span>.</p>
        {/if}
        <dl class="kv small">
          <dt>Report ID</dt><dd class="mono break">{r.id}</dd>
          <dt>Parent</dt><dd class="break">{name(r.parent.id)} <span class="mono xs muted" title={r.parent.sha256}>{shortHash(r.parent.sha256)}</span></dd>
          <dt>Child</dt><dd class="break"><a href="#/edits?rev={encodeURIComponent(r.child.id)}" class="mono">{r.child.id}</a> <span class="mono xs muted" title={r.child.sha256}>{shortHash(r.child.sha256)}</span></dd>
          <dt>Activations</dt><dd>Layer <span class="num">{r.layer}</span>, <span class="mono">{r.role}</span></dd>
          {#if r.width !== undefined}<dt>Activation size</dt><dd class="num">{r.width}</dd>{/if}
          <dt>Dictionary</dt><dd>{[r.method, `${r.features.length} features`, r.k !== undefined ? `k = ${r.k}` : ''].filter(Boolean).join(', ')}</dd>
          {#if pairs}<dt>Trace pairs</dt><dd class="num">{pairs.total} ({pairs.train} train, {pairs.heldout} held out)</dd>{/if}
          <dt>Qualification</dt><dd>Not release-qualified</dd>
        </dl>
        <div class="row">
          <a class="btn" href={weightsUrl(r.id)} download="crosscoder-{r.id.slice(0, 12)}.safetensors">
            Download crosscoder weights<span class="muted"> (SafeTensors{r.weights_bytes !== undefined ? `, ${formatBytes(r.weights_bytes)}` : ''})</span>
          </a>
          {#if r.weights_sha256}<span class="xs muted">SHA-256 <span class="mono" title={r.weights_sha256}>{shortHash(r.weights_sha256)}</span></span>{/if}
        </div>
      </div>
    </section>

    <div class="grid-2 top">
      <section class="panel" aria-labelledby="xc-m-h">
        <header><h2 id="xc-m-h">Reconstruction</h2></header>
        <div class="body">
          <div class="table-wrap">
            <table class="data num">
              <caption class="sr-only">Reconstruction metrics</caption>
              <thead><tr><th scope="col">Metric</th><th scope="col">Parent</th><th scope="col">Child</th></tr></thead>
              <tbody>
                {#each METRICS as m (m.key)}
                  <tr>
                    <th scope="row">{m.label} <span class="xs muted block">{m.hint}</span></th>
                    <td title={String(r.metrics.parent[m.key])}>{m.show(r.metrics.parent[m.key])}</td>
                    <td title={String(r.metrics.child[m.key])}>{m.show(r.metrics.child[m.key])}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section class="panel" aria-labelledby="xc-c-h">
        <header><h2 id="xc-c-h">Features</h2><span class="xs muted num">{r.features.length} total</span></header>
        <div class="body">
          <ul class="counts" aria-label="Features per class">
            {#each CLASSES as c (c)}
              <li>
                <Badge tone={CLASS_INFO[c].tone} label={CLASS_INFO[c].label} />
                <span class="num count">{counts?.[c] ?? 0}</span>
              </li>
            {/each}
          </ul>
        </div>
      </section>
    </div>

    <section class="panel" aria-labelledby="xc-s-h">
      <header><h2 id="xc-s-h">Child share and decoder cosine</h2></header>
      <div class="body stack">
        <p class="xs muted" id="xc-s-d">
          Child share is child / (parent + child) decoder norm. 0.5: equal norms. Near 0: near-zero child decoder norm. Near 1: near-zero parent decoder norm. Low cosine: the direction changed.
          {#if scatter.omitted}{scatter.omitted} features have no child share or cosine and are not plotted.{/if}
        </p>
        {#if scatter.points.length}
          <FeatureScatter
            points={scatter.points}
            label="Scatter of {scatter.points.length} features: child share of decoder norm against decoder cosine. The table below has the values."
            describedby="xc-s-d"
          />
        {:else}
          <p class="empty">No feature has both a ratio and a cosine.</p>
        {/if}
        <details class="disclosure">
          <summary>All features as a table</summary>
          <FeatureTable features={r.features} cls="all" caption="All features" sort="feature" />
        </details>
      </div>
    </section>

    {#each TABLES as t (t.cls)}
      <section class="panel" aria-labelledby="xc-t-{t.cls}">
        <header><h2 id="xc-t-{t.cls}">{CLASS_INFO[t.cls].label} <span class="num muted">({counts?.[t.cls] ?? 0})</span></h2></header>
        <div class="body stack">
          <p class="xs muted">{t.text}</p>
          <FeatureTable features={r.features} cls={t.cls} caption={t.caption} sort={t.sort} />
        </div>
      </section>
    {/each}
  {/if}
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
    align-items: flex-start;
  }
  .kv {
    display: grid;
    grid-template-columns: minmax(96px, max-content) minmax(0, 1fr);
    gap: 4px var(--s-3);
    margin: 0;
  }
  .kv dt {
    color: var(--text-2);
  }
  .kv dd {
    margin: 0;
    min-width: 0;
  }
  .top {
    align-items: start;
  }
  .block {
    display: block;
    font-weight: 400;
  }
  .counts {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .counts li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s-3);
    padding-bottom: var(--s-2);
    border-bottom: 1px solid var(--line-soft);
  }
  .counts li:last-child {
    border-bottom: 0;
    padding-bottom: 0;
  }
  .count {
    font-size: var(--fs-lg);
    font-weight: 600;
  }
</style>

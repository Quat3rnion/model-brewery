<script lang="ts">
  import { brewery } from '../../lib/store.svelte';
  import { formatTime, modelLabel, shortHash } from '../../lib/format';
  import { fmt } from '../../lib/weightspace';
  import {
    STATE_INFO,
    VERDICT_INFO,
    agentLabel,
    agentSessionHref,
    apiHref,
    compareRows,
    findingHref,
    viewHref,
    type Finding,
  } from '../../lib/findings';
  import Badge from '../Badge.svelte';

  let { id, finding, error }: { id: string; finding: Finding | null; error: string | null } = $props();

  const models = $derived(brewery.state?.models ?? []);
  const modelName = (mid: string) => {
    const m = models.find((x) => x.id === mid);
    return m ? modelLabel(m) : mid;
  };
  const rows = $derived(finding ? compareRows(finding.comparison, finding.experiments) : []);
  const changed = $derived(rows.filter((r) => r.changed).length);
  const signed = (v: number) => (v > 0 ? `+${fmt(v)}` : fmt(v));
</script>

<div class="head">
  <a class="small" href="#/findings">← All findings</a>
  <h1 tabindex="-1">Finding</h1>
  {#if finding}<p class="hypothesis break">{finding.hypothesis}</p>{/if}
</div>

{#if error}
  <p class="note bad small" role="alert">Could not read finding <span class="mono break">{id}</span>: {error}</p>
{:else if !finding}
  <p class="muted small" role="status">Reading the finding…</p>
{:else}
  {@const f = finding}
  <section class="panel" aria-labelledby="fd-sum-h">
    <header>
      <h2 id="fd-sum-h">Version <span class="num">{f.version}</span></h2>
      <div class="row">
        <Badge tone={STATE_INFO[f.state].tone} label={STATE_INFO[f.state].label} />
        <Badge tone="warn" label="Not release-qualified" />
      </div>
    </header>
    <div class="body stack">
      {#if f.state === 'failed'}
        <p class="note bad small" role="alert">The finding failed{f.error ? `: ${f.error}` : '. The controller did not report a reason.'}</p>
      {:else if f.state === 'running'}
        <p class="note info small" role="status">The experiments are running. Reload the page to see new evidence.</p>
      {/if}
      <dl class="kv small">
        <dt>Finding ID</dt><dd class="mono break">{f.id}</dd>
        <dt>Model</dt><dd class="break">{modelName(f.model.id)} <span class="mono xs muted" title={f.model.sha256}>{shortHash(f.model.sha256)}</span></dd>
        <dt>Created</dt><dd>{formatTime(f.created_at_ms)}</dd>
        <dt>Lineage</dt>
        <dd>
          {#if f.replay_of}Replay of <a class="mono" href={findingHref(f.replay_of)}>{f.replay_of}</a>{#if f.parent_id && f.parent_id !== f.replay_of}, from <a class="mono" href={findingHref(f.parent_id)}>{f.parent_id}</a>{/if}
          {:else if f.parent_id}From <a class="mono" href={findingHref(f.parent_id)}>{f.parent_id}</a>
          {:else}First version{/if}
        </dd>
        <dt>Qualification</dt><dd>Not release-qualified</dd>
      </dl>
    </div>
  </section>

  <div class="grid-2 top">
    <section class="panel" aria-labelledby="fd-v-h">
      <header>
        <h2 id="fd-v-h">Verdict</h2>
        {#if f.verdict}<Badge tone={VERDICT_INFO[f.verdict.status].tone} label={VERDICT_INFO[f.verdict.status].label} />{/if}
      </header>
      <div class="body">
        {#if f.verdict}<p class="reason break">{f.verdict.reasoning}</p>
        {:else}<p class="small muted">No verdict yet.</p>{/if}
      </div>
    </section>

    <section class="panel" aria-labelledby="fd-p-h">
      <header><h2 id="fd-p-h">Provenance</h2></header>
      <div class="body">
        {#if f.provenance}
          {@const href = agentSessionHref(f.provenance.agent)}
          <dl class="kv small">
            <dt>Agent</dt><dd>{agentLabel(f.provenance.agent.kind)}</dd>
            <dt>{href ? 'Session' : 'Agent ID'}</dt>
            <dd class="mono break">{#if href}<a {href}>{f.provenance.agent.id}</a>{:else}{f.provenance.agent.id}{/if}</dd>
            <dt>Tool version</dt><dd class="mono">{f.provenance.tool_version}</dd>
          </dl>
        {:else}<p class="small muted">The controller did not report provenance.</p>{/if}
      </div>
    </section>
  </div>

  <section class="panel" aria-labelledby="fd-x-h">
    <header><h2 id="fd-x-h">Experiments <span class="num muted">({f.experiments.length})</span></h2></header>
    <div class="body">
      {#if !f.experiments.length}
        <p class="empty">No experiments yet.</p>
      {:else}
        <ol class="timeline" aria-label="Experiments">
          {#each f.experiments as e, i (i)}
            {@const view = viewHref(e.artifact.view_url)}
            {@const json = apiHref(e.artifact.api_url)}
            <li>
              <span class="step num" aria-hidden="true">{i + 1}</span>
              <div class="entry stack-sm">
                <div class="row between">
                  <h3 class="mono"><span class="sr-only">Experiment {i + 1}: </span>{e.spec.kind}</h3>
                  <div class="row">
                    {#if view}<a class="btn" href={view}>Open {e.artifact.id}</a>{:else}<span class="xs muted">No view for {e.artifact.id}</span>{/if}
                    {#if json}<a class="small" href={json}>JSON<span class="sr-only"> for {e.artifact.id}</span></a>{/if}
                  </div>
                </div>
                <p class="xs muted">Artifact <span class="mono break">{e.artifact.id}</span>{#if e.artifact.job_id && e.artifact.job_id !== e.artifact.id}, job <span class="mono">{e.artifact.job_id}</span>{/if} · SHA-256 <span class="mono" title={e.artifact.sha256}>{shortHash(e.artifact.sha256)}</span></p>
                {#if e.metrics.length}
                  <dl class="metrics small">
                    {#each e.metrics as m (m.path)}
                      <div><dt class="mono break">{m.path}</dt><dd class="num" title={String(m.value)}>{fmt(m.value)}</dd></div>
                    {/each}
                  </dl>
                {:else}<p class="xs muted">No metrics.</p>{/if}
              </div>
            </li>
          {/each}
        </ol>
      {/if}
    </div>
  </section>

  {#if rows.length}
    <section class="panel" aria-labelledby="fd-c-h">
      <header>
        <h2 id="fd-c-h">Replay comparison</h2>
        <span class="small" class:changed-count={changed > 0}>{changed} of {rows.length} values changed</span>
      </header>
      <div class="body">
        <div class="table-wrap">
          <table class="data stackable">
            <caption class="sr-only">Original and replay values</caption>
            <thead>
              <tr>
                <th scope="col">Experiment</th>
                <th scope="col">Metric</th>
                <th scope="col" class="r">Original</th>
                <th scope="col" class="r">Replay</th>
                <th scope="col" class="r">Difference</th>
                <th scope="col">Result</th>
              </tr>
            </thead>
            <tbody>
              {#each rows as r (r.key)}
                <tr class:diff={r.changed}>
                  <td data-label="Experiment" class="mono">{r.experiment}</td>
                  <th scope="row" data-label="Metric" class="mono break">{r.path}</th>
                  <td data-label="Original" class="num r" title={String(r.original)}>{fmt(r.original)}</td>
                  <td data-label="Replay" class="num r" title={String(r.replay)}><strong class:val={r.changed}>{fmt(r.replay)}</strong></td>
                  <td data-label="Difference" class="num r">{r.changed ? signed(r.delta) : '0'}</td>
                  <td data-label="Result">{#if r.changed}<Badge tone="warn" label="Changed" />{:else}<Badge tone="muted" label="Same" />{/if}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  {/if}
{/if}

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
    align-items: flex-start;
  }
  .hypothesis {
    margin: 0;
    max-width: 72ch;
    font-size: var(--fs-lg);
    line-height: 1.4;
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
  .reason {
    margin: 0;
    max-width: 72ch;
    line-height: 1.5;
    white-space: pre-line;
  }
  .timeline {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .timeline li {
    position: relative;
    display: grid;
    grid-template-columns: 28px minmax(0, 1fr);
    gap: var(--s-3);
    padding-bottom: var(--s-4);
  }
  .timeline li:last-child {
    padding-bottom: 0;
  }
  /* The rail joins the step numbers in run order. */
  .timeline li:not(:last-child)::before {
    content: '';
    position: absolute;
    left: 13px;
    top: 28px;
    bottom: 0;
    width: 2px;
    background: var(--line);
  }
  .step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border: 1px solid var(--line-strong);
    border-radius: 50%;
    background: var(--surface-2);
    font-size: var(--fs-xs);
    font-weight: 600;
  }
  .entry {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    min-width: 0;
    padding-top: 3px;
  }
  .between {
    justify-content: space-between;
  }
  h3 {
    margin: 0;
    font-size: var(--fs-md);
    font-weight: 600;
  }
  .entry p {
    margin: 0;
  }
  .metrics {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2);
    margin: 0;
  }
  .metrics div {
    display: flex;
    gap: var(--s-2);
    align-items: baseline;
    padding: 2px var(--s-2);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-sm);
    background: var(--surface-2);
    min-width: 0;
  }
  .metrics dt {
    color: var(--text-2);
  }
  .metrics dd {
    margin: 0;
    font-weight: 600;
  }
  .r {
    text-align: right;
  }
  /* Changed rows use the warn tint and a Changed badge, so colour is not the only signal. */
  tr.diff,
  table.data tbody tr.diff:hover {
    background: var(--warn-bg);
  }
  tr.diff td:first-child {
    box-shadow: inset 3px 0 0 var(--warn);
  }
  tr.diff th {
    background: transparent;
  }
  strong {
    font-weight: 400;
  }
  strong.val {
    font-weight: 600;
  }
  .changed-count {
    color: var(--warn);
    font-weight: 600;
  }
  @media (max-width: 640px) {
    .r {
      text-align: left;
    }
    /* Stacked rows: the tint and the Changed badge mark the row. */
    tr.diff td:first-child {
      box-shadow: none;
    }
  }
</style>

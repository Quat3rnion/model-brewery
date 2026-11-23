<script lang="ts">
  import { brewery } from '../../lib/store.svelte';
  import { formatTime, modelLabel, shortHash } from '../../lib/format';
  import { STATE_INFO, VERDICT_INFO, findingHref, type FindingSummary } from '../../lib/findings';
  import Badge from '../Badge.svelte';

  let { findings, error }: { findings: FindingSummary[] | null; error: string | null } = $props();

  const models = $derived(brewery.state?.models ?? []);
  const modelName = (id: string) => {
    const m = models.find((x) => x.id === id);
    return m ? modelLabel(m) : id;
  };
  const sorted = $derived(findings ? [...findings].sort((a, b) => b.created_at_ms - a.created_at_ms || b.version - a.version) : null);
</script>

<div class="head">
  <h1 tabindex="-1">Findings</h1>
  <p class="muted small">A finding is a hypothesis with experiment evidence, a verdict and provenance. Each change to the evidence or verdict makes a new version. A replay runs the experiments again and compares the values.</p>
</div>

<section class="panel" aria-labelledby="fd-list-h">
  <header>
    <h2 id="fd-list-h">All findings{#if sorted}{' '}<span class="num muted">({sorted.length})</span>{/if}</h2>
    <Badge tone="warn" label="Not release-qualified" />
  </header>
  <div class="body">
    {#if error}
      <p class="note bad small" role="alert">Could not read findings: {error}</p>
    {:else if sorted === null}
      <p class="muted small" role="status">Reading findings…</p>
    {:else if !sorted.length}
      <p class="empty">No findings yet. An agent records findings with the findings tools.</p>
    {:else}
      <div class="table-wrap">
        <table class="data stackable">
          <caption class="sr-only">Findings</caption>
          <thead>
            <tr>
              <th scope="col">Hypothesis</th>
              <th scope="col">Verdict</th>
              <th scope="col">State</th>
              <th scope="col">Model</th>
              <th scope="col">Version</th>
              <th scope="col">Lineage</th>
            </tr>
          </thead>
          <tbody>
            {#each sorted as f (f.id)}
              <tr>
                <th scope="row" class="hyp">
                  <a href={findingHref(f.id)} class="break">{f.hypothesis}</a>
                  <span class="xs muted block"><span class="mono">{f.id}</span> · {formatTime(f.created_at_ms)}</span>
                </th>
                <td data-label="Verdict">
                  {#if f.verdict}<Badge tone={VERDICT_INFO[f.verdict.status].tone} label={VERDICT_INFO[f.verdict.status].label} />
                  {:else}<span class="xs muted">No verdict</span>{/if}
                </td>
                <td data-label="State"><Badge tone={STATE_INFO[f.state].tone} label={STATE_INFO[f.state].label} /></td>
                <td data-label="Model" class="model">{modelName(f.model.id)} <span class="mono xs muted block nowrap" title={f.model.sha256}>{shortHash(f.model.sha256)}</span></td>
                <td data-label="Version" class="num">v{f.version}</td>
                <td data-label="Lineage" class="small">
                  {#if f.replay_of}Replay of <a class="mono" href={findingHref(f.replay_of)}>{f.replay_of}</a>
                  {:else if f.parent_id}From <a class="mono" href={findingHref(f.parent_id)}>{f.parent_id}</a>
                  {:else}<span class="muted">First version</span>{/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>
</section>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .head p {
    max-width: 72ch;
  }
  .hyp {
    color: var(--text);
    font-weight: 500;
    min-width: 16rem;
  }
  .block {
    display: block;
    font-weight: 400;
    margin-top: 2px;
  }
  .model {
    min-width: 9rem;
    overflow-wrap: anywhere;
  }
  @media (max-width: 640px) {
    .model,
    .hyp {
      min-width: 0;
    }
  }
</style>

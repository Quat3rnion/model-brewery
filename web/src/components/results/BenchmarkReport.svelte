<script lang="ts">
  import Badge from '../Badge.svelte';
  import { directionText, formatValue, statusTone, type BenchmarkReport } from '../../lib/benchmark';

  let { report, missing = [], requestedTarget = null }: { report: BenchmarkReport; missing?: string[]; requestedTarget?: string | null } = $props();
  const uid = $props.id();
</script>

<div class="stack">
  <dl class="meta small">
    <dt>Endpoint target</dt><dd class="mono break">{report.target}</dd>
    <dt>Model (as reported)</dt><dd class="mono break">{report.model}</dd>
    <dt>Report status</dt><dd><Badge tone={statusTone(report.status)} label={report.status} /></dd>
    <dt>Release</dt><dd><Badge tone="muted" label="Not release-qualified" /></dd>
  </dl>
  {#if requestedTarget && requestedTarget !== report.target}
    <p class="note warn">The report names target <code>{report.target}</code>, but the job requested <code>{requestedTarget}</code>.</p>
  {/if}

  {#if report.runs.length === 0}
    <p class="muted small">The report has no benchmark runs.</p>
  {:else}
    <div class="table-wrap">
      <table class="data runs stackable">
        <caption class="sr-only">Benchmark runs for target {report.target}</caption>
        <thead>
          <tr>
            <th scope="col">Benchmark</th>
            <th scope="col">Version · protocol</th>
            <th scope="col">Status</th>
            <th scope="col" class="r">Samples</th>
            <th scope="col">Metrics</th>
          </tr>
        </thead>
        <tbody>
          {#each report.runs as run, i (i)}
            <tr>
              <th scope="row" class="mono">{run.benchmark}</th>
              <td class="mono xs break" data-label="Version · protocol">{run.version} · {run.protocol}</td>
              <td data-label="Status"><Badge tone={statusTone(run.status)} label={run.status} /></td>
              <td class="num r" data-label="Samples">{run.samples !== undefined ? run.samples.toLocaleString() : '—'}</td>
              <td data-label="Metrics">
                {#if run.metrics.length}
                  <ul class="metrics">
                    {#each run.metrics as m, k (k)}
                      <li>
                        <span class="muted">{m.name}</span>
                        <span class="num val">{formatValue(m.value)}</span>{#if m.unit}{' '}<span class="muted xs">{m.unit}</span>{/if}
                        <span class="dir xs muted" title={directionText(m.direction)}
                          ><span aria-hidden="true">{m.direction === 'higher' ? '↑' : m.direction === 'lower' ? '↓' : '·'}</span><span class="sr-only">{directionText(m.direction)}</span></span
                        >
                      </li>
                    {/each}
                  </ul>
                {:else}<span class="xs muted">No metrics</span>{/if}
                {#if run.error}<p class="xs err break" id="{uid}-e{i}">{run.error}</p>{/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  {#if missing.length}
    <p class="note warn">Requested but not in the report: <span class="mono">{missing.join(', ')}</span>.</p>
  {/if}
</div>

<style>
  .meta {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 6px var(--s-3);
    margin: 0;
    align-items: center;
  }
  .meta dt {
    color: var(--text-2);
  }
  .meta dd {
    margin: 0;
  }
  .runs th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
  }
  .r {
    text-align: right;
  }
  @media (max-width: 640px) {
    .r {
      text-align: left;
    }
  }
  .stack {
    min-width: 0;
  }
  .metrics {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .val {
    margin-left: 6px;
    font-weight: 600;
  }
  .dir {
    margin-left: 4px;
  }
  .err {
    color: var(--bad);
    margin-top: 4px;
  }
</style>

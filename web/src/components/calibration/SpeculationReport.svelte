<script lang="ts">
  import Badge from '../Badge.svelte';
  import Ids from './Ids.svelte';
  import { count, fixed, pct, type SpeculationReport } from '../../lib/calibration';

  let { report }: { report: SpeculationReport } = $props();
  const about: [string, string][] = $derived(
    (
      [
        ['producer version', report.producerVersion],
        ['version evidence', report.versionEvidence],
        ['model (alias)', report.model],
        ['input sha256', report.inputSha256],
        ['checkpoint sha256', report.checkpointSha256 ?? 'Not stated'],
      ] as [string, string | null][]
    ).filter((x): x is [string, string] => x[1] !== null),
  );
</script>

<div class="stack">
  <div class="row">
    <Badge tone="muted" label="Not tuning-eligible" />
    <Badge tone="muted" label="Not release-qualified" />
  </div>
  <p class="note info">
    Drafted and accepted token counts from a Halogen response timing export. They describe these requests only. They are not per-depth MTP accuracy, not a speed result and not a quality result. Fractions are shown as reported; the viewer does not compute missing ones.
  </p>

  <div class="grid-2">
    <Ids title="Source" items={about} />
    <dl class="totals small num">
      <div><dt>Requests</dt><dd>{count(report.requests)}</dd></div>
      <div><dt>With speculation data</dt><dd>{count(report.withSpeculation)}</dd></div>
      <div><dt>Without speculation data</dt><dd>{count(report.missingSpeculation)}</dd></div>
    </dl>
  </div>

  <div class="table-wrap">
    <table class="data stackable">
      <caption class="sr-only">Draft acceptance per speculation method</caption>
      <thead>
        <tr>
          <th scope="col">Method</th>
          <th scope="col" class="r">Rounds</th>
          <th scope="col" class="r">Drafted</th>
          <th scope="col" class="r">Accepted</th>
          <th scope="col" class="r">Accepted / drafted</th>
          <th scope="col" class="r">Accepted per round</th>
        </tr>
      </thead>
      <tbody>
        {#each report.methods as m (m.name)}
          <tr>
            <th scope="row">{m.name}</th>
            {#if !m.stats}
              <td colspan="5" class="muted">Not observed</td>
            {:else}
              <td class="num r" data-label="Rounds">{count(m.stats.rounds)}</td>
              <td class="num r" data-label="Drafted">{count(m.stats.drafted)}</td>
              <td class="num r" data-label="Accepted">{count(m.stats.accepted)}</td>
              <td class="num r" data-label="Accepted / drafted">{m.stats.acceptance_fraction !== null ? pct(m.stats.acceptance_fraction) : m.stats.drafted === 0 ? 'No drafts' : 'Not reported'}</td>
              <td class="num r" data-label="Accepted per round">{m.stats.accepted_per_round !== null ? fixed(m.stats.accepted_per_round, 3) : m.stats.rounds === 0 ? 'No rounds' : 'Not reported'}</td>
            {/if}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <p class="xs muted">
    {report.hasByDepth ? 'The file has per-depth acceptance data; this viewer does not show it.' : 'No per-depth acceptance in this export.'}
    {report.hasExpertRouting ? 'The file has expert routing data; this viewer does not show it.' : 'No expert routing in this export.'}
  </p>
</div>

<style>
  .totals {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-5);
    margin: 0;
    align-content: start;
  }
  .totals dt {
    color: var(--text-2);
    font-size: var(--fs-xs);
  }
  .totals dd {
    margin: 0;
    font-weight: 600;
  }
  .r {
    text-align: right;
  }
  th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
  }
  @media (max-width: 640px) {
    .r {
      text-align: left;
    }
  }
</style>

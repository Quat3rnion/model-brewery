<script lang="ts">
  import Badge from '../Badge.svelte';
  import { shortHash } from '../../lib/format';
  import { count } from '../../lib/calibration';
  import { IDS_SHOWN, pValueText, type PairedQuality, type QualitySide } from '../../lib/buildreports';

  let { report }: { report: PairedQuality } = $props();
  const t = $derived(report.transitions);
  const u = $derived(report.uncertainty);
  const acc = (x: QualitySide) => {
    const sc = x.score;
    if (!sc || sc.correct === null || sc.total === null) return 'Not reported';
    const pct = sc.accuracyBasisPoints !== null ? ` (${(sc.accuracyBasisPoints / 100).toFixed(2)}%)` : '';
    return `${sc.correct.toLocaleString()} / ${sc.total.toLocaleString()}${pct}`;
  };
  const signed = (v: number | null, digits = 0) => (v === null ? 'Not reported' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}`);
  const sameSource = $derived(report.left.sourceSha256 && report.right.sourceSha256 ? report.left.sourceSha256 === report.right.sourceSha256 : null);
</script>

{#snippet ids(title: string, list: number[], expected: number | null)}
  <div>
    <h4 class="h">{title} <span class="num">({list.length}{expected !== null && expected !== list.length ? ` listed, ${expected} reported` : ''})</span></h4>
    {#if list.length === 0}
      <p class="xs muted">None listed.</p>
    {:else}
      <p class="xs mono break">{list.slice(0, IDS_SHOWN).join(', ')}{list.length > IDS_SHOWN ? ' …' : ''}</p>
      {#if list.length > IDS_SHOWN}
        <details><summary class="xs">All {list.length} question IDs</summary><p class="xs mono break">{list.join(', ')}</p></details>
      {/if}
    {/if}
  </div>
{/snippet}

<div class="stack">
  <div class="row">
    {#if report.synthetic === true}<Badge tone="warn" label="Synthetic fixture" />{:else if report.synthetic === null}<Badge tone="muted" label="Real data not stated" />{/if}
    {#if report.datasetFull === false}<Badge tone="warn" label="Not the full split" />{:else if report.datasetFull === null}<Badge tone="muted" label="Full split not stated" />{/if}
    {#if sameSource === false}<Badge tone="warn" label="Different source checkpoints" />{:else if sameSource === null}<Badge tone="muted" label="Source match unknown" />{/if}
    <Badge tone="muted" label="Not a release qualification" />
  </div>
  <p class="note info">
    The same GSM8K questions answered by two evaluation jobs. Delta is candidate (right) minus reference (left). The p-value is from an exact two-sided McNemar test on the questions where only one side is correct. A large p-value means the paired data does not show a difference; it does not show that the builds are equal. This report does not say why answers changed.
  </p>
  {#if report.warnings.length}
    <div class="note warn"><strong class="xs">Warnings</strong><ul class="xs">{#each report.warnings as w, i (i)}<li>{w}</li>{/each}</ul></div>
  {/if}

  <div class="table-wrap">
    <table class="data ids">
      <caption class="sr-only">Reference and candidate evaluation</caption>
      <thead><tr><th scope="col"></th><th scope="col">Reference (left)</th><th scope="col">Candidate (right)</th></tr></thead>
      <tbody>
        <tr><th scope="row">Job</th><td class="mono xs break">{report.left.jobId ?? 'Not stated'}</td><td class="mono xs break">{report.right.jobId ?? 'Not stated'}</td></tr>
        <tr><th scope="row">Host</th><td class="xs">{report.left.host ?? 'Not stated'}</td><td class="xs">{report.right.host ?? 'Not stated'}</td></tr>
        <tr><th scope="row">Score</th><td class="xs num">{acc(report.left)}</td><td class="xs num">{acc(report.right)}</td></tr>
        <tr><th scope="row">Unparsed</th><td class="xs num">{count(report.left.score?.unparsed ?? null)}</td><td class="xs num">{count(report.right.score?.unparsed ?? null)}</td></tr>
        <tr><th scope="row">Source SHA-256</th><td class="mono xs" title={report.left.sourceSha256 ?? ''}>{shortHash(report.left.sourceSha256)}</td><td class="mono xs" title={report.right.sourceSha256 ?? ''}>{shortHash(report.right.sourceSha256)}</td></tr>
        <tr><th scope="row">Manifest SHA-256</th><td class="mono xs" title={report.left.manifestSha256 ?? ''}>{shortHash(report.left.manifestSha256)}</td><td class="mono xs" title={report.right.manifestSha256 ?? ''}>{shortHash(report.right.manifestSha256)}</td></tr>
        <tr><th scope="row">Provenance</th><td class="xs">{report.left.provenanceMethod ?? 'Not stated'}</td><td class="xs">{report.right.provenanceMethod ?? 'Not stated'}</td></tr>
      </tbody>
    </table>
  </div>
  <p class="xs muted">
    Workload: {report.dataset.name ?? 'dataset not stated'}{report.dataset.split ? ` · ${report.dataset.split}` : ''}{report.dataset.rows !== null ? ` · ${report.dataset.rows.toLocaleString()} questions` : ''}{report.protocolId ? ` · protocol ${report.protocolId}` : ''}{report.dataset.sha256 ? ` · data ${shortHash(report.dataset.sha256)}` : ''}
  </p>

  <dl class="totals small num">
    <div><dt>Delta correct</dt><dd>{signed(report.deltaCorrect)}</dd></div>
    <div><dt>Delta</dt><dd>{report.deltaPp === null ? 'Not reported' : `${signed(report.deltaPp, 2)} pp`}</dd></div>
    <div><dt>Gains (wrong → right)</dt><dd>{count(t.improvements)}</dd></div>
    <div><dt>Regressions (right → wrong)</dt><dd>{count(t.regressions)}</dd></div>
    <div><dt>Both correct</dt><dd>{count(t.bothCorrect)}</dd></div>
    <div><dt>Both incorrect</dt><dd>{count(t.bothIncorrect)}</dd></div>
    <div><dt>Discordant pairs</dt><dd>{count(u.discordantPairs)}</dd></div>
    <div><dt>p-value{u.method ? ` (${u.method})` : ''}</dt><dd>{pValueText(u)}</dd></div>
  </dl>
  {#if u.assumption}<p class="xs muted">Assumption: {u.assumption}</p>{/if}

  <div class="grid-3">
    {@render ids('Gains, question IDs', report.improvementIds, t.improvements)}
    {@render ids('Regressions, question IDs', report.regressionIds, t.regressions)}
    {@render ids('Parsed answer changed', report.parsedChangeIds, null)}
  </div>
  <p class="xs muted">Question IDs are 1-based positions in the pinned split. No question or answer text is in this report.</p>
</div>

<style>
  .note ul {
    margin: 4px 0 0;
    padding-left: 1.1em;
  }
  .h {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
    margin-bottom: 4px;
  }
  .ids td {
    overflow-wrap: anywhere;
  }
  .ids th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
    font-weight: 500;
    white-space: nowrap;
  }
  .totals {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-5);
    margin: 0;
  }
  .totals dt {
    color: var(--text-2);
    font-size: var(--fs-xs);
  }
  .totals dd {
    margin: 0;
    font-weight: 600;
  }
</style>

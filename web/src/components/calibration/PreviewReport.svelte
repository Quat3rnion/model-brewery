<script lang="ts">
  import Badge from '../Badge.svelte';
  import Ids from './Ids.svelte';
  import { formatBytes, shortHash } from '../../lib/format';
  import { CHANGES_PAGE, codecTransitions, count, signedBytes, type PreviewReport } from '../../lib/calibration';

  let { report }: { report: PreviewReport } = $props();
  let shown = $state(CHANGES_PAGE);
  const transitions = $derived(codecTransitions(report.changes));
  const shape = (v: number[] | null | undefined) => (v ? v.join('×') : '—');
</script>

<div class="stack">
  <div class="row">
    <Badge tone="muted" label="Not runnable" />
    <Badge tone="muted" label="Not release-eligible" />
    {#each report.compatibility.filter(([k]) => k === 'status' || k === 'weight_identity') as [k, v] (k)}
      <Badge tone="warn" label="{k.replace(/_/g, ' ')}: {v.replace(/_/g, ' ')}" />
    {/each}
  </div>
  {#if report.claimedRunnable}
    <p class="note warn">The file marks itself executable or release-eligible. The viewer ignores this: a profile preview is never a runnable profile.</p>
  {/if}
  <p class="note info">
    A size prediction from two saved plans and a recipe. No weights were read, and no usage or trace data was used. It does not say which tensors should change, and it is not a quality result.
  </p>
  {#if report.limits.length}
    <div class="note warn">
      <strong class="xs">Limits</strong>
      <ul class="xs">{#each report.limits as l, i (i)}<li>{l}</li>{/each}</ul>
    </div>
  {/if}

  <div class="grid-3">
    <Ids title="Base plan" items={report.base} />
    <Ids title="Alternate plan" items={report.alternate} />
    <Ids title="Source metadata" items={report.compatibility.filter(([k]) => k !== 'status' && k !== 'weight_identity')} />
  </div>
  {#if report.recipeSha256}<p class="xs muted">Resolved recipe SHA-256 <span class="mono" title={report.recipeSha256}>{shortHash(report.recipeSha256)}</span></p>{/if}

  <dl class="totals small num">
    <div><dt>Base HGN size</dt><dd>{formatBytes(report.size.base)}</dd></div>
    <div><dt>Predicted HGN size</dt><dd>{formatBytes(report.size.predicted)}</dd></div>
    <div><dt>Change</dt><dd>{signedBytes(report.size.delta)}</dd></div>
    <div><dt>Changed entries</dt><dd>{count(report.changes.length)}</dd></div>
  </dl>

  {#if report.files.length}
    <div class="table-wrap">
      <table class="data stackable">
        <caption class="sr-only">Predicted size per HGN file</caption>
        <thead><tr><th scope="col">File</th><th scope="col" class="r">Base</th><th scope="col" class="r">Predicted</th><th scope="col" class="r">Change</th><th scope="col" class="r">Overhead base / predicted</th></tr></thead>
        <tbody>
          {#each report.files as f, i (i)}
            <tr>
              <th scope="row" class="mono break">{f.filename}</th>
              <td class="num r" data-label="Base">{formatBytes(f.base)}</td>
              <td class="num r" data-label="Predicted">{formatBytes(f.predicted)}</td>
              <td class="num r" data-label="Change">{signedBytes(f.delta)}</td>
              <td class="num r" data-label="Overhead base / predicted">{formatBytes(f.baseOverhead)} / {formatBytes(f.predictedOverhead)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  {#if transitions.length}
    <div class="table-wrap">
      <table class="data stackable">
        <caption class="sr-only">Encoding changes grouped by kind</caption>
        <thead><tr><th scope="col">Encoding</th><th scope="col">Kind</th><th scope="col" class="r">Entries</th><th scope="col" class="r">Payload change</th></tr></thead>
        <tbody>
          {#each transitions as t (t.kind + t.from + t.to)}
            <tr>
              <th scope="row" class="mono">{t.from === t.to ? `${t.from} (unchanged)` : `${t.from} → ${t.to}`}</th>
              <td data-label="Kind">{t.kind}</td>
              <td class="num r" data-label="Entries">{count(t.count)}</td>
              <td class="num r" data-label="Payload change">{signedBytes(t.delta)}{#if t.unknown}<span class="xs muted"> ({t.unknown} unknown)</span>{/if}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  {#if report.changes.length}
    <details>
      <summary class="small">Changed entries ({report.changes.length})</summary>
      <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard scrolling of wide changes) -->
      <div class="table-wrap" role="region" aria-label="Changed entries" tabindex="0">
        <table class="data changes">
          <caption class="sr-only">Each changed HGN entry, before and after</caption>
          <thead><tr><th scope="col">Tensor</th><th scope="col">Layer · role · kind</th><th scope="col">Before</th><th scope="col">After</th><th scope="col" class="r">Change</th></tr></thead>
          <tbody>
            {#each report.changes.slice(0, shown) as c, i (i)}
              <tr>
                <th scope="row" class="mono xs break">{c.tensor}<div class="muted">{c.filename}</div></th>
                <td class="xs">{c.layer ?? '—'} · {c.role || '—'} · {c.kind}</td>
                <td class="xs num"><span class="mono">{c.before?.encoding ?? '—'}</span> {shape(c.before?.shape)}<div class="muted">{formatBytes(c.before?.payload_bytes)}</div></td>
                <td class="xs num"><span class="mono">{c.after?.encoding ?? '—'}</span> {shape(c.after?.shape)}<div class="muted">{formatBytes(c.after?.payload_bytes)}</div></td>
                <td class="num r xs">{signedBytes(c.delta)}{#if c.shapeChanged}<div class="muted">shape changed</div>{/if}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      {#if report.changes.length > shown}
        <button type="button" class="btn small" onclick={() => (shown += CHANGES_PAGE)}>Show {Math.min(CHANGES_PAGE, report.changes.length - shown)} more of {report.changes.length - shown}</button>
      {/if}
    </details>
  {:else}
    <p class="muted small">The recipe has no changes.</p>
  {/if}
</div>

<style>
  .note ul {
    margin: 4px 0 0;
    padding-left: 1.1em;
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
  .r {
    text-align: right;
  }
  th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
    font-weight: 500;
  }
  details > .table-wrap {
    margin: var(--s-2) 0;
  }
  @media (max-width: 640px) {
    .r {
      text-align: left;
    }
  }
</style>

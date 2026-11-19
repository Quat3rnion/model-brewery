<script lang="ts">
  import { count, fixed, pct, topExperts, usageCoverage, type Phase, type UsageReport } from '../../lib/calibration';
  import { shortHash } from '../../lib/format';
  import Ids from './Ids.svelte';

  let { report }: { report: UsageReport } = $props();
  const uid = $props.id();
  let open = $state<number | null>(null);
  const t = $derived(report.totals);
  const TRACES_SHOWN = 20;

  function distinct(p: Phase | null): string {
    return p ? String(p.experts.length) : '—';
  }
</script>

{#snippet coverage(p: Phase | null)}
  {#if !p}
    <span class="muted">Not captured</span>
  {:else}
    {@const c = usageCoverage(p)}
    <span class="num">{p.observed_positions === 0 ? 'None observed' : pct(c)}</span>
    <span class="xs muted num">{count(p.observed_positions)} observed · {count(p.missing_positions)} missing</span>
  {/if}
{/snippet}

{#snippet experts(name: string, p: Phase | null)}
  <div class="ex">
    <h4 class="h">{name}</h4>
    {#if !p}
      <p class="xs muted">Not captured.</p>
    {:else if p.experts.length === 0}
      <p class="xs muted">No expert selections recorded.</p>
    {:else}
      {@const top = topExperts(p.experts)}
      <table class="data tight">
        <caption class="sr-only">{name}: most selected experts</caption>
        <thead>
          <tr><th scope="col">Expert</th><th scope="col" class="r">Selections</th><th scope="col" class="r">Selection share</th><th scope="col" class="r">Mean router coeff.</th><th scope="col" class="r">Coeff. samples</th></tr>
        </thead>
        <tbody>
          {#each top.shown as e (e.expert)}
            <tr>
              <th scope="row" class="mono">{e.expert}</th>
              <td class="num r">{count(e.selections)}</td>
              <td class="num r">{pct(e.selection_fraction)}</td>
              <td class="num r">{fixed(e.weight_mean)}</td>
              <td class="num r">{count(e.weight_samples)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      {#if top.hidden}<p class="xs muted">{top.hidden} more selected experts are in the file but not listed.</p>{/if}
    {/if}
  </div>
{/snippet}

<div class="stack">
  <p class="note info">
    Selection share is selections divided by observed routing positions of that layer and phase. Mean router coefficient is the average captured coefficient of the selected expert. It is not a probability and not a model weight. Missing positions are excluded, not counted as zero.
  </p>
  {#if report.caveats.length}
    <div class="note warn">
      <strong class="xs">Report caveats</strong>
      <ul class="xs">{#each report.caveats as c, i (i)}<li>{c}</li>{/each}</ul>
    </div>
  {/if}

  <div class="grid-3 ids">
    <Ids title="Model" items={report.identity.model} />
    <Ids title="Runtime" items={report.identity.runtime} />
    <Ids title="Tokenizer" items={report.identity.tokenizer} />
  </div>

  {#if t.observed_positions === 0}
    <p class="note warn">No routing was observed in any layer. A dense model has no routed experts; missing routing is not evidence that experts are unused.</p>
  {/if}

  <dl class="totals small num">
    <div><dt>Traces</dt><dd>{count(t.trace_count)}</dd></div>
    <div><dt>Prefill tokens</dt><dd>{count(t.prefill_tokens)}</dd></div>
    <div><dt>Decode tokens</dt><dd>{count(t.decode_tokens)}</dd></div>
    <div><dt>Observed positions</dt><dd>{count(t.observed_positions)}</dd></div>
    <div><dt>Missing positions</dt><dd>{count(t.missing_positions)}</dd></div>
    <div><dt>Selections</dt><dd>{count(t.selections)}</dd></div>
  </dl>
  {#if report.traces.length}
    <details>
      <summary class="xs">Sample traces ({report.traces.length})</summary>
      <ul class="traces xs">
        {#each report.traces.slice(0, TRACES_SHOWN) as tr, i (i)}
          <li><span class="mono" title={tr.sha256 ?? ''}>{shortHash(tr.sha256)}</span> <span class="muted num">{count(tr.prefill_tokens)} prefill · {count(tr.decode_tokens)} decode</span></li>
        {/each}
      </ul>
      {#if report.traces.length > TRACES_SHOWN}<p class="xs muted">{report.traces.length - TRACES_SHOWN} more traces are not listed.</p>{/if}
    </details>
  {/if}

  {#if report.layers.length === 0}
    <p class="muted small">The report has no layers.</p>
  {:else}
    <div class="table-wrap">
      <table class="data layers stackable">
        <caption class="sr-only">Routing coverage per layer and phase</caption>
        <thead>
          <tr>
            <th scope="col">Layer</th>
            <th scope="col">Prefill coverage</th>
            <th scope="col" class="r">Prefill experts selected</th>
            <th scope="col">Decode coverage</th>
            <th scope="col" class="r">Decode experts selected</th>
            <th scope="col"><span class="sr-only">Details</span></th>
          </tr>
        </thead>
        <tbody>
          {#each report.layers as l (l.layer)}
            <tr>
              <th scope="row" class="mono">{l.layer}</th>
              <td data-label="Prefill coverage"><div class="cov">{@render coverage(l.prefill)}</div></td>
              <td data-label="Prefill distinct experts" class="num r">{distinct(l.prefill)}</td>
              <td data-label="Decode coverage"><div class="cov">{@render coverage(l.decode)}</div></td>
              <td data-label="Decode distinct experts" class="num r">{distinct(l.decode)}</td>
              <td>
                {#if (l.prefill?.experts.length ?? 0) + (l.decode?.experts.length ?? 0) > 0}
                <button type="button" class="btn small" aria-expanded={open === l.layer} aria-controls="{uid}-l{l.layer}" onclick={() => (open = open === l.layer ? null : l.layer)}>
                  {open === l.layer ? 'Hide experts' : 'Experts'}<span class="sr-only"> for layer {l.layer}</span>
                </button>
                {/if}
              </td>
            </tr>
            {#if open === l.layer}
              <tr class="detail" id="{uid}-l{l.layer}">
                <td colspan="6">
                  <div class="grid-2">
                    {@render experts('Prefill', l.prefill)}
                    {@render experts('Decode', l.decode)}
                  </div>
                </td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
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
  .traces {
    list-style: none;
    margin: 6px 0 0;
    padding: 0;
  }
  .cov {
    display: flex;
    flex-direction: column;
  }
  .r {
    text-align: right;
  }
  .layers th[scope='row'],
  .tight th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
  }
  .detail td {
    background: var(--bg);
  }
  .tight th,
  .tight td {
    padding: 4px 8px;
  }
  .h {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
    margin-bottom: 4px;
  }
  .ex {
    min-width: 0;
    overflow-x: auto;
  }
  @media (max-width: 640px) {
    .r {
      text-align: left;
    }
  }
</style>

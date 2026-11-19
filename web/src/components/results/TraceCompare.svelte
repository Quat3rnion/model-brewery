<script lang="ts">
  import KeyValues from '../KeyValues.svelte';
  import Badge from '../Badge.svelte';
  import SignalPanel from './SignalPanel.svelte';
  import { obj } from '../../lib/reports';
  import { tokenRows, comparability } from '../../lib/trace';

  let { result }: { result: Record<string, unknown> } = $props();
  const traces = $derived(Array.isArray(result.traces) ? (result.traces as unknown[]).map((t) => obj(t) ?? {}) : []);
  const alignment = $derived(result.alignment);
  const verdict = $derived(comparability(alignment));
</script>

<div class="stack">
  <div class="note {verdict.comparable ? 'info' : 'warn'}">
    <div class="row">
      <Badge
        tone={verdict.status === 'aligned' ? 'ok' : 'warn'}
        label={verdict.status === 'aligned' ? 'Aligned' : verdict.status === 'prefix_only' ? `Same input prefix: ${verdict.prefix ?? '?'} tokens` : 'Side by side only'}
      />
    </div>
    <p class="small" style="margin-top:6px">{verdict.text}</p>
    <p class="xs muted" style="margin-top:4px">Layer signals are shown per model. They are not subtracted across models.</p>
  </div>

  <div class="traces" style:--n={Math.max(1, traces.length)}>
    {#each traces as trace, i (i)}
      {@const rows = tokenRows(trace)}
      <section class="panel trace" aria-labelledby="trace-{i}">
        <header>
          <h3 id="trace-{i}">{i === 0 ? 'Model A' : i === 1 ? 'Model B' : `Model ${i + 1}`}</h3>
        </header>
        <div class="body stack">
          <div>
            <h4 class="sub">Model identity</h4>
            <KeyValues data={trace.meta ?? trace.input} />
          </div>
          {#if trace.template !== undefined}
            <div>
              <h4 class="sub">Template</h4>
              {#if typeof trace.template === 'string'}<pre class="block">{trace.template}</pre>{:else}<KeyValues data={trace.template} />{/if}
            </div>
          {/if}
          {#if trace.execution !== undefined}
            <div>
              <h4 class="sub">Execution and settings</h4>
              <KeyValues data={trace.execution} />
            </div>
          {/if}
          <div>
            <h4 class="sub">Output</h4>
            {#if typeof trace.output_text === 'string'}
              <p class="output">{trace.output_text || '—'}</p>
              <p class="xs muted" style="margin-top:4px">Decoded text as reported by the engine.</p>
            {:else}
              <p class="output">{rows.filter((r) => r.phase !== 'prefill').map((r) => r.text).join('') || '—'}</p>
              <p class="xs muted" style="margin-top:4px">Joined from decode tokens (older receipt without output text).</p>
            {/if}
          </div>
          <div>
            <h4 class="sub">Tokens by position</h4>
            {#if rows.length}
              <div class="table-wrap tokens">
                <table class="data">
                  <caption class="sr-only">Tokens by position for this model</caption>
                  <thead><tr><th scope="col">Pos</th><th scope="col"><span class="sr-only">Same input as other model</span></th><th scope="col">Phase</th><th scope="col">ID</th><th scope="col">Token</th><th scope="col">Next candidates</th></tr></thead>
                  <tbody>
                    {#each rows as r, ri (r.key)}
                      {@const paired = verdict.status === 'aligned' || (verdict.status === 'prefix_only' && verdict.prefix !== null && ri < verdict.prefix)}
                      <tr class:paired>
                        <td class="num mono">{r.pos}</td>
                        <td class="xs">{#if paired}<span class="pair" title="Same input token as the other model at this position">same</span>{/if}</td>
                        <td class="xs muted">{r.phase}</td>
                        <td class="num mono">{r.id ?? '—'}</td>
                        <td class="mono tok">{JSON.stringify(r.text)}</td>
                        <td class="mono xs">{r.candidates || '—'}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {:else}
              <p class="muted small">No tokens in this trace.</p>
            {/if}
          </div>
          <div>
            <h4 class="sub">Layer signals</h4>
            <SignalPanel raw={trace.signals} {rows} label={i === 0 ? 'model A' : i === 1 ? 'model B' : `model ${i + 1}`} />
          </div>
          {#if trace.viz_file}
            <p class="xs muted">Trace file on worker: <span class="mono break">{String(trace.viz_file)}</span></p>
          {/if}
        </div>
      </section>
    {/each}
  </div>

  {#if alignment !== undefined && alignment !== null}
    <details><summary class="small">Alignment record</summary><pre class="block">{JSON.stringify(alignment, null, 2)}</pre></details>
  {/if}
</div>

<style>
  .traces {
    display: grid;
    grid-template-columns: repeat(var(--n), minmax(0, 1fr));
    gap: var(--s-4);
  }
  @media (max-width: 900px) {
    .traces {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .sub {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
    margin-bottom: 6px;
  }
  .output {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-size: var(--fs-md);
    padding: var(--s-3);
    background: var(--bg);
    border-radius: var(--r-sm);
    border: 1px solid var(--line);
  }
  .tokens {
    max-height: 360px;
    overflow: auto;
  }
  .pair {
    color: var(--ok);
    font-weight: 600;
  }
  tr.paired td:first-child {
    box-shadow: inset 3px 0 0 var(--ok);
  }
  .tok {
    white-space: pre;
  }
</style>

<script lang="ts">
  // Results of one succeeded trace job: summary, layer x token activation
  // norms, logit lens, model logits and attention. Every slice is a GET.
  import { api, errorText } from '../../lib/api';
  import { formatBytes, shortHash } from '../../lib/format';
  import { obj } from '../../lib/reports';
  import {
    ROLE_LABELS,
    attentionMatrix,
    fmt,
    lensGrid,
    normMatrix,
    type Matrix,
    type TopLogit,
    type TraceSummary,
    type TraceTensor,
    type TraceRole,
  } from '../../lib/weightspace';
  import MatrixHeat from './MatrixHeat.svelte';

  let { jobId, result, hasSae = false }: { jobId: string; result: unknown; hasSae?: boolean } = $props();

  const promptCount = $derived(Math.max(1, (obj(result)?.prompts as unknown[] | undefined)?.length ?? 1));
  let prompt = $state(0);
  let summary = $state<TraceSummary | null>(null);
  let summaryError = $state<string | null>(null);

  $effect(() => {
    const id = jobId;
    const p = prompt;
    summary = null;
    summaryError = null;
    const ac = new AbortController();
    api
      .trace(id, { prompt: p }, '', ac.signal)
      .then((s) => (summary = s as TraceSummary))
      .catch((e) => {
        if (!ac.signal.aborted) summaryError = errorText(e);
      });
    return () => ac.abort();
  });

  const tokenAt = $derived(new Map((summary?.tokens ?? []).map((t) => [t.position, t.token_id])));
  const posLabel = (p: number) => (tokenAt.has(p) ? `${p} (#${tokenAt.get(p)})` : String(p));
  const plainRoles = $derived((summary?.spec.roles ?? []).filter((r) => r !== 'kq_soft_max') as TraceRole[]);
  const hasAttention = $derived(summary?.spec.roles.includes('kq_soft_max') ?? false);

  type FeatureRow = { position: number; top: { feature: number; activation: number }[] };
  let featurePosition = $state<number | undefined>(undefined);
  let featureRow = $state<FeatureRow | undefined>(undefined);
  let featureError = $state<string | null>(null);
  let featureBusy = $state(false);
  const capturedPositions = $derived((summary?.tokens ?? []).map((t) => t.position)
    .filter((p) => summary?.spec.positions.includes(p)));
  $effect(() => {
    const s = summary;
    const id = jobId;
    const position = featurePosition;
    featureRow = undefined;
    featureError = null;
    featureBusy = false;
    if (!hasSae || !s || position === undefined || !capturedPositions.includes(position)) return;
    const ac = new AbortController();
    featureBusy = true;
    api.trace(id, { prompt: s.prompt, start: position, end: position + 1 }, 'features', ac.signal)
      .then((data) => {
        if (!ac.signal.aborted) featureRow = (data as { rows: FeatureRow[] }).rows[0];
      })
      .catch((e) => {
        if (!ac.signal.aborted) featureError = errorText(e);
      })
      .finally(() => {
        if (!ac.signal.aborted) featureBusy = false;
      });
    return () => ac.abort();
  });

  // ---- Activation norms ----
  let role = $state<TraceRole | ''>('');
  $effect(() => {
    if (plainRoles.length && (!role || !plainRoles.includes(role))) role = plainRoles[0];
  });
  let norms = $state<Matrix | null>(null);
  let normsBusy = $state(false);
  let normsError = $state<string | null>(null);
  let normCell = $state<{ r: number; c: number; row: number; col: number; value: number | null } | null>(null);

  async function tensors(layers: number[], r: string, signal: AbortSignal): Promise<(TraceTensor & { layer: number })[]> {
    const out: (TraceTensor & { layer: number })[] = [];
    // At most four requests at a time: each one reads files on the worker.
    for (let i = 0; i < layers.length; i += 4) {
      const part = await Promise.all(
        layers.slice(i, i + 4).map(async (layer) => ({ ...((await api.trace(jobId, { prompt, layer, role: r }, 'tensor', signal)) as TraceTensor), layer })),
      );
      out.push(...part);
    }
    return out;
  }

  $effect(() => {
    const s = summary;
    const r = role;
    norms = null;
    normCell = null;
    normsError = null;
    if (!s || !r || !s.spec.layers.length) return;
    const ac = new AbortController();
    normsBusy = true;
    tensors(s.spec.layers, r, ac.signal)
      .then((list) => (norms = normMatrix(list, s.spec.positions)))
      .catch((e) => {
        if (!ac.signal.aborted) normsError = errorText(e);
      })
      .finally(() => {
        if (!ac.signal.aborted) normsBusy = false;
      });
    return () => ac.abort();
  });

  // ---- Logit lens and model logits ----
  let lens = $state<ReturnType<typeof lensGrid> | null>(null);
  let logits = $state<{ position: number; top: TopLogit[] }[] | null>(null);
  let logitError = $state<string | null>(null);
  let lensPick = $state<{ layer: number; position: number } | null>(null);
  $effect(() => {
    const s = summary;
    lens = null;
    logits = null;
    lensPick = null;
    logitError = null;
    if (!s || s.spec.top_k <= 0) return;
    const ac = new AbortController();
    const q = { prompt: s.prompt };
    Promise.all([
      s.logit_lens_count ? api.trace(jobId, q, 'logit-lens', ac.signal) : Promise.resolve({ rows: [] }),
      s.model_logits_count ? api.trace(jobId, q, 'model-logits', ac.signal) : Promise.resolve({ rows: [] }),
    ])
      .then(([l, m]) => {
        lens = lensGrid(((obj(l)?.rows as unknown[]) ?? []) as { layer: number; position: number; top: TopLogit[] }[]);
        logits = ((obj(m)?.rows as unknown[]) ?? []) as { position: number; top: TopLogit[] }[];
      })
      .catch((e) => {
        if (!ac.signal.aborted) logitError = errorText(e);
      });
    return () => ac.abort();
  });
  const lensTop = $derived(lensPick && lens ? (lens.at(lensPick.layer, lensPick.position) ?? []) : []);

  // ---- Attention ----
  let attnLayer = $state<number | null>(null);
  let head = $state(0);
  let attn = $state<TraceTensor | null>(null);
  let attnError = $state<string | null>(null);
  $effect(() => {
    if (summary && hasAttention && (attnLayer === null || !summary.spec.layers.includes(attnLayer))) attnLayer = summary.spec.layers[0] ?? null;
  });
  $effect(() => {
    const s = summary;
    const layer = attnLayer;
    attn = null;
    attnError = null;
    if (!s || !hasAttention || layer === null) return;
    const ac = new AbortController();
    api
      .trace(jobId, { prompt: s.prompt, layer, role: 'kq_soft_max' }, 'tensor', ac.signal)
      .then((t) => (attn = t as TraceTensor))
      .catch((e) => {
        if (!ac.signal.aborted) attnError = errorText(e);
      });
    return () => ac.abort();
  });
  const attnMatrix = $derived(attn ? attentionMatrix(attn, head) : null);
  const uid = $props.id();
</script>

<div class="stack">
  {#if promptCount > 1}
    <div class="field narrow">
      <label for="{uid}-prompt">Prompt</label>
      <select id="{uid}-prompt" class="input" bind:value={prompt}>
        {#each Array.from({ length: promptCount }, (_, i) => i) as i (i)}<option value={i}>Prompt {i + 1}</option>{/each}
      </select>
    </div>
  {/if}

  {#if summaryError}
    <p class="note bad" role="alert">{summaryError}</p>
  {:else if !summary}
    <p class="muted small" aria-live="polite">Loading trace summary…</p>
  {:else}
    <dl class="facts small">
      <div><dt>Model</dt><dd class="mono break">{summary.model_id}</dd></div>
      <div><dt>Model SHA-256</dt><dd class="mono" title={summary.model_sha256}>{shortHash(summary.model_sha256)}</dd></div>
      <div><dt>Lanternfish</dt><dd class="mono" title={summary.lanternfish_commit}>{shortHash(summary.lanternfish_commit)}</dd></div>
      <div><dt>Export</dt><dd class="num">{formatBytes(summary.bytes)}{#if summary.spec.byte_cap}{` of ${formatBytes(summary.spec.byte_cap)} cap`}{/if}</dd></div>
      <div><dt>Tokens</dt><dd class="num">{summary.tokens.length}</dd></div>
      <div><dt>Tensors</dt><dd class="num">{summary.tensor_count}{#if summary.missing_count}{` (${summary.missing_count} not captured)`}{/if}</dd></div>
      <div><dt>Interventions</dt><dd class="num">{summary.intervention_count}</dd></div>
    </dl>
    {#if summary.intervention_count}
      <p class="note info small">This trace ran with {summary.intervention_count} activation interventions. Values after an edited node show the edited run.</p>
    {/if}

    {#if hasSae}
      <section class="sub" aria-labelledby="{uid}-sae-h">
        <h3 id="{uid}-sae-h">Top SAE features</h3>
        <div class="field inline">
          <label for="{uid}-sae-pos">Token position</label>
          <select id="{uid}-sae-pos" class="input" bind:value={featurePosition}>
            <option value={undefined}>Select a position</option>
            {#each capturedPositions as p (p)}<option value={p}>{posLabel(p)}</option>{/each}
          </select>
        </div>
        {#if featureError}
          <p class="note bad" role="alert">{featureError}</p>
        {:else if featureBusy}
          <p class="muted small" aria-live="polite">Reading features…</p>
        {:else if featureRow?.top.length}
          <ul class="toplist num" aria-label="Feature IDs and activations">
            {#each featureRow.top as f (f.feature)}<li>#{f.feature} {fmt(f.activation)}</li>{/each}
          </ul>
        {:else if featurePosition !== undefined}
          <p class="empty">No active features at this position.</p>
        {/if}
      </section>
    {/if}

    <section class="sub" aria-labelledby="{uid}-norm-h">
      <div class="sub-head">
        <h3 id="{uid}-norm-h">Activation norm per layer and token</h3>
        {#if plainRoles.length > 1}
          <div class="field inline">
            <label for="{uid}-role">Role</label>
            <select id="{uid}-role" class="input" bind:value={role}>
              {#each plainRoles as r (r)}<option value={r}>{ROLE_LABELS[r]}</option>{/each}
            </select>
          </div>
        {:else if plainRoles.length}
          <span class="xs muted">{ROLE_LABELS[plainRoles[0]]}</span>
        {/if}
      </div>
      {#if !plainRoles.length}
        <p class="empty">This trace has no activation roles. It has logits only.</p>
      {:else if normsError}
        <p class="note bad" role="alert">{normsError}</p>
      {:else if normsBusy || !norms}
        <p class="muted small" aria-live="polite">Reading {summary.spec.layers.length} layers…</p>
      {:else}
        <MatrixHeat matrix={norms} label="L2 norm of {role} per layer and token position" rowName="Layer" colName="Position" colLabel={posLabel} selected={normCell} onselect={(c) => (normCell = c)} />
        <p class="small" aria-live="polite">
          {#if normCell}Layer {normCell.row}, position {posLabel(normCell.col)}: L2 norm <span class="num">{fmt(normCell.value)}</span>.{:else}<span class="muted">Select a cell to read its value.</span>{/if}
        </p>
      {/if}
    </section>

    {#if summary.spec.top_k > 0}
      <section class="sub" aria-labelledby="{uid}-lens-h">
        <h3 id="{uid}-lens-h">Logit lens</h3>
        <p class="xs muted">Top token ID per layer and position after the output projection of l_out. The trace has token IDs only, not token text.</p>
        {#if logitError}
          <p class="note bad" role="alert">{logitError}</p>
        {:else if !lens}
          <p class="muted small" aria-live="polite">Loading logits…</p>
        {:else if !lens.layers.length}
          <p class="empty">No logit lens rows in this trace.</p>
        {:else}
          <div class="table-wrap lens">
            <table class="data num">
              <caption class="sr-only">Top-1 token ID per layer (rows) and position (columns). Select a cell to show the top {summary.spec.top_k}.</caption>
              <thead><tr><th scope="col">Layer</th>{#each lens.positions as p (p)}<th scope="col">Pos {posLabel(p)}</th>{/each}</tr></thead>
              <tbody>
                {#each lens.layers as layer (layer)}
                  <tr>
                    <th scope="row">{layer}</th>
                    {#each lens.positions as p (p)}
                      {@const top = lens.at(layer, p)}
                      <td>
                        {#if top?.length}
                          <button type="button" class="cellbtn" aria-pressed={lensPick?.layer === layer && lensPick?.position === p} aria-label="#{top[0].token_id} {fmt(top[0].logit, 3)}, layer {layer}, position {p}" onclick={() => (lensPick = { layer, position: p })}>
                            #{top[0].token_id} <span class="muted">{fmt(top[0].logit, 3)}</span>
                          </button>
                        {:else}—{/if}
                      </td>
                    {/each}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          {#if lensPick}
            <div class="topk" aria-live="polite">
              <p class="small">Layer {lensPick.layer}, position {posLabel(lensPick.position)}: top {lensTop.length}</p>
              <ol class="toplist num">{#each lensTop as t, i (i)}<li><span class="muted">{i + 1}.</span> <span class="mono">#{t.token_id}</span> {fmt(t.logit)}</li>{/each}</ol>
            </div>
          {/if}
        {/if}

        <h3 class="h-gap">Model logits</h3>
        {#if logits && logits.length}
          <div class="table-wrap">
            <table class="data num stackable">
              <thead><tr><th scope="col">Position</th><th scope="col">Top tokens (ID and logit)</th></tr></thead>
              <tbody>
                {#each logits as row (row.position)}
                  <tr>
                    <th scope="row">{posLabel(row.position)}</th>
                    <td data-label="Top tokens"><span class="inline-top">{#each row.top as t, i (i)}<span><span class="mono">#{t.token_id}</span> {fmt(t.logit, 3)}</span>{/each}</span></td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {:else if logits}
          <p class="empty">No model logits in this trace.</p>
        {/if}
      </section>
    {/if}

    {#if hasAttention}
      <section class="sub" aria-labelledby="{uid}-attn-h">
        <div class="sub-head">
          <h3 id="{uid}-attn-h">Attention pattern</h3>
          <div class="row">
            <div class="field inline">
              <label for="{uid}-alayer">Layer</label>
              <select id="{uid}-alayer" class="input" bind:value={attnLayer}>
                {#each summary.spec.layers as l (l)}<option value={l}>{l}</option>{/each}
              </select>
            </div>
            {#if attnMatrix}
              <div class="field inline">
                <label for="{uid}-head">Head</label>
                <input id="{uid}-head" class="input num" type="number" min="0" max={attnMatrix.heads - 1} step="1" bind:value={head} inputmode="numeric" />
                <span class="xs muted">of {attnMatrix.heads}</span>
              </div>
            {/if}
          </div>
        </div>
        {#if attnError}
          <p class="note bad" role="alert">{attnError}</p>
        {:else if !attn}
          <p class="muted small" aria-live="polite">Loading attention…</p>
        {:else if !attnMatrix}
          <p class="note warn small">The attention rows do not have a head x key shape, or the head number is out of range.</p>
        {:else}
          <MatrixHeat matrix={attnMatrix} label="Attention probabilities of layer {attnLayer}, head {head}: query position by key position" rowName="Query" colName="Key" />
        {/if}
      </section>
    {/if}
  {/if}
</div>

<style>
  .facts {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-5);
    margin: 0;
  }
  .facts dt {
    color: var(--text-2);
    font-size: var(--fs-xs);
  }
  .facts dd {
    margin: 0;
  }
  .sub {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    padding-top: var(--s-3);
    border-top: 1px solid var(--line-soft);
    min-width: 0;
  }
  .sub-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--s-2) var(--s-3);
  }
  h3 {
    font-size: var(--fs-md);
    font-weight: 600;
  }
  .h-gap {
    margin-top: var(--s-3);
  }
  .field.inline {
    flex-direction: row;
    align-items: center;
    gap: var(--s-2);
  }
  .field.inline .input {
    width: auto;
    min-width: 5rem;
  }
  .narrow {
    max-width: 14rem;
  }
  .lens {
    max-height: 420px;
  }
  .cellbtn {
    border: 1px solid transparent;
    background: none;
    color: inherit;
    font: inherit;
    padding: 2px 4px;
    border-radius: var(--r-sm);
    cursor: pointer;
    white-space: nowrap;
  }
  .cellbtn:hover {
    background: var(--accent-soft);
  }
  .cellbtn[aria-pressed='true'] {
    border-color: var(--accent);
  }
  .toplist {
    display: flex;
    flex-wrap: wrap;
    gap: 4px var(--s-4);
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: var(--fs-sm);
  }
  .inline-top {
    display: flex;
    flex-wrap: wrap;
    gap: 2px var(--s-3);
  }
</style>

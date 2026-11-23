<script lang="ts">
  import { api, errorText } from '../../lib/api';
  import { shortHash } from '../../lib/format';

  interface Sentence {
    index: number;
    text: string;
    logprob_gain: number;
    share_of_net_gain: number | null;
    token_count: number;
  }
  interface Token {
    position: number;
    id: number;
    piece: string;
    parent_logprob: number;
    child_logprob: number;
    loss_delta: number;
  }
  interface Report {
    check_bundle_id: string;
    artifact_sha256: string;
    heldout_sha256: string;
    total_logprob_gain: number;
    total_tokens: number;
    sentences: Sentence[];
    sentence: number;
    start: number;
    end: number;
    tokens: Token[];
    top_tokens: Token[];
  }

  let { parent, child, available }: { parent: string; child: string; available: boolean } = $props();
  let report = $state<Report>();
  let error = $state('');
  let busy = $state(false);
  let active: AbortController | undefined;
  const current = $derived(report?.sentences.find((s) => s.index === report?.sentence));
  const magnitude = $derived(Math.max(0, ...((report?.tokens ?? []).map((t) => Math.abs(t.loss_delta)))));
  const signed = (n: number, digits = 3) => `${n > 0 ? '+' : ''}${n.toFixed(digits)}`;
  const share = (s: Sentence) => s.share_of_net_gain === null ? 'No net change' : `${(s.share_of_net_gain * 100).toFixed(1)}% of net gain`;
  const label = (token: Token) => token.piece.trim() ? token.piece : token.piece ? 'Space' : `Token ${token.id}`;
  const lossLabel = (n: number) => n < 0 ? 'Lower loss' : n > 0 ? 'Higher loss' : 'No change';

  $effect(() => {
    void child;
    return () => active?.abort();
  });

  async function load(sentence?: number, start?: number) {
    active?.abort();
    const request = new AbortController();
    active = request;
    busy = true;
    error = '';
    const q = new URLSearchParams({ a: parent, b: child });
    if (sentence !== undefined) q.set('sentence', String(sentence));
    if (start !== undefined) q.set('start', String(start));
    try {
      const next = await api.diff('token-loss', q.toString(), request.signal) as Report;
      if (!request.signal.aborted) report = next;
    } catch (e) {
      if (!request.signal.aborted) error = errorText(e);
    } finally {
      if (!request.signal.aborted) busy = false;
    }
  }
</script>

{#if available}
  <section class="stack" aria-label="Token loss diff" aria-busy={busy}>
    <div><button type="button" class="btn small" disabled={busy} onclick={() => load()}>Token loss diff</button></div>
    {#if busy}<p class="small muted" role="status">Reading token scores…</p>{/if}
    {#if error}<p class="note bad small" role="alert">{error}</p>{/if}
    {#if report && current}
      <p class="small num">{signed(report.total_logprob_gain)} net log-probability gain across {report.total_tokens} predicted tokens. Not release-qualified.</p>
      <label class="field">
        <span>Heldout sentence</span>
        <select class="input" disabled={busy} value={report.sentence} onchange={(event) => {
          const select = event.currentTarget;
          const next = Number(select.value);
          select.value = String(report?.sentence);
          void load(next);
        }}>
          {#each report.sentences as sentence (sentence.index)}
            <option value={sentence.index}>Sentence {sentence.index + 1} · {signed(sentence.logprob_gain)} · {share(sentence)}</option>
          {/each}
        </select>
      </label>
      <div class="sentence">
        <p class="small">Sentence {current.index + 1}: <strong>{share(current)}</strong> <span class="muted num">({signed(current.logprob_gain)} log-probability gain)</span></p>
        <p>{current.text}</p>
        <p class="xs muted">Tokens below exclude the prompt. Loss change is child minus parent. A negative change means lower loss.</p>
        <div class="pieces" aria-label="Token loss changes">
          {#each report.tokens as token (token.position)}
            <span class="piece" class:gain={token.loss_delta < 0} class:loss={token.loss_delta > 0}
              style:--heat={`${magnitude ? 8 + 20 * Math.abs(token.loss_delta) / magnitude : 0}%`}
              aria-label={`${label(token)}: ${lossLabel(token.loss_delta)}, ${signed(token.loss_delta)}`}
              title={`${lossLabel(token.loss_delta)} (${signed(token.loss_delta)})`}>{label(token)}<small class="num">{signed(token.loss_delta)}</small></span>
          {/each}
        </div>
        {#if report.start > 0 || report.end < current.token_count}
          <div class="row small">
            <button class="btn small" disabled={busy || report.start === 0} onclick={() => load(report?.sentence, Math.max(0, (report?.start ?? 0) - 128))}>Previous tokens</button>
            <span class="num">{report.start + 1}–{report.end} of {current.token_count}</span>
            <button class="btn small" disabled={busy || report.end >= current.token_count} onclick={() => load(report?.sentence, report?.end)}>Next tokens</button>
          </div>
        {/if}
      </div>
      <div class="table-wrap">
        <table class="data num stackable">
          <caption>Top token contributors</caption>
          <thead><tr><th scope="col">Token</th><th scope="col">Parent log probability</th><th scope="col">Child log probability</th><th scope="col">Loss change</th></tr></thead>
          <tbody>
            {#each report.top_tokens as token (token.position)}
              <tr><th scope="row">{label(token)} <span class="muted">#{token.position + 1}</span></th><td data-label="Parent log probability">{token.parent_logprob.toFixed(3)}</td><td data-label="Child log probability">{token.child_logprob.toFixed(3)}</td><td data-label="Loss change">{signed(token.loss_delta)}</td></tr>
            {/each}
          </tbody>
        </table>
      </div>
      <p class="xs muted">Bundle <span class="mono" title={report.check_bundle_id}>{report.check_bundle_id.slice(0, 8)}</span> · Scores <span class="mono" title={report.artifact_sha256}>{shortHash(report.artifact_sha256)}</span> · Heldout split <span class="mono" title={report.heldout_sha256}>{shortHash(report.heldout_sha256)}</span></p>
    {/if}
  </section>
{:else}
  <p class="xs muted">No token loss scores. Run checks with private GGUF files and a runner that records token scores.</p>
{/if}

<style>
  section { margin-block: var(--s-3); min-width: 0; }
  .sentence { display: grid; gap: var(--s-2); background: var(--surface-2); border-radius: var(--r-md); padding: var(--s-3); }
  p { margin: 0; }
  select { max-width: 100%; min-width: 0; }
  .pieces { display: flex; flex-wrap: wrap; gap: var(--s-1); }
  .piece { background: var(--surface); border-radius: var(--r-sm); padding: var(--s-1) var(--s-2); white-space: pre-wrap; overflow-wrap: anywhere; min-width: 0; }
  .piece.gain { background: color-mix(in srgb, var(--ok) var(--heat), var(--surface)); }
  .piece.loss { background: color-mix(in srgb, var(--bad) var(--heat), var(--surface)); }
  .piece small { display: block; font-size: var(--fs-xs); }
  caption { text-align: left; font-weight: 600; padding-bottom: var(--s-1); }
  th { overflow-wrap: anywhere; }
</style>

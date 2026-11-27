<script lang="ts">
  // Built-in workbench agent (W15). The operator starts a session with a goal
  // and a target model, follows its requests and tool calls, and can cancel it.
  // The agent makes scratch revisions only. Approval stays in Edits.
  import { api, errorText } from '../lib/api';
  import { brewery } from '../lib/store.svelte';
  import { hashParam, route } from '../lib/route.svelte';
  import { Submission } from '../lib/submit.svelte';
  import { captureResultFocus } from '../lib/focus';
  import { modelLabel } from '../lib/format';
  import { visibleBlockers, type Blocker } from '../lib/blockers';
  import {
    PROVIDER_KINDS, SESSION_STATUS, entryText, goalProblem, parseProviderSettings, parseSession, parseSessions,
    type ProviderSettings, type SessionDetail, type SessionSummary,
  } from '../lib/agent';
  import Badge from '../components/Badge.svelte';
  import Blockers from '../components/Blockers.svelte';
  import ModelSelect from '../components/ModelSelect.svelte';

  const LIST_MS = 3000;
  const DETAIL_MS = 1000;
  // Any Library model is a valid target. GGUF models support traces only.
  const TARGET_KINDS = ['hf', 'safetensors', 'gguf'];

  let provider = $state<ProviderSettings | null>(null);
  let providerError = $state<string | null>(null);
  $effect(() => {
    const ac = new AbortController();
    api.agentProvider(ac.signal).then((raw) => { provider = parseProviderSettings(raw); providerError = null; })
      .catch((e) => { if (!ac.signal.aborted) providerError = errorText(e); });
    return () => ac.abort();
  });
  const ready = $derived(!!provider?.provider && provider.keyConfigured);

  // ---- Session list ----
  let sessions = $state<SessionSummary[] | null>(null);
  let listError = $state<string | null>(null);
  async function loadList(signal?: AbortSignal) {
    try {
      sessions = parseSessions(await api.agentSessions(signal));
      listError = null;
    } catch (e) {
      if (!signal?.aborted) listError = errorText(e);
    }
  }
  $effect(() => {
    const ac = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      await loadList(ac.signal);
      if (!ac.signal.aborted) timer = setTimeout(tick, document.visibilityState === 'hidden' ? 15000 : LIST_MS);
    };
    void tick();
    return () => { ac.abort(); if (timer) clearTimeout(timer); };
  });

  // ---- Selected session, polled while it runs ----
  const selectedId = $derived(hashParam(route.hash, 'session'));
  let detail = $state<SessionDetail | null>(null);
  let detailError = $state<string | null>(null);
  $effect(() => {
    const id = selectedId;
    detail = null;
    detailError = null;
    confirming = false;
    cancelError = null;
    if (!id) return;
    const ac = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      try {
        const next = parseSession(await api.agentSession(id, ac.signal));
        if (ac.signal.aborted) return;
        if (!next) throw new Error('The controller sent a session in an unknown format.');
        detail = next;
        detailError = null;
        if (next.status !== 'running') return;
      } catch (e) {
        if (ac.signal.aborted) return;
        detailError = errorText(e);
      }
      timer = setTimeout(tick, document.visibilityState === 'hidden' ? 15000 : DETAIL_MS);
    };
    void tick();
    return () => { ac.abort(); if (timer) clearTimeout(timer); };
  });

  // ---- Start ----
  let goal = $state('');
  let modelId = $state('');
  let touched = $state<Record<string, boolean>>({});
  let attempted = $state(false);
  const start = new Submission();
  const problems = $derived.by(() => {
    const out: Blocker[] = [];
    const g = goalProblem(goal);
    if (g) out.push({ field: 'goal', text: g });
    if (!modelId) out.push({ field: 'model', text: 'Select a target model' });
    return out;
  });
  const shown = $derived(visibleBlockers(problems, touched, attempted));
  const goalError = $derived(shown.find((b) => b.field === 'goal')?.text);
  const modelError = $derived(shown.find((b) => b.field === 'model')?.text);
  let startStatus = $state<HTMLElement>();

  async function submit(event: MouseEvent) {
    attempted = true;
    if (problems.length) return;
    const focus = captureResultFocus(event);
    const created = await start.send('agent-session', { goal, model_id: modelId }, async (body, key) => {
      const row = parseSession(await api.startAgentSession(body, key));
      if (!row) throw new Error('The controller did not return a session.');
      return row;
    }, event);
    if (created) {
      goal = '';
      touched = {};
      attempted = false;
      void loadList();
      location.hash = `#/agent?session=${encodeURIComponent(created.id)}`;
    } else if (start.error) {
      void focus?.(() => startStatus?.querySelector<HTMLElement>('[role="alert"]'));
    }
  }

  // ---- Cancel ----
  let confirming = $state(false);
  let cancelling = $state(false);
  let cancelError = $state<string | null>(null);
  async function cancel() {
    if (!detail) return;
    cancelling = true;
    cancelError = null;
    try {
      const row = parseSession(await api.cancelAgentSession(detail.id));
      if (row) detail = row;
      confirming = false;
      void loadList();
    } catch (e) {
      cancelError = errorText(e);
    } finally { cancelling = false; }
  }

  const models = $derived(brewery.state?.models ?? []);
  const modelName = (id: string) => {
    const m = models.find((x) => x.id === id);
    return m ? modelLabel(m) : id;
  };
  const kindLabel = (k: string) => PROVIDER_KINDS.find((p) => p.id === k)?.label ?? k;
  const short = (id: string) => id.slice(0, 8);
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Agent</h1>
    <p class="muted small">
      The built-in agent uses a hosted model and all workbench tools to work on a goal. It can trace models and make scratch revisions. It cannot approve a revision. Only an operator approves a publish request in Edits. Agent results are not release-qualified.
    </p>
  </div>

  <div class="layout">
    <div class="stack">
      <section class="panel" aria-labelledby="ag-start-h">
        <header><h2 id="ag-start-h">New session</h2></header>
        <div class="body">
          <form class="stack" novalidate onsubmit={(e) => e.preventDefault()}>
            {#if providerError}
              <p class="note warn small">Provider status is unavailable: {providerError}</p>
            {:else if provider && !ready}
              <p class="note warn small">
                {provider.provider ? 'The provider has no API key.' : 'No provider is set.'}
                <a href="#/settings">Set the agent provider in Settings</a>.
              </p>
            {:else if provider?.provider}
              <p class="small muted">Provider: {kindLabel(provider.provider.kind)}, <span class="mono">{provider.provider.model}</span>. Limit: {provider.provider.max_steps} requests, {provider.provider.max_session_tokens} tokens, {provider.provider.timeout_seconds} seconds.</p>
            {/if}
            <div class="field">
              <label for="ag-goal">Goal</label>
              <textarea id="ag-goal" class="input" rows="4" bind:value={goal}
                aria-invalid={goalError ? 'true' : undefined} aria-describedby="ag-goal-hint"
                onblur={() => (touched.goal = true)}></textarea>
              <span class="hint" id="ag-goal-hint">Tell the agent what to find or change. Up to 4096 bytes. The controller sends the goal and tool results to the hosted provider. Do not include secrets.</span>
            </div>
            <ModelSelect bind:value={modelId} kinds={TARGET_KINDS} label="Target model" id="ag-model"
              hint="The agent gets this model as context. Scratch revisions need a SafeTensors model. A GGUF model supports traces." error={modelError} onblur={() => (touched.model = true)} />
            {#if attempted}<Blockers blockers={problems} />{/if}
            <div class="row">
              <button class="btn primary" type="button" disabled={start.busy} aria-busy={start.busy} onclick={submit}>{start.busy ? 'Starting…' : 'Start session'}</button>
            </div>
            <div aria-live="polite" class="xs" bind:this={startStatus}>
              {#if start.error}<p class="note bad" role="alert" tabindex="-1">{start.error}</p>{/if}
            </div>
          </form>
        </div>
      </section>

      <section class="panel" aria-labelledby="ag-list-h">
        <header><h2 id="ag-list-h">Sessions</h2>{#if sessions}<span class="xs muted">{sessions.length}</span>{/if}</header>
        <div class="body">
          {#if listError}<p class="note warn small" role="alert">Could not read agent sessions: {listError}</p>{/if}
          {#if sessions === null && !listError}<p class="small muted">Loading…</p>
          {:else if sessions && !sessions.length}<p class="empty">No agent sessions.</p>
          {:else if sessions}
            <ul class="sessions">
              {#each sessions as s (s.id)}
                {@const st = SESSION_STATUS[s.status]}
                <li>
                  <a class="node" href="#/agent?session={encodeURIComponent(s.id)}" aria-current={s.id === selectedId ? 'true' : undefined}>
                    <span class="line1"><span class="goal">{s.goal}</span><Badge tone={st.tone} label={st.label} /></span>
                    <span class="xs muted break">{modelName(s.model_id)} · {s.steps} requests · {s.scratch_ids.length} revisions</span>
                  </a>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      </section>
    </div>

    <section class="panel detail" aria-labelledby="ag-detail-h">
      {#if !selectedId}
        <header><h2 id="ag-detail-h">Session</h2></header>
        <div class="body"><p class="empty">Start a session or select one in the list.</p></div>
      {:else}
        {@const st = detail ? SESSION_STATUS[detail.status] : null}
        <header>
          <h2 id="ag-detail-h">Session <span class="mono">{short(selectedId)}</span></h2>
          {#if st}<Badge tone={st.tone} label={st.label} />{/if}
        </header>
        <div class="body stack">
          {#if detailError}<p class="note warn small" role="alert">Could not read the session: {detailError}</p>{/if}
          {#if !detail && !detailError}<p class="small muted">Loading…</p>{/if}
          {#if detail}
            <p class="goal-text">{detail.goal}</p>
            <dl class="kv small">
              <dt>Target model</dt><dd class="break">{modelName(detail.model_id)}</dd>
              <dt>Progress</dt><dd class="num" aria-live="polite">{detail.steps} of {provider?.provider?.max_steps ?? '?'} requests, {detail.tokens} tokens</dd>
              {#if detail.message}<dt>Message</dt><dd>{detail.message}</dd>{/if}
            </dl>
            {#if detail.status === 'running'}
              <div class="row">
                {#if confirming}
                  <button type="button" class="btn danger" disabled={cancelling} onclick={cancel}>{cancelling ? 'Cancelling…' : 'Confirm cancel'}</button>
                  <button type="button" class="btn ghost" disabled={cancelling} onclick={() => (confirming = false)}>Keep running</button>
                {:else}
                  <button type="button" class="btn danger" onclick={() => (confirming = true)}>Cancel session</button>
                {/if}
                {#if cancelError}<span class="xs err" role="alert">{cancelError}</span>{/if}
              </div>
            {/if}

            <div>
              <h3>Scratch revisions</h3>
              {#if detail.scratch_ids.length}
                <ul class="revs">
                  {#each detail.scratch_ids as rev (rev)}
                    <li><a href="#/edits?rev={encodeURIComponent(rev)}">Open revision <span class="mono">{short(rev)}</span> in Edits</a></li>
                  {/each}
                </ul>
                <p class="xs muted">Only an operator approves a publish request in Edits. Review the checks there first.</p>
              {:else}
                <p class="small muted">No scratch revisions yet.</p>
              {/if}
            </div>

            <div>
              <h3>Transcript</h3>
              {#if !detail.entries.length}
                <p class="small muted">{detail.status === 'running' ? 'Waiting for the first reply.' : 'No entries.'}</p>
              {:else}
                <ol class="entries">
                  {#each detail.entries as e, i (i)}
                    <li class="entry {e.kind}">
                      {#if e.kind === 'assistant'}
                        <p class="label xs">Agent</p>
                        <p class="text">{e.text}</p>
                      {:else if e.kind === 'tool_call'}
                        <p class="label xs">Tool call: <span class="mono">{e.name}</span></p>
                        <pre class="mono xs">{entryText(e.arguments)}</pre>
                      {:else}
                        <p class="label xs">Tool result: <span class="mono">{e.name}</span></p>
                        <pre class="mono xs">{entryText(e.result)}</pre>
                      {/if}
                    </li>
                  {/each}
                </ol>
              {/if}
              {#if detail.status === 'running'}<p class="xs muted">Updates every second while the session runs.</p>{/if}
            </div>
          {/if}
        </div>
      {/if}
    </section>
  </div>
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(280px, 1fr) minmax(0, 2fr);
    gap: var(--s-4);
    align-items: start;
  }
  @media (max-width: 960px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .panel > header {
    gap: var(--s-2);
  }
  textarea {
    resize: vertical;
    min-height: 88px;
  }
  .sessions,
  .revs,
  .entries {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .node {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 6px 8px;
    margin: 2px 0;
    border-radius: var(--r-md);
    border: 1px solid transparent;
    color: var(--text);
    text-decoration: none;
  }
  .node:hover {
    background: var(--accent-soft);
  }
  .node[aria-current='true'] {
    border-color: var(--accent);
    box-shadow: inset 3px 0 0 var(--accent);
  }
  .line1 {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--s-2);
  }
  .goal {
    min-width: 0;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow-wrap: anywhere;
  }
  .goal-text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .break {
    overflow-wrap: anywhere;
  }
  .kv {
    display: grid;
    grid-template-columns: minmax(96px, max-content) minmax(0, 1fr);
    gap: 4px var(--s-3);
    margin: 0;
  }
  .kv dt {
    color: var(--muted);
  }
  .kv dd {
    margin: 0;
  }
  h3 {
    font-size: var(--fs-md);
    font-weight: 600;
    margin-bottom: var(--s-1);
  }
  .revs li {
    padding: 2px 0;
  }
  .entries {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .entry {
    border-left: 3px solid var(--line-strong);
    padding: var(--s-1) 0 var(--s-1) var(--s-3);
  }
  .entry.assistant {
    border-left-color: var(--accent);
  }
  .entry.tool_result {
    border-left-color: var(--line);
  }
  .label {
    color: var(--text-2);
    font-weight: 600;
    margin-bottom: 2px;
  }
  .text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  pre {
    margin: 0;
    padding: var(--s-2);
    background: var(--surface-2);
    border-radius: var(--r-sm);
    max-height: 240px;
    overflow: auto;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .err {
    color: var(--bad);
  }
</style>

<script lang="ts">
  // Edit sessions: scratch revisions that agents or operators make, as a
  // lineage tree with provenance, checks, diffs and the operator approval.
  // The list polls GET /api/scratch so an agent's edit loop shows live.
  import { api, errorText } from '../lib/api';
  import { brewery } from '../lib/store.svelte';
  import { hashParam } from '../lib/route.svelte';
  import { formatBytes, formatTime, modelLabel, newestFirst, shortHash } from '../lib/format';
  import { obj } from '../lib/reports';
  import {
    SCRATCH_STATE,
    checkRows,
    flatten,
    fmt,
    lineage,
    operationText,
    traceDiffQuery,
    type LineageNode,
    type ScratchRevision,
  } from '../lib/weightspace';
  import Badge from '../components/Badge.svelte';
  import JobLink from '../components/JobLink.svelte';
  import ApproveScratch from '../components/trace/ApproveScratch.svelte';
  import TokenLossDiff from '../components/trace/TokenLossDiff.svelte';
  import WeightHealth from '../components/spectra/WeightHealth.svelte';

  const POLL_MS = 3000;
  const models = $derived(brewery.state?.models ?? []);
  const jobs = $derived(brewery.state?.jobs ?? []);
  const traces = $derived(newestFirst((brewery.state?.jobs ?? []).filter((j) => j.kind === 'trace' && j.state === 'succeeded')));

  let list = $state<ScratchRevision[] | null>(null);
  let listError = $state<string | null>(null);
  let updated = $state<Date | null>(null);
  let seen = new Set<string>();
  let fresh = $state<Set<string>>(new Set());

  async function load(signal?: AbortSignal) {
    try {
      const data = await api.scratch(signal);
      const next = Array.isArray(data) ? (data as ScratchRevision[]).filter((r) => r && typeof r.id === 'string' && r.provenance) : [];
      // Mark revisions that appeared after the first load, so new agent steps stand out.
      if (list !== null) fresh = new Set([...fresh, ...next.filter((r) => !seen.has(r.id)).map((r) => r.id)]);
      seen = new Set(next.map((r) => r.id));
      list = next;
      listError = null;
      updated = new Date();
    } catch (e) {
      if (!signal?.aborted) listError = errorText(e);
    }
  }
  $effect(() => {
    const ac = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      await load(ac.signal);
      if (!ac.signal.aborted) timer = setTimeout(tick, document.visibilityState === 'hidden' ? 15000 : POLL_MS);
    };
    void tick();
    return () => {
      ac.abort();
      if (timer) clearTimeout(timer);
    };
  });

  const forest = $derived(lineage(list ?? []));
  const all = $derived(forest.flatMap((t) => flatten(t.nodes)));
  const pending = $derived(all.filter((n) => n.rev.state === 'publish_requested').length);

  let hash = $state(typeof window === 'undefined' ? '' : window.location.hash);
  $effect(() => {
    const on = () => (hash = window.location.hash);
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  });
  const selectedId = $derived(hashParam(hash, 'rev'));
  const selected = $derived(all.find((n) => n.rev.id === selectedId)?.rev ?? null);
  const byId = $derived(new Map(all.map((n) => [n.rev.id, n.rev])));
  const sourceName = (id: string) => {
    const m = models.find((x) => x.id === id);
    return m ? modelLabel(m) : id;
  };
  const parentLabel = (r: ScratchRevision) => (byId.has(r.provenance.parent_id) ? `scratch ${r.provenance.parent_id.slice(0, 8)}` : sourceName(r.provenance.parent_id));

  // ---- Diffs of the selected revision against its parent (GET only) ----
  let tensorDiff = $state<Record<string, unknown> | null>(null);
  let behaviorDiff = $state<Record<string, unknown> | null>(null);
  let diffBusy = $state<'' | 'tensors' | 'behavior'>('');
  let diffError = $state<string | null>(null);
  $effect(() => {
    void selectedId;
    tensorDiff = null;
    behaviorDiff = null;
    diffError = null;
  });
  async function runDiff(kind: 'tensors' | 'behavior') {
    if (!selected) return;
    diffBusy = kind;
    diffError = null;
    try {
      const q = new URLSearchParams({ a: selected.provenance.parent_id, b: selected.id }).toString();
      const r = obj(await api.diff(kind, q));
      if (kind === 'tensors') tensorDiff = r;
      else behaviorDiff = r;
    } catch (e) {
      diffError = errorText(e);
    } finally {
      diffBusy = '';
    }
  }
  const tensorRows = $derived((tensorDiff?.tensors as Record<string, unknown>[] | undefined) ?? []);
  const metricRows = $derived((behaviorDiff?.metrics as Record<string, unknown>[] | undefined) ?? []);
  const FLAG: Record<string, { label: string; tone: 'ok' | 'bad' | 'warn' | 'info' | 'muted' }> = {
    unchanged: { label: 'Unchanged', tone: 'muted' },
    missing: { label: 'Missing', tone: 'warn' },
    within_threshold: { label: 'Within threshold', tone: 'muted' },
    improved: { label: 'Improved', tone: 'ok' },
    regressed: { label: 'Regressed', tone: 'bad' },
    changed: { label: 'Changed', tone: 'info' },
  };

  // ---- Trace diff ----
  let ta = $state('');
  let tb = $state('');
  let tPrompt = $state(0);
  let tLayer = $state('');
  let traceDiff = $state<Record<string, unknown> | null>(null);
  let traceBusy = $state(false);
  let traceError = $state<string | null>(null);
  const traceBlock = $derived(!ta || !tb ? 'Select two traces.' : ta === tb ? 'Select two different traces.' : tLayer.trim() && !/^\d{1,4}$/.test(tLayer.trim()) ? 'The layer must be a whole number.' : '');
  async function runTraceDiff() {
    traceBusy = true;
    traceError = null;
    traceDiff = null;
    try {
      traceDiff = obj(await api.diff('traces', traceDiffQuery(ta, tb, tPrompt, tLayer)));
    } catch (e) {
      traceError = errorText(e);
    } finally {
      traceBusy = false;
    }
  }
  const activations = $derived((traceDiff?.activations as Record<string, unknown>[] | undefined) ?? []);
  const logitShifts = $derived([...((traceDiff?.logit_lens as Record<string, unknown>[] | undefined) ?? []), ...((traceDiff?.model_logits as Record<string, unknown>[] | undefined) ?? [])]);

  function approved(r: ScratchRevision) {
    list = (list ?? []).map((x) => (x.id === r.id ? r : x));
    approvedNote = r.library_model_id ? `Approved. Library entry ${r.library_model_id}. It is not release-qualified and not deployed.` : 'Approved.';
    void brewery.refresh();
  }
  let approvedNote = $state<string | null>(null);
  $effect(() => {
    void selectedId;
    approvedNote = null;
  });
  const traceLabel = (id: string) => {
    const j = traces.find((x) => x.id === id);
    return j ? `${id.slice(0, 8)} · ${sourceName(j.input_ids[0])} · ${formatTime(j.created_at)}` : id;
  };
  const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;
  const methodLabel = (method: string) => ({
    copy: 'Copy', direct: 'Direct edit', linear: 'Linear merge', task_arithmetic: 'Task arithmetic',
    ties: 'TIES', dare_linear: 'DARE-linear', dare_ties: 'DARE-TIES',
  } as Record<string, string>)[method] ?? method;
  const vec = (v: unknown) => (Array.isArray(v) ? `[${v.join(', ')}]` : '—');
</script>

{#snippet node(n: LineageNode)}
  {@const st = SCRATCH_STATE[n.rev.state] ?? { label: n.rev.state, tone: 'muted' as const }}
  {@const checks = checkRows(n.rev.check_bundle)}
  <li>
    <a class="node" href="#/edits?rev={encodeURIComponent(n.rev.id)}" aria-current={n.rev.id === selectedId ? 'true' : undefined}>
      <span class="line1">
        <span class="mono">{n.rev.id.slice(0, 8)}</span>
        <Badge tone={st.tone} label={st.label} />
        {#if fresh.has(n.rev.id)}<span class="new xs">New</span>{/if}
      </span>
      <span class="line2 xs muted">
        {#if n.rev.provenance.compiled_program}
          {n.rev.provenance.compiled_program.program}
        {:else if n.rev.provenance.method === 'copy'}
          Copy of the parent
        {:else if n.rev.provenance.method === 'direct'}
          {plural(n.rev.provenance.operations.length, 'edit')}, {plural(n.rev.delta.changed_tensors, 'tensor')}
        {:else}
          {methodLabel(n.rev.provenance.method)}, {plural(n.rev.delta.changed_tensors, 'tensor')}
        {/if}
        · {checks.length ? checks.map((c) => `${c.name} ${c.label.toLowerCase()}`).join(', ') : 'No checks'}
      </span>
    </a>
    {#if n.children.length}
      <ul>{#each n.children as c (c.rev.id)}{@render node(c)}{/each}</ul>
    {/if}
  </li>
{/snippet}

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Edit sessions</h1>
    <p class="muted small">
      Scratch revisions are SafeTensors models. An edit makes a child with a reversible delta. A compiled program has no parent. Only an operator can approve a publish request here. A scratch revision is not release-qualified.
    </p>
  </div>

  <p class="xs muted live" aria-live="polite">
    {#if listError}<span class="err">Could not read scratch revisions: {listError}</span>
    {:else if updated}Updates every {POLL_MS / 1000} seconds. Last update {updated.toLocaleTimeString()}.{#if pending}{` ${pending} publish ${pending === 1 ? 'request waits' : 'requests wait'} for approval.`}{/if}
    {:else}Loading…{/if}
  </p>

  <div class="layout">
    <section class="panel tree-panel" aria-labelledby="ed-tree-h">
      <header><h2 id="ed-tree-h">Lineage</h2><span class="xs muted">{all.length} revisions</span></header>
      <div class="body">
        {#if list && !forest.length}
          <p class="empty">No scratch revisions. An agent or the API creates them from a SafeTensors model.</p>
        {/if}
        {#each forest as tree (tree.source)}
          {@const source = tree.nodes.every((n) => n.rev.provenance.compiled_program) ? 'Compiled programs' : sourceName(tree.source)}
          <nav class="tree" aria-label="Revisions of {source}">
            <p class="src small"><span class="muted xs">Source</span> <span class="break">{source}</span></p>
            <ul>{#each tree.nodes as n (n.rev.id)}{@render node(n)}{/each}</ul>
          </nav>
        {/each}
      </div>
    </section>

    <section class="panel detail" aria-labelledby="ed-rev-h">
      {#if !selected}
        <header><h2 id="ed-rev-h">Revision</h2></header>
        <p class="empty">{selectedId && list ? 'This revision is not in the list. It can be discarded.' : 'Select a revision in the lineage.'}</p>
      {:else}
        {@const st = SCRATCH_STATE[selected.state] ?? { label: selected.state, tone: 'muted' as const }}
        {@const p = selected.provenance}
        {@const program = p.compiled_program}
        {@const conversion = newestFirst(jobs.filter((j) => j.kind === 'scratch_gguf' && j.input_ids[0] === selectedId))[0]}
        {@const workerChecks = newestFirst(jobs.filter((j) => j.kind === 'scratch_checks' && j.input_ids[3] === selectedId))[0]}
        <header>
          <h2 id="ed-rev-h">Revision <span class="mono">{selected.id.slice(0, 8)}</span></h2>
          <div class="row"><Badge tone={st.tone} label={st.label} /><Badge tone="warn" label="Not release-qualified" /></div>
        </header>
        <div class="body stack">
          {#if approvedNote}<p class="note info small" role="status">{approvedNote}</p>{/if}
          <dl class="kv small">
            <dt>ID</dt><dd class="mono break">{selected.id}</dd>
            <dt>SHA-256</dt><dd class="mono break">{selected.sha256}</dd>
            {#if program}
              <dt>Parent</dt><dd>None</dd>
              <dt>Method</dt><dd>Compile program</dd>
              <dt>Program</dt><dd class="mono">{program.program}</dd>
              <dt>Config SHA-256</dt><dd class="mono break">{program.config_sha256}</dd>
            {:else}
              <dt>Parent</dt><dd class="break">{#if byId.has(p.parent_id)}<a href="#/edits?rev={encodeURIComponent(p.parent_id)}">{parentLabel(selected)}</a>{:else}{parentLabel(selected)}{/if} <span class="mono xs muted" title={p.parent_sha256}>{shortHash(p.parent_sha256)}</span></dd>
              <dt>Method</dt><dd>{methodLabel(p.method)}</dd>
            {/if}
            <dt>Worker</dt><dd class="mono">{p.worker}</dd>
            <dt>Tool</dt><dd class="mono break">{p.tool} {p.tool_version}</dd>
            <dt>Runtime</dt><dd class="mono break">{p.runtime_version}</dd>
            <dt>Created</dt><dd class="num">{formatTime(p.created_at_ms)}</dd>
            {#if selected.library_model_id}<dt>Library entry</dt><dd class="mono break">{selected.library_model_id}</dd>{/if}
          </dl>

          {#if p.operations.length}
            <div>
              <h3>Edits</h3>
              <ol class="ops small">
                {#each p.operations as op, i (i)}<li><span class="mono">{op.tensor}</span>: {operationText(op)}</li>{/each}
              </ol>
            </div>
          {/if}

          <div>
            <h3>Delta</h3>
            {#if program}
              <p class="small muted">No delta. This program has no parent.</p>
            {:else}
              <p class="small num break">{plural(selected.delta.changed_tensors, 'tensor')}, {formatBytes(selected.delta.changed_bytes)} changed, delta file {formatBytes(selected.delta.delta_bytes)} <span class="mono xs muted" title={selected.delta.delta_sha256}>{shortHash(selected.delta.delta_sha256)}</span></p>
            {/if}
          </div>

          <div>
            <h3>Checks</h3>
            {#if program}
              <p class="small muted">GGUF does not support this program.</p>
            {:else if selected.gguf}
              <p class="small">Private GGUF: Ready <span class="mono xs muted" title={selected.gguf.sha256}>{shortHash(selected.gguf.sha256)}</span> · <JobLink id={selected.gguf.job_id} label="Conversion job" /></p>
            {:else if conversion}
              <p class="small">Private GGUF: <JobLink id={conversion.id} label={conversion.state} /></p>
            {:else}
              <p class="small muted">Private GGUF: Not converted</p>
            {/if}
            {#if workerChecks && workerChecks.id !== selected.check_bundle?.measured_job_id}
              <p class="small">Worker check job: <JobLink id={workerChecks.id} label={workerChecks.state} /></p>
            {/if}
            {#if !selected.check_bundle}
              <p class="small muted">No check bundle. An agent runs checks before a publish request.</p>
            {:else}
              {@const b = selected.check_bundle}
              <ul class="checklist">
                {#each checkRows(b) as c (c.name)}
                  <li><Badge tone={c.tone} label="{c.name}: {c.label}" />{#if c.reason}<span class="xs muted">{c.reason}</span>{/if}</li>
                {/each}
              </ul>
              {#if b.program_cases !== undefined}
                <p class="small">Exact program outputs: {b.program_cases}</p>
              {/if}
              <p class="xs muted">
                Bundle <span class="mono">{b.id.slice(0, 8)}</span>, {formatTime(b.created_at_ms)}.{#if b.revision_sha256 !== selected.sha256}{' '}<strong class="err">The bundle is for another hash.</strong>{/if}{#if b.acknowledged_at_ms}{` Acknowledged ${formatTime(b.acknowledged_at_ms)}.`}{/if}
              </p>
              {#if b.measured_job_id}
                <p class="small"><JobLink id={b.measured_job_id} label="Worker measurements" /></p>
              {/if}
              {#if b.job_refs.length}
                <p class="xs">Referenced jobs (not verified against this revision): {#each b.job_refs as j, i (j.job_id)}{i ? ', ' : ''}<JobLink id={j.job_id} label="{j.kind} ({j.role})" />{/each}</p>
              {/if}
            {/if}
          </div>

          {#if selected.state === 'publish_requested'}
            <div class="approval">
              <h3>Publish request</h3>
              <p class="small">An agent asks to publish this exact hash. Approval is an operator action. Agents cannot approve through MCP.</p>
              <ApproveScratch rev={selected} onapproved={approved} />
            </div>
          {/if}

          {#if !program}
          <div class="diffs">
            <h3>Compare with the parent</h3>
            <div class="row">
              <button type="button" class="btn small" disabled={diffBusy !== ''} aria-busy={diffBusy === 'tensors'} onclick={() => runDiff('tensors')}>{diffBusy === 'tensors' ? 'Reading…' : 'Tensor diff'}</button>
              <button type="button" class="btn small" disabled={diffBusy !== '' || p.method === 'copy'} aria-busy={diffBusy === 'behavior'} onclick={() => runDiff('behavior')}>{diffBusy === 'behavior' ? 'Reading…' : 'Behavior diff'}</button>
            </div>
            {#key selected.check_bundle?.id}
              <TokenLossDiff parent={p.parent_id} child={selected.id} available={!!selected.check_bundle?.token_loss} />
            {/key}
            {#if diffError}<p class="note bad small" role="alert">{diffError}</p>{/if}
            {#if tensorDiff}
              {#if !tensorRows.length}
                <p class="empty">No tensor changed.</p>
              {:else}
                <div class="table-wrap">
                  <table class="data num stackable">
                    <caption class="sr-only">Changed tensors against the parent</caption>
                    <thead><tr><th scope="col">Tensor</th><th scope="col">Changed</th><th scope="col">Region</th><th scope="col">L2</th><th scope="col">L∞</th><th scope="col">Relative</th><th scope="col">Sample rank</th></tr></thead>
                    <tbody>
                      {#each tensorRows as t (String(t.name))}
                        {@const bounds = obj(t.bounds)}
                        {@const lr = obj(t.low_rank)}
                        <tr>
                          <th scope="row" class="mono break">{String(t.name)} <span class="xs muted">{String(t.dtype)} {vec(t.shape)}</span></th>
                          <td data-label="Changed">{String(t.changed_elements)}</td>
                          <td data-label="Region" class="mono">{vec(bounds?.start)}–{vec(bounds?.end)}</td>
                          <td data-label="L2" title={String(t.l2)}>{fmt(t.l2)}</td>
                          <td data-label="L∞" title={String(t.linf)}>{fmt(t.linf)}</td>
                          <td data-label="Relative" title={String(t.relative_frobenius)}>{fmt(t.relative_frobenius)}</td>
                          <td data-label="Sample rank">{lr ? `${String(lr.sample_rank)} (${vec(lr.sample_shape)})` : '—'}</td>
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              {/if}
            {/if}
            {#if behaviorDiff}
              {#if !metricRows.length}
                <p class="empty">No behavior metrics.</p>
              {:else}
                <div class="table-wrap">
                  <table class="data num stackable">
                    <caption class="sr-only">Behavior metrics, parent and child</caption>
                    <thead><tr><th scope="col">Metric</th><th scope="col">Parent</th><th scope="col">Child</th><th scope="col">Change</th><th scope="col">Result</th></tr></thead>
                    <tbody>
                      {#each metricRows as m (String(m.name))}
                        {@const f = FLAG[String(m.flag)] ?? { label: String(m.flag), tone: 'muted' as const }}
                        <tr>
                          <th scope="row" class="mono break">{String(m.name)}</th>
                          <td data-label="Parent">{fmt(m.parent)}</td>
                          <td data-label="Child">{fmt(m.child)}</td>
                          <td data-label="Change">{fmt(m.delta)}</td>
                          <td data-label="Result"><Badge tone={f.tone} label={f.label} /></td>
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              {/if}
            {/if}
          </div>
          {/if}
        </div>
      {/if}
    </section>
  </div>

  {#if selected}<WeightHealth rev={selected} revisions={all.map((n) => n.rev)} />{/if}

  <section class="panel" aria-labelledby="ed-td-h">
    <header><h2 id="ed-td-h">Trace diff</h2></header>
    <div class="body stack">
      <p class="small muted">Compare two finished traces with the same prompts, capture and tokens, for example the parent and the edited model. A is the baseline.</p>
      <div class="grid-2">
        <div class="field">
          <label for="ed-ta">Trace A</label>
          <select id="ed-ta" class="input" bind:value={ta}>
            <option value="">Select a trace</option>
            {#each traces as j (j.id)}<option value={j.id}>{traceLabel(j.id)}</option>{/each}
          </select>
        </div>
        <div class="field">
          <label for="ed-tb">Trace B</label>
          <select id="ed-tb" class="input" bind:value={tb}>
            <option value="">Select a trace</option>
            {#each traces as j (j.id)}<option value={j.id}>{traceLabel(j.id)}</option>{/each}
          </select>
        </div>
      </div>
      <div class="grid-3">
        <div class="field">
          <label for="ed-tp">Prompt index</label>
          <input id="ed-tp" class="input num" type="number" min="0" max="3" step="1" bind:value={tPrompt} inputmode="numeric" />
        </div>
        <div class="field">
          <label for="ed-tl">Layer <span class="muted">(optional)</span></label>
          <input id="ed-tl" class="input num" bind:value={tLayer} inputmode="numeric" autocomplete="off" aria-describedby="ed-tl-h" />
          <span class="hint" id="ed-tl-h">Use a layer if the full diff is too large.</span>
        </div>
      </div>
      <div class="row">
        <button type="button" class="btn primary" disabled={!!traceBlock || traceBusy} aria-busy={traceBusy} aria-describedby={traceBlock ? 'ed-td-why' : undefined} onclick={runTraceDiff}>{traceBusy ? 'Comparing…' : 'Compare traces'}</button>
        {#if traceBlock}<span class="xs muted" id="ed-td-why">{traceBlock}</span>{/if}
      </div>
      {#if traceError}<p class="note bad small" role="alert">{traceError}</p>{/if}
      {#if traceDiff}
        <p class="xs muted"><Badge tone="warn" label="Not release-qualified" /> Prompt {String(traceDiff.prompt)}. {activations.length} activation rows, {logitShifts.length} logit rows.</p>
        {#if activations.length}
          <div class="table-wrap tall">
            <table class="data num stackable">
              <caption class="sr-only">Activation shifts from A to B</caption>
              <thead><tr><th scope="col">Layer</th><th scope="col">Role</th><th scope="col">Position</th><th scope="col">L2</th><th scope="col">L∞</th><th scope="col">Cosine</th></tr></thead>
              <tbody>
                {#each activations as a, i (i)}
                  <tr>
                    <th scope="row">{String(a.layer)}</th>
                    <td data-label="Role" class="mono">{String(a.role)}</td>
                    <td data-label="Position">{String(a.position)}</td>
                    <td data-label="L2">{a.missing_a || a.missing_b ? `Missing in ${a.missing_a ? 'A' : 'B'}` : fmt(a.l2)}</td>
                    <td data-label="L∞">{fmt(a.linf)}</td>
                    <td data-label="Cosine">{fmt(a.cosine)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
        {#if logitShifts.length}
          <div class="table-wrap tall">
            <table class="data num stackable">
              <caption class="sr-only">Top-k logit shifts from A to B</caption>
              <thead><tr><th scope="col">Layer</th><th scope="col">Position</th><th scope="col">Entered top-k</th><th scope="col">Left top-k</th><th scope="col">Shared</th><th scope="col">KL on shared</th></tr></thead>
              <tbody>
                {#each logitShifts as l, i (i)}
                  <tr>
                    <th scope="row">{l.layer === null ? 'Model' : String(l.layer)}</th>
                    <td data-label="Position">{String(l.position)}</td>
                    <td data-label="Entered top-k" class="mono">{vec(l.entered)}</td>
                    <td data-label="Left top-k" class="mono">{vec(l.exited)}</td>
                    <td data-label="Shared">{String(l.shared_count)}</td>
                    <td data-label="KL on shared">{fmt(l.kl_a_to_b_shared)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          <p class="xs muted">KL is on the shared top-k token IDs only, not on the full vocabulary.</p>
        {/if}
      {/if}
    </div>
  </section>
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .live {
    margin-top: calc(-1 * var(--s-2));
  }
  .err {
    color: var(--bad);
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(260px, 1fr) minmax(0, 2fr);
    gap: var(--s-4);
    align-items: start;
  }
  @media (max-width: 960px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .tree-panel .body {
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
    max-height: 640px;
    overflow: auto;
  }
  .src {
    margin-bottom: var(--s-1);
  }
  .tree ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .tree ul ul {
    margin-left: 10px;
    padding-left: 10px;
    border-left: 1px solid var(--line);
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
  /* Border and bar only: status badges keep their contrast on the plain surface. */
  .node[aria-current='true'] {
    border-color: var(--accent);
    box-shadow: inset 3px 0 0 var(--accent);
  }
  .line1 {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }
  .new {
    color: var(--info);
    font-weight: 600;
  }
  h3 {
    font-size: var(--fs-md);
    font-weight: 600;
    margin-bottom: var(--s-1);
  }
  .kv {
    display: grid;
    grid-template-columns: minmax(96px, max-content) minmax(0, 1fr);
    gap: 4px var(--s-3);
    margin: 0;
  }
  .kv dt {
    color: var(--text-2);
  }
  .kv dd {
    margin: 0;
    min-width: 0;
  }
  .ops {
    margin: 0;
    padding-left: 1.4em;
    overflow-wrap: anywhere;
  }
  .checklist {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin: 0 0 var(--s-1);
    padding: 0;
    list-style: none;
  }
  .checklist li {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px var(--s-2);
  }
  .approval {
    padding: var(--s-3);
    border: 1px solid var(--warn-line);
    border-radius: var(--r-tile);
    background: linear-gradient(var(--warn-bg), var(--warn-bg)), var(--surface);
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .diffs {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .tall {
    max-height: 420px;
  }
</style>

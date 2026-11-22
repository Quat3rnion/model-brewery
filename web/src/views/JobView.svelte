<script lang="ts">
  import { untrack } from 'svelte';
  import JobProgress from '../components/JobProgress.svelte';
  import { detailLines, readDetail, summarizeHistory } from '../lib/progress';
  import { brewery } from '../lib/store.svelte';
  import { api, ApiError, errorText } from '../lib/api';
  import { canRetryManagedHgn, managedRetryHref, STAGES } from '../lib/conversion';
  import { conversions } from '../lib/conversions.svelte';
  import { formatTime, hostName, isActive, isCancelling, isTerminal, kindLabel, kindTitle, modelLabel, progressText, stateLabel } from '../lib/format';
  import JobState from '../components/JobState.svelte';
  import CancelButton from '../components/CancelButton.svelte';
  import KeyValues from '../components/KeyValues.svelte';
  import JobResult from '../components/results/JobResult.svelte';
  import { obj } from '../lib/reports';
  import { rerunAction, rerunHint } from '../lib/rerun';
  import type { JobDetail, JobEvent } from '../lib/types';

  let { id }: { id: string } = $props();

  const summary = $derived(brewery.state?.jobs.find((j) => j.id === id));
  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);
  const prompts = $derived(brewery.state?.prompts ?? []);

  let record = $state<{ id: string; detail?: JobDetail; error?: string; notFound?: boolean }>();
  const detail = $derived(record?.id === id && record.detail?.id === id ? record.detail : null);
  const error = $derived(record?.id === id ? record.error : undefined);
  const notFound = $derived(record?.id === id && record.notFound);

  // Reload the full record when the job changes state; poll while it is active.
  $effect(() => {
    const state = summary?.state;
    const jobId = id;
    const previous = untrack(() => record);
    if (previous?.id !== jobId) record = { id: jobId };
    else if (previous.notFound) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const load = async () => {
      try {
        // untrack: loadJob reads and writes brewery.details; tracking it here re-ran this effect in a loop.
        const d = await untrack(() => brewery.loadJob(jobId, !isTerminal(state)));
        if (stop) return;
        if (d.id !== jobId) throw new Error('The controller returned a different job.');
        record = { id: jobId, detail: d };
        if (isTerminal(d.state)) return;
      } catch (e) {
        if (stop) return;
        if (e instanceof ApiError && e.status === 404) {
          record = { id: jobId, notFound: true, error: 'This job does not exist or was removed.' };
          return;
        }
        record = { id: jobId, detail: record?.detail, error: errorText(e) };
      }
      if (!stop && !isTerminal(state)) timer = setTimeout(load, 3000);
    };
    void load();
    return () => {
      stop = true;
      if (timer) clearTimeout(timer);
    };
  });

  const history = $derived(summarizeHistory(detail?.events ?? []));
  const job = $derived(detail && summary ? { ...detail, ...summary, result: detail.result, events: detail.events } : (detail ?? summary));
  const retry = $derived(job ? rerunAction(job) : null);
  const findManagedRetry = $derived(job && STAGES.some(stage => stage === job.kind) &&
    job.format !== 'gguf' && !job.options?.hy4_overrides && !job.options?.layer_recipe ? job.id : null);
  const parent = $derived(findManagedRetry ? conversions.list.find(run => Object.values(run.jobIds).includes(findManagedRetry)) : undefined);
  const managedRetry = $derived(parent && canRetryManagedHgn(parent) ? parent.id : null);
  $effect(() => { if (findManagedRetry) return conversions.watch(); });
  // The scheduler may advance the parent after the last child event. Watch
  // the state cursor as well as polling active conversions until they settle.
  let lastProbe: { jobId: string; cursor: number | undefined; state: string | undefined } | undefined;
  $effect(() => {
    const jobId = findManagedRetry;
    const cursor = brewery.state?.event_cursor;
    const state = job?.state;
    if (!jobId) return;
    const changed = lastProbe?.jobId === jobId && (lastProbe.cursor !== cursor || lastProbe.state !== state);
    lastProbe = { jobId, cursor, state };
    if (changed && untrack(() => !parent || parent.state === 'running')) void conversions.refresh();
  });
  const hint = $derived(retry ? rerunHint(job?.error) : null);
  const prompt = $derived(job?.prompt_id ? prompts.find((p) => p.id === job.prompt_id) : null);
  const downloadId = $derived(job?.kind === 'hf_download' && job.state === 'succeeded' ? obj(job.result)?.library_model_id : null);
  const download = $derived(typeof downloadId === 'string' ? models.find((m) => m.id === downloadId) : null);

  function eventProgress(event: JobEvent) {
    const pd = readDetail(event.progress_detail);
    const labels = [event.state, stateLabel(event.state)].map((label) => label.toLowerCase());
    const text = pd ? detailLines(pd, event.at).main : progressText(event.progress);
    return text.split(' · ').filter((part) => part && !labels.includes(part.trim().toLowerCase())).join(' · ');
  }
</script>

<div class="stack">
  <p class="small"><a href="#/queue">← All jobs</a></p>
  {#if notFound || !job}
    <h1 tabindex="-1">Job <span class="mono">{id}</span></h1>
    {#if error}<p class="note bad" role="alert">{error}</p>{:else}<p class="muted">Loading…</p>{/if}
  {:else}
    <div class="head">
      <div class="row">
        <h1 tabindex="-1">{kindTitle(job.kind)}</h1>
        <JobState {job} />
      </div>
      <p class="xs muted mono break">{job.id}</p>
    </div>

    {#if isActive(job.state)}
      <section class="panel">
        <div class="body stack">
          {#if job.state === 'queued' && !isCancelling(job)}
            <p class="small">Waiting in the queue. It starts when the jobs before it end.</p>
          {:else}
            <JobProgress {job} label="Job progress" />
          {/if}
          <div><CancelButton {job} small={false} /></div>
          <p class="xs muted">Results appear here after the job finishes.</p>
        </div>
      </section>
    {/if}

    {#if job.error}<p class="note bad" role="alert">{job.error}</p>{/if}
    {#if job.state === 'interrupted' || job.state === 'missing'}
      <p class="note warn">The controller lost this job (restart or worker loss). Partial outputs are excluded. Submit the job again to retry.</p>
    {/if}
    {#if job.state === 'cancelled'}
      <p class="note warn">This job was cancelled. Partial outputs are excluded.</p>
    {/if}
    {#if retry || managedRetry}
      <div class="row">
        {#if retry}<a class="btn" href={retry.href}>{retry.label}</a>{/if}
        {#if managedRetry}<a class="btn" href={managedRetryHref(managedRetry)}>Run again</a>{/if}
        {#if hint}<p class="small muted">{hint}</p>{/if}
      </div>
    {/if}

    <div class="layout">
      <section class="panel" aria-labelledby="res-h">
        <header><h2 id="res-h">Result</h2></header>
        <div class="body">
          {#if download}
            <p class="note info">Verified download: <a href="#/library?entry={encodeURIComponent(download.id)}">View {modelLabel(download)} in Library</a></p>
          {/if}
          {#if job.state === 'succeeded' || job.has_result || (detail && detail.result !== undefined && detail.result !== null)}
            {#if detail}
              <JobResult job={detail} />
            {:else if error}
              <p class="note bad" role="alert">{error}</p>
            {:else}
              <p class="muted small">Loading result…</p>
            {/if}
          {:else}
            <p class="muted small">No result{isTerminal(job.state) ? '.' : ' yet.'}</p>
          {/if}
        </div>
      </section>

      <aside class="stack">
        <section class="panel" aria-labelledby="req-h">
          <header><h2 id="req-h">Request</h2></header>
          <div class="body stack">
            <dl class="kv small">
              <dt>Host</dt><dd>{hostName(hosts, job.host)}</dd>
              {#if job.created_at}<dt>Created</dt><dd>{formatTime(job.created_at)}</dd>{/if}
              {#if job.started_at}<dt>Started</dt><dd>{formatTime(job.started_at)}</dd>{/if}
              {#if job.finished_at}<dt>Finished</dt><dd>{formatTime(job.finished_at)}</dd>{/if}
              {#if job.updated_at}<dt>Updated</dt><dd>{formatTime(job.updated_at)}</dd>{/if}
              {#if job.requested_host === 'auto'}<dt>Requested</dt><dd>Any pool worker</dd>{/if}
            </dl>
            {#if job.input_ids.length}
              <div>
                <h3 class="h">Inputs</h3>
                <ol class="inputs small">
                  {#each job.input_ids as mid, i (i)}
                    {@const m = models.find((x) => x.id === mid)}
                    <li>
                      <span>{modelLabel(m)}</span>
                      {#if m}<span class="xs muted"> · {kindLabel(m.kind)} · {hostName(hosts, m.host)}</span>{:else}<span class="xs muted mono"> {mid}</span>{/if}
                    </li>
                  {/each}
                </ol>
              </div>
            {/if}
            {#if detail && (detail.prompt_text != null || job.prompt_id)}
              <div>
                <h3 class="h">Prompt{detail.prompt_text == null && prompt ? `: ${prompt.name}` : ''}</h3>
                {#if detail.prompt_text != null}
                  <pre class="block">{detail.prompt_text}</pre>
                {:else if prompt}
                  <p class="small muted">Current saved prompt. The text may have changed after this job ran.</p>
                  <pre class="block">{prompt.text}</pre>
                {:else}
                  <p class="small muted">The text used by this job is not available.</p>
                {/if}
              </div>
            {/if}
            {#if job.options && Object.keys(job.options).length}
              <div>
                <h3 class="h">Options</h3>
                <KeyValues data={job.options} />
              </div>
            {/if}
          </div>
        </section>

        {#if detail?.events?.length}
          <section class="panel" aria-labelledby="ev-h">
            <header><h2 id="ev-h">Event log</h2></header>
            <ol class="events">
              {#each history as { event: ev, collapsed }, i (i)}
                {@const progress = eventProgress(ev)}
                <li>
                  <span class="xs muted num">{formatTime(ev.at)}</span>
                  <span class="small"><strong>{stateLabel(ev.state)}</strong>{#if progress}{' · '}{progress}{/if}</span>
                  {#if collapsed}<span class="xs muted">{collapsed} earlier progress update{collapsed === 1 ? '' : 's'} in this phase not listed</span>{/if}
                  {#if ev.error}<span class="xs err break">{ev.error}</span>{/if}
                </li>
              {/each}
            </ol>
          </section>
        {/if}
      </aside>
    </div>
  {/if}
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(260px, 340px);
    gap: var(--s-4);
    align-items: start;
  }
  @media (max-width: 1000px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .h {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
    margin-bottom: 6px;
  }
  .kv {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 4px var(--s-3);
    margin: 0;
  }
  .kv dt {
    color: var(--text-2);
  }
  .kv dd {
    margin: 0;
  }
  .inputs {
    margin: 0;
    padding-left: 1.3em;
  }
  .events {
    list-style: none;
    margin: 0;
    padding: var(--s-2) var(--s-4);
    max-height: 360px;
    overflow: auto;
  }
  .events li {
    display: flex;
    flex-direction: column;
    padding: 6px 0;
    border-bottom: 1px solid var(--line);
  }
  .events li:last-child {
    border-bottom: none;
  }
  .err {
    color: var(--bad);
  }
</style>

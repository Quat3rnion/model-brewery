<script lang="ts">
  import JobProgress from './JobProgress.svelte';
  import { brewery } from '../lib/store.svelte';
  import { errorText } from '../lib/api';
  import { formatTime, hostName, isCancelling, isTerminal, kindTitle, modelLabel, newestFirst, shortHash } from '../lib/format';
  import JobState from './JobState.svelte';
  import CancelButton from './CancelButton.svelte';
  import JobResult from './results/JobResult.svelte';
  import type { JobDetail } from '../lib/types';

  let { kinds, title, limit = 5 }: { kinds: string[]; title: string; limit?: number } = $props();

  const jobs = $derived(newestFirst((brewery.state?.jobs ?? []).filter((j) => kinds.includes(j.kind))).slice(0, limit));
  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);
  const uid = $props.id();

  let open = $state<string | null>(null);
  let detail = $state<JobDetail | null>(null);
  let loadError = $state<string | null>(null);

  const openJob = $derived(jobs.find((j) => j.id === open));
  $effect(() => {
    const j = openJob;
    if (!j) {
      detail = null;
      return;
    }
    if (!isTerminal(j.state)) {
      detail = null;
      return;
    }
    const id = j.id;
    loadError = null;
    brewery
      .loadJob(id)
      .then((d) => {
        if (open === id) detail = d;
      })
      .catch((e) => {
        if (open === id) loadError = errorText(e);
      });
  });

  function where(host: string | undefined, requested: string | undefined) {
    const name = hostName(hosts, host);
    return requested === 'auto' && host && host !== 'auto' ? `${name} (assigned from pool)` : name;
  }
</script>

<section class="panel" aria-labelledby="{uid}-h">
  <header><h2 id="{uid}-h">{title}</h2><a class="xs" href="#/queue">All jobs</a></header>
  {#if jobs.length === 0}
    <p class="empty">None yet.</p>
  {:else}
    <ul class="list">
      {#each jobs as job (job.id)}
        {@const problemId = job.error ? `${uid}-error-${encodeURIComponent(job.id)}` : undefined}
        <li>
          <div class="line">
            <div class="info">
              <div class="row">
                <JobState {job} />
                <a class="name" href="#/jobs/{encodeURIComponent(job.id)}" aria-describedby={problemId}>{kindTitle(job.kind)} <span class="xs muted mono" title={job.id}>{shortHash(job.id)}</span></a>
                <span class="xs muted">on {where(job.host, job.requested_host)}</span>
              </div>
              <p class="xs muted break">
                {job.input_ids.map((id) => modelLabel(models.find((m) => m.id === id))).join(' · ') || 'No inputs'}{#if job.created_at} · {formatTime(job.created_at)}{/if}
              </p>
              {#if job.state === 'running' || isCancelling(job)}<JobProgress {job} label="Progress of job {job.id}" />{/if}
              {#if job.error}<p class="xs err break" id={problemId}>{job.error}</p>{/if}
            </div>
            <div class="row side">
              <CancelButton {job} />
              {#if isTerminal(job.state) && (job.has_result || job.state === 'succeeded')}
                <button
                  type="button"
                  class="btn small"
                  aria-expanded={open === job.id}
                  aria-controls={open === job.id ? `${uid}-r-${job.id}` : undefined}
                  aria-label="{open === job.id ? 'Hide result' : 'Show result'}: {kindTitle(job.kind)} {shortHash(job.id)}"
                  onclick={() => (open = open === job.id ? null : job.id)}>{open === job.id ? 'Hide result' : 'Show result'}</button
                >
              {/if}
            </div>
          </div>
          {#if open === job.id}
            <div class="result" id="{uid}-r-{job.id}">
              {#if loadError}<p class="note bad" role="alert">{loadError}</p>
              {:else if detail && detail.id === job.id}<JobResult job={detail} />
              {:else}<p class="xs muted">Loading result…</p>{/if}
            </div>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .list {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .list > li {
    border-bottom: 1px solid var(--line);
    padding: var(--s-3) var(--s-4);
  }
  .list > li:last-child {
    border-bottom: none;
  }
  .line {
    display: flex;
    gap: var(--s-3);
    justify-content: space-between;
    align-items: flex-start;
  }
  .info {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
    flex: 1;
  }
  .name {
    font-weight: 600;
    color: var(--text);
    text-decoration: none;
  }
  .name:hover {
    text-decoration: underline;
  }
  .side {
    flex-wrap: nowrap;
  }
  .result {
    margin-top: var(--s-3);
    padding-top: var(--s-3);
    border-top: 1px dashed var(--line);
  }
  .err {
    color: var(--bad);
  }
  @media (max-width: 560px) {
    .line {
      flex-direction: column;
    }
  }
</style>

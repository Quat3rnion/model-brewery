<script lang="ts">
  import JobProgress from '../components/JobProgress.svelte';
  import { brewery } from '../lib/store.svelte';
  import { hostName, isActive, kindTitle, modelLabel, newestFirst, formatTime } from '../lib/format';
  import JobState from '../components/JobState.svelte';
  import CancelButton from '../components/CancelButton.svelte';

  const jobs = $derived(brewery.state?.jobs ?? []);
  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);
  let show = $state<'all' | 'active' | 'done'>('all');
  const ordered = $derived(
    // Active jobs first (they need attention), then finished ones; each group newest first.
    newestFirst(jobs)
      .filter((j) => show === 'all' || (show === 'active' ? isActive(j.state) : !isActive(j.state)))
      .sort((a, b) => Number(isActive(b.state)) - Number(isActive(a.state))),
  );
  const counts = $derived({
    all: jobs.length,
    active: jobs.filter((j) => isActive(j.state)).length,
    done: jobs.filter((j) => !isActive(j.state)).length,
  });

  function inputs(ids: string[]) {
    return ids.map((id) => modelLabel(models.find((m) => m.id === id))).join(' · ');
  }
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Jobs</h1>
    <p class="muted small">One queue for all hosts. Jobs run one at a time, except conversions on pool workers, which run in parallel with one per worker. The controller keeps the queue; closing this tab does not stop a job.</p>
  </div>

  <div class="tabs" role="group" aria-label="Show jobs">
    {#each [['all', 'All'], ['active', 'Active'], ['done', 'Finished']] as [v, l] (v)}
      <button type="button" class="btn small {show === v ? 'on' : 'ghost'}" aria-pressed={show === v} onclick={() => (show = v as typeof show)}>
        {l} <span class="num muted">{counts[v as keyof typeof counts]}</span>
      </button>
    {/each}
  </div>

  {#if ordered.length === 0}
    <p class="empty panel">{jobs.length ? 'No job in this group.' : 'No jobs yet. Start one from Transfers, Compare, Convert or Evaluate.'}</p>
  {:else}
    <ol class="jobs">
      {#each ordered as job (job.id)}
        <li class="panel job" class:running={job.state === 'running'} class:done={!isActive(job.state)}>
          <div class="main">
            <div class="row">
              <JobState {job} />
              <a class="title" href="#/jobs/{encodeURIComponent(job.id)}">{kindTitle(job.kind)}</a>
              <span class="xs muted mono">{job.id}</span>
            </div>
            <p class="xs muted break">
              {hostName(hosts, job.host)}{#if job.requested_host === 'auto' && job.host && job.host !== 'auto'} (assigned from pool){/if}{#if job.input_ids.length} · {inputs(job.input_ids)}{/if}{#if job.created_at} · {formatTime(job.created_at)}{/if}
            </p>
            {#if job.state === 'running' || (job.state === 'queued' && job.progress)}
              <JobProgress {job} label="Progress of job {job.id}" />
            {/if}
            {#if job.error}<p class="xs err break">{job.error}</p>{/if}
          </div>
          <div class="side">
            <CancelButton {job} />
            {#if job.has_result}<a class="btn small" href="#/jobs/{encodeURIComponent(job.id)}">Result</a>{/if}
          </div>
        </li>
      {/each}
    </ol>
  {/if}
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .tabs {
    display: flex;
    gap: 4px;
  }
  .btn.on {
    background: var(--accent-soft);
    border-color: var(--accent);
  }
  .jobs {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  .job {
    display: flex;
    gap: var(--s-4);
    align-items: flex-start;
    justify-content: space-between;
    padding: var(--s-3) var(--s-4);
  }
  .job.running {
    border-color: color-mix(in srgb, var(--info) 45%, var(--line));
  }
  /* Finished jobs are history: lighter rows than the active cards. */
  .job.done {
    background: transparent;
    padding-top: var(--s-2);
    padding-bottom: var(--s-2);
  }
  .main {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
    flex: 1;
  }
  .title {
    font-weight: 600;
    color: var(--text);
    text-decoration: none;
  }
  .title:hover {
    text-decoration: underline;
  }
  .side {
    display: flex;
    align-items: center;
    gap: var(--s-2);
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  .err {
    color: var(--bad);
  }
  @media (max-width: 560px) {
    .job {
      flex-direction: column;
    }
    .side {
      justify-content: flex-start;
    }
  }
</style>

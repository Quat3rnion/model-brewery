<script lang="ts">
  import JobProgress from './JobProgress.svelte';
  import JobState from './JobState.svelte';
  import CancelButton from './CancelButton.svelte';
  import JobLink from './JobLink.svelte';
  import { brewery } from '../lib/store.svelte';
  import { hostName, kindTitle } from '../lib/format';
  import type { JobSummary } from '../lib/types';

  let { jobs, queued }: { jobs: JobSummary[]; queued: number } = $props();
  const hosts = $derived(brewery.state?.hosts ?? []);
</script>

<section class="active" aria-label="Running jobs">
  {#each jobs as job (job.id)}
    <div class="line">
      <div class="what">
        <JobState {job} />
        <span class="title">{kindTitle(job.kind)}</span>
        <span class="xs muted">on {hostName(hosts, job.host)} · <JobLink id={job.id} /></span>
      </div>
      <div class="bar"><JobProgress {job} compact label="Progress of job {job.id}" /></div>
      <div class="actions"><CancelButton {job} /></div>
    </div>
  {/each}
  {#if queued}
    <p class="xs muted queued">
      {queued} job{queued === 1 ? '' : 's'} queued{jobs.length ? '' : '; none running'}. <a href="#/queue">Open jobs</a>
    </p>
  {/if}
</section>

<style>
  .active {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    padding: var(--s-2) var(--s-5);
    background: var(--surface-2);
    border-bottom: 1px solid var(--line-soft);
  }
  /* Fixed columns so the progress of every running job starts at the same place. */
  .line {
    display: grid;
    grid-template-columns: minmax(200px, 340px) minmax(0, 1fr) auto;
    align-items: start;
    gap: var(--s-2) var(--s-4);
  }
  .line + .line {
    padding-top: var(--s-2);
    border-top: 1px solid var(--line-soft);
  }
  .what {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  .title {
    font-weight: 600;
    font-size: var(--fs-md);
  }
  .actions {
    display: flex;
    align-items: center;
    gap: var(--s-3);
  }
  @media (max-width: 760px) {
    .active {
      padding: var(--s-2) var(--s-3);
    }
    .line {
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
    }
    .bar {
      grid-column: 1 / -1;
      grid-row: 2;
    }
  }
</style>

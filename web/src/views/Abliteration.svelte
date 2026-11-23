<script lang="ts">
  import { untrack } from 'svelte';
  import Badge from '../components/Badge.svelte';
  import JobLink from '../components/JobLink.svelte';
  import DirectionResult from '../components/abliteration/DirectionResult.svelte';
  import HereticResult from '../components/abliteration/HereticResult.svelte';
  import SearchResult from '../components/abliteration/SearchResult.svelte';
  import { abliterationKinds as kinds, behaviorNames, type DirectionArtifact, type SearchReceipt } from '../lib/abliteration';
  import { api, ApiError, errorText } from '../lib/api';
  import { isActive, newestFirst, stateLabel, stateTone } from '../lib/format';
  import { obj } from '../lib/reports';
  import { hashParam, route } from '../lib/route.svelte';
  import { brewery } from '../lib/store.svelte';
  import type { JobDetail, JobSummary } from '../lib/types';
  import type { ScratchRevision } from '../lib/weightspace';

  const jobs = $derived(newestFirst((brewery.state?.jobs ?? []).filter(j => Object.hasOwn(kinds, j.kind))));
  const selected = $derived(hashParam(route.hash, 'job'));
  const selectedState = $derived(jobs.find(j => j.id === selected)?.state);
  const label = (job: JobSummary) => {
    const behavior = job.options?.behavior;
    return kinds[job.kind] + (typeof behavior === 'string' ? `: ${behaviorNames[behavior] ?? behavior}` : '');
  };
  let detail = $state<JobDetail | null>(null);
  let direction = $state<DirectionArtifact | null>(null);
  let child = $state<ScratchRevision | null>(null);
  let childError = $state<string | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(false);
  const search = $derived(detail?.kind === 'search_ablation' && obj(obj(detail.result)?.screen)
    ? detail.result as SearchReceipt : null);

  $effect(() => {
    const id = selected;
    // Subscribe to state changes, not unrelated changes to the jobs list.
    void selectedState;
    const known = untrack(() => jobs.find(j => j.id === id));
    const abort = new AbortController();
    const signal = abort.signal;
    detail = null;
    direction = null;
    child = null;
    childError = null;
    error = null;
    loading = !!id;
    async function load(id: string) {
      // Current direction jobs write immutable version 2. Old jobs keep v1.
      async function loadDirection() {
        try {
          return await api.direction(id, 2, signal);
        } catch (e) {
          if (!(e instanceof ApiError) || e.status !== 404) throw e;
          return await api.direction(id, 1, signal);
        }
      }
      try {
        if (known?.kind === 'behavior_direction' && known.state === 'succeeded') {
          const result = await loadDirection();
          if (!signal.aborted) { direction = result; detail = known; }
          return;
        }
        const row = await api.job(id, signal);
        if (signal.aborted) return;
        detail = row;
        if (row.kind === 'behavior_direction' && row.state === 'succeeded') {
          const result = await loadDirection();
          if (!signal.aborted) direction = result;
        } else if (row.kind === 'heretic' || row.kind === 'heretic_reproduce') {
          const childId = obj(row.result)?.scratch_id;
          if (typeof childId === 'string') {
            try {
              const rev = await api.scratchRevision(childId, signal);
              if (!signal.aborted) child = rev;
            } catch (e) {
              if (!signal.aborted) childError = errorText(e);
            }
          }
        }
      } catch (e) {
        if (!signal.aborted) error = errorText(e);
      } finally {
        if (!signal.aborted) loading = false;
      }
    }
    if (id) void load(id);
    return () => abort.abort();
  });
</script>

<div class="stack">
  <div>
    <h1 tabindex="-1">Abliteration</h1>
    <p class="muted">Inspect fitted directions, native searches, and Heretic imports. These results are not release-qualified.</p>
  </div>
  <div class="workbench">
    <section class="panel" aria-labelledby="abliteration-runs">
      <header><h2 id="abliteration-runs">Runs</h2><span class="small muted">{jobs.length}</span></header>
      <div class="body">
        {#if jobs.length}
          <ul class="run-list">
            {#each jobs as job (job.id)}
              <li>
                <a href="#/abliteration?job={encodeURIComponent(job.id)}" aria-current={job.id === selected ? 'true' : undefined}>
                  <span>{label(job)}</span><span class="mono xs break">{job.id}</span>
                </a>
                <Badge tone={stateTone(job.state)} label={stateLabel(job.state)} />
              </li>
            {/each}
          </ul>
        {:else}
          <p class="muted">No direction, search, or Heretic jobs.</p>
        {/if}
      </div>
    </section>

    <div class="stack readout">
      {#if !selected}
        <p class="note">Select a run to read its results. Use an agent or the API to start a run.</p>
      {:else if error}
        <p class="note err" role="alert">Could not read the run: {error}</p>
      {:else if loading}
        <p class="note" role="status">Reading run…</p>
      {:else if detail && Object.hasOwn(kinds, detail.kind)}
        <div class="row"><JobLink id={detail.id} label="Job details" />
          <Badge tone={stateTone(detail.state)} label={stateLabel(detail.state)} />
        </div>
        {#if detail.error}<p class="note err" role="alert">{detail.error}</p>{/if}
        {#if direction}
          <DirectionResult artifact={direction} />
        {:else if search}
          <SearchResult result={search} />
        {:else if detail.kind === 'heretic' || detail.kind === 'heretic_reproduce'}
          <HereticResult job={detail} {child} {childError} />
        {:else}
          <p class="note">{isActive(detail.state) ? 'Results will appear after the job finishes.' : 'No results are available for this run.'}</p>
        {/if}
      {:else}
        <p class="note">This job is not an abliteration run.</p>
      {/if}
    </div>
  </div>
</div>

<style>
  .workbench { display: grid; align-items: start; grid-template-columns: minmax(13rem, 1fr) minmax(0, 3fr); gap: var(--s-4); }
  .readout { min-width: 0; }
  .run-list { list-style: none; padding: 0; margin: 0; }
  .run-list li { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2); padding-block: var(--s-2); }
  .run-list li + li { border-top: 1px solid var(--line-soft); }
  .run-list a { display: grid; min-width: 0; border-radius: var(--r-sm); padding: var(--s-1); }
  .run-list a[aria-current] { background: var(--surface-2); font-weight: 600; }
  @media (max-width: 760px) { .workbench { grid-template-columns: minmax(0, 1fr); } }
</style>

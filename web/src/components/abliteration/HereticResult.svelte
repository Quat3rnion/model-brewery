<script lang="ts">
  import { obj } from '../../lib/reports';
  import type { JobDetail } from '../../lib/types';
  import type { ScratchRevision } from '../../lib/weightspace';

  let { job, child, childError }: { job: JobDetail; child: ScratchRevision | null; childError: string | null } = $props();
  const journal = $derived(obj(child?.provenance.heretic));
  const req = $derived(obj(journal?.invocation ?? job.options?.invocation));
  const result = $derived(obj(job.result));
  const childId = $derived(typeof result?.scratch_id === 'string' ? result.scratch_id : null);
  const runtime = $derived(obj(journal?.runtime));
  const trials = $derived(Array.isArray(journal?.trials)
    ? journal.trials as { index: number; scores: { name: string; baseline: number; value: number }[] }[] : []);
  const text = (value: unknown) => typeof value === 'string' || typeof value === 'number' ? String(value) : 'Not reported';
</script>

<section class="panel" aria-labelledby="heretic-title">
  <header><h2 id="heretic-title">{job.kind === 'heretic_reproduce' ? 'Heretic reproduction' : 'Heretic run'}</h2></header>
  <div class="body stack">
    <dl>
      {#if req?.profile}<dt>Profile</dt><dd>{text(req.profile)}</dd>{/if}
      {#if req?.trials !== undefined}<dt>Trial budget</dt><dd class="num">{text(req.trials)}</dd>{/if}
      {#if req?.trial_index !== undefined}<dt>Requested Pareto index</dt><dd class="num">{text(req.trial_index)}</dd>{/if}
      {#if journal?.selected_study_trial !== undefined}<dt>Selected study trial</dt><dd class="num">{text(journal.selected_study_trial)}</dd>{/if}
      <dt>Seed</dt><dd class="num">{text(journal?.seed ?? req?.seed)}</dd>
      <dt>Base SHA-256</dt><dd class="mono break">{text(journal?.base_sha256 ?? job.options?.base_sha256)}</dd>
      {#if runtime}
        <dt>Runtime</dt><dd>Heretic {text(runtime.heretic)} · {text(runtime.licence)}</dd>
        <dt>Lock SHA-256</dt><dd class="mono break">{text(runtime.lock_sha256)}</dd>
      {/if}
      {#if childId}
        <dt>Output</dt><dd><a href="#/edits?rev={encodeURIComponent(childId)}">Imported child</a></dd>
        <dt>Output SHA-256</dt><dd class="mono break">{text(result?.sha256)}</dd>
      {/if}
    </dl>
    {#if childError}<p class="note err" role="alert">Could not read the imported child: {childError}</p>{/if}
    {#if journal}
      {#if trials.length}
        <p class="small muted">Heretic's trial scores use its optimization prompts. They are not Brewery's held-out checks.</p>
        <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard users must be able to scroll the table) -->
        <div class="table-wrap" tabindex="0" role="region" aria-label="Heretic trial scores">
          <table class="data num" aria-label="Heretic trials">
            <thead><tr><th scope="col">Trial</th><th scope="col">Scorer</th><th scope="col">Baseline</th><th scope="col">Trial score</th></tr></thead>
            <tbody>
              {#each trials as trial (trial.index)}
                {#each trial.scores as s}
                  <tr><th scope="row">{trial.index}{trial.index === journal.selected_study_trial ? ' Selected' : ''}</th>
                    <td class="break">{s.name}</td><td>{s.baseline}</td><td>{s.value}</td></tr>
                {/each}
              {/each}
            </tbody>
          </table>
        </div>
      {:else}
        <p class="small muted">This journal does not contain per-trial scores.</p>
      {/if}
      {#if typeof journal.source_config === 'string'}
        <details class="disclosure"><summary>Source config</summary><pre>{journal.source_config}</pre></details>
      {/if}
      {#if typeof journal.effective_config === 'string'}
        <details class="disclosure"><summary>Effective config</summary><pre>{journal.effective_config}</pre></details>
      {/if}
      <details class="disclosure"><summary>Reproduce journal</summary><pre>{JSON.stringify(journal, null, 2)}</pre></details>
    {:else if childId && !childError}
      <p class="small muted">No reproduce journal was available in the imported child.</p>
    {/if}
  </div>
</section>

<style>
  dl { margin: 0; display: grid; grid-template-columns: minmax(5rem, max-content) minmax(0, 1fr); gap: var(--s-2) var(--s-4); }
  dt { color: var(--text-2); }
  dd { margin: 0; min-width: 0; }
  pre { margin: var(--s-3) 0 0; padding: var(--s-3); background: var(--surface-2); white-space: pre-wrap; overflow-wrap: anywhere; }
</style>

<script lang="ts">
  // Progress of one job: controller progress_detail first, then a fresh
  // legacy live sample for the same running job, then the plain progress.
  // The rate comes from the shared tracker in lib/live.svelte.ts, so every
  // view of the same job shows the same measured value.
  import Progress from './Progress.svelte';
  import { detailFraction, detailLines, legacyDetail, phaseEtaFor, shownDetail, threadsFor, unitLabel } from '../lib/progress';
  import { etaText, rateText } from '../lib/rate';
  import { describeStep } from '../lib/steps';
  import { live } from '../lib/live.svelte';
  import { progressText } from '../lib/format';
  import type { JobSummary } from '../lib/types';

  // compact: bar, main line, then rate and age on one line. No input or file lines.
  let { job, label, compact = false }: { job: JobSummary; label: string; compact?: boolean } = $props();
  const shown = $derived(shownDetail(job, live.sample, live.now));
  const detail = $derived(shown?.detail ?? null);
  const sample = $derived(shown?.observed ?? false);
  const lines = $derived(detail ? detailLines(detail, live.now) : null);
  // Who runs this check, and whether this page saw the same job move to it.
  const step = $derived.by(() => {
    if (!detail) return null;
    const s = live.steps(job.id);
    return describeStep(job, detail, s.history, s.dropped);
  });
  const threads = $derived(detail && !compact ? threadsFor(job, detail, live.now) : null);
  const fraction = $derived(detail ? detailFraction(detail) : null);
  const legacy = $derived(progressText(job.progress));
  const footer = $derived(
    [sample ? 'Observed from the running process, not reported by the controller.' : '', lines?.age ?? '', fraction !== null ? 'Percent covers this phase only.' : '']
      .filter(Boolean)
      .join(' · '),
  );
  const legacyCounter = $derived(detail ? null : legacyDetail(job.progress));
  const measuring = $derived(job.state === 'running' && !job.cancel_requested);
  // The counter that is measured, its rate view, and the phase time left from the same view.
  const measured = $derived.by(() => {
    if (!measuring) return null;
    const m = shown ? { source: shown.source, detail: shown.detail } : legacyCounter ? { source: 'legacy' as const, detail: legacyCounter } : null;
    if (!m) return null;
    const view = live.rate(job.id, m.source, m.detail);
    return { rate: rateText(view, unitLabel(m.detail)), eta: phaseEtaFor(job, view, m.detail, live.now) };
  });
  const rate = $derived(measured?.rate ?? null);
  const eta = $derived(etaText(measured?.eta ?? null, compact));
</script>

{#if detail && lines}
  <div class="jp">
    {#if fraction !== null}
      <div class="track" role="progressbar" aria-label={label} aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.floor(fraction * 100)} aria-valuetext={lines.main}>
        <div class="fill" style:width="{fraction * 100}%"></div>
      </div>
    {:else}
      <div class="track indeterminate" role="progressbar" aria-label={label} aria-valuetext={lines.main}><div class="fill"></div></div>
    {/if}
    <p class="xs main num break">{lines.main}</p>
    {#if compact}
      <!-- One muted line: the banner shows the job kind and ID beside this. -->
      <p class="xs muted num break">
        {#if step}<span class="role">{step.compactRole}</span>{' · '}{/if}{#if step?.compactTransition}<strong class="new">{step.compactTransition}</strong>{' · '}{/if}{[rate, eta, sample ? 'Observed from the running process' : '', lines.age].filter(Boolean).join(' · ')}
      </p>
    {:else}
      {#if step}<p class="xs role break">{step.role}</p>{/if}
      {#if step?.purpose}<p class="xs muted break">{step.purpose}</p>{/if}
      {#if step?.transition}<p class="xs transition break">{step.transition}</p>{/if}
      {#each lines.extra as x (x)}<p class="xs muted num break">{x}</p>{/each}
      {#if rate}<p class="xs num rate">{rate}</p>{/if}
      {#if eta}<p class="xs num rate">{eta}</p>{/if}
      {#if threads}
        <details class="threads">
          <summary class="xs">CPU thread activity ({threads.length} {threads.length === 1 ? 'slot' : 'slots'} reported)</summary>
          <p class="xs muted">Snapshot by logical worker slot, not CPU core. Only slots the worker reported are listed.</p>
          <table class="xs num">
            <thead><tr><th scope="col">Slot</th><th scope="col">Task</th></tr></thead>
            <tbody>
              {#each threads as t (t.slot)}<tr><td>{t.slot}</td><td class="break">{t.task}</td></tr>{/each}
            </tbody>
          </table>
        </details>
      {/if}
      {#if step?.sequence}<p class="xs muted break">{step.sequence}</p>{/if}
      {#if footer}<p class="xs muted">{footer}</p>{/if}
    {/if}
  </div>
{:else}
  <Progress progress={job.progress} {label} />
  {#if rate}<p class="xs num rate">{compact && eta ? `${rate} · ${eta}` : rate}</p>{/if}
  {#if eta && !compact}<p class="xs num rate">{eta}</p>{/if}
  {#if !legacy && job.state === 'running'}<p class="xs muted">No measured progress yet.</p>{/if}
{/if}

<style>
  .jp {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  .track {
    position: relative;
    height: 6px;
    border-radius: 3px;
    background: var(--surface-2);
    overflow: hidden;
    margin-bottom: 2px;
  }
  .fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.3s ease;
  }
  .indeterminate .fill {
    width: 30%;
    animation: slide 1.4s ease-in-out infinite;
  }
  @keyframes slide {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(340%);
    }
  }
  .main {
    color: var(--text);
  }
  .role {
    color: var(--text);
    font-weight: 600;
  }
  .new {
    color: var(--text);
    font-weight: 600;
  }
  .transition {
    color: var(--text);
    border-left: 3px solid var(--accent);
    padding-left: 6px;
    margin: 1px 0;
  }
  .rate {
    color: var(--text);
    margin: 0;
  }
  p {
    margin: 0;
  }
  .threads {
    margin: 2px 0;
  }
  .threads summary {
    cursor: pointer;
    color: var(--text);
    width: fit-content;
    min-height: 24px;
    display: list-item;
  }
  .threads summary:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 2px;
  }
  .threads table {
    border-collapse: collapse;
    margin-top: 2px;
    width: 100%;
    max-width: 40rem;
  }
  .threads th,
  .threads td {
    text-align: left;
    padding: 1px var(--s-2, 8px) 1px 0;
    vertical-align: top;
  }
  .threads th {
    font-weight: 600;
    color: var(--text-2, inherit);
  }
  .threads td:first-child {
    width: 3.5em;
    font-variant-numeric: tabular-nums;
  }
</style>

<script lang="ts">
  // One durable conversion workflow: stage list with job links, output entry, cancel.
  import Badge from './Badge.svelte';
  import JobProgress from './JobProgress.svelte';
  import ComponentEvidence from './ComponentEvidence.svelte';
  import { selectionFrom } from '../lib/components';
  import { api, errorText } from '../lib/api';
  import { brewery } from '../lib/store.svelte';
  import { conversions } from '../lib/conversions.svelte';
  import { formatTime, hostName, modelLabel } from '../lib/format';
  import { parseConversion, stageRows, STAGE_LABEL, type Conversion, type StageStatus } from '../lib/conversion';
  import type { Tone } from '../lib/format';

  let { c }: { c: Conversion } = $props();
  const uid = $props.id();
  const jobs = $derived(brewery.state?.jobs ?? []);
  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);
  const states = $derived(Object.fromEntries(jobs.map((j) => [j.id, String(j.state)])));
  const rows = $derived(stageRows(c, states));
  const current = $derived(c.stage && c.state === 'running' ? jobs.find((j) => j.id === c.jobIds[c.stage!]) : undefined);
  const source = $derived(models.find((m) => m.id === c.inputIds[0]));
  const output = $derived(c.artifactModelId ? models.find((m) => m.id === c.artifactModelId) : undefined);
  const ggufPreset = $derived(c.format === 'gguf' && typeof c.options.preset === 'string' ? c.options.preset : null);
  const profile = $derived(c.format === 'gguf' ? `GGUF ${c.profileRevision ? `${c.profileName ?? 'profile'} · ${c.profileRevision.slice(0, 10)}` : (ggufPreset ?? '')}` : c.profileRevision ? `${c.profileName ?? 'Custom profile'} · ${c.profileRevision.slice(0, 10)}` : typeof c.options.profile === 'string' ? (c.options.profile === 'v2' ? 'HGN v2' : c.options.profile === 'ht43' ? 'HT43' : c.options.profile) : 'Converter default');

  const TONE: Record<StageStatus, Tone> = { pending: 'muted', queued: 'muted', running: 'info', succeeded: 'ok', failed: 'bad', cancelled: 'warn', skipped: 'muted', unknown: 'muted' };
  const LABEL: Record<StageStatus, string> = { pending: 'Waiting', queued: 'Queued', running: 'Running', succeeded: 'Done', failed: 'Failed', cancelled: 'Cancelled', skipped: 'Not run', unknown: 'Unknown' };
  const OVERALL: Record<string, [Tone, string]> = { running: ['info', 'Running'], succeeded: ['ok', 'Verified bundle ready'], failed: ['bad', 'Failed'], cancelled: ['warn', 'Cancelled'] };
  const requested = $derived(selectionFrom(c.inputIds, c.options));
  const replaced = $derived(Object.values(requested).some(Boolean));
  // Newest stage result that states components (plan, convert or verify).
  const evidenceJob = $derived(
    (['verify', 'convert', 'plan', 'inspect'] as const).map((st) => c.jobIds[st]).filter((id): id is string => !!id).map((id) => brewery.details[id]).find((d) => d && d.result && typeof d.result === 'object' && 'components' in (d.result as object)),
  );
  $effect(() => {
    // Load finished stage details once so their component evidence can be shown.
    for (const st of ['inspect', 'plan', 'convert', 'verify'] as const) {
      const id = c.jobIds[st];
      const j = id ? jobs.find((x) => x.id === id) : undefined;
      if (id && j && j.state === 'succeeded' && !brewery.details[id]) void brewery.loadJob(id).catch(() => {});
    }
  });
  const overall = $derived(OVERALL[c.state] ?? (['muted', c.state] as [Tone, string]));

  let confirming = $state(false);
  let busy = $state(false);
  let error = $state<string | null>(null);
  async function cancel() {
    busy = true;
    error = null;
    try {
      const r = parseConversion(await api.cancelConversion(c.id));
      if (r) conversions.upsert(r);
      confirming = false;
      void conversions.refresh();
    } catch (e) {
      error = errorText(e);
    } finally {
      busy = false;
    }
  }
</script>

<article class="run" aria-labelledby="{uid}-h">
  <header class="top">
    <div class="who">
      <h3 id="{uid}-h" class="title">{source ? modelLabel(source) : (c.inputIds[0] ?? 'Conversion')} → {profile}</h3>
      <p class="xs muted break">
        on {c.requestedHost === 'auto' && c.host ? `${hostName(hosts, c.host)} (pool)` : hostName(hosts, c.host)}{#if c.createdAt}{' · started '}{formatTime(c.createdAt)}{/if}{' · '}<span class="mono">{c.id}</span>
      </p>
    </div>
    <div class="side">
      <Badge tone={overall[0]} label={overall[1]} />
      {#if c.state === 'running'}
        {#if c.cancelRequested}
          <span class="xs muted">Cancel requested</span>
        {:else if confirming}
          <button type="button" class="btn danger small" disabled={busy} onclick={cancel}>{busy ? 'Cancelling…' : 'Confirm cancel'}</button>
          <button type="button" class="btn ghost small" disabled={busy} onclick={() => (confirming = false)}>Keep running</button>
        {:else}
          <button type="button" class="btn danger small" onclick={() => (confirming = true)} aria-label="Cancel conversion {c.id}">Cancel</button>
        {/if}
      {/if}
    </div>
  </header>
  {#if error}<p class="note bad" role="alert">{error}</p>{/if}

  <ol class="stages" aria-label="Stages">
    {#each rows as r (r.stage)}
      <li class:now={r.status === 'running'}>
        <span class="sname">{STAGE_LABEL[r.stage]}</span>
        <Badge tone={TONE[r.status]} label={LABEL[r.status]} />
        {#if r.jobId}<a class="xs" href="#/jobs/{encodeURIComponent(r.jobId)}" aria-label="{STAGE_LABEL[r.stage]} job {r.jobId}">Job</a>{/if}
      </li>
    {/each}
  </ol>

  {#if current}
    <JobProgress job={current} compact label="Progress of {current.kind} stage" />
  {/if}
  {#if replaced || evidenceJob}
    <details class="disclosure">
      <summary><span class="xs"><strong>Components</strong></span><span class="xs muted">{replaced ? 'Replacements requested' : 'Defaults'}{evidenceJob ? ' · reported' : ''}</span></summary>
      <ComponentEvidence {requested} result={evidenceJob?.result} title="Per component" />
    </details>
  {/if}
  {#if c.error}<p class="note bad">{c.error}</p>{/if}
  {#if c.state === 'succeeded'}
    <p class="small">
      {#if output}Output: <strong>{modelLabel(output)}</strong> in the Library.{:else if c.artifactModelId}Output entry <span class="mono">{c.artifactModelId}</span>.{:else}Finished; the controller did not name an output entry.{/if}
      File verification passed. This does not prove GPU execution or quality.
    </p>
  {/if}
</article>

<style>
  .run {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    padding: var(--s-3) var(--s-4);
    border-bottom: 1px solid var(--line-soft);
  }
  .run:last-child {
    border-bottom: none;
  }
  .top {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-4);
    justify-content: space-between;
    align-items: flex-start;
  }
  .who {
    min-width: 0;
  }
  .title {
    font-size: var(--fs-md);
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .side {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--s-2);
  }
  .stages {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--s-2);
  }
  .stages li {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 8px;
    padding: 6px 8px;
    border: 1px solid var(--line-soft);
    border-radius: var(--r-md);
    min-width: 0;
  }
  .stages li.now {
    border-color: var(--info);
  }
  .sname {
    font-size: var(--fs-sm);
    font-weight: 500;
    width: 100%;
  }
  @media (max-width: 560px) {
    .stages {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>

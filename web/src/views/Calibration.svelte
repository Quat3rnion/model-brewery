<script lang="ts">
  import { formatBytes } from '../lib/format';
  import { readFileBounded, MAX_FILE_BYTES, type Parsed } from '../lib/calibration';
  import UsageReportView from '../components/calibration/UsageReport.svelte';
  import PreviewReportView from '../components/calibration/PreviewReport.svelte';
  import SpeculationReportView from '../components/calibration/SpeculationReport.svelte';

  interface Loaded {
    key: string;
    name: string;
    size: number;
    parsed: Parsed;
  }

  let loaded = $state<Loaded[]>([]);
  let busy = $state(false);
  let seq = 0;
  let input: HTMLInputElement | undefined = $state();

  async function open(files: FileList | null) {
    if (!files?.length) return;
    busy = true;
    const next: Loaded[] = [];
    for (const f of Array.from(files)) next.push({ key: `${seq++}`, name: f.name, size: f.size, parsed: await readFileBounded(f) });
    loaded = [...loaded, ...next];
    busy = false;
    if (input) input.value = '';
  }
  function remove(key: string) {
    loaded = loaded.filter((l) => l.key !== key);
  }
  const titles = { usage: 'Expert usage', preview: 'Profile size preview', speculation: 'Speculation acceptance', error: 'Not opened' } as const;
  const ok = $derived(loaded.filter((l) => l.parsed.kind !== 'error').length);
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Calibration inputs</h1>
    <p class="muted small">
      Read derived reports from <code>expert_usage.py</code>, <code>profile_preview.py</code> and the Halogen speculation export. These are descriptive statistics and size predictions. They are not calibration or quality results, and a preview is not a runnable profile.
    </p>
  </div>

  <section class="panel" aria-labelledby="open-h">
    <header><h2 id="open-h">Open reports</h2></header>
    <div class="body stack">
      <div class="field">
        <label for="cal-files">Report files (JSON)</label>
        <input id="cal-files" bind:this={input} class="input" type="file" accept=".json,application/json" multiple disabled={busy} aria-describedby="cal-files-hint" onchange={(e) => open(e.currentTarget.files)} />
        <span class="hint" id="cal-files-hint">
          Files stay in this browser tab. Nothing goes to the controller, and no model is loaded. At most {formatBytes(MAX_FILE_BYTES)} per file. Raw traces are not accepted.
        </span>
      </div>
      <p class="sr-only" aria-live="polite">{busy ? 'Reading files…' : loaded.length ? `${ok} of ${loaded.length} files opened.` : ''}</p>
    </div>
  </section>

  {#if loaded.length === 0}
    <p class="empty">No reports open.</p>
  {/if}

  {#each loaded as l (l.key)}
    <section class="panel" aria-labelledby="rep-{l.key}-h">
      <header>
        <h2 id="rep-{l.key}-h" class="rh"><span>{titles[l.parsed.kind]}</span> <span class="mono xs muted break">{l.name}</span></h2>
        <button type="button" class="btn small ghost" onclick={() => remove(l.key)} aria-label="Close {l.name}">Close</button>
      </header>
      <div class="body">
        {#if l.parsed.kind === 'error'}
          <p class="note bad" role="alert">{l.parsed.message}</p>
        {:else if l.parsed.kind === 'usage'}
          <UsageReportView report={l.parsed.report} />
        {:else if l.parsed.kind === 'preview'}
          <PreviewReportView report={l.parsed.report} />
        {:else}
          <SpeculationReportView report={l.parsed.report} />
        {/if}
      </div>
    </section>
  {/each}

  {#if ok > 1}
    <p class="xs muted">Each report stands alone. The viewer does not match usage, preview or speculation reports to each other, or to traces and plans, even when names look alike.</p>
  {/if}
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .rh {
    display: flex;
    flex-wrap: wrap;
    gap: 0 8px;
    align-items: baseline;
    min-width: 0;
  }
  input[type='file'] {
    padding: 6px;
  }
</style>

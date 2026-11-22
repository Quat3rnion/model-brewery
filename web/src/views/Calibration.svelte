<script module lang="ts">
  import { readFileBounded, MAX_FILE_BYTES, type Parsed } from '../lib/calibration';
  interface Loaded {
    key: string;
    name: string;
    parsed: Parsed;
  }

  // Keep local reports in this tab when the router unmounts this view.
  let loaded = $state.raw<Loaded[]>([]);
  let busy = $state(false);
  let seq = 0;

  async function open(files: File[]) {
    if (!files.length || busy) return;
    busy = true;
    try {
      const next: Loaded[] = [];
      for (const f of files) next.push({ key: `${seq++}`, name: f.name, parsed: await readFileBounded(f) });
      loaded = [...loaded, ...next];
    } finally {
      busy = false;
    }
  }
  const commands = [
    ['Expert usage', 'python3 qualification/expert_usage.py trace.json > usage.json'],
    ['Expert map', 'python3 qualification/expert_plan.py frequency --usage usage.json --source-sha256 "$SOURCE_SHA256" > expert-map.json'],
    ['Profile size preview', 'python3 qualification/profile_preview.py --recipe recipe.json --v2-plan v2.json --ht43-plan ht43.json > preview.json'],
    ['Speculation acceptance', 'python3 qualification/halogen_speculation.py --producer-version "$HALOGEN_VERSION" responses.jsonl > speculation.json'],
  ];
</script>

<script lang="ts">
  import { formatBytes } from '../lib/format';
  import { focusAfterRemoval } from '../lib/focus';
  import Badge from '../components/Badge.svelte';
  import Ids from '../components/calibration/Ids.svelte';
  import UsageReportView from '../components/calibration/UsageReport.svelte';
  import PreviewReportView from '../components/calibration/PreviewReport.svelte';
  import SpeculationReportView from '../components/calibration/SpeculationReport.svelte';

  let input: HTMLInputElement | undefined = $state();
  let heading: HTMLHeadingElement | undefined = $state();
  function readSelection(field: HTMLInputElement) {
    const files = Array.from(field.files ?? []);
    field.value = '';
    void open(files);
  }
  const remove = (key: string) => focusAfterRemoval(
    () => { loaded = loaded.filter((l) => l.key !== key); },
    () => input?.disabled ? heading : input,
  );
  const titles = { expertMap: 'Expert map review', usage: 'Expert usage', preview: 'Profile size preview', speculation: 'Speculation acceptance', error: 'Not opened' } as const;
  const ok = $derived(loaded.filter((l) => l.parsed.kind !== 'error').length);
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Calibration inputs</h1>
    <p class="muted small">
      Open expert usage, expert maps, profile size previews, or Halogen speculation reports. Reviewing a file does not change a checkpoint.
    </p>
  </div>

  <section class="panel" aria-labelledby="open-h">
    <header><h2 id="open-h" bind:this={heading} tabindex="-1">Open reports</h2></header>
    <div class="body stack">
      <div class="field">
        <label for="cal-files">Report files (JSON)</label>
        <input id="cal-files" bind:this={input} class="input" type="file" accept=".json,application/json" multiple disabled={busy} aria-describedby="cal-files-hint" onchange={(e) => readSelection(e.currentTarget)} />
        <span class="hint" id="cal-files-hint">
          Files stay in this browser tab. Nothing goes to the controller, and no model is loaded. At most {formatBytes(MAX_FILE_BYTES)} per file. Raw traces are not accepted.
        </span>
      </div>
      <p class="sr-only" aria-live="polite">{busy ? 'Reading files…' : loaded.length ? `${ok} of ${loaded.length} files opened.` : ''}</p>
    </div>
  </section>

  {#if loaded.length === 0}
    <section class="panel" aria-labelledby="commands-h">
      <header><h2 id="commands-h">Make a report</h2></header>
      <div class="body stack">
        <p class="small">No reports open.</p>
        <p class="small muted">Run a command from the repository, then open its JSON output. Use your own input paths, source SHA-256 or Halogen version.</p>
        <dl class="commands stack small">
          {#each commands as [title, command] (title)}
            <div><dt>{title}</dt><dd><code class="break">{command}</code></dd></div>
          {/each}
        </dl>
      </div>
    </section>
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
        {:else if l.parsed.kind === 'expertMap'}
          <div class="stack">
            <div><Badge tone="muted" label="Not applied" /></div>
            <Ids title="Map details" compact={false} items={[
              ['Source checkpoint SHA-256', l.parsed.report.source],
              ['Map file SHA-256', l.parsed.fileSha ?? 'Unavailable'],
              ['Ranking method', l.parsed.report.method],
            ]} />
            <p class="small">Each list shows the original expert IDs in the proposed order. The first ID goes to slot 0.</p>
            {#each l.parsed.report.layers as layer (layer.layer)}
              <details><summary>Layer {layer.layer}: {layer.order.length} retained</summary><p class="mono break">{layer.order.join(', ')}</p></details>
            {/each}
          </div>
        {:else}
          <SpeculationReportView report={l.parsed.report} />
        {/if}
      </div>
    </section>
  {/each}

  {#if ok > 1}
    <p class="xs muted">Each report stands alone. The viewer does not match reports to each other or to traces and plans.</p>
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
  .commands, .commands dd { margin: 0; }
  .commands dt { font-weight: 600; }
</style>

<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { BUNDLE_KINDS, capability, hostName, isTerminal, modelLabel } from '../lib/format';
  import { benchmarkLine, generationLine, latestJobFor, lineTone, verificationLine, type ReportLine } from '../lib/reports';
  import HostSelect from '../components/HostSelect.svelte';
  import Combobox from '../components/Combobox.svelte';
  import { idOptions, promptOptions } from '../lib/pickers';
  import ModelSelect from '../components/ModelSelect.svelte';
  import Submit from '../components/Submit.svelte';
  import SubmitStatus from '../components/SubmitStatus.svelte';
  import RecentJobs from '../components/RecentJobs.svelte';
  import Badge from '../components/Badge.svelte';
  import type { Host, JobSummary } from '../lib/types';

  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);
  const jobs = $derived(brewery.state?.jobs ?? []);
  const prompts = $derived(brewery.state?.prompts ?? []);
  const bundles = $derived(models.filter((m) => BUNDLE_KINDS.has(m.kind)));

  function sources(h: Host | undefined): { id: string; label?: string }[] {
    const list = h?.baseline_sources;
    return Array.isArray(list) ? list.filter((x): x is { id: string; label?: string } => !!x && typeof x.id === 'string') : [];
  }
  function hostOf(id: string, override: string) {
    return hosts.find((h) => h.id === (override || models.find((m) => m.id === id)?.host));
  }
  function capBlock(kind: string, h: Host | undefined, out: string[]) {
    if (!h) return;
    const c = capability(h, kind);
    if (!c.allowed && c.reason) out.push(c.reason);
  }

  // ---- Report status per bundle (three separate lines) ----
  // Load terminal detail records for the latest verify/generate/evaluate job of each bundle.
  $effect(() => {
    for (const b of bundles) {
      for (const kind of ['verify', 'generate', 'evaluate']) {
        const j = latestJobFor(jobs, kind, b.id);
        if (j && isTerminal(j.state) && !brewery.details[j.id]) void brewery.loadJob(j.id).catch(() => {});
      }
    }
  });
  function lines(id: string): { name: string; line: ReportLine }[] {
    const pick = (kind: string) => {
      const j: JobSummary | null = latestJobFor(jobs, kind, id);
      return [j, j ? brewery.details[j.id] : undefined] as const;
    };
    return [
      { name: 'File verification', line: verificationLine(...pick('verify')) },
      { name: 'Generation', line: generationLine(...pick('generate')) },
      { name: 'Benchmark quality', line: benchmarkLine(...pick('evaluate')) },
    ];
  }

  // ---- Generate ----
  let gBundle = $state('');
  let gHost = $state('');
  let gMode = $state<'saved' | 'text'>('saved');
  let gPromptId = $state('');
  let gText = $state('');
  const gen = new Submitter();
  const gBlockers = $derived.by(() => {
    const out: string[] = [];
    if (!gBundle) out.push('Select a bundle.');
    if (gMode === 'saved' && !gPromptId) out.push('Select a saved prompt.');
    if (gMode === 'text' && !gText) out.push('Enter a prompt.');
    capBlock('generate', hostOf(gBundle, gHost), out);
    return out;
  });

  // ---- Evaluate ----
  let eBundle = $state('');
  let eHost = $state('');
  let eBaseMode = $state<'none' | 'file' | 'source'>('none');
  let eBaseFile = $state('');
  let eSource = $state('');
  const eSources = $derived(sources(hostOf(eBundle, eHost)));
  const ev = new Submitter();
  const eBlockers = $derived.by(() => {
    const out: string[] = [];
    if (!eBundle) out.push('Select a bundle.');
    if (eBaseMode === 'file' && !eBaseFile) out.push('Select reference results.');
    if (eBaseMode === 'source' && !eSources.some(s=>s.id===eSource)) out.push('Select a configured comparison endpoint.');
    capBlock('evaluate', hostOf(eBundle, eHost), out);
    return out;
  });

  // ---- Baseline ----
  let bMode = $state<'file' | 'source'>('file');
  let bFile = $state('');
  let bHost = $state('');
  let bSource = $state('');
  const bSources = $derived(sources(hosts.find((h) => h.id === bHost)));
  const base = new Submitter();
  const bBlockers = $derived.by(() => {
    const out: string[] = [];
    if (bMode === 'file' && !bFile) out.push('Select a predictions or baseline file.');
    if (bMode === 'source') {
      if (!bHost) out.push('Select a host.');
      if (!bSource) out.push('Select a baseline source.');
    }
    capBlock('baseline', bMode === 'source' ? hosts.find((h) => h.id === bHost) : hostOf(bFile, bHost), out);
    return out;
  });
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Evaluate</h1>
    <p class="muted small">
      Score a bundle on the pinned GSM8K split. To measure a difference, compare it with another model’s results from the same protocol. File verification and generated text are separate checks.
    </p>
  </div>

  {#if hosts.length && !hosts.some((h) => capability(h, 'evaluate').allowed)}
    <p class="note info">No worker can run GSM8K evaluation yet. <a href="#/settings">Check evaluation settings</a>.</p>
  {/if}

  <section class="panel" aria-labelledby="st-h">
    <header><h2 id="st-h">Bundle status</h2></header>
    {#if bundles.length === 0}
      <p class="empty">No bundles in the library. Convert a checkpoint, or register a bundle directory.</p>
    {:else}
      <div class="table-wrap">
        <table class="data status stackable">
          <caption class="sr-only">Latest file verification, generation and benchmark result for each bundle</caption>
          <thead>
            <tr>
              <th scope="col">Bundle</th>
              <th scope="col">File verification</th>
              <th scope="col">Generation</th>
              <th scope="col">Benchmark quality</th>
            </tr>
          </thead>
          <tbody>
            {#each bundles as b (b.id)}
              <tr>
                <th scope="row">
                  <div class="break">{modelLabel(b)}</div>
                  <div class="xs muted">{b.kind} · {hostName(hosts, b.host)}</div>
                </th>
                {#each lines(b.id) as { name, line } (name)}
                  <td data-label={name}>
                    <Badge tone={lineTone(line.status)} label={line.label} />
                    <div class="xs muted break" style="margin-top:4px">{line.detail}</div>
                    {#if line.jobId}<a class="xs" href="#/jobs/{encodeURIComponent(line.jobId)}" aria-label="{name} job for {modelLabel(b)}">Job</a>{/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>

  <h2 class="group-h">Benchmark quality</h2>
  {#snippet prepareReference()}
    <details class="disclosure">
      <summary>Prepare a reusable comparison result</summary>
      <div class="body stack">
        <p class="small muted">Import predictions from a reference model, or score a configured endpoint on the same questions.</p>
        <div class="seg" role="radiogroup" aria-label="Baseline input">
          <label><input type="radio" bind:group={bMode} value="file" /> Import predictions file</label>
          <label><input type="radio" bind:group={bMode} value="source" /> Configured source</label>
        </div>
        <div class="grid-2">
          {#if bMode === 'file'}
            <ModelSelect bind:value={bFile} kinds={['gsm8k-predictions', 'gsm8k-baseline']} label="Predictions or baseline file" />
            <HostSelect bind:value={bHost} kind="baseline" label="Run on" optional optionalLabel="Host of the file" />
          {:else}
            <HostSelect bind:value={bHost} kind="baseline" label="Host" />
            <div class="field">
              <Combobox bind:value={bSource} id="b-src" label="Baseline source" options={idOptions(bSources)} disabled={!bHost || bSources.length === 0} placeholder={bHost && bSources.length === 0 ? 'None configured on this host' : bHost ? 'Select or type a source' : 'Select a host first'} emptyText="No source matches." />
              <span class="hint">Comparison endpoints are configured in <a href="#/settings">Settings</a>.</span>
            </div>
          {/if}
        </div>
        <Submit
          blockers={bBlockers}
          busy={base.busy}
          onclick={() =>
            base.submit(
              bMode === 'file'
                ? { kind: 'baseline', host: bHost || undefined, input_ids: [bFile] }
                : { kind: 'baseline', host: bHost, input_ids: [], options: { baseline_source: bSource } },
            )}>Queue comparison result</Submit
        >
        <SubmitStatus s={base} />
      </div>
    </details>
  {/snippet}
    <section class="panel" aria-labelledby="ev-h">
      <header><h3 id="ev-h" class="ph">GSM8K evaluation</h3></header>
      <div class="body stack">
        <ModelSelect bind:value={eBundle} kinds={BUNDLE_KINDS} label="Bundle" />
        <HostSelect bind:value={eHost} kind="evaluate" label="Run on" optional optionalLabel="Host of the bundle" />
        <fieldset class="plain">
          <legend class="small muted">Scoring mode</legend>
          <div class="seg">
            <label><input type="radio" bind:group={eBaseMode} value="none" /> Standalone score</label>
            <label><input type="radio" bind:group={eBaseMode} value="file" /> Compare with a file</label>
            <label><input type="radio" bind:group={eBaseMode} value="source" /> Compare with an endpoint</label>
          </div>
          {#if eBaseMode === 'file'}
            <ModelSelect bind:value={eBaseFile} kinds={['gsm8k-baseline','gsm8k-evaluation','gsm8k-predictions']} label="Reference results" hint="Select results or predictions for the same GSM8K protocol. The reference can use a different model or precision." />
            {@render prepareReference()}
          {:else if eBaseMode === 'source'}
            <div class="field">
              {#if eSources.length}
                <Combobox bind:value={eSource} id="e-src" label="Comparison endpoint" options={idOptions(eSources)} placeholder="Select an endpoint" emptyText="No endpoint matches." />
              {:else if !hostOf(eBundle,eHost)}<p class="xs muted">Select the bundle or worker to see its comparison endpoints.</p>
              {:else}<p class="xs muted">No comparison endpoint on this worker. <a href="#/settings">Check Settings</a> or compare with a file instead.</p>{/if}
            </div>
          {/if}
        </fieldset>
        <Submit
          blockers={eBlockers}
          busy={ev.busy}
          onclick={() =>
            ev.submit({
              kind: 'evaluate',
              host: eHost || undefined,
              input_ids: eBaseMode === 'file' ? [eBundle, eBaseFile] : [eBundle],
              options: eBaseMode === 'source' ? { baseline_source: eSource } : undefined,
            })}>Queue full GSM8K evaluation</Submit
        >
        <SubmitStatus s={ev} />
      </div>
    </section>

  <h2 class="group-h">Generation check</h2>
  <section class="panel" aria-labelledby="gen-h">
    <header><h3 id="gen-h" class="ph">Generate</h3></header>
    <div class="body stack">
      <ModelSelect bind:value={gBundle} kinds={BUNDLE_KINDS} label="Bundle" />
      <HostSelect bind:value={gHost} kind="generate" label="Run on" optional optionalLabel="Host of the bundle" hint="Needs a worker with an approved Halogen runtime. The run records which files it opens." />
      <div class="seg" role="radiogroup" aria-label="Prompt source">
        <label><input type="radio" bind:group={gMode} value="saved" /> Saved prompt</label>
        <label><input type="radio" bind:group={gMode} value="text" /> Type a prompt</label>
      </div>
      {#if gMode === 'saved'}
        <div class="field">
          <Combobox bind:value={gPromptId} id="g-prompt" label="Saved prompt" labelHidden options={promptOptions(prompts)} disabled={!prompts.length} placeholder={prompts.length ? 'Select or type a prompt name' : 'No saved prompts'} emptyText="No saved prompt matches." />
        </div>
      {:else}
        <div class="field"><label for="g-text" class="sr-only">Prompt text</label><textarea id="g-text" class="input" rows="3" bind:value={gText}></textarea></div>
      {/if}
      <Submit blockers={gBlockers} busy={gen.busy} onclick={() => gen.submit({ kind: 'generate', host: gHost || undefined, input_ids: [gBundle], prompt_id: gMode === 'saved' ? gPromptId : undefined, prompt: gMode === 'text' ? gText : undefined })}>Queue generation</Submit>
      <SubmitStatus s={gen} />
    </div>
  </section>

  <RecentJobs kinds={['generate', 'evaluate', 'baseline']} title="Recent generation and evaluation jobs" limit={8} />
</div>

<style>
  .group-h {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
    margin: var(--s-2) 0 calc(-1 * var(--s-2));
  }
  .ph {
    font-size: var(--fs-title);
    letter-spacing: -0.005em;
  }
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .status th[scope='row'] {
    font-weight: 500;
    min-width: 160px;
  }
  .status td {
    min-width: 170px;
  }
  .seg {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-4);
    font-size: var(--fs-sm);
  }
  .seg label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 32px;
    cursor: pointer;
  }
  input[type='radio'] {
    accent-color: var(--accent);
    width: 16px;
    height: 16px;
  }
  .plain {
    border: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
    min-width: 0;
  }
  .plain legend {
    padding: 0;
    margin-bottom: 6px;
  }
</style>

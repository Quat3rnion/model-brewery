<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { BUNDLE_KINDS, capability, hostName, isTerminal, modelLabel } from '../lib/format';
  import { benchmarkLine, generationLine, latestJobFor, lineTone, verificationLine, type ReportLine } from '../lib/reports';
  import { fieldError, visibleBlockers, type Blocker } from '../lib/blockers';
  import { hostWorkflowEligibility } from '../lib/eligibility';
  import { unsavedChanges } from '../lib/unsaved-changes.svelte';
  import HostSelect from '../components/HostSelect.svelte';
  import Combobox from '../components/Combobox.svelte';
  import { idOptions, promptOptions } from '../lib/pickers';
  import ModelSelect from '../components/ModelSelect.svelte';
  import WorkflowGate from '../components/WorkflowGate.svelte';
  import Blockers from '../components/Blockers.svelte';
  import FieldError from '../components/FieldError.svelte';
  import SubmitStatus from '../components/SubmitStatus.svelte';
  import RecentJobs from '../components/RecentJobs.svelte';
  import Badge from '../components/Badge.svelte';
  import type { Host, JobSummary, Workflow } from '../lib/types';

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
  function hostBlockers(workflow: Workflow, host: Host | undefined): Blocker[] {
    if (!host) return [];
    const entry = hostWorkflowEligibility(brewery.state?.eligibility, host.id, workflow);
    const legacy = entry ? undefined : capability(host, workflow);
    if (entry?.eligible !== false && legacy?.allowed !== false) return [];
    const capabilityReason = host.capability_reasons?.[workflow];
    const reasons = entry?.reasons ?? (capabilityReason ? [capabilityReason] : []);
    return reasons.length
      ? reasons.map((r) => ({ text: r.text, fixAnchor: r.fix_anchor }))
      : [{ host: host.name, text: legacy?.reason ?? 'This host cannot run this workflow.', fixAnchor: evaluationSettings(host) }];
  }
  const evaluationSettings = (host: Host) => `#/settings/evaluation?host=${encodeURIComponent(host.id)}`;
  const touched = $state<Record<string, boolean>>({});
  const touch = (field: string) => () => { touched[field] = true; };
  const refInputs = ['gsm8k-predictions', 'gsm8k-baseline'];
  const refResults = [...refInputs, 'gsm8k-evaluation'];

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
      { name: 'GSM8K score', line: benchmarkLine(...pick('evaluate')) },
    ];
  }

  // ---- Generate ----
  let gBundle = $state('');
  let gHost = $state('');
  let gMode = $state<'saved' | 'text'>('saved');
  let gPromptId = $state('');
  let gText = $state('');
  let submittedText = $state('');
  unsavedChanges(() => gText !== '' && gText !== submittedText);
  const gen = new Submitter();
  const gHostBlockers = $derived(hostBlockers('generate', hostOf(gBundle, gHost)));
  const gFields = $derived<Blocker[]>([
    ...(!gBundle ? [{ field: 'g-bundle', text: 'Select a bundle.' }] : []),
    ...(gMode === 'saved' && !prompts.some((p) => p.id === gPromptId) ? [{ field: 'g-prompt', text: 'Select a saved prompt.' }] : []),
    ...(gMode === 'text' && !gText.trim() ? [{ field: 'g-text', text: 'Enter a prompt.' }] : []),
  ]);
  async function generate() {
    const text = gMode === 'text' ? gText : undefined;
    const job = await gen.submit({ kind: 'generate', host: gHost || undefined, input_ids: [gBundle], prompt_id: gMode === 'saved' ? gPromptId : undefined, prompt: text });
    if (job && text !== undefined) submittedText = text;
  }

  // ---- Evaluate ----
  let eBundle = $state('');
  let eHost = $state('');
  let eBaseMode = $state<'none' | 'file' | 'source'>('none');
  let eBaseFile = $state('');
  let eSource = $state('');
  const eWorker = $derived(hostOf(eBundle, eHost));
  const eSources = $derived(sources(eWorker));
  const ev = new Submitter();
  const eHostBlockers = $derived(hostBlockers('evaluate', eWorker));
  const eFields = $derived<Blocker[]>([
    ...(!eBundle ? [{ field: 'e-bundle', text: 'Select a bundle.' }] : []),
    ...(eBaseMode === 'file' && !eBaseFile ? [{ field: 'e-file', text: 'Select reference results.' }] : []),
    ...(eBaseMode === 'source' && !eSources.some((s) => s.id === eSource) ? [{ field: 'e-src', text: 'Select a reference endpoint.' }] : []),
  ]);

  // ---- Reference ----
  let bMode = $state<'file' | 'source'>('file');
  let bFile = $state('');
  let bHost = $state('');
  let bSource = $state('');
  const bWorker = $derived(hostOf(bMode === 'file' ? bFile : '', bHost));
  const bSources = $derived(sources(bWorker));
  const base = new Submitter();
  const bHostBlockers = $derived(hostBlockers('baseline', bWorker));
  const bFields = $derived<Blocker[]>([
    ...(bMode === 'file' && !bFile ? [{ field: 'b-file', text: 'Select reference results or predictions.' }] : []),
    ...(bMode === 'source' && !bHost ? [{ field: 'b-host', text: 'Select a host.' }] : []),
    ...(bMode === 'source' && !bSources.some((s) => s.id === bSource) ? [{ field: 'b-src', text: 'Select a reference endpoint.' }] : []),
  ]);
  const visible = $derived(visibleBlockers([...gFields, ...eFields, ...bFields], touched));
  const error = (field: string) => fieldError(visible, field);
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Evaluate</h1>
    <p class="muted small">
      Score a bundle on the pinned GSM8K split. To measure a difference, compare it with another model’s results from the same protocol. File verification and generated text are separate checks.
    </p>
  </div>

  <section class="panel" aria-labelledby="st-h">
    <header><h2 id="st-h">Bundle status</h2></header>
    {#if bundles.length === 0}
      <p class="empty">No bundles in the <a href="#/library">Library</a>. Convert a checkpoint, or register a bundle directory.</p>
    {:else}
      <div class="table-wrap">
        <table class="data status stackable">
          <caption class="sr-only">Latest file verification, generation and GSM8K score for each bundle</caption>
          <thead>
            <tr>
              <th scope="col">Bundle</th>
              <th scope="col">File verification</th>
              <th scope="col">Generation</th>
              <th scope="col">GSM8K score</th>
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

  <h2 class="group-h">GSM8K score</h2>
  <section class="panel" aria-labelledby="ev-h">
    <header><h3 id="ev-h" class="ph">GSM8K evaluation</h3></header>
    <WorkflowGate workflow="evaluate">
      <div class="body stack">
        <Blockers blockers={eHostBlockers} />
        <ModelSelect bind:value={eBundle} kinds={BUNDLE_KINDS} label="Bundle" error={error('e-bundle')} onblur={touch('e-bundle')} />
        <HostSelect bind:value={eHost} workflow="evaluate" kind="evaluate" label="Run on" optional optionalLabel="Host of the bundle" />
        <fieldset class="plain">
          <legend class="small muted">Scoring mode</legend>
          <div class="seg">
            <label><input type="radio" name="evaluation-mode" bind:group={eBaseMode} value="none" /> Standalone score</label>
            <label><input type="radio" name="evaluation-mode" bind:group={eBaseMode} value="file" /> Use a reference file</label>
            <label><input type="radio" name="evaluation-mode" bind:group={eBaseMode} value="source" /> Use a reference endpoint</label>
          </div>
          {#if eBaseMode === 'file'}
            <ModelSelect bind:value={eBaseFile} kinds={refResults} label="Reference results" hint="Select results or predictions for the same GSM8K protocol. The reference can use a different model or precision." error={error('e-file')} onblur={touch('e-file')} />
            {#if !models.some((model) => refResults.includes(model.kind))}
              <p class="hint">Register a reference file in <a href="#/library">Library</a>, or prepare reference results below.</p>
            {/if}
          {:else if eBaseMode === 'source'}
            <div class="field">
              {#if eSources.length}
                <Combobox bind:value={eSource} id="e-src" label="Reference endpoint" options={idOptions(eSources)} placeholder="Select an endpoint" emptyText="No endpoint matches." invalid={!!error('e-src')} describedby={error('e-src') ? 'e-src-error' : undefined} onblur={touch('e-src')} />
                <FieldError id="e-src-error" error={error('e-src')} />
              {:else if eWorker}
                <p class="small muted">No reference endpoint on this host. <a href={evaluationSettings(eWorker)}>Set up an endpoint for {eWorker.name}</a>, or use a reference file.</p>
              {:else}
                <p class="small muted">Select a bundle or host to see its reference endpoints.</p>
              {/if}
            </div>
          {/if}
        </fieldset>
        {@render queue('Queue full GSM8K evaluation', eFields, eHostBlockers, ev, () =>
          ev.submit({
            kind: 'evaluate',
            host: eHost || undefined,
            input_ids: eBaseMode === 'file' ? [eBundle, eBaseFile] : [eBundle],
            options: eBaseMode === 'source' ? { baseline_source: eSource } : undefined,
          }))}
      </div>
    </WorkflowGate>
  </section>

  <section class="panel" aria-labelledby="ref-h">
    <header><h3 id="ref-h" class="ph">Prepare reference results</h3></header>
    <WorkflowGate workflow="baseline">
      <div class="body stack">
        <Blockers blockers={bHostBlockers} />
        <p class="small muted">Import predictions from a reference model, or score a configured endpoint on the same questions.</p>
        <fieldset class="plain">
          <legend class="small muted">Reference input</legend>
          <div class="seg">
            <label><input type="radio" name="reference-input" bind:group={bMode} value="file" /> Import a file</label>
            <label><input type="radio" name="reference-input" bind:group={bMode} value="source" /> Use a reference endpoint</label>
          </div>
        </fieldset>
        <div class="grid-2">
          {#if bMode === 'file'}
            <div class="stack">
              <ModelSelect bind:value={bFile} kinds={refInputs} label="Reference results or predictions" error={error('b-file')} onblur={touch('b-file')} />
              {#if !models.some((model) => refInputs.includes(model.kind))}
                <p class="hint">Register a reference file in <a href="#/library">Library</a>.</p>
              {/if}
            </div>
            <HostSelect bind:value={bHost} workflow="baseline" kind="baseline" label="Run on" optional optionalLabel="Host of the file" />
          {:else}
            <HostSelect bind:value={bHost} workflow="baseline" kind="baseline" label="Host" error={error('b-host')} onblur={touch('b-host')} />
            <div class="field">
              <Combobox bind:value={bSource} id="b-src" label="Reference endpoint" options={idOptions(bSources)} disabled={!bSources.length} placeholder={!bWorker ? 'Select a host first' : bSources.length ? 'Select an endpoint' : 'No endpoint on this host'} emptyText="No endpoint matches." invalid={!!error('b-src')} describedby={error('b-src') ? 'b-src-error' : undefined} onblur={touch('b-src')} />
              <FieldError id="b-src-error" error={error('b-src')} />
              {#if bWorker && !bSources.length}<span class="hint"><a href={evaluationSettings(bWorker)}>Set up a reference endpoint for {bWorker.name}</a>.</span>{/if}
            </div>
          {/if}
        </div>
        {@render queue('Queue reference results', bFields, bHostBlockers, base, () => base.submit(
          bMode === 'file'
            ? { kind: 'baseline', host: bHost || undefined, input_ids: [bFile] }
            : { kind: 'baseline', host: bHost, input_ids: [], options: { baseline_source: bSource } },
        ))}
      </div>
    </WorkflowGate>
  </section>

  <h2 class="group-h">Generation check</h2>
  <section class="panel" aria-labelledby="gen-h">
    <header><h3 id="gen-h" class="ph">Generate</h3></header>
    <WorkflowGate workflow="generate">
      <div class="body stack">
        <Blockers blockers={gHostBlockers} />
        <ModelSelect bind:value={gBundle} kinds={BUNDLE_KINDS} label="Bundle" error={error('g-bundle')} onblur={touch('g-bundle')} />
        <HostSelect bind:value={gHost} workflow="generate" kind="generate" label="Run on" optional optionalLabel="Host of the bundle" hint="Needs a worker with an approved Halogen runtime. The run records which files it opens." />
        <fieldset class="plain">
          <legend class="small muted">Prompt source</legend>
          <div class="seg">
            <label><input type="radio" name="generation-prompt" bind:group={gMode} value="saved" /> Saved prompt</label>
            <label><input type="radio" name="generation-prompt" bind:group={gMode} value="text" /> Type a prompt</label>
          </div>
          {#if gMode === 'saved'}
            <div class="field">
              <Combobox bind:value={gPromptId} id="g-prompt" label="Saved prompt" labelHidden options={promptOptions(prompts)} disabled={!prompts.length} placeholder={prompts.length ? 'Select or type a prompt name' : 'No saved prompts'} emptyText="No saved prompt matches." invalid={!!error('g-prompt')} describedby={error('g-prompt') ? 'g-prompt-error' : undefined} onblur={touch('g-prompt')} />
              <FieldError id="g-prompt-error" error={error('g-prompt')} />
              {#if !prompts.length}<span class="hint">Save a prompt in <a href="#/library">Library</a>, or type a prompt for this run.</span>{/if}
            </div>
          {:else}
            <div class="field">
              <label for="g-text">Prompt text</label>
              <textarea id="g-text" class="input" rows="3" bind:value={gText} aria-invalid={!!error('g-text')} aria-describedby={error('g-text') ? 'g-text-error' : undefined} onblur={touch('g-text')}></textarea>
              <FieldError id="g-text-error" error={error('g-text')} />
            </div>
          {/if}
        </fieldset>
        {@render queue('Queue generation', gFields, gHostBlockers, gen, generate)}
      </div>
    </WorkflowGate>
  </section>

  <RecentJobs kinds={['generate', 'evaluate', 'baseline']} title="Recent generation, evaluation and reference jobs" limit={8} />
</div>

{#snippet queue(label: string, fields: Blocker[], hosts: Blocker[], s: Submitter, submit: () => Promise<unknown>)}
  <div>
    <button type="button" class="btn primary" disabled={s.busy || fields.length > 0 || hosts.length > 0} aria-busy={s.busy} onclick={submit}>{s.busy ? 'Submitting…' : label}</button>
  </div>
  <SubmitStatus {s} />
{/snippet}

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

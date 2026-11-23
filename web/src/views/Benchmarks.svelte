<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { fieldError, visibleBlockers } from '../lib/blockers';
  import { hostWorkflowEligibility } from '../lib/eligibility';
  import { formatTime, hostName, isTerminal, modelLabel, newestFirst, stateLabel } from '../lib/format';
  import { benchmarkLine, lineTone, accuracyText, readScore, obj } from '../lib/reports';
  import {
    benchmarkChoices,
    buildBenchmarkJob,
    buildScorecard,
    exportBlockers,
    freshDetail,
    choiceLabel,
    compareRows,
    directionText,
    formatValue,
    missingSuites,
    orderedSuites,
    provenanceRows,
    readBenchmarkReport,
    requestedOf,
    statusTone,
    MAX_SUITES,
  } from '../lib/benchmark';
  import HostSelect from '../components/HostSelect.svelte';
  import Combobox from '../components/Combobox.svelte';
  import { idOptions, visibleChecklist } from '../lib/pickers';
  import WorkflowGate from '../components/WorkflowGate.svelte';
  import Blockers from '../components/Blockers.svelte';
  import FieldError from '../components/FieldError.svelte';
  import SubmitStatus from '../components/SubmitStatus.svelte';
  import RecentJobs from '../components/RecentJobs.svelte';
  import Badge from '../components/Badge.svelte';
  import JobState from '../components/JobState.svelte';
  import BenchmarkReport from '../components/results/BenchmarkReport.svelte';
  import Ids from '../components/calibration/Ids.svelte';
  import type { JobDetail, JobSummary } from '../lib/types';
  import { errorText } from '../lib/api';
  import { availableId, loadRetryJob, omittedInputs, RetryPrefill, watchPrefill } from '../lib/retry-prefill.svelte';
  import RetryOffer from '../components/RetryOffer.svelte';
  import RetryRecovery from '../components/RetryRecovery.svelte';
  import RetryWarnings from '../components/RetryWarnings.svelte';

  const hosts = $derived(brewery.state?.hosts ?? []);
  const models = $derived(brewery.state?.models ?? []);
  const jobs = $derived(brewery.state?.jobs ?? []);

  // ---- Launcher ----
  let hostId = $state('');
  let target = $state('');
  let picked = $state<string[]>([]);
  let suiteOrder = $state<string[]>([]);
  const host = $derived(hosts.find((h) => h.id === hostId));
  const offer = $derived(benchmarkChoices(host));
  // Drop selections that the newly selected host does not advertise.
  $effect(() => {
    const ids = new Set(offer.suites.map((s) => s.id));
    if (picked.some((p) => !ids.has(p))) {
      retry.prune('picked', values => values.filter(id => ids.has(id)));
      picked = picked.filter(id => ids.has(id));
    }
    if (target && !offer.targets.some((t) => t.id === target)) target = '';
  });
  let suiteFilter = $state('');
  const suitesShown = $derived(visibleChecklist(offer.suites.map((x) => ({ ...x, text: `${x.label ?? ''} ${x.id}` })), suiteFilter, picked));
  const hostDecision = $derived(hostWorkflowEligibility(brewery.state?.eligibility, hostId, 'benchmark'));
  const setupLink = $derived(host && `#/settings/evaluation?host=${encodeURIComponent(host.id)}`);
  const run = new Submitter('benchmarks.run');
  const retry = new RetryPrefill({
    slot: 'benchmarks', defaults: { hostId: '', target: '', picked: [] as string[], suiteOrder: [] as string[] },
    read: () => ({ hostId, target, picked, suiteOrder }),
    write: value => { ({ hostId, target, picked, suiteOrder } = value); },
    load: loadRetryJob(['benchmark']),
    restore: job => {
      const request = requestedOf(job);
      const hostId = availableId(job.requested_host ?? job.host, hosts);
      const choices = benchmarkChoices(hosts.find(h => h.id === hostId));
      const picked = request.benchmarks.filter(id => choices.suites.some(suite => suite.id === id));
      const target = choices.targets.find(t => t.id === request.benchmark_target)?.id ?? '';
      return {
        actions: ['benchmark'], patch: { hostId, target, picked, suiteOrder: request.benchmarks }, focus: 'bm-host',
        retains: draft => draft.picked.some(id => request.benchmarks.includes(id)),
        omissions: draft => omittedInputs(request.benchmarks, draft.picked).map(id => ({ id,
          text: `Suite ${id} ${benchmarkChoices(hosts.find(h => h.id === draft.hostId)).suites.some(s => s.id === id)
            ? 'was not restored. Select it to include it.' : 'is not available on this host.'}` })),
      };
    },
  });
  const ordered = $derived(orderedSuites(picked, retry.pending ? requestedOf(retry.pending).benchmarks : suiteOrder));
  const built = $derived(buildBenchmarkJob(host, target, ordered, hostDecision));
  const touched = $state({ host: false, target: false, suites: false });
  const visible = $derived(visibleBlockers(built.blockers, touched));
  const error = (field: string) => fieldError(visible, field);
  watchPrefill(retry, () => !!brewery.state);
  const launch = $derived(retry.action('benchmark'));
  function queueBenchmark() {
    if (!built.job || launch.blocked || run.busy) return;
    void run.submit(built.job, launch);
  }

  // ---- Scorecard selection ----
  const benchJobs = $derived(newestFirst(jobs.filter((j) => j.kind === 'benchmark')));
  const gsmJobs = $derived(newestFirst(jobs.filter((j) => j.kind === 'evaluate' && isTerminal(j.state))));
  const canSelectJobs = $derived(benchJobs.length > 0 || gsmJobs.length > 0);
  let selected = $state<string[]>([]);
  let gsmSelected = $state<string[]>([]);
  let seeded = false;
  $effect(() => {
    // Select the newest finished benchmark job once, so the scorecard is not empty on first visit.
    if (seeded) return;
    const first = benchJobs.find((j) => j.state === 'succeeded') ?? benchJobs.find((j) => isTerminal(j.state));
    if (first) {
      seeded = true;
      selected = [first.id];
    }
  });
  // Load the detail of each selected terminal job. A detail cached while the
  // job ran is stale once its summary is terminal, so it is fetched again.
  let loadErrors = $state<Record<string, string>>({});
  const inflight = new Set<string>();
  function load(id: string) {
    if (inflight.has(id)) return;
    inflight.add(id);
    delete loadErrors[id];
    brewery
      .loadJob(id, true)
      .catch((e) => (loadErrors[id] = errorText(e)))
      .finally(() => inflight.delete(id));
  }
  // Auto-load once per job and state; Retry loads again on request.
  const attempted = new Set<string>();
  $effect(() => {
    for (const id of [...selected, ...gsmSelected]) {
      const j = jobs.find((x) => x.id === id);
      if (!j || !isTerminal(j.state) || loadErrors[id] || freshDetail(j, brewery.details[id])) continue;
      const key = `${id}\u0000${j.state}`;
      if (attempted.has(key)) {
        if (!inflight.has(id)) loadErrors[id] = 'The loaded record does not match the current job state.';
        continue;
      }
      attempted.add(key);
      load(id);
    }
  });
  const fresh = (j: JobSummary): JobDetail | undefined => freshDetail(j, brewery.details[j.id]);
  const chosen = $derived(benchJobs.filter((j) => selected.includes(j.id)));
  const chosenGsm = $derived(gsmJobs.filter((j) => gsmSelected.includes(j.id)));
  const columns = $derived(
    chosen.flatMap((j) => {
      const report = readBenchmarkReport(fresh(j)?.result);
      return report ? [{ jobId: j.id, host: j.host, report }] : [];
    }),
  );
  const rows = $derived(compareRows(columns));

  function toggle(list: string[], id: string, on: boolean): string[] {
    return on ? (list.includes(id) ? list : [...list, id]) : list.filter((x) => x !== id);
  }

  function suiteSummary(j: JobSummary): string {
    const r = requestedOf(j);
    return `${r.benchmark_target ?? 'no target'} · ${r.benchmarks.join(', ') || 'no suites'}`;
  }

  let exportNote = $state('');
  function exportJson() {
    if (blockers.length) return;
    const pair = (j: JobSummary) => ({ summary: j, detail: fresh(j) });
    const card = buildScorecard(chosen.map(pair), chosenGsm.map(pair), hosts, new Date().toISOString());
    const blob = new Blob([JSON.stringify(card, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `brewery-scorecard-${card.generated_at.replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    exportNote = `Exported ${card.benchmark_jobs.length} benchmark job${card.benchmark_jobs.length === 1 ? '' : 's'} and ${card.gsm8k_jobs.length} GSM8K job${card.gsm8k_jobs.length === 1 ? '' : 's'}.`;
  }
  const blockers = $derived(exportBlockers([...chosen, ...chosenGsm], brewery.details, loadErrors));
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Benchmarks</h1>
    <p class="muted small">
      Run benchmark suites against an operator-configured endpoint. Endpoint runs never qualify a release. Each job keeps its own target, model and protocol; the scorecard does not average across benchmarks or jobs.
    </p>
  </div>

  <section class="panel" aria-labelledby="bl-h">
    <header><h2 id="bl-h">Run benchmarks</h2></header>
    {#if retry.failed}<div class="body"><RetryRecovery reason={retry.failure} reload={retry.reload} startNew={retry.startNew} /></div>{/if}
    <WorkflowGate workflow="benchmark">
    <div class="body stack" oninputcapture={retry.touchField} onchangecapture={retry.touchField} onclickcapture={retry.touchField} onkeydowncapture={retry.touchField}>
      {#if retry.loadingReason}<p class="hint" role="status">{retry.loadingReason}</p>{/if}
      {#if retry.offered}<RetryOffer apply={retry.usePrevious} />{/if}
      <RetryWarnings items={retry.omissions('benchmark')} label="Acknowledge omitted suites"
        acknowledged={retry.acknowledged('benchmark')}
        acknowledge={() => { retry.acknowledge('benchmark'); document.getElementById('bm-host')?.focus(); }} />
      <Blockers blockers={built.blockers.filter((b) => !b.field)} />
      <div class="grid-2">
        <HostSelect id="bm-host" bind:value={hostId} workflow="benchmark" kind="benchmark" label="Host" hint="The host runs the suites against the endpoint. No library input is used." error={error('host')} onblur={() => { touched.host = true; }} />
        <div class="field">
          <Combobox bind:value={target} id="bm-target" label="Endpoint target" options={idOptions(offer.targets)} disabled={!host || offer.targets.length === 0} invalid={!!error('target')} describedby="bm-target-hint{error('target') ? ' bm-target-error' : ''}" placeholder={!host ? 'Select a host first' : offer.targets.length ? 'Select or type a target' : 'None advertised'} emptyText="No target matches." onblur={() => { touched.target = true; }} />
          <FieldError id="bm-target-error" error={error('target')} />
          <span class="hint" id="bm-target-hint">
            {#if setupLink && !offer.targets.length}<a href={setupLink}>Set up a benchmark target for {host?.name ?? hostId}</a>.
            {:else}The run sends only the target ID.{/if}
          </span>
        </div>
      </div>

      <fieldset class="suites" aria-describedby="bm-count{error('suites') ? ' bm-suites-error' : ''}">
        <legend class="small">Suites <span class="muted xs" id="bm-count">{picked.length} of at most {MAX_SUITES} selected</span></legend>
        {#if !host}
          <p class="xs muted">Select a host to see its suites.</p>
        {:else if offer.suites.length === 0}
          <p class="xs muted">No suites are enabled on this host. <a href={setupLink}>Set up suites for {host.name ?? hostId}</a>.</p>
        {:else}
          {#if offer.suites.length > 6}
            <div class="sfilter">
              <label class="sr-only" for="bm-suite-filter">Filter suites</label>
              <input id="bm-suite-filter" class="input" type="search" bind:value={suiteFilter} placeholder="Filter suites…" autocomplete="off" spellcheck="false" />
              {#if suiteFilter.trim()}<span class="xs muted" aria-live="polite">{suitesShown.length} of {offer.suites.length} shown; selected suites stay listed</span>{/if}
            </div>
          {/if}
          <div class="checks">
            {#each suitesShown as s (s.id)}
              {@const on = picked.includes(s.id)}
              <label class="check" class:off={!on && picked.length >= MAX_SUITES}>
                <input type="checkbox" checked={on} disabled={!on && picked.length >= MAX_SUITES} onblur={() => { touched.suites = true; }} onchange={(e) => (picked = toggle(picked, s.id, e.currentTarget.checked))} />
                <span>{s.label ?? s.id}{#if s.label && s.label !== s.id}{' '}<span class="mono xs muted">{s.id}</span>{/if}</span>
              </label>
            {/each}
          </div>
        {/if}
        <FieldError id="bm-suites-error" error={error('suites')} />
        {#if ordered.length > 1}
          <p class="hint">Run order: {ordered.map(id => choiceLabel(offer.suites, id)).join(', ')}.</p>
        {/if}
      </fieldset>

      <div><button type="button" class="btn primary" disabled={launch.blocked || !built.job || run.busy} aria-busy={run.busy} onclick={queueBenchmark}>{run.busy ? 'Submitting…' : 'Queue benchmark run'}</button></div>
      <SubmitStatus s={run} current={built.job} />
    </div>
    </WorkflowGate>
  </section>

  <section class="panel" aria-labelledby="sc-h">
    <header>
      <h2 id="sc-h">Scorecard</h2>
      {#if canSelectJobs}
        <button type="button" class="btn small" onclick={exportJson} disabled={blockers.length > 0} aria-describedby="sc-export-hint{blockers.length ? ' sc-export-why' : ''}">Export JSON</button>
      {/if}
    </header>
    <div class="body stack">
      {#if canSelectJobs}
        <p class="xs muted" id="sc-export-hint">
          The export lists each selected job with its ID, host, request, full report, and any failed or missing suites. GSM8K evaluations have separate entries with their own identity and reference provenance. The export does not match them to benchmark jobs.
        </p>
        {#if blockers.length}<ul class="why xs" id="sc-export-why">{#each blockers as b (b)}<li>{b}</li>{/each}</ul>{/if}
        <p class="sr-only" aria-live="polite">{exportNote}</p>
      {/if}

      <div class="grid-2 pick">
        <fieldset>
          <legend class="small">Benchmark jobs</legend>
          {#if benchJobs.length === 0}
            <p class="xs muted">No benchmark jobs yet.</p>
          {:else}
            <ul class="joblist">
              {#each benchJobs as j (j.id)}
                <li>
                  <label class="check">
                    <input type="checkbox" checked={selected.includes(j.id)} onchange={(e) => (selected = toggle(selected, j.id, e.currentTarget.checked))} />
                    <span class="jl">
                      <span class="row"><JobState job={j} /><span class="mono xs">{j.id}</span></span>
                      <span class="xs muted break">{hostName(hosts, j.host)} · {suiteSummary(j)}{#if j.created_at}{' · '}{formatTime(j.created_at)}{/if}</span>
                    </span>
                  </label>
                </li>
              {/each}
            </ul>
          {/if}
        </fieldset>
        <fieldset>
          <legend class="small">GSM8K evaluations (separate)</legend>
          {#if gsmJobs.length === 0}
            <p class="xs muted">No finished GSM8K evaluations.</p>
          {:else}
            <ul class="joblist">
              {#each gsmJobs as j (j.id)}
                <li>
                  <label class="check">
                    <input type="checkbox" checked={gsmSelected.includes(j.id)} onchange={(e) => (gsmSelected = toggle(gsmSelected, j.id, e.currentTarget.checked))} />
                    <span class="jl">
                      <span class="row"><JobState job={j} /><span class="mono xs">{j.id}</span></span>
                      <span class="xs muted break">{hostName(hosts, j.host)} · {j.input_ids.map((id) => modelLabel(models.find((m) => m.id === id))).join(' · ') || 'No inputs'}</span>
                    </span>
                  </label>
                </li>
              {/each}
            </ul>
          {/if}
        </fieldset>
      </div>
    </div>
  </section>

  {#snippet loadState(id: string)}
    {#if loadErrors[id]}
      <div class="note bad row" role="alert">
        <span>Result could not be loaded: {loadErrors[id]}</span>
        <button type="button" class="btn small" onclick={() => load(id)}>Retry</button>
      </div>
    {:else}
      <p class="xs muted">Loading result…</p>
    {/if}
  {/snippet}

  {#each chosen as j (j.id)}
    {@const d = fresh(j)}
    {@const report = readBenchmarkReport(d?.result)}
    {@const req = requestedOf(j)}
    <section class="panel" aria-labelledby="job-{j.id}-h">
      <header>
        <h2 id="job-{j.id}-h" class="jh">
          <span>Benchmark job</span> <a class="mono" href="#/jobs/{encodeURIComponent(j.id)}">{j.id}</a>
        </h2>
        <span class="xs muted">on {hostName(hosts, j.host)}</span>
      </header>
      <div class="body stack">
        <p class="xs muted break">
          Requested target <span class="mono">{req.benchmark_target ?? '—'}</span>{#if req.benchmark_target}{@const tl = choiceLabel(benchmarkChoices(hosts.find((h) => h.id === j.host)).targets, req.benchmark_target)}{#if tl !== req.benchmark_target}{' '}({tl}){/if}{/if} · suites <span class="mono">{req.benchmarks.join(', ') || '—'}</span>
        </p>
        {#if !isTerminal(j.state)}
          <p class="small muted">{stateLabel(j.state)}. No result yet.</p>
        {:else if !d}
          {@render loadState(j.id)}
        {:else if report}
          <BenchmarkReport {report} missing={missingSuites(j, report)} requestedTarget={req.benchmark_target} />
        {:else}
          {#if d.error ?? j.error}<p class="note bad">{d.error ?? j.error}</p>{/if}
          <p class="small muted">
            {d.result === undefined || d.result === null ? 'No report recorded.' : 'The result is not a brewery-benchmark-report-v1 report.'} All requested suites count as missing.
          </p>
        {/if}
      </div>
    </section>
  {/each}

  {#each chosenGsm as j (j.id)}
    {@const d = fresh(j)}
    {@const line = benchmarkLine(j, d)}
    {@const score = readScore(obj(d?.result))}
    <section class="panel" aria-labelledby="gsm-{j.id}-h">
      <header>
        <h2 id="gsm-{j.id}-h" class="jh"><span>GSM8K evaluation</span> <a class="mono" href="#/jobs/{encodeURIComponent(j.id)}">{j.id}</a></h2>
        <span class="xs muted">on {hostName(hosts, j.host)}</span>
      </header>
      <div class="body stack">
        {#if !d}
          {@render loadState(j.id)}
        {:else}
          {@const r = obj(d.result)}
          {@const id = obj(r?.identity)}
          <div class="row"><Badge tone={lineTone(line.status)} label={line.label} />{#if score}<span class="num small">{accuracyText(score)}</span>{/if}</div>
          <p class="xs muted break">{line.detail}</p>
          <dl class="prov xs">
            <dt>Source SHA-256</dt><dd class="mono break">{typeof id?.source_sha256 === 'string' ? id.source_sha256 : 'Not stated'}</dd>
            <dt>Data</dt><dd>{id?.synthetic === false ? 'Real (not synthetic)' : id?.synthetic === true ? 'Synthetic fixture' : 'Not stated'}</dd>
          </dl>
          <Ids title="Provenance" items={provenanceRows(id?.provenance)} compact={false} />
          {#if r?.baseline_identity}<Ids title="Reference identity" items={provenanceRows(r.baseline_identity)} compact={false} />{/if}
        {/if}
        <p class="xs muted">Separate evaluation of library input {j.input_ids.map((id) => modelLabel(models.find((m) => m.id === id))).join(' · ') || '—'}. It is not matched to any benchmark job's endpoint or model.</p>
      </div>
    </section>
  {/each}

  {#if columns.length >= 2}
    <section class="panel" aria-labelledby="cmp-h">
      <header><h2 id="cmp-h">Side by side</h2></header>
      <div class="body stack">
        <p class="note info">
          Rows align only when benchmark, version, protocol, metric and unit are identical. A shared row does not make the runs comparable: the jobs can use different endpoints, models or hosts. No combined score is computed.
        </p>
        <!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard scrolling of wide metrics) -->
        <div class="table-wrap" role="region" aria-label="Benchmark metrics" tabindex="0">
          <table class="data cmp">
            <caption class="sr-only">Metric values per selected benchmark job. Empty cells mean the job has no such metric.</caption>
            <thead>
              <tr>
                <th scope="col">Benchmark · version · protocol</th>
                <th scope="col">Metric</th>
                {#each columns as c (c.jobId)}
                  <th scope="col" class="r">
                    <span class="mono">{c.jobId}</span>
                    <span class="sub xs">{c.report.target} · {c.report.model}</span>
                    <span class="sub xs">{hostName(hosts, c.host)}</span>
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each rows as r (r.key)}
                <tr>
                  <th scope="row"><span class="mono">{r.benchmark}</span> <span class="xs muted mono">{r.version} · {r.protocol}</span></th>
                  <td>{#if r.metric}{r.metric}{#if r.unit}{' '}<span class="xs muted">({r.unit})</span>{/if}{' '}<span class="xs muted">{directionText(r.direction)}</span>{:else}<span class="xs muted">No metrics</span>{/if}</td>
                  {#each r.cells as v, i (i)}
                    <td class="num r">
                      {#if r.statuses[i] === undefined}<span class="muted" aria-label="Not run in this job">—</span>
                      {:else if r.statuses[i] !== 'succeeded'}<Badge tone={statusTone(r.statuses[i] ?? '')} label={r.statuses[i] ?? ''} />{#if v !== undefined}{' '}{formatValue(v)}{/if}
                      {:else if v === undefined}<span class="muted" aria-label="Not reported">—</span>
                      {:else}{formatValue(v)}{/if}
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  {:else if chosen.length >= 2}
    <p class="xs muted">Side by side needs at least two selected jobs with a v1 report.</p>
  {/if}

  <RecentJobs kinds={['benchmark']} title="Recent benchmark jobs" limit={6} />
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  fieldset {
    border: none;
    margin: 0;
    padding: 0;
    min-width: 0;
  }
  legend {
    padding: 0;
    margin-bottom: var(--s-2);
    font-weight: 500;
  }
  .sfilter {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--s-2) var(--s-3);
    margin-bottom: var(--s-2);
  }
  .sfilter .input {
    flex: 0 1 260px;
    min-height: 32px;
  }
  .checks {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 2px var(--s-4);
  }
  .check {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    min-height: 32px;
    padding: 6px 0;
    cursor: pointer;
    font-size: var(--fs-sm);
  }
  .check input {
    margin: 2px 0 0;
    flex: none;
  }
  .check.off {
    color: var(--text-2);
    cursor: not-allowed;
  }
  .joblist {
    list-style: none;
    margin: 0;
    padding: 0;
    max-height: 280px;
    overflow-y: auto;
    border: 1px solid var(--line);
    border-radius: var(--r-sm);
  }
  .joblist li {
    padding: 0 var(--s-3);
    border-bottom: 1px solid var(--line);
  }
  .joblist li:last-child {
    border-bottom: none;
  }
  .jl {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  .pick {
    align-items: start;
  }
  .jh {
    display: flex;
    flex-wrap: wrap;
    gap: 0 8px;
    align-items: baseline;
  }
  .jh a {
    font-size: var(--fs-sm);
    font-weight: 500;
  }
  .r {
    text-align: right;
  }
  .prov {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 2px var(--s-3);
    margin: 0;
  }
  .prov dt {
    color: var(--text-2);
  }
  .prov dd {
    margin: 0;
  }
  .why {
    margin: 0;
    padding-left: 1.1em;
    color: var(--text-2);
  }
  .cmp th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
  }
  .cmp thead th {
    vertical-align: bottom;
  }
  .sub {
    display: block;
    font-weight: 400;
  }
</style>

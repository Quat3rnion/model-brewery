<script lang="ts">
  import Badge from '../Badge.svelte';
  import KeyValues from '../KeyValues.svelte';
  import TraceCompare from './TraceCompare.svelte';
  import RegisterCopy from '../RegisterCopy.svelte';
  import BenchmarkReport from './BenchmarkReport.svelte';
  import ComponentEvidence from '../ComponentEvidence.svelte';
  import WeightReport from '../weights/WeightReport.svelte';
  import ConversionSummary from './ConversionSummary.svelte';
  import SensitivityReport from './SensitivityReport.svelte';
  import { selectionFrom } from '../../lib/components';
  import { missingSuites, readBenchmarkReport, requestedOf } from '../../lib/benchmark';
  import { formatBytes, shortHash } from '../../lib/format';
  import { referenceInfo, accuracyText, isFullSplit, isSynthetic, loadProofText, obj, readScore, fileVerificationStatus, explicitVerdict, NOT_RELEASE_QUALIFIED } from '../../lib/reports';
  import type { JobDetail } from '../../lib/types';

  let { job }: { job: JobDetail } = $props();
  const result = $derived(obj(job.result));
  const artifact = $derived(obj(result?.artifact));
  const kind = $derived(String(job.kind));
  const conversion = $derived(kind === 'convert' || result?.kind === 'hy4_plan');
  const bench = $derived(readBenchmarkReport(job.result));
  const requestedRoles = $derived(selectionFrom(job.input_ids, job.options));
  const benchShown = ['kind', 'schema', 'target', 'model', 'status', 'runs', 'release_qualified'];
  const conversionShown = ['runtime_ready', 'payload_verified', 'file_bytes', 'base_file_bytes', 'resident_bytes', 'base_resident_bytes', 'resident_delta_bytes', 'source_metadata_read_bytes', 'reflink_bytes', 'engram', 'release_qualified'];
  const shown = $derived([...(bench ? benchShown : []), ...(conversion ? conversionShown : []), 'artifact', 'traces', 'alignment', 'report', 'text', 'score', 'comparison', 'disk_preflight', 'file_verification', 'hardware_proof', 'load_proof', 'identity', 'kind', ...(result?.score ? ['baseline_score','baseline_identity','reference_scope'] : []), 'dataset', 'protocol_id', 'inputs', 'bytes', 'missing_bytes', 'available_bytes', 'removed', 'skipped', 'components', ...(kind.startsWith('weight_') ? ['proposal'] : [])]);
</script>

{#if !result}
  {#if job.result !== undefined && job.result !== null}
    <pre class="block">{JSON.stringify(job.result, null, 2)}</pre>
  {:else}
    <p class="muted small">No result recorded.</p>
  {/if}
{:else if kind === 'sensitivity' && result.schema_version === 1}
  <SensitivityReport {result} />
{:else}
  <div class="stack">
    {#if conversion}<ConversionSummary {job} {result} />{/if}
    {#if ['inspect', 'plan', 'convert', 'verify'].includes(kind) && (result.components || Object.values(requestedRoles).some(Boolean))}
      <ComponentEvidence requested={requestedRoles} result={job.result} />
    {/if}

    {#if bench}
      <section>
        <h3 class="h">Benchmark report</h3>
        <BenchmarkReport report={bench} missing={missingSuites(job, bench)} requestedTarget={requestedOf(job).benchmark_target} />
      </section>
    {/if}

    {#if kind.startsWith('weight_')}
      <WeightReport {job} />
    {/if}

    {#if kind === 'compare' || result.kind === 'trace-comparison'}
      <TraceCompare {result} />
    {/if}

    {#if kind === 'transfer_preview' || kind === 'transfer'}
      {@const inputs = Array.isArray(result.inputs) ? (result.inputs as unknown[]).map((x) => obj(x) ?? {}) : []}
      {@const missing = typeof result.missing_bytes === 'number' ? result.missing_bytes : null}
      {@const avail = typeof result.available_bytes === 'number' ? result.available_bytes : null}
      <section class="stack">
        <dl class="art small num">
          <dt>Total size</dt><dd>{formatBytes(result.bytes)}</dd>
          <dt>Missing on destination</dt><dd>{formatBytes(missing)}</dd>
          <dt>Free on destination</dt><dd>{formatBytes(avail)}</dd>
        </dl>
        {#if missing !== null && avail !== null}
          <div class="row">
            <Badge tone={missing <= avail ? 'ok' : 'bad'} label={missing <= avail ? 'Missing bytes fit in free space' : 'Not enough free space'} />
          </div>
          <p class="xs muted">Free space is measured, not reserved. Other writes can use it before the transfer runs.</p>
        {/if}
        {#if inputs.length}
          <!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users must be able to scroll this table.) -->
          <div class="table-wrap" role="region" aria-label="Transfer inputs scroll area" tabindex="0">
            <table class="data">
              <caption class="sr-only">Inputs of this transfer</caption>
              <thead><tr><th scope="col">{kind === 'transfer' && job.state === 'succeeded' ? 'Copy' : 'Source'}</th><th scope="col">Kind</th><th scope="col">Size</th><th scope="col">Bundle SHA-256</th><th scope="col">On destination</th>{#if kind === 'transfer' && job.state === 'succeeded'}<th scope="col"><span class="sr-only">Library</span></th>{/if}</tr></thead>
              <tbody>
                {#each inputs as it, i (i)}
                  <tr>
                    <td class="mono xs break">
                      {#if kind === 'transfer' && job.state === 'succeeded'}{String(it.host ?? '')}:{String(it.path ?? '—')}<div class="muted">from {String(it.source_host ?? '?')}</div>
                      {:else}{String(it.source_host ?? '')}{it.source_host ? ':' : ''}{String(it.path ?? '—')}{/if}
                    </td>
                    <td class="xs">{String(it.kind ?? '—')}</td>
                    <td class="num xs">{formatBytes(it.bytes)}</td>
                    <td class="mono xs" title={String(it.bundle_sha256 ?? '')}>{shortHash(it.bundle_sha256)}</td>
                    <td>
                      {#if it.cached === true}<Badge tone="ok" label="Verified copy reused" />
                      {:else if it.cached === false}<Badge tone="muted" label={kind === 'transfer' ? 'Copied' : 'Needs copy'} />
                      {:else}<span class="xs muted">—</span>{/if}
                    </td>
                    {#if kind === 'transfer' && job.state === 'succeeded'}
                      <td>
                        {#if typeof it.host === 'string' && typeof it.path === 'string' && typeof it.kind === 'string'}
                          <RegisterCopy host={it.host} path={it.path} kind={it.kind} />
                        {/if}
                      </td>
                    {/if}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </section>
    {/if}

    {#if kind === 'cache_cleanup'}
      {@const removed = Array.isArray(result.removed) ? (result.removed as unknown[]) : []}
      {@const skipped = Array.isArray(result.skipped) ? (result.skipped as unknown[]) : []}
      <section class="stack">
        <p class="small num">Removed {removed.length} cached bundle{removed.length === 1 ? '' : 's'}; skipped {skipped.length}. Originals are never removed by cleanup.</p>
        {#if removed.length}<details><summary class="xs">Removed digests</summary><pre class="block">{removed.join('\n')}</pre></details>{/if}
        {#if skipped.length}<details><summary class="xs">Skipped digests</summary><pre class="block">{skipped.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join('\n')}</pre></details>{/if}
      </section>
    {/if}

    {#if result.disk_preflight}
      {@const d = obj(result.disk_preflight)}
      <section>
        <h3 class="h">Disk preflight</h3>
        {#if d}
          <div class="row" style="margin-bottom:6px">
            {#if typeof d.fits === 'boolean'}<Badge tone={d.fits ? 'ok' : 'bad'} label={d.fits ? 'Fits' : 'Does not fit'} />{/if}
          </div>
          <p class="small num">
            Needs at least {formatBytes(d.required_bytes)}; {formatBytes(d.available_bytes)} available. This is the minimum planned HGN payload, not a space reservation.
          </p>
        {/if}
      </section>
    {/if}

    {#if kind === 'verify' || result.file_verification}
      {@const v = fileVerificationStatus(result)}
      <section>
        <h3 class="h">File verification</h3>
        <div class="row">
          <Badge tone={v === 'passed' ? 'ok' : v === 'failed' ? 'bad' : 'muted'} label={v === 'passed' ? 'Passed' : v === 'failed' ? 'Failed' : v === 'not_performed' ? 'Not performed' : 'No verdict'} />
        </div>
        {#if result.file_verification}<KeyValues data={result.file_verification} />{/if}
      </section>
    {/if}

    {#if result.hardware_proof}
      {@const hp = obj(result.hardware_proof)}
      <section>
        <h3 class="h">Hardware proof</h3>
        <div class="row">
          <Badge tone={hp?.status === 'proved' ? 'ok' : 'warn'} label={hp?.status === 'proved' ? 'Proved' : String(hp?.status ?? 'unknown').replace(/_/g, ' ')} />
        </div>
        <p class="xs muted" style="margin-top:4px">Hardware proof is separate from file verification. A file pass does not prove GPU execution.</p>
        <KeyValues data={hp} skip={['status']} />
      </section>
    {/if}

    {#if result.report !== undefined && kind.startsWith('weight_')}
      <details class="disclosure"><summary class="xs">Raw report</summary><pre class="block">{JSON.stringify(result.report, null, 2)}</pre></details>
    {:else if result.report !== undefined && kind !== 'compare'}
      <section>
        <h3 class="h">{kind === 'inspect' ? 'Inspection report' : kind === 'plan' ? 'Plan report' : kind === 'verify' ? 'Verify report' : 'Report'}</h3>
        {#if kind !== 'verify' && explicitVerdict(result.report) !== null}
          <div class="row"><Badge tone={explicitVerdict(result.report) ? 'ok' : 'bad'} label={explicitVerdict(result.report) ? 'Passed' : 'Failed'} /></div>
        {/if}
        <KeyValues data={result.report} />
      </section>
    {/if}

    {#if kind === 'generate' || typeof result.text === 'string'}
      <section>
        <h3 class="h">Generated text</h3>
        {#if typeof result.text === 'string'}
          <p class="gen">{result.text}</p>
        {:else}
          <p class="muted small">No text in the result.</p>
        {/if}
        <p class="xs muted" style="margin-top:6px">{loadProofText(result.load_proof)} Generated text is not a quality result.</p>
        {#if result.load_proof}<details><summary class="xs">Load proof record</summary><KeyValues data={result.load_proof} /></details>{/if}
      </section>
    {/if}

    {#if result.score}
      {@const score = readScore(result)}
      {@const full = isFullSplit(result)}
      {@const cmp = obj(result.comparison)}
      {@const reference = referenceInfo(result)}
      <section class="stack">
        <h3 class="h">GSM8K score</h3>
        {#if score}<p class="num">{accuracyText(score)}{typeof score.unparsed === 'number' ? ` · ${score.unparsed} unparsed` : ''}</p>{/if}
        <div class="row">
          <Badge tone="muted" label={NOT_RELEASE_QUALIFIED} />
          {#if full === true}
            <Badge tone="muted" label="Full pinned split{typeof result.protocol_id === 'string' ? ` · ${result.protocol_id}` : ''}" />
          {:else if isSynthetic(result)}
            <Badge tone="warn" label="Synthetic fixture" />
          {:else if full === false}
            <Badge tone="warn" label="Not the full split" />
          {:else}
            <Badge tone="warn" label="Split not confirmed" />
          {/if}
          {#if kind === 'evaluate' && !reference}<Badge tone="muted" label="Standalone evaluation" />{/if}
        </div>
        {#if reference}
          <h4>Reference comparison</h4>
          <p class="small">{reference.scope === 'attested_bf16_same_source' ? 'Attested BF16 reference from the same source.' : 'Diagnostic reference.'} Precision: {reference.precision ?? 'not reported'}.</p>
          {@const rs = readScore({score:reference.score})}
          {#if rs}<p class="small num">Reference: {accuracyText(rs)}</p>{/if}
          {#if reference.identity}<details><summary class="small">Reference identity and provenance</summary><KeyValues data={reference.identity} /></details>{/if}
        {/if}
        {#if cmp}
          <KeyValues data={cmp} skip={['changed_answers', 'within_two_percentage_points']} />
          {#if Array.isArray(cmp.changed_answers)}
            <details>
              <summary class="small">{cmp.changed_answers.length} changed answers</summary>
              <pre class="block">{JSON.stringify(cmp.changed_answers, null, 2)}</pre>
            </details>
          {/if}
        {/if}
      </section>
    {/if}

    {#if result.dataset || result.protocol_id}
      <section>
        <h3 class="h">Protocol and dataset</h3>
        <KeyValues data={{ protocol_id: result.protocol_id, ...(obj(result.dataset) ?? {}) }} />
      </section>
    {/if}

    {#if result.identity}
      <section>
        <h3 class="h">Identity</h3>
        <KeyValues data={result.identity} />
      </section>
    {/if}

    {#if artifact}
      <section>
        <h3 class="h">Artifact</h3>
        <dl class="art small">
          <dt>Kind</dt><dd>{String(artifact.kind ?? '—')}</dd>
          <dt>Path</dt><dd class="mono break">{String(artifact.path ?? '—')}</dd>
          <dt>SHA-256</dt><dd class="mono break" title={String(artifact.sha256 ?? '')}>{shortHash(artifact.sha256)}</dd>
          {#if !conversion}<dt>Size</dt><dd class="num">{formatBytes(artifact.bytes)}</dd>{/if}
        </dl>
        {#if Array.isArray(artifact.files)}
          <details><summary class="xs">{artifact.files.length} files</summary><pre class="block">{JSON.stringify(artifact.files, null, 2)}</pre></details>
        {/if}
      </section>
    {/if}

    {#if Object.keys(result).some((k) => !shown.includes(k))}
      <section>
        <h3 class="h">Other fields</h3>
        <KeyValues data={result} skip={shown} />
      </section>
    {/if}

    <details>
      <summary class="xs muted">Raw result JSON</summary>
      <pre class="block">{JSON.stringify(job.result, null, 2)}</pre>
    </details>
  </div>
{/if}

<style>
  .h {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
    margin-bottom: 6px;
  }
  .gen {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    padding: var(--s-3);
    background: var(--bg);
    border: 1px solid var(--line);
    border-radius: var(--r-sm);
  }
  .art {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 4px var(--s-3);
    margin: 0;
  }
  .art dt {
    color: var(--text-2);
  }
  .art dd {
    margin: 0;
  }
</style>

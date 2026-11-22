<script lang="ts">
  import Badge from '../Badge.svelte';
  import { api, errorText } from '../../lib/api';
  import { basename, formatBytes, MODEL_KINDS } from '../../lib/format';
  import { fileVerificationStatus, NOT_RELEASE_QUALIFIED, obj } from '../../lib/reports';
  import { brewery } from '../../lib/store.svelte';
  import type { JobDetail, Model, NewModel } from '../../lib/types';

  let { job, result }: { job: JobDetail; result: Record<string, unknown> } = $props();
  const plan = $derived(result.kind === 'hy4_plan');
  const artifact = $derived(obj(result.artifact));
  const registered = $derived((brewery.state?.models ?? []).find((m) =>
    m.job_id === job.id || (m.host === job.host && m.path === artifact?.path && m.kind === artifact?.kind)));
  let added = $state<Model>();
  const entry = $derived(registered ?? (added?.host === job.host && added?.path === artifact?.path ? added : undefined));
  const registration = $derived(job.host ? brewery.state?.eligibility?.hosts[job.host]?.register_manual : undefined);
  const candidate = $derived.by((): NewModel | null => {
    if (!job.host || job.state !== 'succeeded' || plan || !artifact ||
        typeof artifact.path !== 'string' || !artifact.path.startsWith('/') || typeof artifact.kind !== 'string' ||
        !MODEL_KINDS.some((kind) => kind.value === artifact.kind)) return null;
    const verified = fileVerificationStatus(result) === 'passed' || (result.kind === 'hy4_gate_encode' && result.verified === true);
    return verified ? { host: job.host, path: artifact.path, kind: artifact.kind, name: basename(artifact.path) } : null;
  });

  const sizes = $derived([
    [plan ? 'Planned output size' : 'Output size', plan ? result.file_bytes : (artifact?.bytes ?? result.file_bytes)],
    ['Base file size', result.base_file_bytes],
    ['Resident data', result.resident_bytes],
    ['Base resident data', result.base_resident_bytes],
    ['Change in resident data', result.resident_delta_bytes],
    ['Engram on disk', obj(result.engram)?.disk_bytes],
    ['Source metadata read', result.source_metadata_read_bytes],
    ['Bytes shared with the source', result.reflink_bytes],
  ] as const);

  let busy = $state(false);
  let error = $state('');
  async function register() {
    if (!candidate || entry || busy || registration?.eligible === false) return;
    const model = candidate;
    busy = true;
    error = '';
    try {
      added = await api.addModel(model);
      await brewery.refresh();
    } catch (e) {
      error = errorText(e);
    } finally {
      busy = false;
    }
  }
</script>

<section class="stack">
  <h3>{plan ? 'Hy4 metadata plan' : 'Conversion summary'}</h3>
  {#if plan}<p class="small">The plan describes an output. It does not write model weights.</p>{/if}
  {#if sizes.some(([, bytes]) => typeof bytes === 'number')}
    <dl class="sizes small">
      {#each sizes as [label, bytes] (label)}
        {#if typeof bytes === 'number' && Number.isFinite(bytes)}
          <dt>{label}</dt><dd class="num">{bytes < 0 ? '-' : ''}{formatBytes(Math.abs(bytes))}</dd>
        {/if}
      {/each}
    </dl>
  {/if}
  <div class="row"><Badge tone="muted" label={NOT_RELEASE_QUALIFIED} /></div>
  {#if typeof result.payload_verified === 'boolean'}
    <p class="small">{result.payload_verified ? 'The job verified the tensor payload.' : 'The job did not verify the tensor payload.'}</p>
  {/if}
  {#if typeof result.runtime_ready === 'boolean'}
    <p class="small">{result.runtime_ready ? 'The worker reports that the output can run. This does not show model quality.' : 'The result does not show that the model can run.'}</p>
  {/if}
  {#if job.state === 'succeeded' && !plan}
    {#if entry}
      <div class="row"><a class="btn small" href="#/library?entry={encodeURIComponent(entry.id)}">View in Library</a></div>
    {:else if candidate && registration?.eligible !== false}
      <div class="row"><button class="btn small" type="button" disabled={busy} onclick={register}>{busy ? 'Registering…' : 'Register in Library'}</button></div>
      <p class="xs muted">Registration adds a Library entry. It does not show that the model can run or is release-qualified.</p>
    {:else if candidate && registration?.eligible === false}
      {#each registration.reasons as reason}<p class="xs"><a href={reason.fix_anchor}>{reason.text}</a></p>{/each}
    {:else if artifact}
      <p class="xs muted">The Library has no entry for this output. Verify it before you register it.</p>
    {/if}
    {#if error}<p class="note bad" role="alert">{error}</p>{/if}
  {/if}
</section>

<style>
  h3 { font-size: var(--fs-sm); }
  .sizes {
    display: grid;
    grid-template-columns: minmax(0, max-content) minmax(0, 1fr);
    gap: 4px var(--s-3);
    margin: 0;
  }
  dt { color: var(--text-2); }
  dd { margin: 0; }
</style>

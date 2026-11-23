<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import type { RetrySubmission } from '../lib/retry-prefill.svelte';
  import JobLink from './JobLink.svelte';
  import ModelSelect from './ModelSelect.svelte';
  import Submit from './Submit.svelte';

  let { sourceId = $bindable(''), encode, restored, retry, loading }: {
    sourceId?: string; encode: boolean; restored: boolean; retry?: RetrySubmission; loading: boolean;
  } = $props();
  let open = $state(false);
  const action = $derived(encode ? 'conversion' : 'plan');
  const capability = $derived(encode ? 'hy4_encode' : 'hy4_plan');
  const tensor = $derived(encode ? 'blk.1.attn_gate.weight' : 'blk.1.ffn_down_exps.weight');
  $effect(() => {
    if (restored) open = true;
  });
  const source = $derived(brewery.state?.models.find(m => m.id === sourceId));
  const host = $derived(brewery.state?.hosts.find(h => h.id === source?.host));
  const problem = $derived(!source || source.kind !== 'hf' ? 'Select the registered HF directory for the pinned revision.'
    : source.available === false ? 'The source is unavailable. Restore it or select another registered copy.'
    : host?.scheduling_paused ? 'Resume job scheduling on the source host.'
    : host?.capabilities?.[encode ? 'convert' : 'plan'] !== true ? `Enable ${encode ? 'Convert' : 'Plan'} on the source host.`
    : host?.capabilities?.[capability] !== true ? `Select a copy on a worker that supports this Hy4 ${action}. ${host?.reasons?.[capability] ?? ''}` : '');
  const submission = new Submitter(job => job.kind === 'convert' ? 'convert.hy4.encode' : 'convert.hy4.plan');

  async function queue() {
    if (loading || retry?.blocked || submission.busy || problem || !source) return;
    await submission.submit({ kind: encode ? 'convert' : 'plan', host: source.host, input_ids: [source.id],
      options: { hy4_overrides: { [tensor]: 'Q8_0' } } }, retry);
  }
</script>

<details class="disclosure" bind:open>
  <summary><strong>{encode ? 'Hy4 gate conversion' : 'Hy4 metadata plan'}</strong> <span class="xs muted">Text-only GGUF layout</span></summary>
  <div class="body stack">
    <p class="small">Target: <strong>tencent/Hy4-flash</strong>, revision <code>b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f</code>. The worker checks the pinned metadata, not the Library name.</p>
    <ModelSelect id="hy4-source" bind:value={sourceId} kinds={['hf']} label="Hy4 source" hint="Choose the registered HF directory on a worker that supports this job." />
    {#if source?.kind === 'hf'}<p class="xs muted">Source: <code>{source.path}</code> on <strong>{source.host}</strong>.</p>{/if}
    <p class="small">Fixed override: <code>{tensor}</code> to <strong>Q8_0</strong>.</p>
    {#if encode}
      <p class="xs muted">Writes a new GGUF file. This job does not show that the model can run or measure its quality.</p>
    {:else}
      <p class="small">Expected file size: <strong>224,927,386,624 bytes</strong>, including the <abbr title="Tables that store values for sequences of tokens">Engram</abbr> tables. The region that stays in memory grows by 603,979,776 bytes in the plan. These are layout sizes, not measured memory use.</p>
      <p class="xs muted">Reads metadata only. Does not read or verify weight payloads, write a GGUF file, or test runtime support or quality.</p>
    {/if}
    <Submit blockers={problem ? [problem] : []} busy={submission.busy} disabled={loading || retry?.blocked} onclick={queue}>Queue Hy4 {action}</Submit>
    <div aria-live="polite">
      {#if submission.error}
        <p class="note bad" role="alert">{submission.error}</p>
      {:else if submission.job}
        <p class="muted">{submission.replayed ? 'This request was already queued as job' : 'Queued as job'}
          <JobLink id={submission.job.id} label={'View job result: ' + submission.job.id} />.</p>
      {/if}
    </div>
  </div>
</details>

<style>
  .body { overflow-wrap: anywhere; }
</style>

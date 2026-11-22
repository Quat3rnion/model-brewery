<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { newKey } from '../lib/conversion';
  import { api, errorText } from '../lib/api';
  import type { JobDetail, NewJob } from '../lib/types';
  import JobLink from './JobLink.svelte';
  import ModelSelect from './ModelSelect.svelte';
  import Submit from './Submit.svelte';

  let { retry = null }: { retry?: JobDetail | null } = $props();
  let sourceId = $state('');
  let open = $state(false);
  const encode = $derived(retry?.kind === 'convert');
  const action = $derived(encode ? 'conversion' : 'plan');
  const capability = $derived(encode ? 'hy4_encode' : 'hy4_plan');
  const tensor = $derived(encode ? 'blk.1.attn_gate.weight' : 'blk.1.ffn_down_exps.weight');
  $effect(() => {
    if (!retry) return;
    sourceId = retry.input_ids[0] ?? '';
    open = true;
  });
  const source = $derived(brewery.state?.models.find(m => m.id === sourceId));
  const host = $derived(brewery.state?.hosts.find(h => h.id === source?.host));
  const problem = $derived(!source || source.kind !== 'hf' ? 'Select the registered HF directory for the pinned revision.'
    : source.available === false ? 'The source is unavailable. Restore it or select another registered copy.'
    : host?.capabilities?.[capability] !== true ? `Select a copy on a worker that supports this Hy4 ${action}. ${host?.reasons?.[capability] ?? ''}` : '');
  let busy = $state(false);
  let outcome = $state<{ source: string; error?: string; id?: string } | null>(null);
  let lastBody = '', key = '';

  async function queue() {
    if (busy || problem || !source) return;
    const id = source.id;
    const body: NewJob = { kind: encode ? 'convert' : 'plan', host: source.host, input_ids: [id], options: { hy4_overrides: { [tensor]: 'Q8_0' } } };
    const json = JSON.stringify(body);
    if (lastBody !== json) { key = newKey(); lastBody = json; }
    busy = true;
    outcome = null;
    try {
      const job = await api.submitJob(body, key);
      if (typeof job?.id !== 'string' || !job.id) throw new Error('The controller did not return a job ID.');
      outcome = { source: id, id: job.id };
      void brewery.refresh();
    } catch (e) {
      outcome = { source: id, error: errorText(e) };
    } finally {
      busy = false;
    }
  }
</script>

<details class="disclosure" bind:open>
  <summary><strong>{encode ? 'Hy4 gate conversion' : 'Hy4 metadata plan'}</strong> <span class="xs muted">Text-only GGUF layout</span></summary>
  <div class="body stack">
    <p class="small">Target: <strong>tencent/Hy4-flash</strong>, revision <code>b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f</code>. The worker checks the pinned metadata, not the Library name.</p>
    <ModelSelect bind:value={sourceId} kinds={['hf']} label="Hy4 source" hint="Choose the registered HF directory on a worker that supports this job." />
    {#if source?.kind === 'hf'}<p class="xs muted">Source: <code>{source.path}</code> on <strong>{source.host}</strong>.</p>{/if}
    <p class="small">Fixed override: <code>{tensor}</code> to <strong>Q8_0</strong>.</p>
    {#if encode}
      <p class="xs muted">Writes a new GGUF file. This job does not show that the model can run or measure its quality.</p>
    {:else}
      <p class="small">Expected file size: <strong>224,927,386,624 bytes</strong>, including the <abbr title="Tables that store values for sequences of tokens">Engram</abbr> tables. The region that stays in memory grows by 603,979,776 bytes in the plan. These are layout sizes, not measured memory use.</p>
      <p class="xs muted">Reads metadata only. Does not read or verify weight payloads, write a GGUF file, or test runtime support or quality.</p>
    {/if}
    <Submit blockers={problem ? [problem] : []} {busy} onclick={queue}>Queue Hy4 {action}</Submit>
    <div aria-live="polite">
      {#if outcome?.source === sourceId && outcome.error}
        <p class="note bad" role="alert">{outcome.error} Retry with the same source to reuse this request.</p>
      {:else if outcome?.source === sourceId && outcome.id}
        <JobLink id={outcome.id} label={'View job result: ' + outcome.id} />
      {/if}
    </div>
  </div>
</details>

<style>
  .body { overflow-wrap: anywhere; }
</style>

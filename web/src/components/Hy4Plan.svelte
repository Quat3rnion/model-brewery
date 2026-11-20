<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { newKey } from '../lib/conversion';
  import { errorText } from '../lib/api';
  import JobLink from './JobLink.svelte';
  import ModelSelect from './ModelSelect.svelte';
  import Submit from './Submit.svelte';

  let { sourceId = $bindable('') }: { sourceId?: string } = $props();
  const source = $derived(brewery.state?.models.find(m => m.id === sourceId));
  const host = $derived(brewery.state?.hosts.find(h => h.id === source?.host));
  const problem = $derived(!source || source.kind !== 'hf' ? 'Select the registered HF directory for the pinned revision.'
    : source.available === false ? 'The source is unavailable. Restore it or select another registered copy.'
    : host?.capabilities?.hy4_plan !== true ? `Select a copy on a worker that advertises hy4_plan. ${host?.reasons?.hy4_plan ?? ''}` : '');
  const tensor = 'blk.1.ffn_down_exps.weight';
  let busy = $state(false);
  let outcome = $state<{ source: string; error?: string; id?: string } | null>(null);
  let lastBody = '', key = '';

  async function queue() {
    if (busy || problem || !source) return;
    const id = source.id;
    const body = JSON.stringify({ kind: 'plan', host: source.host, input_ids: [id], options: { hy4_overrides: { [tensor]: 'Q8_0' } } });
    if (lastBody !== body) { key = newKey(); lastBody = body; }
    busy = true;
    outcome = null;
    try {
      const response = await fetch('/api/jobs', { method: 'POST', body,
        headers: { 'Content-Type': 'application/json', 'Idempotency-Key': key } });
      const job = await response.json();
      if (!response.ok || typeof job?.id !== 'string' || !job.id) throw new Error(job?.error ?? `Cannot confirm plan submission (HTTP ${response.status}).`);
      outcome = { source: id, id: job.id };
      void brewery.refresh();
    } catch (e) {
      outcome = { source: id, error: errorText(e) };
    } finally {
      busy = false;
    }
  }
</script>

<details class="disclosure">
  <summary><strong>Hy4 metadata plan</strong> <span class="xs muted">Text-only GGUF layout</span></summary>
  <div class="body stack">
    <p class="small">Target: <strong>tencent/Hy4-flash</strong>, revision <code>b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f</code>. The worker checks the pinned metadata, not the Library name.</p>
    <ModelSelect bind:value={sourceId} kinds={['hf']} label="Hy4 source" hint="Choose the registered HF directory on its planner worker." />
    {#if source?.kind === 'hf'}<p class="xs muted">Source: <code>{source.path}</code> on <strong>{source.host}</strong>.</p>{/if}
    <p class="small">Fixed override: <code>{tensor}</code> from Q4_K to <strong>Q8_0</strong>. Other routed experts stay Q4_K.</p>
    <p class="small">Expected file size: <strong>224,927,386,624 bytes</strong>, including the Engram tables. The planned resident region grows by 603,979,776 bytes. These are layout sizes, not measured memory use.</p>
    <p class="xs muted">Reads metadata only. Does not read or verify weight payloads, write a GGUF file, or test runtime support or quality.</p>
    <Submit blockers={problem ? [problem] : []} {busy} onclick={queue}>Queue Hy4 plan</Submit>
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

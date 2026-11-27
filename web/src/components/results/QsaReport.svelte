<script lang="ts">
  import { obj } from '../../lib/reports';
  import KeyValues from '../KeyValues.svelte';
  let { result }: { result: Record<string,unknown> } = $props();
  const metrics = $derived(obj(result.metrics));
  const teacher = $derived(obj(result.teacher));
  const id = $derived(result.library_model_id);
</script>

<section class="stack">
  <h3>QSA training result</h3>
  {#if typeof id === 'string'}
    <div class="row"><a href="#/library?entry={encodeURIComponent(id)}">View indexer in Library</a>
      {#if typeof teacher?.id === 'string'}
        <a class="btn small" href="#/convert?model={encodeURIComponent(teacher.id)}&indexer={encodeURIComponent(id)}">Select trained indexer in checkpoint components</a>
      {/if}
    </div>
  {/if}
  <div class="grid-2">
    {#each ['before','after'] as stage}
      {@const row = obj(metrics?.[stage])}
      <section class="stack">
        <h4>{stage === 'before' ? 'Before training' : 'After training'}</h4>
        <p class="small break">Held-out KL: {row?.kl ?? 'Not recorded'}<br>
          Capture: {row?.capture ?? row?.capture_unavailable_reason ?? 'Not recorded'}<br>
          Scored queries: {row?.queries ?? 'Not recorded'} · Capture queries: {row?.capture_queries ?? 'Not recorded'}</p>
      </section>
    {/each}
  </div>
  {#each [['Teacher',teacher],['Indexer file',result.artifact],['Metrics file',result.metrics_file]] as [label,data]}
    <section><h4>{label}</h4><KeyValues {data} /></section>
  {/each}
  <details><summary>Training details</summary><KeyValues data={metrics} skip={['before','after']} /></details>
</section>

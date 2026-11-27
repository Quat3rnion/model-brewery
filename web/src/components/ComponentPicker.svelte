<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { modelLabel, hostName } from '../lib/format';
  import { COMPONENT_ROLES, defaultText, type RoleSelection } from '../lib/components';
  import { resolveSourceComponents, sourceComponentJobs } from '../lib/sourceComponents';
  import type { Model } from '../lib/types';

  let { value = $bindable(), source, idPrefix, gguf = false, linkedIndexer }: {
    value?:RoleSelection; source:Model; idPrefix:string; gguf?:boolean; linkedIndexer?:string | null;
  } = $props();
  const models = $derived(brewery.state?.models ?? []);
  // Reference selection only: never add this ID to RoleSelection or conversion inputs.
  const indexers = $derived(models.filter(m => {
    const p = m.qsa_provenance;
    return m.kind === 'qsa-indexer' && m.host === source.host && m.available !== false &&
      p?.teacher_model_id === source.id && !!p.training_job_id && p.training_job_id === m.job_id &&
      /^[a-f0-9]{64}$/.test(p.teacher_sha256) && (!source.sha256 || p.teacher_sha256 === source.sha256);
  }));
  let selection = $state<{ teacher: string; id: string } | null>(null);
  const indexer = $derived(indexers.find(m => m.id === (selection?.teacher === source.id ? selection.id : linkedIndexer)));
  const candidates = $derived(sourceComponentJobs(source, brewery.state?.jobs ?? [], models));
  const evidence = $derived(resolveSourceComponents(source,
    candidates.flatMap(j=>brewery.details[j.id] ? [brewery.details[j.id]] : []), models));
  const attempted = new Set<string>();
  let failures = $state<string[]>([]);
  let retryCount = $state(0);
  let active=0;
  $effect(()=>{
    void retryCount;
    for (const j of candidates) {
      if (active>=2) break;
      if (brewery.details[j.id] || attempted.has(j.id)) continue;
      attempted.add(j.id);
      active++;
      void brewery.loadJob(j.id).catch(()=>{failures.push(j.id);}).finally(()=>{active--; retryCount++;});
    }
  });
  const failed = $derived(candidates.some(j=>failures.includes(j.id)));
  const loading = $derived(candidates.some(j=>!brewery.details[j.id] && !failures.includes(j.id)));
  function retry() {
    for (const id of failures) attempted.delete(id);
    failures=[];
    retryCount++;
  }
</script>

<section class="stack" aria-labelledby="{idPrefix}-title">
  <h3 id="{idPrefix}-title">Checkpoint components</h3>
  <p class="small muted">{gguf
    ? 'GGUF output takes no separate component inputs. The source evidence below does not guarantee that the exporter writes each component.'
    : value ? 'Use the checkpoint defaults, or select a registered replacement. The conversion job reports what it used.'
    : 'Inspect the source evidence and select a trained indexer for reference.'}</p>
  {#if failed}<div class="small" role="status">Some saved job details could not be loaded. <button type="button" class="btn small" onclick={retry}>Retry</button></div>{/if}
  <div class="roles">
    {#each COMPONENT_ROLES as r (r.kind)}
      {@const fact=evidence[r.kind]}
      {@const options=models.filter(m=>m.kind===r.kind && m.available!==false)}
      {@const label=source.kind==='gguf' && r.kind!=='ngram' ? 'Matching Library input required' : fact.status==='detected' ? `Use checkpoint ${r.label.toLowerCase()}` : fact.status==='none' ? 'No checkpoint component reported' : defaultText(source,r.kind)}
      <div class="field">
        {#if gguf}
          <strong class="small" title={r.kind === 'mtp' ? 'Multi-token prediction head' : undefined}>{r.label}</strong>
        {:else if value}
          <label for="{idPrefix}-{r.kind}" title={r.kind === 'mtp' ? 'Multi-token prediction head' : undefined}>{r.label}</label>
          <select id="{idPrefix}-{r.kind}" class="input" bind:value={value[r.kind]} aria-describedby="{idPrefix}-{r.kind}-hint">
            <option value="">{label}</option>
            {#if value[r.kind] && !options.some(m=>m.id===value?.[r.kind])}<option value={value[r.kind]}>Unavailable Library entry</option>{/if}
            {#if options.length}<optgroup label="Registered replacements">
              {#each options as m (m.id)}<option value={m.id}>{modelLabel(m)} · {hostName(brewery.state?.hosts ?? [],m.host)}</option>{/each}
            </optgroup>{/if}
          </select>
        {:else}
          <strong class="small">{r.label}</strong>
        {/if}
        <span class="hint" id="{idPrefix}-{r.kind}-hint">
          {loading && fact.status==='unknown' ? 'Reading saved job evidence…' : fact.summary}
          {#if fact.jobId}<a href="#/jobs/{encodeURIComponent(fact.jobId)}" aria-label="{r.label} evidence job">Job evidence</a>{/if}
        </span>
      </div>
    {/each}
    {#if source.kind === 'hf'}
      <div class="field">
        <label for="{idPrefix}-indexer">Trained QSA indexer</label>
        <select class="input" id="{idPrefix}-indexer" disabled={!indexers.length} value={indexer?.id ?? ''}
          onchange={e => selection = { teacher:source.id, id:e.currentTarget.value }} aria-describedby="{idPrefix}-indexer-hint">
          <option value="">{indexers.length ? 'Choose a trained indexer' : 'No matching trained indexer'}</option>
          {#each indexers as m (m.id)}<option value={m.id}>{modelLabel(m)}</option>{/each}
        </select>
        <span class="hint" id="{idPrefix}-indexer-hint">Reference only. HGN and GGUF conversion do not apply this component.</span>
        {#if indexer?.qsa_provenance}
          <span class="hint">Teacher SHA-256 at training: <span class="mono">{indexer.qsa_provenance.teacher_sha256}</span>.
            {#if !source.sha256}The current teacher identity has not been verified.{/if}
            <a href="#/jobs/{encodeURIComponent(indexer.qsa_provenance.training_job_id)}">Training job</a>
          </span>
        {/if}
      </div>
    {/if}
  </div>
</section>

<style>
  .roles { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:var(--s-3) var(--s-4); }
  .hint { overflow-wrap:anywhere; }
  @media(max-width:760px){.roles{grid-template-columns:minmax(0,1fr);}}
</style>

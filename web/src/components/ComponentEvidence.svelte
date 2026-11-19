<script lang="ts">
  // What each component role used, as reported by a job result
  // (result.components), next to what was requested. Never infers.
  import Badge from './Badge.svelte';
  import { brewery } from '../lib/store.svelte';
  import { modelLabel, shortHash } from '../lib/format';
  import { evidenceText, readComponents, roleRows, type RoleSelection } from '../lib/components';

  let { requested, result, title = 'Components' }: { requested: RoleSelection; result?: unknown; title?: string } = $props();
  const models = $derived(brewery.state?.models ?? []);
  const evidence = $derived(result === undefined ? null : readComponents(result));
  const rows = $derived(roleRows(requested, evidence));
  const any = $derived(rows.some((r) => r.requested || r.reported));
  const name = (id: string | null) => (id ? modelLabel(models.find((m) => m.id === id)) : '');
</script>

{#if any || result !== undefined}
  <div class="ce">
    <h4 class="h">{title}</h4>
    <div class="table-wrap">
      <table class="data stackable">
        <caption class="sr-only">Requested and reported source of each component</caption>
        <thead><tr><th scope="col">Component</th><th scope="col">Requested</th><th scope="col">Reported by the job</th></tr></thead>
        <tbody>
          {#each rows as r (r.kind)}
            <tr>
              <th scope="row">{r.label}</th>
              <td data-label="Requested" class="xs">{r.requested ? name(r.requested) : 'Default'}</td>
              <td data-label="Reported by the job" class="xs">
                {#if result === undefined}<span class="muted">No result yet</span>
                {:else if evidence === null}<span class="muted">Not reported by this job</span>
                {:else}
                  {evidenceText(r.evidence)}{#if r.evidence?.inputId}{': '}{name(r.evidence.inputId)}{/if}
                  {#if r.evidence?.sha256}<span class="mono muted" title={r.evidence.sha256}> {shortHash(r.evidence.sha256)}</span>{/if}
                  {#if r.match === false}<Badge tone="warn" label="Differs from request" />{/if}
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>
{/if}

<style>
  .h {
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-2);
    margin-bottom: 4px;
  }
  th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
    font-weight: 500;
  }
</style>

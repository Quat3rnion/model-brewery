<script lang="ts">
  import type { Snippet } from 'svelte';
  import { blockerText } from '../lib/blockers';
  import { brewery } from '../lib/store.svelte';
  import type { EligibilityEntry, Workflow } from '../lib/types';

  let { workflow, decision, children }: { workflow: Workflow; decision?: EligibilityEntry; children: Snippet } = $props();
  const entry = $derived(decision ?? brewery.state?.eligibility?.global[workflow]);
  const id = $props.id();
</script>

{#if entry?.eligible === false}
  <section class="note warn stack" aria-labelledby={id}>
    <h3 id={id}>Setup required</h3>
    <p>No host can run this workflow.</p>
    <ul>
      {#each entry.reasons as reason}
        <li><a href={reason.fix_anchor}>{blockerText(reason)}</a></li>
      {/each}
    </ul>
  </section>
{:else}
  {@render children()}
{/if}

<style>
  section { margin: var(--s-4); }
  ul { margin: 0; padding-inline-start: var(--s-5); }
  li a { display: inline-block; min-height: 24px; min-width: 24px; }
</style>

<script lang="ts">
  // Per-role component choice for a conversion: the checkpoint default, or an
  // explicit Library entry of that role. Collapsed by default; the summary
  // names any replacement so a choice is never hidden.
  import Advanced from './Advanced.svelte';
  import ModelSelect from './ModelSelect.svelte';
  import { COMPONENT_ROLES, defaultText, type RoleSelection } from '../lib/components';
  import type { Model } from '../lib/types';

  let { value = $bindable(), source, idPrefix }: { value: RoleSelection; source: Model | undefined; idPrefix: string } = $props();
  const changed = $derived(COMPONENT_ROLES.filter((r) => value[r.kind]).map((r) => `${r.label} replaced`));
</script>

<Advanced title="Components" changed={changed}>
  <p class="xs muted">
    Each component comes from the source unless you pick a Library entry for it. The page cannot see inside the checkpoint; the job result reports what was actually used.
  </p>
  <div class="roles">
    {#each COMPONENT_ROLES as r (r.kind)}
      <div class="role">
        <div class="def">
          <span class="rl">{r.label}</span>
          <span class="xs muted">Default: {defaultText(source, r.kind)}</span>
        </div>
        <ModelSelect bind:value={value[r.kind]} kinds={[r.kind]} label="Replacement for {r.label.toLowerCase()}" optional optionalLabel="Use default" hint={r.formats} id="{idPrefix}-{r.kind}" />
      </div>
    {/each}
  </div>
</Advanced>

<style>
  .roles {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--s-3) var(--s-4);
  }
  .role {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
    padding: var(--s-2) var(--s-3);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-md);
  }
  .def {
    display: flex;
    flex-direction: column;
  }
  .rl {
    font-weight: 600;
    font-size: var(--fs-sm);
  }
  @media (max-width: 760px) {
    .roles {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>

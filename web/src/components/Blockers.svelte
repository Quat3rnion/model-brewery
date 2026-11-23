<script lang="ts">
  import { blockerText, type Blocker } from '../lib/blockers';
  let { blockers, id }: { blockers: Blocker[]; id?: string } = $props();
</script>

<div aria-live="polite">
  {#if blockers.length}
    <ul class="note warn" {id} aria-label="Changes needed">
      {#each blockers as blocker}
        <li>
          {#if blocker.fixAnchor}
            <a href={blocker.fixAnchor}>{blockerText(blocker)}</a>
          {:else}
            {blockerText(blocker)}
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  ul.note {
    padding-inline-start: var(--s-6);
  }
</style>

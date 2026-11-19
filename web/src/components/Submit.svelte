<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    blockers = [],
    busy = false,
    onclick,
    variant = 'primary',
    children,
  }: { blockers?: string[]; busy?: boolean; onclick: () => void; variant?: 'primary' | 'secondary'; children: Snippet } = $props();

  const id = $props.id();
  const blocked = $derived(blockers.length > 0);
</script>

<div class="submit">
  <button
    type="button"
    class="btn {variant === 'primary' ? 'primary' : ''}"
    disabled={blocked || busy}
    aria-describedby={blocked ? id : undefined}
    aria-busy={busy}
    {onclick}
  >
    {#if busy}Submitting…{:else}{@render children()}{/if}
  </button>
  {#if blocked}
    <!-- One quiet line beside the button, not a list under every form. -->
    <p class="why xs" {id}><span class="lead">Needed:</span> {blockers.join(' ')}</p>
  {/if}
</div>

<style>
  .submit {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px var(--s-3);
  }
  .why {
    margin: 0;
    flex: 1 1 220px;
    max-width: 60ch;
    color: var(--text-2);
  }
  .lead {
    font-weight: 600;
  }
</style>

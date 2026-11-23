<script lang="ts">
  let { prepare, current, changed = false, disabled = false }: {
    prepare: () => boolean; current?: () => boolean; changed?: boolean; disabled?: boolean;
  } = $props();
  function open(prepare: () => boolean) {
    if (prepare()) document.querySelector<HTMLElement>('main h1')?.focus();
  }
</script>

<div class="row">
  <button type="button" class="btn" {disabled} onclick={() => {
    if (!changed || current || window.confirm('Replace the current form with the submitted inputs?')) open(prepare);
  }}>Start a new run</button>
  {#if changed && current}
    <button type="button" class="btn" {disabled} onclick={() => current && open(current)}>Start a new run with the current form</button>
  {/if}
</div>
<p class="hint">Start a new run opens the plain form with the submitted inputs, not the current form. Review the inputs and queue the new run separately.</p>

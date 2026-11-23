<script lang="ts">
  let { prepare, disabled = false }: { prepare: () => boolean; disabled?: boolean } = $props();
  function open() {
    if (prepare()) document.querySelector<HTMLElement>('main h1')?.focus();
  }
</script>

<button type="button" class="btn" {disabled} onclick={open}>Start a new run</button>
<p class="hint">Opens the plain form with these inputs. Review them and queue the new run separately.</p>

<script lang="ts">
  let { apply }: { apply: () => void } = $props();
</script>

<div class="row">
  <p class="small" role="status">Previous inputs are ready. You can use them to replace this draft.</p>
  <button type="button" class="btn" onclick={apply}>Use previous inputs</button>
</div>

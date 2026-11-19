<script lang="ts">
  let { id, label }: { id: string; label?: string } = $props();
</script>

<a class="mono" href="#/jobs/{encodeURIComponent(id)}">{label ?? id}</a>

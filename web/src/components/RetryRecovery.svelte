<script lang="ts">
  let { id, reason, reload, startNew }: { id?: string; reason: string; reload: () => void; startNew: () => void } = $props();
  function act(action: () => void) {
    action();
    document.querySelector<HTMLElement>('main h1')?.focus();
  }
</script>

<div class="note warn stack">
  <p {id} role="status">Cannot open the previous request. {reason} Queue stays blocked on this retry page.</p>
  <div class="row">
    <button type="button" class="btn" onclick={() => act(reload)}>Reload previous request</button>
    <button type="button" class="btn" onclick={() => act(startNew)}>Start a new run instead</button>
  </div>
</div>

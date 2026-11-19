<script lang="ts">
  // Enter the infrastructure admin key for this tab. Kept in memory only.
  import { adminSession } from '../lib/admin.svelte';
  import { isAdminKey } from '../lib/admin';

  let { purpose }: { purpose: string } = $props();
  const uid = $props.id();
  let value = $state('');
  let touched = $state(false);
  const ok = $derived(isAdminKey(value.trim()));
  function use(e: Event) {
    e.preventDefault();
    touched = true;
    if (!ok) return;
    adminSession.set(value.trim());
    value = '';
  }
</script>

<div class="ak">
  <p class="small">{purpose}</p>
  <div class="row">
    <div class="field grow">
      <label for="{uid}-k">Admin key</label>
      <input
        id="{uid}-k"
        class="input mono"
        type="password"
        bind:value
        autocomplete="off"
        spellcheck="false"
        aria-invalid={touched && !ok ? 'true' : undefined}
        aria-describedby="{uid}-h"
        onkeydown={(e) => e.key === 'Enter' && use(e)}
      />
      <span class="hint" id="{uid}-h">{touched && !ok ? '64 lowercase hexadecimal characters.' : 'Kept only in this tab’s memory. It is not saved, and a reload forgets it.'}</span>
    </div>
    <button type="button" class="btn" onclick={use} disabled={!value}>Use key</button>
  </div>
</div>

<style>
  .ak {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    padding: var(--s-3);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
  }
  .grow {
    flex: 1 1 260px;
  }
  .row {
    align-items: flex-end;
  }
</style>

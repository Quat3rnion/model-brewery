<script lang="ts">
  import { api, errorText } from '../lib/api';
  import { brewery } from '../lib/store.svelte';
  import { basename } from '../lib/format';

  let { host, path, kind }: { host: string; path: string; kind: string } = $props();
  let busy = $state(false);
  let error = $state<string | null>(null);
  const exists = $derived((brewery.state?.models ?? []).some((m) => m.host === host && m.path === path));

  async function add() {
    busy = true;
    error = null;
    try {
      await api.addModel({ host, path, kind, name: `${basename(path)} (copy on ${host})` });
      await brewery.refresh();
    } catch (e) {
      error = errorText(e);
    } finally {
      busy = false;
    }
  }
</script>

{#if exists}
  <span class="xs muted">In library</span>
{:else}
  <button type="button" class="btn small" disabled={busy} onclick={add}>{busy ? 'Adding…' : 'Add copy to library'}</button>
  {#if error}<span class="xs" style="color:var(--bad)" role="alert">{error}</span>{/if}
{/if}

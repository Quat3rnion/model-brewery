<script lang="ts">
  import { onMount } from 'svelte';

  type Worker = { id: string; configured: boolean };
  let workers = $state<Worker[]>([]);
  let host = $state('');
  const worker = $derived(workers.find((w) => w.id === host));
  let field = $state<HTMLInputElement>(null!);
  let loading = $state(true);
  let busy = $state(false);
  let confirming = $state(false);
  let notice = $state('');
  let error = $state('');
  const policy = { cache: 'no-store', credentials: 'same-origin', redirect: 'error', mode: 'same-origin' } as const;

  function clear() {
    if (field) field.value = '';
    confirming = false;
  }
  async function load() {
    loading = true;
    error = '';
    clear();
    try {
      const r = await fetch('/api/settings/hf-tokens', policy);
      if (!r.ok) throw new Error();
      const { hosts } = await r.json();
      if (!Array.isArray(hosts)) throw new Error();
      workers = hosts.map((w) => {
        if (!w || typeof w.id !== 'string' || typeof w.configured !== 'boolean') throw new Error();
        return { id: w.id, configured: w.configured };
      });
      host = workers[0]?.id ?? '';
    } catch {
      workers = [];
      error = 'Token status is unavailable.';
    } finally { loading = false; }
  }
  onMount(() => { void load(); });

  async function change(remove: boolean) {
    if (!worker || busy || (remove && !confirming) || (!remove && !field?.value)) return;
    const id = worker.id;
    busy = true;
    notice = error = '';
    try {
      // The secret lives only in the uncontrolled field and the outbound request.
      const pending = fetch(`/api/settings/${encodeURIComponent(id)}/hf-token`, {
        ...policy, method: remove ? 'DELETE' : 'PUT', headers: { 'Content-Type': 'application/json' },
        body: remove ? '{}' : JSON.stringify({ token: field.value }),
      });
      clear();
      const r = await pending;
      if (!r.ok) {
        error = r.status === 409 ? 'This worker is busy or no longer eligible. Refresh its status before retrying.' : 'The controller did not accept the token change.';
        return; // Never display a response body: an upstream error could echo input.
      }
      const data = await r.json();
      if (data.id !== id || data.configured !== !remove) throw new Error();
      workers = workers.map((w) => w.id === id ? { id, configured: !remove } : w);
      notice = remove ? 'Token removed.' : 'Token saved.';
    } catch { error = 'Could not confirm the change. Refresh the token status.'; }
    finally { clear(); busy = false; }
  }
</script>

<svelte:window onpagehide={clear} />
<section class="panel" aria-labelledby="hf-token-heading">
  <header><h2 id="hf-token-heading">Hugging Face access token</h2></header>
  <div class="body stack">
    <p class="small muted">Set a read token for a download worker to access private or gated repositories. The saved token is never shown.</p>
    {#if loading}
      <p role="status">Checking token status…</p>
    {:else if worker}
      <div class="field">
        <label for="hf-token-worker">Download worker</label>
        <select class="input" id="hf-token-worker" bind:value={host} disabled={busy} onchange={() => { clear(); notice = error = ''; }}>
          {#each workers as w (w.id)}<option value={w.id}>{w.id}</option>{/each}
        </select>
      </div>
      <p class="small" aria-live="polite">Token status: <strong>{worker.configured ? 'Configured' : 'Not configured'}</strong></p>
      <form class="stack" onsubmit={(e) => { e.preventDefault(); void change(false); }}>
        <div class="field">
          <label for="hf-new-token">New access token</label>
          <input bind:this={field} class="input" id="hf-new-token" type="password" autocomplete="off" spellcheck="false" autocapitalize="none" required disabled={busy || confirming} />
        </div>
        <div class="row">
          <button class="btn primary" type="submit" disabled={busy || confirming}>{worker.configured ? 'Replace token' : 'Set token'}</button>
          {#if worker.configured && !confirming}
            <button class="btn" type="button" disabled={busy} onclick={() => { clear(); confirming = true; notice = error = ''; }}>Remove token</button>
          {/if}
        </div>
      </form>
      {#if confirming}
        <p class="small">Remove the token from {worker.id}? Downloads that need it will no longer have access.</p>
        <div class="row">
          <button class="btn danger" type="button" onclick={() => change(true)}>Confirm removal</button>
          <button class="btn" type="button" onclick={() => confirming = false}>Keep token</button>
        </div>
      {/if}
    {:else if !error}
      <p class="muted small">No local download worker has token management enabled.</p>
    {/if}
    {#if notice}<p role="status" class="small">{notice}</p>{/if}
    {#if error}<p class="note warn" role="alert">{error}</p>{/if}
    {#if !loading && (error || !worker)}<div><button class="btn small" type="button" disabled={busy} onclick={load}>Refresh token status</button></div>{/if}
  </div>
</section>

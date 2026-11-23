<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { focusAfterRemoval } from '../lib/focus';

  let { confirmRefresh, onDirty }: { confirmRefresh: () => boolean; onDirty: (dirty: boolean) => void } = $props();
  const serviceId = 'backend-hf';
  let configured = $state<boolean | null>(null);
  let field = $state<HTMLInputElement>(null!);
  let loading = $state(true);
  let busy = $state(false);
  let confirming = $state(false);
  let notice = $state('');
  let error = $state('');
  let removeButton = $state<HTMLButtonElement>();
  let removeConfirm = $state<HTMLButtonElement>();
  const policy = { cache: 'no-store', credentials: 'same-origin', redirect: 'error', mode: 'same-origin' } as const;

  function clear() {
    if (field) field.value = '';
    onDirty(false);
    confirming = false;
  }
  async function load() {
    loading = true;
    error = '';
    try {
      const r = await fetch('/api/settings/hf-tokens', policy);
      if (!r.ok) throw new Error();
      const { hosts } = await r.json();
      if (!Array.isArray(hosts)) throw new Error();
      const service = hosts.find((w) => w?.id === serviceId);
      if (service && typeof service.configured !== 'boolean') throw new Error();
      configured = service?.configured ?? null;
    } catch {
      configured = null;
      error = 'Token status is unavailable.';
    } finally { loading = false; }
  }
  onMount(() => { void load(); });

  async function change(remove: boolean) {
    if (configured === null || busy || (remove && !confirming)) return;
    const token = remove ? '' : field?.value.trim();
    if (!remove && !token) {
      error = 'Enter an access token.';
      field?.focus();
      return;
    }
    busy = true;
    notice = error = '';
    try {
      // Never put the token in reactive state or persistent storage.
      const pending = fetch(`/api/settings/${serviceId}/hf-token`, {
        // Firefox otherwise sends Origin: null under the page's no-referrer policy.
        ...policy, referrerPolicy: 'same-origin', method: remove ? 'DELETE' : 'PUT', headers: { 'Content-Type': 'application/json' },
        body: remove ? '{}' : JSON.stringify({ token }),
      });
      if (remove) await focusAfterRemoval(clear, () => field);
      else clear();
      const r = await pending;
      if (!r.ok) {
        const advice: Record<number, string> = {
          400: 'Check the request. For a save, paste only the token and try again.',
          401: 'Sign in to the workbench again, then retry.',
          403: 'Access denied. Open Settings from the workbench address and check that you have permission to change token settings.',
          404: 'Ask your administrator to check that the built-in downloader is available.',
          409: 'The downloader is busy or unavailable. Check Jobs, then refresh the token status before retrying.',
          422: 'The controller could not process the request. Reload Settings and try again.',
          503: 'Ask your administrator to check the downloader’s token storage and its write permissions.',
        };
        error = `HTTP ${r.status}: ${advice[r.status] ?? 'Refresh the token status. If the problem persists, report this HTTP status to your administrator.'}`;
        return; // Never display a response body: an upstream error could echo input.
      }
      const data = await r.json();
      if (data.id !== serviceId || data.configured !== !remove) throw new Error();
      configured = data.configured;
      notice = remove ? 'Token removed.' : 'Token saved.';
    } catch { error = 'Could not confirm the change. Refresh the token status.'; }
    finally {
      clear(); busy = false;
      if (remove && document.activeElement === document.body) {
        await tick();
        if (document.activeElement === document.body) field?.focus();
      }
    }
  }
  async function reviewRemoval() {
    if (field?.value && !window.confirm('Discard the unsaved token?')) return;
    await focusAfterRemoval(() => { clear(); confirming = true; notice = error = ''; }, () => removeConfirm);
  }
</script>

<svelte:window onpagehide={clear} />
<section class="panel" aria-labelledby="hf-token-heading">
  <header><h2 id="hf-token-heading">Hugging Face access token</h2></header>
  <div class="body stack">
    <p class="small muted">Set a read token so the built-in downloader can access private or gated repositories. The saved token is never shown.</p>
    {#if loading}
      <p role="status">Checking token status…</p>
    {/if}
    {#if configured !== null}
      <p class="small" aria-live="polite">Token status: <strong>{configured ? 'Configured' : 'Not configured'}</strong></p>
      <form class="stack" onsubmit={(e) => { e.preventDefault(); void change(false); }}>
        <div class="field">
          <label for="hf-new-token">New access token</label>
          <input bind:this={field} class="input" id="hf-new-token" type="password" autocomplete="off" spellcheck="false" autocapitalize="none" required disabled={busy || confirming} oninput={(e) => onDirty(!!e.currentTarget.value)} />
        </div>
        <div class="row">
          <button class="btn primary" type="submit" disabled={busy || confirming}>{configured ? 'Replace token' : 'Set token'}</button>
          {#if configured && !confirming}
            <button bind:this={removeButton} class="btn" type="button" disabled={busy} onclick={reviewRemoval}>Remove token</button>
          {/if}
        </div>
      </form>
      {#if confirming}
        <p class="small">Remove the saved token? Downloads that need it will no longer have access.</p>
        <div class="row">
          <button bind:this={removeConfirm} class="btn danger" type="button" onclick={() => change(true)}>Confirm removal</button>
          <button class="btn" type="button" onclick={() => focusAfterRemoval(() => confirming = false, () => removeButton)}>Keep token</button>
        </div>
      {/if}
    {:else if !loading && !error}
      <p class="muted small">Token settings for the built-in downloader are unavailable.</p>
    {/if}
    {#if notice}<p role="status" class="small">{notice}</p>{/if}
    {#if error}<p class="note warn" role="alert">{error}</p>{/if}
    {#if !loading && (error || configured === null)}<div><button class="btn small" type="button" disabled={busy} onclick={() => {
      if (confirmRefresh()) {
        clear();
        void load();
      }
    }}>Refresh token status</button></div>{/if}
  </div>
</section>

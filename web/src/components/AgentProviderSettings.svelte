<script lang="ts">
  // Hosted model provider for the built-in workbench agent. The API key is
  // write-only: it is never read back, kept in reactive state or stored.
  import { onMount, tick } from 'svelte';
  import { api, errorText } from '../lib/api';
  import { fieldError, visibleBlockers } from '../lib/blockers';
  import { focusAfterRemoval } from '../lib/focus';
  import {
    DEFAULT_PROVIDER, LIMITS, PROVIDER_KINDS, defaultBaseUrl, endpointChanged, parseProviderSettings,
    providerBody, providerDirty, providerProblems, type Provider, type ProviderKind,
  } from '../lib/agent';
  import Blockers from './Blockers.svelte';
  import FieldError from './FieldError.svelte';

  let { confirmRefresh, onDirty }: { confirmRefresh: () => boolean; onDirty: (dirty: boolean) => void } = $props();
  const uid = $props.id();
  const policy = { cache: 'no-store', credentials: 'same-origin', redirect: 'error', mode: 'same-origin' } as const;

  let loading = $state(true);
  let loadError = $state('');
  let saved = $state<Provider | null>(null);
  let keyConfigured = $state(false);
  let draft = $state<Provider>({ ...DEFAULT_PROVIDER });
  let touched = $state<Record<string, boolean>>({});
  let attempted = $state(false);
  let saving = $state(false);
  let saveNotice = $state('');
  let saveError = $state('');

  let keyField = $state<HTMLInputElement>(null!);
  let keyDirty = $state(false);
  let keyBusy = $state(false);
  let confirming = $state(false);
  let keyNotice = $state('');
  let keyError = $state('');
  let removeButton = $state<HTMLButtonElement>();
  let removeConfirm = $state<HTMLButtonElement>();

  const problems = $derived(providerProblems(draft));
  const shown = $derived(visibleBlockers(problems, touched, attempted));
  const err = (field: string) => fieldError(shown, field);
  const dirty = $derived(saved ? providerDirty(draft, saved) : providerDirty(draft, DEFAULT_PROVIDER));
  const removesKey = $derived(keyConfigured && endpointChanged(saved, draft));
  $effect(() => { onDirty(dirty || keyDirty); });

  function reset(provider: Provider | null) {
    draft = { ...(provider ?? DEFAULT_PROVIDER) };
    touched = {};
    attempted = false;
  }
  async function load() {
    loading = true;
    loadError = '';
    try {
      const data = parseProviderSettings(await api.agentProvider());
      saved = data.provider;
      keyConfigured = data.keyConfigured;
      reset(data.provider);
    } catch (e) {
      loadError = `Provider settings are unavailable. ${errorText(e)}`;
    } finally { loading = false; }
  }
  onMount(() => { void load(); });

  function setKind(kind: ProviderKind) {
    // Replace the base URL only when it is still the default of the other kind.
    if (draft.base_url.trim() === '' || PROVIDER_KINDS.some((k) => defaultBaseUrl(k.id) === draft.base_url.trim())) {
      draft.base_url = defaultBaseUrl(kind);
    }
    draft.kind = kind;
  }

  async function save(event: SubmitEvent) {
    event.preventDefault();
    if (saving) return;
    attempted = true;
    saveNotice = saveError = '';
    if (problems.length) {
      await tick();
      document.getElementById(`${uid}-${problems[0].field}`)?.focus();
      return;
    }
    saving = true;
    const hadKey = keyConfigured;
    try {
      const data = parseProviderSettings(await api.saveAgentProvider(providerBody($state.snapshot(draft))));
      saved = data.provider;
      keyConfigured = data.keyConfigured;
      reset(data.provider);
      saveNotice = hadKey && !data.keyConfigured ? 'Provider saved. Set the API key again.' : 'Provider saved.';
    } catch (e) {
      saveError = errorText(e);
    } finally { saving = false; }
  }

  function clearKey() {
    if (keyField) keyField.value = '';
    keyDirty = false;
    confirming = false;
  }
  async function changeKey(remove: boolean) {
    if (!saved || keyBusy || (remove && !confirming)) return;
    const key = remove ? '' : keyField?.value.trim();
    if (!remove && !key) {
      keyError = 'Enter an API key.';
      keyField?.focus();
      return;
    }
    keyBusy = true;
    keyNotice = keyError = '';
    try {
      const pending = fetch('/api/settings/agent-provider/key', {
        ...policy, referrerPolicy: 'same-origin', method: remove ? 'DELETE' : 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: remove ? undefined : JSON.stringify({ key }),
      });
      if (remove) await focusAfterRemoval(clearKey, () => keyField);
      else clearKey();
      const r = await pending;
      if (!r.ok) {
        const advice: Record<number, string> = {
          400: 'Check the key. Paste only the API key and try again.',
          403: 'Access denied. Open Settings from the workbench address.',
          409: 'Save the provider first. You cannot change the key while an agent session runs.',
          503: 'Ask your administrator to check the private key storage of the controller.',
        };
        // Never show a response body: an upstream error could echo the key.
        keyError = `HTTP ${r.status}: ${advice[r.status] ?? 'Refresh the provider settings and try again.'}`;
        return;
      }
      const data = parseProviderSettings(await r.json());
      if (data.keyConfigured !== !remove) throw new Error();
      keyConfigured = data.keyConfigured;
      keyNotice = remove ? 'Key removed.' : 'Key saved.';
    } catch { keyError = 'Could not confirm the change. Refresh the provider settings.'; }
    finally {
      clearKey();
      keyBusy = false;
      if (remove && document.activeElement === document.body) {
        await tick();
        if (document.activeElement === document.body) keyField?.focus();
      }
    }
  }
  async function reviewRemoval() {
    if (keyField?.value && !window.confirm('Discard the unsaved key?')) return;
    await focusAfterRemoval(() => { clearKey(); confirming = true; keyNotice = keyError = ''; }, () => removeConfirm);
  }
</script>

<svelte:window onpagehide={clearKey} />
<section class="panel" aria-labelledby="{uid}-h">
  <header><h2 id="{uid}-h">Agent model provider</h2></header>
  <div class="body stack">
    <p class="small muted">The built-in agent sends its requests to this hosted model. It uses the workbench tools and makes scratch revisions only. Only an operator approves a revision in Edits.</p>
    {#if loading}<p role="status" class="small">Loading provider settings…</p>{/if}
    {#if loadError}
      <p class="note warn" role="alert">{loadError}</p>
      <div><button class="btn small" type="button" onclick={() => { if (confirmRefresh()) void load(); }}>Refresh provider settings</button></div>
    {:else if !loading}
      <form class="stack" novalidate onsubmit={save}>
        <div class="grid-2">
          <div class="field">
            <label for="{uid}-kind">Provider kind</label>
            <select class="input" id="{uid}-kind" value={draft.kind} onchange={(e) => setKind(e.currentTarget.value as ProviderKind)}>
              {#each PROVIDER_KINDS as k (k.id)}<option value={k.id}>{k.label}</option>{/each}
            </select>
          </div>
          <div class="field">
            <label for="{uid}-model">Model ID</label>
            <input class="input mono" id="{uid}-model" bind:value={draft.model} autocomplete="off" spellcheck="false" autocapitalize="none"
              aria-invalid={err('model') ? 'true' : undefined} aria-describedby={err('model') ? `${uid}-model-error` : undefined}
              onblur={() => (touched.model = true)} />
            <FieldError id="{uid}-model-error" error={err('model')} />
          </div>
        </div>
        <div class="field">
          <label for="{uid}-base_url">Base URL</label>
          <input class="input mono" id="{uid}-base_url" type="url" bind:value={draft.base_url} autocomplete="off" spellcheck="false" autocapitalize="none"
            aria-invalid={err('base_url') ? 'true' : undefined} aria-describedby="{uid}-base_url-hint{err('base_url') ? ` ${uid}-base_url-error` : ''}"
            onblur={() => (touched.base_url = true)} />
          <span class="hint" id="{uid}-base_url-hint">Use HTTPS. Include a path such as /v1 if the provider needs it, but not /messages or /chat/completions.</span>
          <FieldError id="{uid}-base_url-error" error={err('base_url')} />
        </div>
        {#if removesKey}
          <p class="note warn small">If you save a different provider kind or base URL, the controller removes the saved API key.</p>
        {/if}
        <fieldset class="limits">
          <legend>Limits</legend>
          <div class="grid-3">
            {#each LIMITS as limit (limit.field)}
              <div class="field">
                <label for="{uid}-{limit.field}">{limit.label}</label>
                <input class="input num" id="{uid}-{limit.field}" type="number" inputmode="numeric" min="1" max={limit.max} step="1"
                  bind:value={draft[limit.field]}
                  aria-invalid={err(limit.field) ? 'true' : undefined}
                  aria-describedby="{uid}-{limit.field}-hint{err(limit.field) ? ` ${uid}-{limit.field}-error` : ''}"
                  onblur={() => (touched[limit.field] = true)} />
                <span class="hint" id="{uid}-{limit.field}-hint">{limit.hint}</span>
                <FieldError id="{uid}-{limit.field}-error" error={err(limit.field)} />
              </div>
            {/each}
          </div>
        </fieldset>
        {#if attempted}<Blockers blockers={problems} />{/if}
        <div class="row">
          <button class="btn primary" type="submit" disabled={saving || (!!saved && !dirty)}>{saving ? 'Saving…' : 'Save provider'}</button>
          {#if dirty}<button class="btn" type="button" disabled={saving} onclick={() => { reset(saved); saveNotice = saveError = ''; }}>Discard changes</button>{/if}
        </div>
        {#if saveNotice}<p role="status" class="small">{saveNotice}</p>{/if}
        {#if saveError}<p class="note warn" role="alert">{saveError}</p>{/if}
      </form>

      <div class="stack key">
        <h3 class="small">API key</h3>
        <p class="small" aria-live="polite">Key status: <strong>{keyConfigured ? 'Configured' : 'Not configured'}</strong></p>
        <p class="small muted" id="{uid}-key-hint">{saved ? 'The saved key is never shown. Set a new key to replace it.' : 'Save the provider before you set its API key.'}</p>
        <form class="stack" onsubmit={(e) => { e.preventDefault(); void changeKey(false); }}>
          <div class="field">
            <label for="{uid}-key">New API key</label>
            <input bind:this={keyField} class="input" id="{uid}-key" type="password" autocomplete="off" spellcheck="false" autocapitalize="none"
              aria-describedby="{uid}-key-hint" disabled={!saved || keyBusy || confirming} oninput={(e) => (keyDirty = !!e.currentTarget.value)} />
          </div>
          <div class="row">
            <button class="btn primary" type="submit" disabled={!saved || keyBusy || confirming}>{keyConfigured ? 'Replace key' : 'Set key'}</button>
            {#if keyConfigured && !confirming}
              <button bind:this={removeButton} class="btn" type="button" disabled={keyBusy} onclick={reviewRemoval}>Remove key</button>
            {/if}
          </div>
        </form>
        {#if confirming}
          <p class="small">Remove the saved key? The agent cannot start a session until you set a key again.</p>
          <div class="row">
            <button bind:this={removeConfirm} class="btn danger" type="button" onclick={() => changeKey(true)}>Confirm removal</button>
            <button class="btn" type="button" onclick={() => focusAfterRemoval(() => (confirming = false), () => removeButton)}>Keep key</button>
          </div>
        {/if}
        {#if keyNotice}<p role="status" class="small">{keyNotice}</p>{/if}
        {#if keyError}<p class="note warn" role="alert">{keyError}</p>{/if}
      </div>
    {/if}
  </div>
</section>

<style>
  .limits {
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    padding: var(--s-3) var(--s-4) var(--s-4);
    margin: 0;
    min-width: 0;
  }
  .limits legend {
    padding: 0 var(--s-1);
    font-size: var(--fs-sm);
    font-weight: 600;
  }
  .key {
    border-top: 1px solid var(--line-soft);
    padding-top: var(--s-4);
  }
  h3 {
    font-weight: 600;
  }
</style>

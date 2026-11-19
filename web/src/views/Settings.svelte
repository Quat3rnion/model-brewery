<script lang="ts">
  import Infrastructure from '../components/Infrastructure.svelte';
  import HfTokenSettings from '../components/HfTokenSettings.svelte';
  import { api, ApiError, errorText } from '../lib/api';
  import { brewery } from '../lib/store.svelte';
  import { buildPut, effectiveReadiness, endpointProblem, hostDirty, idProblem, parseHost, parseSettings, validateHost, type Endpoint, type HostSettings } from '../lib/settings';
  import Badge from '../components/Badge.svelte';

  let loaded = $state(false);
  let status = $state<'ok' | 'unavailable' | 'error'>('ok');
  let loadError = $state<string | null>(null);
  let saved = $state<HostSettings[]>([]);
  let selectedId = $state('');
  let draft = $state<HostSettings | null>(null);
  let saving = $state(false);
  let saveError = $state<string | null>(null);
  let saveNote = $state<string | null>(null);

  const hosts = $derived(brewery.state?.hosts ?? []);
  const savedHost = $derived(saved.find((h) => h.id === selectedId));
  const profileOnly = $derived(savedHost?.evaluationStatus === 'profile_only');
  const problems = $derived(draft ? validateHost(draft, savedHost) : []);
  // Baseline setup locked by the controller (baseline_allowed: false). Existing entries stay as they are.
  const baselineLocked = $derived(draft?.baselineAllowed === false);
  const baselineWhy = $derived.by(() => {
    if (!savedHost) return '';
    const h = hosts.find((x) => x.id === savedHost.id);
    const reason = h?.reasons?.baseline;
    return reason ?? 'The controller does not allow baseline endpoints on this worker. An operator must provide an attested baseline runtime first.';
  });
  const dirty = $derived(!!draft && !!savedHost && hostDirty(draft, savedHost));
  const readiness = $derived(savedHost ? effectiveReadiness(savedHost, hosts.find((h) => h.id === savedHost.id)) : []);

  async function load(keepSelection = true) {
    try {
      saved = parseSettings(await api.settings());
      status = 'ok';
      loadError = null;
      if (!keepSelection || !saved.some((h) => h.id === selectedId)) selectedId = saved.find((h) => h.managed)?.id ?? saved[0]?.id ?? '';
      reset();
    } catch (e) {
      status = e instanceof ApiError && (e.status === 404 || e.status === 405) ? 'unavailable' : 'error';
      loadError = errorText(e);
    } finally {
      loaded = true;
    }
  }
  $effect(() => {
    void load(false);
  });
  function reset() {
    const h = saved.find((x) => x.id === selectedId);
    draft = h ? structuredClone($state.snapshot(h)) : null;
    saveError = null;
  }
  function select(id: string) {
    if (saving || id === selectedId) return;
    if (dirty && !confirm('Discard unsaved changes for this host?')) return;
    selectedId = id;
    saveNote = null;
    reset();
  }

  async function save(e: SubmitEvent) {
    e.preventDefault();
    if (!draft || problems.length || saving) return;
    saving = true;
    saveError = null;
    saveNote = null;
    try {
      const h = parseHost(await api.saveSettings(draft.id, buildPut(draft)));
      if (!h || h.id !== draft.id) throw new Error('The controller did not return settings for this worker. Reload saved settings to check the result.');
      saved = saved.map((x) => (x.id === h.id ? h : x));
      reset();
      saveNote = h.evaluationStatus === 'profile_only'
        ? `Profile saved on the controller for ${h.id}. Nothing was applied to the worker.`
        : `Evaluation settings saved on ${h.id}. No job was started.`;
      void brewery.refresh();
    } catch (err) {
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        saveError = `${errorText(err)} Your session may have expired; sign in again, then save. Your edits are kept until you reload.`;
      } else if (err instanceof ApiError && err.status === 409) {
        saveError = `${errorText(err)} Your edits are kept. Reload the saved settings to see the current version, then apply your changes again.`;
      } else saveError = errorText(err);
    } finally {
      saving = false;
    }
  }

  function addEndpoint(list: 'benchmarkTargets' | 'baselineEndpoints') {
    if (!draft) return;
    const port = draft.approvedPorts[0] ?? 8000;
    const base = { id: '', endpoint: `http://127.0.0.1:${port}/v1`, model: '' };
    draft[list] = [...draft[list], list === 'baselineEndpoints' ? { ...base, source_sha256: '', runtime_id: '' } : base];
  }
  function removeEndpoint(list: 'benchmarkTargets' | 'baselineEndpoints', i: number) {
    if (!draft) return;
    draft[list] = draft[list].filter((_, j) => j !== i);
  }
  function toggleSuite(id: string, on: boolean) {
    if (!draft) return;
    draft.enabledSuites = on ? [...draft.enabledSuites, id] : draft.enabledSuites.filter((x) => x !== id);
  }
</script>

{#snippet endpointList(title: string, list: 'benchmarkTargets' | 'baselineEndpoints', hint: string)}
  {@const items: Endpoint[] = draft ? draft[list] : []}
  {@const locked = list === 'baselineEndpoints' && baselineLocked}
  <fieldset class="group">
    <legend>{title}</legend>
    <p class="xs muted">{hint}</p>
    {#if locked}
      <p class="note warn xs" id="{list}-locked">
        Baseline setup is unavailable: {baselineWhy}{items.length ? ' Existing entries are kept unchanged when you save; you can remove them.' : ''}
      </p>
    {/if}
    {#if items.length === 0}<p class="xs muted">None.</p>{/if}
    {#each items as ep, i (i)}
      {@const idP = ep.id ? idProblem(ep.id) : null}
      {@const urlP = ep.endpoint ? endpointProblem(ep.endpoint) : null}
      <div class="ep">
        <div class="field">
          <label for="{list}-{i}-id">ID</label>
          <input id="{list}-{i}-id" readonly={locked} class="input mono" bind:value={ep.id} autocomplete="off" spellcheck="false" aria-invalid={idP ? 'true' : undefined} aria-describedby="{list}-{i}-id-h" />
          <span class="hint" class:bad={!!idP} id="{list}-{i}-id-h">{idP ?? 'Lowercase letters, digits, _ or -.'}</span>
        </div>
        <div class="field">
          <label for="{list}-{i}-label">Label <span class="muted">(optional)</span></label>
          <input id="{list}-{i}-label" readonly={locked} class="input" bind:value={ep.label} autocomplete="off" />
        </div>
        <div class="field wide">
          <label for="{list}-{i}-url">Endpoint on the worker</label>
          <input id="{list}-{i}-url" readonly={locked} class="input mono" bind:value={ep.endpoint} autocomplete="off" spellcheck="false" inputmode="url" aria-invalid={urlP ? 'true' : undefined} aria-describedby="{list}-{i}-url-h" />
          <span class="hint" class:bad={!!urlP} id="{list}-{i}-url-h">{urlP ?? (profileOnly ? 'Loopback on the selected worker. Saving this address does not connect to it or approve its port.' : `Loopback on the selected worker. Approved ports: ${draft?.approvedPorts.join(', ') || 'none'}.`)}</span>
        </div>
        <div class="field">
          <label for="{list}-{i}-model">Model name</label>
          <input id="{list}-{i}-model" readonly={locked} class="input mono" bind:value={ep.model} autocomplete="off" spellcheck="false" />
        </div>
        {#if list === 'baselineEndpoints'}
          {@const shaP = ep.source_sha256 && !/^[0-9a-f]{64}$/.test(ep.source_sha256) ? '64 lowercase hexadecimal characters.' : null}
          <div class="field wide">
            <label for="{list}-{i}-sha">Source checkpoint SHA-256</label>
            <input id="{list}-{i}-sha" readonly={locked} class="input mono" bind:value={ep.source_sha256} autocomplete="off" spellcheck="false" aria-invalid={shaP ? 'true' : undefined} aria-describedby="{list}-{i}-sha-h" />
            <span class="hint" class:bad={!!shaP} id="{list}-{i}-sha-h">{shaP ?? 'Identity of the exact BF16 checkpoint this endpoint serves. Evaluations match it to the bundle source.'}</span>
          </div>
          <div class="field">
            <label for="{list}-{i}-rt">Runtime ID</label>
            <input id="{list}-{i}-rt" readonly={locked} class="input mono" bind:value={ep.runtime_id} autocomplete="off" spellcheck="false" />
          </div>
        {/if}
        <div class="rm"><button type="button" class="btn small ghost" onclick={() => removeEndpoint(list, i)} aria-label="Remove {title.toLowerCase()} {ep.id || i + 1}">Remove</button></div>
      </div>
    {/each}
    {#if !locked}<div><button type="button" class="btn small" onclick={() => addEndpoint(list)}>Add {list === 'benchmarkTargets' ? 'target' : 'baseline endpoint'}</button></div>{/if}
  </fieldset>
{/snippet}

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Settings</h1>
    <p class="muted small">
      Select a host to manage weight editing, admission policy and resource limits. Evaluation settings are separate.
      <a href="#/settings/infrastructure">Infrastructure controls and audit log</a>
    </p>
  </div>

  {#if !loaded}
    <p class="muted">Loading…</p>
  {:else if status === 'unavailable'}
    <p class="note warn">This controller does not provide browser-managed settings yet. Benchmark and evaluation setup is in the worker configuration; see the README section "Managed benchmarks".</p>
  {:else if status === 'error'}
    <div class="note bad" role="alert"><p>Settings could not be loaded: {loadError}</p><button type="button" class="btn small" onclick={() => load()}>Retry</button></div>
  {:else if saved.length === 0}
    <p class="empty panel">No benchmark or evaluation worker settings reported.</p>
  {:else}
    <div class="layout">
      <nav class="hosts panel" aria-label="Workers">
        <ul>
          {#each saved as h (h.id)}
            <li>
              <button type="button" class="hb" disabled={saving} aria-current={h.id === selectedId ? 'true' : undefined} onclick={() => select(h.id)}>
                <span class="hn">{h.name ?? h.id}</span>
                <span class="xs muted">{h.id}</span>
              </button>
            </li>
          {/each}
        </ul>
      </nav>

      {#if draft && savedHost}
        <div class="stack">
          <section class="panel" aria-labelledby="host-mgmt-h">
            <header><h2 id="host-mgmt-h">Manage {savedHost.name ?? savedHost.id}</h2></header>
            <div class="body stack">
              {#if savedHost.managementAvailable === false}<p class="note warn">{savedHost.managementReason ?? 'The controller does not permit changes on this host.'}</p>{/if}
              <div class="row"><a class="btn" href="#/settings/infrastructure?host={encodeURIComponent(savedHost.id)}&section=weights">Weight editing settings</a>
                <a class="btn" href="#/settings/infrastructure?host={encodeURIComponent(savedHost.id)}&section=policy">Admission and resource limits</a></div>
            </div>
          </section>
          <section class="panel" aria-labelledby="rd-h">
            <header><h2 id="rd-h">Evaluation readiness</h2><span class="xs muted">Revision {savedHost.revision}</span></header>
            <div class="body stack">
              <p class="small">
                <Badge tone={savedHost.ready ? 'ok' : 'muted'} label={savedHost.ready ? 'Configuration ready' : profileOnly ? 'Profile only' : 'Configuration not ready'} />
                {#if savedHost.reason}<span class="muted">{savedHost.reason}</span>{/if}
              </p>
              <div class="table-wrap">
                <table class="data">
                  <caption class="sr-only">What the controller currently allows on this worker</caption>
                  <thead><tr><th scope="col">Job</th><th scope="col">Controller reports</th></tr></thead>
                  <tbody>
                    {#each readiness as r (r.what)}
                      <tr><th scope="row">{r.what}</th><td class="rc"><Badge tone={r.allowed ? 'ok' : 'muted'} label={r.allowed ? 'Available' : 'Not available'} />{#if r.reason}<span class="xs muted">{r.reason}</span>{/if}</td></tr>
                    {/each}
                  </tbody>
                </table>
              </div>
              {#if dirty}<p class="xs muted">You have unsaved edits; readiness shows the saved configuration.</p>{/if}
            </div>
          </section>

          {#if savedHost.evaluationStatus !== 'managed' && !profileOnly}
            <p class="note info">Evaluation settings unavailable: {savedHost.evaluationReason ?? 'This controller did not report an editable profile for this worker.'}</p>
          {:else}
          <form class="panel" aria-labelledby="ed-h" onsubmit={save} novalidate>
            <header><h2 id="ed-h">Evaluation configuration</h2></header>
            <fieldset class="body stack plain" disabled={saving}>
              <p class="small">{profileOnly ? 'Save target profiles on the controller for this worker. These profiles are not installed on the worker and do not enable execution.' : 'Save operational settings on this worker. Only approved ports, suites and runtimes can be used.'}</p>
              {#if !profileOnly && savedHost.savedProfileTargets?.length}
                <details class="group"><summary>Saved controller targets ({savedHost.savedProfileTargets.length})</summary>
                  <p class="xs muted">These targets are not applied automatically. Copy them to replace the targets in this form, then review and save.</p>
                  <ul>{#each savedHost.savedProfileTargets as ep}<li class="small break">{ep.id} · {ep.model} · <code>{ep.endpoint}</code></li>{/each}</ul>
                  <button class="btn small" type="button" onclick={() => {if(draft && savedHost?.savedProfileTargets) draft.benchmarkTargets=savedHost.savedProfileTargets.map(e=>({...e}));}}>Copy targets to form</button>
                </details>
              {/if}
              {@render endpointList('Benchmark targets', 'benchmarkTargets', profileOnly ? 'OpenAI-compatible target definitions. The model name identifies the model served by each endpoint.' : 'OpenAI-compatible endpoints that benchmark runs send requests to.')}

              {#if !profileOnly}
              <fieldset class="group">
                <legend>Benchmark suites</legend>
                {#if draft.approvedSuites.length === 0}
                  <p class="xs muted">No suites are installed and approved on this worker. Installing suites is an infrastructure task.</p>
                {:else}
                  <p class="xs muted">Installed and approved suites. Enable the ones benchmark runs may use.</p>
                  <div class="suites">
                    {#each draft.approvedSuites as s (s.id)}
                      <label class="chk"><input type="checkbox" checked={draft.enabledSuites.includes(s.id)} onchange={(e) => toggleSuite(s.id, e.currentTarget.checked)} /> {s.label ?? s.id}{#if s.label}{' '}<span class="mono xs muted">{s.id}</span>{/if}</label>
                    {/each}
                  </div>
                {/if}
              </fieldset>

              <fieldset class="group">
                <legend>GSM8K evaluation</legend>
                <label class="chk"><input type="checkbox" bind:checked={draft.evaluationEnabled} disabled={!draft.evaluationAllowed && !draft.evaluationEnabled} aria-describedby="ev-allow-h" /> Allow GSM8K evaluation on this worker</label>
                <p class="xs muted" id="ev-allow-h">{draft.evaluationAllowed ? 'The controller still checks the Halogen runtime; this only allows it.' : 'Not available: this worker has no approved Halogen runtime. An operator must provision one first.'}</p>
              </fieldset>

              {@render endpointList('BF16 baseline endpoints', 'baselineEndpoints', 'Endpoints that serve the exact BF16 source checkpoint, used to score a source baseline.')}
              {/if}

              {#if problems.length}
                <ul class="note warn xs probs" aria-label="Problems to fix before saving">{#each problems as p (p)}<li>{p}</li>{/each}</ul>
              {/if}
              <div class="row">
                <button type="submit" class="btn primary" disabled={!dirty || problems.length > 0 || saving}>{saving ? 'Saving…' : 'Save evaluation settings'}</button>
                <button type="button" class="btn ghost" disabled={!dirty || saving} onclick={reset}>Discard changes</button>
                {#if !dirty}<span class="xs muted">No unsaved changes.</span>{/if}
              </div>
              <div aria-live="polite">
                {#if saveError}<div class="note bad" role="alert"><p>{saveError}</p>{#if saveError.includes('Reload')}<button type="button" class="btn small" onclick={() => load()}>Reload saved settings</button>{/if}</div>{/if}
                {#if saveNote}<p class="note info">{saveNote}</p>{/if}
              </div>
            </fieldset>
          </form>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
  <Infrastructure />
  <HfTokenSettings />
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .layout {
    display: grid;
    grid-template-columns: 220px minmax(0, 1fr);
    gap: var(--s-4);
    align-items: start;
  }
  .hosts ul {
    list-style: none;
    margin: 0;
    padding: 4px;
  }
  .hb {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    width: 100%;
    padding: 8px 10px;
    border: 1px solid transparent;
    border-radius: var(--r-md);
    background: transparent;
    cursor: pointer;
    text-align: left;
  }
  .hb:hover {
    background: var(--accent-soft);
  }
  .hb[aria-current='true'] {
    border-color: var(--accent);
    background: var(--accent-soft);
  }
  .hn {
    font-weight: 600;
    font-size: var(--fs-sm);
    overflow-wrap: anywhere;
  }
  .plain {
    border: none;
    margin: 0;
    min-width: 0;
  }
  .group {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
    padding: var(--s-2) var(--s-3) var(--s-3);
    margin: 0;
    min-width: 0;
  }
  .group legend {
    padding: 0 4px;
    font-weight: 600;
    font-size: var(--fs-sm);
  }
  .ep {
    display: grid;
    grid-template-columns: minmax(120px, 1fr) minmax(120px, 1fr) minmax(220px, 2fr) minmax(140px, 1fr) auto;
    gap: var(--s-2) var(--s-3);
    align-items: start;
    padding-bottom: var(--s-2);
    border-bottom: 1px dashed var(--line-soft);
  }
  .rm {
    padding-top: 22px;
  }
  .rc {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 8px;
  }
  .hint.bad {
    color: var(--bad);
  }
  .suites {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 2px var(--s-4);
  }
  .chk {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 32px;
    font-size: var(--fs-sm);
    cursor: pointer;
  }
  .probs {
    margin: 0;
    padding-left: 2em;
  }
  @media (max-width: 1000px) {
    .ep {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }
    .ep .wide {
      grid-column: 1 / -1;
    }
    .rm {
      padding-top: 0;
    }
  }
  @media (max-width: 760px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    .ep {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>

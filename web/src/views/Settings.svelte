<script lang="ts">
  import { SvelteMap } from 'svelte/reactivity';
  import Infrastructure from '../components/Infrastructure.svelte';
  import HfTokenSettings from '../components/HfTokenSettings.svelte';
  import AgentProviderSettings from '../components/AgentProviderSettings.svelte';
  import Blockers from '../components/Blockers.svelte';
  import FieldError from '../components/FieldError.svelte';
  import { api, ApiError, errorText } from '../lib/api';
  import { fieldError, visibleBlockers } from '../lib/blockers';
  import { focusAfterRemoval } from '../lib/focus';
  import { hashParam, route } from '../lib/route.svelte';
  import { brewery } from '../lib/store.svelte';
  import { buildPut, effectiveReadiness, evaluationBlockers, hostDirty, parseHost, parseSettings, type Endpoint, type HostSettings } from '../lib/settings';
  import { unsavedChanges } from '../lib/unsaved-changes.svelte';
  import Badge from '../components/Badge.svelte';

  let loaded = $state(false);
  let status = $state<'ok' | 'unavailable' | 'error'>('ok');
  let loadError = $state<string | null>(null);
  let saved = $state<HostSettings[]>([]);
  let selectedId = $state('');
  type EvaluationEdit = { draft: HostSettings; base: HostSettings; touched: Record<string, boolean>; attempted: boolean };
  let edits = $state<Record<string, EvaluationEdit>>({});
  const otherDrafts = new SvelteMap<string, string>();
  let saving = $state(false);
  let saveError = $state<string | null>(null);
  let saveNote = $state<string | null>(null);
  let settingsHeading = $state<HTMLElement>();
  let evaluationHeading = $state<HTMLElement>();
  let readinessHeading = $state<HTMLElement>();
  let focusRequest = $state('');
  const key = (id: string) => `host:${id}`;
  function trackDraft(id: string, host: string, dirty: boolean) {
    if (dirty) otherDrafts.set(id, host);
    else otherDrafts.delete(id);
  }
  const unsaved = $derived(Object.values(edits).some((edit) => hostDirty(edit.draft, edit.base)) || otherDrafts.size > 0);
  unsavedChanges(() => unsaved);
  function confirmRefresh(clearToken = false) {
    const message = clearToken && otherDrafts.has('token')
      ? 'Refresh token status and discard the unsaved token?'
      : 'Refresh saved settings? Your unsaved edits stay in the form.';
    return !unsaved || window.confirm(message);
  }

  const hosts = $derived(brewery.state?.hosts ?? []);
  const savedHost = $derived(saved.find((h) => h.id === selectedId));
  const edit = $derived(edits[key(selectedId)]);
  const draft = $derived(edit?.draft);
  const profileOnly = $derived(savedHost?.evaluationStatus === 'profile_only');
  // The built-in download service never runs evaluation, even when it reports
  // the generic controller-only profile contract.
  const downloadOnly = $derived(selectedId === 'backend-hf');
  const problems = $derived(draft ? evaluationBlockers(draft, edit?.base) : []);
  const formBlockers = $derived(visibleBlockers(problems, edit?.touched ?? {}, edit?.attempted));
  const canConfigureTargets = $derived(!downloadOnly && (profileOnly || !!draft && (
    draft.benchmarkTargets.length > 0 || draft.approvedSuites.length > 0 ||
    hosts.find(h => h.id === draft.id)?.capabilities?.benchmark === true ||
    savedHost?.readiness.benchmark?.ready === true
  )));
  // Baseline setup locked by the controller (baseline_allowed: false). Existing entries stay as they are.
  const baselineLocked = $derived(draft?.baselineAllowed === false);
  const baselineWhy = $derived.by(() => {
    if (!savedHost) return '';
    const h = hosts.find((x) => x.id === savedHost.id);
    const reason = h?.reasons?.baseline;
    return reason ?? 'The controller does not allow reference endpoints on this worker. Ask the administrator to configure a verified reference runtime.';
  });
  const dirty = $derived(!!edit && hostDirty(edit.draft, edit.base));
  const stale = $derived(!!edit && !!savedHost && dirty && hostDirty(edit.base, savedHost));
  const readiness = $derived(savedHost ? effectiveReadiness(savedHost, hosts.find((h) => h.id === savedHost.id)) : []);
  // Older controllers do not send readiness rows and may report a static ready flag.
  const ready = $derived(savedHost && (Object.keys(savedHost.readiness).length > 0
    ? savedHost.ready
    : readiness.some((r) => r.allowed)));
  const evaluationReadiness = $derived(readiness.find((r) => r.kind === 'evaluate'));

  function resetHost(host: HostSettings) {
    const base = structuredClone($state.snapshot(host));
    edits[key(host.id)] = { base, draft: structuredClone(base), touched: {}, attempted: false };
  }
  let readinessRequest = 0;
  async function load() {
    const request = ++readinessRequest;
    try {
      const next = parseSettings(await api.settings());
      if (request !== readinessRequest) return;
      saved = next;
      status = 'ok';
      loadError = null;
      for (const host of saved) {
        const previous = edits[key(host.id)];
        if (!previous || !hostDirty(previous.draft, previous.base)) resetHost(host);
      }
      if (!selectedId) selectedId = saved.find((h) => h.managed)?.id ?? saved[0]?.id ?? '';
    } catch (e) {
      if (request !== readinessRequest) return;
      status = e instanceof ApiError && (e.status === 404 || e.status === 405) ? 'unavailable' : 'error';
      loadError = errorText(e);
    } finally {
      if (request === readinessRequest) loaded = true;
    }
  }
  $effect(() => { void load(); });
  $effect(() => {
    const hash = route.hash;
    const path = hash.split('?')[0];
    const id = hashParam(hash, 'host');
    if (id) selectedId = id;
    focusRequest = path === '#/settings/evaluation' ? 'evaluation' : hashParam(hash, 'section')
      ?? (path === '#/settings/infrastructure' ? (id ? 'weights' : 'infrastructure') : '');
  });
  const allowsEvaluationFocus = (el: Element | null) => el === document.body || el === settingsHeading ||
    (el instanceof HTMLAnchorElement && el.hash === route.hash);
  $effect(() => {
    if (focusRequest !== 'evaluation') return;
    const cancel = (event: FocusEvent) => {
      if (event.target instanceof Element && !allowsEvaluationFocus(event.target)) focusRequest = '';
    };
    document.addEventListener('focusin', cancel);
    return () => document.removeEventListener('focusin', cancel);
  });
  $effect(() => {
    if (focusRequest !== 'evaluation' || !loaded) return;
    const heading = evaluationHeading ?? readinessHeading ?? settingsHeading;
    if (heading) {
      focusRequest = '';
      if (allowsEvaluationFocus(document.activeElement)) heading.focus();
    }
  });
  function reset() {
    if (savedHost) resetHost(savedHost);
    saveError = null;
  }
  function select(id: string): boolean {
    if (id === selectedId) return true;
    if (saving || ((dirty || [...otherDrafts.values()].includes(selectedId)) &&
      !window.confirm('Change host? Your unsaved edits stay with this host.'))) return false;
    selectedId = id;
    focusRequest = '';
    saveNote = null;
    saveError = null;
    return true;
  }

  async function save(e: SubmitEvent) {
    e.preventDefault();
    if (edit) edit.attempted = true;
    if (!draft || !dirty || stale || problems.length || saving) return;
    const submitted = draft;
    saving = true;
    saveError = null;
    saveNote = null;
    try {
      const h = parseHost(await api.saveSettings(submitted.id, buildPut(submitted)));
      if (!h || h.id !== submitted.id) throw new Error('The controller did not return settings for this worker. Reload saved settings to check the result.');
      saved = saved.map((x) => (x.id === h.id ? h : x));
      resetHost(h);
      saveNote = h.evaluationStatus === 'profile_only'
        ? `Profile saved on the controller for ${h.id}. Nothing was applied to the worker.`
        : `Evaluation settings saved on ${h.id}. No job was started.`;
      void brewery.refresh();
    } catch (err) {
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        saveError = `${errorText(err)} Your session may have expired; sign in again, then save. Your edits are kept until you reload.`;
      } else if (err instanceof ApiError && err.status === 409) {
        saveError = `${errorText(err)} Your edits are kept. Reload saved settings before you review your edits again.`;
      } else saveError = errorText(err);
    } finally {
      saving = false;
    }
  }
  function touch(field: string) {
    if (edit) edit.touched[field] = true;
  }

  function addEndpoint(list: 'benchmarkTargets' | 'baselineEndpoints') {
    if (!draft) return;
    const port = draft.approvedPorts[0] ?? 8000;
    const base = { id: '', endpoint: `http://127.0.0.1:${port}/v1`, model: '' };
    draft[list] = [...draft[list], list === 'baselineEndpoints' ? { ...base, source_sha256: '', runtime_id: '' } : base];
  }
  function removeEndpoint(list: 'benchmarkTargets' | 'baselineEndpoints', i: number, fallback: string) {
    if (!draft) return;
    const current = draft;
    void focusAfterRemoval(() => {
      current[list] = current[list].filter((_, j) => j !== i);
      if (edit) edit.touched = {};
    }, () => document.getElementById(fallback));
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
        Reference setup is unavailable: {baselineWhy}{items.length ? ' Existing entries stay unchanged. You can remove them.' : ''}
      </p>
    {/if}
    {#if items.length === 0}<p class="xs muted">None.</p>{/if}
    {#each items as ep, i (i)}
      {@const idP = fieldError(formBlockers, `${list}-${i}-id`)}
      {@const urlP = fieldError(formBlockers, `${list}-${i}-url`)}
      {@const modelP = fieldError(formBlockers, `${list}-${i}-model`)}
      <div class="ep">
        <div class="field">
          <label for="{list}-{i}-id">ID</label>
          <input id="{list}-{i}-id" readonly={locked} class="input mono" bind:value={ep.id} onblur={() => touch(`${list}-${i}-id`)} autocomplete="off" spellcheck="false" aria-invalid={idP ? 'true' : undefined} aria-describedby={idP ? `${list}-${i}-id-err` : `${list}-${i}-id-h`} />
          <FieldError id="{list}-{i}-id-err" error={idP} />
          {#if !idP}<span class="hint" id="{list}-{i}-id-h">Lowercase letters, digits, _ or -.</span>{/if}
        </div>
        <div class="field">
          <label for="{list}-{i}-label">Label <span class="muted">(optional)</span></label>
          <input id="{list}-{i}-label" readonly={locked} class="input" bind:value={ep.label} autocomplete="off" />
        </div>
        <div class="field wide">
          <label for="{list}-{i}-url">Endpoint on the worker</label>
          <input id="{list}-{i}-url" readonly={locked} class="input mono" bind:value={ep.endpoint} onblur={() => touch(`${list}-${i}-url`)} autocomplete="off" spellcheck="false" inputmode="url" aria-invalid={urlP ? 'true' : undefined} aria-describedby={urlP ? `${list}-${i}-url-err` : `${list}-${i}-url-h`} />
          <FieldError id="{list}-{i}-url-err" error={urlP} />
          {#if !urlP}<span class="hint" id="{list}-{i}-url-h">{profileOnly ? 'Use a loopback address on the worker. Saving does not connect to the address or approve the port.' : `Use a loopback address on the worker. Approved ports: ${draft?.approvedPorts.join(', ') || 'none'}.`}</span>{/if}
        </div>
        <div class="field">
          <label for="{list}-{i}-model">Model name</label>
          <input id="{list}-{i}-model" readonly={locked} class="input mono" bind:value={ep.model} onblur={() => touch(`${list}-${i}-model`)} autocomplete="off" spellcheck="false" aria-invalid={modelP ? 'true' : undefined} aria-describedby={modelP ? `${list}-${i}-model-err` : undefined} />
          <FieldError id="{list}-{i}-model-err" error={modelP} />
        </div>
        {#if list === 'baselineEndpoints'}
          {@const shaP = fieldError(formBlockers, `${list}-${i}-sha`)}
          {@const runtimeP = fieldError(formBlockers, `${list}-${i}-rt`)}
          {@const precisionP = fieldError(formBlockers, `${list}-${i}-precision`)}
          <div class="field wide">
            <label for="{list}-{i}-sha">Reference model SHA-256</label>
            <input id="{list}-{i}-sha" readonly={locked} class="input mono" bind:value={ep.source_sha256} onblur={() => touch(`${list}-${i}-sha`)} autocomplete="off" spellcheck="false" aria-invalid={shaP ? 'true' : undefined} aria-describedby={shaP ? `${list}-${i}-sha-err` : `${list}-${i}-sha-h`} />
            <FieldError id="{list}-{i}-sha-err" error={shaP} />
            {#if !shaP}<span class="hint" id="{list}-{i}-sha-h">Use the hash of the model that this endpoint serves.</span>{/if}
          </div>
          <div class="field">
            <label for="{list}-{i}-rt">Runtime ID</label>
            <input id="{list}-{i}-rt" readonly={locked} class="input mono" bind:value={ep.runtime_id} onblur={() => touch(`${list}-${i}-rt`)} autocomplete="off" spellcheck="false" aria-invalid={runtimeP ? 'true' : undefined} aria-describedby={runtimeP ? `${list}-${i}-rt-err` : undefined} />
            <FieldError id="{list}-{i}-rt-err" error={runtimeP} />
          </div>
          <div class="field">
            <label for="{list}-{i}-precision">Reference precision (optional)</label>
            <input id="{list}-{i}-precision" readonly={locked} class="input mono" bind:value={ep.precision} onblur={() => touch(`${list}-${i}-precision`)} maxlength="32" autocomplete="off" spellcheck="false" placeholder="e.g. Q4_K_M" aria-invalid={precisionP ? 'true' : undefined} aria-describedby={precisionP ? `${list}-${i}-precision-err` : undefined} />
            <FieldError id="{list}-{i}-precision-err" error={precisionP} />
          </div>
        {/if}
        <div class="rm"><button type="button" class="btn small ghost" onclick={() => removeEndpoint(list, i, locked ? 'reference-summary' : `${list}-add`)} aria-label="Remove {list === 'benchmarkTargets' ? 'target' : 'reference endpoint'} {ep.id || i + 1}">Remove</button></div>
      </div>
    {/each}
    {#if !locked}<div><button id="{list}-add" type="button" class="btn small" onclick={() => addEndpoint(list)}>Add {list === 'benchmarkTargets' ? 'target' : 'reference endpoint'}</button></div>{/if}
  </fieldset>
{/snippet}

<div class="stack">
  <div class="head">
    <h1 tabindex="-1" bind:this={settingsHeading}>Settings</h1>
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
    <div class="note bad" role="alert"><p>Settings could not be loaded: {loadError}</p><button type="button" class="btn small" onclick={() => { if (confirmRefresh()) void load(); }}>Retry</button></div>
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
                {#if h.name && h.name !== h.id}<span class="xs muted">{h.id}</span>{/if}
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
              <div class="row"><a class="btn" href="#/settings/infrastructure?host={encodeURIComponent(savedHost.id)}&section=weights"
                  onclick={(e) => { if (e.currentTarget.hash === route.hash) focusRequest = 'weights'; }}>Weight editing settings</a>
                <a class="btn" href="#/settings/infrastructure?host={encodeURIComponent(savedHost.id)}&section=policy"
                  onclick={(e) => { if (e.currentTarget.hash === route.hash) focusRequest = 'policy'; }}>Admission and resource limits</a></div>
            </div>
          </section>
          <section class="panel" aria-labelledby="rd-h">
            <header><h2 id="rd-h" tabindex="-1" bind:this={readinessHeading}>Evaluation readiness</h2><span class="xs muted" title="The version of the saved settings">Revision {savedHost.revision}</span></header>
            <div class="body stack">
              <p class="small">
                <Badge tone={ready ? 'ok' : 'muted'} label={ready ? 'Configuration ready' : profileOnly ? 'Profile only' : 'Configuration not ready'} />
                {#if savedHost.reason}<span class="muted">{savedHost.reason}</span>{/if}
              </p>
              {#if profileOnly && !downloadOnly}<p class="xs muted">Profile only: the controller stores these targets but does not apply them to this worker.</p>{/if}
              <div class="table-wrap">
                <table class="data">
                  <caption class="sr-only">What the controller currently allows on this worker</caption>
                  <thead><tr><th scope="col">Job</th><th scope="col">Controller reports</th></tr></thead>
                  <tbody>
                    {#each readiness as r (r.kind)}
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
            <header><h2 tabindex="-1" bind:this={evaluationHeading} id="ed-h">Evaluation configuration</h2></header>
            <fieldset class="body stack plain" disabled={saving}>
              {#if !downloadOnly}<p class="small">{profileOnly ? 'Save target profiles on the controller for this worker. These profiles do not enable execution.' : 'Save settings on this worker. Use only approved ports, suites and runtimes.'}</p>{/if}
              {#if canConfigureTargets && !profileOnly && savedHost.savedProfileTargets?.length}
                <details class="group"><summary>Saved controller targets ({savedHost.savedProfileTargets.length})</summary>
                  <p class="xs muted">These targets are not applied automatically. Copy them to replace the targets in this form, then review and save.</p>
                  <ul>{#each savedHost.savedProfileTargets as ep}<li class="small break">{ep.id} · {ep.model} · <code>{ep.endpoint}</code></li>{/each}</ul>
                  <button class="btn small" type="button" onclick={() => {if(draft && savedHost?.savedProfileTargets) draft.benchmarkTargets=savedHost.savedProfileTargets.map(e=>({...e}));}}>Copy targets to form</button>
                </details>
              {/if}
              {#if canConfigureTargets}
                {@render endpointList('Benchmark targets', 'benchmarkTargets', profileOnly ? 'OpenAI-compatible target definitions. The model name identifies the model served by each endpoint.' : 'OpenAI-compatible endpoints that benchmark runs send requests to.')}
              {:else}
                <p class="small muted">{downloadOnly ? 'The built-in download service does not run evaluation.' : 'This host does not report benchmark target support.'}</p>
              {/if}

              {#if !profileOnly}
              <fieldset class="group">
                <legend>Benchmark suites</legend>
                {#if draft.approvedSuites.length === 0}
                  <p class="xs muted">No suites are installed and approved on this worker. Installing suites is an infrastructure task.</p>
                {:else}
                  <p class="xs muted">Installed and approved suites. Enable the ones benchmark runs may use.</p>
                  <div class="suites">
                    {#each draft.approvedSuites as s (s.id)}
                      <label class="chk"><input type="checkbox" checked={draft.enabledSuites.includes(s.id)} onchange={(e) => { touch('suites'); toggleSuite(s.id, e.currentTarget.checked); }} /> {s.label ?? s.id}{#if s.label && s.label !== s.id}{' '}<span class="mono xs muted">{s.id}</span>{/if}</label>
                    {/each}
                  </div>
                {/if}
              </fieldset>

              <fieldset class="group">
                <legend>GSM8K evaluation</legend>
                <label class="chk"><input type="checkbox" bind:checked={draft.evaluationEnabled} onchange={() => touch('gsm8k')} disabled={!draft.evaluationAllowed && !draft.evaluationEnabled} aria-describedby="ev-allow-h" /> Allow GSM8K evaluation on this worker</label>
                <p class="xs muted" id="ev-allow-h">{draft.evaluationAllowed ? 'The controller still checks the Halogen runtime (the program that runs the model).' : evaluationReadiness?.reason ?? 'GSM8K evaluation is not allowed on this worker.'}</p>
              </fieldset>

              {#key draft.id}
                <details class="disclosure">
                  <summary id="reference-summary">Reference endpoints{draft.baselineEndpoints.length ? ` · ${draft.baselineEndpoints.length}` : ' (optional)'}</summary>
                  {@render endpointList('Reference endpoints', 'baselineEndpoints', 'Compare with predictions from a model served on this worker. Standalone evaluation does not use these endpoints.')}
                </details>
              {/key}
              {/if}

              <Blockers blockers={formBlockers} />
              {#if stale}<p class="note warn">Saved settings changed. Your edits are kept. Discard your edits to load the new version.</p>{/if}
              {#if !downloadOnly}<div class="row">
                <button type="submit" class="btn primary" disabled={!dirty || stale || problems.length > 0 || saving}>{saving ? 'Saving…' : 'Save evaluation settings'}</button>
                <button type="button" class="btn ghost" disabled={!dirty || saving} onclick={() => { reset(); evaluationHeading?.focus(); }}>Discard changes</button>
                {#if !dirty}<span class="xs muted">No unsaved changes.</span>{/if}
              </div>{/if}
              <div aria-live="polite">
                {#if saveError}<div class="note bad" role="alert"><p>{saveError}</p>{#if saveError.includes('Reload')}<button type="button" class="btn small" onclick={() => { if (confirmRefresh()) void load(); }}>Reload saved settings</button>{/if}</div>{/if}
                {#if saveNote}<p class="note info" role="status">{saveNote}</p>{/if}
              </div>
            </fieldset>
          </form>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
  <Infrastructure hostId={selectedId} selectHost={select} {confirmRefresh} {trackDraft} {focusRequest} settingsLoaded={loaded}
    clearFocus={() => focusRequest = ''} refreshReadiness={load} />
  <HfTokenSettings confirmRefresh={() => confirmRefresh(true)} onDirty={(dirty) => trackDraft('token', '', dirty)} />
  <AgentProviderSettings confirmRefresh={() => confirmRefresh()} onDirty={(dirty) => trackDraft('agent-provider', '', dirty)} />
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

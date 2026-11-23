<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { api, errorText } from '../lib/api';
  import { brewery } from '../lib/store.svelte';
  import { WEIGHT_SOURCES, MODEL_KINDS, basename, formatBytes, hostName, isActive, kindLabel, modelLabel } from '../lib/format';
  import { weightAccess } from '../lib/weight-access';
  import HostSelect from '../components/HostSelect.svelte';
  import HfDownload from '../components/HfDownload.svelte';
  import { hashParam, route } from '../lib/route.svelte';
  import Combobox from '../components/Combobox.svelte';
  import { hostOptions, nameSuggestions } from '../lib/pickers';
  import { completionText, IDLE, PathSuggester, type PathState } from '../lib/paths';
  import type { ComboOption } from '../lib/combobox';
  import Badge from '../components/Badge.svelte';
  import WorkflowGate from '../components/WorkflowGate.svelte';
  import { hostWorkflowEligibility } from '../lib/eligibility';
  import { focusAfterRemoval } from '../lib/focus';
  import { unsavedChanges } from '../lib/unsaved-changes.svelte';
  import type { Model, Prompt } from '../lib/types';

  const models = $derived(brewery.state?.models ?? []);
  const prompts = $derived(brewery.state?.prompts ?? []);
  const hosts = $derived(brewery.state?.hosts ?? []);
  const jobs = $derived(brewery.state?.jobs ?? []);

  // Register form
  let host = $state('');
  let path = $state('');
  let kind = $state('gguf');
  let name = $state('');
  let regBusy = $state(false);
  let regError = $state<string | null>(null);
  let regDone = $state<string | null>(null);
  const pathProblem = $derived(path.trim() === '' ? null : path.trim().startsWith('/') ? null : 'Use an absolute path that starts with /.');
  // Path suggestions from the selected worker. They only help typing:
  // a manual absolute path is valid when the worker is offline.
  let paths = $state<PathState>(IDLE);
  const suggester = new PathSuggester((h, prefix, signal) => api.suggestPaths(h, prefix, signal), (s) => (paths = s));
  $effect(() => () => suggester.cancel());
  $effect(() => {
    // A host change drops the old host's results.
    const h = host;
    suggester.cancel();
    paths = { ...IDLE, host: h };
    if (document.activeElement?.id === 'reg-path') untrack(() => suggester.request(h, path));
  });
  const pathOptions = $derived<ComboOption[]>(
    paths.host === host
      ? paths.items.map((it) => ({ value: completionText(it), label: completionText(it), detail: it.kind === 'directory' ? 'Directory' : 'File' }))
      : [],
  );
  const pathStatus = $derived.by(() => {
    if (!host) return 'Select a host to list paths on it. You can also type the full path.';
    if (paths.status === 'loading') return `Listing ${paths.prefix || 'roots'} on ${hostName(hosts, host)}…`;
    if (paths.status === 'error' || paths.status === 'unavailable' || (paths.status === 'idle' && paths.message)) return `${paths.message} You can still type the full path.`;
    if (paths.status === 'ready' && paths.truncated) return 'More entries exist. Type more of the name to narrow the list.';
    return null;
  });
  const nameOptions = $derived(nameSuggestions(path, models));
  const nameInUse = $derived(name.trim() !== '' && models.some((m) => m.name === name.trim()));

  const registration = $derived(hostWorkflowEligibility(brewery.state?.eligibility, host, 'register_manual'));
  const regBlockers = $derived([
    ...(!host ? ['Select a host.'] : registration?.eligible === false
      ? [...registration.reasons.map((reason) => reason.text), 'Select a different host.'] : []),
    ...(path.trim() ? [] : ['Enter the path on that host.']),
    ...(pathProblem ? [pathProblem] : []),
  ]);

  async function register(e: SubmitEvent) {
    e.preventDefault();
    if (regBlockers.length) return;
    regBusy = true;
    regError = null;
    regDone = null;
    try {
      const m = await api.addModel({ host, path: path.trim(), kind, ...(name.trim() ? { name: name.trim() } : {}) });
      regDone = `Registered ${modelLabel(m)}. The file is not read or copied until a job uses it.`;
      path = '';
      name = '';
      await brewery.refresh();
    } catch (err) {
      regError = errorText(err);
    } finally {
      regBusy = false;
    }
  }

  // Filter
  let filter = $state('');
  let hostFilter = $state('');
  $effect(() => {
    const entry = hashParam(route.hash, 'entry');
    filter = entry ?? '';
    if (entry) hostFilter = '';
  });
  const exactEntry = $derived(models.some((m) => m.id === filter));
  const shown = $derived(
    models.filter(
      (m) =>
        (!hostFilter || m.host === hostFilter) &&
        (!filter || (exactEntry ? m.id === filter : `${m.name ?? ''} ${m.path} ${m.kind}`.toLowerCase().includes(filter.toLowerCase()))),
    ),
  );

  // Removal
  let removing = $state<string | null>(null);
  let removeBusy = $state(false);
  let removeError = $state<string | null>(null);
  const removeButtonId = (id: string) => `remove-${encodeURIComponent(id)}`;
  function usedBy(m: Model) {
    return jobs.filter((j) => isActive(j.state) && j.input_ids.includes(m.id));
  }
  async function remove(m: Model) {
    if (removeBusy || usedBy(m).length) return;
    removeBusy = true;
    removeError = null;
    try {
      await api.removeModel(m.id);
      await focusAfterRemoval(() => { removing = null; }, () => document.getElementById('lib-filter'));
      await brewery.refresh();
    } catch (err) {
      removeError = errorText(err);
    } finally {
      removeBusy = false;
    }
  }

  // Prompts
  let pName = $state('');
  let pText = $state('');
  let pBusy = $state(false);
  let pError = $state<string | null>(null);
  let openPrompt = $state<string | null>(null);
  let editing = $state<Prompt | null>(null);
  let deleting = $state<string | null>(null);
  let deleteBusy = $state(false);
  let deleteError = $state<string | null>(null);
  const promptToggleId = (id: string) => `prompt-${encodeURIComponent(id)}`;
  const promptGuard = unsavedChanges(() => (editing?.name ?? '') !== pName || (editing?.text ?? '') !== pText);
  async function editPrompt(p: Prompt) {
    if (pBusy || !promptGuard.confirm()) return;
    editing = p;
    pName = p.name;
    pText = p.text;
    pError = null;
    await tick();
    document.getElementById('p-name')?.focus();
  }
  function resetPrompt() {
    editing = null;
    pName = '';
    pText = '';
    pError = null;
  }
  function cancelPromptEdit() {
    if (!editing || !promptGuard.confirm()) return;
    const id = editing.id;
    void focusAfterRemoval(resetPrompt, () => document.getElementById(promptToggleId(id)));
  }
  async function deletePrompt(id: string) {
    if (deleteBusy) return;
    deleteBusy = true;
    deleteError = null;
    try {
      await api.deletePrompt(id);
      await focusAfterRemoval(() => {
        deleting = null;
        openPrompt = null;
      }, () => document.getElementById('prompts-h'));
      await brewery.refresh();
    } catch (err) {
      deleteError = errorText(err);
    } finally {
      deleteBusy = false;
    }
  }
  async function savePrompt(e: SubmitEvent) {
    e.preventDefault();
    if (pBusy || !pName.trim() || !pText) return;
    pBusy = true;
    pError = null;
    try {
      const input = { name: pName.trim(), text: pText };
      if (editing) {
        const id = editing.id;
        await api.updatePrompt(id, input);
        openPrompt = id;
        await focusAfterRemoval(resetPrompt, () => document.getElementById(promptToggleId(id)));
      } else {
        await api.addPrompt(input);
        resetPrompt();
      }
      await brewery.refresh();
    } catch (err) {
      pError = errorText(err);
    } finally {
      pBusy = false;
    }
  }

  const selectedKind = $derived(MODEL_KINDS.find((k) => k.value === kind));
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Library</h1>
    <p class="muted small">Entries point to files on a host. Adding or removing an entry never changes or deletes the original files.</p>
    <div class="row">
      <a class="btn small" href="#/weights">Open weight editor</a>
      <span class="muted small">Edit a GGUF, native HGN, SafeTensors file or indexed Hugging Face checkpoint on its registered host.</span>
    </div>
  </div>

  <HfDownload />
  <section class="panel" aria-labelledby="reg-h">
    <header><h2 id="reg-h">Register an entry</h2></header>
    <WorkflowGate workflow="register_manual">
    <form class="body stack" onsubmit={register} novalidate>
      <div class="reg">
        <HostSelect bind:value={host} id="reg-host" workflow="register_manual" hideIneligible />
        <div class="field grow">
          <Combobox
            bind:value={path}
            id="reg-path"
            label="Path on host"
            free
            filter={false}
            mono
            options={pathOptions}
            loading={paths.status === 'loading'}
            status={pathStatus}
            emptyText={paths.status === 'ready' ? 'No entries with this prefix.' : ''}
            placeholder="/models/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf…"
            invalid={!!pathProblem}
            describedby="reg-path-hint"
            onfocus={() => {
              if (paths.status === 'idle' || paths.host !== host) suggester.request(host, path);
            }}
            ontextinput={(t) => suggester.request(host, t)}
            onpick={(o) => o.detail === 'Directory'}
          />
          <span class="hint" id="reg-path-hint">{pathProblem ?? 'Absolute path. Pick a directory to list it. The worker checks access and hashes the file when a job uses it.'}</span>
        </div>
        <div class="field">
          <label for="reg-kind">Kind</label>
          <select id="reg-kind" class="input" bind:value={kind} aria-describedby="reg-kind-hint">
            {#each MODEL_KINDS as k (k.value)}<option value={k.value}>{k.label}</option>{/each}
          </select>
          <span class="hint" id="reg-kind-hint">{selectedKind?.hint}</span>
        </div>
        <div class="field">
          <Combobox
            bind:value={name}
            id="reg-name"
            label="Name (optional)"
            free
            options={nameOptions}
            placeholder={path ? basename(path) : 'Defaults to file name'}
            emptyText="No suggestion. Type any name."
            describedby="reg-name-hint"
          />
          <span class="hint" id="reg-name-hint">{nameInUse ? 'Another entry uses this name. Pickers add the entry ID to tell them apart.' : 'Leave empty to use the file or directory name.'}</span>
        </div>
      </div>
      <div class="row">
        <button type="submit" class="btn primary" disabled={regBusy || regBlockers.length > 0} aria-describedby={regBlockers.length ? 'reg-why' : undefined}>
          {regBusy ? 'Registering…' : 'Register'}
        </button>
        {#if regBlockers.length}<span class="xs muted" id="reg-why">{regBlockers.join(' ')}</span>{/if}
      </div>
      <div aria-live="polite">
        {#if regError}<p class="note bad" role="alert">{regError}</p>{/if}
        {#if regDone}<p class="note info">{regDone}</p>{/if}
      </div>
    </form>
    </WorkflowGate>
  </section>

  <section class="panel" aria-labelledby="models-h">
    <header>
      <h2 id="models-h">Library entries <span class="muted num small">({models.length})</span></h2>
      <div class="filters">
        <label class="sr-only" for="lib-filter">Filter entries</label>
        <input id="lib-filter" class="input" type="search" bind:value={filter} placeholder="Filter…" autocomplete="off" />
        <Combobox bind:value={hostFilter} id="lib-host" label="Host filter" labelHidden optional optionalLabel="All hosts" options={hostOptions(hosts)} emptyText="No host matches." />
      </div>
    </header>
    {#if removeError}<p class="note bad" role="alert" style="margin:12px 16px 0">{removeError}</p>{/if}
    {#if models.length === 0}
      <p class="empty">No entries yet. Register a GGUF, checkpoint or bundle above.</p>
    {:else if shown.length === 0}
      <p class="empty">No entry matches the filter.</p>
    {:else}
      <div class="table-wrap">
        <table class="data stackable">
          <caption class="sr-only">Library entries{#if filter || hostFilter}{' '}matching the filter ({shown.length} of {models.length}){/if}</caption>
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Kind</th>
              <th scope="col">Host</th>
              <th scope="col">Path</th>
              <th scope="col">Identity</th>
              <th scope="col"><span class="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            {#each shown as m (m.id)}
              {@const busyJobs = usedBy(m)}
              {@const edit = WEIGHT_SOURCES.includes(m.kind) ? weightAccess(m, hosts.find((host) => host.id === m.host)) : null}
              <tr>
                <td>
                  <div class="name break">{modelLabel(m)}</div>
                </td>
                <td class="nowrap" data-label="Kind">{kindLabel(m.kind)}</td>
                <td class="nowrap" data-label="Host">{hostName(hosts, m.host)}</td>
                <td class="mono xs break path" data-label="Path">{m.path}</td>
                <td class="xs identity" data-label="Identity">
                  <Badge tone={m.available === true ? 'ok' : m.available === false ? 'warn' : 'muted'} label={m.available === true ? 'Available' : m.available === false ? 'Unavailable' : 'Unknown'} />
                  {#if m.reason || m.available !== true}<div class="muted break">{m.reason || (m.available === false ? 'The source file is not available.' : 'The controller has not checked this file.')}</div>{/if}
                  {#if m.sha256}<div class="break">SHA-256 <span class="mono">{m.sha256}</span></div>{:else}<div class="muted">Not yet hashed</div>{/if}
                  {#if typeof m.bytes === 'number'}<div class="num muted" title={`${m.bytes} bytes`}>{formatBytes(m.bytes)}</div>{/if}
                  {#if m.job_id}<a class="break" href="#/jobs/{encodeURIComponent(m.job_id)}">Output of job <span class="mono">{m.job_id}</span></a>{/if}
                </td>
                <td class="actions">
                  {#if removing === m.id}
                    <div class="confirm" role="group" aria-label="Confirm removal">
                      <span class="xs">This removes only the entry. The files stay on the host.{#if m.host === 'backend-hf'} You cannot register it again by hand. To add the source again, start a verified download or submit the job from its view.{/if}</span>
                      <div class="row">
                        <button type="button" class="btn danger small" aria-disabled={removeBusy || busyJobs.length > 0} onclick={() => remove(m)}>{removeBusy ? 'Removing…' : 'Remove entry'}</button>
                        <button id="keep-entry" type="button" class="btn ghost small" aria-disabled={removeBusy} onclick={() => {
                          if (!removeBusy) void focusAfterRemoval(() => { removing = null; }, () => document.getElementById(removeButtonId(m.id)));
                        }}>Keep</button>
                      </div>
                    </div>
                  {:else}
                    {#if edit?.allowed}
                      <a class="btn ghost small" href="#/weights?source={encodeURIComponent(m.id)}" aria-label="Edit weights of {modelLabel(m)}">Edit weights</a>
                    {:else if edit}
                      <div class="xs muted edit-reason">
                        <button type="button" class="btn ghost small" disabled aria-describedby="edit-why-{m.id}">Edit weights</button>
                        <div id="edit-why-{m.id}">{edit.reason}</div>
                        {#if m.available !== false}<a class="btn ghost small" href={edit.fixAnchor}>Host settings</a>{/if}
                      </div>
                    {/if}
                    <button
                      id={removeButtonId(m.id)}
                      type="button"
                      class="btn ghost small"
                      disabled={busyJobs.length > 0 || removeBusy}
                      title={busyJobs.length ? `Used by active job ${busyJobs[0].id}` : undefined}
                      aria-label="Remove {modelLabel(m)} from library"
                      onclick={() => {
                        void focusAfterRemoval(() => {
                          removing = m.id;
                          removeError = null;
                        }, () => document.getElementById('keep-entry'));
                      }}>Remove…</button
                    >
                    {#if busyJobs.length}<div class="xs muted">In use by an active job</div>{/if}
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>

  <div class="grid-2">
    <section class="panel" aria-labelledby="prompt-h">
      <header><h2 id="prompt-h">{editing ? 'Edit prompt' : 'Save a prompt'}</h2></header>
      <form class="body stack" onsubmit={savePrompt} aria-busy={pBusy}>
        <div class="field">
          <label for="p-name">Name</label>
          <input id="p-name" class="input" bind:value={pName} autocomplete="off" required />
        </div>
        <div class="field">
          <label for="p-text">Prompt text</label>
          <textarea id="p-text" class="input" bind:value={pText} rows="5" required></textarea>
          <span class="hint num">{pText.length.toLocaleString()} characters. Saved exactly as typed.</span>
        </div>
        <div class="row">
          {#if editing}
            <button type="submit" class="btn primary" aria-disabled={pBusy || !pName.trim() || !pText}>{pBusy ? 'Saving…' : 'Save changes'}</button>
            <button type="button" class="btn ghost" disabled={pBusy} onclick={cancelPromptEdit}>Cancel edit</button>
          {:else}
            <button type="submit" class="btn primary" disabled={pBusy || !pName.trim() || !pText}>{pBusy ? 'Saving…' : 'Save prompt'}</button>
          {/if}
        </div>
        <div aria-live="polite">{#if pError}<p class="note bad" role="alert">{pError}</p>{/if}</div>
      </form>
    </section>

    <section class="panel" aria-labelledby="prompts-h">
      <header><h2 id="prompts-h" tabindex="-1">Saved prompts <span class="muted num small">({prompts.length})</span></h2></header>
      {#if prompts.length === 0}
        <p class="empty">No saved prompts.</p>
      {:else}
        <ul class="prompts">
          {#each prompts as p (p.id)}
            <li>
              <button id={promptToggleId(p.id)} type="button" class="ptoggle" aria-expanded={openPrompt === p.id} aria-disabled={deleteBusy} onclick={() => {
                if (deleteBusy) return;
                openPrompt = openPrompt === p.id ? null : p.id;
                deleting = null;
                deleteError = null;
              }}>
                <span class="truncate">{p.name}</span>
                <span class="xs muted truncate">{p.text.slice(0, 80)}</span>
              </button>
              {#if openPrompt === p.id}
                <div class="body stack">
                  <pre class="block">{p.text}</pre>
                  {#if deleting === p.id}
                    <div class="stack" role="group" aria-label="Confirm prompt deletion">
                      <p class="small">Delete this saved prompt? You cannot undo this action.</p>
                      {#if deleteError}<p class="note bad" role="alert">{deleteError}</p>{/if}
                      <div class="row">
                        <button type="button" class="btn danger small" aria-disabled={deleteBusy} onclick={() => deletePrompt(p.id)}>{deleteBusy ? 'Deleting…' : 'Delete prompt'}</button>
                        <button id="keep-prompt" type="button" class="btn ghost small" aria-disabled={deleteBusy} onclick={() => {
                          if (!deleteBusy) void focusAfterRemoval(() => { deleting = null; }, () => document.getElementById('delete-prompt'));
                        }}>Keep prompt</button>
                      </div>
                    </div>
                  {:else}
                    <div class="row">
                      <button type="button" class="btn small" disabled={pBusy} onclick={() => editPrompt(p)}>Edit prompt</button>
                      <button id="delete-prompt" type="button" class="btn ghost small" disabled={editing?.id === p.id} onclick={() => {
                        void focusAfterRemoval(() => { deleting = p.id; deleteError = null; }, () => document.getElementById('keep-prompt'));
                      }}>Delete prompt</button>
                    </div>
                  {/if}
                </div>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}
    </section>
  </div>
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .reg {
    display: grid;
    grid-template-columns: minmax(140px, 0.8fr) minmax(220px, 2.2fr) minmax(160px, 1fr) minmax(140px, 1fr);
    gap: var(--s-3);
    align-items: start;
  }
  @media (max-width: 1000px) {
    .reg {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }
    .grow {
      grid-column: 1 / -1;
    }
  }
  @media (max-width: 560px) {
    .reg {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .filters {
    display: flex;
    gap: var(--s-2);
  }
  .filters :global(.input) {
    min-height: 32px;
    width: 180px;
    max-width: 100%;
  }
  .name {
    font-weight: 500;
  }
  .path {
    max-width: 360px;
    min-width: 160px;
  }
  .actions {
    text-align: right;
  }
  .edit-reason {
    max-width: 220px;
    margin-left: auto;
  }
  .identity {
    min-width: 125px;
  }
  @media (max-width: 640px) {
    .actions, .edit-reason {
      text-align: left;
      margin-left: 0;
    }
    .confirm {
      align-items: flex-start;
      margin-left: 0;
      max-width: none;
    }
  }
  .confirm {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
    white-space: normal;
    max-width: 220px;
    margin-left: auto;
  }
  .prompts {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .prompts li {
    border-bottom: 1px solid var(--line);
  }
  .prompts li:last-child {
    border-bottom: none;
  }
  .ptoggle {
    display: flex;
    flex-direction: column;
    gap: 2px;
    width: 100%;
    padding: 10px var(--s-4);
    background: none;
    border: none;
    text-align: left;
    cursor: pointer;
  }
  .ptoggle:hover {
    background: var(--surface-2);
  }
  .prompts pre {
    margin: 0;
    overflow-wrap: anywhere;
  }
  @media (max-width: 760px) {
    header {
      flex-wrap: wrap;
    }
    .filters {
      width: 100%;
    }
    .filters .input {
      max-width: none;
      flex: 1;
      min-width: 0;
    }
  }
</style>

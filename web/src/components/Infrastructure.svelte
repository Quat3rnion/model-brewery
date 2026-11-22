<script lang="ts">
  import { api, ApiError, errorText } from '../lib/api';
  import { onMount, tick } from 'svelte';
  import { adminActionAllowed, parseAdminState, type AdminAction, type AdminState } from '../lib/admin';
  import { focusAfterRemoval } from '../lib/focus';
  import { formatTime } from '../lib/format';
  import { brewery } from '../lib/store.svelte';
  import type { PolicyAction } from '../lib/host-policy';
  import WeightEditorSettings from './WeightEditorSettings.svelte';
  import ModelDirectories from './ModelDirectories.svelte';
  import { directoryReceipt, type DirectoryChange } from '../lib/model-directories';
  import HostPolicy from './HostPolicy.svelte';
  import SshHosts from './SshHosts.svelte';
  import Badge from '../components/Badge.svelte';

  let { hostId, selectHost, confirmRefresh, trackDraft, focusRequest, clearFocus, settingsLoaded }:
    { hostId: string; selectHost: (id: string) => boolean; confirmRefresh: () => boolean;
      trackDraft: (id: string, host: string, dirty: boolean) => void; focusRequest: string;
      clearFocus: () => void; settingsLoaded: boolean } = $props();
  let st = $state<AdminState | null>(null);
  let status = $state<'idle' | 'loading' | 'ok' | 'unavailable' | 'error'>('idle');
  let error = $state<string | null>(null);
  let busy = $state<string | null>(null);
  let actionError = $state<string | null>(null);
  let actionNote = $state('');
  let visitedHosts = $state<string[]>([]);
  let directoryStatus = $state<{host:string;message:string;error?:boolean}|null>(null);
  let mounted = false;
  const sections: Record<string, string> = { infrastructure: 'infrastructure', weights: 'weight-settings-h',
    policy: 'host-policy-h', directories: 'model-directories-h' };
  const policy = $derived(hostId ? st?.hosts?.find(h => h.id === hostId) : st?.hosts?.[0]);
  $effect(() => {
    if (settingsLoaded && !hostId && st) {
      const firstHost = policy?.id ?? st.workers[0]?.id;
      if (firstHost) selectHost(firstHost);
    }
    if (policy && !visitedHosts.includes(policy.id)) visitedHosts = [...visitedHosts, policy.id];
  });
  $effect(() => {
    if (!focusRequest || !Object.hasOwn(sections, focusRequest)) return;
    if (focusRequest !== 'infrastructure' && (!settingsLoaded || !st || (policy && !visitedHosts.includes(policy.id)))) return;
    const id = sections[focusRequest];
    void tick().then(() => {
      if (sections[focusRequest] !== id) return;
      (document.getElementById(id) ?? document.getElementById('infrastructure'))?.focus();
      clearFocus();
    });
  });
  let confirmDeactivate = $state<string | null>(null);
  let reviewPool = $state<{ worker: string; allow: boolean } | null>(null);
  let reviewButton = $state<HTMLButtonElement>();
  let deactivateButton = $state<HTMLButtonElement>();
  const selectedWorker = $derived(st?.workers.find(w => w.id === hostId));
  const blocked = $derived(!!busy || status !== 'ok');

  async function load() {
    if (busy || status === 'loading') return;
    status = 'loading';
    try {
      const s = parseAdminState(await api.adminState());
      if (!s) throw new Error('The controller answered without admin state.');
      st = s;
      status = 'ok';
      error = null;
    } catch (e) {
      handle(e);
    }
  }
  function refresh() { if (confirmRefresh()) void load(); }
  function handle(e: unknown) {
    if (e instanceof ApiError && (e.status === 503 || e.status === 404 || e.status === 405)) {
      status = 'unavailable';
      error = e.status === 503 ? errorText(e) : 'Infrastructure settings are not available on this controller.';
    } else {
      status = 'error';
      error = errorText(e);
    }
  }
  onMount(() => {
    mounted = true;
    void load();
    return () => { mounted = false; };
  });

  async function saveDirectory(change:DirectoryChange) {
    if (blocked || !policy?.directories?.available || policy.id !== change.worker || policy.revision !== change.revision) return false;
    busy='directory';
    directoryStatus={host:change.worker,message:'Checking the directory change…'};
    try {
      const receipt=directoryReceipt(await api.planDirectory(change),change);
      if (!mounted || !receipt || policy?.id!==change.worker || policy.revision!==change.revision) throw new Error('The validation did not match the reviewed change. Refresh infrastructure before trying again.');
      const result=parseAdminState(await api.confirmDirectory({worker:change.worker, revision:change.revision, confirmation:receipt.confirmation}));
      if (!result?.hosts?.some(h=>h.id===change.worker && h.revision>change.revision)) throw new Error('Refresh infrastructure to check whether the change was saved.');
      void focusAfterRemoval(() => { st = result; }, () => document.getElementById('model-directories-h'));
      directoryStatus={host:change.worker,message:`Directory change saved for ${change.worker}.`};
      void brewery.refresh();
      return true;
    } catch(e) {
      directoryStatus={host:change.worker,error:true,message:`${errorText(e)}${e instanceof ApiError && e.status===409 ? ' Refresh infrastructure, review the current paths, then save again.' : ''}`};
      return false;
    } finally { busy=null; }
  }

  async function act(a: AdminAction | PolicyAction, label: string): Promise<boolean> {
    if (!st || blocked) return false;
    busy = label;
    const enrollmentAction = ['set_resource_policy', 'enroll', 'approve', 'deactivate', 'install_runtime', 'enable_runtime', 'disable_runtime'].includes(a.action);
    actionError = null;
    actionNote = '';
    try {
      const s = parseAdminState(await api.adminAction(a));
      if (!s) throw new Error('Invalid infrastructure settings response. Refresh to check the outcome.');
      await focusAfterRemoval(() => {
        st = s;
        confirmDeactivate = null;
        reviewPool = null;
      }, () => document.getElementById('wk-h'));
      if (enrollmentAction && document.activeElement === document.body) document.getElementById('wk-h')?.focus();
      actionNote = a.action === 'set_resource_policy' ? `Pool membership saved for ${a.worker}.`
        : a.action === 'set_host_policy' ? `Admission policy saved for ${a.worker}.`
        : a.action === 'set_host_resources' ? `Resource limits saved for ${a.worker}.`
        : `${label}: complete.`;
      void brewery.refresh();
      return true;
    } catch (e) {
      if (e instanceof ApiError && [401,403].includes(e.status)) handle(e);
      else actionError = `${label}: ${errorText(e)}${e instanceof ApiError && e.status === 409 ? ' Refresh infrastructure and review the latest settings before retrying.' : ''}`;
      return false;
    } finally {
      busy = null;
    }
  }
  async function reviewMembership(worker: string, allowed: boolean) {
    reviewPool = { worker, allow: !allowed };
    await tick();
    reviewButton?.focus();
  }
  function confirmMembership() {
    if (!reviewPool || !selectedWorker || reviewPool.worker !== selectedWorker.id || reviewPool.allow === selectedWorker.allow_pool) return;
    const action = { action: 'set_resource_policy', worker: reviewPool.worker, allow_pool: reviewPool.allow } as const;
    if (!adminActionAllowed(selectedWorker, action)) void act(action, `Change pool membership for ${action.worker}`);
  }
  const STATUS_TONE: Record<string, 'ok' | 'warn' | 'muted' | 'info'> = { active: 'ok', pending: 'warn', unregistered: 'muted', inactive: 'muted' };
</script>

<section class="stack" aria-labelledby="infrastructure">
  <div class="head">
    <h2 id="infrastructure" tabindex="-1">Infrastructure</h2>
    <p class="muted small">Review admission and resource limits for each host. Manage worker enrollment, runtimes and conversion pool membership. The audit log records changes.</p>
  </div>
  <SshHosts {confirmRefresh} onDirty={(dirty) => trackDraft('ssh', '', dirty)} />
  <div><button type="button" class="btn small" onclick={refresh} disabled={!!busy || status === 'loading'}>Refresh infrastructure</button></div>
    {#if status === 'loading' && !st}<p class="muted">Loading…</p>{/if}
    {#if status === 'unavailable'}<p class="note warn">Infrastructure settings unavailable: {error}</p>{/if}
    {#if status === 'error'}<div class="note bad" role="alert"><p>{error}</p><button type="button" class="btn small" onclick={refresh}>Retry</button></div>{/if}
    {#if actionError}<p class="note bad" role="alert">{actionError}</p>{/if}
    {#if actionNote}<p class="note info" role="status">{actionNote}</p>{/if}

    {#if st}
      <section class="panel" aria-labelledby="host-controls-h">
        <header><h3 id="host-controls-h">Host controls</h3></header>
        <div class="body stack">
          {#if st.hosts?.length}
            <div class="field"><label for="infra-policy-host">Defined host</label>
              <select id="infra-policy-host" class="input" value={hostId || policy?.id} disabled={blocked}
                onchange={(e) => { if (!selectHost(e.currentTarget.value)) e.currentTarget.value = hostId; }}>
                {#if hostId && !policy}<option value={hostId}>{hostId} (not defined)</option>{/if}
                {#each st.hosts as h (h.id)}<option value={h.id}>{h.name === h.id ? h.id : `${h.name} · ${h.id}`}</option>{/each}
              </select>
            </div>
            {#each st.hosts.filter(h => visitedHosts.includes(h.id)) as host (host.id)}
              {@const active = host.id === policy?.id}
              <div class:stack={active} hidden={!active}>
                <HostPolicy {host} {active} busy={blocked} save={act}
                  onDirty={(dirty) => trackDraft(`policy:${host.id}`, host.id, dirty)} />
                <WeightEditorSettings {host} {active} policies={st.hosts} busy={blocked} save={act}
                  onDirty={(dirty) => trackDraft(`editor:${host.id}`, host.id, dirty)} />
                <ModelDirectories {host} {active} busy={blocked} save={saveDirectory}
                  onDirty={(dirty) => trackDraft(`directory:${host.id}`, host.id, dirty)} />
                {#if directoryStatus?.host===host.id}<p class="note small" class:bad={directoryStatus.error} role={directoryStatus.error?'alert':'status'}>{directoryStatus.message}</p>{/if}
              </div>
            {/each}
          {/if}
          {#if !policy}<p class="muted small">{hostId && st.hosts ? `The selected host (${hostId}) is not defined here.` : st.hosts ? 'No defined hosts reported.' : 'This controller did not report editable host policies.'}</p>{/if}
        </div>
      </section>
      <section class="panel" aria-labelledby="wk-h">
        <header><h3 id="wk-h" tabindex="-1">Runtime enrollment</h3></header>
        {#if !selectedWorker}<p class="empty">The selected host is not in the runtime enrollment catalog.</p>{/if}
        <ul class="workers">
          {#each selectedWorker ? [selectedWorker] : [] as w (w.id)}
            <li class="w">
              <div class="wh">
                <span class="wn">{w.name ?? w.id}</span>
                {#if w.name && w.name !== w.id}<span class="mono xs muted">{w.id}</span>{/if}
                <Badge tone={STATUS_TONE[w.status] ?? 'muted'} label={w.status} />
              </div>
              <div class="row acts">
                {#each [['enroll', 'Enroll', w.can_enroll], ['approve', 'Approve', w.can_approve]] as const as [a, label, can] (a)}
                  {#if can}<button type="button" class="btn small" disabled={blocked} onclick={() => act({ action: a, worker: w.id }, `${label} ${w.id}`)}>{busy === `${label} ${w.id}` ? `${label}…` : label}</button>{/if}
                {/each}
                {#if w.can_deactivate}
                  {#if confirmDeactivate === w.id}
                    <button bind:this={deactivateButton} type="button" class="btn small danger" disabled={blocked} onclick={() => act({ action: 'deactivate', worker: w.id }, `Deactivate ${w.id}`)}>Confirm deactivate</button>
                    <button type="button" class="btn small ghost" onclick={() => focusAfterRemoval(() => confirmDeactivate = null, () => document.getElementById('deactivate-worker'))}>Keep active</button>
                  {:else}
                    <button id="deactivate-worker" type="button" class="btn small danger" disabled={blocked} onclick={async () => { confirmDeactivate = w.id; await tick(); deactivateButton?.focus(); }}>Deactivate</button>
                  {/if}
                {/if}
                {#if !w.can_enroll && !w.can_approve && !w.can_deactivate}<span class="xs muted">No enrollment action is possible now.</span>{/if}
              </div>
              <p class="small"><strong>Conversion pool:</strong> {w.allow_pool ? 'Member' : 'Not a member'}</p>
              <span class="xs muted" id="pool-help">{!w.pool_ceiling && !w.allow_pool ? adminActionAllowed(w, { action: 'set_resource_policy', worker: w.id, allow_pool: true }) : 'The controller can assign conversions to pool members when you do not specify a worker.'}</span>
              {#if reviewPool?.worker === w.id && reviewPool.allow !== w.allow_pool}
                <div class="note stack">
                  <p>{reviewPool.allow ? 'Add' : 'Remove'} {w.name ?? w.id} {reviewPool.allow ? 'to' : 'from'} the conversion pool?</p>
                  <div class="row"><button bind:this={reviewButton} class="btn primary" disabled={blocked} onclick={confirmMembership}>Confirm pool membership</button>
                    <button class="btn" disabled={blocked} onclick={() => focusAfterRemoval(() => reviewPool = null, () => document.getElementById('review-pool'))}>Cancel pool change</button></div>
                </div>
              {:else if w.allow_pool || w.pool_ceiling}
                <div><button id="review-pool" class="btn small" disabled={blocked} onclick={() => reviewMembership(w.id, w.allow_pool)} aria-describedby="pool-help">Review pool membership</button></div>
              {/if}
              {#if w.runtimes.length}
                <div class="table-wrap">
                  <table class="data">
                    <caption class="sr-only">Runtimes on {w.name ?? w.id}</caption>
                    <thead><tr><th scope="col">Runtime</th><th scope="col">State</th><th scope="col"><span class="sr-only">Action</span></th></tr></thead>
                    <tbody>
                      {#each w.runtimes as r (r.id)}
                        {@const why = adminActionAllowed(w, { action: r.enabled ? 'disable_runtime' : 'enable_runtime', worker: w.id, runtime: r.id })}
                        {@const installWhy = adminActionAllowed(w, { action: 'install_runtime', worker: w.id, runtime: r.id })}
                        <tr>
                          <th scope="row">{r.label ?? r.id}{#if r.label && r.label !== r.id}{' '}<span class="mono xs muted">{r.id}</span>{/if}</th>
                          <td class="st"><Badge tone={r.enabled ? 'ok' : 'muted'} label={r.enabled ? 'Enabled' : 'Disabled'} /><Badge tone={r.installed ? 'ok' : 'muted'} label={r.installed ? 'Installed' : 'Not installed'} />{#if !r.available}{' '}<span class="xs muted">{r.reason ?? 'Not available on the worker'}</span>{/if}</td>
                          <td class="r">
                            {#if !r.installed && !installWhy}
                              <button type="button" class="btn small" disabled={blocked} onclick={() => act({ action: 'install_runtime', worker: w.id, runtime: r.id }, `Install ${r.id} on ${w.id}`)}>{busy === `Install ${r.id} on ${w.id}` ? 'Installing…' : 'Install'}</button>
                            {:else if why && r.available}<span class="xs muted">{why}</span>{:else if why}<span class="xs muted">Cannot enable</span>
                            {:else}<button type="button" class="btn small" disabled={blocked} onclick={() => act({ action: r.enabled ? 'disable_runtime' : 'enable_runtime', worker: w.id, runtime: r.id }, `${r.enabled ? 'Disable' : 'Enable'} ${r.id} on ${w.id}`)}>{r.enabled ? 'Disable' : 'Enable'}</button>{/if}
                          </td>
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
                <p class="xs muted">Install runs only the operator-provisioned helper on the worker for that runtime; it accepts no paths or commands. Enabling uses an installed runtime and installs nothing. Install results appear in the audit log.</p>
              {/if}
            </li>
          {/each}
        </ul>
      </section>

      {#if st.unavailable.length}
        <section class="panel" aria-labelledby="na-h">
          <header><h3 id="na-h">Not available from the browser</h3></header>
          <ul class="body na">{#each st.unavailable as u (u.what)}<li><strong>{u.what}:</strong> <span class="muted">{u.reason}</span></li>{/each}</ul>
        </section>
      {/if}

      <section class="panel" aria-labelledby="au-h">
        <header><h3 id="au-h">Audit log</h3></header>
        {#if st.audit.length === 0}<p class="empty">No infrastructure actions yet.</p>
        {:else}
          <!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users must be able to scroll the audit table.) -->
          <div class="table-wrap" role="region" aria-label="Audit table" tabindex="0">
            <table class="data">
              <caption class="sr-only">Infrastructure actions, newest first</caption>
              <thead><tr><th scope="col">Time</th><th scope="col">Action</th><th scope="col">Worker</th><th scope="col">Detail</th><th scope="col">Outcome</th></tr></thead>
              <tbody>
                {#each st.audit as a (a.id)}
                  <tr><td class="xs num">{a.at ? formatTime(a.at) : '—'}</td><td>{a.action.replace(/_/g, ' ')}</td><td class="mono xs">{a.worker ?? '—'}</td><td class="xs break">{a.detail ?? '—'}</td><td>{#if a.outcome}<Badge tone={a.outcome === 'applied' ? 'ok' : a.outcome === 'failed' ? 'bad' : 'info'} label={a.outcome} />{:else}<span class="xs muted">—</span>{/if}</td></tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </section>
    {/if}
</section>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .workers {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .w {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    padding: var(--s-3) var(--s-4);
    border-bottom: 1px solid var(--line-soft);
  }
  .w:last-child {
    border-bottom: none;
  }
  .wh {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 10px;
  }
  .wn {
    font-weight: 600;
  }
  .r {
    text-align: right;
  }
  .st {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 6px;
  }
  th[scope='row'] {
    color: var(--text);
    background: transparent;
    position: static;
    font-weight: 500;
  }
  .na {
    margin: 0;
    padding-left: 2.2em;
    font-size: var(--fs-sm);
  }
</style>

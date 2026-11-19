<script lang="ts">
  // Infrastructure admin. Separate from the shared workbench login: every
  // request carries the admin key, checked by the server. Only actions the
  // backend implements and marks possible are offered.
  import { api, ApiError, errorText } from '../lib/api';
  import { adminSession } from '../lib/admin.svelte';
  import { adminActionAllowed, parseAdminState, type AdminAction, type AdminState } from '../lib/admin';
  import { formatTime } from '../lib/format';
  import { brewery } from '../lib/store.svelte';
  import Badge from '../components/Badge.svelte';
  import AdminKey from '../components/AdminKey.svelte';

  let st = $state<AdminState | null>(null);
  let status = $state<'idle' | 'loading' | 'ok' | 'unavailable' | 'error'>('idle');
  let error = $state<string | null>(null);
  let busy = $state<string | null>(null);
  let actionError = $state<string | null>(null);
  let confirmDeactivate = $state<string | null>(null);

  async function load() {
    const key = adminSession.key;
    if (!key) return;
    status = 'loading';
    try {
      const s = parseAdminState(await api.adminState(key));
      if (!s) throw new Error('The controller answered without admin state.');
      st = s;
      status = 'ok';
      error = null;
    } catch (e) {
      handle(e);
    }
  }
  function handle(e: unknown) {
    if (e instanceof ApiError && e.status === 401) {
      adminSession.clear();
      st = null;
      status = 'idle';
      error = 'The admin key was refused. Enter it again.';
    } else if (e instanceof ApiError && (e.status === 503 || e.status === 404 || e.status === 405)) {
      status = 'unavailable';
      error = e.status === 503 ? errorText(e) : 'This controller has no admin interface.';
    } else {
      status = 'error';
      error = errorText(e);
    }
  }
  $effect(() => {
    if (adminSession.key) void load();
  });

  async function act(a: AdminAction, label: string) {
    const key = adminSession.key;
    if (!key || busy) return;
    busy = label;
    actionError = null;
    try {
      const s = parseAdminState(await api.adminAction(key, a));
      if (s) st = s;
      confirmDeactivate = null;
      void brewery.refresh();
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) handle(e);
      else actionError = `${label}: ${errorText(e)}`;
    } finally {
      busy = null;
    }
  }
  const STATUS_TONE: Record<string, 'ok' | 'warn' | 'muted' | 'info'> = { active: 'ok', pending: 'warn', unregistered: 'muted', inactive: 'muted' };
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Infrastructure admin</h1>
    <p class="muted small">
      Worker enrollment, runtimes and pool policy. This is separate from the shared workbench login: the server checks the admin key on every request. Actions change controller configuration only; none of them starts a job, installs software or runs commands.
      <a href="#/settings">Back to Settings</a>
    </p>
  </div>

  {#if !adminSession.key}
    <section class="panel"><div class="body"><AdminKey purpose="Enter the infrastructure admin key to continue." /></div></section>
    {#if error}<p class="note bad" role="alert">{error}</p>{/if}
  {:else}
    <div class="row">
      <button type="button" class="btn small" onclick={load} disabled={status === 'loading'}>Refresh</button>
      <button type="button" class="btn small ghost" onclick={() => { adminSession.clear(); st = null; status = 'idle'; }}>Forget admin key</button>
    </div>
    {#if status === 'loading' && !st}<p class="muted">Loading…</p>{/if}
    {#if status === 'unavailable'}<p class="note warn">Admin is not available: {error}</p>{/if}
    {#if status === 'error'}<div class="note bad" role="alert"><p>{error}</p><button type="button" class="btn small" onclick={load}>Retry</button></div>{/if}
    {#if actionError}<p class="note bad" role="alert">{actionError}</p>{/if}

    {#if st}
      <section class="panel" aria-labelledby="wk-h">
        <header><h2 id="wk-h">Workers</h2></header>
        {#if st.workers.length === 0}<p class="empty">No workers.</p>{/if}
        <ul class="workers">
          {#each st.workers as w (w.id)}
            <li class="w">
              <div class="wh">
                <span class="wn">{w.name ?? w.id}</span>
                <span class="mono xs muted">{w.id}</span>
                <Badge tone={STATUS_TONE[w.status] ?? 'muted'} label={w.status} />
              </div>
              <div class="row acts">
                {#each [['enroll', 'Enroll', w.can_enroll], ['approve', 'Approve', w.can_approve]] as const as [a, label, can] (a)}
                  {#if can}<button type="button" class="btn small" disabled={!!busy} onclick={() => act({ action: a, worker: w.id }, `${label} ${w.id}`)}>{busy === `${label} ${w.id}` ? `${label}…` : label}</button>{/if}
                {/each}
                {#if w.can_deactivate}
                  {#if confirmDeactivate === w.id}
                    <button type="button" class="btn small danger" disabled={!!busy} onclick={() => act({ action: 'deactivate', worker: w.id }, `Deactivate ${w.id}`)}>Confirm deactivate</button>
                    <button type="button" class="btn small ghost" onclick={() => (confirmDeactivate = null)}>Keep active</button>
                  {:else}
                    <button type="button" class="btn small danger" disabled={!!busy} onclick={() => (confirmDeactivate = w.id)}>Deactivate</button>
                  {/if}
                {/if}
                {#if !w.can_enroll && !w.can_approve && !w.can_deactivate}<span class="xs muted">No enrollment action is possible now.</span>{/if}
              </div>
              <label class="chk">
                <input
                  type="checkbox"
                  checked={w.allow_pool}
                  disabled={!!busy || (!w.allow_pool && !w.pool_ceiling)}
                  aria-describedby="pool-{w.id}"
                  onchange={(e) => act({ action: 'set_resource_policy', worker: w.id, allow_pool: e.currentTarget.checked }, `Pool policy ${w.id}`)}
                />
                Member of the conversion pool
              </label>
              <span class="xs muted" id="pool-{w.id}">{!w.pool_ceiling && !w.allow_pool ? adminActionAllowed(w, { action: 'set_resource_policy', worker: w.id, allow_pool: true }) : 'The controller may assign "any pool worker" conversions to this worker.'}</span>
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
                          <th scope="row">{r.label ?? r.id}{#if r.label}{' '}<span class="mono xs muted">{r.id}</span>{/if}</th>
                          <td class="st"><Badge tone={r.enabled ? 'ok' : 'muted'} label={r.enabled ? 'Enabled' : 'Disabled'} /><Badge tone={r.installed ? 'ok' : 'muted'} label={r.installed ? 'Installed' : 'Not installed'} />{#if !r.available}{' '}<span class="xs muted">{r.reason ?? 'Not available on the worker'}</span>{/if}</td>
                          <td class="r">
                            {#if !r.installed && !installWhy}
                              <button type="button" class="btn small" disabled={!!busy} onclick={() => act({ action: 'install_runtime', worker: w.id, runtime: r.id }, `Install ${r.id} on ${w.id}`)}>{busy === `Install ${r.id} on ${w.id}` ? 'Installing…' : 'Install'}</button>
                            {:else if why && r.available}<span class="xs muted">{why}</span>{:else if why}<span class="xs muted">Cannot enable</span>
                            {:else}<button type="button" class="btn small" disabled={!!busy} onclick={() => act({ action: r.enabled ? 'disable_runtime' : 'enable_runtime', worker: w.id, runtime: r.id }, `${r.enabled ? 'Disable' : 'Enable'} ${r.id} on ${w.id}`)}>{r.enabled ? 'Disable' : 'Enable'}</button>{/if}
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
          <header><h2 id="na-h">Not available from the browser</h2></header>
          <ul class="body na">{#each st.unavailable as u (u.what)}<li><strong>{u.what}:</strong> <span class="muted">{u.reason}</span></li>{/each}</ul>
        </section>
      {/if}

      <section class="panel" aria-labelledby="au-h">
        <header><h2 id="au-h">Audit log</h2></header>
        {#if st.audit.length === 0}<p class="empty">No admin actions yet.</p>
        {:else}
          <div class="table-wrap">
            <table class="data">
              <caption class="sr-only">Admin actions, newest last</caption>
              <thead><tr><th scope="col">Time</th><th scope="col">Action</th><th scope="col">Worker</th><th scope="col">Detail</th><th scope="col">Outcome</th></tr></thead>
              <tbody>
                {#each st.audit as a (a.id)}
                  <tr><td class="xs num">{a.at ? formatTime(a.at) : '—'}</td><td>{a.action.replace(/_/g, ' ')}</td><td class="mono xs">{a.worker ?? '—'}</td><td class="xs">{a.runtime ?? (a.allow_pool !== null ? `pool ${a.allow_pool ? 'on' : 'off'}` : '')}</td><td>{#if a.outcome}<Badge tone={a.outcome === 'applied' ? 'ok' : a.outcome === 'failed' ? 'bad' : 'info'} label={a.outcome} />{:else}<span class="xs muted">—</span>{/if}</td></tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </section>
    {/if}
  {/if}
</div>

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
  .chk {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 32px;
    font-size: var(--fs-sm);
    cursor: pointer;
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

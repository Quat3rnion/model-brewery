<script lang="ts">
  // Operator approval of one publish request. The dialog repeats the exact
  // SHA-256 and the check results; the request sends that hash, so a
  // revision that changed after the page loaded is refused by the controller.
  import Badge from '../Badge.svelte';
  import { api, errorText } from '../../lib/api';
  import { newKey } from '../../lib/conversion';
  import { approvalState, checkRows, type ScratchRevision } from '../../lib/weightspace';

  let { rev, onapproved }: { rev: ScratchRevision; onapproved?: (r: ScratchRevision) => void } = $props();

  const uid = $props.id();
  const gate = $derived(approvalState(rev));
  const checks = $derived(checkRows(rev.check_bundle));
  const failing = $derived(checks.filter((c) => c.status !== 'passed'));
  let dialog: HTMLDialogElement | undefined = $state();
  let opener: HTMLButtonElement | undefined = $state();
  let busy = $state(false);
  let error = $state<string | null>(null);
  // One key per opened dialog: a retry after a lost response does not approve twice.
  let key = '';

  function open() {
    error = null;
    key = newKey().replace(/^conv-/, 'approve-');
    dialog?.showModal();
  }
  function close() {
    dialog?.close();
  }
  async function approve() {
    busy = true;
    error = null;
    try {
      const r = (await api.approveScratch(rev.id, rev.sha256, key)) as ScratchRevision;
      close();
      onapproved?.(r);
    } catch (e) {
      error = errorText(e);
    } finally {
      busy = false;
    }
  }
</script>

<div class="approve">
  <button type="button" class="btn primary" bind:this={opener} disabled={!gate.can} aria-describedby={gate.can ? undefined : `${uid}-why`} onclick={open}>Approve publication…</button>
  {#if !gate.can}<span class="xs muted" id="{uid}-why">{gate.reason}</span>{/if}
</div>

<dialog bind:this={dialog} aria-labelledby="{uid}-h" aria-describedby="{uid}-d" onclose={() => opener?.focus()}>
  <form method="dialog" class="stack" onsubmit={(e) => e.preventDefault()}>
    <h2 id="{uid}-h">Approve publication</h2>
    <p id="{uid}-d" class="small">
      This registers the exact revision below as a SafeTensors model in the Library. It does not qualify or deploy the model. The model stays not release-qualified.
    </p>
    <div class="field">
      <span>SHA-256</span>
      <code class="mono hash break">{rev.sha256}</code>
    </div>
    <div class="field">
      <span>Check bundle <span class="mono xs muted">{rev.check_bundle?.id ?? '—'}</span></span>
      <ul class="checks">
        {#each checks as c (c.name)}
          <li><Badge tone={c.tone} label="{c.name}: {c.label}" />{#if c.reason}<span class="xs muted">{c.reason}</span>{/if}</li>
        {/each}
      </ul>
    </div>
    {#if failing.length}
      <p class="note warn small">
        {failing.length} {failing.length === 1 ? 'check did' : 'checks did'} not pass.{#if rev.check_bundle?.acknowledged_at_ms}{' The requester acknowledged this bundle.'}{/if}{' '}Read the results before you approve.
      </p>
    {/if}
    {#if error}<p class="note bad" role="alert">{error}</p>{/if}
    <div class="row actions">
      <button type="button" class="btn ghost" disabled={busy} onclick={close}>Keep pending</button>
      <button type="button" class="btn primary" disabled={busy} aria-busy={busy} onclick={approve}>{busy ? 'Approving…' : 'Approve this hash'}</button>
    </div>
  </form>
</dialog>

<style>
  .approve {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px var(--s-3);
  }
  dialog {
    width: min(560px, calc(100vw - 2 * var(--s-4)));
    padding: var(--s-5);
    border: 1px solid var(--line);
    border-radius: var(--r-card);
    background: var(--surface);
    color: var(--text);
  }
  dialog::backdrop {
    background: rgba(0, 0, 0, 0.55);
  }
  h2 {
    font-size: var(--fs-lg);
    font-weight: 600;
  }
  .hash {
    display: block;
    padding: var(--s-2);
    background: var(--bg);
    border: 1px solid var(--line-soft);
    border-radius: var(--r-sm);
    font-size: var(--fs-xs);
  }
  .checks {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .checks li {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px var(--s-2);
  }
  .actions {
    justify-content: flex-end;
  }
</style>

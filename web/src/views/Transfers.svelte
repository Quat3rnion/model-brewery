<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { capability, hostName, kindLabel, modelLabel } from '../lib/format';
  import HostSelect from '../components/HostSelect.svelte';
  import Combobox from '../components/Combobox.svelte';
  import Blockers from '../components/Blockers.svelte';
  import WorkflowGate from '../components/WorkflowGate.svelte';
  import { hostOptions, visibleChecklist } from '../lib/pickers';
  import { globalJobEligibility, hostWorkflowEligibility } from '../lib/eligibility';
  import type { Blocker } from '../lib/blockers';
  import type { Host, NewJob } from '../lib/types';
  import SubmitStatus from '../components/SubmitStatus.svelte';
  import RecentJobs from '../components/RecentJobs.svelte';
  import { tick } from 'svelte';
  import { route } from '../lib/route.svelte';

  const models = $derived(brewery.state?.models ?? []);
  const hosts = $derived(brewery.state?.hosts ?? []);
  type CopyKind = 'transfer_preview' | 'transfer';
  const decision = (h: Host, kind: CopyKind) => hostWorkflowEligibility(brewery.state?.eligibility, h.id, kind);
  const allows = (h: Host, kind: CopyKind) => decision(h, kind)?.eligible ?? capability(h, kind).allowed;
  const destinationOptions = $derived(hostOptions(hosts.filter((h) => allows(h, 'transfer') || allows(h, 'transfer_preview'))));
  const setup = $derived.by(() => {
    const copy = globalJobEligibility(brewery.state, 'transfer');
    const preview = globalJobEligibility(brewery.state, 'transfer_preview');
    // Preview and copy share the form; either action can make it useful.
    if (copy?.eligible === false && preview?.eligible === false) {
      const reasons = [...copy.reasons, ...preview.reasons];
      return { eligible: false, reasons: [...new Map(reasons.map((r) => [`${r.text}\n${r.fix_anchor}`, r])).values()] };
    }
    return { eligible: true, reasons: [] };
  });

  let dest = $state('');
  let selected = $state<string[]>([]);
  const destHost = $derived(hosts.find((h) => h.id === dest));
  const candidates = $derived(models.filter((m) => m.host !== dest && m.available !== false));
  let txFilter = $state('');
  const listed = $derived(
    visibleChecklist(
      candidates.map((m) => ({ id: m.id, m, text: `${modelLabel(m)} ${m.name ?? ''} ${m.kind} ${kindLabel(m.kind)} ${m.host} ${hostName(hosts, m.host)} ${m.path} ${m.id}` })),
      txFilter,
      selected,
    ),
  );

  // Drop selections that are no longer in the library or are already on the destination.
  $effect(() => {
    const ok = new Set(candidates.map((m) => m.id));
    if (selected.some((id) => !ok.has(id))) {
      selected = selected.filter(id => ok.has(id));
    }
  });

  function blockers(): Blocker[] {
    const out: Blocker[] = [];
    if (!destHost) out.push({ text: 'Select a destination host.' });
    if (selected.length === 0) out.push({ text: 'Select at least one entry to copy.' });
    return out;
  }

  function hostBlockers(kind: CopyKind): Blocker[] {
    if (!destHost || allows(destHost, kind)) return [];
    const name = hostName(hosts, destHost.id);
    const reasons = decision(destHost, kind)?.reasons ?? [];
    const action = kind === 'transfer' ? 'Transfer' : 'Preview';
    if (reasons.length) {
      return reasons.map((r) => ({ host: name, text: `${action}: ${r.text}`, fixAnchor: r.fix_anchor }));
    }
    const reason = destHost.capability_reasons?.[kind];
    return [{ host: name, text: `${action}: ${reason?.text ?? capability(destHost, kind).reason ?? 'This action is not available.'}`, fixAnchor: reason?.fix_anchor }];
  }

  const copyBlockers = $derived(blockers());
  const previewHostBlockers = $derived(hostBlockers('transfer_preview'));
  const transferHostBlockers = $derived(hostBlockers('transfer'));
  const preview = new Submitter('transfers.preview');
  const transfer = new Submitter('transfers.transfer');

  // Cleanup
  let cleanHost = $state('');
  let digest = $state('');
  let confirmClean = $state(false);
  let confirmation: HTMLElement | undefined = $state();
  let cleanButton: HTMLButtonElement | undefined = $state();
  const cleaner = new Submitter('transfers.cleanup');
  const digestProblem = $derived(digest.trim() && !/^[0-9a-f]{64}$/.test(digest.trim()) ? 'Use a 64-character lowercase SHA-256 hex digest.' : null);
  const cleanBlockers = $derived.by(() => {
    const out: Blocker[] = [];
    if (!cleanHost) out.push({ text: 'Select a host.' });
    else {
      const c = capability(hosts.find((h) => h.id === cleanHost), 'cache_cleanup');
      if (!c.allowed && c.reason) out.push({ text: c.reason });
    }
    if (digestProblem) out.push({ text: digestProblem });
    return out;
  });

  const previewJob = $derived<NewJob>({ kind: 'transfer_preview', host: dest, input_ids: selected });
  const transferJob = $derived<NewJob>({ kind: 'transfer', host: dest, input_ids: selected });
  const cleanupJob = $derived<NewJob>({
    kind: 'cache_cleanup', host: cleanHost, input_ids: [],
    options: digest.trim() ? { bundle_sha256: digest.trim() } : undefined,
  });

  async function closeCleanup(visit: ReturnType<typeof route.capture>, restoreFocus: boolean) {
    confirmClean = false;
    await tick();
    if (restoreFocus && visit.current() && document.activeElement === document.body) {
      const resultLink = cleanButton?.closest('section')?.querySelector<HTMLElement>('a[href^="#/jobs/"]');
      (cleanButton?.disabled ? resultLink : cleanButton)?.focus();
    }
  }

  async function runCleanup() {
    const visit = route.capture();
    const group = confirmation;
    let stayed = group?.contains(document.activeElement) ?? false;
    const observe = (event: Event) => {
      if (event.target instanceof Node && !group?.contains(event.target)) stayed = false;
    };
    document.addEventListener('focusin', observe);
    document.addEventListener('pointerdown', observe);
    const job = await cleaner.submit(cleanupJob);
    document.removeEventListener('focusin', observe);
    document.removeEventListener('pointerdown', observe);
    if (job && visit.current() && confirmation === group) await closeCleanup(visit, stayed);
  }
</script>

<div class="stack">
  <div class="head">
    <h1 tabindex="-1">Transfers</h1>
    <p class="muted small">
      Copy library entries to a worker cache. Preview shows missing bytes and free space first. Interrupted copies resume; copies are hash-verified, and a verified copy on the worker is reused.
    </p>
  </div>

  <section class="panel" aria-labelledby="tx-h">
    <header><h2 id="tx-h">Copy to a host</h2></header>
    <WorkflowGate workflow="transfer" decision={setup}>
    <div class="body stack">
      <div class="grid-2">
        <Combobox id="tx-host" bind:value={dest} options={destinationOptions} label="Destination host" placeholder="Select or type a host" emptyText="No eligible destination matches." />
      </div>
      <fieldset class="pick">
        <legend class="small">Entries to copy</legend>
        {#if !dest}
          <p class="xs muted">Select a destination first.</p>
        {:else if candidates.length === 0}
          <p class="xs muted">No entry on another host. Register sources in Library.</p>
        {:else}
          <div class="tx-filter">
            <label class="sr-only" for="tx-filter">Filter entries to copy</label>
            <input id="tx-filter" class="input" type="search" bind:value={txFilter} placeholder="Filter by name, host or path…" autocomplete="off" spellcheck="false" aria-describedby="tx-filter-hint" />
            <span class="xs muted" id="tx-filter-hint" aria-live="polite">{selected.length} selected{txFilter.trim() ? ` · ${listed.length} of ${candidates.length} shown; selected entries stay listed` : ''}</span>
          </div>
          <ul>
            {#each listed as { m } (m.id)}
              <li>
                <label>
                  <input type="checkbox" value={m.id} bind:group={selected} />
                  <span class="entry">
                    <span>{modelLabel(m)}</span>
                    <span class="xs muted">{kindLabel(m.kind)} · from {hostName(hosts, m.host)}</span>
                  </span>
                </label>
              </li>
            {/each}
          </ul>
          {#if listed.length === 0}<p class="xs muted">No entry matches the filter.</p>{/if}
        {/if}
      </fieldset>
      <div class="row actions">
        <button type="button" class="btn" aria-describedby={copyBlockers.length || previewHostBlockers.length ? 'copy-blockers' : undefined} aria-busy={preview.busy}
          disabled={preview.busy || copyBlockers.length > 0 || previewHostBlockers.length > 0}
          onclick={() => preview.submit(previewJob)}>{preview.busy ? 'Submitting…' : 'Preview'}</button>
        <button type="button" class="btn primary" aria-describedby={copyBlockers.length || transferHostBlockers.length ? 'copy-blockers' : undefined} aria-busy={transfer.busy}
          disabled={transfer.busy || copyBlockers.length > 0 || transferHostBlockers.length > 0}
          onclick={() => transfer.submit(transferJob)}>{transfer.busy ? 'Submitting…' : 'Transfer'}</button>
      </div>
      {#if copyBlockers.length || previewHostBlockers.length || transferHostBlockers.length}
        <Blockers id="copy-blockers" blockers={[...copyBlockers, ...previewHostBlockers, ...transferHostBlockers]} />
      {/if}
      <SubmitStatus s={preview} />
      <SubmitStatus s={transfer} />
      <p class="xs muted">Preview and transfer are queued jobs. Open a job to see the preview result, progress and cancel.</p>
    </div>
    </WorkflowGate>
  </section>

  <RecentJobs kinds={['transfer_preview', 'transfer']} title="Recent previews and transfers" />

  <section class="panel" aria-labelledby="cl-h">
    <header><h2 id="cl-h">Cache cleanup</h2></header>
    <div class="body stack">
      <p class="small muted">Removes worker-owned cached copies only. Registered originals and produced bundles in the library are not touched.</p>
      <div class="grid-2">
        <HostSelect id="cl-host" bind:value={cleanHost} kind="cache_cleanup" />
        <div class="field">
          <label for="cl-digest">Bundle SHA-256 <span class="muted">(optional)</span></label>
          <input id="cl-digest" class="input mono" bind:value={digest} autocomplete="off" spellcheck="false" aria-invalid={digestProblem ? 'true' : undefined} aria-describedby="cl-digest-h" />
          <span class="hint" id="cl-digest-h">{digestProblem ?? 'Leave empty to remove every cached copy that no job uses.'}</span>
        </div>
      </div>
      {#if confirmClean}
        <div class="note warn" role="group" aria-label="Confirm cleanup" bind:this={confirmation}>
          <p class="small">Remove {digest.trim() ? 'this cached bundle' : 'all unused cached copies'} on {hostName(hosts, cleanHost)}? Later jobs copy the files again.</p>
          <div class="row" style="margin-top:8px">
          <button type="button" class="btn danger" disabled={cleaner.busy || cleanBlockers.length > 0} onclick={runCleanup}>{cleaner.busy ? 'Submitting…' : 'Queue cleanup'}</button>
            <button type="button" class="btn ghost" onclick={() => closeCleanup(route.capture(), !!confirmation?.contains(document.activeElement))}>Keep cache</button>
          </div>
        </div>
      {:else}
        <div class="row">
          <button type="button" class="btn" bind:this={cleanButton} aria-describedby={cleanBlockers.length ? 'cleanup-blockers' : undefined}
            disabled={cleanBlockers.length > 0} onclick={() => (confirmClean = true)}>Clean cache…</button>
        </div>
      {/if}
      <Blockers id="cleanup-blockers" blockers={cleanBlockers} />
      <SubmitStatus s={cleaner} />
    </div>
  </section>

  <RecentJobs kinds={['cache_cleanup']} title="Recent cleanups" limit={3} />
</div>

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .pick {
    border: 1px solid var(--line);
    border-radius: var(--r-md);
    padding: var(--s-2) var(--s-3) var(--s-3);
    margin: 0;
    min-width: 0;
  }
  .pick legend {
    padding: 0 4px;
    color: var(--text-2);
    font-weight: 500;
  }
  .tx-filter {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--s-2) var(--s-3);
    margin-bottom: var(--s-2);
  }
  .tx-filter .input {
    flex: 1 1 220px;
    min-height: 32px;
  }
  .pick ul {
    list-style: none;
    margin: 0;
    padding: 0;
    max-height: 280px;
    overflow: auto;
  }
  .pick li label {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: start;
    gap: 10px;
    padding: 6px 4px;
    min-height: 36px;
    cursor: pointer;
    border-radius: var(--r-sm);
  }
  .pick li label:hover {
    background: var(--surface-2);
  }
  .entry {
    display: flex;
    flex-direction: column;
    overflow-wrap: anywhere;
  }
  .actions {
    align-items: flex-start;
    gap: var(--s-3);
  }
</style>

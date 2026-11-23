<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { hashParam, route } from '../lib/route.svelte';
  import { loadRetryJob, RetryPrefill, watchPrefill } from '../lib/retry-prefill.svelte';
  import SubmitStatus from './SubmitStatus.svelte';
  import RetryOffer from './RetryOffer.svelte';
  import RetryRecovery from './RetryRecovery.svelte';
  import { unsavedChanges } from '../lib/unsaved-changes.svelte';
  import { hostName } from '../lib/format';
  import Submit from './Submit.svelte';
  import type { NewJob } from '../lib/types';

  const submitter = new Submitter('library.hf-download');
  let host = $state('backend-hf');
  const hosts = $derived(brewery.state?.hosts ?? []);
  const target = $derived(hosts.find((h) => h.id === host));
  let repo = $state('');
  let revision = $state('');
  let scope = $state('file');
  let filename = $state('');
  let open = $state(false);
  const defaults = { host: 'backend-hf', repo: '', revision: '', scope: 'file', filename: '' };
  const draft = () => JSON.stringify({ repo, revision, scope, filename });
  let savedDraft = $state(draft());
  unsavedChanges(() => draft() !== savedDraft);
  const retry = new RetryPrefill({
    slot: 'download', defaults,
    read: () => ({ host, repo, revision, scope, filename }),
    write: value => { ({ host, repo, revision, scope, filename } = value); },
    load: loadRetryJob(['hf_download']),
    restore: ({ options, requested_host, host }) => {
      const repo = typeof options?.repo_id === 'string' ? options.repo_id : '';
      const revision = typeof options?.revision === 'string' ? options.revision : '';
      const filename = typeof options?.filename === 'string' ? options.filename : '';
      const scope = filename ? 'file' : 'repo';
      return { actions: ['hf_download'], patch: { host: requested_host || host || defaults.host, repo, revision, filename, scope }, focus: 'hf-repo' };
    },
    applied: () => { savedDraft = draft(); },
    clear: () => { savedDraft = JSON.stringify({ repo: '', revision: '', scope: 'file', filename: '' }); },
  });
  watchPrefill(retry);
  $effect(() => {
    if (hashParam(route.hash, 'retry')) open = true;
  });
  const safePart = (s: string) => /^[A-Za-z0-9_][A-Za-z0-9._-]*$/.test(s) && !s.includes('..');
  const blockers = $derived.by(() => {
    const errors: string[] = [];
    const parts = repo.trim().split('/');
    if (parts.length !== 2 || !parts.every(safePart)) errors.push('Enter a repository ID in owner/model form.');
    if (revision.trim() && !revision.trim().split('/').every(safePart)) errors.push('Enter a branch, tag or commit, not a URL.');
    if (scope === 'file' && (!filename.trim().split('/').every(safePart) || !/\.(gguf|safetensors)$/i.test(filename.trim())))
      errors.push('Enter the exact .gguf or .safetensors filename from the repository.');
    if (target?.capabilities?.hf_download !== true || target.scheduling_paused)
      errors.push(`Downloads are not available on ${hostName(hosts, host)}.`);
    return errors;
  });
  const request = $derived<NewJob>({
    kind: 'hf_download', input_ids: [],
    host: host === 'backend-hf' ? undefined : host,
    options: { repo_id: repo.trim(), revision: revision.trim(), filename: scope === 'file' ? filename.trim() : undefined },
  });
  async function start() {
    if (submitter.busy || retry.loading || blockers.length) return;
    const visit = route.capture();
    const submitted = draft();
    const job = await submitter.submit(request, retry.action('hf_download'));
    if (job && visit.current()) {
      savedDraft = submitted;
      if (!submitter.replayed && visit.canPresent()) location.hash = `#/jobs/${encodeURIComponent(job.id)}`;
    }
  }
</script>

<details class="disclosure" bind:open>
  <summary><strong>Download from Hugging Face</strong></summary>
  <div class="stack fields" oninputcapture={retry.touchField} onchangecapture={retry.touchField}>
    {#if retry.loading}<p class="hint" role="status">Loading the previous download.</p>{/if}
    {#if retry.failed}<RetryRecovery reason={retry.failure} reload={retry.reload} startNew={retry.startNew} />{/if}
    {#if retry.offered}<RetryOffer apply={retry.usePrevious} />{/if}
    <div class="grid-2">
      <div class="field">
        <label for="hf-repo">Repository ID</label>
        <input id="hf-repo" class="input" bind:value={repo} placeholder="owner/model" autocomplete="off" spellcheck="false" />
      </div>
      <div class="field">
        <label for="hf-revision">Revision (optional)</label>
        <input id="hf-revision" class="input" bind:value={revision} placeholder="main" aria-describedby="hf-rev-help" autocomplete="off" spellcheck="false" />
        <span class="hint" id="hf-rev-help">Use a branch, tag or commit. Leave blank for main.</span>
      </div>
      <div class="field">
        <label for="hf-scope">Download scope</label>
        <select id="hf-scope" class="input" bind:value={scope}>
          <option value="file">One model file</option>
          <option value="repo">Whole repository</option>
        </select>
      </div>
      {#if scope === 'file'}
        <div class="field">
          <label for="hf-filename">Model filename</label>
          <input id="hf-filename" class="input" bind:value={filename} placeholder="model.safetensors" aria-describedby="hf-file-help" autocomplete="off" spellcheck="false" />
          <span class="hint" id="hf-file-help">Use the exact name in the repository, including any subfolder.</span>
        </div>
      {/if}
    </div>
    {#if host !== 'backend-hf'}<p class="hint">Download on {hostName(hosts, host)}.</p>{/if}
    <p class="hint">Track progress or cancel in Jobs. The download service verifies the files before it adds the entry to Library.</p>
    <Submit {blockers} busy={submitter.busy} disabled={retry.unavailable} onclick={start}>Start download</Submit>
    <SubmitStatus s={submitter} current={request} />
  </div>
</details>

<style>
  .fields { padding-bottom: var(--s-3); }
</style>

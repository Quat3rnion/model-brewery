<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { Submitter } from '../lib/submit.svelte';
  import { hashParam, route } from '../lib/route.svelte';
  import { watchRetry } from '../lib/rerun';
  import Submit from './Submit.svelte';

  const submitter = new Submitter();
  const available = $derived(brewery.state?.hosts.some((h) => h.id === 'backend-hf' && h.capabilities?.hf_download === true));
  let repo = $state('');
  let revision = $state('');
  let scope = $state('file');
  let filename = $state('');
  let open = $state(false);
  let retryError = $state(false);
  $effect(() => {
    const hash = route.hash;
    retryError = false;
    if (hashParam(hash, 'retry')) open = true;
    return watchRetry(hash, ['hf_download'], ({ options }) => {
      repo = typeof options?.repo_id === 'string' ? options.repo_id : '';
      revision = typeof options?.revision === 'string' ? options.revision : '';
      filename = typeof options?.filename === 'string' ? options.filename : '';
      scope = filename ? 'file' : 'repo';
    }, () => { retryError = true; });
  });
  const safePart = (s: string) => /^[A-Za-z0-9_][A-Za-z0-9._-]*$/.test(s) && !s.includes('..');
  const blockers = $derived.by(() => {
    const errors: string[] = [];
    const parts = repo.trim().split('/');
    if (parts.length !== 2 || !parts.every(safePart)) errors.push('Enter a repository ID in owner/model form.');
    if (revision.trim() && !revision.trim().split('/').every(safePart)) errors.push('Enter a branch, tag or commit, not a URL.');
    if (scope === 'file' && (!filename.trim().split('/').every(safePart) || !/\.(gguf|safetensors)$/i.test(filename.trim())))
      errors.push('Enter the exact .gguf or .safetensors filename from the repository.');
    if (!available) errors.push('The built-in downloader is unavailable.');
    return errors;
  });
  async function start() {
    if (submitter.busy || blockers.length) return;
    const job = await submitter.submit({
      kind: 'hf_download', input_ids: [],
      options: { repo_id: repo.trim(), revision: revision.trim(), filename: scope === 'file' ? filename.trim() : undefined },
    });
    if (job) location.hash = `#/jobs/${encodeURIComponent(job.id)}`;
  }
</script>

<details class="disclosure" bind:open>
  <summary><strong>Download from Hugging Face</strong></summary>
  <div class="stack fields">
    {#if retryError}<p class="note warn">Cannot open the previous request. Enter the inputs again.</p>{/if}
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
    <p class="hint">Track progress or cancel in Jobs. Backend downloads verifies the download before it adds the source to Library.</p>
    <Submit {blockers} busy={submitter.busy} onclick={start}>Start download</Submit>
    {#if submitter.error}<p class="note bad" role="alert">{submitter.error}</p>{/if}
  </div>
</details>

<style>
  .fields { padding-bottom: var(--s-3); }
</style>

<script lang="ts">
  import { brewery } from '../lib/store.svelte';
  import { modelLabel } from '../lib/format';
  import { hostWorkflowEligibility } from '../lib/eligibility';
  import { Submitter } from '../lib/submit.svelte';
  import type { Model } from '../lib/types';
  import Submit from './Submit.svelte';
  import SubmitStatus from './SubmitStatus.svelte';

  let { teacher }: { teacher: Model } = $props();
  const worker = $derived(brewery.state?.hosts.find(h => h.id === teacher.host));
  const hostLabel = $derived(worker?.name ?? teacher.host);
  const eligibility = $derived(hostWorkflowEligibility(brewery.state?.eligibility, teacher.host, 'qsa_train'));
  const files = $derived((brewery.state?.models ?? []).filter(m => m.kind === 'qsa-corpus' &&
    m.host === teacher.host && m.available !== false && m.path !== teacher.path &&
    !m.path.startsWith(teacher.path.replace(/\/$/, '') + '/')));
  let train = $state(''), heldout = $state('');
  let steps = $state(100), seq_len = $state(4096), seed = $state(42), lr = $state(0.001);
  const training = $derived(files.find(m => m.id === train));
  const evalFiles = $derived(files.filter(m => m.path !== training?.path));
  const int = (n: number, min: number, max: number) => Number.isInteger(n) && n >= min && n <= max;
  const blockers = $derived([
    ...(eligibility?.eligible === false ? (eligibility.reasons.length ? eligibility.reasons.map(r => r.text) : [`Training is not available on ${hostLabel}.`])
      : worker?.capabilities?.qsa_train === true ? [] : [worker?.capability_reasons?.qsa_train?.text || worker?.reasons?.qsa_train || `${hostLabel} does not report QSA training support.`]),
    ...(teacher.available === false ? [teacher.reason || 'The teacher checkpoint is unavailable.'] : []),
    ...(!training || !evalFiles.some(m => m.id === heldout) ? ['Select two separate corpus files.'] : []),
    ...(!int(steps,1,10000) ? ['Use 1–10000 steps.'] : []),
    ...(!int(seq_len,8,16384) ? ['Use a sequence length of 8–16384.'] : []),
    ...(!int(seed,0,4294967295) ? ['Use a seed of 0–4294967295.'] : []),
    ...(!(lr > 0 && lr <= 1) ? ['Use a learning rate greater than 0 and at most 1.'] : []),
  ]);
  const s = new Submitter('library.qsa-train');
  function queue(event: MouseEvent) {
    if (!blockers.length) void s.submit({ kind:'qsa_train', host:teacher.host, input_ids:[teacher.id,train,heldout], options:{steps,seq_len,seed,lr} },event);
  }
</script>

<section class="panel" aria-labelledby="qsa-heading">
  <header><h2 tabindex="-1" id="qsa-heading">Train a QSA indexer</h2></header>
  <div class="body stack">
    <p class="small break">Teacher: <strong>{modelLabel(teacher)}</strong> on {hostLabel}. The worker checks for a supported dense Qwen checkpoint and trains a new indexer without changing the teacher.</p>
    <div class="grid-2">
      <div class="field"><label for="qsa-train-text">Training corpus</label>
        <select id="qsa-train-text" class="input" bind:value={train}><option value="">Choose training text</option>
          {#each files as m (m.id)}<option value={m.id}>{modelLabel(m)}</option>{/each}
        </select>
      </div>
      <div class="field"><label for="qsa-heldout-text">Held-out corpus</label>
        <select id="qsa-heldout-text" class="input" bind:value={heldout}><option value="">Choose held-out text</option>
          {#each evalFiles as m (m.id)}<option value={m.id}>{modelLabel(m)}</option>{/each}
        </select>
      </div>
    </div>
    <p class="hint">Use “Register an entry” below to add two separate UTF-8 files as QSA text corpora on {hostLabel}. Keep both outside the checkpoint directory.</p>
    <div class="grid-2">
      <label class="field">Training steps<input class="input" type="number" min="1" max="10000" bind:value={steps} /></label>
      <label class="field">Sequence length (tokens)<input class="input" type="number" min="8" max="16384" bind:value={seq_len} /></label>
      <label class="field">Random seed<input class="input" type="number" min="0" max="4294967295" bind:value={seed} /></label>
      <label class="field">Learning rate<input class="input" type="number" min="0" max="1" step="any" bind:value={lr} /></label>
    </div>
    <Submit {blockers} busy={s.busy} onclick={queue}>Queue training on {hostLabel}</Submit>
    <SubmitStatus {s} resultLinkLabel="View training job" allowRepeat={false} />
  </div>
</section>

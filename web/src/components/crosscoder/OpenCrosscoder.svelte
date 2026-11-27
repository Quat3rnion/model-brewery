<script lang="ts">
  // Opens a crosscoder report for a parent/child pair. Reports are trained
  // offline (CLI) and imported; the operator or agent gives the report ID.
  import { crosscoderHref, isCrosscoderId } from '../../lib/crosscoder';
  import FieldError from '../FieldError.svelte';

  let { rev }: { rev: string } = $props();
  const uid = $props.id();
  let value = $state('');
  let attempted = $state(false);
  const id = $derived(value.trim().toLowerCase());
  const error = $derived(attempted && !isCrosscoderId(id) ? 'Enter the 64-character hexadecimal report ID.' : null);
  $effect(() => {
    void rev;
    value = '';
    attempted = false;
  });

  function open(event: SubmitEvent) {
    event.preventDefault();
    attempted = true;
    if (isCrosscoderId(id)) window.location.hash = crosscoderHref(id, rev);
  }
</script>

<section class="xc" aria-labelledby="{uid}-h">
  <h3 id="{uid}-h">Crosscoder diff</h3>
  <p class="small muted" id="{uid}-d">A crosscoder report shows the features that are unique to the parent or the child and the shared features that changed. Train it with the CLI, then open it here by its ID.</p>
  <form class="row" onsubmit={open} novalidate>
    <div class="field grow">
      <label for="{uid}-id">Crosscoder report ID</label>
      <input
        id="{uid}-id"
        class="input mono"
        bind:value
        autocomplete="off"
        spellcheck="false"
        aria-invalid={error ? 'true' : undefined}
        aria-describedby={error ? `${uid}-d ${uid}-e` : `${uid}-d`}
        onblur={() => (attempted = value.trim() !== '')}
      />
    </div>
    <button type="submit" class="btn small">Open report</button>
  </form>
  <FieldError id="{uid}-e" error={error ?? undefined} />
</section>

<style>
  .xc {
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
  }
  h3 {
    font-size: var(--fs-md);
    font-weight: 600;
  }
  form {
    align-items: flex-end;
  }
  .grow {
    flex: 1 1 16rem;
    min-width: 0;
  }
</style>

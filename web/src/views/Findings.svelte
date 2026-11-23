<script lang="ts">
  // X10 findings: a list of versioned hypotheses and one finding's evidence.
  // Routes: #/findings and #/findings/<id>. The view reads with GET only.
  import { tick } from 'svelte';
  import { api, errorText } from '../lib/api';
  import { route } from '../lib/route.svelte';
  import { findingIdFromHash, parseFinding, parseFindingSummaries, type Finding, type FindingSummary } from '../lib/findings';
  import FindingList from '../components/findings/FindingList.svelte';
  import FindingDetail from '../components/findings/FindingDetail.svelte';

  const id = $derived(findingIdFromHash(route.hash));
  const badRoute = $derived(/^#\/findings\/./.test(route.hash) && id === null);

  let list = $state<FindingSummary[] | null>(null);
  let listError = $state<string | null>(null);
  let detail = $state<Finding | null>(null);
  let detailError = $state<string | null>(null);

  $effect(() => {
    if (id !== null || badRoute) return;
    const ac = new AbortController();
    listError = null;
    api
      .findings(ac.signal)
      .then((raw) => {
        const r = parseFindingSummaries(raw);
        if (r.ok) list = r.value;
        else listError = r.error;
      })
      .catch((e) => {
        if (!ac.signal.aborted) listError = errorText(e);
      });
    return () => ac.abort();
  });

  $effect(() => {
    const target = id;
    detail = null;
    detailError = null;
    if (target === null) return;
    const ac = new AbortController();
    api
      .finding(target, ac.signal)
      .then((raw) => {
        const r = parseFinding(raw);
        if (r.ok && r.value.id !== target) detailError = 'The controller returned a finding with another ID.';
        else if (r.ok) detail = r.value;
        else detailError = r.error;
      })
      .catch((e) => {
        if (!ac.signal.aborted) detailError = errorText(e);
      });
    return () => ac.abort();
  });

  // App moves focus only when the section changes. Between list and detail, do it here.
  let root: HTMLElement | undefined = $state();
  let shown: string | null | undefined;
  $effect(() => {
    const target = id;
    if (shown !== undefined && shown !== target) tick().then(() => root?.querySelector<HTMLElement>('h1')?.focus());
    shown = target;
  });
</script>

<div class="stack" bind:this={root}>
  {#if badRoute}
    <h1 tabindex="-1">Finding</h1>
    <p class="note bad small" role="alert">The finding link is not valid. <a href="#/findings">Open all findings</a>.</p>
  {:else if id === null}
    <FindingList findings={list} error={listError} />
  {:else}
    <FindingDetail {id} finding={detail} error={detailError} />
  {/if}
</div>

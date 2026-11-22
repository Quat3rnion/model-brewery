<script lang="ts">
  import Badge from '../Badge.svelte';
  import { shortHash } from '../../lib/format';
  import { coord, exact, percent, readApplied, readInspect, readProposal, readTuneReport, relChange, rounded, signed } from '../../lib/weights';
  import type { JobDetail } from '../../lib/types';

  let { job }: { job: JobDetail } = $props();
  const kind = $derived(String(job.kind));
  const proposal = $derived(readProposal(job.result));
  const tune = $derived(readTuneReport(job.result));
  const inspected = $derived(readInspect(job.result));
  const applied = $derived(readApplied(job.result));
  const roundedCount = $derived(proposal?.edits.filter(rounded).length ?? 0);
  // One tensor for all edits (always so for tuning): name it once above the table.
  const oneTensor = $derived(proposal && proposal.edits.every((e) => e.tensor === proposal.edits[0].tensor) ? proposal.edits[0].tensor : null);
  const uid = $props.id();
</script>

<div class="stack">
  {#if kind === 'weight_inspect'}
    {#if inspected?.mode === 'value'}
      <dl class="kv small">
        <dt>Tensor</dt><dd class="mono break">{inspected.tensor}</dd>
        <dt>Coordinate</dt><dd class="mono num">{coord(inspected.index)}</dd>
        <dt>Shape</dt><dd class="mono num">{coord(inspected.shape)}</dd>
        <dt>Stored type</dt><dd class="mono">{inspected.dtype}</dd>
        <dt>Stored value</dt><dd class="mono num">{exact(inspected.value)}</dd>
        {#if inspected.source_sha256}<dt>Source SHA-256</dt><dd class="mono" title={inspected.source_sha256}>{shortHash(inspected.source_sha256)}</dd>{/if}
      </dl>
    {:else if inspected?.mode === 'inventory'}
      <p class="small muted">{inspected.tensors.length} of {inspected.total.toLocaleString()} tensors on this page. Use Weights to read another page or select a tensor.</p>
      {#if inspected.tensors.length}
        <!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users must be able to scroll this table.) -->
        <div class="table-wrap scroll" role="region" aria-label="Tensor inventory scroll area" tabindex="0">
          <table class="data">
            <caption class="sr-only">Tensor inventory</caption>
            <thead><tr><th scope="col">Tensor</th><th scope="col">Shape</th><th scope="col">Stored type</th></tr></thead>
            <tbody>
              {#each inspected.tensors as tensor (tensor.order)}
                <tr>
                  <td class="mono xs break">{tensor.name}</td>
                  <td class="mono xs num">{coord(tensor.shape)}</td>
                  <td class="mono xs">{tensor.dtype}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    {:else if inspected?.mode === 'tile'}
      <dl class="kv small">
        <dt>Tensor</dt><dd class="mono break">{inspected.tensor}</dd>
        <dt>Origin</dt><dd class="mono num">{coord(inspected.origin)}</dd>
        <dt>Tensor shape</dt><dd class="mono num">{coord(inspected.shape)}</dd>
        <dt>Tile shape</dt><dd class="mono num">{coord(inspected.tile_shape)}</dd>
        <dt>Stored type</dt><dd class="mono">{inspected.dtype}</dd>
      </dl>
      <p class="xs muted">The raw report lists the tile values in row order.</p>
    {:else if job.state === 'succeeded'}
      <p class="note warn">The inspect result has an unknown shape.</p>
    {/if}
    {#if inspected && inspected.mode !== 'value' && job.input_ids[0]}
      <p class="small"><a href="#/weights?source={encodeURIComponent(job.input_ids[0])}">Open source in Weights</a></p>
    {/if}
  {/if}

  {#if kind === 'weight_tune'}
    {#if tune}
      {@const regress = tune.held_out.regression}
      <section class="stack tight" aria-labelledby="{uid}-m">
        <h3 class="h" id="{uid}-m">Layer-output error (MSE)</h3>
        <div class="table-wrap">
          <table class="data num stackable">
            <thead><tr><th scope="col">Split</th><th scope="col">Before</th><th scope="col">After</th><th scope="col">Change</th><th scope="col"><span class="sr-only">Result</span></th></tr></thead>
            <tbody>
              <tr>
                <th scope="row">Calibration (fitted)</th>
                <td class="mono" data-label="Before">{exact(tune.calibration.before)}</td><td class="mono" data-label="After">{exact(tune.calibration.after)}</td>
                <td data-label="Change">{percent(relChange(tune.calibration.before, tune.calibration.after))}</td>
                <td data-label="Role"><Badge tone="muted" label="Used for fitting" /></td>
              </tr>
              <tr>
                <th scope="row">Held-out (measured only)</th>
                <td class="mono" data-label="Before">{exact(tune.held_out.before)}</td><td class="mono" data-label="After">{exact(tune.held_out.after)}</td>
                <td data-label="Change">{percent(relChange(tune.held_out.before, tune.held_out.after))}</td>
                <td data-label="Result"><Badge tone={regress ? 'bad' : tune.held_out.after < tune.held_out.before ? 'ok' : 'muted'} label={regress ? 'Held-out error increased' : tune.held_out.after < tune.held_out.before ? 'Held-out error decreased' : 'No held-out change'} /></td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="xs muted">
          These numbers are for one layer output on the registered splits. They are not a model quality or release result.
          {#if tune.method}Method <span class="mono">{tune.method}</span>{#if tune.max_sweeps !== null}, at most {tune.max_sweeps} sweeps{/if}.{/if}
          {#if tune.selected !== null && tune.proposed !== null} {tune.proposed} of {tune.selected} selected weights changed{#if tune.max_update !== null}, each by at most {exact(tune.max_update)}{/if}.{/if}
        </p>
        {#if regress}<p class="note warn">The tuned weights increase the error on the held-out split. Do not apply this proposal unless you have another reason to.</p>{/if}
      </section>
    {:else if job.state === 'succeeded'}
      <p class="note warn">The tune report is missing or has an unknown shape. Review the proposed values only.</p>
    {/if}
  {/if}

  {#if kind === 'weight_propose' || kind === 'weight_tune'}
    {#if proposal}
      <section class="stack tight" aria-labelledby="{uid}-p">
        <div class="row between">
          <h3 class="h" id="{uid}-p">Proposed values ({proposal.edits.length})</h3>
          <span class="xs muted">{#if oneTensor}Tensor <span class="mono break">{oneTensor}</span>{" · "}{/if}Source SHA-256 <span class="mono" title={proposal.source_sha256}>{shortHash(proposal.source_sha256)}</span></span>
        </div>
        <div class="table-wrap scroll">
          <table class="data stackable">
            <thead>
              <tr>{#if !oneTensor}<th scope="col">Tensor</th>{/if}<th scope="col">Coordinate</th><th scope="col">Type</th><th scope="col">Requested</th><th scope="col">Stored before</th><th scope="col">Stored after</th><th scope="col">Change</th></tr>
            </thead>
            <tbody>
              {#each proposal.edits as e, i (i)}
                <tr>
                  {#if !oneTensor}<td class="mono xs tname" data-label="Tensor">{e.tensor}</td>{/if}
                  <td class="mono num nowrap" data-label="Coordinate">{coord(e.index)}</td>
                  <td class="mono" data-label="Type">{e.dtype}</td>
                  <td class="mono num nowrap" data-label="Requested">{e.op === 'set' ? `set ${exact(e.value)}` : `delta ${signed(e.value)}`}</td>
                  <td class="mono num" data-label="Stored before">{exact(e.before)}</td>
                  <td class="mono num" data-label="Stored after">{exact(e.after)}{#if rounded(e)}{" "}<span class="rnd" title="Rounded to the nearest {e.dtype} value">rounded</span>{/if}</td>
                  <td class="mono num nowrap" data-label="Change" title="Exact: {signed(e.after - e.before)}">{signed(Number((e.after - e.before).toPrecision(6)))} <span class="muted">({percent(relChange(e.before, e.after))})</span></td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
        {#if roundedCount}<p class="xs muted">{roundedCount} {roundedCount === 1 ? 'value was' : 'values were'} rounded to the stored type. The After column shows the value that apply writes.</p>{/if}
      </section>
    {:else if job.state === 'succeeded'}
      <p class="note bad" role="alert">The proposal is missing or has an unknown shape. It cannot be reviewed or applied.</p>
    {/if}
  {/if}

  {#if kind === 'weight_apply'}
    {#if applied}
      <dl class="kv small">
        <dt>Result</dt><dd>{#if applied.verified}<Badge tone="ok" label="Byte-verified copy" title="The worker compared every unchanged byte with the source" />{:else}<Badge tone="warn" label="Not verified" />{/if}</dd>
        <dt>Edits written</dt><dd class="num">{applied.edit_count}</dd>
        <dt>Source SHA-256</dt><dd class="mono" title={applied.source_sha256}>{shortHash(applied.source_sha256)}</dd>
        <dt>Revision SHA-256</dt><dd class="mono" title={applied.output_sha256}>{shortHash(applied.output_sha256)}</dd>
        <dt>Library entry</dt><dd>{#if applied.library_model_id}<a href="#/weights?source={encodeURIComponent(applied.library_model_id)}" class="mono">{applied.library_model_id}</a>{:else}<span class="muted">Not registered</span>{/if}</dd>
      </dl>
      <p class="xs muted">The source is unchanged. The revision is a separate Library entry in the source format. It is not evaluated or release-qualified.</p>
    {:else if job.state === 'succeeded'}
      <p class="note warn">The apply receipt has an unknown shape.</p>
    {/if}
  {/if}
</div>

<style>
  .tight {
    gap: var(--s-2);
  }
  .h {
    margin: 0;
    font-size: var(--fs-sm);
    font-weight: 600;
  }
  .between {
    justify-content: space-between;
  }
  .kv {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 4px var(--s-4);
    margin: 0;
  }
  .kv dt {
    color: var(--text-2);
  }
  .kv dd {
    margin: 0;
  }
  .scroll {
    max-height: 420px;
    overflow: auto;
    border: 1px solid var(--line-soft);
    border-radius: var(--r-tile);
  }
  .tname {
    min-width: 12rem;
    overflow-wrap: anywhere;
  }
  .rnd {
    font-family: var(--font-sans);
    font-size: var(--fs-xs);
    color: var(--warn);
  }
</style>
